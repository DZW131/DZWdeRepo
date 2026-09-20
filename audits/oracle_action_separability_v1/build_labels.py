"""Build component and gate Oracle labels only after feature manifest is sealed."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.oracle_action_separability_v1.freeze_observables import infer
from audits.ucrf_v1.gate import load_model
from network.cirv import extract_regions
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import prediction_from_cam

ALPHAS=np.asarray([0.,.25,.5,1.,1.5,2.,3.,4.])
TIE_ORDER=[3,2,4,1,5,0,6,7]


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(dlag: Path, ids: np.ndarray) -> np.ndarray:
    arrays=[]
    for alpha in ALPHAS:
        path=dlag/f"counterfactuals/alpha_{int(round(alpha*100)):03d}/predictions.npz"
        with np.load(path) as bank:
            if not np.array_equal(bank["image_ids"],ids):
                raise AssertionError(f"DLAG image IDs mismatch: {path}")
            arrays.append(bank["predictions"])
    return np.stack(arrays)


@torch.inference_mode()
def main() -> None:
    parser=argparse.ArgumentParser()
    for name in ("checkpoint","val-root","dlag","ucrf","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    parser.add_argument("--num-workers",type=int,default=2)
    args=parser.parse_args()
    out=args.output
    if (out/"arbitration/oracle_action_labels.parquet").exists():
        raise FileExistsError("Oracle labels already exist")
    gate=json.loads((out/"00_reproduction_gate.json").read_text())
    manifest=json.loads((out/"feature_manifest.json").read_text())
    if not gate["pass"] or not manifest["feature_freeze_before_gt"]:
        raise AssertionError("Reproduction/feature freeze gate failed")
    for relative,expected in manifest["sha256"].items():
        if sha(out/relative)!=expected:
            raise AssertionError(f"Frozen feature hash mismatch: {relative}")
    a=pd.read_parquet(out/"arbitration/observable_features.parquet")
    g=pd.read_parquet(out/"gate/gate_off_pairs.parquet")
    with np.load(out/"baseline_predictions.npz") as pred_bank:
        baseline=pred_bank["predictions"].copy()
        ids=pred_bank["image_ids"].copy()
    bank=load_predictions(args.dlag,ids)
    if not np.array_equal(bank[3],baseline):
        mismatch=int(np.sum(bank[3]!=baseline))
        raise AssertionError(f"GT-free baseline differs from frozen DLAG alpha=1 ({mismatch} pixels)")
    ucrf=pd.read_parquet(args.ucrf/"metrics/component_event_table.parquet")
    m1=set(zip(ucrf.loc[ucrf.cohort=="M1","image_id"].astype(str),
               ucrf.loc[ucrf.cohort=="M1","predicted_class"].astype(int),
               ucrf.loc[ucrf.cohort=="M1","component_id"].astype(int)))
    m1_gate=set(zip(ucrf.loc[ucrf.cohort=="M1","image_id"].astype(str),
                    ucrf.loc[ucrf.cohort=="M1","true_class"].astype(int)))
    if len(m1)!=4440:
        raise AssertionError("Frozen M1 subset mismatch")
    loader=DataLoader(Stage1_InferDataset(str(args.val_root/"img"),img_size=224),
                      batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    model=load_model(args.checkpoint)
    a_rows=[]; g_rows=[]
    a_lookup={(str(r.image_id),int(r.predicted_class),int(r.component_id)):i for i,r in a.iterrows()}
    g_lookup={(str(r.image_id),int(r.candidate_class)):i for i,r in g.iterrows()}
    for image_index,(names,image) in enumerate(loader):
        image_id=str(names[0])
        if image_id!=str(ids[image_index]):
            raise AssertionError("Image-order mismatch")
        truth=np.asarray(Image.open(args.val_root/"mask"/f"{image_id}.png"))
        valid=truth<4
        pred=baseline[image_index]
        if truth.shape!=pred.shape:
            raise AssertionError("Image/GT shape mismatch")
        for region in extract_regions(pred):
            key=(image_id,int(region["class_id"]),int(region["component_id"]))
            if key not in a_lookup:
                raise AssertionError(f"Feature component missing: {key}")
            mask=region["mask"] & valid
            if not mask.any():
                gains=np.zeros(len(ALPHAS),dtype=int)
            else:
                correct=np.asarray([(bank[j,image_index][mask]==truth[mask]).sum() for j in range(len(ALPHAS))])
                gains=correct-correct[3]
            best=int(gains.max())
            choice=next(j for j in TIE_ORDER if gains[j]==best)
            action="KEEP" if best<=0 or choice==3 else "DOWN" if ALPHAS[choice]<1 else "UP"
            a_rows.append({"feature_row":a_lookup[key],"image_id":image_id,
                           "predicted_class":key[1],"component_id":key[2],
                           "area":int(region["area"]),"valid_area":int(mask.sum()),
                           "oracle_alpha":float(ALPHAS[choice]),"oracle_gain_pixels":best,
                           "action":action,"intervene":int(action!="KEEP"),
                           "direction_up":int(action=="UP") if action!="KEEP" else -1,
                           "m1_subset":int(key in m1),
                           "alpha_gain_curve":json.dumps(gains.tolist())})
        bundle=infer(model,image.cuda(non_blocking=True))
        if not np.array_equal(bundle["prediction"],pred):
            raise AssertionError(f"Rerun HQMR prediction mismatch: {image_id}")
        correct_base=int(((pred==truth)&valid).sum())
        for cls in range(4):
            if bundle["label"][cls]!=0:
                continue
            key=(image_id,cls)
            if key not in g_lookup:
                raise AssertionError(f"Frozen gate pair missing: {key}")
            force=bundle["label"].copy(); force[cls]=1
            forced=prediction_from_cam(bundle["maps"]["cam"],force,truth)
            correct_force=int(((forced==truth)&valid).sum())
            gain=correct_force-correct_base
            g_rows.append({"feature_row":g_lookup[key],"image_id":image_id,
                           "candidate_class":cls,"gate_gain_pixels":gain,
                           "beneficial_rescue":int(gain>0),
                           "class_present":int((truth==cls).any()),
                           "correct_baseline":correct_base,"correct_force_on":correct_force,
                           "m1_related":int(key in m1_gate)})
        if (image_index+1)%100==0 or image_index+1==len(loader):
            print(json.dumps({"event":"label_progress","images":image_index+1,
                              "a_labels":len(a_rows),"g_labels":len(g_rows)}),flush=True)
    al=pd.DataFrame(a_rows).sort_values("feature_row")
    gl=pd.DataFrame(g_rows).sort_values("feature_row")
    if not np.array_equal(al.feature_row,np.arange(len(a))) or not np.array_equal(gl.feature_row,np.arange(len(g))):
        raise AssertionError("Label coverage or ordering mismatch")
    if int(al.m1_subset.sum())!=4440:
        raise AssertionError("Frozen M1 component identity mismatch")
    al.to_parquet(out/"arbitration/oracle_action_labels.parquet",index=False)
    gl.to_parquet(out/"gate/rescue_oracle_labels.parquet",index=False)
    checks={"pass":True,"frozen_feature_sha256_verified":True,
            "baseline_equals_dlag_alpha1":True,"all_components":len(al),
            "all_gate_off_pairs":len(gl),"m1_components":int(al.m1_subset.sum()),
            "a_label_sha256":sha(out/"arbitration/oracle_action_labels.parquet"),
            "g_label_sha256":sha(out/"gate/rescue_oracle_labels.parquet"),
            "parameter_updates":0}
    (out/"leakage_audit.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps({"event":"ORACLE_LABELS_BUILT",**checks}),flush=True)


if __name__=="__main__":
    main()
