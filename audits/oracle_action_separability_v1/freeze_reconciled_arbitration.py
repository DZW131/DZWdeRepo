"""GT-free FP32 re-freeze of Track A after sealed DLAG baseline mismatch audit.

The original observable feature tables remain immutable as a forensic record.
This pass computes all A features/components with the exact sealed DLAG
FP32-resize/torch-mean CAM path; Gate universe/features are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from audits.oracle_action_separability_v1.freeze_observables import component_features,digest,infer
from audits.ucrf_v1.gate import load_model
from network.cirv import extract_regions
from tool.GenDataset import Stage1_InferDataset


def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","val-root","dlag","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--num-workers",type=int,default=2)
    args=p.parse_args();out=args.output
    original_path=out/"feature_manifest_final.json"
    final_path=out/"feature_manifest_action.json"
    if not original_path.exists() or final_path.exists():
        raise AssertionError("Original two-pass GT-free freeze required; reconciled manifest absent")
    if (out/"arbitration/oracle_action_labels.parquet").exists() or (out/"gate/rescue_oracle_labels.parquet").exists():
        raise AssertionError("GT/Oracle labels already exist; cannot re-freeze")
    manifest=json.loads(original_path.read_text())
    sources=json.loads((out/"source_manifest.json").read_text())
    for relative,expected in manifest["sha256"].items():
        if digest(out/relative)!=expected:raise AssertionError(f"Original GT-free table changed: {relative}")
    bank_path=args.dlag/"counterfactuals/alpha_100/predictions.npz"
    if digest(bank_path)!=sources["source_sha256"][str(bank_path)]:
        raise AssertionError("DLAG alpha=1 source hash mismatch")
    with np.load(bank_path) as file:
        bank=file["predictions"].copy();ids=file["image_ids"].copy()
    with np.load(out/"baseline_predictions.npz") as file:
        old=file["predictions"].copy()
    gates=pd.read_parquet(out/"gate/gate_off_pairs.parquet")
    gate_keys=set(zip(gates.image_id.astype(str),gates.candidate_class.astype(int)))
    model=load_model(args.checkpoint)
    loader=DataLoader(Stage1_InferDataset(str(args.val_root/"img"),img_size=224),
                      batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    rows=[]; predictions=[];mismatch=0;start=time.perf_counter()
    for index,(names,image) in enumerate(loader):
        image_id=str(names[0])
        if image_id!=str(ids[index]):raise AssertionError("Image ID order mismatch")
        bundle=infer(model,image.cuda(non_blocking=True))
        prediction=bundle["prediction"].astype(np.uint8)
        if not np.array_equal(prediction,bank[index]):
            raise AssertionError(f"FP32 HQMR baseline still differs from DLAG bank at {image_id}")
        mismatch+=int(np.count_nonzero(prediction!=old[index]))
        predictions.append(prediction)
        for cls in range(4):
            if (bundle["label"][cls]==0)!=( (image_id,cls) in gate_keys):
                raise AssertionError(f"Gate-off universe changed: {image_id}, class {cls}")
        regions=extract_regions(prediction)
        counts={cls:sum(int(r["class_id"]==cls) for r in regions) for cls in range(4)}
        for region in regions:
            rows.append(component_features(bundle,region,image_id,index,counts))
        if (index+1)%100==0 or index+1==len(loader):
            print(json.dumps({"event":"reconciled_progress","images":index+1,
                              "components":len(rows),"old_pixel_drift":mismatch,
                              "elapsed_s":round(time.perf_counter()-start,1)}),flush=True)
    path=out/"arbitration/observable_features_fp32.parquet"
    pred_path=out/"baseline_predictions_fp32.npz"
    pd.DataFrame(rows).to_parquet(path,index=False)
    np.savez_compressed(pred_path,predictions=np.stack(predictions),image_ids=ids)
    if not np.array_equal(np.stack(predictions),bank):
        raise AssertionError("Final prediction bank mismatch")
    final=json.loads(original_path.read_text())
    final["initial_two_pass_manifest_sha256"]=digest(original_path)
    final["fp32_reconciliation"]={"old_pixel_drift":mismatch,
        "old_pixel_drift_fraction":mismatch/(len(ids)*224*224),
        "new_components":len(rows),"bank_exact_pixel_match":True,
        "reason":"BF16 interpolation before resize in original feature pass; sealed DLAG uses FP32 before resize"}
    final["sha256"][str(path.relative_to(out))]=digest(path)
    final["sha256"][str(pred_path.relative_to(out))]=digest(pred_path)
    final["active_A_feature_table"]=str(path.relative_to(out))
    final["active_baseline_predictions"]=str(pred_path.relative_to(out))
    final_path.write_text(json.dumps(final,indent=2),encoding="utf-8")
    (out/"feature_table_sha256_action.txt").write_text(
        "\n".join(f"{value}  {relative}" for relative,value in final["sha256"].items())+"\n",
        encoding="utf-8")
    print(json.dumps({"event":"RECONCILED_A_FROZEN","components":len(rows),
                      "old_pixel_drift":mismatch,"manifest_sha256":digest(final_path)}),flush=True)


if __name__=="__main__":main()
