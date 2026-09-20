"""Second GT-free pass: sealed, true W(q,c) × query-basis support features.

This runs after the full observable pass but still *before* any GT or Oracle label read.
The original manifest/table remain untouched; a final manifest includes both hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from audits.ucrf_v1.gate import load_model
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import TTA


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def query_features(model,image):
    dummy=torch.ones((1,4),device=image.device)
    weights2=[];weights3=[];responses=[]
    for input_flip,_ in TTA:
        value=torch.flip(image,dims=input_flip) if input_flip else image
        with torch.autocast("cuda",dtype=torch.bfloat16):
            output=model(value,dummy,step=29275,hqmr_mode="full")
            stage2=output["stages"][1]["hqmr"]
            stage3=output["stages"][2]["hqmr"]
            w2=stage2["weights"][0].detach().float().cpu().numpy()
            w3=stage3["weights"][0].detach().float().cpu().numpy()
            basis=stage3["basis"][0].detach().float().flatten(1).mean(1).cpu().numpy()
        weights2.append(w2);weights3.append(w3);responses.append(basis)
    w2=np.mean(weights2,axis=0);w3=np.mean(weights3,axis=0)
    response=np.mean(responses,axis=0)
    return w2,w3,response


def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","val-root","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--num-workers",type=int,default=2)
    args=p.parse_args();out=args.output
    original=out/"feature_manifest.json"; final=out/"feature_manifest_final.json"
    if not original.exists() or final.exists():
        raise AssertionError("Original GT-free freeze must exist; final freeze must not")
    if (out/"arbitration/oracle_action_labels.parquet").exists() or (out/"gate/rescue_oracle_labels.parquet").exists():
        raise AssertionError("Oracle/GT labels already exist; query augmentation forbidden")
    base=json.loads(original.read_text())
    for relative,expected in base["sha256"].items():
        if digest(out/relative)!=expected:
            raise AssertionError(f"Initial GT-free feature hash changed: {relative}")
    pairs=pd.read_parquet(out/"gate/gate_off_pairs.parquet")
    model=load_model(args.checkpoint)
    loader=DataLoader(Stage1_InferDataset(str(args.val_root/"img"),img_size=224),
                      batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    by_image={key:frame for key,frame in pairs.groupby("image_id",sort=False)}
    rows=[]
    for index,(names,image) in enumerate(loader,1):
        name=str(names[0]);w2,w3,response=query_features(model,image.cuda(non_blocking=True))
        local=by_image.get(name)
        if local is None: continue
        for row in local.itertuples():
            cls=int(row.candidate_class)
            support=np.maximum(w3[:,cls],0.)
            mass=support/max(float(support.sum()),1e-8)
            evidence=support*response
            earlier=np.maximum(w2[:,cls],0.)
            earlier=earlier/max(float(earlier.sum()),1e-8)
            rows.append({"image_id":name,"candidate_class":cls,
                         "g_query_weight_mass":float(support.sum()),
                         "g_query_top_response":float(evidence.max()),
                         "g_query_entropy_true":float(-(mass*np.log(np.maximum(mass,1e-8))).sum()/np.log(len(mass))),
                         "g_query_weight_persistence":float(np.dot(mass,earlier)/(max(np.linalg.norm(mass)*np.linalg.norm(earlier),1e-8))),
                         "g_query_top_identity_same":int(np.argmax(mass)==np.argmax(earlier))})
        if index%200==0 or index==len(loader):
            print(json.dumps({"event":"query_freeze_progress","images":index,"pairs":len(rows)}),flush=True)
    extra=pd.DataFrame(rows)
    if len(extra)!=len(pairs) or not np.array_equal(extra.image_id,pairs.image_id) or not np.array_equal(extra.candidate_class,pairs.candidate_class):
        raise AssertionError("Query features do not align with frozen gate pairs")
    path=out/"gate/gate_query_features.parquet"
    extra.to_parquet(path,index=False)
    new=json.loads(original.read_text())
    new["initial_manifest_sha256"]=digest(original)
    new["two_pass_GT_free_freeze"]=True
    new["features_G"] += [c for c in extra if c.startswith("g_")]
    if len(new["features_G"])>=50:
        raise AssertionError("Gate feature count cap exceeded")
    new["sha256"][str(path.relative_to(out))]=digest(path)
    final.write_text(json.dumps(new,indent=2),encoding="utf-8")
    (out/"feature_table_sha256_final.txt").write_text(
        "\n".join(f"{value}  {relative}" for relative,value in new["sha256"].items())+"\n",
        encoding="utf-8")
    print(json.dumps({"event":"FINAL_GT_FREE_FEATURES_FROZEN","gate_features":len(new["features_G"]),
                      "manifest_sha256":digest(final),"extra_sha256":digest(path)}),flush=True)


if __name__=="__main__":main()
