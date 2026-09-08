"""Read-only archived FOMD endpoint compatibility audit for GCQM."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from network.gcqm_net import GCQMNet
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_counterfactuals import materialize
from tools.gcqm_diagnostics import batch_health,summarize
from tools.hqrf_phase0_io import install_train_access_guard,write_json
from train_cqrf_phase0 import MonitorDataset


def main():
    p=argparse.ArgumentParser(); p.add_argument("--endpoint",required=True); p.add_argument("--cohort-json",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    install_train_access_guard(); raw=json.loads(Path(a.cohort_json).read_text()); cohort=[(x["path"],torch.tensor(x["label"],dtype=torch.float32)) for x in raw["images"]]
    if any("/bcss-wsss/training/" not in Path(path).as_posix().lower() for path,_ in cohort): raise ValueError("BCSS train-only cohort required")
    bank=permutation_payload(196); model=GCQMNet().cuda(); model.load_state_dict(torch.load(a.endpoint,map_location="cuda",weights_only=True),strict=True); model.eval(); batches=[]; pmec=[]; exact=[]; detached=[]; finite=[]
    with torch.no_grad():
        for _,images,labels in DataLoader(MonitorDataset(cohort),batch_size=8,num_workers=4,pin_memory=True):
            images=images.cuda(); labels=labels.cuda()
            with torch.autocast("cuda",dtype=torch.bfloat16): out=model(images,labels,step=5855,run_pmec=True)
            batches.append(batch_health(out,labels,bank["permutations"])); pmec.extend(out["pmec_rows"])
            for stage in out["stages"][1:]:
                cf=materialize(stage,bank["permutations"]); exact.append(torch.equal(cf["primary"],stage["gcqm"]["mixture"].detach().float())); detached.append(all(not x.requires_grad for x in cf.values())); finite.append(all(bool(torch.isfinite(x).all()) for x in cf.values()))
    summary=summarize("historical_fomd_endpoint",batches,pmec,model,8); ok=all(exact+detached+finite) and bank["all_derangements"]
    result={"decision":"PROCEED" if ok else "ENGINEERING_BLOCKED","endpoint":str(Path(a.endpoint).resolve()),"cohort_images":len(cohort),"primary_exact":all(exact),"references_detached":all(detached),"all_finite":all(finite),"all_derangements":bank["all_derangements"],"permutation_bank_sha256":bank["sha256"],"parameter_delta":0,"summary":summary,"exploratory_only":True,"thresholds_modified":False}
    write_json(a.output,result); print(json.dumps(result,allow_nan=False));
    if not ok: raise RuntimeError("GCQM compatibility blocked")


if __name__=="__main__": main()
