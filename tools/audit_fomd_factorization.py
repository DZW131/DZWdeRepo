"""Read-only archived MOMD endpoint audit for FOMD counterfactuals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from network.momd_net import MOMDNet
from tools.fomd_counterfactuals import materialize, permutation_payload
from tools.fomd_diagnostics import batch_health, summarize
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, write_json
from train_cqrf_phase0 import MonitorDataset


def load_cohort(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    return payload,[(x["path"],torch.tensor(x["label"],dtype=torch.float32)) for x in payload["images"]]


def main():
    p=argparse.ArgumentParser(); p.add_argument("--endpoint",required=True); p.add_argument("--cohort-json",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for archived endpoint audit")
    install_train_access_guard(); payload,cohort=load_cohort(a.cohort_json)
    for path,_ in cohort: check_train_path(path)
    bank=permutation_payload(196); permutations=bank["permutations"]
    model=MOMDNet().cuda(); state=torch.load(a.endpoint,map_location="cuda",weights_only=True); model.load_state_dict(state,strict=True); model.eval()
    count_before=sum(p.numel() for p in model.parameters()); batches=[]; pmec=[]; exact=[]; detached=[]; finite=[]
    loader=DataLoader(MonitorDataset(cohort),batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    with torch.no_grad():
        for _,images,labels in loader:
            images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=5855,run_pmec=True)
            batches.append(batch_health(result,labels,permutations)); pmec.extend(result["pmec_rows"])
            for stage in result["stages"][1:]:
                cf=materialize(stage,permutations)
                exact.append(torch.equal(cf["full"],stage["momd"]["mixture"].detach().float()))
                detached.append(all(not value.requires_grad for value in cf.values()))
                finite.append(all(bool(torch.isfinite(value).all()) for value in cf.values()))
    summary=summarize("historical_momd_endpoint",batches,pmec,model,len(permutations))
    count_after=sum(p.numel() for p in model.parameters())
    result={"decision":"PROCEED" if all(exact+detached+finite) and bank["all_derangements"] and count_before==count_after else "FOMD_ENGINEERING_BLOCKED",
            "endpoint":str(Path(a.endpoint).resolve()),"cohort_images":len(cohort),"F_full_equals_MOMD":all(exact),
            "counterfactuals_detached":all(detached),"all_finite":all(finite),"all_derangements":bank["all_derangements"],
            "permutation_bank_sha256":bank["sha256"],"parameter_delta":count_after-count_before,"summary":summary,
            "exploratory_only":True,"thresholds_modified":False}
    write_json(a.output,result); print(json.dumps(result,allow_nan=False),flush=True)
    if result["decision"]!="PROCEED": raise RuntimeError("FOMD historical audit blocked formal training")


if __name__=="__main__": main()
