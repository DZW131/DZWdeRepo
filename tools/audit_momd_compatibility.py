"""Read-only MOMD compatibility audit of the frozen CQRF endpoint and cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from network.momd_net import MOMDNet
from tools.momd_diagnostics import batch_health, summarize
from train_cqrf_phase0 import MonitorDataset


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",required=True); parser.add_argument("--cohort-json",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); payload=json.loads(Path(args.cohort_json).read_text(encoding="utf-8"))
    rows=[(x["path"],torch.tensor(x["label"],dtype=torch.float32)) for x in payload["images"]]
    model=MOMDNet().cuda().eval(); state=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    missing,unexpected=model.load_state_dict(state,strict=True); assert not missing and not unexpected
    batches=[]; pmec=[]
    loader=DataLoader(MonitorDataset(rows),batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    with torch.no_grad():
        for _,images,labels in loader:
            images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=0,run_pmec=True)
            batches.append(batch_health(result,labels)); pmec.extend(result["pmec_rows"])
    summary=summarize("frozen_cqrf_endpoint",batches,pmec,model)
    integrity=next(x for x in summary["routing_integrity"] if x["stage"]==3)
    capacity=next(x for x in summary["capacity_preservation"] if x["stage"]==3)
    decision="PROCEED" if integrity["all_finite"] and integrity["max_A_sum_error"]<=1e-5 and integrity["max_C_sum_error"]<=1e-6 and integrity["max_Q_sum_error"]<=1e-4 else "BLOCK"
    output={"audit":"read-only frozen CQRF endpoint MOMD compatibility","checkpoint":str(Path(args.checkpoint).resolve()),
            "cohort":str(Path(args.cohort_json).resolve()),"nontriviality":{"decision":decision},
            "routing_integrity":integrity,"capacity_reference":capacity,"summary":summary}
    Path(args.output).write_text(json.dumps(output,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"decision":decision,"routing":integrity,"capacity":capacity},allow_nan=False))
    if decision!="PROCEED": raise SystemExit(2)


if __name__=="__main__": main()
