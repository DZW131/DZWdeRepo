"""Read-only diagnostic comparing HQMR observable and DLAG bank inference paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from audits.dlag_oracle_v1.oracle_bank import infer_alpha_bank
from audits.oracle_action_separability_v1 import freeze_observables as freeze
from audits.ucrf_v1.gate import load_model
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import TTA,normalize_cam,resize_unflip
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--val-root",type=Path,required=True);p.add_argument("--index",type=int,required=True)
    args=p.parse_args()
    model=load_model(args.checkpoint)
    dataset=Stage1_InferDataset(str(args.val_root/"img"),img_size=224)
    name,img=dataset[args.index]
    x=img[None].cuda()
    original=model.forward
    capture=[]
    def wrapped(*a,**kw):
        output=original(*a,**kw)
        capture.append(output["primary_output"].detach().float().cpu().numpy().copy())
        return output
    model.forward=wrapped
    spatial_calls=[]
    original_spatial=freeze.spatial
    def spy(value,hw,flip):
        result=original_spatial(value,hw,flip)
        spatial_calls.append(result.copy())
        return result
    freeze.spatial=spy
    observed=freeze.infer(model,x); a=capture.copy();capture.clear()
    freeze.spatial=original_spatial
    reference=infer_alpha_bank(model,x,(224,224));b=capture.copy()
    print("name",name)
    print("forward_per_view_max_abs",[float(np.max(np.abs(i-j))) for i,j in zip(a,b)])
    print("cam_max_abs",float(np.max(np.abs(observed["maps"]["cam"]-reference["baseline_cam"]))))
    print("gate_max_abs",float(np.max(np.abs(observed["gate"]-reference["gate_score"]))))
    rec=torch.stack([resize_unflip(torch.from_numpy(value).cuda(),(224,224),flip).float().cpu()
                     for value,(_,flip) in zip(a,TTA)]).mean(0).numpy()
    normalized=normalize_cam(rec)
    print("manual_vs_observed",float(np.max(np.abs(normalized-observed["maps"]["cam"]))))
    print("manual_vs_reference",float(np.max(np.abs(normalized-reference["baseline_cam"]))))
    print("spatial_calls",len(spatial_calls))
    for i,(value,(_,flip)) in enumerate(zip(a,TTA)):
        manual=resize_unflip(torch.from_numpy(value).cuda(),(224,224),flip).float().cpu().numpy()
        print("view",i,"vs_spatial_cam",float(np.max(np.abs(manual-spatial_calls[i*6+5]))),
              "c3_cam",float(np.max(np.abs(spatial_calls[i*6+4]-spatial_calls[i*6+5]))))


if __name__=="__main__":main()
