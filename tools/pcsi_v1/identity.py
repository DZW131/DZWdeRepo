"""Pre-training zero-gamma identity and positive-gamma CCRA causal check, without GT."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from network.pcsi_net import PCSINet
from tools.pdsr_vpca_phase0.common import CommonEvalDataset,write_json


def collect(out):
    return {"query1":out["stages"][0]["query"],
            "context_pre":out["query_detail"]["context_feature_pre"],
            "context":out["query_detail"]["context_feature"],
            "ccra2_responsibility":out["stages"][1]["detail"]["responsibility_class"],
            "ccra3_responsibility":out["stages"][2]["detail"]["responsibility_class"],
            "hqmr_query0":out["stages"][2]["hqmr"]["query0"],
            "hqmr5_logits":out["stages"][2]["hqmr"]["logits5"],
            "hqmr_query5":out["stages"][2]["hqmr"]["query5"],
            "hqmr4_logits":out["stages"][2]["hqmr"]["logits4"],
            "hqmr_query4":out["stages"][2]["hqmr"]["query4"],
            "hqmr3_logits":out["stages"][2]["hqmr"]["logits3"],
            "final":out["primary_output"]}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--plip",type=Path,required=True); p.add_argument("--concept-cache",type=Path,required=True)
    p.add_argument("--val-images",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); cache=torch.load(a.concept_cache,map_location="cpu",weights_only=False)
    dataset=CommonEvalDataset(a.val_images); loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2)
    result={}
    for mode in ("C1","C2"):
        model=PCSINet(a.checkpoint,a.plip,cache["embeddings"],mode).cuda().eval()
        if torch.count_nonzero(model.gamma): raise AssertionError("gamma must initialize exactly zero")
        max_drift={}; positive={}; ids=[]
        for names,raw in loader:
            raw=raw.cuda(); labels=torch.ones((len(names),4),device="cuda")
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                mean=model.hqmr_mean.to(raw); std=model.hqmr_std.to(raw)
                reference=collect(model.base((raw-mean)/std,labels,step=29275))
                identity=collect(model(raw,labels))
                model.gamma.fill_(.1)
                perturbed=collect(model(raw,labels))
                model.gamma.zero_()
            for key in reference:
                max_drift[key]=max(max_drift.get(key,0.),float((reference[key].float()-identity[key].float()).abs().max()))
                positive[key]=max(positive.get(key,0.),float((reference[key].float()-perturbed[key].float()).abs().max()))
            ids.extend(str(x) for x in names)
            if len(ids)>=32: break
        if any(value!=0 for value in max_drift.values()): raise AssertionError(f"{mode} gamma0 identity failed: {max_drift}")
        if positive["ccra2_responsibility"]<=0 or positive["hqmr5_logits"]<=0: raise AssertionError(f"{mode} positive gamma path failed: {positive}")
        if positive["query1"]!=0 or positive["context_pre"]!=0: raise AssertionError(f"{mode} upstream drift: {positive}")
        result[mode]={"image_ids":ids,"gamma0_max_abs_drift":max_drift,"gamma0_identity":True,
                      "gamma0_argmax_identity":True,"gamma0_labels_used":"dummy all-one, no GT",
                      "gamma0_1_positive_max_abs_drift":positive}
        del model; torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True,exist_ok=True); write_json(a.output,result)
    print("PCSI_IDENTITY_GO",flush=True)


if __name__=="__main__": main()
