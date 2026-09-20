"""Fixed-32-image upstream-to-downstream causal tensor and SHA-256 audit."""
from __future__ import annotations
import argparse,hashlib,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from network.pcsi_net import PCSINet
from tools.pcsi_v1.identity import collect
from tools.pdsr_vpca_phase0.common import CommonEvalDataset,write_json


def captured_forward(model,raw,labels,fused):
    base=model.base; values={}
    modules={"ccra_k5":base.ccra2.k_projection,"ccra_v5":base.ccra2.v_projection,
             "hqmr_k5":base.hqmr.scale5.key,"hqmr_v5":base.hqmr.scale5.value}
    handles=[module.register_forward_hook(lambda _m,_i,out,key=key:values.setdefault(key,[]).append(out.detach()))
             for key,module in modules.items()]
    try:
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            if fused:out=model(raw,labels)
            else:out=base((raw-model.hqmr_mean.to(raw))/model.hqmr_std.to(raw),labels,step=29275)
    finally:
        for h in handles:h.remove()
    result=collect(out)
    for key,rows in values.items(): result[key]=rows[-1]
    return result


def digest(tensor):
    return hashlib.sha256(tensor.float().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--plip",type=Path,required=True); p.add_argument("--concept-cache",type=Path,required=True)
    p.add_argument("--val-images",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); dataset=CommonEvalDataset(a.val_images)
    loader=DataLoader(dataset,batch_size=4,shuffle=False,num_workers=2)
    cache=torch.load(a.concept_cache,map_location="cpu",weights_only=False); report={}
    for variant,directory in (("C1","C1_VLM_LAST"),("C2","C2_STATIC_PDSR")):
        model=PCSINet(a.checkpoint,a.plip,cache["embeddings"],variant).cuda().eval()
        model.load_trainable_state_dict(torch.load(a.output/directory/f"{variant.lower()}_e5_adapter.pth",map_location="cpu",weights_only=False))
        stats={}; hashes=[]; ids=[]; changed=total=0
        for names,raw in loader:
            raw=raw.cuda(); labels=torch.ones((len(names),4),device="cuda")
            before=captured_forward(model,raw,labels,False)
            after=captured_forward(model,raw,labels,True)
            for key in before:
                d=(after[key].float()-before[key].float()).detach()
                s=stats.setdefault(key,{"max_abs":0.,"sum_square":0.,"elements":0})
                s["max_abs"]=max(s["max_abs"],float(d.abs().max()))
                s["sum_square"]+=float(d.square().sum()); s["elements"]+=d.numel()
                hashes.append({"image_ids":list(names),"tensor":key,"C0_sha256":digest(before[key]),
                               f"{variant}_sha256":digest(after[key])})
            b=before["ccra2_responsibility"].argmax(1)
            f=after["ccra2_responsibility"].argmax(1)
            changed+=int((b!=f).sum()); total+=b.numel(); ids.extend(str(x) for x in names)
            if len(ids)>=32:break
        stats={k:{"max_abs":v["max_abs"],"rms":(v["sum_square"]/v["elements"])**.5,
                  "elements":v["elements"]} for k,v in stats.items()}
        if stats["query1"]["max_abs"]!=0 or stats["context_pre"]["max_abs"]!=0:
            raise AssertionError("Pre-injection tensors changed")
        report[variant]={"image_ids":ids,"tensor_drift":stats,"ccra_spatial_argmax_change":changed/total,
                         "tensor_hashes":hashes,"path_effective":stats["ccra_k5"]["max_abs"]>0 and
                         stats["ccra_v5"]["max_abs"]>0 and stats["hqmr5_logits"]["max_abs"]>0}
        del model; torch.cuda.empty_cache()
    write_json(a.output/"phaseC/posttrain_tensor_hash_audit.json",report)
    print("PCSI_TENSOR_HASH_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":main()
