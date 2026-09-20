"""PCSI C1/C2 five-epoch training; frozen HQMR/PLIP, original Phase0 optimizer."""
from __future__ import annotations
import argparse,csv,json,platform,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import train_sshr as official
from network.pcsi_net import PCSINet
from tool.torchutils import PolyOptimizer
from tools.pdsr_vpca_phase0.common import ACCUMULATION,EPOCHS,MICRO_BATCH,TOTAL_STEPS,CommonAugmentTrainDataset,set_seed,sha256,write_json


def append_csv(path:Path,row:dict):
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if not exists: writer.writeheader()
        writer.writerow(row)


def grad_sums(model):
    return {n:(None if p.grad is None else float(p.grad.detach().float().abs().sum())) for n,p in model.named_parameters()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--variant",choices=("C1","C2"),required=True)
    p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--plip",type=Path,required=True)
    p.add_argument("--concept-cache",type=Path,required=True); p.add_argument("--train-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--smoke-effective-steps",type=int,default=0); a=p.parse_args()
    gate=json.loads((a.output/"phaseA_forward/phaseA_decision.json").read_text())
    if gate["PHASE_A_DECISION"]!="GO": raise AssertionError("Phase A GO required")
    variant_dir=a.output/("C1_VLM_LAST" if a.variant=="C1" else "C2_STATIC_PDSR")
    variant_dir.mkdir(parents=True,exist_ok=True)
    if any(variant_dir.iterdir()): raise FileExistsError(f"Refusing populated output: {variant_dir}")
    set_seed(42); dataset=CommonAugmentTrainDataset(a.train_root); generator=torch.Generator().manual_seed(42)
    loader=DataLoader(dataset,batch_size=MICRO_BATCH,shuffle=True,num_workers=a.num_workers,pin_memory=True,drop_last=True,
                      worker_init_fn=official.seed_worker,generator=generator,persistent_workers=a.num_workers>0)
    if len(dataset)!=23422 or len(loader)!=4684: raise AssertionError("BCSS train cardinality changed")
    cache=torch.load(a.concept_cache,map_location="cpu",weights_only=False)
    model=PCSINet(a.checkpoint,a.plip,cache["embeddings"],a.variant).cuda().train()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    weights=[p for n,p in trainable if p.ndim>1]; scalars=[p for n,p in trainable if p.ndim<=1]
    optimizer=PolyOptimizer([{"params":weights,"lr":.1,"weight_decay":.0005},{"params":scalars,"lr":.1,"weight_decay":0.}],
                            lr=.1,weight_decay=.0005,max_step=TOTAL_STEPS)
    target=a.smoke_effective_steps or TOTAL_STEPS
    optimizer.zero_grad(set_to_none=True); started=time.perf_counter(); first_backward=None; post_update=None
    torch.cuda.reset_peak_memory_stats(); micro=0; rolling={}; rolling_count=0
    for epoch in range(1,EPOCHS+1):
        epoch_start=time.perf_counter()
        for names,raw,labels in loader:
            raw=raw.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): result=model(raw,labels); loss=result["losses"]["loss"]/ACCUMULATION
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite PCSI loss")
            loss.backward(); micro+=1; rolling_count+=1
            for key in ("loss","loss_hqmr","loss_semantic"):
                rolling[key]=rolling.get(key,0.)+float(result["losses"][key].detach())
            if first_backward is None: first_backward=grad_sums(model)
            if micro%ACCUMULATION: continue
            if optimizer.global_step==1 and post_update is None: post_update=grad_sums(model)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step=optimizer.global_step
            if step%100==0 or step==target:
                beta=result["phase0"]["pdsr"].get("layer_weights")
                row={"variant":a.variant,"epoch":epoch,"step":step,"total_loss":rolling["loss"]/rolling_count,
                     "hqmr_loss":rolling["loss_hqmr"]/rolling_count,"semantic_loss":rolling["loss_semantic"]/rolling_count,
                     "gamma_abs_mean":float(model.gamma.detach().abs().mean()),
                     "beta_entropy":float((-(beta.clamp_min(1e-8)*beta.clamp_min(1e-8).log()).sum(1).mean()).detach()) if beta is not None else 0.,
                     "lr":optimizer.param_groups[0]["lr"],"gpu_memory_gib":torch.cuda.max_memory_allocated()/1024**3}
                append_csv(variant_dir/"train_log.csv",row); print("PCSI_TRAIN_STEP "+json.dumps(row),flush=True); rolling={}; rolling_count=0
            if step>=target: break
        append_csv(variant_dir/"epoch_summary.csv",{"variant":a.variant,"epoch":epoch,"step":optimizer.global_step,
                   "seconds":time.perf_counter()-epoch_start,"gamma_abs_mean":float(model.gamma.detach().abs().mean()),
                   "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3})
        if optimizer.global_step>=target: break
    frozen_bad=[n for n,v in first_backward.items() if n.startswith(("base.","plip.")) and v is not None]
    new_first={n:v for n,v in first_backward.items() if not n.startswith(("base.","plip."))}
    new_post={n:v for n,v in (post_update or {}).items() if not n.startswith(("base.","plip."))}
    gradient={"variant":a.variant,"frozen_grad_none":not frozen_bad,"unexpected_frozen":frozen_bad,
              "new_first_backward":new_first,"new_after_first_update":new_post,
              "contract_note":"Exact gamma=0 blocks projection on first backward, then gamma update opens path."}
    if frozen_bad or not (new_first.get("gamma") and new_first["gamma"]>0): raise AssertionError(gradient)
    projection=[n for n in new_post if "projection" in n]
    if not projection or any(new_post[n] is None or new_post[n]<=0 for n in projection): raise AssertionError(gradient)
    write_json(variant_dir/"gradient_manifest.json",gradient)
    state_path=variant_dir/f"{a.variant.lower()}_e5_adapter.pth"; torch.save(model.trainable_state_dict(),state_path)
    runtime={"status":"SMOKE_COMPLETE" if a.smoke_effective_steps else "TRAINING_COMPLETE","variant":a.variant,
             "epochs":epoch,"optimizer_steps":optimizer.global_step,"effective_batch":20,"micro_batch":MICRO_BATCH,
             "accumulation":ACCUMULATION,"train_seconds":time.perf_counter()-started,
             "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3,"checkpoint":str(state_path),
             "checkpoint_sha256":sha256(state_path),"validation_accessed":False,"segmentation_gt_accessed":False,
             "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"platform":platform.platform()}}
    write_json(variant_dir/"runtime.json",runtime); print("PCSI_TRAINING_DONE "+json.dumps(runtime),flush=True)


if __name__=="__main__": main()
