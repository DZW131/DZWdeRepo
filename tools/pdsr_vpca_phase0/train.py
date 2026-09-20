"""Five-epoch frozen-HQMR/frozen-PLIP Phase0 trainer for P1/P2/P3."""
from __future__ import annotations
import argparse,csv,json,platform,sys,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import train_sshr as official
from network.pdsr_vpca_hqmr import PDSRVPCAHQMR
from tool.torchutils import PolyOptimizer
from tools.pdsr_vpca_phase0.common import ACCUMULATION,EPOCHS,MICRO_BATCH,STEPS_PER_EPOCH,TOTAL_STEPS,CommonAugmentTrainDataset,set_seed,sha256,write_json

def append_csv(path:Path,row:dict):
    path.parent.mkdir(parents=True,exist_ok=True); exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)

def grad_sums(model): return {n:(None if p.grad is None else float(p.grad.detach().float().abs().sum())) for n,p in model.named_parameters()}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--variant",choices=("P1","P2","P3"),required=True); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--plip",type=Path,required=True); p.add_argument("--train-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num-workers",type=int,default=4); p.add_argument("--smoke-effective-steps",type=int,default=0); a=p.parse_args()
    root=a.output; prep=json.loads((root/"manifests/preparation_complete.json").read_text())
    if not prep["pass"]: raise AssertionError("Preparation gate missing")
    variant_dir=root/{"P1":"P1_VLM_LAST","P2":"P2_STATIC_PDSR","P3":"P3_VPCA_PDSR"}[a.variant]
    if any(variant_dir.iterdir()): raise FileExistsError(f"Refusing populated variant output: {variant_dir}")
    set_seed(42); dataset=CommonAugmentTrainDataset(a.train_root); generator=torch.Generator().manual_seed(42)
    loader=DataLoader(dataset,batch_size=MICRO_BATCH,shuffle=True,num_workers=a.num_workers,pin_memory=True,drop_last=True,
                      worker_init_fn=official.seed_worker,generator=generator,persistent_workers=a.num_workers>0)
    if len(dataset)!=23422 or len(loader)!=4684: raise AssertionError("Frozen BCSS train cardinality changed")
    cache=torch.load(root/"concept_bank/concept_embeddings.pt",map_location="cpu",weights_only=False)
    model=PDSRVPCAHQMR(a.checkpoint,a.plip,cache["embeddings"],a.variant).cuda().train()
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    weights=[p for n,p in trainable if p.ndim>1]; scalars=[p for n,p in trainable if p.ndim<=1]
    optimizer=PolyOptimizer([{"params":weights,"lr":.1,"weight_decay":.0005},{"params":scalars,"lr":.1,"weight_decay":0.}],lr=.1,weight_decay=.0005,max_step=TOTAL_STEPS)
    target=a.smoke_effective_steps or TOTAL_STEPS; optimizer.zero_grad(set_to_none=True); started=time.perf_counter(); first_backward=None; post_update=None
    torch.cuda.reset_peak_memory_stats(); micro=0; rolling={}; rolling_count=0
    for epoch in range(1,EPOCHS+1):
        epoch_start=time.perf_counter()
        for names,raw,labels in loader:
            raw=raw.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): result=model(raw,labels); loss=result["losses"]["loss"]/ACCUMULATION
            if not torch.isfinite(loss): raise FloatingPointError("Non-finite Phase0 loss")
            loss.backward(); micro+=1; rolling_count+=1
            for key in ("loss","loss_hqmr","loss_semantic"):
                rolling[key]=rolling.get(key,0.)+float(result["losses"][key].detach())
            if first_backward is None: first_backward=grad_sums(model)
            if micro%ACCUMULATION: continue
            if optimizer.global_step==1 and post_update is None: post_update=grad_sums(model)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            step=optimizer.global_step
            if step==1 and post_update is None: pass
            if step%100==0 or step==target:
                aux=result["phase0"]["pdsr"]; provider=result["phase0"]["provider"]
                beta=aux.get("layer_weights"); entropy=provider.get("concept_entropy")
                row={"variant":a.variant,"epoch":epoch,"step":step,"total_loss":rolling["loss"]/rolling_count,
                     "hqmr_loss":rolling["loss_hqmr"]/rolling_count,"semantic_loss":rolling["loss_semantic"]/rolling_count,
                     "gamma_abs_mean":float(model.gamma.detach().abs().mean()),"beta_entropy":float((-(beta.clamp_min(1e-8)*beta.clamp_min(1e-8).log()).sum(1).mean()).detach()) if beta is not None else 0.,
                     "concept_entropy":float(entropy.mean().detach()) if entropy is not None else 0.,"lr":optimizer.param_groups[0]["lr"],
                     "gpu_memory_gib":torch.cuda.max_memory_allocated()/1024**3}
                append_csv(variant_dir/"train_log.csv",row); print("PDSR_PHASE0_STEP "+json.dumps(row),flush=True); rolling={}; rolling_count=0
            if step>=target: break
        append_csv(variant_dir/"epoch_summary.csv",{"variant":a.variant,"epoch":epoch,"step":optimizer.global_step,"seconds":time.perf_counter()-epoch_start,
                   "gamma_abs_mean":float(model.gamma.detach().abs().mean()),"peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3})
        if optimizer.global_step>=target: break
    # Obtain post-update adapter gradients explicitly if the short run reached only one update.
    if post_update is None and optimizer.global_step>=1:
        names,raw,labels=next(iter(loader)); optimizer.zero_grad(set_to_none=True); raw=raw.cuda(); labels=labels.cuda()
        with torch.autocast("cuda",dtype=torch.bfloat16): model(raw,labels)["losses"]["loss"].backward()
        post_update=grad_sums(model); optimizer.zero_grad(set_to_none=True)
    base_bad=[n for n,v in first_backward.items() if n.startswith("base.") and v is not None]
    plip_bad=[n for n,v in first_backward.items() if n.startswith("plip.") and v is not None]
    new_first={n:v for n,v in first_backward.items() if not n.startswith(("base.","plip."))}
    new_post={n:v for n,v in (post_update or {}).items() if not n.startswith(("base.","plip."))}
    gradient={"variant":a.variant,"frozen_hqmr_grad_none":not base_bad,"frozen_plip_grad_none":not plip_bad,"unexpected_hqmr":base_bad,"unexpected_plip":plip_bad,
              "new_first_backward":new_first,"new_after_first_update":new_post,
              "contract_note":"P1 P_last and P2/P3 Ps are blocked by exact gamma=0 on first backward; they must become nonzero after the first gamma update."}
    if base_bad or plip_bad or not (new_first.get("gamma") and new_first["gamma"]>0): raise AssertionError(f"Frozen/gradient contract failed: {gradient}")
    required_after=[n for n in new_post if "projection" in n]
    if not required_after or any(new_post[n] is None or new_post[n]<=0 for n in required_after): raise AssertionError(f"Adapter projection gradients missing: {gradient}")
    write_json(variant_dir/"gradient_manifest.json",gradient)
    state_path=variant_dir/f"{a.variant.lower()}_e5_adapter.pth"; torch.save(model.trainable_state_dict(),state_path)
    runtime={"status":"SMOKE_COMPLETE" if a.smoke_effective_steps else "TRAINING_COMPLETE","variant":a.variant,"epochs":epoch,"optimizer_steps":optimizer.global_step,
             "effective_batch":20,"micro_batch":MICRO_BATCH,"accumulation":ACCUMULATION,"train_seconds":time.perf_counter()-started,
             "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3,"checkpoint":str(state_path),"checkpoint_sha256":sha256(state_path),
             "validation_accessed":False,"segmentation_gt_accessed":False,"environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"platform":platform.platform()}}
    write_json(variant_dir/"runtime.json",runtime); print("PDSR_PHASE0_TRAINING_DONE "+json.dumps(runtime),flush=True)
if __name__=="__main__": main()
