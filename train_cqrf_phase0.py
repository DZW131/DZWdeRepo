"""Train-only CQRF CCRA responsibility gate on BCSS Seed42."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import train_sshr as official
from network.cqrf_net import CQRFNet, STAGE_WEIGHTS
from network.hqrf_targets import FULL25_STEPS
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.cqrf_diagnostics import apply_epoch2_screen, apply_final_gate, batch_health, summarize
from tools.cqrf_report import render_report
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json


ROOT = Path(__file__).resolve().parent
STEPS_PER_EPOCH = 1171
PHASE0_STEPS = 5855
SNAPSHOTS = {250:"step0250",500:"step0500",1000:"step1000",1171:"epoch1",2342:"epoch2",3513:"epoch3",4684:"epoch4",5855:"epoch5"}
VISUAL_SNAPSHOTS = {500,1000,2342,3513,4684,5855}
CONFIG = {
    "experiment":"CQRF-Net Phase-0 CCRA Responsibility Allocation","dataset":"BCSS training only","seed":42,
    "gpu":"RTX4090D","batch":20,"epochs_max":5,"steps_per_epoch":STEPS_PER_EPOCH,"phase0_steps_max":PHASE0_STEPS,
    "full25_schedule_denominator":FULL25_STEPS,"amp":"bf16","image_size":224,"query_grid":[14,14],"query_count":196,
    "dimension":256,"patch_size":16,"stage1_order":["cross_attention","norm","self_attention","norm","ffn","norm"],
    "stage2_stage3":"CCRA then FFN; no self-attention","ccra":{"lambda_class":1.0,"temperature":1.0,"epsilon":1e-8,"softmax_dimension":"query","prior":"detach(Pclass)*detach(sigmoid(deep_logits))","fallback":"normalized Pclass if all gates tiny"},
    "memory_position":"parameter-free normalized 2-D sine/cosine Kp","memory_detach":"detach CNN-side input before trainable projections",
    "f4_chpf":"one coherent 128-channel transform feeding separate 256 memory and 256 pixel projections",
    "stage_loss_weights":list(STAGE_WEIGHTS),"loss_weights":{"deep":.5,"PCA":.25,"mask":.25},
    "pseudo":{"positive_floor":.60,"top_ratio":.15,"class_margin":.10,"background_ceiling":.10,"positive_dilation":1,"min_labeled_pixels":4},
    "locality":{"radii":[1,5],"denominator":FULL25_STEPS},"pmec":{"tau_bin":.70,"tau_low":.40,"tau_high":.50,"T":5,"stage":3},
    "optimizer":{"base_lr":.01,"weight_decay":.0005,"multipliers":[1,2,10,20],"poly_max_step":PHASE0_STEPS},
    "monitor_snapshots":list(SNAPSHOTS),"validation_access":False,
    "prohibited":["responsibility entropy loss","diversity loss","repulsion","orthogonality","Sinkhorn","balanced assignment","hard assignment","GT-label routing"],
}


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


class MonitorDataset(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        from PIL import Image
        path,label=self.rows[index]; image=Image.open(path).convert("RGB").resize((224,224))
        value=transforms.functional.to_tensor(image); value=transforms.functional.normalize(value,[.485,.456,.406],[.229,.224,.225])
        return Path(path).stem,value,label.float()


def select_monitor_cohort(rows):
    single=[r for r in rows if int(r[1].sum())==1]; multi=[r for r in rows if int(r[1].sum())>=2]; rng=random.Random(42)
    rng.shuffle(single); rng.shuffle(multi); chosen=single[:8]+multi[:16]; used={str(r[0]) for r in chosen}
    rest=[r for r in rows if str(r[0]) not in used]; rng.shuffle(rest); chosen+=rest[:32-len(chosen)]
    if len(chosen)!=32 or sum(int(r[1].sum())==1 for r in chosen)<8 or sum(int(r[1].sum())>=2 for r in chosen)<16:
        raise RuntimeError("Unable to freeze the preregistered cohort")
    return chosen


def gradient_health(model,step):
    buckets=defaultdict(list); zeros=defaultdict(list); nonfinite=defaultdict(list)
    for name,p in model.named_parameters():
        if p.grad is None: continue
        if name.startswith("backbone."): group="backbone"
        elif name.startswith("deep_head."): group="deep"
        elif name.startswith("pixel_decoder."): group="pixel"
        elif name.startswith("mask_embeddings."): group="mask"
        elif name.startswith("pca_heads."): group="pca"
        elif name.startswith(("ccra2.","ccra3.")): group="ccra"
        else: group="query"
        grad=p.grad.detach().float(); finite=torch.isfinite(grad)
        buckets[group].append(float(torch.where(finite,grad,torch.zeros_like(grad)).square().mean().sqrt()))
        zeros[group].append(float((grad==0).float().mean())); nonfinite[group].append(float((~finite).float().mean()))
    rows=[]
    for group,values in buckets.items():
        rows.append({"step":step,"module":group,"grad_rms_mean":float(np.mean(values)),"grad_rms_p50":float(np.median(values)),"grad_rms_p90":float(np.quantile(values,.9)),"zero_fraction":float(np.mean(zeros[group])),"nonfinite_fraction":float(np.mean(nonfinite[group]))})
    return rows


def _rgb(images):
    mean=images.new_tensor([.485,.456,.406])[None,:,None,None]; std=images.new_tensor([.229,.224,.225])[None,:,None,None]
    return (images*std+mean).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()


def save_visualizations(directory,snapshot,names,images,labels,output):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=Path(directory)/"visualizations"/snapshot; target.mkdir(parents=True,exist_ok=True); rgb=_rgb(images)
    for image in range(min(8,len(names))):
        panels=[(rgb[image],"input"),(output["normalized_cam"][image].max(0).values.float().cpu().numpy(),"deep CAM"),(output["targets"][image].float().amax(0).cpu().numpy(),"tri-state")]
        tops=[]
        for stage_index,stage in enumerate(output["stages"],1):
            score=stage["confidence"]["joint"][image].float().max(-1).values; top=torch.argsort(score,descending=True,stable=True)[:5]; tops.append(top)
            for rank,index in enumerate(top): panels.append((stage["mask_logits"][image,index].float().sigmoid().cpu().numpy(),f"S{stage_index} mask {rank+1}"))
        for stage_index in (2,3):
            stage=output["stages"][stage_index-1]; cls=int(torch.where(labels[image].bool())[0][0]); detail=stage["detail"]; h,w=stage["memory_hw"]
            mass=detail["responsibility_class"][image,:,:,cls].sum(-1); top=torch.argsort(mass,descending=True,stable=True)[:5]
            for rank,index in enumerate(top): panels.append((detail["responsibility_class"][image,index,:,cls].reshape(h,w).cpu().numpy(),f"S{stage_index} R {rank+1}"))
            owner=detail["responsibility_class"][image,:,:,cls].argmax(0).reshape(h,w).float().cpu().numpy(); panels.append((owner,f"S{stage_index} owner"))
        assignment=output["stages"][2]["confidence"]["p_class"][image].argmax(-1).reshape(14,14).float().cpu().numpy()
        panels += [(assignment,"S3 PCA"),(output["pmec_region"][image].amax(0).float().cpu().numpy(),"S3 PMEC")]
        figure,axes=plt.subplots(6,6,figsize=(18,18))
        for axis in axes.flat: axis.axis("off")
        for axis,(value,title) in zip(axes.flat,panels): axis.imshow(value,cmap=None if value.ndim==3 else "viridis"); axis.set_title(title)
        figure.suptitle(f"{names[image]} | present={torch.where(labels[image].bool())[0].tolist()}"); figure.tight_layout(); figure.savefig(target/f"{names[image]}.png",dpi=110); plt.close(figure)


@torch.no_grad()
def monitor(model,loader,step,snapshot,histories,summary_history,output_dir,visualize=False):
    was_training=model.training; model.eval(); batches=[]; pmec_rows=[]; first=None
    for names,images,labels in loader:
        images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=step,run_pmec=True)
        batches.append(batch_health(result,labels)); pmec_rows.extend(result["pmec_rows"])
        if first is None: first=(names,images.detach(),labels.detach(),result)
    summary=summarize(snapshot,batches,pmec_rows,model); summary_history.append(summary)
    mapping={"stagewise_mask_area":"cqrf_stagewise_mask_area.csv","stagewise_query_redundancy":"cqrf_stagewise_query_redundancy.csv","stagewise_embedding_diversity":"cqrf_stagewise_embedding_diversity.csv","responsibility_integrity":"cqrf_responsibility_integrity.csv","responsibility_utilization":"cqrf_responsibility_utilization.csv","responsibility_complementarity":"cqrf_responsibility_complementarity.csv","query_update_health":"cqrf_query_update_health.csv","over_fragmentation":"cqrf_over_fragmentation.csv","semantic_selectivity":"cqrf_semantic_selectivity.csv","pca_health":"cqrf_pca_health.csv","pmec_health":"cqrf_pmec_health.csv","chpf_health":"cqrf_chpf_health.csv","deep_gate_health":"cqrf_deep_gate_health.csv"}
    for key,filename in mapping.items():
        rows=summary[key] if isinstance(summary[key],list) else [summary[key]]; histories[key].extend(rows); write_csv(Path(output_dir)/filename,histories[key])
    temporal=[r for r in histories["stagewise_query_redundancy"] if r["stage"]==3 and str(r["snapshot"]).startswith("epoch")]
    if temporal: write_csv(Path(output_dir)/"cqrf_temporal_redundancy.csv",temporal)
    if visualize and first is not None: save_visualizations(output_dir,snapshot,*first)
    print("CQRF_MONITOR "+json.dumps(summary,allow_nan=False),flush=True)
    if was_training: model.train()
    return summary


def checkpoint(model,output):
    path=Path(output)/"cqrf_phase0_endpoint.pth"; torch.save(model.state_dict(),path); digest=sha256(path)
    (Path(output)/"cqrf_phase0_endpoint_sha256.txt").write_text(digest+"\n",encoding="utf-8"); return path,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--trainroot",required=True); parser.add_argument("--weights",required=True); parser.add_argument("--output",required=True); parser.add_argument("--smoke-steps",type=int,choices=(0,2),default=0); args=parser.parse_args()
    check_train_path(args.trainroot); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("Native CUDA BF16 required")
    if "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("Registered RTX4090D required")
    protected=protected_sources(ROOT); accesses=install_train_access_guard(); official.set_seed(42); output.mkdir(parents=True)
    log_handle=(output/"cqrf_phase0_train.log").open("w",encoding="utf-8",buffering=1); sys.stdout=Tee(sys.__stdout__,log_handle); sys.stderr=Tee(sys.__stderr__,log_handle)
    source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); config={**CONFIG,"source_commit":source_commit,"smoke_steps":args.smoke_steps,"trainroot":str(Path(args.trainroot).resolve())}
    write_json(output/"cqrf_phase0_config.json",config); config_hash=sha256(output/"cqrf_phase0_config.json"); (output/"cqrf_phase0_config_sha256.txt").write_text(config_hash+"\n",encoding="utf-8"); (output/"cqrf_phase0_source_commit.txt").write_text(source_commit+"\n",encoding="utf-8")
    sources=["network/cqrf_query.py","network/cqrf_net.py","tools/cqrf_diagnostics.py","tools/cqrf_report.py","train_cqrf_phase0.py"]
    (output/"cqrf_phase0_source_snapshot.txt").write_text("\n".join(f"{sha256(ROOT/p)}  {p}" for p in sources)+"\n",encoding="utf-8")
    model=CQRFNet().cuda(); init=model.backbone.load_official_initialization(args.weights); write_json(output/"cqrf_phase0_init_identity.json",init)
    module_counts={name:sum(p.numel() for p in module.parameters()) for name,module in (("backbone",model.backbone),("deep",model.deep_head),("query",model.patch_queries),("stage1",model.decoder1),("ccra2",model.ccra2),("ccra3",model.ccra3),("pixel",model.pixel_decoder),("mask",model.mask_embeddings),("pca",model.pca_heads))}
    write_json(output/"cqrf_phase0_parameter_counts.json",{**module_counts,"total":sum(p.numel() for p in model.parameters())})
    dataset=Stage1_TrainDataset(args.trainroot,transform=transforms.Compose([transforms.ToTensor()]),dataset="bcss",img_size=224)
    if len(dataset)!=23422: raise RuntimeError("Expected 23,422 BCSS training images")
    cohort=select_monitor_cohort(dataset.object); write_json(output/"cqrf_phase0_monitor_cohort.json",{"seed":42,"images":[{"path":str(Path(p).resolve()),"label":[int(v) for v in y.tolist()]} for p,y in cohort]})
    monitor_loader=DataLoader(MonitorDataset(cohort),batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    generator=torch.Generator().manual_seed(42); loader=DataLoader(dataset,batch_size=20,shuffle=True,num_workers=8,pin_memory=True,drop_last=True,worker_init_fn=official.seed_worker,generator=generator)
    if len(loader)!=STEPS_PER_EPOCH: raise RuntimeError("Expected 1,171 steps per epoch")
    groups=model.get_parameter_groups(); optimizer=PolyOptimizer([{"params":g,"lr":.01*m,"weight_decay":d} for g,m,d in zip(groups,(1,2,10,20),(.0005,0,.0005,0))],lr=.01,weight_decay=.0005,max_step=PHASE0_STEPS)
    write_json(output/"cqrf_phase0_provenance.json",{"protected_sources":protected,"environment":{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},"argv":sys.argv,"config_sha256":config_hash})
    print("CQRF_PROTOCOL "+json.dumps(config,allow_nan=False),flush=True)
    history_keys=("stagewise_mask_area","stagewise_query_redundancy","stagewise_embedding_diversity","responsibility_integrity","responsibility_utilization","responsibility_complementarity","query_update_health","over_fragmentation","semantic_selectivity","pca_health","pmec_health","chpf_health","deep_gate_health")
    histories={k:[] for k in history_keys}; summary_history=[]; losses=[]; gradients=[]; completed=0; started=time.perf_counter(); final_summary=None; epoch2_screen=None; gate=None
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        for epoch in range(1,6):
            epoch_rows=[]
            for _,images,labels in loader:
                images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=optimizer.global_step,run_pmec=False)
                loss=result["losses"]["loss"]
                if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Nonfinite loss at step {optimizer.global_step+1}")
                loss.backward(); next_step=optimizer.global_step+1
                if next_step%100==0 or next_step in SNAPSHOTS:
                    new_grad=gradient_health(model,next_step); gradients.extend(new_grad); write_csv(output/"cqrf_gradient_health.csv",gradients)
                    if any(r["nonfinite_fraction"]>0 for r in new_grad): raise FloatingPointError("Nonfinite gradient")
                optimizer.step(); step=optimizer.global_step
                row={"epoch":epoch,"step":step,**{k:float(v.detach()) for k,v in result["losses"].items()},"lr":optimizer.param_groups[0]["lr"]}; epoch_rows.append(row)
                if step%100==0 or step in SNAPSHOTS:
                    losses.append(row); write_csv(output/"cqrf_losses.csv",losses); print("CQRF_STEP "+json.dumps({**row,"peak_memory":torch.cuda.max_memory_allocated(),"elapsed_seconds":time.perf_counter()-started}),flush=True)
                if step in SNAPSHOTS: final_summary=monitor(model,monitor_loader,step,SNAPSHOTS[step],histories,summary_history,output,step in VISUAL_SNAPSHOTS)
                if args.smoke_steps and step>=args.smoke_steps: break
            completed=epoch; print("CQRF_EPOCH "+json.dumps({"epoch":epoch,"step":optimizer.global_step,**{k:float(np.mean([r[k] for r in epoch_rows])) for k in ("loss","loss_deep","loss_pca","loss_mask")}}),flush=True)
            if args.smoke_steps: break
            if epoch==2:
                epoch2_screen=apply_epoch2_screen(final_summary); write_json(output/"cqrf_phase0_epoch2_screen.json",epoch2_screen)
                if epoch2_screen["decision"]=="CQRF_CCRA_NOGO": break
        elapsed=time.perf_counter()-started
        if args.smoke_steps:
            runtime={"smoke":True,"steps":optimizer.global_step,"epochs":completed,"all_finite":True,"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"validation_accessed":False,"training_paths_opened_parent":len(accesses)}; write_json(output/"cqrf_phase0_runtime.json",runtime); print("CQRF_SMOKE_PASS "+json.dumps(runtime),flush=True); return
        endpoint,digest=checkpoint(model,output)
        if epoch2_screen and epoch2_screen["decision"]=="CQRF_CCRA_NOGO": gate=epoch2_screen; decision="CQRF_CCRA_NOGO"
        else: gate=apply_final_gate(summary_history); decision=gate["decision"]
        runtime={"smoke":False,"steps":optimizer.global_step,"epochs":completed,"all_finite":bool(final_summary["all_finite"]),"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_memory_gib":torch.cuda.max_memory_allocated()/1024**3,"validation_accessed":False,"test_accessed":False,"luad_accessed":False,"training_samples_consumed":optimizer.global_step*20,"training_paths_opened_parent":len(accesses),"decision":decision,"checkpoint":str(endpoint),"checkpoint_sha256":digest}
        write_json(output/"cqrf_phase0_runtime.json",runtime); result={**runtime,"source_commit":source_commit,"final_summary":final_summary,"summary_history":summary_history,"epoch2_screen":epoch2_screen,"gate":gate}; write_json(output/"cqrf_phase0_gate_result.json",result)
        report=render_report(output,result); print("CQRF_FINAL "+json.dumps({"decision":decision,"report":str(report),"gate":gate}),flush=True); print(f"DECISION = {decision}",flush=True)
    except Exception as error:
        failure={"decision":"CQRF_ENGINEERING_BLOCKED","error":repr(error),"source_commit":source_commit,"epochs":completed,"steps":optimizer.global_step,"all_finite":False,"final_summary":final_summary or {},"epoch2_screen":epoch2_screen,"gate":{}}
        write_json(output/"cqrf_phase0_engineering_failure.json",failure); render_report(output,failure); print("DECISION = CQRF_ENGINEERING_BLOCKED",flush=True); raise


if __name__=="__main__": main()
