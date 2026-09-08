"""Train-only MOMD normalized ownership mixture on BCSS Seed42."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import train_sshr as official
from network.cqrf_net import CQRFNet, STAGE_WEIGHTS
from network.hqrf_targets import FULL25_STEPS
from network.momd import route_one_class
from network.momd_net import MOMDNet
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.cqrf_diagnostics import _components
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.momd_diagnostics import apply_epoch2_screen, apply_final_gate, batch_health, summarize
from tools.momd_report import render_report
from train_cqrf_phase0 import MonitorDataset, Tee


ROOT=Path(__file__).resolve().parent; STEPS_PER_EPOCH=1171; PHASE0_STEPS=5855
SNAPSHOTS={250:"step0250",500:"step0500",1000:"step1000",1171:"epoch1",2342:"epoch2",3513:"epoch3",4684:"epoch4",5855:"epoch5"}
VISUAL_SNAPSHOTS={500,1000,2342,3513,4684,5855}
CONFIG={"experiment":"MOMD Phase-0 Mass-Preserving Ownership Mixture Decoding","dataset":"BCSS training only","seed":42,
"gpu":"RTX4090D","batch":20,"epochs_max":5,"steps_per_epoch":STEPS_PER_EPOCH,"phase0_steps_max":PHASE0_STEPS,
"full25_schedule_denominator":FULL25_STEPS,"amp":"bf16","image_size":224,"query_count":196,
"frozen_cqrf_training_source":"49392bce089586450cf6506c47a5d0f3499b6fc8","audited_cqrf_source":"330be92040a3ea973404e4c2ba60f625d56c2498",
"historical_rpmc_source":"0b2018d1551968da78201369ef5d6ade7f4b51bb","historical_comd_source":"aacd0d17adddc36d025d3ad4f13ba7ad2bda6b62",
"momd":{"resize":"FP32 bilinear 56x56 align_corners=False","locality":"frozen circular mask","normalization":"query axis",
"fallback":"global resized R","base_expert":"sigmoid(original mask logit)","mixture":"F=sum_i A_ixc B_i","contribution":"C=A*B","posterior":"Q=C/F","parameters":0},
"loss":{"stage1":"frozen query BCE","stage2_stage3":"class-level mixture BCE","stage_weights":list(STAGE_WEIGHTS),"macro":{"deep":.5,"PCA":.25,"mask":.25}},
"pmec":"historical detached diagnostic only from base query logits/PCA","validation_access":False,
"prohibited":["R power","temperature","top-k","hard ownership","load balancing","extra loss","threshold change","LR change","PCA multiplier"]}


def load_cohort(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8")); return payload,[(x["path"],torch.tensor(x["label"],dtype=torch.float32)) for x in payload["images"]]


def gradient_health(model,step):
    buckets=defaultdict(list); zeros=defaultdict(list); bad=defaultdict(list)
    for name,p in model.named_parameters():
        if p.grad is None: continue
        group="backbone" if name.startswith("backbone.") else "mask_heads" if name.startswith("mask_embeddings.") else "ccra" if name.startswith(("ccra2.","ccra3.")) else "pca" if name.startswith("pca_heads.") else "pixel" if name.startswith("pixel_decoder.") else "deep" if name.startswith("deep_head.") else "query"
        g=p.grad.detach().float(); finite=torch.isfinite(g); clean=torch.where(finite,g,torch.zeros_like(g))
        buckets[group].append(float(clean.square().mean().sqrt())); zeros[group].append(float((g==0).float().mean())); bad[group].append(float((~finite).float().mean()))
    return [{"step":step,"module":k,"grad_rms_mean":float(np.mean(v)),"grad_rms_p50":float(np.median(v)),"grad_rms_p90":float(np.quantile(v,.9)),"zero_fraction":float(np.mean(zeros[k])),"nonfinite_fraction":float(np.mean(bad[k]))} for k,v in buckets.items()]


def gradient_routing_audit(result,step):
    stage=result["stages"][2]; grad=stage["base_mask_logits"].grad.detach().float().abs(); rows=[]
    for cls in range(4):
        route=route_one_class(stage["detail"]["responsibility_class"],stage["memory_hw"],grad.shape[-2:],result["locality"],cls)["routing"]
        a=route.flatten(); g=grad.flatten(); a=a-a.mean(); g=g-g.mean()
        corr=float((a*g).sum()/(a.square().sum().sqrt()*g.square().sum().sqrt()).clamp_min(1e-8))
        rows.append({"step":step,"class":cls,"pearson_A_abs_dL_dZ":corr,"gradient_mean":float(g.abs().mean())})
    return rows


def _rgb(images):
    mean=images.new_tensor([.485,.456,.406])[None,:,None,None]; std=images.new_tensor([.229,.224,.225])[None,:,None,None]
    return (images*std+mean).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()


def save_visualizations(directory,snapshot,names,images,labels,output):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=Path(directory)/"visualizations"/snapshot; target.mkdir(parents=True,exist_ok=True); rgb=_rgb(images)
    for i in range(min(8,len(names))):
        cls=int(torch.where(labels[i].bool())[0][0]); panels=[(rgb[i],"input"),(output["normalized_cam"][i,cls].float().cpu(),"CAM"),(output["targets"][i,cls].float().cpu(),"target")]
        for si in (1,2):
            m=output["stages"][si]["momd"]; ids=torch.argsort(output["stages"][si]["confidence"]["joint"][i,:,cls],descending=True,stable=True)[:3]
            panels += [(m["responsibility_resized"][i,q,cls].cpu(),f"S{si+1} R{j+1}") for j,q in enumerate(ids)]
            panels += [(m["routing"][i,q,cls].cpu(),f"S{si+1} A{j+1}") for j,q in enumerate(ids)]
            panels += [(m["base_probability"][i,q].cpu(),f"S{si+1} B{j+1}") for j,q in enumerate(ids)]
            panels += [(m["posterior_share"][i,q,cls].cpu(),f"S{si+1} Q{j+1}") for j,q in enumerate(ids)]
            panels += [(m["mixture"][i,cls].cpu(),f"S{si+1} F"),(m["reference_envelope"][i,cls].cpu(),f"S{si+1} E_ref")]
        panels.append((output["pmec_region"][i,cls].float().cpu(),"historical PMEC"))
        fig,axes=plt.subplots(6,5,figsize=(15,18))
        for ax in axes.flat: ax.axis("off")
        for ax,(v,t) in zip(axes.flat,panels): ax.imshow(v,cmap=None if getattr(v,"ndim",0)==3 else "viridis"); ax.set_title(t)
        fig.tight_layout(); fig.savefig(target/f"{names[i]}.png",dpi=100); plt.close(fig)


@torch.no_grad()
def monitor(model,loader,step,snapshot,histories,summaries,output,visualize=False):
    training=model.training; model.eval(); batches=[]; pmec=[]; first=None
    for names,images,labels in loader:
        images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=step,run_pmec=True)
        batches.append(batch_health(result,labels)); pmec.extend(result["pmec_rows"])
        if first is None: first=(names,images.detach(),labels.detach(),result)
    summary=summarize(snapshot,batches,pmec,model); summaries.append(summary); write_json(Path(output)/"momd_phase0_summary_history.json",summaries)
    mapping={"routing_integrity":"momd_routing_integrity.csv","locality_fallback":"momd_locality_fallback.csv","capacity_preservation":"momd_capacity_preservation.csv","negative_capacity":"momd_negative_capacity.csv","primary_mask_health":"momd_primary_mask_health.csv","class_mask_area":"momd_class_mask_area.csv","fragmentation":"momd_fragmentation.csv","responsibility_complementarity":"momd_responsibility_complementarity.csv","contribution_complementarity":"momd_contribution_complementarity.csv","r_to_q_transfer":"momd_r_to_q_transfer.csv","expert_owner_contrast":"momd_expert_owner_contrast.csv","expert_ownership_correlation":"momd_expert_ownership_correlation.csv","base_expert_redundancy":"momd_base_expert_redundancy.csv","routed_expert_utilization":"momd_routed_expert_utilization.csv","pca_health":"momd_pca_health.csv","deep_gate_health":"momd_deep_gate_health.csv","query_update_health":"momd_query_update_health.csv","historical_pmec_diagnostic":"momd_historical_pmec_diagnostic.csv"}
    for key,file in mapping.items():
        rows=summary[key] if isinstance(summary[key],list) else [summary[key]]; histories[key].extend(rows); write_csv(Path(output)/file,histories[key])
    # Required semantic-selectivity alias records the frozen negative/capacity view.
    histories["semantic_selectivity"].extend(summary["negative_capacity"]); write_csv(Path(output)/"momd_semantic_selectivity.csv",histories["semantic_selectivity"])
    if visualize and first: save_visualizations(output,snapshot,*first)
    print("MOMD_MONITOR "+json.dumps({"snapshot":snapshot,"capacity":next(x for x in summary["capacity_preservation"] if x["stage"]==3),"negative":next(x for x in summary["negative_capacity"] if x["stage"]==3),"contribution":next(x for x in summary["contribution_complementarity"] if x["stage"]==3)},allow_nan=False),flush=True)
    if training: model.train()
    return summary


def checkpoint(model,output):
    path=Path(output)/"momd_phase0_endpoint.pth"; torch.save(model.state_dict(),path); digest=sha256(path); (Path(output)/"momd_phase0_endpoint_sha256.txt").write_text(digest+"\n",encoding="utf-8"); return path,digest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trainroot",required=True); p.add_argument("--weights",required=True); p.add_argument("--output",required=True); p.add_argument("--cohort-json",required=True); p.add_argument("--compatibility-json",required=True); p.add_argument("--smoke-steps",type=int,choices=(0,2),default=0); args=p.parse_args()
    check_train_path(args.trainroot); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("Registered RTX4090D BF16 GPU required")
    compatibility=json.loads(Path(args.compatibility_json).read_text(encoding="utf-8"))
    if compatibility.get("nontriviality",{}).get("decision")!="PROCEED": raise RuntimeError("MOMD compatibility audit blocked training")
    protected=protected_sources(ROOT); accesses=install_train_access_guard(); official.set_seed(42); output.mkdir(parents=True)
    log=(output/"momd_phase0_train.log").open("w",encoding="utf-8",buffering=1); sys.stdout=Tee(sys.__stdout__,log); sys.stderr=Tee(sys.__stderr__,log)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); config={**CONFIG,"source_commit":source,"smoke_steps":args.smoke_steps,"trainroot":str(Path(args.trainroot).resolve())}
    write_json(output/"momd_phase0_config.json",config); config_hash=sha256(output/"momd_phase0_config.json"); (output/"momd_phase0_source_commit.txt").write_text(source+"\n",encoding="utf-8"); (output/"momd_phase0_config_sha256.txt").write_text(config_hash+"\n",encoding="utf-8")
    write_json(output/"cqrf_endpoint_momd_compatibility_audit.json",compatibility)
    model=MOMDNet().cuda(); init=model.backbone.load_official_initialization(args.weights); write_json(output/"momd_phase0_init_identity.json",init)
    base_count=sum(x.numel() for x in CQRFNet().parameters()); count=sum(x.numel() for x in model.parameters())
    if count!=base_count: raise RuntimeError("MOMD must add zero parameters")
    write_json(output/"momd_phase0_parameter_counts.json",{"cqrf_total":base_count,"momd_total":count,"momd_delta":0})
    cohort_payload,cohort=load_cohort(args.cohort_json); write_json(output/"momd_phase0_monitor_cohort.json",cohort_payload)
    monitor_loader=DataLoader(MonitorDataset(cohort),batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    dataset=Stage1_TrainDataset(args.trainroot,transform=transforms.Compose([transforms.ToTensor()]),dataset="bcss",img_size=224)
    if len(dataset)!=23422: raise RuntimeError("Expected 23,422 BCSS train images")
    loader=DataLoader(dataset,batch_size=20,shuffle=True,num_workers=8,pin_memory=True,drop_last=True,worker_init_fn=official.seed_worker,generator=torch.Generator().manual_seed(42))
    if len(loader)!=STEPS_PER_EPOCH: raise RuntimeError("Expected 1,171 steps/epoch")
    groups=model.get_parameter_groups(); optimizer=PolyOptimizer([{"params":g,"lr":.01*m,"weight_decay":d} for g,m,d in zip(groups,(1,2,10,20),(.0005,0,.0005,0))],lr=.01,weight_decay=.0005,max_step=PHASE0_STEPS)
    write_json(output/"momd_phase0_provenance.json",{"protected_sources":protected,"environment":{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},"argv":sys.argv,"config_sha256":config_hash,"cohort_sha256":sha256(args.cohort_json),"compatibility_sha256":sha256(args.compatibility_json),"momd_delta_parameters":0})
    print("MOMD_PROTOCOL "+json.dumps(config,allow_nan=False),flush=True)
    keys=("routing_integrity","locality_fallback","capacity_preservation","negative_capacity","primary_mask_health","class_mask_area","fragmentation","responsibility_complementarity","contribution_complementarity","r_to_q_transfer","expert_owner_contrast","expert_ownership_correlation","base_expert_redundancy","routed_expert_utilization","pca_health","deep_gate_health","query_update_health","historical_pmec_diagnostic","semantic_selectivity")
    histories={k:[] for k in keys}; summaries=[]; losses=[]; gradients=[]; grad_audit=[]; completed=0; final=None; epoch2=None; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        for epoch in range(1,6):
            rows=[]
            for _,images,labels in loader:
                images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=optimizer.global_step,run_pmec=False)
                next_step=optimizer.global_step+1
                audit=next_step in SNAPSHOTS or bool(args.smoke_steps and next_step==args.smoke_steps)
                result["stages"][2]["base_mask_logits"].retain_grad() if audit else None
                loss=result["losses"]["loss"]
                if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite loss")
                loss.backward()
                if next_step%100==0 or audit:
                    g=gradient_health(model,next_step); gradients.extend(g); write_csv(output/"momd_gradient_health.csv",gradients)
                    if any(x["nonfinite_fraction"]>0 for x in g): raise FloatingPointError("Nonfinite gradient")
                if audit:
                    grad_audit.extend(gradient_routing_audit(result,next_step)); write_csv(output/"momd_gradient_routing_audit.csv",grad_audit)
                optimizer.step(); step=optimizer.global_step
                row={"epoch":epoch,"step":step,**{k:float(v.detach()) for k,v in result["losses"].items()},"lr":optimizer.param_groups[0]["lr"]}; rows.append(row)
                if step%100==0 or step in SNAPSHOTS:
                    losses.append(row); write_csv(output/"momd_losses.csv",losses); print("MOMD_STEP "+json.dumps({**row,"peak_memory":torch.cuda.max_memory_allocated(),"elapsed_seconds":time.perf_counter()-started}),flush=True)
                if step in SNAPSHOTS: final=monitor(model,monitor_loader,step,SNAPSHOTS[step],histories,summaries,output,step in VISUAL_SNAPSHOTS)
                if args.smoke_steps and step>=args.smoke_steps: break
            completed=epoch; print("MOMD_EPOCH "+json.dumps({"epoch":epoch,"step":optimizer.global_step,**{k:float(np.mean([x[k] for x in rows])) for k in ("loss","loss_deep","loss_pca","loss_mask")}}),flush=True)
            if args.smoke_steps: break
            if epoch==2:
                epoch2=apply_epoch2_screen(final); write_json(output/"momd_phase0_epoch2_screen.json",epoch2)
                if epoch2["decision"]=="MOMD_PHASE0_NOGO": break
        elapsed=time.perf_counter()-started
        if args.smoke_steps:
            runtime={"smoke":True,"steps":optimizer.global_step,"epochs":completed,"all_finite":True,"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"momd_delta_parameters":0,"validation_accessed":False}
            write_json(output/"momd_phase0_runtime.json",runtime); print("MOMD_SMOKE_PASS "+json.dumps(runtime),flush=True); return
        endpoint,digest=checkpoint(model,output); gate=epoch2 if epoch2 and epoch2["decision"]=="MOMD_PHASE0_NOGO" else apply_final_gate(summaries); decision=gate["decision"]
        runtime={"smoke":False,"steps":optimizer.global_step,"epochs":completed,"all_finite":bool(final["all_finite"]),"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_memory_gib":torch.cuda.max_memory_allocated()/1024**3,"validation_accessed":False,"test_accessed":False,"luad_accessed":False,"training_samples_consumed":optimizer.global_step*20,"decision":decision,"checkpoint":str(endpoint),"checkpoint_sha256":digest,"cqrf_parameters":base_count,"momd_parameters":count,"momd_delta_parameters":0}
        write_json(output/"momd_phase0_runtime.json",runtime); payload={**runtime,"source_commit":source,"final_summary":final,"summary_history":summaries,"epoch2_screen":epoch2,"gate":gate,"cqrf_compatibility":compatibility}; write_json(output/"momd_phase0_gate_result.json",payload)
        report=render_report(output,payload); print("MOMD_FINAL "+json.dumps({"decision":decision,"report":str(report),"gate":gate}),flush=True); print(f"DECISION = {decision}",flush=True)
    except Exception as error:
        failure={"decision":"MOMD_ENGINEERING_BLOCKED","error":repr(error),"source_commit":source,"epochs":completed,"steps":optimizer.global_step,"all_finite":False,"final_summary":final or {},"summary_history":summaries,"epoch2_screen":epoch2,"gate":{},"cqrf_compatibility":compatibility}
        write_json(output/"momd_phase0_engineering_failure.json",failure); render_report(output,failure); print("DECISION = MOMD_ENGINEERING_BLOCKED",flush=True); raise


if __name__=="__main__": main()
