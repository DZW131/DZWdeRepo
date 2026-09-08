"""Train-only FOMD Phase-0: archived MOMD training plus detached factorization audits."""
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
from torch.utils.data import DataLoader
from torchvision import transforms

import train_sshr as official
from network.cqrf_net import CQRFNet, STAGE_WEIGHTS
from network.hqrf_targets import FULL25_STEPS
from network.momd_net import MOMDNet
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.fomd_counterfactuals import materialize, permutation_payload
from tools.fomd_diagnostics import apply_epoch2_screen, apply_final_gate, batch_health, summarize
from tools.fomd_report import render_report
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from train_cqrf_phase0 import MonitorDataset, Tee
from train_momd_phase0 import gradient_health


ROOT=Path(__file__).resolve().parent; STEPS_PER_EPOCH=1171; PHASE0_STEPS=5855
SNAPSHOTS={250:"step0250",500:"step0500",1000:"step1000",1171:"epoch1",2342:"epoch2",3513:"epoch3",4684:"epoch4",5855:"epoch5"}
VISUAL={500,1000,2342,3513,4684,5855}
CONFIG={"experiment":"FOMD Phase-0 Factorized Ownership Mask Decoding","dataset":"BCSS training only","seed":42,"gpu":"RTX4090D","batch":20,"epochs_max":5,"steps_per_epoch":1171,"phase0_steps_max":5855,"full25_schedule_denominator":FULL25_STEPS,"amp":"bf16","image_size":224,"query_count":196,"frozen_momd_source":"aecc292f539a2810c938d250cd3fcc913a4a6cd9","architecture":"numerically identical MOMD-v1","factorization":{"A":"class-conditioned CCRA ownership","B":"class-agnostic query region support","F":"sum_i A_ixc*B_ix","parameters":0},"counterfactuals":{"training_gradient":False,"fixed_derangements":8,"seeds":list(range(1001,1009)),"global_A":"spatial mean then query renormalize","pca":"historical E_ref","uniform":"mean_i B"},"loss":{"stage1":"frozen query BCE","stage2_stage3":"class-level MOMD mixture BCE","stage_weights":list(STAGE_WEIGHTS),"macro":{"deep":.5,"PCA":.25,"mask":.25},"auxiliary":False},"pmec":"historical detached diagnostic only","validation_access":False,"prohibited":["OEC","extra loss","new B head","class-conditioned B","tuning after step0"]}


def load_cohort(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8")); return payload,[(x["path"],torch.tensor(x["label"],dtype=torch.float32)) for x in payload["images"]]


def _rgb(images):
    mean=images.new_tensor([.485,.456,.406])[None,:,None,None]; std=images.new_tensor([.229,.224,.225])[None,:,None,None]
    return (images*std+mean).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()


def save_visualizations(directory,snapshot,names,images,labels,output,permutations):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=Path(directory)/"visualizations"/snapshot; target.mkdir(parents=True,exist_ok=True); rgb=_rgb(images)
    stage=output["stages"][2]; momd=stage["momd"]; cf=materialize(stage,permutations)
    for image in range(min(8,len(names))):
        cls=int(torch.where(labels[image].bool())[0][0]); ids=torch.argsort(stage["confidence"]["joint"][image,:,cls],descending=True,stable=True)[:5]
        panels=[(rgb[image],"input"),(output["normalized_cam"][image,cls].float().cpu(),"deep CAM"),(output["targets"][image,cls].float().cpu(),"tri-state")]
        panels += [(momd["routing"][image,q,cls].cpu(),f"A{j+1}") for j,q in enumerate(ids)]
        panels += [(momd["base_probability"][image,q].cpu(),f"B{j+1}") for j,q in enumerate(ids)]
        panels += [(cf["full"][image,cls].cpu(),"F_full"),(cf["global"][image,cls].cpu(),"F_global"),(cf["pca"][image,cls].cpu(),"F_pca"),(cf["perm"][0,image,cls].cpu(),"F_perm1001"),(cf["perm"][1,image,cls].cpu(),"F_perm1002")]
        panels += [((cf["full"][image,cls]-cf[key][image,cls]).abs().cpu(),f"|full-{key}|") for key in ("global","pca")]
        panels += [((cf["full"][image,cls]-cf["perm"][0,image,cls]).abs().cpu(),"|full-perm|")]
        panels += [(momd["contribution"][image,q,cls].cpu(),f"C{j+1}") for j,q in enumerate(ids)]
        panels += [(momd["posterior_share"][image,q,cls].cpu(),f"Q{j+1}") for j,q in enumerate(ids)]
        fig,axes=plt.subplots(6,5,figsize=(15,18))
        for ax in axes.flat: ax.axis("off")
        for ax,(value,title) in zip(axes.flat,panels): ax.imshow(value,cmap=None if getattr(value,"ndim",0)==3 else "viridis"); ax.set_title(title)
        fig.tight_layout(); fig.savefig(target/f"{names[image]}.png",dpi=100); plt.close(fig)


@torch.no_grad()
def monitor(model,loader,step,snapshot,histories,summaries,output,permutations,visualize=False):
    training=model.training; model.eval(); batches=[]; pmec=[]; first=None
    for names,images,labels in loader:
        images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=step,run_pmec=True)
        batches.append(batch_health(result,labels,permutations)); pmec.extend(result["pmec_rows"])
        if first is None: first=(names,images.detach(),labels.detach(),result)
    summary=summarize(snapshot,batches,pmec,model,len(permutations)); summaries.append(summary); write_json(Path(output)/"fomd_phase0_summary_history.json",summaries)
    mapping={"full_semantic_health":"fomd_full_semantic_health.csv","perm_semantic_health":"fomd_perm_semantic_health.csv","global_semantic_health":"fomd_global_semantic_health.csv","pca_semantic_health":"fomd_pca_semantic_health.csv","uniform_reference":"fomd_uniform_reference.csv","query_identity_sensitivity":"fomd_query_identity_sensitivity.csv","spatial_ownership_sensitivity":"fomd_spatial_ownership_sensitivity.csv","pca_sensitivity":"fomd_pca_sensitivity.csv","factorization_synergy":"fomd_factorization_synergy.csv","A_class_conditioning":"fomd_A_class_conditioning.csv","A_top_owner_disagreement":"fomd_A_top_owner_disagreement.csv","B_union_foreground_support":"fomd_B_union_foreground_support.csv","B_locality_mass":"fomd_B_locality_mass.csv","B_centroid_consistency":"fomd_B_centroid_consistency.csv","B_area_health":"fomd_B_area_health.csv","B_fragmentation":"fomd_B_fragmentation.csv","B_redundancy":"fomd_B_redundancy.csv","ccra_health":"fomd_ccra_health.csv","conservation":"fomd_conservation.csv","contribution_complementarity":"fomd_contribution_complementarity.csv","r_to_q_transfer":"fomd_r_to_q_transfer.csv","pca_health":"fomd_pca_health.csv","deep_gate_health":"fomd_deep_gate_health.csv","query_update_health":"fomd_query_update_health.csv","counterfactual_detach_audit":"fomd_counterfactual_detach_audit.csv"}
    for key,file in mapping.items():
        rows=summary[key] if isinstance(summary[key],list) else [summary[key]]; histories[key].extend(rows); write_csv(Path(output)/file,histories[key])
    if visualize and first: save_visualizations(output,snapshot,*first,permutations)
    f=next(x for x in summary["full_semantic_health"] if x["stage"]==3); q=next(x for x in summary["query_identity_sensitivity"] if x["stage"]==3)
    print("FOMD_MONITOR "+json.dumps({"snapshot":snapshot,"CPR":next(x for x in summary["capacity_preservation"] if x["stage"]==3)["CPR"],"positive_recall":f["positive_recall"],"gap":f["probability_gap"],"D_perm":q["D_perm_mean"]},allow_nan=False),flush=True)
    if training: model.train()
    return summary


def checkpoint(model,output):
    path=Path(output)/"fomd_phase0_endpoint.pth"; torch.save(model.state_dict(),path); digest=sha256(path); (Path(output)/"fomd_phase0_endpoint_sha256.txt").write_text(digest+"\n",encoding="utf-8"); return path,digest


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trainroot",required=True); p.add_argument("--weights",required=True); p.add_argument("--output",required=True); p.add_argument("--cohort-json",required=True); p.add_argument("--factorization-audit-json",required=True); p.add_argument("--smoke-steps",type=int,choices=(0,2),default=0); a=p.parse_args()
    check_train_path(a.trainroot); output=Path(a.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("Registered RTX4090D BF16 GPU required")
    audit=json.loads(Path(a.factorization_audit_json).read_text(encoding="utf-8"))
    required=(audit.get("decision")=="PROCEED" and audit.get("F_full_equals_MOMD") and audit.get("counterfactuals_detached") and audit.get("all_derangements") and audit.get("parameter_delta")==0)
    if not required: raise RuntimeError("FOMD historical engineering audit blocked training")
    protected=protected_sources(ROOT); accesses=install_train_access_guard(); official.set_seed(42); output.mkdir(parents=True)
    log=(output/"fomd_train.log").open("w",encoding="utf-8",buffering=1); sys.stdout=Tee(sys.__stdout__,log); sys.stderr=Tee(sys.__stderr__,log)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); config={**CONFIG,"source_commit":source,"smoke_steps":a.smoke_steps,"trainroot":str(Path(a.trainroot).resolve())}
    write_json(output/"fomd_phase0_config.json",config); (output/"fomd_source_commit.txt").write_text(source+"\n",encoding="utf-8"); write_json(output/"momd_endpoint_fomd_factorization_audit.json",audit)
    bank=permutation_payload(196); write_json(output/"fomd_query_permutations.json",bank); (output/"fomd_query_permutations_sha256.txt").write_text(sha256(output/"fomd_query_permutations.json")+"\n",encoding="utf-8"); permutations=bank["permutations"]
    model=MOMDNet().cuda(); init=model.backbone.load_official_initialization(a.weights); write_json(output/"fomd_init_identity.json",init)
    momd_count=sum(x.numel() for x in MOMDNet().parameters()); count=sum(x.numel() for x in model.parameters())
    if count!=momd_count: raise RuntimeError("FOMD must add zero parameters")
    write_json(output/"fomd_phase0_parameter_counts.json",{"momd_total":momd_count,"fomd_total":count,"fomd_delta":0})
    cohort_payload,cohort=load_cohort(a.cohort_json); write_json(output/"fomd_phase0_monitor_cohort.json",cohort_payload); monitor_loader=DataLoader(MonitorDataset(cohort),batch_size=8,num_workers=4,pin_memory=True)
    dataset=Stage1_TrainDataset(a.trainroot,transform=transforms.Compose([transforms.ToTensor()]),dataset="bcss",img_size=224)
    if len(dataset)!=23422: raise RuntimeError("Expected 23,422 BCSS train images")
    loader=DataLoader(dataset,batch_size=20,shuffle=True,num_workers=8,pin_memory=True,drop_last=True,worker_init_fn=official.seed_worker,generator=torch.Generator().manual_seed(42))
    if len(loader)!=STEPS_PER_EPOCH: raise RuntimeError("Expected 1,171 steps/epoch")
    groups=model.get_parameter_groups(); optimizer=PolyOptimizer([{"params":g,"lr":.01*m,"weight_decay":d} for g,m,d in zip(groups,(1,2,10,20),(.0005,0,.0005,0))],lr=.01,weight_decay=.0005,max_step=PHASE0_STEPS)
    write_json(output/"fomd_phase0_provenance.json",{"protected_sources":protected,"environment":{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},"cohort_sha256":sha256(a.cohort_json),"audit_sha256":sha256(a.factorization_audit_json),"zero_training_equation_delta":True,"fomd_delta_parameters":0})
    print("FOMD_PROTOCOL "+json.dumps(config,allow_nan=False),flush=True)
    history_keys=("full_semantic_health","perm_semantic_health","global_semantic_health","pca_semantic_health","uniform_reference","query_identity_sensitivity","spatial_ownership_sensitivity","pca_sensitivity","factorization_synergy","A_class_conditioning","A_top_owner_disagreement","B_union_foreground_support","B_locality_mass","B_centroid_consistency","B_area_health","B_fragmentation","B_redundancy","ccra_health","conservation","contribution_complementarity","r_to_q_transfer","pca_health","deep_gate_health","query_update_health","counterfactual_detach_audit")
    histories={k:[] for k in history_keys}; summaries=[]; losses=[]; gradients=[]; completed=0; final=None; epoch2=None; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        for epoch in range(1,6):
            rows=[]
            for _,images,labels in loader:
                images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=optimizer.global_step,run_pmec=False)
                loss=result["losses"]["loss"]
                if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite loss")
                loss.backward(); next_step=optimizer.global_step+1
                if next_step%100==0 or next_step in SNAPSHOTS or (a.smoke_steps and next_step==a.smoke_steps):
                    g=gradient_health(model,next_step); gradients.extend(g); write_csv(output/"fomd_gradient_health.csv",gradients)
                    if any(x["nonfinite_fraction"]>0 for x in g): raise FloatingPointError("Nonfinite gradient")
                optimizer.step(); step=optimizer.global_step
                row={"epoch":epoch,"step":step,**{k:float(v.detach()) for k,v in result["losses"].items()},"lr":optimizer.param_groups[0]["lr"]}; rows.append(row)
                if step%100==0 or step in SNAPSHOTS:
                    losses.append(row); write_csv(output/"fomd_losses.csv",losses); print("FOMD_STEP "+json.dumps({**row,"peak_memory":torch.cuda.max_memory_allocated(),"elapsed_seconds":time.perf_counter()-started}),flush=True)
                if step in SNAPSHOTS: final=monitor(model,monitor_loader,step,SNAPSHOTS[step],histories,summaries,output,permutations,step in VISUAL)
                if a.smoke_steps and step>=a.smoke_steps: break
            completed=epoch; print("FOMD_EPOCH "+json.dumps({"epoch":epoch,"step":optimizer.global_step,"loss":float(np.mean([x["loss"] for x in rows]))}),flush=True)
            if a.smoke_steps: break
            if epoch==2:
                epoch2=apply_epoch2_screen(final); write_json(output/"fomd_phase0_epoch2_screen.json",epoch2)
                if epoch2["decision"]=="FOMD_PHASE0_NOGO": break
        elapsed=time.perf_counter()-started
        if a.smoke_steps:
            runtime={"smoke":True,"steps":optimizer.global_step,"epochs":completed,"all_finite":True,"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"fomd_delta_parameters":0,"counterfactual_training_gradient":False,"validation_accessed":False}; write_json(output/"fomd_phase0_runtime.json",runtime); print("FOMD_SMOKE_PASS "+json.dumps(runtime),flush=True); return
        endpoint,digest=checkpoint(model,output); gate=epoch2 if epoch2 and epoch2["decision"]=="FOMD_PHASE0_NOGO" else apply_final_gate(summaries); decision=gate["decision"]
        runtime={"smoke":False,"steps":optimizer.global_step,"epochs":completed,"all_finite":bool(final["all_finite"]),"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_memory_gib":torch.cuda.max_memory_allocated()/1024**3,"validation_accessed":False,"test_accessed":False,"luad_accessed":False,"training_samples_consumed":optimizer.global_step*20,"decision":decision,"checkpoint":str(endpoint),"checkpoint_sha256":digest,"momd_parameters":momd_count,"fomd_parameters":count,"fomd_delta_parameters":0}
        write_json(output/"fomd_phase0_runtime.json",runtime); payload={**runtime,"source_commit":source,"final_summary":final,"summary_history":summaries,"epoch2_screen":epoch2,"gate":gate,"momd_factorization_audit":audit}; write_json(output/"fomd_phase0_gate_result.json",payload); report=render_report(output,payload); print("FOMD_FINAL "+json.dumps({"decision":decision,"report":str(report),"gate":gate}),flush=True); print(f"DECISION = {decision}",flush=True)
    except Exception as error:
        failure={"decision":"FOMD_ENGINEERING_BLOCKED","error":repr(error),"source_commit":source,"epochs":completed,"steps":optimizer.global_step,"all_finite":False,"final_summary":final or {},"summary_history":summaries,"epoch2_screen":epoch2,"gate":{},"momd_factorization_audit":audit}; write_json(output/"fomd_phase0_engineering_failure.json",failure); render_report(output,failure); print("DECISION = FOMD_ENGINEERING_BLOCKED",flush=True); raise


if __name__=="__main__": main()
