"""Train-only GCQM Phase-0 on BCSS Seed42."""
from __future__ import annotations
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import train_sshr as official
from network.gcqm_net import GCQMNet
from network.hqrf_targets import FULL25_STEPS
from network.momd_net import MOMDNet
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_counterfactuals import materialize
from tools.gcqm_diagnostics import batch_health,summarize,apply_epoch2_screen,apply_final_gate
from tools.gcqm_report import render_report
from tools.hqrf_phase0_io import check_train_path,install_train_access_guard,protected_sources,sha256,write_csv,write_json
from train_cqrf_phase0 import MonitorDataset,Tee
from train_momd_phase0 import gradient_health


ROOT=Path(__file__).resolve().parent; STEPS_PER_EPOCH=1171; TOTAL=5855
SNAPSHOTS={250:"step0250",500:"step0500",1000:"step1000",1171:"epoch1",2342:"epoch2",3513:"epoch3",4684:"epoch4",5855:"epoch5"}; VIS={500,1000,2342,3513,4684,5855}
CONFIG={"experiment":"GCQM Phase-0 Global Class-Conditioned Query Mixture","dataset":"BCSS training only","seed":42,"gpu":"RTX4090D","batch":20,"epochs_max":5,"steps_per_epoch":1171,"total_steps":5855,"full25_schedule_denominator":FULL25_STEPS,"amp":"bf16","query_count":196,"parent":"c479ac4d1f3dca5231fb73435e59269b3407d85f","primary":"F_gcqm=sum_i w_ic B_i(x)","w":"detach(query_renorm(spatial_mean(A)))","new_parameters":0,"new_aux_loss":0,"counterfactuals":"8 fixed perm/PCA/pixel/uniform detached","validation_access":False}


def load_cohort(path):
    p=json.loads(Path(path).read_text()); return p,[(x["path"],torch.tensor(x["label"],dtype=torch.float32)) for x in p["images"]]


def _rgb(images):
    mean=images.new_tensor([.485,.456,.406])[None,:,None,None]; std=images.new_tensor([.229,.224,.225])[None,:,None,None]
    return (images*std+mean).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()


def save_visuals(directory,snapshot,names,images,labels,out,permutations):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=Path(directory)/"visualizations"/snapshot; target.mkdir(parents=True,exist_ok=True); rgb=_rgb(images); stage=out["stages"][2]; cf=materialize(stage,permutations)
    for i in range(min(8,len(names))):
        cls=int(torch.where(labels[i].bool())[0][0]); w=cf["weights"][i,:,cls]; ids=torch.argsort(w,descending=True,stable=True)[:10]; top5=ids[:5]
        panels=[(rgb[i],"input"),(out["normalized_cam"][i,cls].float().cpu(),"deep CAM"),(out["targets"][i,cls].float().cpu(),"tri-state"),(w.cpu().reshape(14,14),"global w")]
        panels += [(stage["gcqm"]["base_probability"][i,q].cpu(),f"B{j+1}") for j,q in enumerate(top5)]
        panels += [(cf[k][i,cls].cpu(),k) for k in ("primary","pixel","pca")]
        panels += [(cf["perm"][k,i,cls].cpu(),f"perm{k+1}") for k in (0,1)]
        panels += [((cf["primary"][i,cls]-cf[k][i,cls]).abs().cpu(),f"|primary-{k}|") for k in ("pixel","pca")]
        panels += [((cf["primary"][i,cls]-cf["perm"][0,i,cls]).abs().cpu(),"|primary-perm|")]
        fig,axes=plt.subplots(4,4,figsize=(12,12));
        for ax in axes.flat: ax.axis("off")
        for ax,(v,t) in zip(axes.flat,panels): ax.imshow(v,cmap=None if getattr(v,"ndim",0)==3 else "viridis"); ax.set_title(t)
        fig.tight_layout(); fig.savefig(target/f"{names[i]}.png",dpi=100); plt.close(fig)


@torch.no_grad()
def monitor(model,loader,step,snapshot,histories,summaries,output,permutations,visualize=False):
    training=model.training; model.eval(); batches=[]; pmec=[]; first=None
    for names,images,labels in loader:
        images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=step,run_pmec=True)
        batches.append(batch_health(result,labels,permutations)); pmec.extend(result["pmec_rows"])
        if first is None: first=(names,images.detach(),labels.detach(),result)
    summary=summarize(snapshot,batches,pmec,model,len(permutations)); summaries.append(summary); write_json(Path(output)/"gcqm_phase0_summary_history.json",summaries)
    mapping={"weight_conservation":"gcqm_weight_conservation.csv","weight_class_conditioning":"gcqm_weight_class_conditioning.csv","weight_utilization":"gcqm_weight_utilization.csv","primary_semantic_health":"gcqm_primary_semantic_health.csv","query_identity_sensitivity":"gcqm_query_identity_sensitivity.csv","perm_semantic_health":"gcqm_perm_semantic_health.csv","pca_semantic_health":"gcqm_pca_semantic_health.csv","pixel_reference_health":"gcqm_pixel_reference_health.csv","vs_pca_gain":"gcqm_vs_pca_gain.csv","vs_pixel_noninferiority":"gcqm_vs_pixel_noninferiority.csv","B_basis_health":"gcqm_B_basis_health.csv","contribution_utilization":"gcqm_contribution_utilization.csv","ccra_health":"gcqm_ccra_health.csv","pca_health":"gcqm_pca_health.csv","deep_gate_health":"gcqm_deep_gate_health.csv","query_update_health":"gcqm_query_update_health.csv","counterfactual_detach_audit":"gcqm_counterfactual_detach_audit.csv"}
    for key,file in mapping.items():
        rows=summary[key] if isinstance(summary[key],list) else [summary[key]]; histories[key].extend(rows); write_csv(Path(output)/file,histories[key])
    if visualize and first: save_visuals(output,snapshot,*first,permutations)
    sem=next(x for x in summary["primary_semantic_health"] if x["stage"]==3); sens=next(x for x in summary["query_identity_sensitivity"] if x["stage"]==3); gain=next(x for x in summary["vs_pca_gain"] if x["stage"]==3); ni=next(x for x in summary["vs_pixel_noninferiority"] if x["stage"]==3)
    print("GCQM_MONITOR "+json.dumps({"snapshot":snapshot,"CPR":sem["CPR"],"recall":sem["positive_recall"],"gap":sem["probability_gap"],"D_perm":sens["D_perm_mean"],"PCA_gap_gain":gain["gap_gain"],"pixel_gap_delta":ni["gap_delta"]},allow_nan=False),flush=True)
    if training:model.train()
    return summary


def checkpoint(model,output):
    p=Path(output)/"gcqm_phase0_endpoint.pth"; torch.save(model.state_dict(),p); d=sha256(p); (Path(output)/"gcqm_phase0_endpoint_sha256.txt").write_text(d+"\n"); return p,d


def main():
    p=argparse.ArgumentParser(); p.add_argument("--trainroot",required=True); p.add_argument("--weights",required=True); p.add_argument("--output",required=True); p.add_argument("--cohort-json",required=True); p.add_argument("--compatibility-json",required=True); p.add_argument("--smoke-steps",type=int,choices=(0,2),default=0); a=p.parse_args(); check_train_path(a.trainroot); output=Path(a.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("RTX4090D BF16 required")
    audit=json.loads(Path(a.compatibility_json).read_text());
    if audit.get("decision")!="PROCEED" or not audit.get("references_detached") or not audit.get("all_derangements"): raise RuntimeError("GCQM compatibility blocked")
    protected=protected_sources(ROOT); accesses=install_train_access_guard(); official.set_seed(42); output.mkdir(parents=True); log=(output/"gcqm_train.log").open("w",buffering=1); sys.stdout=Tee(sys.__stdout__,log); sys.stderr=Tee(sys.__stderr__,log)
    source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); config={**CONFIG,"source_commit":source,"smoke_steps":a.smoke_steps,"trainroot":str(Path(a.trainroot).resolve())}; write_json(output/"gcqm_phase0_config.json",config); (output/"gcqm_source_commit.txt").write_text(source+"\n"); write_json(output/"fomd_endpoint_gcqm_compatibility_audit.json",audit)
    bank=permutation_payload(196); write_json(output/"gcqm_query_permutations.json",bank); model=GCQMNet().cuda(); init=model.backbone.load_official_initialization(a.weights); write_json(output/"gcqm_init_identity.json",init)
    old=sum(x.numel() for x in MOMDNet().parameters()); count=sum(x.numel() for x in model.parameters());
    if count!=old: raise RuntimeError("GCQM parameter delta must be zero")
    write_json(output/"gcqm_parameter_counts.json",{"momd":old,"gcqm":count,"delta":0}); cohort_payload,cohort=load_cohort(a.cohort_json); write_json(output/"gcqm_monitor_cohort.json",cohort_payload); monitor_loader=DataLoader(MonitorDataset(cohort),batch_size=8,num_workers=4,pin_memory=True)
    dataset=Stage1_TrainDataset(a.trainroot,transform=transforms.Compose([transforms.ToTensor()]),dataset="bcss",img_size=224); loader=DataLoader(dataset,batch_size=20,shuffle=True,num_workers=8,pin_memory=True,drop_last=True,worker_init_fn=official.seed_worker,generator=torch.Generator().manual_seed(42))
    if len(dataset)!=23422 or len(loader)!=1171: raise RuntimeError("Frozen BCSS cardinality changed")
    groups=model.get_parameter_groups(); optimizer=PolyOptimizer([{"params":g,"lr":.01*m,"weight_decay":d} for g,m,d in zip(groups,(1,2,10,20),(.0005,0,.0005,0))],lr=.01,weight_decay=.0005,max_step=TOTAL)
    write_json(output/"gcqm_provenance.json",{"protected_sources":protected,"environment":{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},"cohort_sha256":sha256(a.cohort_json),"compatibility_sha256":sha256(a.compatibility_json),"parameter_delta":0,"validation_access":False})
    keys=("weight_conservation","weight_class_conditioning","weight_utilization","primary_semantic_health","query_identity_sensitivity","perm_semantic_health","pca_semantic_health","pixel_reference_health","vs_pca_gain","vs_pixel_noninferiority","B_basis_health","contribution_utilization","ccra_health","pca_health","deep_gate_health","query_update_health","counterfactual_detach_audit"); histories={k:[] for k in keys}; summaries=[]; losses=[]; gradients=[]; final=None; epoch2=None; completed=0; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); print("GCQM_PROTOCOL "+json.dumps(config),flush=True)
    try:
        model.train()
        for epoch in range(1,6):
            rows=[]
            for _,images,labels in loader:
                images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=optimizer.global_step,run_pmec=False)
                loss=result["losses"]["loss"]
                if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite loss")
                loss.backward(); nxt=optimizer.global_step+1
                if nxt%100==0 or nxt in SNAPSHOTS or (a.smoke_steps and nxt==2):
                    g=gradient_health(model,nxt); gradients.extend(g); write_csv(output/"gcqm_gradient_health.csv",gradients)
                    if any(x["nonfinite_fraction"]>0 for x in g): raise FloatingPointError("Nonfinite gradient")
                optimizer.step(); step=optimizer.global_step; row={"epoch":epoch,"step":step,**{k:float(v.detach()) for k,v in result["losses"].items()},"lr":optimizer.param_groups[0]["lr"]}; rows.append(row)
                if step%100==0 or step in SNAPSHOTS: losses.append(row); write_csv(output/"gcqm_losses.csv",losses); print("GCQM_STEP "+json.dumps({**row,"peak_memory":torch.cuda.max_memory_allocated(),"elapsed_seconds":time.perf_counter()-started}),flush=True)
                if step in SNAPSHOTS or (a.smoke_steps and step==2): final=monitor(model,monitor_loader,step,SNAPSHOTS.get(step,"smoke_step2"),histories,summaries,output,bank["permutations"],step in VIS)
                if a.smoke_steps and step>=2: break
            completed=epoch; print("GCQM_EPOCH "+json.dumps({"epoch":epoch,"step":optimizer.global_step,"loss":float(np.mean([x["loss"] for x in rows]))}),flush=True)
            if a.smoke_steps: break
            if epoch==2:
                epoch2=apply_epoch2_screen(final); write_json(output/"gcqm_epoch2_screen.json",epoch2)
                if epoch2["decision"]!="CONTINUE_TO_E5_UNCHANGED": break
        elapsed=time.perf_counter()-started
        if a.smoke_steps: write_json(output/"gcqm_runtime.json",{"smoke":True,"steps":optimizer.global_step,"all_finite":True,"seconds":elapsed,"parameter_delta":0,"validation_accessed":False}); print("GCQM_SMOKE_PASS",flush=True); return
        endpoint,digest=checkpoint(model,output)
        gate=epoch2 if epoch2 and epoch2["decision"]!="CONTINUE_TO_E5_UNCHANGED" else apply_final_gate(summaries)
        decision=gate["decision"]
        runtime={"smoke":False,"steps":optimizer.global_step,"epochs":completed,"train_seconds":elapsed,"peak_cuda_memory_gib":torch.cuda.max_memory_allocated()/1024**3,"all_finite":bool(final["all_finite"]),"validation_accessed":False,"test_accessed":False,"luad_accessed":False,"decision":decision,"checkpoint":str(endpoint),"checkpoint_sha256":digest,"gcqm_delta_parameters":0}
        write_json(output/"gcqm_runtime.json",runtime)
        payload={**runtime,"source_commit":source,"final_summary":final,"summary_history":summaries,"epoch2_screen":epoch2,"gate":gate,"fomd_compatibility":audit}
        write_json(output/"gcqm_gate_result.json",payload)
        report=render_report(output,payload)
        print("GCQM_FINAL "+json.dumps({"decision":decision,"report":str(report),"gate":gate}),flush=True)
        print(f"DECISION = {decision}",flush=True)
    except Exception as e:
        failure={"decision":"GCQM_ENGINEERING_BLOCKED","error":repr(e),"source_commit":source,"steps":optimizer.global_step,"epochs":completed,"final_summary":final or {},"summary_history":summaries,"epoch2_screen":epoch2,"gate":{},"fomd_compatibility":audit}; write_json(output/"gcqm_engineering_failure.json",failure); render_report(output,failure); print("DECISION = GCQM_ENGINEERING_BLOCKED",flush=True); raise


if __name__=="__main__": main()
