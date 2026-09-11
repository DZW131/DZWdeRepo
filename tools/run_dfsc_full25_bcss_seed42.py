#!/usr/bin/env python3
"""Fresh BCSS Seed42 Full25 training for frozen CCRA/GCQM plus DFSC."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import train_sshr as official
from network.dfsc_net import DFSCNet
from network.gcqm_net import GCQMNet
from network.hqrf_targets import FULL25_STEPS
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_diagnostics import batch_health, summarize
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.run_gcqm_full25_bcss_seed42 import _canonical_hash, _finite_model, _git, _gradient_health, _rows, _save_recovery
from train_cqrf_phase0 import MonitorDataset, Tee
from train_gcqm_phase0 import load_cohort, save_visuals


INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 25, 1171, 29275
MILESTONES = {5, 10, 15, 20, 25}
CONFIG = {"experiment": "CCRA DFSC BCSS Seed42 Full25", "dataset": "BCSS training only", "seed": 42,
    "epochs": 25, "batch_size": 20, "effective_batch_size": 20, "image_size": 224, "precision": "bf16",
    "base_lr": .01, "weight_decay": .0005, "poly_power": .9, "steps_per_epoch": 1171,
    "total_steps": 29275, "locality_denominator": 29275, "base_loss": ".50*deep+.25*PCA+.25*mask",
    "total_loss": "L_base+L_rel", "stage_mask_weights": [.20, .30, .50], "dfra_channels": "256->32 x2",
    "new_parameters": 16386, "low_frequency": "AvgPool5x5", "high_frequency": "P-AvgPool3x3",
    "relation": "exp(-softplus(theta_L)*dL-softplus(theta_H)*dH)", "relation_feature_detached": True,
    "completion_affinity_detached": True, "completion": "MCC max(S,D), T=2, Stage2/3", "rival_gate": False,
    "checkpoint_selection": "fixed Epoch25 FINAL only", "validation_during_training": False}


def gradient_contract(model, output) -> dict:
    base_params = [p for n, p in model.named_parameters() if not n.startswith("dfra.") and p.requires_grad]
    relation_params = [p for n, p in model.named_parameters() if n.startswith("dfra.") and p.requires_grad]
    base_from_total = torch.autograd.grad(output["losses"]["loss"], base_params, retain_graph=True, allow_unused=True)
    base_from_base = torch.autograd.grad(output["losses"]["loss_base"], base_params, retain_graph=True, allow_unused=True)
    relation_from_total = torch.autograd.grad(output["losses"]["loss"], relation_params, retain_graph=True, allow_unused=True)
    relation_from_relation = torch.autograd.grad(output["losses"]["loss_relation"], relation_params, retain_graph=True, allow_unused=True)
    relation_from_base = torch.autograd.grad(output["losses"]["loss_base"], relation_params, retain_graph=True, allow_unused=True)
    def maximum_difference(left, right):
        return max((float((a-b).abs().max()) for a, b in zip(left, right) if a is not None and b is not None), default=0.)
    result = {"base_total_vs_base_max_abs": maximum_difference(base_from_total, base_from_base),
              "relation_total_vs_relation_max_abs": maximum_difference(relation_from_total, relation_from_relation),
              "relation_from_base_max_abs": max((float(v.abs().max()) for v in relation_from_base if v is not None), default=0.),
              "relation_nonzero_tensors": sum(v is not None and float(v.abs().sum()) > 0 for v in relation_from_relation)}
    if result["base_total_vs_base_max_abs"] > 1e-7 or result["relation_total_vs_relation_max_abs"] > 1e-7 or result["relation_from_base_max_abs"] != 0 or result["relation_nonzero_tensors"] == 0:
        raise AssertionError(f"DFSC gradient isolation failed: {result}")
    return result


@torch.no_grad()
def mechanism_snapshot(model, loader, epoch, output, permutations):
    training = model.training; model.eval(); batches, pmec, first = [], [], None
    scores, truth, d_low_pos, d_low_neg, d_high_pos, d_high_neg = [], [], [], [], [], []
    entropy, boundary_entropy, self_mass, top_neighbor = [], [], [], []
    completion = []
    for names, images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=epoch*STEPS_PER_EPOCH, run_pmec=True)
        batches.append(batch_health(result, labels, permutations)); pmec.extend(result["pmec_rows"]); rel = result["relation"]
        pos, neg, raw, alpha = rel["positive_pairs"], rel["negative_pairs"], rel["raw_affinity"], rel["affinity"]
        scores.extend(raw[pos].float().cpu().tolist()); truth.extend([1]*int(pos.sum())); scores.extend(raw[neg].float().cpu().tolist()); truth.extend([0]*int(neg.sum()))
        d_low_pos.extend(rel["distance_low"][pos].cpu().tolist()); d_low_neg.extend(rel["distance_low"][neg].cpu().tolist())
        d_high_pos.extend(rel["distance_high"][pos].cpu().tolist()); d_high_neg.extend(rel["distance_high"][neg].cpu().tolist())
        h = -(alpha.clamp_min(1e-8)*alpha.clamp_min(1e-8).log()).sum(1); entropy.extend(h.flatten().cpu().tolist())
        boundary = neg.any(1); boundary_entropy.extend(h[boundary].cpu().tolist()); self_mass.extend(alpha[:,4].flatten().cpu().tolist())
        off = alpha.clone(); off[:,4] = -1; top_neighbor.extend(off.max(1).values.flatten().cpu().tolist())
        for stage_index in (2,3):
            value = result["stages"][stage_index-1]["dfsc"]; delta=value["restored"]-value["base"]
            completion.append({"snapshot":f"epoch{epoch}","stage":stage_index,"completion_mass_mean":float(delta.mean()),
                "changed_fraction":float((delta>1e-7).float().mean()),"base_mean":float(value["base"].mean()),"restored_mean":float(value["restored"].mean()),
                "base_positive_fraction":float((value["base"]>.5).float().mean()),"restored_positive_fraction":float((value["restored"]>.5).float().mean()),
                "max_negative_change":float((-delta).clamp_min(0).max())})
        if first is None: first=(names,images.detach(),labels.detach(),result)
    summary = summarize(f"epoch{epoch}", batches, pmec, model, len(permutations))
    y, s = np.asarray(truth), np.asarray(scores); pos_s, neg_s = s[y==1], s[y==0]
    relation = {"snapshot":f"epoch{epoch}","positive_pairs":int((y==1).sum()),"negative_pairs":int((y==0).sum()),
        "positive_affinity_mean":float(pos_s.mean()),"negative_affinity_mean":float(neg_s.mean()),"pos_neg_gap":float(pos_s.mean()-neg_s.mean()),
        "pair_auroc":float(roc_auc_score(y,s)),"pair_ap":float(average_precision_score(y,s)),"row_entropy":float(np.mean(entropy)),
        "effective_neighbor_count":float(math.exp(np.mean(entropy))),"boundary_neighborhood_entropy":float(np.mean(boundary_entropy)),
        "self_mass":float(np.mean(self_mass)),"top_neighbor_mass":float(np.mean(top_neighbor)),
        "beta_low":float(torch.nn.functional.softplus(model.dfra.theta_low.detach().float())),"beta_high":float(torch.nn.functional.softplus(model.dfra.theta_high.detach().float()))}
    branches = {"snapshot":f"epoch{epoch}","positive_dL":float(np.mean(d_low_pos)),"negative_dL":float(np.mean(d_low_neg)),
        "positive_dH":float(np.mean(d_high_pos)),"negative_dH":float(np.mean(d_high_neg)),"beta_low":relation["beta_low"],"beta_high":relation["beta_high"]}
    aggregated=[]
    for stage in (2,3):
        rows=[x for x in completion if x["stage"]==stage]; aggregated.append({"snapshot":f"epoch{epoch}","stage":stage,
            **{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in {"snapshot","stage"}}})
    save_visuals(output,f"epoch{epoch}",*first,permutations)
    if training:model.train()
    return summary,relation,branches,aggregated


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--trainroot",required=True); p.add_argument("--weights",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--cohort-json",required=True); p.add_argument("--ccac-result-json",required=True)
    p.add_argument("--failure-anatomy-json",required=True); p.add_argument("--num-workers",type=int,default=8); p.add_argument("--smoke-steps",type=int,choices=(0,2),default=0); return p.parse_args()


def main():
    args=parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("DFSC Full25 requires RTX4090D BF16")
    output,weights=Path(args.output_dir).resolve(),Path(args.weights).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing populated output: {output}")
    if not args.smoke_steps and _git("status","--porcelain"): raise AssertionError("Formal step0 requires clean source")
    ccac=json.loads(Path(args.ccac_result_json).read_text()); anatomy=json.loads(Path(args.failure_anatomy_json).read_text())
    if ccac.get("decision")!="CCAC_FULL25_NOGO" or anatomy.get("decision")!="MISSING_SPATIAL_COHERENCE": raise AssertionError("Frozen upstream evidence mismatch")
    if sha256(weights)!=INIT_SHA256 or weights.suffix!=".params": raise AssertionError("Official initialization mismatch")
    for name in ("provenance","tests","logs","metrics","mechanism","checkpoints","recovery","visualizations","evaluation","ablation","report","smoke"):(output/name).mkdir(parents=True,exist_ok=True)
    log=(output/"logs/dfsc_full25_train.log").open("w",buffering=1); sys.stdout=Tee(sys.__stdout__,log);sys.stderr=Tee(sys.__stderr__,log)
    accesses=install_train_access_guard();official.set_seed(42);source=_git("rev-parse","HEAD");config={**CONFIG,"source_commit":source,"trainroot":str(Path(args.trainroot).resolve()),"weights":str(weights),"smoke_steps":args.smoke_steps}
    write_json(output/"provenance/dfsc_config.json",config);(output/"provenance/dfsc_config_sha256.txt").write_text(_canonical_hash(config)+"\n");(output/"provenance/dfsc_source_commit.txt").write_text(source+"\n")
    (output/"provenance/dfsc_git_diff.patch").write_text(subprocess.check_output(["git","show","--format=","--binary","HEAD"],cwd=ROOT,text=True));(output/"provenance/dfsc_environment.txt").write_text(f"python\t{sys.version.replace(chr(10),' ')}\ntorch\t{torch.__version__}\ncuda\t{torch.version.cuda}\ngpu\t{torch.cuda.get_device_name(0)}\nplatform\t{platform.platform()}\n")
    write_json(output/"provenance/dfsc_upstream_evidence.json",{"ccac_result":ccac,"failure_anatomy":anatomy,"ccac_result_sha256":sha256(args.ccac_result_json),"failure_anatomy_sha256":sha256(args.failure_anatomy_json)})
    dataset=Stage1_TrainDataset(args.trainroot,dataset="bcss",img_size=224);generator=torch.Generator().manual_seed(42);loader=DataLoader(dataset,batch_size=20,shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=True,worker_init_fn=official.seed_worker,generator=generator)
    if len(dataset)!=23422 or len(loader)!=STEPS_PER_EPOCH or FULL25_STEPS!=TOTAL_STEPS:raise AssertionError("Frozen cardinality mismatch")
    names=sorted(Path(p).name for p,_ in dataset.object);write_json(output/"provenance/dfsc_dataset.json",{"samples":len(dataset),"filename_manifest_sha256":hashlib.sha256("\n".join(names).encode()).hexdigest(),"training_only":True,"validation_accessed":False})
    model=DFSCNet();init=model.backbone.load_official_initialization(str(weights));init.update({"sha256":sha256(weights),"fresh_official_initialization":True,"trained_checkpoint_loaded":False});write_json(output/"provenance/dfsc_init_identity.json",init)
    base_count=sum(p.numel() for p in GCQMNet().parameters());count=sum(p.numel() for p in model.parameters())
    if count-base_count!=16386:raise AssertionError(f"DFSC parameter delta {count-base_count}")
    write_json(output/"provenance/dfsc_parameter_counts.json",{"gcqm":base_count,"dfsc":count,"delta":count-base_count})
    model=model.cuda();groups=model.get_parameter_groups();optimizer=PolyOptimizer([{"params":g,"lr":.01*m,"weight_decay":d} for g,m,d in zip(groups,(1,2,10,20),(.0005,0,.0005,0))],lr=.01,weight_decay=.0005,max_step=TOTAL_STEPS)
    cohort_payload,cohort=load_cohort(args.cohort_json);write_json(output/"provenance/dfsc_monitor_cohort.json",cohort_payload);monitor_loader=DataLoader(MonitorDataset(cohort),batch_size=8,num_workers=4,pin_memory=True);permutations=permutation_payload(196)["permutations"]
    loss_rows=[];epoch_rows=[];summaries=[];relation_rows=[];branch_rows=[];completion_rows=[];ccra_rows=[];started=time.perf_counter();peak=0.;target=2 if args.smoke_steps else TOTAL_STEPS;contract=None
    torch.cuda.reset_peak_memory_stats();print("DFSC_FULL25_PROTOCOL "+json.dumps(config,sort_keys=True),flush=True)
    for epoch in range(1,EPOCHS+1):
        epoch_started=time.perf_counter();model.train();sums={};batches=0
        for _,images,labels in loader:
            images,labels=images.cuda(non_blocking=True),labels.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):result=model(images,labels,step=optimizer.global_step)
            loss=result["losses"]["loss"]
            if not bool(torch.isfinite(loss)):raise FloatingPointError("Non-finite DFSC loss")
            if optimizer.global_step==0:contract=gradient_contract(model,result);write_json(output/"tests/dfsc_gradient_contract.json",contract)
            loss.backward();health=_gradient_health(model) if optimizer.global_step==0 or (optimizer.global_step+1)%100==0 else {};optimizer.step();batches+=1
            if not _finite_model(model):raise FloatingPointError("Non-finite DFSC parameter")
            for key,value in result["losses"].items():sums[key]=sums.get(key,0.)+float(value.detach())
            if optimizer.global_step%100==0 or optimizer.global_step==target:
                row={"epoch":epoch,"step":optimizer.global_step,**{k:v/batches for k,v in sums.items()},"lr":optimizer.param_groups[0]["lr"],**health};loss_rows.append(row);write_csv(output/"metrics/train_loss.csv",loss_rows);print("DFSC_FULL25_STEP "+json.dumps(row,sort_keys=True),flush=True)
            if optimizer.global_step>=target:break
        peak=max(peak,torch.cuda.max_memory_allocated()/1024**3);erow={"epoch":epoch,"step":optimizer.global_step,**{k:v/batches for k,v in sums.items()},"lr":optimizer.param_groups[0]["lr"],"epoch_seconds":time.perf_counter()-epoch_started,"peak_cuda_memory_gib":peak};epoch_rows.append(erow);write_csv(output/"metrics/epoch_summary.csv",epoch_rows);print("DFSC_FULL25_EPOCH "+json.dumps(erow,sort_keys=True),flush=True)
        if args.smoke_steps:
            summary,relation,branches,completion=mechanism_snapshot(model,monitor_loader,0,output,permutations)
            if not np.isfinite(relation["pair_auroc"]) or relation["positive_pairs"]==0 or relation["negative_pairs"]==0 or any(x["max_negative_change"]>0 for x in completion):raise AssertionError("DFSC functional smoke failed")
            write_json(output/"tests/dfsc_smoke_summary.json",{"steps":optimizer.global_step,"finite":True,"parameter_delta":16386,"gradient_contract":contract,"relation":relation,"completion":completion,"validation_accessed":False,"checkpoint_written":False});print("DFSC_FULL25_SMOKE_PASS",flush=True);return
        if epoch in MILESTONES:
            summary,relation,branches,completion=mechanism_snapshot(model,monitor_loader,epoch,output,permutations);summaries.append(summary);relation_rows.append(relation);branch_rows.append(branches);completion_rows+=completion
            _,_,c,_=_rows(summary);ccra_rows+=c;write_json(output/"mechanism/dfsc_summary_history.json",summaries);write_csv(output/"mechanism/dfra_relation_health.csv",relation_rows);write_csv(output/"mechanism/dfra_low_high_health.csv",branch_rows);write_csv(output/"mechanism/dfsc_completion_health.csv",completion_rows);write_csv(output/"mechanism/ccra_health.csv",ccra_rows)
            checkpoint=output/f"checkpoints/dfsc_epoch{epoch:02d}.pth";torch.save(model.state_dict(),checkpoint);write_json(checkpoint.with_suffix(".json"),{"epoch":epoch,"step":optimizer.global_step,"sha256":sha256(checkpoint),"scientific_endpoint":epoch==25});_save_recovery(output/"recovery/latest.pth",model,optimizer,epoch,generator)
    if optimizer.global_step!=TOTAL_STEPS or len(epoch_rows)!=25:raise AssertionError("DFSC did not reach E25")
    source_checkpoint=output/"checkpoints/dfsc_epoch25.pth";final=output/"checkpoints/dfsc_epoch25_final.pth";os.replace(source_checkpoint,final);digest=sha256(final);(output/"checkpoints/dfsc_epoch25_final_sha256.txt").write_text(digest+"\n")
    metadata=json.loads((output/"checkpoints/dfsc_epoch25.json").read_text());metadata.update({"sha256":digest,"sealed_before_segmentation_evaluation":True,"selection":"E25 FINAL only"});write_json(output/"checkpoints/dfsc_epoch25_final.json",metadata);(output/"checkpoints/dfsc_epoch25.json").unlink()
    runtime={"status":"DFSC_FULL25_TRAINING_COMPLETE","epochs":25,"steps":optimizer.global_step,"train_seconds":time.perf_counter()-started,"peak_cuda_memory_gib":peak,"all_finite":True,"validation_accessed":False,"test_accessed":False,"training_paths_accessed":len(accesses),"checkpoint":str(final),"checkpoint_sha256":digest};write_json(output/"provenance/dfsc_runtime.json",runtime);print("DFSC_FULL25_TRAINING_COMPLETE "+json.dumps(runtime,sort_keys=True),flush=True)


if __name__=="__main__":main()
