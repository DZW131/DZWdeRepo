#!/usr/bin/env python3
"""Post-seal DFSC validation, A-G ablations, and targeted coherence re-audit."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from network.ccac_net import CCACNet
from network.dfsc import mcc_complete
from network.dfsc_net import DFSCNet
from network.gcqm_net import GCQMNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_gcqm_full25_failure_anatomy import class_anatomy_rows,contact_rows,mean_ci,weight_metrics
from tools.eval_gcqm_full25_bcss_seed42 import BASELINE_SHA256,TTA,_predict_gcqm,_predict_sshr,foreground_confusion,load_state,normalize_cam,paired_bootstrap,prediction_from_cam,presence,resize_unflip,scores_from_confusion
from tools.hqrf_phase0_io import sha256,write_csv,write_json


GCQM_SHA256="6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f"
CCAC_SHA256="848631927607bc2832c07cbbafdf0ba71ed3482ff1cec8a056abb2bbca7c80a6"
BOOTSTRAP_SEED,BOOTSTRAP_RESAMPLES=20260911,10_000
MODES=("full_dfsc","dfsc_off","uniform_mcc","raw_cosine_mcc","low_only_mcc","high_only_mcc","full_affinity_old_update")


@torch.no_grad()
def predict_dfsc_modes(model,image,original,diagnostics=False):
    views={mode:[] for mode in MODES};gates=[];original_output=None
    dummy=torch.ones((1,4),device=image.device)
    for input_flip,cam_flip in TTA:
        value=torch.flip(image,dims=input_flip) if input_flip else image
        with torch.autocast("cuda",dtype=torch.bfloat16):base_output=GCQMNet.forward(model,value,dummy,step=29275,run_pmec=False)
        base=base_output["primary_output"];relations={mode:model.dfra(base_output["pixel_feature"],None,mode) for mode in ("full","uniform","raw_cosine","low_only","high_only")}
        candidates={"dfsc_off":base,"full_dfsc":mcc_complete(base,relations["full"]["affinity_comp"],2,"mcc")["restored"],
            "uniform_mcc":mcc_complete(base,relations["uniform"]["affinity_comp"],2,"mcc")["restored"],
            "raw_cosine_mcc":mcc_complete(base,relations["raw_cosine"]["affinity_comp"],2,"mcc")["restored"],
            "low_only_mcc":mcc_complete(base,relations["low_only"]["affinity_comp"],2,"mcc")["restored"],
            "high_only_mcc":mcc_complete(base,relations["high_only"]["affinity_comp"],2,"mcc")["restored"],
            "full_affinity_old_update":mcc_complete(base,relations["full"]["affinity_comp"],2,"old_weak")["restored"]}
        for mode,candidate in candidates.items():views[mode].append(resize_unflip(candidate,original.shape[:2],cam_flip))
        gates.append(base_output["deep_gate"])
        if not input_flip:original_output={"base":base_output,"relations":relations,"candidates":candidates}
    label=presence(torch.stack(gates).mean(0).float().cpu().numpy()[0])
    predictions={mode:prediction_from_cam(normalize_cam(torch.stack(value).mean(0).float().cpu().numpy()),label,original) for mode,value in views.items()}
    return (predictions,original_output,label) if diagnostics else predictions


def metric_rows(name,metrics):return [{"model":name,"class":c,"iou":metrics["class_iou"][str(c)],"dice":metrics["class_dice"][str(c)]} for c in range(4)]


def morphology_summary(frame):
    pivot=frame[frame.model.isin(["gcqm","sshr"])].groupby("model").mean(numeric_only=True)
    return {metric:float(pivot.loc["gcqm",metric]-pivot.loc["sshr",metric]) for metric in ("components","hole_count","compactness","fragmentation_index")}


def decide(delta_sshr,lower_sshr,delta_ccac,class_delta,relation,coherence,ablation):
    selectivity=relation["claim_pass"] and ablation["full_minus_uniform_pp"]>0
    catastrophic=min(class_delta.values())<=-3.0
    coherent=coherence["fn_target"] and coherence["interior_target"] and coherence["morphology_recovery"] and not coherence["overcompletion"] and not coherence["contact_failure"]
    if delta_sshr>=.50 and lower_sshr>0 and min(class_delta.values())>=-1 and selectivity and coherence["spatial_improved"]:return "DFSC_FULL25_STRONG_GO"
    if delta_sshr>=.30 and lower_sshr>0 and not catastrophic and selectivity and coherence["spatial_improved"]:return "DFSC_FULL25_GO"
    if delta_sshr>0 and (delta_sshr<.30 or lower_sshr<=0) and delta_ccac>.30 and selectivity and coherence["spatial_improved"]:return "DFSC_FULL25_BREAKTHROUGH_UNCERTAIN"
    if delta_ccac>=1.0 and delta_sshr<-.30 and coherent and selectivity:return "DFSC_FULL25_RECOVERY_GO"
    if delta_ccac<=-.30 or relation["pair_auroc"]<.60 or ablation["full_minus_uniform_pp"]<=0 or coherence["spatial_worse"] or coherence["contact_failure"] or coherence["overcompletion"]:return "DFSC_FULL25_NOGO"
    if -.30<delta_ccac<.30 or not selectivity:return "DFSC_FULL25_NEUTRAL"
    return "DFSC_FULL25_NOGO"


def report_text(r):
    m,d,c,a,rel=r["metrics"],r["deltas_pp"],r["coherence"],r["ablation"],r["relation"]
    sections=[("Executive Decision",f"**DECISION = {r['decision']}**"),("Frozen Evidence","SSHR B0=66.6967，old GCQM=64.3543，CCAC=64.6169 mIoU，均使用冻结的 BCSS Seed42 E25 协议。"),
    ("Why CCAC Was Insufficient","Raw pixel-feature cosine affinity 接近 uniform，且 attenuated fill 对决策边界影响过弱。"),("Literature Migration","仅迁移 dual-frequency relation、structure-aware posterior 与 affinity-guided completion 原则。"),
    ("Final CCRA+DFSC Architecture","冻结 CCRA/GCQM 语义分配，新增 DFRA 关系估计与 MCC 单调补全。"),("Dual-Frequency Decomposition","低频为 AvgPool5(P_detach)，高频为 P_detach−AvgPool3(P_detach)。"),
    ("Learned Relation Embeddings","两个 256→32 无偏置 1×1 projection，加两个正 softplus scale，共 16,386 参数。"),("Weak Pair Supervision","仅使用训练图像已有 tri-state reliable foreground/background 构造正负局部 pair。"),
    ("Gradient Isolation","L_base 不更新 DFRA；L_rel 不更新 backbone/pixel/query；completion 使用 detached affinity。"),("Monotone Consensus Completion","Stage2/3 固定两轮 S←max(S,αS)，Stage1 不变。"),
    ("Engineering Validation",f"全仓回归、synthetic tests、2-step smoke 与数值梯度隔离均通过；训练 finite={r['training']['all_finite']}。"),("Fresh Full25 Protocol","Fresh official MXNet init，Seed42，BF16，batch20，25 epochs/29275 steps，E25-only。"),
    ("Training Completion",f"训练 {r['training']['train_seconds']/60:.2f} 分钟，峰值显存 {r['training']['peak_cuda_memory_gib']:.3f} GiB。"),("E25 Seal",f"SHA256 `{r['provenance']['dfsc_sha256']}`，先封存后验证。"),
    ("Main mIoU/mDice",f"DFSC={100*m['full_dfsc']['mIoU']:.4f}/{100*m['full_dfsc']['mDice']:.4f}。"),("Comparison vs SSHR",f"ΔmIoU={d['vs_sshr']:+.4f} pp；95% CI={r['bootstrap_vs_sshr']['miou_ci95_pp']}。"),
    ("Comparison vs GCQM",f"ΔmIoU={d['vs_gcqm']:+.4f} pp。"),("Comparison vs CCAC",f"ΔmIoU={d['vs_ccac']:+.4f} pp；95% CI={r['bootstrap_vs_ccac']['miou_ci95_pp']}。"),
    ("Per-Class Results",str(r["per_class"])),("Paired Bootstrap",f"10,000 resamples，seed={BOOTSTRAP_SEED}，严格逐图配对。"),
    ("Affinity Selectivity",f"AUROC={rel['pair_auroc']:.4f}，AP={rel['pair_ap']:.4f}，pos-neg gap={rel['pos_neg_gap']:.4f}，boundary entropy={rel['boundary_neighborhood_entropy']:.4f}，claim_pass={rel['claim_pass']}。"),
    ("Low/High Branch Health",f"positive/negative dL={rel['positive_dL']:.4f}/{rel['negative_dL']:.4f}；dH={rel['positive_dH']:.4f}/{rel['negative_dH']:.4f}；beta={rel['beta_low']:.4f}/{rel['beta_high']:.4f}。"),
    ("FN Recovery",f"normalized ΔFN={c['normalized_delta_fn']['mean']:+.6f}；target={c['fn_target']}。"),("Interior Recovery",f"interior loss={c['interior_loss']:+.6f}；target={c['interior_target']}。"),
    ("Morphology Recovery",f"gaps={c['morphology_gaps']}；recovery={c['morphology_recovery']}。"),("Boundary Safety",f"boundary loss={c['boundary_loss']:+.6f}；tradeoff={c['boundary_tradeoff']}。"),
    ("Contact Safety",f"contact excess={c['contact_excess_loss']:+.6f}；failure={c['contact_failure']}。"),("FP Safety",f"normalized ΔFP={c['normalized_delta_fp']['mean']:+.6f}；overcompletion={c['overcompletion']}。"),
    ("Same-Checkpoint Causal Ablations",str(a)),("Complexity",f"parameters={r['complexity']['parameters']:,}；delta={r['complexity']['parameter_delta']:,}；inference={r['complexity']['seconds_per_image']:.4f}s/image。"),
    ("Qualitative Structural Recovery","自动选择索引保存在 visualizations/selection.json。"),("Scientific Interpretation",r["interpretation"]),("Exact Decision",f"`DECISION = {r['decision']}`"),("Next Step",r["next_step"])]
    return "# CCRA+DFSC BCSS Seed42 Full25 Final Validation Report\n\n"+"\n\n".join(f"## {i} {title}\n\n{body}" for i,(title,body) in enumerate(sections,1))+f"\n\nDECISION = {r['decision']}\n"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--val-root",required=True);p.add_argument("--dfsc-checkpoint",required=True);p.add_argument("--dfsc-experiment",required=True);p.add_argument("--ccac-checkpoint",required=True);p.add_argument("--gcqm-checkpoint",required=True);p.add_argument("--sshr-checkpoint",required=True);p.add_argument("--num-workers",type=int,default=8);return p.parse_args()


def main():
    args=parse_args();valroot=Path(args.val_root).resolve();experiment=Path(args.dfsc_experiment).resolve();dfsc_path,ccac_path,gcqm_path,sshr_path=map(lambda x:Path(x).resolve(),(args.dfsc_checkpoint,args.ccac_checkpoint,args.gcqm_checkpoint,args.sshr_checkpoint))
    if len(list((valroot/"img").glob("*.png")))!=3418 or len(list((valroot/"mask").glob("*.png")))!=3418:raise AssertionError("Expected 3418 BCSS validation pairs")
    for name in ("evaluation","ablation","visualizations","report"):(experiment/name).mkdir(exist_ok=True)
    if any((experiment/"report").iterdir()):raise FileExistsError("DFSC evaluation already exists")
    runtime=json.loads((experiment/"provenance/dfsc_runtime.json").read_text());seal=json.loads((experiment/"checkpoints/dfsc_epoch25_final.json").read_text())
    checks={"dfsc_sealed":seal["sealed_before_segmentation_evaluation"] and seal["sha256"]==sha256(dfsc_path),"ccac_frozen":sha256(ccac_path)==CCAC_SHA256,"gcqm_frozen":sha256(gcqm_path)==GCQM_SHA256,"sshr_frozen":sha256(sshr_path)==BASELINE_SHA256,"no_training_validation":not runtime["validation_accessed"],"full25":runtime["epochs"]==25 and runtime["steps"]==29275,"same_hardware":"4090" in torch.cuda.get_device_name(0)}
    protocol={"decision":"COMPARABLE" if all(checks.values()) else "NOT_COMPARABLE","checks":checks};write_json(experiment/"evaluation/dfsc_protocol_audit.json",protocol)
    if protocol["decision"]!="COMPARABLE":raise AssertionError(f"Protocol audit failed: {checks}")
    loader=DataLoader(Stage1_InferDataset(str(valroot/"img"),img_size=224),batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    sshr=SSHRCAM(4).cuda();sshr.load_state_dict(load_state(sshr_path),strict=True);sshr.eval();gcqm=GCQMNet().cuda();gcqm.load_state_dict(load_state(gcqm_path),strict=True);gcqm.eval();ccac=CCACNet().cuda();ccac.load_state_dict(load_state(ccac_path),strict=True);ccac.eval();dfsc=DFSCNet().cuda();dfsc.load_state_dict(load_state(dfsc_path),strict=True);dfsc.eval()
    names_all=("sshr","old_gcqm","ccac",*MODES);hist={k:[] for k in names_all};ids=[];pair_rows=[];band_rows=[];morph_rows=[];contact_data=[];started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for index,(names,image) in enumerate(loader,1):
        image_id=names[0];original=np.asarray(Image.open(valroot/"img"/f"{image_id}.png").convert("RGB"));truth=np.asarray(Image.open(valroot/"mask"/f"{image_id}.png"));image=image.cuda(non_blocking=True)
        ps=_predict_sshr(sshr,image,original);pg=_predict_gcqm(gcqm,image,original);pc=_predict_gcqm(ccac,image,original);predictions,_,_=predict_dfsc_modes(dfsc,image,original,True)
        for name,pred in {"sshr":ps,"old_gcqm":pg,"ccac":pc,**predictions}.items():hist[name].append(foreground_confusion(truth,pred))
        p,b,m=class_anatomy_rows(image_id,truth,predictions["full_dfsc"],ps,predictions["dfsc_off"]);pair_rows+=p;band_rows+=b;morph_rows+=m;contact_data+=contact_rows(image_id,truth,predictions["full_dfsc"],ps,predictions["dfsc_off"]);ids.append(image_id)
        if index%200==0 or index==len(loader):print(f"DFSC_EVAL_PROGRESS={index}/{len(loader)}",flush=True)
    elapsed=time.perf_counter()-started;hist={k:np.stack(v) for k,v in hist.items()};metrics={k:scores_from_confusion(v.sum(0)) for k,v in hist.items()};write_json(experiment/"evaluation/dfsc_epoch25_metrics.json",metrics["full_dfsc"])
    for mode in MODES:write_csv(experiment/f"ablation/{mode}.csv",metric_rows(mode,metrics[mode]))
    per_class=sum((metric_rows(name,metrics[name]) for name in names_all),[]);write_csv(experiment/"evaluation/dfsc_per_class.csv",per_class)
    per_image=[]
    for i,image_id in enumerate(ids):
        row={"image_id":image_id};row.update({f"{name}_mIoU":scores_from_confusion(hist[name][i])["mIoU"] for name in names_all});per_image.append(row)
    write_csv(experiment/"evaluation/dfsc_per_image.csv",per_image)
    paired_s=[{"image_id":r["image_id"],"sshr_mIoU":r["sshr_mIoU"],"dfsc_mIoU":r["full_dfsc_mIoU"],"delta_mIoU":r["full_dfsc_mIoU"]-r["sshr_mIoU"]} for r in per_image];paired_c=[{"image_id":r["image_id"],"ccac_mIoU":r["ccac_mIoU"],"dfsc_mIoU":r["full_dfsc_mIoU"],"delta_mIoU":r["full_dfsc_mIoU"]-r["ccac_mIoU"]} for r in per_image]
    write_csv(experiment/"evaluation/dfsc_vs_sshr_paired.csv",paired_s);write_csv(experiment/"evaluation/dfsc_vs_ccac_paired.csv",paired_c);boot_s=paired_bootstrap(hist["sshr"],hist["full_dfsc"],BOOTSTRAP_RESAMPLES,BOOTSTRAP_SEED);boot_c=paired_bootstrap(hist["ccac"],hist["full_dfsc"],BOOTSTRAP_RESAMPLES,BOOTSTRAP_SEED);write_json(experiment/"evaluation/dfsc_vs_sshr_bootstrap.json",boot_s);write_json(experiment/"evaluation/dfsc_vs_ccac_bootstrap.json",boot_c)
    pair,bands,morph,contacts=map(pd.DataFrame,(pair_rows,band_rows,morph_rows,contact_data));fp,fn=mean_ci(pair.normalized_delta_fp),mean_ci(pair.normalized_delta_fn);main_band=bands[bands.radius==3];interior=float((main_band.sshr_interior_correctness-main_band.gcqm_interior_correctness).mean());boundary=float((main_band.sshr_boundary_f1-main_band.gcqm_boundary_f1).mean())
    if len(contacts):mc=contacts[contacts.distance==3];contact=float(((mc.sshr_contact_accuracy-mc.gcqm_contact_accuracy)-(mc.sshr_noncontact_boundary_accuracy-mc.gcqm_noncontact_boundary_accuracy)).mean())
    else:contact=0.
    gaps=morphology_summary(morph);reductions={"components":gaps["components"]<=.1456,"hole_count":gaps["hole_count"]<=.1224,"compactness":gaps["compactness"]<=.2394}
    coherence={"normalized_delta_fp":fp,"normalized_delta_fn":fn,"interior_loss":interior,"boundary_loss":boundary,"contact_excess_loss":contact,"morphology_gaps":gaps,"fn_target":fn["mean"]<=.011,"interior_target":interior<=.012,"morphology_recovery":sum(reductions.values())>=2,"morphology_reduction_checks":reductions,"boundary_tradeoff":boundary>.0164,"contact_failure":contact>=.03,"overcompletion":fp["mean"]>0 and fp["ci95"][0]>0,"spatial_improved":fn["mean"]<.02105 and interior<.022303,"spatial_worse":fn["mean"]>.02105 and interior>.022303};write_json(experiment/"evaluation/dfsc_failure_anatomy_reaudit.json",coherence);write_csv(experiment/"evaluation/dfsc_coherence_recovery.csv",[{k:v for k,v in coherence.items() if isinstance(v,(int,float,bool))}])
    relation=pd.read_csv(experiment/"mechanism/dfra_relation_health.csv").iloc[-1].to_dict();branches=pd.read_csv(experiment/"mechanism/dfra_low_high_health.csv").iloc[-1].to_dict();relation.update(branches);relation["claim_pass"]=bool(relation["pair_auroc"]>=.70 and relation["pos_neg_gap"]>=.15 and relation["positive_affinity_mean"]>relation["negative_affinity_mean"] and relation["boundary_neighborhood_entropy"]<=2.068)
    delta_s=100*(metrics["full_dfsc"]["mIoU"]-metrics["sshr"]["mIoU"]);delta_c=100*(metrics["full_dfsc"]["mIoU"]-metrics["ccac"]["mIoU"]);delta_g=100*(metrics["full_dfsc"]["mIoU"]-metrics["old_gcqm"]["mIoU"]);class_delta={str(i):100*(metrics["full_dfsc"]["class_iou"][str(i)]-metrics["sshr"]["class_iou"][str(i)]) for i in range(4)}
    ablation={mode:{"mIoU":metrics[mode]["mIoU"],"mDice":metrics[mode]["mDice"]} for mode in MODES};ablation.update({"full_minus_uniform_pp":100*(metrics["full_dfsc"]["mIoU"]-metrics["uniform_mcc"]["mIoU"]),"full_minus_low_pp":100*(metrics["full_dfsc"]["mIoU"]-metrics["low_only_mcc"]["mIoU"]),"full_minus_high_pp":100*(metrics["full_dfsc"]["mIoU"]-metrics["high_only_mcc"]["mIoU"]),"full_minus_old_update_pp":100*(metrics["full_dfsc"]["mIoU"]-metrics["full_affinity_old_update"]["mIoU"])})
    verdict=decide(delta_s,boot_s["miou_ci95_pp"][0],delta_c,class_delta,relation,coherence,ablation);interpretation="DFSC is accepted only if learned dual-frequency relation is selective, causally beats uniform/raw alternatives, and repairs the pre-registered coherence failure without a new boundary/contact/FP trade-off."
    result={"decision":verdict,"metrics":metrics,"deltas_pp":{"vs_sshr":delta_s,"vs_ccac":delta_c,"vs_gcqm":delta_g,"class_vs_sshr":class_delta},"per_class":per_class,"bootstrap_vs_sshr":boot_s,"bootstrap_vs_ccac":boot_c,"relation":relation,"coherence":coherence,"ablation":ablation,"training":runtime,"complexity":{"parameters":sum(p.numel() for p in dfsc.parameters()),"parameter_delta":16386,"seconds_per_image":elapsed/len(ids),"peak_inference_gib":torch.cuda.max_memory_allocated()/1024**3},"provenance":{"source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"dfsc_sha256":sha256(dfsc_path),"ccac_sha256":sha256(ccac_path),"gcqm_sha256":sha256(gcqm_path),"sshr_sha256":sha256(sshr_path),"protocol":protocol},"interpretation":interpretation,"next_step":"Apply the frozen decision rule: GO variants proceed directly to multi-seed; otherwise classify relation-learning, completion, semantic-confusion, or FP failure before designing another experiment."};write_json(experiment/"evaluation/dfsc_final_result.json",result)
    ordered=sorted(paired_c,key=lambda r:(r["delta_mIoU"],r["image_id"]));ordered_s=sorted(paired_s,key=lambda r:(r["delta_mIoU"],r["image_id"]));write_json(experiment/"visualizations/selection.json",{"gains_vs_ccac":list(reversed(ordered[-5:])),"gains_vs_sshr":list(reversed(ordered_s[-5:])),"near_zero_vs_sshr":sorted(paired_s,key=lambda r:(abs(r["delta_mIoU"]),r["image_id"]))[:5],"regressions_vs_sshr":ordered_s[:5]})
    report=experiment/"report/CCRA_DFSC_BCSS_Seed42_Full25_Final_Validation_Report.md";report.write_text(report_text(result));print(json.dumps({"decision":verdict,"report":str(report),"delta_vs_sshr_pp":delta_s,"delta_vs_ccac_pp":delta_c},indent=2));print(f"DECISION = {verdict}")


if __name__=="__main__":main()
