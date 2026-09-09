"""GCQM train-cohort metrics and preregistered Phase-0 gates."""
from __future__ import annotations

from collections import defaultdict
import itertools
import math
import numpy as np
import torch
from torch.nn import functional as F

from tools.cqrf_diagnostics import batch_health as cqrf_batch_health, summarize as cqrf_summarize, _components
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_counterfactuals import materialize
from tools.momd_diagnostics import _mean,_median,_q,_pair_iou,_pair_soft_iou


def _semantic(v,prefix,mask,pos,rival,bg):
    selected=mask>=.5; v[f"{prefix}_area"].append(float(selected.float().mean())); v[f"{prefix}_components"].append(_components(selected.cpu().numpy())); v[f"{prefix}_empty"].append(float(not bool(selected.any())))
    if bool(pos.any()): v[f"{prefix}_recall"].append(float(selected[pos].float().mean())); v[f"{prefix}_pos"].extend(mask[pos].cpu().tolist())
    if bool(rival.any()): v[f"{prefix}_rival_leak"].append(float(selected[rival].float().mean())); v[f"{prefix}_rival"].extend(mask[rival].cpu().tolist())
    if bool(bg.any()): v[f"{prefix}_bg_leak"].append(float(selected[bg].float().mean())); v[f"{prefix}_bg"].extend(mask[bg].cpu().tolist())
    neg=rival|bg
    if bool(neg.any()): v[f"{prefix}_negative"].extend(mask[neg].cpu().tolist())


def _js(a,b):
    m=.5*(a+b); return .5*((a*(a.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum()+(b*(b.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum())


@torch.no_grad()
def batch_health(output,labels,permutations):
    values=cqrf_batch_health(output,labels)
    positive=F.interpolate(output["target_detail"]["positive"].float(),(56,56),mode="nearest").bool(); bgall=F.interpolate(output["target_detail"]["reliable_background"][:,None].float(),(56,56),mode="nearest")[:,0].bool()
    for stage_index in (2,3):
        stage=output["stages"][stage_index-1]; p=f"s{stage_index}"; g=stage["gcqm"]; cf=materialize(stage,permutations); w=cf["weights"]; b=g["base_probability"].detach().float()
        values[f"{p}_w_error"].append(g["weight_sum_error_max"]); values[f"{p}_c_error"].append(g["contribution_sum_error_max"]); values[f"{p}_detach"].append(float(all(not x.requires_grad for x in cf.values())))
        for image in range(labels.shape[0]):
            present=torch.where(labels[image].bool())[0].tolist()
            for cls in present:
                pos=positive[image,cls]; rival=positive[image,[c for c in range(4) if c!=cls]].any(0); bg=bgall[image]
                for branch in ("primary","pca","pixel","uniform"): _semantic(values,f"{p}_{branch}",cf[branch][image,cls],pos,rival,bg)
                for k in range(len(permutations)): _semantic(values,f"{p}_perm{k}",cf["perm"][k,image,cls],pos,rival,bg)
                if bool(pos.any()): values[f"{p}_cpr"].append(float(cf["primary"][image,cls][pos].mean()/cf["pca"][image,cls][pos].mean().clamp_min(1e-8)))
                for region,mask in (("all",torch.ones_like(pos)),("positive",pos),("rival",rival),("background",bg)):
                    if bool(mask.any()): values[f"{p}_dperm_{region}"].extend((cf["perm"][:,image,cls][:,mask]-cf["primary"][image,cls][mask]).abs().mean(1).cpu().tolist())
                dist=w[image,:,cls]; entropy=-(dist*dist.clamp_min(1e-8).log()).sum()
                values[f"{p}_w_dominant"].append(float(dist.max())); values[f"{p}_w_effective"].append(float(entropy.exp())); values[f"{p}_w_entropy"].append(float(entropy/math.log(len(dist)))); values[f"{p}_w_above_uniform"].append(float((dist>1/len(dist)).sum())); order=torch.argsort(dist,descending=True,stable=True); values[f"{p}_w_top5"].append(float(dist[order[:5]].sum())); values[f"{p}_w_top10"].append(float(dist[order[:10]].sum()))
                spatial_mass=b[image].flatten(1).sum(1); contribution=dist*spatial_mass; contribution/=contribution.sum().clamp_min(1e-8); ce=-(contribution*contribution.clamp_min(1e-8).log()).sum(); values[f"{p}_contrib_effective"].append(float(ce.exp())); values[f"{p}_contrib_entropy"].append(float(ce/math.log(len(dist)))); values[f"{p}_contrib_top5"].append(float(contribution.topk(5).values.sum()))
            if len(present)>=2:
                for c1,c2 in itertools.combinations(present,2):
                    x,y=w[image,:,c1],w[image,:,c2]; values[f"{p}_w_js"].append(float(_js(x,y))); oa=torch.argsort(x,descending=True,stable=True); ob=torch.argsort(y,descending=True,stable=True); values[f"{p}_w_top1_diff"].append(float(oa[0]!=ob[0]))
                    for k in (3,5):
                        inter=sum(int(q in ob[:k]) for q in oa[:k]); values[f"{p}_w_top{k}_jaccard"].append(inter/(2*k-inter))
            ranked=[]
            for cls in present: ranked.extend(torch.argsort(w[image,:,cls],descending=True,stable=True)[:20].tolist())
            ranked=sorted(set(ranked))
            if ranked:
                maps=b[image,ranked]; binary=maps>=.5; area=binary.flatten(1).float().mean(1); values[f"{p}_B_empty"].extend((area==0).float().cpu().tolist()); values[f"{p}_B_area"].extend(area.cpu().tolist()); values[f"{p}_B_components"].extend([_components(x.cpu().numpy()) for x in binary]); values[f"{p}_B_finite"].append(float(torch.isfinite(maps).all()))
                if len(ranked)>1: values[f"{p}_B_binary_iou"].extend(_pair_iou(binary).cpu().tolist()); values[f"{p}_B_soft_iou"].extend(_pair_soft_iou(maps).cpu().tolist()); emb=stage["mask_embedding"][image,ranked].detach().float(); values[f"{p}_B_embed_cos"].extend((F.normalize(emb,dim=-1)@F.normalize(emb,dim=-1).T)[torch.triu(torch.ones(len(ranked),len(ranked),dtype=torch.bool,device=emb.device),diagonal=1)].cpu().tolist())
    return values


def _sem(snapshot,stage,m,prefix):
    pos=m[f"{prefix}_pos"]; neg=m[f"{prefix}_negative"]; pm,pn=_mean(pos),_mean(neg); areas=m[f"{prefix}_area"]
    return {"snapshot":snapshot,"stage":stage,"positive_recall":_median(m[f"{prefix}_recall"]),"positive_confidence":pm,"positive_F_median":_median(pos),"rival_leakage":_mean(m[f"{prefix}_rival_leak"]),"background_leakage":_mean(m[f"{prefix}_bg_leak"]),"probability_gap":pm-pn,"rival_gap":pm-_mean(m[f"{prefix}_rival"]),"background_gap":pm-_mean(m[f"{prefix}_bg"]),"empty_fraction":_mean(m[f"{prefix}_empty"]),"median_area":_median(areas),"fraction_area_lt_001":_mean([x<.01 for x in areas]),"fraction_area_gt_090":_mean([x>.9 for x in areas]),"component_p90":_q(m[f"{prefix}_components"],.9)}


def summarize(snapshot,batches,pmec_rows,model,permutation_count=8):
    base=cqrf_summarize(snapshot,batches,pmec_rows,model); m=defaultdict(list)
    for batch in batches:
        for k,v in batch.items(): m[k].extend(v)
    primary=[]; pca=[]; pixel=[]; perms=[]; sensitivity=[]; conservation=[]; conditioning=[]; utilization=[]; basis=[]; contrib=[]; detach=[]; vpca=[]; vpixel=[]
    for stage in (2,3):
        p=f"s{stage}"; pr=_sem(snapshot,stage,m,f"{p}_primary"); pc=_sem(snapshot,stage,m,f"{p}_pca"); px=_sem(snapshot,stage,m,f"{p}_pixel"); prows=[_sem(snapshot,stage,m,f"{p}_perm{i}")|{"permutation":i} for i in range(permutation_count)]; primary.append(pr|{"CPR":_median(m[f"{p}_cpr"])}); pca.append(pc); pixel.append(px); perms.extend(prows)
        d=m[f"{p}_dperm_all"]; sensitivity.append({"snapshot":snapshot,"stage":stage,"D_perm_mean":_mean(d),"D_perm_std":float(np.std(d)) if d else 0,"D_perm_min":min(d or [0]),"D_perm_max":max(d or [0]),**{f"D_perm_{r}":_mean(m[f"{p}_dperm_{r}"]) for r in ("positive","rival","background")}})
        conservation.append({"snapshot":snapshot,"stage":stage,"max_weight_sum_error":max(m[f"{p}_w_error"] or [0]),"max_contribution_sum_error":max(m[f"{p}_c_error"] or [0]),"all_finite":bool(base["all_finite"])})
        conditioning.append({"snapshot":snapshot,"stage":stage,"JS_median":_median(m[f"{p}_w_js"]),"JS_mean":_mean(m[f"{p}_w_js"]),"top1_difference":_mean(m[f"{p}_w_top1_diff"]),"top3_jaccard":_mean(m[f"{p}_w_top3_jaccard"]),"top5_jaccard":_mean(m[f"{p}_w_top5_jaccard"])})
        utilization.append({"snapshot":snapshot,"stage":stage,"dominant_share":_median(m[f"{p}_w_dominant"]),"effective_queries":_median(m[f"{p}_w_effective"]),"normalized_entropy":_median(m[f"{p}_w_entropy"]),"queries_above_uniform":_median(m[f"{p}_w_above_uniform"]),"top5_mass":_median(m[f"{p}_w_top5"]),"top10_mass":_median(m[f"{p}_w_top10"])})
        areas=m[f"{p}_B_area"]; basis.append({"snapshot":snapshot,"stage":stage,"empty_fraction":_mean(m[f"{p}_B_empty"]),"fraction_area_gt_090":_mean([x>.9 for x in areas]),"median_area":_median(areas),"component_p90":_q(m[f"{p}_B_components"],.9),"binary_iou_median":_median(m[f"{p}_B_binary_iou"]),"soft_iou_median":_median(m[f"{p}_B_soft_iou"]),"embedding_cosine_median":_median(m[f"{p}_B_embed_cos"]),"all_finite":bool(min(m[f"{p}_B_finite"] or [0]))})
        contrib.append({"snapshot":snapshot,"stage":stage,"effective_contributing_queries":_median(m[f"{p}_contrib_effective"]),"normalized_entropy":_median(m[f"{p}_contrib_entropy"]),"top5_mass":_median(m[f"{p}_contrib_top5"])})
        detach.append({"snapshot":snapshot,"stage":stage,"all_references_detached":bool(min(m[f"{p}_detach"] or [0])),"w_detached":True})
        vpca.append({"snapshot":snapshot,"stage":stage,"gap_gain":pr["probability_gap"]-pc["probability_gap"],"recall_gain":pr["positive_recall"]-pc["positive_recall"],"confidence_gain":pr["positive_confidence"]-pc["positive_confidence"],"rival_reduction":pc["rival_leakage"]-pr["rival_leakage"],"background_reduction":pc["background_leakage"]-pr["background_leakage"]})
        vpixel.append({"snapshot":snapshot,"stage":stage,"gap_delta":pr["probability_gap"]-px["probability_gap"],"recall_delta":pr["positive_recall"]-px["positive_recall"],"confidence_delta":pr["positive_confidence"]-px["positive_confidence"],"rival_delta":pr["rival_leakage"]-px["rival_leakage"],"background_delta":pr["background_leakage"]-px["background_leakage"]})
    return {**base,"primary_semantic_health":primary,"perm_semantic_health":perms,"pca_semantic_health":pca,"pixel_reference_health":pixel,"query_identity_sensitivity":sensitivity,"weight_conservation":conservation,"weight_class_conditioning":conditioning,"weight_utilization":utilization,"B_basis_health":basis,"contribution_utilization":contrib,"counterfactual_detach_audit":detach,"vs_pca_gain":vpca,"vs_pixel_noninferiority":vpixel,"ccra_health":base["responsibility_integrity"]}


def _row(s,k,stage=3): return next(x for x in s[k] if x["stage"]==stage)


def apply_epoch2_screen(s):
    con=_row(s,"weight_conservation"); sem=_row(s,"primary_semantic_health"); rc=_row(s,"responsibility_complementarity"); u=_row(s,"weight_utilization"); det=_row(s,"counterfactual_detach_audit")
    checks={"engineering":not con["all_finite"] or con["max_weight_sum_error"]>1e-5 or con["max_contribution_sum_error"]>1e-6 or not det["all_references_detached"],"semantic":sem["CPR"]<.40 or sem["positive_recall"]<.50 or sem["positive_F_median"]<.40 or sem["empty_fraction"]>.40,"specificity":sem["rival_leakage"]>.70 or sem["background_leakage"]>.50 or sem["probability_gap"]<=0,"ccra":rc["responsibility_iou_median"]>.75 and rc["distinct_peak_fraction"]<.40,"monopoly":u["dominant_share"]>.90 and u["effective_queries"]<2}
    failed=[k for k,v in checks.items() if v]
    decision="GCQM_ENGINEERING_BLOCKED" if checks["engineering"] else "GCQM_PHASE0_NOGO" if failed else "CONTINUE_TO_E5_UNCHANGED"
    return {"decision":decision,"checks":checks,"failed_criteria":failed}


def apply_final_gate(history):
    s=history[-1]; r=_row(s,"responsibility_integrity"); rc=_row(s,"responsibility_complementarity"); ru=_row(s,"responsibility_utilization"); con=_row(s,"weight_conservation"); sem=_row(s,"primary_semantic_health"); sens=_row(s,"query_identity_sensitivity"); gain=_row(s,"vs_pca_gain"); ni=_row(s,"vs_pixel_noninferiority"); wc=_row(s,"weight_class_conditioning"); wu=_row(s,"weight_utilization"); b=_row(s,"B_basis_health"); pca=_row(s,"pca_health"); deep=s["deep_gate_health"]
    prows=[x for x in s["perm_semantic_health"] if x["stage"]==3]; pm={k:_mean([x[k] for x in prows]) for k in ("probability_gap","rival_leakage","background_leakage","positive_confidence","positive_recall")}; permchecks=[sem["probability_gap"]>=pm["probability_gap"]+.03,sem["rival_leakage"]<=pm["rival_leakage"]-.02,sem["background_leakage"]<=pm["background_leakage"]-.005,sem["positive_confidence"]>=pm["positive_confidence"]+.02,sem["positive_recall"]>=pm["positive_recall"]+.02]
    pcachecks=[gain["gap_gain"]>=.10,gain["rival_reduction"]>=.03,gain["background_reduction"]>=.005,gain["confidence_gain"]>=.03,gain["recall_gain"]>=.03]
    by={_row(x,"primary_semantic_health")["snapshot"]:x for x in history}; e3=by.get("epoch3"); late=bool(e3) and sens["D_perm_mean"]>=_row(e3,"query_identity_sensitivity")["D_perm_mean"]-.03 and gain["gap_gain"]>=_row(e3,"vs_pca_gain")["gap_gain"]-.05 and sem["CPR"]>=_row(e3,"primary_semantic_health")["CPR"]-.10
    groups={"A_CCRA_preserved":r["max_abs_sum_error"]<=1e-5 and rc["responsibility_iou_median"]<=.35 and rc["distinct_peak_fraction"]>=.80 and ru["dominant_share_median"]<.75 and ru["effective_queries_median"]>1.5,"B_GCQM_conservation":con["max_weight_sum_error"]<=1e-6 and con["max_contribution_sum_error"]<=1e-6 and con["all_finite"],"C_primary_mask_healthy":sem["CPR"]>=.80 and sem["positive_recall"]>=.90 and sem["positive_F_median"]>=.70 and sem["empty_fraction"]<.05 and sem["rival_leakage"]<.30 and sem["background_leakage"]<.10 and sem["probability_gap"]>.40 and .05<=sem["median_area"]<=.80 and sem["fraction_area_lt_001"]<.10 and sem["fraction_area_gt_090"]<.20 and sem["component_p90"]<=15,"D_query_identity":sens["D_perm_mean"]>=.05 and sum(permchecks)>=2,"E_beats_PCA":sum(pcachecks)>=2 and (gain["gap_gain"]>=.10 or gain["rival_reduction"]>=.03),"F_pixel_unnecessary":ni["gap_delta"]>=-.03 and ni["recall_delta"]>=-.03 and ni["confidence_delta"]>=-.03 and ni["rival_delta"]<=.03 and ni["background_delta"]<=.01,"G_class_conditioned_allocation":wc["JS_median"]>=.05 and wc["top1_difference"]>=.50 and wu["dominant_share"]<.80 and wu["effective_queries"]>=2,"H_B_nondegenerate":b["empty_fraction"]<.70 and b["fraction_area_gt_090"]<.30 and b["all_finite"],"I_late_stability":late,"J_PCA_deep":pca["present_class_query_coverage"]>=.93 and pca["absent_class_dominance"]<=.20 and deep["present_absent_gap"]>.40 and s["all_finite"]}
    go=all(groups.values()); strong=go and sem["CPR"]>=1 and sem["positive_recall"]>=.95 and sem["positive_F_median"]>=.75 and sem["rival_leakage"]<.20 and sem["background_leakage"]<.05 and sem["probability_gap"]>.55 and sens["D_perm_mean"]>=.10 and gain["gap_gain"]>=.20 and gain["rival_reduction"]>=.05 and ni["gap_delta"]>=-.01 and ni["recall_delta"]>=-.01 and wc["JS_median"]>=.15 and wc["top1_difference"]>=.80 and wu["dominant_share"]<.60 and wu["effective_queries"]>=5 and late
    return {"decision":"GCQM_PHASE0_STRONG_GO" if strong else "GCQM_PHASE0_GO" if go else "GCQM_PHASE0_NOGO","gate_groups":groups,"strong_go":strong,"permutation_checks":permchecks,"pca_checks":pcachecks}


__all__=["permutation_payload","batch_health","summarize","apply_epoch2_screen","apply_final_gate"]
