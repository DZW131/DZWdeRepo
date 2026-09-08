"""Train-cohort diagnostics and preregistered MOMD Phase-0 gates."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np
import torch
from torch.nn import functional as F

from tools.cqrf_diagnostics import _components, batch_health as cqrf_batch_health, summarize as cqrf_summarize


def _mean(x): return float(np.mean(x)) if x else 0.0
def _median(x): return float(np.median(x)) if x else 0.0
def _q(x, q): return float(np.quantile(x, q)) if x else 0.0
def _upper(n, device): return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def _pair_iou(binary):
    inter = (binary[:, None] & binary[None]).flatten(2).sum(-1).float()
    union = (binary[:, None] | binary[None]).flatten(2).sum(-1).float().clamp_min(1)
    return (inter / union)[_upper(binary.shape[0], binary.device)]


def _pair_soft_iou(value):
    inter = torch.minimum(value[:, None], value[None]).flatten(2).sum(-1)
    union = torch.maximum(value[:, None], value[None]).flatten(2).sum(-1).clamp_min(1e-8)
    return (inter / union)[_upper(value.shape[0], value.device)]


def _exact_top_fraction_support(value, fraction=.20):
    """Select exactly ceil(fraction*HW) pixels per map with stable tie handling."""
    flat=value.flatten(1); count=math.ceil(fraction*flat.shape[1])
    indices=torch.argsort(flat,dim=1,descending=True,stable=True)[:,:count]
    support=torch.zeros_like(flat,dtype=torch.bool); support.scatter_(1,indices,True)
    return support.reshape_as(value)


def _rank_corr(a, b):
    ar = torch.argsort(torch.argsort(a.flatten())).float(); br = torch.argsort(torch.argsort(b.flatten())).float()
    ar -= ar.mean(); br -= br.mean()
    return float((ar * br).sum() / (ar.square().sum().sqrt() * br.square().sum().sqrt()).clamp_min(1e-8))


@torch.no_grad()
def batch_health(output, labels):
    values = cqrf_batch_health(output, labels)
    positive = F.interpolate(output["target_detail"]["positive"].float(), (56, 56), mode="nearest").bool()
    background = F.interpolate(output["target_detail"]["reliable_background"][:, None].float(), (56, 56), mode="nearest")[:, 0].bool()
    for stage_index in (2, 3):
        stage = output["stages"][stage_index - 1]; prefix = f"s{stage_index}"; momd = stage["momd"]
        base = momd["base_probability"].float(); final = momd["mixture"].float()
        route = momd["routing"].float(); contribution = momd["contribution"].float()
        posterior = momd["posterior_share"].float(); reference = momd["reference_envelope"].float()
        values[f"{prefix}_routing_max"].append(momd["routing_sum_error_max"])
        values[f"{prefix}_routing_mean"].append(momd["routing_sum_error_mean"])
        values[f"{prefix}_contribution_max"].append(momd["contribution_sum_error_max"])
        values[f"{prefix}_posterior_max"].append(momd["posterior_sum_error_max"])
        values[f"{prefix}_fallback"].append(momd["fallback_fraction"])
        for image in range(labels.shape[0]):
            for cls in torch.where(labels[image].bool())[0].tolist():
                scores = stage["confidence"]["joint"][image, :, cls].float()
                indices = torch.argsort(scores, descending=True, stable=True)[:20]
                qmaps = posterior[image, indices, cls]; amaps = route[image, indices, cls]
                bmaps = base[image, indices]
                qsupport = _exact_top_fraction_support(qmaps, .20)
                values[f"{prefix}_q_iou"].extend(_pair_iou(qsupport).cpu().tolist())
                values[f"{prefix}_q_gt075"].extend((_pair_iou(qsupport) > .75).float().cpu().tolist())
                values[f"{prefix}_q_gt090"].extend((_pair_iou(qsupport) > .90).float().cpu().tolist())
                values[f"{prefix}_distinct_q"].append(float(torch.unique(qmaps[:5].flatten(1).argmax(-1)).numel() / 5))
                values[f"{prefix}_base_binary_iou"].extend(_pair_iou(bmaps >= .5).cpu().tolist())
                values[f"{prefix}_base_soft_iou"].extend(_pair_soft_iou(bmaps).cpu().tolist())
                for query in indices[:8].tolist():
                    a = route[image, query, cls]; b = base[image, query]
                    hi = a >= torch.quantile(a, .8); lo = a <= torch.quantile(a, .5)
                    contrast = b[hi].mean() - b[lo].mean()
                    values[f"{prefix}_owner_contrast"].append(float(contrast))
                    values[f"{prefix}_owner_corr"].append(_rank_corr(a, b))
                mass = route[image, :, cls].mean((-2, -1)); distribution = mass / mass.sum().clamp_min(1e-8)
                entropy = -(distribution * distribution.clamp_min(1e-8).log()).sum()
                values[f"{prefix}_routed_dominant"].append(float(distribution.max()))
                values[f"{prefix}_routed_effective"].append(float(entropy.exp()))

                pos = positive[image, cls]
                rival = positive[image, [c for c in range(4) if c != cls]].any(0)
                bg = background[image]; selected = final[image, cls] >= .5
                values[f"{prefix}_area"].append(float(selected.float().mean()))
                values[f"{prefix}_components"].append(_components(selected.cpu().numpy()))
                values[f"{prefix}_empty"].append(float(not bool(selected.any())))
                if bool(pos.any()):
                    pos_f = final[image, cls][pos]; pos_ref = reference[image, cls][pos]
                    values[f"{prefix}_positive_recall"].append(float(selected[pos].float().mean()))
                    values[f"{prefix}_positive_f"].extend(pos_f.cpu().tolist())
                    values[f"{prefix}_cpr"].append(float(pos_f.mean() / pos_ref.mean().clamp_min(1e-8)))
                    values[f"{prefix}_base_pos_logit"].extend(stage["base_mask_logits"][image, indices[:8]][:, pos].flatten().float().cpu().tolist())
                negative = rival | bg
                if bool(rival.any()): values[f"{prefix}_rival_leak"].append(float(selected[rival].float().mean()))
                if bool(bg.any()): values[f"{prefix}_background_leak"].append(float(selected[bg].float().mean()))
                if bool(negative.any()):
                    values[f"{prefix}_negative_f"].extend(final[image, cls][negative].cpu().tolist())
                    values[f"{prefix}_ncr"].append(float(final[image, cls][negative].mean() / reference[image, cls][negative].mean().clamp_min(1e-8)))
                    values[f"{prefix}_base_neg_logit"].extend(stage["base_mask_logits"][image, indices[:8]][:, negative].flatten().float().cpu().tolist())
    return values


def summarize(snapshot, batches, pmec_rows, model):
    base = cqrf_summarize(snapshot, batches, pmec_rows, model)
    merged = defaultdict(list)
    for batch in batches:
        for key, value in batch.items(): merged[key].extend(value)
    routing=[]; fallback=[]; capacity=[]; negative=[]; mask=[]; fragment=[]; contribution=[]; transfer=[]
    contrast=[]; correlation=[]; redundancy=[]; utilization=[]
    for stage in (2, 3):
        p=f"s{stage}"; q_iou=merged[f"{p}_q_iou"]; pos=_mean(merged[f"{p}_positive_f"]); neg=_mean(merged[f"{p}_negative_f"])
        routing.append({"snapshot":snapshot,"stage":stage,"max_A_sum_error":max(merged[f"{p}_routing_max"] or [0]),
                        "mean_A_sum_error":_mean(merged[f"{p}_routing_mean"]),"max_C_sum_error":max(merged[f"{p}_contribution_max"] or [0]),
                        "max_Q_sum_error":max(merged[f"{p}_posterior_max"] or [0]),"all_finite":bool(base["all_finite"])})
        fallback.append({"snapshot":snapshot,"stage":stage,"fallback_fraction":_mean(merged[f"{p}_fallback"])})
        capacity.append({"snapshot":snapshot,"stage":stage,"CPR":_median(merged[f"{p}_cpr"]),
                         "positive_seed_recall_median":_median(merged[f"{p}_positive_recall"]),
                         "positive_F_median":_median(merged[f"{p}_positive_f"]),"present_class_empty_fraction":_mean(merged[f"{p}_empty"])})
        negative.append({"snapshot":snapshot,"stage":stage,"NCR":_median(merged[f"{p}_ncr"]),
                         "rival_leakage":_mean(merged[f"{p}_rival_leak"]),"background_leakage":_mean(merged[f"{p}_background_leak"]),
                         "positive_probability":pos,"negative_probability":neg,"probability_gap":pos-neg,
                         "base_positive_logit":_mean(merged[f"{p}_base_pos_logit"]),"base_negative_logit":_mean(merged[f"{p}_base_neg_logit"]),
                         "base_logit_gap":_mean(merged[f"{p}_base_pos_logit"])-_mean(merged[f"{p}_base_neg_logit"])})
        areas=merged[f"{p}_area"]
        mask.append({"snapshot":snapshot,"stage":stage,"median_area":_median(areas),"mean_area":_mean(areas),
                     "fraction_area_lt_001":_mean([v<.01 for v in areas]),"fraction_area_gt_090":_mean([v>.90 for v in areas])})
        fragment.append({"snapshot":snapshot,"stage":stage,"component_mean":_mean(merged[f"{p}_components"]),
                         "component_median":_median(merged[f"{p}_components"]),"component_p90":_q(merged[f"{p}_components"],.9)})
        contribution.append({"snapshot":snapshot,"stage":stage,"posterior_support_iou_mean":_mean(q_iou),
                             "posterior_support_iou_median":_median(q_iou),"fraction_gt_075":_mean(merged[f"{p}_q_gt075"]),
                             "fraction_gt_090":_mean(merged[f"{p}_q_gt090"]),"distinct_contribution_peaks":_mean(merged[f"{p}_distinct_q"])})
        resp=next(x for x in base["responsibility_complementarity"] if x["stage"]==stage)
        transfer.append({"snapshot":snapshot,"stage":stage,"responsibility_iou":resp["responsibility_iou_median"],
                         "posterior_iou":_median(q_iou),"T_R_to_Q":(1-_median(q_iou))/(1-resp["responsibility_iou_median"]+1e-8)})
        contrast.append({"snapshot":snapshot,"stage":stage,"median":_median(merged[f"{p}_owner_contrast"]),"mean":_mean(merged[f"{p}_owner_contrast"])})
        correlation.append({"snapshot":snapshot,"stage":stage,"spearman_median":_median(merged[f"{p}_owner_corr"]),"spearman_mean":_mean(merged[f"{p}_owner_corr"])})
        redundancy.append({"snapshot":snapshot,"stage":stage,"binary_iou_median":_median(merged[f"{p}_base_binary_iou"]),
                           "binary_iou_gt090":_mean([x>.9 for x in merged[f"{p}_base_binary_iou"]]),
                           "soft_iou_median":_median(merged[f"{p}_base_soft_iou"])})
        utilization.append({"snapshot":snapshot,"stage":stage,"dominant_share":_median(merged[f"{p}_routed_dominant"]),
                            "effective_queries":_median(merged[f"{p}_routed_effective"])})
    return {**base,"routing_integrity":routing,"locality_fallback":fallback,"capacity_preservation":capacity,
            "negative_capacity":negative,"primary_mask_health":mask,"class_mask_area":mask,"fragmentation":fragment,
            "contribution_complementarity":contribution,"r_to_q_transfer":transfer,"expert_owner_contrast":contrast,
            "expert_ownership_correlation":correlation,"base_expert_redundancy":redundancy,"routed_expert_utilization":utilization,
            "historical_pmec_diagnostic":base["pmec_health"]}


def _row(summary, key, stage=3): return next(x for x in summary[key] if x["stage"]==stage)


def apply_epoch2_screen(summary):
    cap=_row(summary,"capacity_preservation"); neg=_row(summary,"negative_capacity"); routing=_row(summary,"routing_integrity")
    resp=_row(summary,"responsibility_complementarity"); contrib=_row(summary,"contribution_complementarity")
    checks={"engineering":not routing["all_finite"] or routing["max_A_sum_error"]>1e-5 or routing["max_C_sum_error"]>1e-6 or routing["max_Q_sum_error"]>1e-4,
            "capacity":cap["positive_seed_recall_median"]<.20 or cap["positive_F_median"]<.20 or cap["present_class_empty_fraction"]>.70,
            "specificity":neg["rival_leakage"]>.70 or neg["background_leakage"]>.50 or neg["probability_gap"]<=0,
            "ownership_lost":resp["responsibility_iou_median"]>.75 and contrib["distinct_contribution_peaks"]<.40,
            "fallback":_row(summary,"locality_fallback")["fallback_fraction"]>.80}
    failed=[k for k,v in checks.items() if v]
    return {"decision":"MOMD_PHASE0_NOGO" if failed else "CONTINUE_TO_EPOCH5","checks":checks,"failed_criteria":failed}


def apply_final_gate(history):
    s=history[-1]; r=_row(s,"responsibility_integrity"); rc=_row(s,"responsibility_complementarity")
    ru=_row(s,"responsibility_utilization"); mix=_row(s,"routing_integrity"); cap=_row(s,"capacity_preservation")
    neg=_row(s,"negative_capacity"); cc=_row(s,"contribution_complementarity"); tr=_row(s,"r_to_q_transfer")
    con=_row(s,"expert_owner_contrast"); area=_row(s,"primary_mask_health"); frag=_row(s,"fragmentation")
    pca=_row(s,"pca_health"); deep=s["deep_gate_health"]
    by={next(iter(x.get("routing_integrity",[])),{}).get("snapshot"):x for x in history}
    e3,e4,e5=(by.get(f"epoch{i}") for i in (3,4,5))
    late=all((e3,e4,e5)) and _row(e4,"contribution_complementarity")["posterior_support_iou_median"]<=_row(e3,"contribution_complementarity")["posterior_support_iou_median"]+.05 and cc["posterior_support_iou_median"]<=_row(e4,"contribution_complementarity")["posterior_support_iou_median"]+.05 and con["median"]>=_row(e3,"expert_owner_contrast")["median"]-.05 and cap["CPR"]>=_row(e3,"capacity_preservation")["CPR"]-.10
    groups={
      "A_CCRA_preserved":r["max_abs_sum_error"]<=1e-5 and rc["responsibility_iou_median"]<=.35 and rc["distinct_peak_fraction"]>=.80 and ru["dominant_share_median"]<.75 and ru["effective_queries_median"]>1.5,
      "B_mixture_conservation":mix["max_A_sum_error"]<=1e-5 and mix["max_C_sum_error"]<=1e-6 and mix["max_Q_sum_error"]<=1e-4 and _row(s,"locality_fallback")["fallback_fraction"]<=.50 and mix["all_finite"],
      "C_semantic_capacity":cap["CPR"]>=.70 and cap["positive_seed_recall_median"]>=.80 and cap["positive_F_median"]>=.60 and cap["present_class_empty_fraction"]<.10,
      "D_semantic_correctness":neg["rival_leakage"]<.30 and neg["background_leakage"]<.15 and neg["probability_gap"]>.25 and neg["base_logit_gap"]>5,
      "E_ownership_transfer":cc["posterior_support_iou_median"]<=.35 and cc["distinct_contribution_peaks"]>=.80 and tr["T_R_to_Q"]>=.70 and con["median"]>=.05,
      "F_no_late_homogenization":late,
      "G_output_health":.05<=area["median_area"]<=.80 and area["fraction_area_lt_001"]<.10 and area["fraction_area_gt_090"]<.20 and frag["component_p90"]<=15,
      "H_PCA_deep":pca["present_class_query_coverage"]>=.93 and pca["absent_class_dominance"]<=.20 and deep["present_absent_gap"]>.40 and s["all_finite"]}
    go=all(groups.values())
    strong=go and cap["CPR"]>=.90 and cap["positive_seed_recall_median"]>=.90 and cap["positive_F_median"]>=.75 and neg["rival_leakage"]<.20 and neg["background_leakage"]<.05 and neg["probability_gap"]>.40 and cc["posterior_support_iou_median"]<=.25 and cc["distinct_contribution_peaks"]>=.90 and tr["T_R_to_Q"]>=.85 and con["median"]>=.10 and pca["present_class_query_coverage"]>=.95 and pca["absent_class_dominance"]<=.10
    return {"decision":"MOMD_PHASE0_STRONG_GO" if strong else "MOMD_PHASE0_GO" if go else "MOMD_PHASE0_NOGO",
            "gate_groups":groups,"strong_go":strong}
