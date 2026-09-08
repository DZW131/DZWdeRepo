"""Preregistered FOMD factorization diagnostics and Phase-0 gates."""
from __future__ import annotations

from collections import defaultdict
import itertools
import math

import numpy as np
import torch
from torch.nn import functional as F

from tools.cqrf_diagnostics import _components
from tools.fomd_counterfactuals import materialize
from tools.momd_diagnostics import batch_health as momd_batch_health
from tools.momd_diagnostics import summarize as momd_summarize
from tools.momd_diagnostics import _mean, _median, _q, _pair_iou, _pair_soft_iou


def _semantic(values, prefix, mask, pos, rival, bg):
    selected = mask >= .5
    values[f"{prefix}_area"].append(float(selected.float().mean()))
    values[f"{prefix}_components"].append(_components(selected.cpu().numpy()))
    values[f"{prefix}_empty"].append(float(not bool(selected.any())))
    if bool(pos.any()):
        values[f"{prefix}_recall"].append(float(selected[pos].float().mean()))
        values[f"{prefix}_pos"].extend(mask[pos].cpu().tolist())
    if bool(rival.any()):
        values[f"{prefix}_rival_leak"].append(float(selected[rival].float().mean()))
        values[f"{prefix}_rival"].extend(mask[rival].cpu().tolist())
    if bool(bg.any()):
        values[f"{prefix}_bg_leak"].append(float(selected[bg].float().mean()))
        values[f"{prefix}_bg"].extend(mask[bg].cpu().tolist())
    negative = rival | bg
    if bool(negative.any()): values[f"{prefix}_negative"].extend(mask[negative].cpu().tolist())


def _js(a, b):
    m = .5 * (a + b)
    return .5 * ((a * (a.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(0) +
                 (b * (b.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(0))


@torch.no_grad()
def batch_health(output, labels, permutations):
    values = momd_batch_health(output, labels)
    positive = F.interpolate(output["target_detail"]["positive"].float(), (56, 56), mode="nearest").bool()
    background = F.interpolate(output["target_detail"]["reliable_background"][:, None].float(),
                               (56, 56), mode="nearest")[:, 0].bool()
    locality = output["locality"].bool()
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 56, device=labels.device),
                            torch.linspace(0, 1, 56, device=labels.device), indexing="ij")
    coords = torch.stack((yy, xx), 0)
    local_flat = locality.float().flatten(1)
    origins = (locality.float()[:, None] * coords[None]).flatten(2).sum(-1) / local_flat.sum(-1, keepdim=True).clamp_min(1)

    for stage_index in (2, 3):
        stage = output["stages"][stage_index - 1]; p = f"s{stage_index}"
        cf = materialize(stage, permutations)
        a = stage["momd"]["routing"].detach().float(); b = stage["momd"]["base_probability"].detach().float()
        values[f"{p}_detach"].append(float(all(not x.requires_grad for x in cf.values())))
        values[f"{p}_full_equal"].append(float(torch.equal(cf["full"], stage["momd"]["mixture"].detach().float())))
        values[f"{p}_d_perm"].extend((cf["perm"] - cf["full"][None]).abs().mean((1, 2, 3, 4)).cpu().tolist())
        values[f"{p}_d_pca"].append(float((cf["full"] - cf["pca"]).abs().mean()))

        for image in range(labels.shape[0]):
            present = torch.where(labels[image].bool())[0].tolist()
            for cls in present:
                pos = positive[image, cls]
                rival = positive[image, [c for c in range(4) if c != cls]].any(0)
                bg = background[image]
                for branch in ("full", "global", "pca", "uniform"):
                    _semantic(values, f"{p}_{branch}", cf[branch][image, cls], pos, rival, bg)
                for perm_index in range(len(permutations)):
                    _semantic(values, f"{p}_perm{perm_index}", cf["perm"][perm_index, image, cls], pos, rival, bg)

                delta_global = (cf["full"][image, cls] - cf["global"][image, cls]).abs()
                delta_pca = (cf["full"][image, cls] - cf["pca"][image, cls]).abs()
                for region, region_mask in (("all", torch.ones_like(pos)), ("positive", pos),
                                            ("rival", rival), ("background", bg)):
                    if bool(region_mask.any()):
                        values[f"{p}_d_global_{region}"].append(float(delta_global[region_mask].mean()))
                        values[f"{p}_d_pca_{region}"].append(float(delta_pca[region_mask].mean()))

            if len(present) >= 2:
                eligible = positive[image, present].any(0)
                if bool(eligible.any()):
                    for c1, c2 in itertools.combinations(present, 2):
                        aa, ab = a[image, :, c1][:, eligible], a[image, :, c2][:, eligible]
                        values[f"{p}_A_js"].extend(_js(aa, ab).cpu().tolist())
                        order_a = torch.argsort(aa, dim=0, descending=True, stable=True)
                        order_b = torch.argsort(ab, dim=0, descending=True, stable=True)
                        values[f"{p}_A_top1_diff"].extend((order_a[0] != order_b[0]).float().cpu().tolist())
                        for k in (3, 5):
                            left, right = order_a[:k].T, order_b[:k].T
                            inter = (left[:, :, None] == right[:, None, :]).any(-1).sum(-1).float()
                            values[f"{p}_A_top{k}_jaccard"].extend((inter / (2 * k - inter).clamp_min(1)).cpu().tolist())

            # Evaluate B on the union of PCA top-20 queries for all present classes.
            ranked = []
            for cls in present:
                ranked.extend(torch.argsort(stage["confidence"]["joint"][image, :, cls], descending=True,
                                            stable=True)[:20].tolist())
            ranked = sorted(set(ranked))
            if ranked:
                maps = b[image, ranked]; binary = maps >= .5
                union_pos = positive[image, present].any(0); bg = background[image]
                bmax, bmean = maps.max(0).values, maps.mean(0)
                if bool(union_pos.any()):
                    values[f"{p}_Bmax_pos"].extend(bmax[union_pos].cpu().tolist())
                    values[f"{p}_Bmean_pos"].extend(bmean[union_pos].cpu().tolist())
                if bool(bg.any()):
                    values[f"{p}_Bmax_bg"].extend(bmax[bg].cpu().tolist())
                    values[f"{p}_Bmean_bg"].extend(bmean[bg].cpu().tolist())
                values[f"{p}_B_empty"].extend((binary.flatten(1).sum(1) == 0).float().cpu().tolist())
                area = binary.flatten(1).float().mean(1)
                values[f"{p}_B_area"].extend(area.cpu().tolist())
                values[f"{p}_B_components"].extend([_components(x.cpu().numpy()) for x in binary])
                if len(ranked) > 1:
                    values[f"{p}_B_binary_iou"].extend(_pair_iou(binary).cpu().tolist())
                    values[f"{p}_B_soft_iou"].extend(_pair_soft_iou(maps).cpu().tolist())
                ids = torch.as_tensor(ranked, device=labels.device)
                mass = maps.flatten(1).sum(1).clamp_min(1e-8)
                local_mass = (maps * locality[ids]).flatten(1).sum(1) / mass
                values[f"{p}_B_locality"].extend(local_mass.cpu().tolist())
                centroid = (maps[:, None] * coords[None]).flatten(2).sum(-1) / mass[:, None]
                dist = (centroid - origins[ids]).square().sum(-1).sqrt()
                values[f"{p}_B_centroid"].extend(dist.cpu().tolist())
    return values


def _semantic_row(snapshot, stage, merged, prefix):
    pos = merged[f"{prefix}_pos"]; rival = merged[f"{prefix}_rival"]
    bg = merged[f"{prefix}_bg"]; negative = merged[f"{prefix}_negative"]
    pos_mean, neg_mean = _mean(pos), _mean(negative)
    return {"snapshot": snapshot, "stage": stage,
            "positive_recall": _median(merged[f"{prefix}_recall"]),
            "positive_confidence": pos_mean, "positive_F_median": _median(pos),
            "rival_leakage": _mean(merged[f"{prefix}_rival_leak"]),
            "background_leakage": _mean(merged[f"{prefix}_bg_leak"]),
            "rival_probability": _mean(rival), "background_probability": _mean(bg),
            "negative_probability": neg_mean, "probability_gap": pos_mean - neg_mean,
            "rival_gap": pos_mean - _mean(rival), "background_gap": pos_mean - _mean(bg),
            "empty_fraction": _mean(merged[f"{prefix}_empty"]),
            "median_area": _median(merged[f"{prefix}_area"]),
            "component_p90": _q(merged[f"{prefix}_components"], .9)}


def summarize(snapshot, batches, pmec_rows, model, permutation_count=8):
    base = momd_summarize(snapshot, batches, pmec_rows, model)
    merged = defaultdict(list)
    for batch in batches:
        for key, value in batch.items(): merged[key].extend(value)
    full=[]; global_rows=[]; pca=[]; uniform=[]; perm=[]; query=[]; spatial=[]; pca_sens=[]
    synergy=[]; acond=[]; atop=[]; bsupport=[]; bloc=[]; bcent=[]; barea=[]; bfrag=[]; bred=[]; detach=[]
    for stage in (2, 3):
        p=f"s{stage}"
        frow=_semantic_row(snapshot,stage,merged,f"{p}_full"); grow=_semantic_row(snapshot,stage,merged,f"{p}_global")
        prow=_semantic_row(snapshot,stage,merged,f"{p}_pca"); urow=_semantic_row(snapshot,stage,merged,f"{p}_uniform")
        prows=[_semantic_row(snapshot,stage,merged,f"{p}_perm{i}") | {"permutation":i} for i in range(permutation_count)]
        full.append(frow); global_rows.append(grow); pca.append(prow); uniform.append(urow); perm.extend(prows)
        pmean={k:_mean([r[k] for r in prows]) for k in ("positive_recall","positive_confidence","rival_leakage","background_leakage","probability_gap","component_p90")}
        d=merged[f"{p}_d_perm"]
        query.append({"snapshot":snapshot,"stage":stage,"D_perm_mean":_mean(d),"D_perm_std":float(np.std(d)) if d else 0.0,"D_perm_min":min(d or [0]),"D_perm_max":max(d or [0])})
        spatial.append({"snapshot":snapshot,"stage":stage,**{f"D_global_{r}":_mean(merged[f"{p}_d_global_{r}"]) for r in ("all","positive","rival","background")}})
        pca_sens.append({"snapshot":snapshot,"stage":stage,**{f"D_pca_{r}":_mean(merged[f"{p}_d_pca_{r}"]) for r in ("all","positive","rival","background")}})
        synergy.append({"snapshot":snapshot,"stage":stage,**{f"full_minus_perm_{k}":frow[k]-pmean[k] for k in pmean},
                        **{f"full_minus_global_{k}":frow[k]-grow[k] for k in pmean},
                        **{f"full_minus_pca_{k}":frow[k]-prow[k] for k in pmean}})
        acond.append({"snapshot":snapshot,"stage":stage,"JS_median":_median(merged[f"{p}_A_js"]),"JS_mean":_mean(merged[f"{p}_A_js"])})
        atop.append({"snapshot":snapshot,"stage":stage,"top_owner_difference":_mean(merged[f"{p}_A_top1_diff"]),"top3_jaccard":_mean(merged[f"{p}_A_top3_jaccard"]),"top5_jaccard":_mean(merged[f"{p}_A_top5_jaccard"])})
        bmaxp,bmaxb=_mean(merged[f"{p}_Bmax_pos"]),_mean(merged[f"{p}_Bmax_bg"])
        bsupport.append({"snapshot":snapshot,"stage":stage,"Bmax_union_positive":bmaxp,"Bmax_reliable_background":bmaxb,"Bmax_gap":bmaxp-bmaxb,"Bmean_union_positive":_mean(merged[f"{p}_Bmean_pos"]),"Bmean_reliable_background":_mean(merged[f"{p}_Bmean_bg"])})
        bloc.append({"snapshot":snapshot,"stage":stage,"locality_mass_median":_median(merged[f"{p}_B_locality"]),"locality_mass_mean":_mean(merged[f"{p}_B_locality"])})
        bcent.append({"snapshot":snapshot,"stage":stage,"centroid_distance_median":_median(merged[f"{p}_B_centroid"]),"centroid_distance_p90":_q(merged[f"{p}_B_centroid"],.9)})
        areas=merged[f"{p}_B_area"]
        barea.append({"snapshot":snapshot,"stage":stage,"empty_fraction":_mean(merged[f"{p}_B_empty"]),"median_area":_median(areas),"fraction_area_gt_090":_mean([x>.9 for x in areas])})
        bfrag.append({"snapshot":snapshot,"stage":stage,"component_p90":_q(merged[f"{p}_B_components"],.9),"component_median":_median(merged[f"{p}_B_components"])})
        bred.append({"snapshot":snapshot,"stage":stage,"binary_iou_median":_median(merged[f"{p}_B_binary_iou"]),"soft_iou_median":_median(merged[f"{p}_B_soft_iou"])})
        detach.append({"snapshot":snapshot,"stage":stage,"all_counterfactuals_detached":bool(min(merged[f"{p}_detach"] or [0])),"F_full_equals_MOMD":bool(min(merged[f"{p}_full_equal"] or [0]))})
    return {**base,"full_semantic_health":full,"perm_semantic_health":perm,"global_semantic_health":global_rows,
            "pca_semantic_health":pca,"uniform_reference":uniform,"query_identity_sensitivity":query,
            "spatial_ownership_sensitivity":spatial,"pca_sensitivity":pca_sens,"factorization_synergy":synergy,
            "A_class_conditioning":acond,"A_top_owner_disagreement":atop,"B_union_foreground_support":bsupport,
            "B_locality_mass":bloc,"B_centroid_consistency":bcent,"B_area_health":barea,
            "B_fragmentation":bfrag,"B_redundancy":bred,"counterfactual_detach_audit":detach,
            "ccra_health":base["responsibility_integrity"],"conservation":base["routing_integrity"]}


def _row(summary,key,stage=3): return next(x for x in summary[key] if x["stage"]==stage)


def apply_epoch2_screen(s):
    m=_row(s,"routing_integrity"); cap=_row(s,"capacity_preservation"); neg=_row(s,"full_semantic_health")
    rc=_row(s,"responsibility_complementarity"); detach=_row(s,"counterfactual_detach_audit")
    checks={"engineering":not s["all_finite"] or m["max_A_sum_error"]>1e-5 or m["max_C_sum_error"]>1e-6 or m["max_Q_sum_error"]>1e-4 or not detach["all_counterfactuals_detached"] or not detach["F_full_equals_MOMD"],
            "capacity":cap["CPR"]<.40 or cap["positive_seed_recall_median"]<.50 or cap["positive_F_median"]<.40 or cap["present_class_empty_fraction"]>.40,
            "specificity":neg["rival_leakage"]>.70 or neg["background_leakage"]>.50 or neg["probability_gap"]<=0,
            "allocator":rc["responsibility_iou_median"]>.75 and rc["distinct_peak_fraction"]<.40}
    failed=[k for k,v in checks.items() if v]
    return {"decision":"FOMD_PHASE0_NOGO" if failed else "CONTINUE_TO_E5_UNCHANGED","checks":checks,"failed_criteria":failed}


def apply_final_gate(history):
    s=history[-1]; r=_row(s,"responsibility_integrity"); rc=_row(s,"responsibility_complementarity"); ru=_row(s,"responsibility_utilization")
    m=_row(s,"routing_integrity"); cap=_row(s,"capacity_preservation"); full=_row(s,"full_semantic_health")
    q=_row(s,"query_identity_sensitivity"); g=_row(s,"global_semantic_health"); pca_sem=_row(s,"pca_semantic_health")
    b=_row(s,"B_area_health"); bs=_row(s,"B_union_foreground_support"); bl=_row(s,"B_locality_mass"); bf=_row(s,"B_fragmentation")
    ac=_row(s,"A_class_conditioning"); at=_row(s,"A_top_owner_disagreement"); spatial=_row(s,"spatial_ownership_sensitivity")
    pca=_row(s,"pca_health"); deep=s["deep_gate_health"]
    perms=[x for x in s["perm_semantic_health"] if x["stage"]==3]
    pm={k:_mean([x[k] for x in perms]) for k in ("probability_gap","rival_leakage","background_leakage","positive_confidence","positive_recall")}
    perm_checks=[full["probability_gap"]>=pm["probability_gap"]+.03,full["rival_leakage"]<=pm["rival_leakage"]-.02,full["background_leakage"]<=pm["background_leakage"]-.005,full["positive_confidence"]>=pm["positive_confidence"]+.02,full["positive_recall"]>=pm["positive_recall"]+.02]
    global_checks=[full["probability_gap"]>=g["probability_gap"]+.03,full["rival_leakage"]<=g["rival_leakage"]-.02,full["background_leakage"]<=g["background_leakage"]-.005,full["positive_confidence"]>=g["positive_confidence"]+.02,full["positive_recall"]>=g["positive_recall"]+.02]
    pca_checks=[full["probability_gap"]>=pca_sem["probability_gap"]+.03,full["rival_leakage"]<=pca_sem["rival_leakage"]-.02,full["background_leakage"]<=pca_sem["background_leakage"]-.005,full["positive_confidence"]>=pca_sem["positive_confidence"]+.02,full["positive_recall"]>=pca_sem["positive_recall"]+.02,full["component_p90"]<=pca_sem["component_p90"]]
    by={_row(x,"query_identity_sensitivity")["snapshot"]:x for x in history}; e3=by.get("epoch3")
    e3perm=None
    if e3:
        e3full=_row(e3,"full_semantic_health"); e3ps=[x for x in e3["perm_semantic_health"] if x["stage"]==3]
        e3perm=e3full["probability_gap"]-_mean([x["probability_gap"] for x in e3ps])
    late=bool(e3) and q["D_perm_mean"]>=_row(e3,"query_identity_sensitivity")["D_perm_mean"]-.005 and full["probability_gap"]-pm["probability_gap"]>=e3perm-.03 and cap["CPR"]>=_row(e3,"capacity_preservation")["CPR"]-.10
    area=_row(s,"primary_mask_health"); frag=_row(s,"fragmentation")
    groups={"A_CCRA_preserved":r["max_abs_sum_error"]<=1e-5 and rc["responsibility_iou_median"]<=.35 and rc["distinct_peak_fraction"]>=.80 and ru["dominant_share_median"]<.75 and ru["effective_queries_median"]>1.5,
            "B_conservation":m["max_A_sum_error"]<=1e-5 and m["max_C_sum_error"]<=1e-6 and m["max_Q_sum_error"]<=1e-4 and _row(s,"locality_fallback")["fallback_fraction"]<=.50 and m["all_finite"],
            "C_final_mask_healthy":cap["CPR"]>=.80 and full["positive_recall"]>=.90 and full["positive_F_median"]>=.70 and full["empty_fraction"]<.05 and full["rival_leakage"]<.30 and full["background_leakage"]<.10 and full["probability_gap"]>.40 and .05<=area["median_area"]<=.80 and area["fraction_area_lt_001"]<.10 and area["fraction_area_gt_090"]<.20 and frag["component_p90"]<=15,
            "D_query_identity":q["D_perm_mean"]>=.01 and sum(perm_checks)>=2,
            "E_pixelwise_ownership":spatial["D_global_all"]>=.01 and any(global_checks),
            "F_beats_PCA":sum(pca_checks)>=2 and cap["CPR"]>=.80,
            "G_B_region_support":b["empty_fraction"]<.70 and b["fraction_area_gt_090"]<.30 and bf["component_p90"]<=15 and bl["locality_mass_median"]>=.50 and bs["Bmax_union_positive"]>bs["Bmax_reliable_background"],
            "H_A_class_conditioned":ac["JS_median"]>0 and at["top_owner_difference"]>=.10,
            "I_late_stability":late,
            "J_PCA_deep":pca["present_class_query_coverage"]>=.93 and pca["absent_class_dominance"]<=.20 and deep["present_absent_gap"]>.40 and s["all_finite"]}
    go=all(groups.values())
    strong=go and cap["CPR"]>=1 and full["positive_recall"]>=.95 and full["positive_F_median"]>=.75 and full["rival_leakage"]<.20 and full["background_leakage"]<.05 and full["probability_gap"]>.55 and q["D_perm_mean"]>=.03 and (full["probability_gap"]>=pm["probability_gap"]+.08 or full["rival_leakage"]<=pm["rival_leakage"]-.05) and full["probability_gap"]>=g["probability_gap"]+.05 and full["probability_gap"]>=pca_sem["probability_gap"]+.05 and sum(perm_checks)>=3 and bl["locality_mass_median"]>=.65 and at["top_owner_difference"]>=.20 and late
    return {"decision":"FOMD_PHASE0_STRONG_GO" if strong else "FOMD_PHASE0_GO" if go else "FOMD_PHASE0_NOGO","gate_groups":groups,"strong_go":strong,"permutation_semantic_checks":perm_checks,"global_semantic_checks":global_checks,"pca_semantic_checks":pca_checks}


__all__=["batch_health","summarize","apply_epoch2_screen","apply_final_gate"]
