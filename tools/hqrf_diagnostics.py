"""Frozen-train-cohort diagnostics and Phase-0 gate evaluation."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: 0.0 for key in ("mean", "median", "p10", "p25", "p75", "p90")}
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p10": float(np.quantile(array, .10)), "p25": float(np.quantile(array, .25)),
        "p75": float(np.quantile(array, .75)), "p90": float(np.quantile(array, .90)),
    }


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def batch_health(output, labels):
    probability = output["mask_logits"].float().sigmoid()
    logits = output["mask_logits"].float()
    joint = output["confidence"]["joint"].float()
    p_class = output["confidence"]["p_class"].float()
    p_patch = output["confidence"]["p_patch"].float()
    locality = output["locality"]
    embedding = F.normalize(output["mask_embedding"].float(), dim=-1)
    assigned = p_class.argmax(-1)
    top_scores = joint.max(-1).values
    top20 = torch.argsort(top_scores, dim=1, descending=True, stable=True)[:, :20]
    target_hw = logits.shape[-2:]
    positive = F.interpolate(output["target_detail"]["positive"].float(), target_hw, mode="nearest").bool()
    background = F.interpolate(output["target_detail"]["reliable_background"][:, None].float(), target_hw, mode="nearest")[:, 0].bool()

    values = defaultdict(list)
    image_rows = []
    for image in range(logits.shape[0]):
        indices = top20[image]
        raw_binary = probability[image, indices] > .5
        local = locality[indices]
        local_binary = raw_binary & local
        global_area = raw_binary.flatten(1).float().mean(1)
        local_area = local_binary.flatten(1).sum(1).float() / local.flatten(1).sum(1).clamp_min(1)
        selected_logits = logits[image, indices]
        values["global_area"].extend(global_area.cpu().tolist())
        values["local_area"].extend(local_area.cpu().tolist())
        values["logit"].extend(selected_logits.flatten().cpu().tolist())
        intersections = (raw_binary[:, None] & raw_binary[None]).flatten(2).sum(-1).float()
        unions = (raw_binary[:, None] | raw_binary[None]).flatten(2).sum(-1).float().clamp_min(1)
        upper = torch.triu(torch.ones((20, 20), dtype=torch.bool, device=logits.device), diagonal=1)
        pair_iou = (intersections / unions)[upper]
        cosine = (embedding[image, indices] @ embedding[image, indices].T)[upper]
        values["pair_iou"].extend(pair_iou.cpu().tolist())
        values["embedding_cosine"].extend(cosine.cpu().tolist())

        rival_rates, background_rates, positive_logits, negative_logits = [], [], [], []
        for cls in torch.where(labels[image].bool())[0].tolist():
            ranked = torch.argsort(joint[image, :, cls], descending=True, stable=True)[:8]
            rival = positive[image, [other for other in range(4) if other != cls]].any(0)
            pos = positive[image, cls]
            bg = background[image]
            for query in ranked.tolist():
                support = locality[query]
                selected = probability[image, query] > .5
                if bool((pos & support).any()):
                    values["positive_recall"].append(float((selected & pos & support).sum() / (pos & support).sum()))
                    positive_logits.extend(logits[image, query][pos & support].cpu().tolist())
                if bool((rival & support).any()):
                    rate = float((selected & rival & support).sum() / (rival & support).sum())
                    values["rival_leakage"].append(rate); rival_rates.append(rate)
                    negative_logits.extend(logits[image, query][rival & support].cpu().tolist())
                if bool((bg & support).any()):
                    rate = float((selected & bg & support).sum() / (bg & support).sum())
                    values["background_leakage"].append(rate); background_rates.append(rate)
                    negative_logits.extend(logits[image, query][bg & support].cpu().tolist())
        values["positive_logit"].extend(positive_logits)
        values["negative_logit"].extend(negative_logits)

        present_pairs = int(labels[image].sum())
        present_with_query = sum(bool((assigned[image] == cls).any()) for cls in torch.where(labels[image].bool())[0].tolist())
        top_mass = top_scores[image, indices]
        absent_top = ~labels[image, assigned[image, indices]].bool()
        absent_dominance = float(top_mass[absent_top].sum() > .5 * top_mass.sum())
        values["present_pairs"].append(present_pairs)
        values["present_pairs_with_query"].append(present_with_query)
        values["absent_dominance"].append(absent_dominance)
        values["assigned_absent_fraction"].append(float((~labels[image, assigned[image]].bool()).float().mean()))

        pmec_rows = [row for row in output["pmec_rows"] if row["image"] == image]
        pmec_full = _safe_mean([row["selected_region_area"] > .90 for row in pmec_rows]) > .5
        pmec_candidates = [row["candidate_masks"] >= 1 for row in pmec_rows]
        pmec_diff = [row["differs_from_top1"] for row in pmec_rows]
        values["pmec_pairs"].extend([1] * len(pmec_rows))
        values["pmec_candidate"].extend(pmec_candidates)
        values["pmec_diff"].extend(pmec_diff)
        image_rows.append({
            "median_local_area_gt_090": float(float(local_area.median()) > .90),
            "fraction_local_area_gt_090_gt_080": float(float((local_area > .90).float().mean()) > .80),
            "rival_leakage_gt_090": float(_safe_mean(rival_rates) > .90),
            "background_leakage_gt_090": float(_safe_mean(background_rates) > .90),
            "mean_pair_iou_gt_090": float(float(pair_iou.mean()) > .90),
            "fraction_logits_positive_gt_095": float(float((selected_logits > 0).float().mean()) > .95),
            "no_valid_class_selective_queries": float(present_with_query == 0),
            "pmec_almost_full": float(pmec_full),
        })

    class_entropy = -(p_class * p_class.clamp_min(1e-8).log()).sum(-1)
    patch_entropy = -(p_patch * p_patch.clamp_min(1e-8).log()).sum(1)
    sorted_class = p_class.sort(-1, descending=True).values
    values["class_entropy"].extend(class_entropy.flatten().cpu().tolist())
    values["patch_entropy"].extend(patch_entropy.flatten().cpu().tolist())
    values["class_margin"].extend((sorted_class[..., 0] - sorted_class[..., 1]).flatten().cpu().tolist())
    values["top1_confidence"].extend(top_scores.max(1).values.cpu().tolist())
    values["top5_cumulative"].extend(top_scores.topk(5, dim=1).values.sum(1).cpu().tolist())
    for prefix, raw, context in (
        ("F5", output["query_detail"]["context_raw"], output["query_detail"]["context_feature"]),
        ("F4", output["pixel_detail"]["F4_raw"], output["pixel_detail"]["F4_context"]),
    ):
        values[f"{prefix}_residual_norm"].append(float((context.float()-raw.float()).square().mean().sqrt()))
        values[f"{prefix}_raw_context_cosine"].append(float(F.cosine_similarity(raw.float(),context.float(),dim=1).mean()))
    return values, image_rows


def summarize(snapshot, batches, image_rows, pmec_rows, model):
    merged = defaultdict(list)
    for batch in batches:
        for key, values in batch.items(): merged[key].extend(values)
    area_global, area_local, logit = _stats(merged["global_area"]), _stats(merged["local_area"]), _stats(merged["logit"])
    pair, cosine = _stats(merged["pair_iou"]), _stats(merged["embedding_cosine"])
    positive_logit = _safe_mean(merged["positive_logit"])
    negative_logit = _safe_mean(merged["negative_logit"])
    mask_area = {"snapshot": snapshot, **{f"global_{k}": v for k,v in area_global.items()}, **{f"local_{k}": v for k,v in area_local.items()},
                 "global_fraction_gt_090": _safe_mean([v > .90 for v in merged["global_area"]]), "global_fraction_gt_075": _safe_mean([v > .75 for v in merged["global_area"]]), "global_fraction_lt_005": _safe_mean([v < .05 for v in merged["global_area"]]),
                 "local_fraction_gt_090": _safe_mean([v > .90 for v in merged["local_area"]]), "local_fraction_gt_075": _safe_mean([v > .75 for v in merged["local_area"]]), "local_fraction_lt_005": _safe_mean([v < .05 for v in merged["local_area"]])}
    mask_logits = {"snapshot": snapshot, **logit, "std": float(np.std(merged["logit"])), "fraction_gt_zero": _safe_mean([v > 0 for v in merged["logit"]]), "fraction_sigmoid_gt_050": _safe_mean([v > 0 for v in merged["logit"]])}
    diversity = {"snapshot": snapshot, "pair_iou_mean": pair["mean"], "pair_iou_median": pair["median"], "pair_iou_fraction_gt_090": _safe_mean([v > .90 for v in merged["pair_iou"]]), "embedding_cosine_mean": cosine["mean"], "embedding_cosine_p90": cosine["p90"]}
    selectivity = {"snapshot": snapshot, "positive_seed_recall": _safe_mean(merged["positive_recall"]), "rival_seed_leakage": _safe_mean(merged["rival_leakage"]), "background_leakage": _safe_mean(merged["background_leakage"]), "mean_positive_logit": positive_logit, "mean_negative_logit": negative_logit, "positive_negative_logit_gap": positive_logit-negative_logit}
    pca = {"snapshot": snapshot, "class_entropy": _safe_mean(merged["class_entropy"]), "query_entropy": _safe_mean(merged["patch_entropy"]), "top1_confidence": _safe_mean(merged["top1_confidence"]), "top5_cumulative_confidence": _safe_mean(merged["top5_cumulative"]), "top1_top2_class_margin": _safe_mean(merged["class_margin"]), "present_pair_query_coverage": sum(merged["present_pairs_with_query"])/max(sum(merged["present_pairs"]),1), "assigned_absent_fraction": _safe_mean(merged["assigned_absent_fraction"]), "absent_confidence_dominance_image_fraction": _safe_mean(merged["absent_dominance"]), "all_finite": True}
    pmec_summary = {"snapshot": snapshot, "present_class_pairs": len(pmec_rows), "candidate_pair_fraction": _safe_mean([r["candidate_masks"] >= 1 for r in pmec_rows]), "differs_from_top1_fraction": _safe_mean([r["differs_from_top1"] for r in pmec_rows]), "candidate_masks_per_pair": _safe_mean([r["candidate_masks"] for r in pmec_rows]), "selected_masks_per_pair": _safe_mean([r["selected_masks"] for r in pmec_rows]), "region_groups_per_pair": _safe_mean([r["region_groups"] for r in pmec_rows]), "selected_region_area": _safe_mean([r["selected_region_area"] for r in pmec_rows]), "redundancy_rejections_per_pair": _safe_mean([r["redundancy_rejections"] for r in pmec_rows]), "new_references_per_pair": _safe_mean([r["new_references"] for r in pmec_rows])}
    chpf = {"snapshot": snapshot, "gamma5": model.f5_chpf.gamma.detach().item(), "gamma4": model.pixel_decoder.f4_chpf.gamma.detach().item(),
            "F5_context_residual_norm": _safe_mean(merged["F5_residual_norm"]), "F4_context_residual_norm": _safe_mean(merged["F4_residual_norm"]),
            "F5_raw_context_cosine": _safe_mean(merged["F5_raw_context_cosine"]), "F4_raw_context_cosine": _safe_mean(merged["F4_raw_context_cosine"])}
    catastrophic = {key: _safe_mean([row[key] for row in image_rows]) for key in image_rows[0]}
    catastrophic["snapshot"] = snapshot
    return {"mask_area": mask_area, "mask_logits": mask_logits, "query_diversity": diversity, "semantic_selectivity": selectivity, "pca_health": pca, "pmec_health": pmec_summary, "chpf": chpf, "catastrophic": catastrophic}


def apply_epoch2_screen(summary):
    fractions = {key:value for key,value in summary["catastrophic"].items() if key != "snapshot"}
    failed = [key for key,value in fractions.items() if value > .80]
    return {"decision": "HQRF_QUERY_REGION_NOGO" if failed else "CONTINUE_EPOCH3", "failed_criteria": failed, "fractions": fractions}


def apply_final_gate(summary):
    area=summary["mask_area"]; diversity=summary["query_diversity"]; selectivity=summary["semantic_selectivity"]; pca=summary["pca_health"]; pmec=summary["pmec_health"]
    checks={
        "A_fraction_gt90": area["local_fraction_gt_090"] < .40,
        "A_median_area": area["local_median"] < .75,
        "B_mean_iou": diversity["pair_iou_mean"] < .75,
        "B_high_iou": diversity["pair_iou_fraction_gt_090"] < .50,
        "C_rival": selectivity["rival_seed_leakage"] < .70,
        "C_background": selectivity["background_leakage"] < .70,
        "C_one_below50": min(selectivity["rival_seed_leakage"],selectivity["background_leakage"]) < .50,
        "D_logit_gap": selectivity["positive_negative_logit_gap"] > .50,
        "E_present_queries": pca["present_pair_query_coverage"] > .95,
        "E_absent_not_dominant": pca["absent_confidence_dominance_image_fraction"] < .05,
        "E_finite": bool(pca["all_finite"]),
        "F_candidates": pmec["candidate_pair_fraction"] > .80,
        "F_non_top1": pmec["differs_from_top1_fraction"] > .50,
    }
    go=all(checks.values())
    strong=go and area["local_median"] < .60 and selectivity["rival_seed_leakage"] < .50 and selectivity["background_leakage"] < .50 and diversity["pair_iou_mean"] < .60 and selectivity["positive_negative_logit_gap"] > 1.0
    decision="HQRF_PHASE0_STRONG_GO" if strong else "HQRF_PHASE0_GO" if go else "HQRF_QUERY_REGION_NOGO"
    return {"decision":decision,"checks":checks,"strong_go":strong}
