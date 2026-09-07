"""Train-cohort diagnostics and preregistered FQRF Phase-0 gates."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _stats(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {key: 0.0 for key in ("mean", "median", "p90")}
    return {"mean": float(array.mean()), "median": float(np.median(array)), "p90": float(np.quantile(array, .90))}


def _upper(count, device):
    return torch.triu(torch.ones((count, count), dtype=torch.bool, device=device), diagonal=1)


def _pair_iou(binary):
    intersections = (binary[:, None] & binary[None]).flatten(2).sum(-1).float()
    unions = (binary[:, None] | binary[None]).flatten(2).sum(-1).float().clamp_min(1)
    return (intersections / unions)[_upper(binary.shape[0], binary.device)]


def _pair_cosine(value):
    value = F.normalize(value.float(), dim=-1)
    return (value @ value.T)[_upper(value.shape[0], value.device)]


def _stage_top20(stage):
    score = stage["confidence"]["joint"].float().max(-1).values
    return torch.argsort(score, dim=1, descending=True, stable=True)[:, :20]


@torch.no_grad()
def batch_health(output, labels):
    locality = output["locality"]
    values = defaultdict(list)
    all_finite = True
    for stage_index, stage in enumerate(output["stages"], start=1):
        prefix = f"s{stage_index}"
        logits = stage["mask_logits"].float()
        probability = logits.sigmoid()
        embedding = stage["mask_embedding"].float()
        attention = stage["attention"]["cross_attention"].float()
        top20 = _stage_top20(stage)
        all_finite &= bool(torch.isfinite(logits).all() and torch.isfinite(attention).all())
        height, width = stage["memory_hw"]
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, height, device=logits.device),
            torch.linspace(0, 1, width, device=logits.device), indexing="ij",
        )
        coordinates = torch.stack((yy.flatten(), xx.flatten()), dim=-1)
        for image in range(logits.shape[0]):
            indices = top20[image]
            binary = probability[image, indices] > .5
            local = locality[indices]
            local_binary = binary & local
            global_area = binary.flatten(1).float().mean(1)
            local_area = local_binary.flatten(1).sum(1).float() / local.flatten(1).sum(1).clamp_min(1)
            pair_iou = _pair_iou(binary)
            pair_cosine = _pair_cosine(embedding[image, indices])
            values[f"{prefix}_global_area"].extend(global_area.cpu().tolist())
            values[f"{prefix}_local_area"].extend(local_area.cpu().tolist())
            values[f"{prefix}_pair_iou"].extend(pair_iou.cpu().tolist())
            values[f"{prefix}_embedding_cosine"].extend(pair_cosine.cpu().tolist())

            focused = attention[image, indices]
            focused = focused / focused.sum(-1, keepdim=True).clamp_min(1e-8)
            entropy = -(focused * focused.clamp_min(1e-8).log()).sum(-1)
            attention_cosine = _pair_cosine(focused)
            binary_attention = focused >= focused.mean(-1, keepdim=True)
            attention_iou = _pair_iou(binary_attention)
            centroids = focused @ coordinates
            centroid_distance = torch.cdist(centroids, centroids)[_upper(20, logits.device)]
            values[f"{prefix}_attention_entropy"].extend(entropy.cpu().tolist())
            values[f"{prefix}_attention_cosine"].extend(attention_cosine.cpu().tolist())
            values[f"{prefix}_attention_iou"].extend(attention_iou.cpu().tolist())
            values[f"{prefix}_centroid_distance"].extend(centroid_distance.cpu().tolist())

        confidence = stage["confidence"]
        p_class, p_query = confidence["p_class"].float(), confidence["p_patch"].float()
        assigned = p_class.argmax(-1)
        top_scores = confidence["joint"].float().max(-1).values
        present_pairs = int(labels.sum())
        present_with_query = 0
        absent_dominance = []
        assigned_absent = []
        for image in range(labels.shape[0]):
            present_with_query += sum(bool((assigned[image] == cls).any()) for cls in torch.where(labels[image].bool())[0].tolist())
            indices = top20[image]
            mass = top_scores[image, indices]
            absent = ~labels[image, assigned[image, indices]].bool()
            absent_dominance.append(float(mass[absent].sum() > .5 * mass.sum()))
            assigned_absent.append(float((~labels[image, assigned[image]].bool()).float().mean()))
        class_entropy = -(p_class * p_class.clamp_min(1e-8).log()).sum(-1)
        query_entropy = -(p_query * p_query.clamp_min(1e-8).log()).sum(1)
        sorted_class = p_class.sort(-1, descending=True).values
        values[f"{prefix}_present_pairs"].append(present_pairs)
        values[f"{prefix}_present_with_query"].append(present_with_query)
        values[f"{prefix}_absent_dominance"].extend(absent_dominance)
        values[f"{prefix}_assigned_absent"].extend(assigned_absent)
        values[f"{prefix}_class_entropy"].extend(class_entropy.flatten().cpu().tolist())
        values[f"{prefix}_query_entropy"].extend(query_entropy.flatten().cpu().tolist())
        values[f"{prefix}_class_margin"].extend((sorted_class[..., 0] - sorted_class[..., 1]).flatten().cpu().tolist())
        values[f"{prefix}_top1_confidence"].extend(top_scores.max(1).values.cpu().tolist())

        if stage_index >= 2:
            health = stage["masked_attention"]
            values[f"{prefix}_visible_ratio"].extend(health["visible_ratio"].flatten().cpu().tolist())
            values[f"{prefix}_fallback"].extend(health["fallback_to_global"].flatten().float().cpu().tolist())
            values[f"{prefix}_all_masked"].extend(health["all_masked_before_fallback"].flatten().float().cpu().tolist())

    base = output["query_detail"]["B"].float()
    for name in ("Qp2", "Qp3"):
        position = output["query_detail"][name].float()
        displacement = (position - base).norm(dim=-1)
        values[f"{name}_displacement"].extend(displacement.flatten().cpu().tolist())
        values[f"{name}_active"].extend((displacement > 1e-6).flatten().float().cpu().tolist())
        for image in range(position.shape[0]):
            values[f"{name}_cosine"].extend(_pair_cosine(position[image]).cpu().tolist())
        all_finite &= bool(torch.isfinite(position).all())

    final = output["stages"][-1]
    logits = final["mask_logits"].float()
    probability = logits.sigmoid()
    joint = final["confidence"]["joint"].float()
    p_class = final["confidence"]["p_class"].float()
    target_hw = logits.shape[-2:]
    positive = F.interpolate(output["target_detail"]["positive"].float(), target_hw, mode="nearest").bool()
    background = F.interpolate(output["target_detail"]["reliable_background"][:, None].float(), target_hw, mode="nearest")[:, 0].bool()
    for image in range(logits.shape[0]):
        for cls in torch.where(labels[image].bool())[0].tolist():
            ranked = torch.argsort(joint[image, :, cls], descending=True, stable=True)[:8]
            rival = positive[image, [other for other in range(4) if other != cls]].any(0)
            pos, bg = positive[image, cls], background[image]
            for query in ranked.tolist():
                support = locality[query]
                selected = probability[image, query] > .5
                if bool((pos & support).any()):
                    values["positive_recall"].append(float((selected & pos & support).sum() / (pos & support).sum()))
                    values["positive_logit"].extend(logits[image, query][pos & support].cpu().tolist())
                if bool((rival & support).any()):
                    values["rival_leakage"].append(float((selected & rival & support).sum() / (rival & support).sum()))
                    values["negative_logit"].extend(logits[image, query][rival & support].cpu().tolist())
                if bool((bg & support).any()):
                    values["background_leakage"].append(float((selected & bg & support).sum() / (bg & support).sum()))
                    values["negative_logit"].extend(logits[image, query][bg & support].cpu().tolist())

    for prefix, raw, context in (
        ("F5", output["query_detail"]["context_raw"], output["query_detail"]["context_feature"]),
        ("F4", output["pixel_detail"]["F4_raw"], output["pixel_detail"]["F4_context"]),
    ):
        values[f"{prefix}_residual_norm"].append(float((context.float() - raw.float()).square().mean().sqrt()))
        values[f"{prefix}_raw_context_cosine"].append(float(F.cosine_similarity(raw.float(), context.float(), dim=1).mean()))
    values["all_finite"].append(float(all_finite))
    return values


def summarize(snapshot, batches, pmec_rows, model):
    merged = defaultdict(list)
    for batch in batches:
        for key, values in batch.items():
            merged[key].extend(values)
    areas, redundancy, embedding, attention, pca = [], [], [], [], []
    for stage in range(1, 4):
        prefix = f"s{stage}"
        global_area = _stats(merged[f"{prefix}_global_area"])
        local_area = _stats(merged[f"{prefix}_local_area"])
        pair = _stats(merged[f"{prefix}_pair_iou"])
        cosine = _stats(merged[f"{prefix}_embedding_cosine"])
        areas.append({
            "snapshot": snapshot, "stage": stage,
            "global_mean": global_area["mean"], "global_median": global_area["median"],
            "local_mean": local_area["mean"], "local_median": local_area["median"],
            "local_fraction_gt_090": _safe_mean([v > .90 for v in merged[f"{prefix}_local_area"]]),
        })
        redundancy.append({
            "snapshot": snapshot, "stage": stage, "pair_iou_mean": pair["mean"],
            "pair_iou_median": pair["median"],
            "pair_iou_fraction_gt_090": _safe_mean([v > .90 for v in merged[f"{prefix}_pair_iou"]]),
            "pair_iou_fraction_gt_075": _safe_mean([v > .75 for v in merged[f"{prefix}_pair_iou"]]),
        })
        embedding.append({
            "snapshot": snapshot, "stage": stage, "cosine_mean": cosine["mean"],
            "cosine_median": cosine["median"], "cosine_p90": cosine["p90"],
            "cosine_fraction_gt_095": _safe_mean([v > .95 for v in merged[f"{prefix}_embedding_cosine"]]),
        })
        attention.append({
            "snapshot": snapshot, "stage": stage,
            "entropy_mean": _safe_mean(merged[f"{prefix}_attention_entropy"]),
            "pairwise_cosine_mean": _safe_mean(merged[f"{prefix}_attention_cosine"]),
            "pairwise_iou_mean": _safe_mean(merged[f"{prefix}_attention_iou"]),
            "centroid_pairwise_distance_mean": _safe_mean(merged[f"{prefix}_centroid_distance"]),
        })
        present_pairs = sum(merged[f"{prefix}_present_pairs"])
        pca.append({
            "snapshot": snapshot, "stage": stage,
            "present_class_query_coverage": sum(merged[f"{prefix}_present_with_query"]) / max(present_pairs, 1),
            "absent_class_dominance": _safe_mean(merged[f"{prefix}_absent_dominance"]),
            "assigned_absent_fraction": _safe_mean(merged[f"{prefix}_assigned_absent"]),
            "top1_confidence": _safe_mean(merged[f"{prefix}_top1_confidence"]),
            "class_entropy": _safe_mean(merged[f"{prefix}_class_entropy"]),
            "query_entropy": _safe_mean(merged[f"{prefix}_query_entropy"]),
            "class_margin": _safe_mean(merged[f"{prefix}_class_margin"]),
            "all_finite": bool(min(merged["all_finite"] or [0])),
        })
    dynamic = []
    for stage, name in ((2, "Qp2"), (3, "Qp3")):
        displacement = _stats(merged[f"{name}_displacement"])
        focus_cosine = _stats(merged[f"{name}_cosine"])
        dynamic.append({
            "snapshot": snapshot, "stage": stage, "displacement_mean": displacement["mean"],
            "displacement_median": displacement["median"], "displacement_p90": displacement["p90"],
            "active_fraction": _safe_mean(merged[f"{name}_active"]),
            "focus_cosine_mean": focus_cosine["mean"], "focus_cosine_p90": focus_cosine["p90"],
        })
    masked = []
    for stage in (2, 3):
        prefix = f"s{stage}"
        visible = _stats(merged[f"{prefix}_visible_ratio"])
        masked.append({
            "snapshot": snapshot, "stage": stage, "visible_ratio_mean": visible["mean"],
            "visible_ratio_median": visible["median"], "fallback_to_global_fraction": _safe_mean(merged[f"{prefix}_fallback"]),
            "all_masked_before_fallback_fraction": _safe_mean(merged[f"{prefix}_all_masked"]),
        })
    positive, negative = _safe_mean(merged["positive_logit"]), _safe_mean(merged["negative_logit"])
    selectivity = {
        "snapshot": snapshot, "positive_seed_recall": _safe_mean(merged["positive_recall"]),
        "rival_leakage": _safe_mean(merged["rival_leakage"]),
        "background_leakage": _safe_mean(merged["background_leakage"]),
        "positive_mean_logit": positive, "negative_mean_logit": negative,
        "positive_negative_gap": positive - negative,
    }
    pmec_health = {
        "snapshot": snapshot, "present_class_pairs": len(pmec_rows),
        "candidate_coverage": _safe_mean([row["candidate_masks"] >= 1 for row in pmec_rows]),
        "differs_from_top1": _safe_mean([row["differs_from_top1"] for row in pmec_rows]),
        "candidate_masks_per_pair": _safe_mean([row["candidate_masks"] for row in pmec_rows]),
        "groups_per_pair": _safe_mean([row["region_groups"] for row in pmec_rows]),
        "selected_region_area": _safe_mean([row["selected_region_area"] for row in pmec_rows]),
    }
    chpf = {
        "snapshot": snapshot, "gamma5": model.f5_chpf.gamma.detach().item(),
        "gamma4": model.pixel_decoder.f4_chpf.gamma.detach().item(),
        "F5_context_residual_norm": _safe_mean(merged["F5_residual_norm"]),
        "F4_context_residual_norm": _safe_mean(merged["F4_residual_norm"]),
        "F5_raw_context_cosine": _safe_mean(merged["F5_raw_context_cosine"]),
        "F4_raw_context_cosine": _safe_mean(merged["F4_raw_context_cosine"]),
    }
    return {
        "stagewise_mask_area": areas, "stagewise_query_redundancy": redundancy,
        "stagewise_embedding_diversity": embedding, "dynamic_focus": dynamic,
        "attention_focus": attention, "masked_attention_health": masked,
        "semantic_selectivity": selectivity, "pca_health": pca,
        "pmec_health": pmec_health, "chpf_health": chpf,
        "all_finite": bool(min(merged["all_finite"] or [0])),
    }


def _row(summary, key, stage):
    return next(row for row in summary[key] if row["stage"] == stage)


def apply_epoch2_screen(summary):
    final_red = _row(summary, "stagewise_query_redundancy", 3)
    final_area = _row(summary, "stagewise_mask_area", 3)
    selectivity = summary["semantic_selectivity"]
    masked2 = _row(summary, "masked_attention_health", 2)
    masked3 = _row(summary, "masked_attention_health", 3)
    checks = {
        "severe_redundancy": final_red["pair_iou_median"] >= .95 and final_red["pair_iou_fraction_gt_090"] >= .60,
        "rival_leakage": selectivity["rival_leakage"] > .70,
        "background_leakage": selectivity["background_leakage"] > .50,
        "nonpositive_gap": selectivity["positive_negative_gap"] <= 0,
        "all_window_area": final_area["local_median"] > .90 or final_area["local_fraction_gt_090"] > .80,
        "fallback_global": max(masked2["fallback_to_global_fraction"], masked3["fallback_to_global_fraction"]) > .50,
        "nonfinite": not summary["all_finite"],
    }
    failed = [key for key, value in checks.items() if value]
    return {"decision": "FQRF_FOCUS_NOGO" if failed else "CONTINUE_EPOCH3", "checks": checks, "failed_criteria": failed}


def apply_final_gate(summary):
    redundancy = _row(summary, "stagewise_query_redundancy", 3)
    area = _row(summary, "stagewise_mask_area", 3)
    dynamic2, dynamic3 = _row(summary, "dynamic_focus", 2), _row(summary, "dynamic_focus", 3)
    masked2 = _row(summary, "masked_attention_health", 2)
    masked3 = _row(summary, "masked_attention_health", 3)
    selectivity = summary["semantic_selectivity"]
    pca = _row(summary, "pca_health", 3)
    pmec = summary["pmec_health"]
    checks = {
        "A_median_iou": redundancy["pair_iou_median"] < .75,
        "A_high_iou": redundancy["pair_iou_fraction_gt_090"] < .30,
        "B_rival": selectivity["rival_leakage"] < .30,
        "B_background": selectivity["background_leakage"] < .10,
        "B_gap": selectivity["positive_negative_gap"] > 5.0,
        "C_median_area": .20 <= area["local_median"] <= .75,
        "C_high_area": area["local_fraction_gt_090"] < .40,
        "D_dynamic_active": min(dynamic2["active_fraction"], dynamic3["active_fraction"]) > .80,
        "D_visible_ratio": all(.10 <= row["visible_ratio_mean"] <= .90 for row in (masked2, masked3)),
        "D_fallback": max(masked2["fallback_to_global_fraction"], masked3["fallback_to_global_fraction"]) < .30,
        "E_present_queries": pca["present_class_query_coverage"] >= .93,
        "E_absent_dominance": pca["absent_class_dominance"] <= .10,
        "E_finite": bool(pca["all_finite"] and summary["all_finite"]),
        "F_candidates": pmec["candidate_coverage"] >= .85,
        "F_non_top1": pmec["differs_from_top1"] >= .70,
    }
    go = all(checks.values())
    strong = go and all((
        redundancy["pair_iou_median"] < .60,
        redundancy["pair_iou_fraction_gt_090"] < .20,
        selectivity["rival_leakage"] < .20,
        selectivity["background_leakage"] < .05,
        selectivity["positive_negative_gap"] > 10.0,
        pca["present_class_query_coverage"] >= .95,
        pmec["candidate_coverage"] >= .90,
    ))
    decision = "FQRF_PHASE0_STRONG_GO" if strong else "FQRF_PHASE0_GO" if go else "FQRF_FOCUS_NOGO"
    return {"decision": decision, "checks": checks, "strong_go": strong}

