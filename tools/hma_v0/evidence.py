"""Deterministic evidence labels and tiered weakness map for HMA-v0."""

from __future__ import annotations

import numpy as np


def build_mechanism_map(summary):
    labels = []
    gamma = summary["gamma"]
    kernels = summary["kernels"]
    validation = summary["validation"]
    gradient = summary["gradient"]
    final = validation["final_variants"]
    pipeline = validation["pipeline_decomposition"]
    gsr_response = validation["gsr_response"]
    present_confusion = validation["present_confusion"]
    spatial = validation["ch_spatial_effect"]
    complementarity = validation["complementarity"]

    all_positive_gamma = all(gamma[stage]["gamma_veto"] > 0 for stage in gamma)
    if all_positive_gamma:
        labels.append("GSR_IS_GLOBAL_AMPLIFICATION")
    absent = gsr_response["absent_primary"]["all_stages"]
    veto_supported = absent["median_delta_logit"] < 0 and absent["fraction_suppressed"] >= 0.60
    if veto_supported:
        labels.append("GSR_VETO_SUPPORTED")
    total_present_raw = sum(item["raw_present_confusion"] for item in present_confusion.values())
    total_present_net = sum(item["net"] for item in present_confusion.values())
    present_net_rate = total_present_net / max(total_present_raw, 1)
    if veto_supported and present_net_rate <= 0.01:
        labels.extend(("GSR_ABSENT_ONLY_EFFECT", "GLOBAL_SEMANTIC_SPATIAL_LIMIT_SUPPORTED"))

    kernel_behaviors = [kernels[stage]["behavior"] for stage in kernels]
    if all(value == "CH_BEHAVES_AS_HOMOGENIZER" for value in kernel_behaviors):
        labels.append("CH_HOMOGENIZER_SUPPORTED")
    if any(value == "CH_FREE_FILTER_BEHAVIOR" for value in kernel_behaviors):
        labels.append("CH_FREE_FILTER_BEHAVIOR")
    interior_net = spatial["raw_to_ch"]["B2_ge_8"]["net"]
    boundary_net = spatial["raw_to_ch"]["B0_le_2"]["net"]
    if interior_net > 0 and boundary_net < 0:
        labels.append("CH_BOUNDARY_TRADEOFF_CONFIRMED")

    raw_miou = final["all_hfrm_off"]["mean_iou"]
    full_miou = final["official_full"]["mean_iou"]
    gsr_gain = final["gsr_only"]["mean_iou"] - raw_miou
    ch_gain = final["ch_only"]["mean_iou"] - raw_miou
    full_gain = full_miou - raw_miou
    recovery_union = (
        complementarity["gsr_recover"] + complementarity["ch_recover"]
        - complementarity["both_recover"]
    )
    g_unique_fraction = complementarity["gsr_unique_recover"] / max(recovery_union, 1)
    c_unique_fraction = complementarity["ch_unique_recover"] / max(recovery_union, 1)
    if (
        gsr_gain > 0 and ch_gain > 0
        and complementarity["recovery_set_jaccard"] < 0.50
        and g_unique_fraction >= 0.10 and c_unique_fraction >= 0.10
    ):
        labels.append("GSR_CH_COMPLEMENTARY")
    elif complementarity["recovery_set_jaccard"] >= 0.70:
        labels.append("GSR_CH_REDUNDANT")
    elif gsr_gain * ch_gain < 0 or full_gain < max(gsr_gain, ch_gain) - 0.0005:
        labels.append("GSR_CH_CONFLICTING")

    impact_28_1 = full_miou - final["hfrm_28_1_off"]["mean_iou"]
    impact_28_2 = full_miou - final["hfrm_28_2_off"]["mean_iou"]
    if impact_28_1 > 0.0005 and impact_28_1 > 1.5 * max(impact_28_2, 1e-12):
        labels.append("HFRM28_1_DOMINANT")
    elif impact_28_2 > 0.0005 and impact_28_2 > 1.5 * max(impact_28_1, 1e-12):
        labels.append("HFRM28_2_DOMINANT")
    else:
        labels.append("NO_SINGLE_STAGE_DOMINANCE")

    feat_norms = gradient["feat_deep_gradient_norm"]
    deep_norm = feat_norms["deep"]["mean"]
    shallow_sum = sum(feat_norms[branch]["mean"] for branch in ("56", "28_1", "28_2"))
    if deep_norm > shallow_sum:
        labels.append("DEEP_SUPERVISION_DOMINANT")
    if shallow_sum >= 0.25 * deep_norm:
        labels.append("SHALLOW_TO_DEEP_GRADIENT_SIGNIFICANT")

    class_gate_gain = (
        pipeline["full_official_gate"]["mean_iou"]
        - pipeline["full_no_gate"]["mean_iou"]
    )
    if abs(class_gate_gain) > 2.0 * max(abs(full_gain), 1e-12):
        labels.append("CLASS_GATE_DOMINANT")

    amplification_cells = 0
    for stage in gsr_response["by_stage_class"].values():
        for cell in stage.values():
            if (
                cell["normalization_amplification_ratio_median"] > 2.0
                and cell["absolute_range_scaled_raw_delta_median"] < 0.01
            ):
                amplification_cells += 1
    if amplification_cells >= 6:
        labels.append("NORMALIZATION_AMPLIFICATION")

    full_errors = validation["error_taxonomy"]["official_full"]
    remaining_absent = full_errors["absent_class"]["candidate_wrong"]
    remaining_present = full_errors["present_confusion"]["candidate_wrong"]
    dominant_remaining_error = (
        "present_class_confusion" if remaining_present > remaining_absent else "absent_class"
    )

    tier_a = []
    if all_positive_gamma:
        tier_a.append(
            "All trained gamma_veto scalars are positive, so the released residual equation performs additive channel amplification/modulation rather than direct feature suppression."
        )
    if not veto_supported:
        tier_a.append(
            "Raw-to-GSR same-forward measurements do not meet the preregistered absent-class veto criterion."
        )
    if present_net_rate <= 0.01:
        tier_a.append(
            "GSR produces at most a 1% net recovery rate on present-class confusion in the standalone stage audit."
        )
    if "CH_FREE_FILTER_BEHAVIOR" in labels:
        tier_a.append(
            "At least one trained CH bank exhibits free-filter rather than constrained low-pass behavior by direct kernel/FFT measurements."
        )
    if "CH_BOUNDARY_TRADEOFF_CONFIRMED" in labels:
        tier_a.append(
            "CH improves interior pixels while causing net near-boundary harm in paired causal predictions."
        )
    if full_gain <= 0:
        tier_a.append(
            "Removing all HFRM residuals from the same trained checkpoint does not reduce official validation mIoU."
        )
    tier_a.append(
        f"The remaining frozen Full errors are dominated by {dominant_remaining_error.replace('_', ' ')} ({remaining_present} present-confusion vs {remaining_absent} absent-class pixels)."
    )

    tier_b = [
        "Each GSR gate is global over space: one channel vector is broadcast to every pixel in a stage.",
        "The same deep feature is directly supervised at weight 0.50 and simultaneously conditions all three HFRMs.",
        "HFRM56 has a training loss but no direct path to the released final inference fusion or deeper backbone stages.",
        "A fixed K=15 spans approximately 60 input pixels at F56 and 120 at F28; this is a structural scale mismatch, not proven performance harm.",
    ]
    tier_c = [
        "Whether a spatially varying semantic mechanism would improve present-class localization is untested.",
        "Whether constraining CH to a low-pass family would improve segmentation is untested.",
        "Whether changing K by stage would improve the smoothing/boundary trade-off is untested.",
        "The historical training-time contribution of HFRM56 cannot be identified by frozen-checkpoint inference ablation.",
    ]
    question = (
        "How can deep semantic guidance resolve spatial present-class confusion while preserving tissue boundaries, and is that limitation causal during retraining?"
        if dominant_remaining_error == "present_class_confusion"
        else "Why do image-absent classes remain competitive after deep semantic conditioning, and which training pathway sustains them?"
    )
    return {
        "completion": "HFRM_MECHANISM_MAP_COMPLETE",
        "labels": list(dict.fromkeys(labels)),
        "measurements": {
            "veto_supported": bool(veto_supported),
            "absent_median_delta_logit": absent["median_delta_logit"],
            "absent_fraction_suppressed": absent["fraction_suppressed"],
            "present_confusion_net_rate": float(present_net_rate),
            "gsr_gain_miou": float(gsr_gain),
            "ch_gain_miou": float(ch_gain),
            "full_gain_miou": float(full_gain),
            "hfrm_28_1_removal_cost_miou": float(impact_28_1),
            "hfrm_28_2_removal_cost_miou": float(impact_28_2),
            "class_gate_gain_miou": float(class_gate_gain),
            "shallow_to_deep_gradient_norm_sum": float(shallow_sum),
            "direct_deep_gradient_norm": float(deep_norm),
            "normalization_amplification_cells": int(amplification_cells),
            "remaining_absent_errors": int(remaining_absent),
            "remaining_present_confusion_errors": int(remaining_present),
        },
        "tier_a_directly_supported": tier_a,
        "tier_b_structurally_plausible": tier_b,
        "tier_c_speculative": tier_c,
        "single_most_important_unresolved_scientific_question": question,
    }
