"""Preregistered OSMF-v1.1 readiness and Phase-0 decisions."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from tools.osmf_phase0_audit.gradients import max_consecutive
from tools.osmf_v11_audit import PARAMETER_NAMES


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["objective"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["step"]))
    return grouped


def _parameter_failures(parameter_summary: Mapping[str, Mapping]) -> list[str]:
    failures = []
    for name in PARAMETER_NAMES:
        health = parameter_summary[name]
        if not bool(health["grad_nonzero"]):
            failures.append(f"DEAD_PATH_GRADIENT_{name}")
        if not bool(health["measurable_update"]):
            failures.append(f"DEAD_PATH_UPDATE_{name}")
    return failures


def _representation_failures(representation_rows: list[dict]):
    nogo, review = [], []
    branch_ratios = [
        float(row["semantic_morphology_rms_ratio"])
        for row in representation_rows
    ]
    response_ratios = [
        float(row["semantic_response_rms_ratio"]) for row in representation_rows
    ]
    if max_consecutive([value <= 0.05 or value >= 20.0 for value in branch_ratios]) >= 2:
        nogo.append("BRANCH_COLLAPSE")
    elif max_consecutive(
        [(0.05 < value < 0.10) or (10.0 < value < 20.0) for value in branch_ratios]
    ) >= 2:
        review.append("SUSTAINED_BRANCH_IMBALANCE")
    if max_consecutive([value < 0.05 for value in response_ratios]) >= 2:
        nogo.append("SEMANTIC_RESPONSE_COLLAPSE")
    end_reconstruction = float(representation_rows[-1]["reconstruction_cosine"])
    if end_reconstruction < 0.90:
        nogo.append("RECONSTRUCTION_DESTABILIZED")
    elif end_reconstruction < 0.95:
        review.append("RECONSTRUCTION_REVIEW_ZONE")
    return nogo, review


def readiness_decision(
    *,
    finite: bool,
    gradient_ratio_rows: list[dict],
    representation_rows: list[dict],
    parameter_summary: Mapping[str, Mapping],
    morphology_eq_gradient_active: bool,
) -> tuple[str, list[str], list[str]]:
    nogo, review, flags = [], [], []
    if not finite:
        nogo.append("NONFINITE_TENSOR_LOSS_OR_GRADIENT")
    grouped = _group(gradient_ratio_rows)
    semantic_rows = grouped["sem_pres"]
    semantic_values = [float(row["ratio"]) for row in semantic_rows]
    if max_consecutive([value > 0.50 for value in semantic_values]) >= 2:
        nogo.append("PERSISTENT_SEMANTIC_PRESERVATION_RATIO_GT_0_50")
    nogo.extend(_parameter_failures(parameter_summary))
    if not morphology_eq_gradient_active:
        nogo.append("MORPHOLOGY_OBJECTIVE_INACTIVE")
    representation_nogo, representation_review = _representation_failures(
        representation_rows
    )
    nogo.extend(representation_nogo)
    review.extend(representation_review)

    semantic_update = sum(
        float(parameter_summary[name]["end_update_norm"])
        for name in ("p_sem.weight", "u_sem.weight")
    )
    morphology_update = sum(
        float(parameter_summary[name]["end_update_norm"])
        for name in ("p_morph.weight", "u_morph.weight")
    )
    update_ratio = max(semantic_update, morphology_update) / (
        min(semantic_update, morphology_update) + 1e-12
    )
    mean_semantic = sum(semantic_values) / len(semantic_values)
    if update_ratio > 20.0 and mean_semantic > 0.30:
        flags.append("SEMANTIC_MORPHOLOGY_UPDATE_IMBALANCE_GT_20")
        review.append("PARAMETER_UPDATE_IMBALANCE")

    late_safe = any(
        int(row["step"]) in (4, 8) and float(row["ratio"]) <= 0.30
        for row in semantic_rows
    )
    semantic_pass = mean_semantic <= 0.30 or late_safe
    if not semantic_pass and not nogo:
        review.append("SEMANTIC_RATIO_REMAINS_ABOVE_PASS_RANGE")

    if nogo:
        return "OSMF_V11_SEMANTIC_READINESS_NOGO", sorted(set(flags)), sorted(set(nogo))
    if review:
        return "OSMF_V11_SEMANTIC_READINESS_REVIEW", sorted(set(flags)), sorted(set(review))
    return "OSMF_V11_SEMANTIC_READINESS_PASS", sorted(set(flags)), []


def phase0_decision(
    *,
    finite: bool,
    gradient_ratio_rows: list[dict],
    gradient_cosine_rows: list[dict],
    representation_rows: list[dict],
    parameter_summary: Mapping[str, Mapping],
    morphology_eq_gradient_active: bool,
    eq_responsive: bool,
    sshr_loss_stable: bool,
    cross_covariance_healthy: bool,
) -> tuple[str, list[str], list[str]]:
    nogo, review, flags = [], [], []
    if not finite:
        nogo.append("NONFINITE_TENSOR_LOSS_OR_GRADIENT")
    if not morphology_eq_gradient_active:
        nogo.append("MORPHOLOGY_OBJECTIVE_INACTIVE")
    if not eq_responsive:
        nogo.append("EQUIVARIANCE_ERROR_UNRESPONSIVE")
    if not sshr_loss_stable:
        nogo.append("SSHR_LOSS_DESTABILIZED")
    if not cross_covariance_healthy:
        nogo.append("CROSS_COVARIANCE_EXPLOSION")
    nogo.extend(_parameter_failures(parameter_summary))
    representation_nogo, representation_review = _representation_failures(
        representation_rows
    )
    nogo.extend(representation_nogo)
    review.extend(representation_review)

    ratios = _group(gradient_ratio_rows)
    cosines = _group(gradient_cosine_rows)
    for objective, rows in ratios.items():
        values = [float(row["ratio"]) for row in rows]
        if max_consecutive([value > 0.50 for value in values]) >= 2:
            nogo.append(f"PERSISTENT_{objective.upper()}_RATIO_GT_0_50")
        elif max_consecutive([0.30 < value <= 0.50 for value in values]) >= 2:
            review.append(f"PERSISTENT_{objective.upper()}_REVIEW_ZONE")
        cosine_values = [float(row["cosine"]) for row in cosines[objective]]
        if sum(cosine_values) / len(cosine_values) < -0.5 and sum(values) / len(values) > 0.30:
            flags.append(f"STRONG_GRADIENT_CONFLICT_{objective.upper()}")
            review.append(f"STRONG_GRADIENT_CONFLICT_{objective.upper()}")

    cross_start = float(representation_rows[0]["cross_covariance"])
    cross_end = float(representation_rows[-1]["cross_covariance"])
    branch_healthy = all(
        0.05 < float(row["semantic_morphology_rms_ratio"]) < 20.0
        for row in representation_rows
    )
    if cross_end < cross_start and branch_healthy:
        flags.append("GENUINE_DECORRELATION_SIGNAL")
    elif cross_end < cross_start:
        flags.append("FALSE_DECORRELATION_BY_COLLAPSE")
        nogo.append("FALSE_DECORRELATION_BY_COLLAPSE")

    if nogo:
        return "OSMF_V11_PHASE0_NOGO", sorted(set(flags)), sorted(set(nogo))
    if review:
        return "OSMF_V11_PHASE0_REVIEW", sorted(set(flags)), sorted(set(review))
    return "OSMF_V11_PHASE0_GO", sorted(set(flags)), []

