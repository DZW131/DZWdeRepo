"""Preregistered OSMF-v1.2 readiness and Phase-0 decisions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping

from tools.osmf_phase0_audit.gradients import max_consecutive
from tools.osmf_v12_audit import PARAMETER_NAMES


def _group(rows: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["objective"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["step"]))
    return grouped


def percentile(values: list[float], quantile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(quantile)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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
    if any(value <= 0.05 or value >= 20.0 for value in branch_ratios):
        nogo.append("BRANCH_COLLAPSE")
    elif any(
        (0.05 < value < 0.10) or (10.0 < value < 20.0)
        for value in branch_ratios
    ):
        review.append("BRANCH_IMBALANCE_REVIEW_ZONE")
    if max_consecutive([value < 0.05 for value in response_ratios]) >= 2:
        nogo.append("SEMANTIC_RESPONSE_COLLAPSE")
    end_reconstruction = float(representation_rows[-1]["reconstruction_cosine"])
    if end_reconstruction < 0.90:
        nogo.append("RECONSTRUCTION_DESTABILIZED")
    elif end_reconstruction < 0.95:
        review.append("RECONSTRUCTION_REVIEW_ZONE")
    return nogo, review


def _strong_conflicts(
    ratio_groups: Mapping[str, list[dict]],
    cosine_groups: Mapping[str, list[dict]],
) -> list[str]:
    conflicts = []
    for objective, rows in ratio_groups.items():
        values = [float(row["ratio"]) for row in rows]
        cosine_values = [
            float(row["cosine"]) for row in cosine_groups[objective]
        ]
        if (
            sum(cosine_values) / len(cosine_values) < -0.5
            and sum(values) / len(values) > 0.20
        ):
            conflicts.append(f"STRONG_GRADIENT_CONFLICT_{objective.upper()}")
    return conflicts


def readiness_decision(
    *,
    finite: bool,
    gradient_ratio_rows: list[dict],
    gradient_cosine_rows: list[dict],
    representation_rows: list[dict],
    parameter_summary: Mapping[str, Mapping],
    morphology_eq_gradient_active: bool,
    sshr_loss_stable: bool,
) -> tuple[str, list[str], list[str]]:
    nogo, review, flags = [], [], []
    if not finite:
        nogo.append("NONFINITE_TENSOR_LOSS_OR_GRADIENT")
    if not morphology_eq_gradient_active:
        nogo.append("MORPHOLOGY_OBJECTIVE_INACTIVE")
    if not sshr_loss_stable:
        nogo.append("SSHR_LOSS_DESTABILIZED")
    nogo.extend(_parameter_failures(parameter_summary))
    representation_nogo, representation_review = _representation_failures(
        representation_rows
    )
    nogo.extend(representation_nogo)
    review.extend(representation_review)

    ratios = _group(gradient_ratio_rows)
    cosines = _group(gradient_cosine_rows)
    for objective in ("sem_pres", "eq"):
        values = [float(row["ratio"]) for row in ratios[objective]]
        mean_value = sum(values) / len(values)
        max_value = max(values)
        if max_consecutive([value > 0.50 for value in values]) >= 2:
            nogo.append(f"PERSISTENT_{objective.upper()}_RATIO_GT_0_50")
        elif max_value > 0.30 or mean_value > 0.20:
            review.append(f"{objective.upper()}_GRADIENT_BUDGET_NOT_MET")
    for objective in ("orth", "rec"):
        if max(float(row["ratio"]) for row in ratios[objective]) > 0.30:
            review.append(f"{objective.upper()}_RATIO_GT_0_30")

    conflicts = _strong_conflicts(ratios, cosines)
    flags.extend(conflicts)
    review.extend(conflicts)

    if nogo:
        return "OSMF_V12_READINESS_NOGO", sorted(set(flags)), sorted(set(nogo))
    if review:
        return "OSMF_V12_READINESS_REVIEW", sorted(set(flags)), sorted(set(review))
    return "OSMF_V12_READINESS_PASS", sorted(set(flags)), []


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
    if not sshr_loss_stable:
        nogo.append("SSHR_LOSS_DESTABILIZED")
    if not cross_covariance_healthy:
        nogo.append("CROSS_COVARIANCE_DESTABILIZED")
    nogo.extend(_parameter_failures(parameter_summary))
    representation_nogo, representation_review = _representation_failures(
        representation_rows
    )
    nogo.extend(representation_nogo)
    review.extend(representation_review)

    ratios = _group(gradient_ratio_rows)
    cosines = _group(gradient_cosine_rows)
    for objective in ("sem_pres", "eq"):
        values = [float(row["ratio"]) for row in ratios[objective]]
        if max_consecutive([value > 0.50 for value in values]) >= 2:
            nogo.append(f"PERSISTENT_{objective.upper()}_RATIO_GT_0_50")
        if sum(values) / len(values) > 0.20:
            review.append(f"{objective.upper()}_MEAN_RATIO_GT_0_20")
        if percentile(values, 0.95) > 0.30:
            review.append(f"{objective.upper()}_P95_RATIO_GT_0_30")
    for objective in ("orth", "rec"):
        if max(float(row["ratio"]) for row in ratios[objective]) > 0.30:
            review.append(f"{objective.upper()}_RATIO_GT_0_30")

    conflicts = _strong_conflicts(ratios, cosines)
    flags.extend(conflicts)
    review.extend(conflicts)

    end_agreement = float(representation_rows[-1]["semantic_agreement"])
    if end_agreement < 0.90:
        review.append("SEMANTIC_AGREEMENT_BELOW_0_90")

    eq_start = float(representation_rows[0]["eq_error_morphology"])
    eq_end = float(representation_rows[-1]["eq_error_morphology"])
    if eq_end >= eq_start or not eq_responsive:
        review.append("MORPHOLOGY_EQUIVARIANCE_NO_FAVORABLE_TREND")

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
    else:
        review.append("CROSS_COVARIANCE_NO_DECREASE")

    if nogo:
        return "OSMF_V12_PHASE0_NOGO", sorted(set(flags)), sorted(set(nogo))
    if review:
        return "OSMF_V12_PHASE0_REVIEW", sorted(set(flags)), sorted(set(review))
    return "OSMF_V12_PHASE0_GO", sorted(set(flags)), []
