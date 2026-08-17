"""Preregistered OSMF Phase-0 GO/REVIEW/NOGO decision logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from .gradients import max_consecutive


def _by_objective(rows: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["objective"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["step"]))
    return grouped


def decide_phase0(
    *,
    finite: bool,
    gradient_ratio_rows: list[dict],
    gradient_cosine_rows: list[dict],
    representation_rows: list[dict],
    parameter_summary: Mapping[str, Mapping],
    eq_responsive: bool,
    morphology_eq_gradient_active: bool,
    sshr_loss_stable: bool,
    cross_covariance_finite: bool,
    cost_overhead_percent: float,
) -> tuple[str, list[str], list[str]]:
    """Return exactly one frozen decision plus evidence flags/reasons."""

    nogo = []
    review = []
    flags = []
    if not finite:
        nogo.append("NONFINITE_TENSOR_LOSS_OR_GRADIENT")
    if not sshr_loss_stable:
        nogo.append("SSHR_LOSS_DESTABILIZED")
    if not cross_covariance_finite:
        nogo.append("CROSS_COVARIANCE_NONFINITE")

    ratios = _by_objective(gradient_ratio_rows)
    cosines = _by_objective(gradient_cosine_rows)
    for objective, rows in ratios.items():
        values = [float(row["ratio"]) for row in rows]
        if max_consecutive([value > 0.50 for value in values]) >= 2:
            nogo.append(f"PERSISTENT_{objective.upper()}_GRADIENT_RATIO_GT_0_50")
        elif max_consecutive([0.30 < value <= 0.50 for value in values]) >= 2:
            review.append(f"PERSISTENT_{objective.upper()}_GRADIENT_REVIEW_ZONE")
        mean_cosine = sum(float(row["cosine"]) for row in cosines[objective]) / len(
            cosines[objective]
        )
        mean_ratio = sum(values) / len(values)
        if mean_cosine < -0.5 and mean_ratio > 0.30:
            flags.append(f"STRONG_GRADIENT_CONFLICT_{objective.upper()}")
            review.append(f"STRONG_GRADIENT_CONFLICT_{objective.upper()}")

    end = representation_rows[-1]
    end_cosine = float(end["reconstruction_cosine"])
    if end_cosine < 0.90:
        nogo.append("RECONSTRUCTION_DESTABILIZED")
    elif end_cosine < 0.95:
        review.append("RECONSTRUCTION_COSINE_REVIEW_ZONE")

    ratios_sm = [float(row["semantic_morphology_rms_ratio"]) for row in representation_rows]
    collapse = [value <= 0.05 or value >= 20.0 for value in ratios_sm]
    imbalance = [
        (0.05 < value < 0.10) or (10.0 < value < 20.0) for value in ratios_sm
    ]
    if max_consecutive(collapse) >= 2:
        nogo.append("BRANCH_COLLAPSE")
    elif max_consecutive(imbalance) >= 2:
        flags.append("BRANCH_IMBALANCE_WARNING")
        review.append("SUSTAINED_BRANCH_IMBALANCE")

    required = (
        "p_sem.weight",
        "p_morph.weight",
        "u_sem.weight",
        "u_morph.weight",
        "semantic_classifier.weight",
        "semantic_classifier.bias",
    )
    for name in required:
        health = parameter_summary[name]
        if not bool(health["grad_nonzero"]):
            nogo.append(f"DEAD_PATH_GRADIENT_{name}")
        if not bool(health["measurable_update"]):
            nogo.append(f"DEAD_PATH_UPDATE_{name}")

    if not morphology_eq_gradient_active:
        nogo.append("MORPHOLOGY_OBJECTIVE_INACTIVE")
    if not eq_responsive:
        nogo.append("EQUIVARIANCE_ERROR_UNRESPONSIVE")
    if cost_overhead_percent > 40.0:
        flags.append("COST_REVIEW")
        review.append("TRAINING_OVERHEAD_GT_40_PERCENT")

    cross_start = float(representation_rows[0]["cross_covariance"])
    cross_end = float(representation_rows[-1]["cross_covariance"])
    branches_healthy = all(0.05 < value < 20.0 for value in ratios_sm)
    if cross_end < cross_start and branches_healthy:
        flags.append("GENUINE_DECORRELATION_SIGNAL")
    elif cross_end < cross_start and not branches_healthy:
        flags.append("FALSE_DECORRELATION_BY_COLLAPSE")
        nogo.append("FALSE_DECORRELATION_BY_COLLAPSE")

    nogo = sorted(set(nogo))
    review = sorted(set(review))
    flags = sorted(set(flags))
    if nogo:
        return "OSMF_PHASE0_NOGO", flags, nogo
    if review:
        return "OSMF_PHASE0_REVIEW", flags, review
    return "OSMF_PHASE0_GO", flags, []
