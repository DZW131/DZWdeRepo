"""Frozen decision contract for the RSBR-v0 three-epoch pilot."""

from __future__ import annotations


KNOWN_PRODUCTION_ENVELOPE_PP = 0.01329944
STRONG_GO = "RSBR_V0_PILOT_STRONG_GO"
GO = "RSBR_V0_PILOT_GO"
REVIEW = "RSBR_V0_PILOT_REVIEW"
NOGO = "RSBR_V0_PILOT_NOGO"


def select_best_epoch(epoch_records):
    """Maximum paired delta with the earliest epoch as the tie-break."""

    if not epoch_records:
        raise ValueError("At least one epoch record is required")
    return max(
        epoch_records,
        key=lambda row: (row["paired_delta_miou_pp"], -row["epoch"]),
    )


def decide_pilot(
    *,
    best_delta_miou_pp: float,
    epoch3_delta_miou_pp: float,
    nonnegative_classes_at_best: int,
    positive_classes_at_best: int,
    safety_failure: bool,
    mechanism_review_evidence: bool,
) -> str:
    """Apply safety precedence and the frozen Stage-1 performance gates."""

    if safety_failure or epoch3_delta_miou_pp < 0.0:
        return NOGO
    if (
        best_delta_miou_pp >= 0.30
        and epoch3_delta_miou_pp >= 0.20
        and nonnegative_classes_at_best >= 3
    ):
        return STRONG_GO
    if (
        best_delta_miou_pp >= 0.15
        and epoch3_delta_miou_pp >= 0.0
        and positive_classes_at_best >= 2
    ):
        return GO
    if 0.05 <= best_delta_miou_pp < 0.15 or mechanism_review_evidence:
        return REVIEW
    return NOGO
