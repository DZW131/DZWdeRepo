"""Frozen decision contract for the RSBR-v0 corrected parity audit."""

from __future__ import annotations


SELF_MIOU_ENVELOPE_PP = 0.01329944
SELF_PIXEL_ENVELOPE = 83_626
MIOU_EPSILON_PP = 0.0005
PIXEL_EPSILON = 4_182
MIOU_ALLOWANCE_PP = SELF_MIOU_ENVELOPE_PP + MIOU_EPSILON_PP
PIXEL_ALLOWANCE = SELF_PIXEL_ENVELOPE + PIXEL_EPSILON

MODEL_IDENTITY_NOGO = "RSBR_V0_PARITY_R1_MODEL_IDENTITY_NOGO"
NUMERICAL_REVIEW = "RSBR_V0_PARITY_R1_NUMERICAL_REVIEW"
PARITY_PASS = "RSBR_V0_PARITY_R1_PASS"


def decide_parity_r1(
    *,
    max_cam_difference: float,
    delta_core_exact_zero: bool,
    delta_transition_exact_zero: bool,
    same_process_prediction_differences: int,
    production_miou_difference_pp: float | None = None,
    production_prediction_differences: int | None = None,
) -> str:
    """Apply the preregistered hard identity gate and frozen envelope gate."""

    identity_pass = (
        max_cam_difference == 0.0
        and delta_core_exact_zero
        and delta_transition_exact_zero
        and same_process_prediction_differences == 0
    )
    if not identity_pass:
        return MODEL_IDENTITY_NOGO
    if production_miou_difference_pp is None or production_prediction_differences is None:
        raise ValueError("Production comparison is required after identity PASS")
    if (
        production_miou_difference_pp <= MIOU_ALLOWANCE_PP
        and production_prediction_differences <= PIXEL_ALLOWANCE
    ):
        return PARITY_PASS
    return NUMERICAL_REVIEW
