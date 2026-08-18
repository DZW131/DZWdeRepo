from __future__ import annotations

from tools.osmf_v13_audit import (
    BATCH_SIZE, FIXED_PROBE_STEPS, OBJECTIVE_WEIGHTS,
    PHASE0S_BATCHES, READINESS_BATCHES, SEED,
)
from tools.osmf_v13_audit.decision import phase0s_decision, readiness_decision


def _rows(values):
    result = []
    for name in ("sem_pres", "struct", "orth", "rec"):
        for value in values:
            result.append({"objective": name, "ratio": value})
    return result


def _representations():
    return [
        {"semantic_morphology_rms_ratio": 1.0, "semantic_response_rms_ratio": 1.0, "reconstruction_cosine": 1.0, "semantic_agreement": 1.0},
        {"semantic_morphology_rms_ratio": 1.1, "semantic_response_rms_ratio": 1.0, "reconstruction_cosine": 0.99, "semantic_agreement": 0.98},
    ]


def _parameters():
    return {name: {"grad_nonzero": True, "measurable_update": True} for name in ("p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight")}


def test_frozen_gate_contract():
    assert SEED == 20260817 and BATCH_SIZE == 20
    assert READINESS_BATCHES == 8 and PHASE0S_BATCHES == 128
    assert FIXED_PROBE_STEPS == (0, 4, 8, 16, 32, 64, 96, 128)
    assert OBJECTIVE_WEIGHTS == {"sem_pres": 0.05, "struct": 0.05, "orth": 0.05, "rec": 0.10}


def test_readiness_pass_requires_favorable_same_pair_causality():
    decision, reasons, _, _ = readiness_decision(
        finite=True, ratio_rows=_rows([0.1, 0.15]),
        representation_rows=_representations(), parameter_summary=_parameters(),
        morph_struct_active=True,
        causal_rows=[{"delta": -0.01}, {"delta": -0.02}],
        sshr_loss_stable=True,
    )
    assert decision == "OSMF_V13_READINESS_PASS" and reasons == []


def test_phase0s_nogo_has_priority_for_invalid_causal_effect():
    fixed = [
        {"affinity_eq_error_morphology": 0.1, "affinity_eq_error_semantic": 0.1},
        {"affinity_eq_error_morphology": 0.09, "affinity_eq_error_semantic": 0.1},
    ]
    decision, reasons, *_ = phase0s_decision(
        finite=True, ratio_rows=_rows([0.1, 0.15]),
        representation_rows=_representations(), parameter_summary=_parameters(),
        morph_struct_active=True,
        causal_rows=[{"delta": 0.01}, {"delta": -0.001}], fixed_rows=fixed,
        sshr_loss_stable=True, cross_covariance_healthy=True,
    )
    assert decision == "OSMF_V13_PHASE0S_NOGO"
    assert "SAME_PAIR_STRUCTURAL_CAUSAL_INVALID" in reasons


def test_phase0s_go_requires_five_percent_fixed_morphology_improvement():
    fixed = [
        {"affinity_eq_error_morphology": 0.1, "affinity_eq_error_semantic": 0.1},
        {"affinity_eq_error_morphology": 0.09, "affinity_eq_error_semantic": 0.099},
    ]
    decision, reasons, _, _, evidence = phase0s_decision(
        finite=True, ratio_rows=_rows([0.1, 0.15]),
        representation_rows=_representations(), parameter_summary=_parameters(),
        morph_struct_active=True,
        causal_rows=[{"delta": -0.01}] * 4, fixed_rows=fixed,
        sshr_loss_stable=True, cross_covariance_healthy=True,
    )
    assert decision == "OSMF_V13_PHASE0S_GO" and reasons == []
    assert evidence["struct_improve_m"] >= 0.05
