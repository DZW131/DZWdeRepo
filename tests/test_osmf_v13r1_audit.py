from __future__ import annotations

from pathlib import Path

from tools.osmf_v13r1_audit import (
    BATCH_SIZE, OBJECTIVE_WEIGHTS, PHASE0S_BATCHES,
    READINESS_BATCHES, SEED,
)
from tools.osmf_v13r1_audit.decision import phase0s_decision, readiness_decision


def _ratios(values=(0.1, 0.15)):
    return [
        {"objective": objective, "ratio": value}
        for objective in ("sem_pres", "struct", "orth", "rec")
        for value in values
    ]


def _representations():
    return [
        {"semantic_morphology_rms_ratio": 1.0, "semantic_response_rms_ratio": 1.0, "reconstruction_cosine": 1.0, "semantic_agreement": 1.0},
        {"semantic_morphology_rms_ratio": 1.1, "semantic_response_rms_ratio": 1.0, "reconstruction_cosine": 0.99, "semantic_agreement": 0.98},
    ]


def _parameters():
    return {
        name: {"grad_nonzero": True, "measurable_update": True}
        for name in ("p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight")
    }


def test_r1_keeps_v13_scientific_contract():
    assert SEED == 20260817 and BATCH_SIZE == 20
    assert READINESS_BATCHES == 8 and PHASE0S_BATCHES == 128
    assert OBJECTIVE_WEIGHTS == {"sem_pres": 0.05, "struct": 0.05, "orth": 0.05, "rec": 0.10}


def test_r1_readiness_pass_accepts_corrected_graph_expectation():
    decision, reasons, *_ = readiness_decision(
        finite=True, ratio_rows=_ratios(), representation_rows=_representations(),
        parameter_summary=_parameters(), morphology_graph_expected=True,
        semantic_path_active=True,
        causal_rows=[{"delta": -0.01}, {"delta": -0.02}],
        sshr_loss_stable=True,
    )
    assert decision == "OSMF_V13R1_READINESS_PASS"
    assert reasons == []


def test_r1_readiness_rejects_wrong_graph_expectation():
    decision, reasons, *_ = readiness_decision(
        finite=True, ratio_rows=_ratios(), representation_rows=_representations(),
        parameter_summary=_parameters(), morphology_graph_expected=False,
        semantic_path_active=True,
        causal_rows=[{"delta": -0.01}, {"delta": -0.02}],
        sshr_loss_stable=True,
    )
    assert decision == "OSMF_V13R1_READINESS_NOGO"
    assert "MORPHOLOGY_GRAPH_EXPECTATION_FAILED" in reasons


def test_r1_phase0s_go_uses_corrected_graph_gate():
    decision, reasons, *_ = phase0s_decision(
        finite=True, ratio_rows=_ratios(), representation_rows=_representations(),
        parameter_summary=_parameters(), morphology_graph_expected=True,
        semantic_path_active=True, causal_rows=[{"delta": -0.01}] * 4,
        fixed_rows=[
            {"affinity_eq_error_morphology": 0.1, "affinity_eq_error_semantic": 0.1},
            {"affinity_eq_error_morphology": 0.09, "affinity_eq_error_semantic": 0.099},
        ],
        sshr_loss_stable=True, cross_covariance_healthy=True,
    )
    assert decision == "OSMF_V13R1_PHASE0S_GO"
    assert reasons == []


def test_r1_tools_do_not_expose_training_extension_or_test_data():
    for name in (
        "tools/audit_osmf_v13r1_graph_parity.py",
        "tools/audit_osmf_v13r1_gradient_gate.py",
    ):
        source = Path(name).read_text(encoding="utf-8")
        assert 'add_argument("--test' not in source
        assert 'add_argument("--luad' not in source
        assert 'add_argument("--epochs' not in source
        assert "torch.save(" not in source
