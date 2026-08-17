from __future__ import annotations

from pathlib import Path

import pytest
import torch

from network.osmf import OSMFFactorizer
from tools.audit_osmf_phase0_128batch import (
    _flip_dimension,
    _hard_stop_reason,
    forward_objectives,
)
from tools.osmf_phase0_audit import (
    AUDIT_STEPS,
    BATCH_SIZE,
    FORMAL_EPOCHS_FOR_POLY_SCHEDULE,
    GRADIENT_STEPS,
    IMAGE_SIZE,
    NUM_REAL_BATCHES,
    OBJECTIVE_WEIGHTS,
    PARAMETER_NAMES,
    SEED,
)
from tools.osmf_phase0_audit.decision import decide_phase0
from tools.osmf_phase0_audit.gradients import (
    gradient_decomposition,
    max_consecutive,
    parameter_gradient_rows,
    parameter_update_rows,
    snapshot_parameters,
)


def test_frozen_phase0_contract_constants():
    assert AUDIT_STEPS == (0, 1, 2, 4, 8, 16, 32, 64, 96, 128)
    assert GRADIENT_STEPS == (1, 2, 4, 8, 16, 32, 64, 96, 128)
    assert NUM_REAL_BATCHES == 128
    assert SEED == 20260817
    assert BATCH_SIZE == 20
    assert IMAGE_SIZE == 224
    assert FORMAL_EPOCHS_FOR_POLY_SCHEDULE == 25
    assert OBJECTIVE_WEIGHTS == {"sem": 0.20, "eq": 0.20, "orth": 0.05, "rec": 0.10}


def test_frozen_parameter_names_are_exact():
    assert PARAMETER_NAMES == (
        "p_sem.weight",
        "p_morph.weight",
        "u_sem.weight",
        "u_morph.weight",
        "semantic_classifier.weight",
        "semantic_classifier.bias",
    )


def test_flip_schedule_uses_only_horizontal_and_vertical():
    assert [_flip_dimension(step) for step in (1, 2, 4, 8, 16)] == [3, 3, 3, 2, 2]
    assert all(_flip_dimension(step) in (2, 3) for step in range(129))


def test_gradient_decomposition_is_finite_and_does_not_populate_grad():
    representation = torch.randn(2, 3, 4, 4, requires_grad=True)
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    losses = {
        "base": representation.square().mean() + 0.1 * parameter.square(),
        "sem": representation.mean() * parameter,
        "eq": representation.square().mean() * parameter,
        "orth": representation.abs().mean() * parameter,
        "rec": (representation - representation.detach()).square().mean()
        + 0.0 * parameter,
    }
    ratios, cosines = gradient_decomposition(
        losses,
        representation,
        (parameter,),
        OBJECTIVE_WEIGHTS,
    )
    assert [row["objective"] for row in ratios] == ["sem", "eq", "orth", "rec"]
    assert all(row["finite"] for row in ratios + cosines)
    assert all(row["ratio"] >= 0.0 for row in ratios)
    assert parameter.grad is None
    assert representation.grad is None


def test_parameter_gradient_and_update_accounting():
    module = OSMFFactorizer(8, n_class=4)
    initial = snapshot_parameters(module, PARAMETER_NAMES)
    before = snapshot_parameters(module, PARAMETER_NAMES)
    feature = torch.randn(2, 8, 4, 4)
    reconstruction, aux = module(feature)
    (reconstruction.square().mean() + aux["semantic_logits"].square().mean()).backward()
    gradients = parameter_gradient_rows(module, PARAMETER_NAMES)
    with torch.no_grad():
        for parameter in module.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-0.01)
    updates = parameter_update_rows(module, PARAMETER_NAMES, initial, before)
    assert all(row["finite"] for row in gradients + updates)
    assert any(row["grad_norm"] > 0.0 for row in gradients)
    assert any(row["cumulative_update_norm"] > 0.0 for row in updates)


class _DummyOSMFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.osmf_28_1 = OSMFFactorizer(4, n_class=4)

    def forward_with_aux(self, images):
        reconstruction, aux = self.osmf_28_1(images)
        logits = reconstruction.mean(dim=(2, 3))
        outputs = (logits, logits, logits, logits, torch.sigmoid(logits), images, images, images, images, images)
        return outputs, aux

    def forward_osmf_features(self, images):
        return self.osmf_28_1(images)[1]


def test_equivariance_is_in_total_only_every_fourth_step():
    model = _DummyOSMFModel()
    images = torch.randn(2, 4, 5, 5)
    labels = torch.randint(0, 2, (2, 4)).float()
    step1 = forward_objectives(model, images, labels, step=1, force_equivariance=True)
    step4 = forward_objectives(model, images, labels, step=4, force_equivariance=False)
    expected_step1 = (
        step1["base"]
        + 0.20 * step1["sem"]
        + 0.05 * step1["orth"]
        + 0.10 * step1["rec"]
    )
    expected_step4 = (
        step4["base"]
        + 0.20 * step4["sem"]
        + 0.20 * step4["eq"]
        + 0.05 * step4["orth"]
        + 0.10 * step4["rec"]
    )
    assert not step1["scheduled_eq"]
    assert step4["scheduled_eq"]
    assert torch.equal(step1["total"], expected_step1)
    assert torch.equal(step4["total"], expected_step4)


def _healthy_inputs():
    ratio_rows = []
    cosine_rows = []
    for step in GRADIENT_STEPS:
        for objective in ("sem", "eq", "orth", "rec"):
            ratio_rows.append({"step": step, "objective": objective, "ratio": 0.10})
            cosine_rows.append({"step": step, "objective": objective, "cosine": 0.0})
    representation_rows = [
        {
            "step": 0,
            "reconstruction_cosine": 1.0,
            "semantic_morphology_rms_ratio": 1.0,
            "cross_covariance": 0.20,
        },
        {
            "step": 128,
            "reconstruction_cosine": 0.99,
            "semantic_morphology_rms_ratio": 1.1,
            "cross_covariance": 0.18,
        },
    ]
    parameter_summary = {
        name: {
            "grad_nonzero": True,
            "measurable_update": True,
            "mean_grad_norm": 1.0,
            "end_update_norm": 1e-3,
            "end_relative_update": 1e-3,
        }
        for name in PARAMETER_NAMES
    }
    return ratio_rows, cosine_rows, representation_rows, parameter_summary


def _decide(**overrides):
    ratios, cosines, representations, parameters = _healthy_inputs()
    values = {
        "finite": True,
        "gradient_ratio_rows": ratios,
        "gradient_cosine_rows": cosines,
        "representation_rows": representations,
        "parameter_summary": parameters,
        "eq_responsive": True,
        "morphology_eq_gradient_active": True,
        "sshr_loss_stable": True,
        "cross_covariance_finite": True,
        "cost_overhead_percent": 20.0,
    }
    values.update(overrides)
    return decide_phase0(**values)


def test_healthy_decision_is_go():
    decision, flags, reasons = _decide()
    assert decision == "OSMF_PHASE0_GO"
    assert "GENUINE_DECORRELATION_SIGNAL" in flags
    assert reasons == []


@pytest.mark.parametrize(
    "value,expected",
    [(0.40, "OSMF_PHASE0_REVIEW"), (0.60, "OSMF_PHASE0_NOGO")],
)
def test_persistent_gradient_ratio_decisions(value, expected):
    ratios, cosines, representations, parameters = _healthy_inputs()
    for row in ratios:
        if row["objective"] == "sem":
            row["ratio"] = value
    decision, _, _ = _decide(gradient_ratio_rows=ratios)
    assert decision == expected


@pytest.mark.parametrize(
    "cosine,expected",
    [(0.93, "OSMF_PHASE0_REVIEW"), (0.89, "OSMF_PHASE0_NOGO")],
)
def test_reconstruction_decisions(cosine, expected):
    _, _, representations, _ = _healthy_inputs()
    representations[-1]["reconstruction_cosine"] = cosine
    decision, _, _ = _decide(representation_rows=representations)
    assert decision == expected


def test_dead_parameter_path_is_nogo():
    _, _, _, parameters = _healthy_inputs()
    parameters["p_morph.weight"]["measurable_update"] = False
    decision, _, reasons = _decide(parameter_summary=parameters)
    assert decision == "OSMF_PHASE0_NOGO"
    assert any("p_morph.weight" in reason for reason in reasons)


def test_cost_overhead_is_review_not_nogo():
    decision, flags, reasons = _decide(cost_overhead_percent=41.0)
    assert decision == "OSMF_PHASE0_REVIEW"
    assert "COST_REVIEW" in flags
    assert "TRAINING_OVERHEAD_GT_40_PERCENT" in reasons


def test_unresponsive_equivariance_is_nogo():
    decision, _, reasons = _decide(eq_responsive=False)
    assert decision == "OSMF_PHASE0_NOGO"
    assert "EQUIVARIANCE_ERROR_UNRESPONSIVE" in reasons


def test_hard_stop_requires_persistent_ratio():
    ratios, _, representations, _ = _healthy_inputs()
    sem_rows = [row for row in ratios if row["objective"] == "sem"]
    sem_rows[0]["ratio"] = 0.60
    assert _hard_stop_reason(ratios, representations, True) is None
    sem_rows[1]["ratio"] = 0.60
    assert "PERSISTENT_SEM" in _hard_stop_reason(ratios, representations, True)


def test_max_consecutive():
    assert max_consecutive([False, True, True, False, True]) == 2
    assert max_consecutive([]) == 0


def test_audit_entrypoint_exposes_no_validation_test_or_luad_option():
    source = Path("tools/audit_osmf_phase0_128batch.py").read_text(encoding="utf-8")
    assert 'add_argument("--val' not in source
    assert 'add_argument("--test' not in source
    assert 'add_argument("--luad' not in source
