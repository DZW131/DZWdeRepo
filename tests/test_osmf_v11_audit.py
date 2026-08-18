from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from network.osmf_v11 import OSMFV11Factorizer
from tools.audit_osmf_v11_gradient_gate import _forward_objectives
from tools.osmf_phase0_audit.gradients import gradient_decomposition
from tools.osmf_v11_audit import (
    BATCH_SIZE,
    IMAGE_SIZE,
    OBJECTIVE_NAMES,
    OBJECTIVE_WEIGHTS,
    PARAMETER_NAMES,
    PHASE0_AUDIT_STEPS,
    PHASE0_BATCHES,
    READINESS_AUDIT_STEPS,
    READINESS_BATCHES,
    SEED,
)
from tools.osmf_v11_audit.decision import phase0_decision, readiness_decision


def test_v11_frozen_audit_contract():
    assert SEED == 20260817
    assert BATCH_SIZE == 20
    assert IMAGE_SIZE == 224
    assert READINESS_BATCHES == 8
    assert PHASE0_BATCHES == 128
    assert READINESS_AUDIT_STEPS == (0, 1, 2, 4, 8)
    assert PHASE0_AUDIT_STEPS == (0, 1, 2, 4, 8, 16, 32, 64, 96, 128)
    assert OBJECTIVE_NAMES == ("sem_pres", "eq", "orth", "rec")
    assert OBJECTIVE_WEIGHTS == {
        "sem_pres": 0.20,
        "eq": 0.20,
        "orth": 0.05,
        "rec": 0.10,
    }
    assert PARAMETER_NAMES == (
        "p_sem.weight",
        "p_morph.weight",
        "u_sem.weight",
        "u_morph.weight",
    )


def test_generic_gradient_decomposition_accepts_sem_pres_name():
    representation = torch.randn(2, 3, 4, 4, requires_grad=True)
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    losses = {
        "base": representation.square().mean(),
        "sem_pres": representation.mean() * parameter,
        "eq": representation.square().mean() * parameter,
        "orth": representation.abs().mean() * parameter,
        "rec": (representation - representation.detach()).square().mean(),
    }
    ratios, cosines = gradient_decomposition(
        losses,
        representation,
        (parameter,),
        OBJECTIVE_WEIGHTS,
        objective_names=OBJECTIVE_NAMES,
    )
    assert [row["objective"] for row in ratios] == list(OBJECTIVE_NAMES)
    assert all(row["finite"] for row in ratios + cosines)
    assert representation.grad is None
    assert parameter.grad is None


class _DummyV11(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.osmf_28_1 = OSMFV11Factorizer(4)
        self.ic1 = torch.nn.Conv2d(4, 4, 1)

    def _attach(self, aux):
        weight = self.ic1.weight.detach()
        bias = self.ic1.bias.detach()
        aux["semantic_teacher_response"] = F.conv2d(
            aux["input"].detach(), weight, bias
        ).detach()
        aux["semantic_student_response"] = F.conv2d(
            aux["semantic_reconstruction"], weight, bias
        )

    def forward_with_aux(self, images):
        reconstruction, aux = self.osmf_28_1(images)
        self._attach(aux)
        logits = reconstruction.mean(dim=(2, 3))
        outputs = (
            logits,
            logits,
            logits,
            logits,
            torch.sigmoid(logits),
            images,
            images,
            images,
            images,
            images,
        )
        return outputs, aux

    def forward_osmf_features(self, images):
        return self.osmf_28_1(images)[1]


def test_equivariance_enters_total_only_at_every_fourth_step():
    model = _DummyV11()
    images = torch.randn(2, 4, 5, 5)
    labels = torch.randint(0, 2, (2, 4)).float()
    step1 = _forward_objectives(model, images, labels, 1, True)
    step4 = _forward_objectives(model, images, labels, 4, False)
    expected1 = (
        step1["base"]
        + 0.20 * step1["sem_pres"]
        + 0.05 * step1["orth"]
        + 0.10 * step1["rec"]
    )
    expected4 = expected1.new_tensor(0.0) + (
        step4["base"]
        + 0.20 * step4["sem_pres"]
        + 0.20 * step4["eq"]
        + 0.05 * step4["orth"]
        + 0.10 * step4["rec"]
    )
    assert not step1["scheduled_eq"]
    assert step4["scheduled_eq"]
    assert torch.equal(step1["total"], expected1)
    assert torch.equal(step4["total"], expected4)


def _parameter_summary():
    return {
        name: {
            "grad_nonzero": True,
            "measurable_update": True,
            "mean_grad_norm": 0.1,
            "end_update_norm": 1e-3,
            "end_relative_update": 1e-3,
        }
        for name in PARAMETER_NAMES
    }


def _representations(end_cosine=0.99, response_ratio=1.0, branch_ratio=1.0):
    return [
        {
            "step": 0,
            "reconstruction_cosine": 1.0,
            "semantic_morphology_rms_ratio": 1.0,
            "semantic_response_rms_ratio": 1.0,
            "cross_covariance": 0.2,
        },
        {
            "step": 8,
            "reconstruction_cosine": end_cosine,
            "semantic_morphology_rms_ratio": branch_ratio,
            "semantic_response_rms_ratio": response_ratio,
            "cross_covariance": 0.18,
        },
    ]


def _rows(sem_values, steps=(1, 2, 4, 8)):
    ratio_rows, cosine_rows = [], []
    for step, sem_value in zip(steps, sem_values):
        for objective in OBJECTIVE_NAMES:
            ratio_rows.append(
                {
                    "step": step,
                    "objective": objective,
                    "ratio": sem_value if objective == "sem_pres" else 0.1,
                }
            )
            cosine_rows.append(
                {"step": step, "objective": objective, "cosine": 0.0}
            )
    return ratio_rows, cosine_rows


def test_readiness_passes_when_late_semantic_ratio_is_safe():
    ratios, _ = _rows((0.42, 0.36, 0.28, 0.24))
    decision, _, reasons = readiness_decision(
        finite=True,
        gradient_ratio_rows=ratios,
        representation_rows=_representations(),
        parameter_summary=_parameter_summary(),
        morphology_eq_gradient_active=True,
    )
    assert decision == "OSMF_V11_SEMANTIC_READINESS_PASS"
    assert reasons == []


def test_readiness_review_when_ratio_improves_but_stays_in_review_zone():
    ratios, _ = _rows((0.45, 0.42, 0.39, 0.36))
    decision, _, reasons = readiness_decision(
        finite=True,
        gradient_ratio_rows=ratios,
        representation_rows=_representations(),
        parameter_summary=_parameter_summary(),
        morphology_eq_gradient_active=True,
    )
    assert decision == "OSMF_V11_SEMANTIC_READINESS_REVIEW"
    assert "SEMANTIC_RATIO_REMAINS_ABOVE_PASS_RANGE" in reasons


def test_readiness_nogo_on_two_persistent_high_points():
    ratios, _ = _rows((0.7, 0.6), steps=(1, 2))
    decision, _, reasons = readiness_decision(
        finite=True,
        gradient_ratio_rows=ratios,
        representation_rows=_representations(),
        parameter_summary=_parameter_summary(),
        morphology_eq_gradient_active=True,
    )
    assert decision == "OSMF_V11_SEMANTIC_READINESS_NOGO"
    assert any("PERSISTENT" in reason for reason in reasons)


def test_phase0_healthy_inputs_are_go():
    ratios, cosines = _rows((0.2, 0.2, 0.2, 0.2))
    decision, flags, reasons = phase0_decision(
        finite=True,
        gradient_ratio_rows=ratios,
        gradient_cosine_rows=cosines,
        representation_rows=_representations(),
        parameter_summary=_parameter_summary(),
        morphology_eq_gradient_active=True,
        eq_responsive=True,
        sshr_loss_stable=True,
        cross_covariance_healthy=True,
    )
    assert decision == "OSMF_V11_PHASE0_GO"
    assert "GENUINE_DECORRELATION_SIGNAL" in flags
    assert reasons == []


def test_phase0_semantic_response_collapse_is_nogo():
    ratios, cosines = _rows((0.2, 0.2, 0.2, 0.2))
    representations = _representations(response_ratio=0.01)
    representations.insert(
        1,
        {
            "step": 4,
            "reconstruction_cosine": 0.99,
            "semantic_morphology_rms_ratio": 1.0,
            "semantic_response_rms_ratio": 0.01,
            "cross_covariance": 0.19,
        },
    )
    decision, _, reasons = phase0_decision(
        finite=True,
        gradient_ratio_rows=ratios,
        gradient_cosine_rows=cosines,
        representation_rows=representations,
        parameter_summary=_parameter_summary(),
        morphology_eq_gradient_active=True,
        eq_responsive=True,
        sshr_loss_stable=True,
        cross_covariance_healthy=True,
    )
    assert decision == "OSMF_V11_PHASE0_NOGO"
    assert "SEMANTIC_RESPONSE_COLLAPSE" in reasons


def test_gradient_gate_cli_exposes_no_validation_test_or_luad_option():
    source = Path("tools/audit_osmf_v11_gradient_gate.py").read_text(
        encoding="utf-8"
    )
    assert 'add_argument("--val' not in source
    assert 'add_argument("--test' not in source
    assert 'add_argument("--luad' not in source

