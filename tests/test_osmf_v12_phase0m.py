from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.osmf_v12_phase0m import (
    AUTHORIZED_BATCHES,
    BATCH_SIZE,
    EQ_STEPS,
    FIXED_PROBE_STEPS,
    GRADIENT_STEPS,
    IMAGE_SIZE,
    OBJECTIVE_WEIGHTS,
    PROBE_IMAGES,
    REPLICATION_AUDIT_STEPS,
    SEED,
)
from tools.osmf_v12_phase0m.diagnostics import (
    affinity_equivariance_error,
    causal_statistics,
    decide_phase0m,
    local_affinity_map,
    morphology_gradient_competition,
    replication_deviations,
)


def test_frozen_phase0m_contract():
    assert SEED == 20260817
    assert BATCH_SIZE == 20
    assert IMAGE_SIZE == 224
    assert AUTHORIZED_BATCHES == 128
    assert PROBE_IMAGES == 64
    assert EQ_STEPS == tuple(range(4, 129, 4))
    assert FIXED_PROBE_STEPS == (0, 4, 8, 16, 32, 64, 96, 128)
    assert GRADIENT_STEPS == (4, 8, 16, 32, 64, 96, 128)
    assert REPLICATION_AUDIT_STEPS == (1, 2, 4, 8, 16, 32, 64, 96, 128)
    assert OBJECTIVE_WEIGHTS == {
        "sem_pres": 0.05,
        "eq": 0.05,
        "orth": 0.05,
        "rec": 0.10,
    }


@pytest.mark.parametrize("flip_dimension", (2, 3))
def test_local_affinity_inverse_alignment_is_exact_for_a_flip(flip_dimension):
    feature = torch.randn(2, 7, 9, 11)
    transformed = torch.flip(feature, dims=(flip_dimension,))
    error = affinity_equivariance_error(feature, transformed, flip_dimension)
    assert error.item() == pytest.approx(0.0, abs=1e-7)


def test_local_affinity_has_eight_channels_and_valid_boundary_mask():
    affinity, mask = local_affinity_map(torch.randn(1, 4, 5, 7))
    assert affinity.shape == (1, 8, 5, 7)
    assert mask.shape == (1, 8, 5, 7)
    assert mask.dtype == torch.bool
    assert 0 < torch.count_nonzero(mask) < mask.numel()


def test_causal_statistics_use_preregistered_neutral_tolerance():
    rows = [
        {"delta": -0.01},
        {"delta": -2e-6},
        {"delta": 5e-7},
        {"delta": 0.02},
    ]
    result = causal_statistics(rows)
    assert result["num_eq_steps"] == 4
    assert result["num_improved"] == 2
    assert result["num_harmed"] == 1
    assert result["num_neutral"] == 1
    assert result["improved_fraction"] == 0.5


def _causal(improved_fraction=0.8, mean_delta=-0.01):
    return {"improved_fraction": improved_fraction, "mean_delta": mean_delta}


def test_decision_valid_requires_same_pair_and_fixed_probe_improvement():
    decision, flags, reasons = decide_phase0m(
        causal=_causal(),
        fixed_raw_delta=-0.002,
        fixed_affinity_delta=-0.001,
        healthy=True,
        mean_eq_base_cosine=0.0,
        replication_instability=False,
    )
    assert decision == "MORPH_EQ_OBJECTIVE_VALID"
    assert "SAME_PAIR_CAUSAL_VALID" in flags
    assert reasons == []


def test_decision_generalization_failure_when_fixed_raw_does_not_improve():
    decision, _, reasons = decide_phase0m(
        causal=_causal(),
        fixed_raw_delta=0.002,
        fixed_affinity_delta=0.001,
        healthy=True,
        mean_eq_base_cosine=0.0,
        replication_instability=False,
    )
    assert decision == "MORPH_EQ_GENERALIZATION_FAILURE"
    assert "SAME_PAIR_EFFECT_DOES_NOT_GENERALIZE_TO_FIXED_PROBE" in reasons


def test_decision_invalid_has_highest_causal_priority():
    decision, flags, _ = decide_phase0m(
        causal=_causal(improved_fraction=0.4, mean_delta=-0.01),
        fixed_raw_delta=-0.01,
        fixed_affinity_delta=-0.01,
        healthy=True,
        mean_eq_base_cosine=-0.6,
        replication_instability=False,
    )
    assert decision == "MORPH_EQ_OBJECTIVE_INVALID"
    assert "SAME_PAIR_CAUSAL_INVALID" in flags
    assert "STRONG_MORPHOLOGY_TASK_CONFLICT" in flags


def test_decision_metric_mismatch_when_affinity_improves_but_raw_fails():
    decision, flags, _ = decide_phase0m(
        causal=_causal(),
        fixed_raw_delta=0.002,
        fixed_affinity_delta=-0.006,
        healthy=True,
        mean_eq_base_cosine=0.0,
        replication_instability=False,
    )
    assert decision == "MORPH_EQ_METRIC_MISMATCH_REVIEW"
    assert "LOCAL_GEOMETRY_IMPROVES_DESPITE_RAW_FEATURE_EQ_FAILURE" in flags


def test_morphology_gradient_competition_does_not_populate_parameter_grads():
    left = torch.nn.Parameter(torch.randn(4))
    right = torch.nn.Parameter(torch.randn(4))
    combined = left + right
    losses = {
        "eq": combined.square().mean(),
        "base": combined.mean(),
        "sem_pres": left.square().mean(),
        "orth": (left * right).mean(),
        "rec": right.square().mean(),
    }
    result = morphology_gradient_competition(losses, (left, right))
    assert result["finite"]
    assert result["eq_grad_norm"] > 0
    assert left.grad is None
    assert right.grad is None


def test_replication_deviation_threshold_is_observational():
    reference = {"a": 1.0, "b": 2.0}
    deviations, unstable = replication_deviations(
        {"a": 1.1, "b": 2.6}, reference
    )
    assert deviations["a"] == pytest.approx(0.1)
    assert deviations["b"] == pytest.approx(0.3)
    assert unstable


def test_cli_exposes_no_validation_test_luad_or_training_extension():
    source = Path("tools/audit_osmf_v12_phase0m.py").read_text(encoding="utf-8")
    assert 'add_argument("--val' not in source
    assert 'add_argument("--test' not in source
    assert 'add_argument("--luad' not in source
    assert 'add_argument("--epochs' not in source
    assert 'torch.save(' not in source
