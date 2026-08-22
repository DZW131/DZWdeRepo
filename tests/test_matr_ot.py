"""Unit tests for balanced Sinkhorn and OT-MTR supervision."""

import torch

from network.matr_multiprototype_head import MultiPrototypeCAMHead
from tool.sinkhorn import marginal_errors, sinkhorn_plan
from tools.matr_objectives import epoch_alpha, ot_mtr_loss


def test_frozen_ot_ramp():
    assert [epoch_alpha(epoch) for epoch in range(1, 7)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]


def test_sinkhorn_plan_is_detached_and_balanced():
    torch.manual_seed(3)
    cost = torch.rand(17, 2, requires_grad=True)
    plan = sinkhorn_plan(cost, epsilon=0.1, iterations=20)
    row_error, col_error = marginal_errors(plan)
    assert not plan.requires_grad
    assert torch.isfinite(plan).all()
    assert abs(plan.sum().item() - 1.0) < 1.0e-6
    assert row_error.item() < 2.0e-3
    assert col_error.item() < 1.0e-6


def test_ot_uses_valid_present_pairs_and_reaches_modes_and_features():
    torch.manual_seed(4)
    head = MultiPrototypeCAMHead(8, 4, 2)
    features = torch.randn(2, 8, 4, 4, requires_grad=True)
    aggregated = torch.full((2, 4, 4, 4), -5.0)
    aggregated[:, 0, :, :2] = 5.0
    aggregated[:, 1, :, 2:] = 5.0
    mode_logits = torch.randn(2, 4, 2, 4, 4)
    labels = torch.tensor([[1., 1., 0., 0.], [1., 1., 0., 0.]])
    result = ot_mtr_loss(
        features, aggregated, mode_logits, head.mode_weights(), labels
    )
    assert result["valid_pairs"].item() == 4.0
    assert result["mean_seeds"].item() == 4.0
    feature_grad, offset_grad = torch.autograd.grad(
        result["loss"], (features, head.d_raw)
    )
    assert feature_grad.abs().sum() > 0
    assert offset_grad.abs().sum() > 0
    assert torch.isfinite(feature_grad).all() and torch.isfinite(offset_grad).all()


def test_absent_classes_and_too_small_candidates_are_skipped():
    features = torch.randn(1, 8, 1, 3, requires_grad=True)
    aggregated = torch.randn(1, 4, 1, 3)
    modes = torch.randn(1, 4, 2, 1, 3)
    weights = torch.randn(4, 2, 8, requires_grad=True)
    labels = torch.tensor([[1., 0., 0., 0.]])
    result = ot_mtr_loss(features, aggregated, modes, weights, labels)
    assert result["valid_pairs"].item() == 0.0
    assert result["loss"].item() == 0.0
