"""Unit tests for the frozen OT-MTR head."""

import torch

from network.matr_multiprototype_head import MultiPrototypeCAMHead


def test_mode_offsets_are_centered_per_class():
    head = MultiPrototypeCAMHead(8, 4, 2)
    centered = head.mode_weights() - head.base.weight[:, None, :, 0, 0]
    assert torch.allclose(centered.mean(dim=1), torch.zeros(4, 8), atol=1.0e-8)
    assert head.d_raw.std().item() > 0.0


def test_zero_offsets_log_mean_exp_matches_base_logit():
    torch.manual_seed(2)
    head = MultiPrototypeCAMHead(8, 4, 2)
    with torch.no_grad():
        head.d_raw.zero_()
    features = torch.randn(3, 8, 5, 5)
    aggregated, modes = head(features)
    base = head.base(features)
    assert torch.equal(modes[:, :, 0], base)
    assert torch.equal(modes[:, :, 1], base)
    assert torch.equal(aggregated, base)


def test_head_shapes_and_mode_gradient():
    head = MultiPrototypeCAMHead(8, 4, 2)
    features = torch.randn(2, 8, 5, 5, requires_grad=True)
    aggregated, modes = head(features)
    assert aggregated.shape == (2, 4, 5, 5)
    assert modes.shape == (2, 4, 2, 5, 5)
    aggregated.square().mean().backward()
    assert head.d_raw.grad is not None
    assert torch.isfinite(head.d_raw.grad).all()
    assert head.d_raw.grad.abs().sum() > 0
