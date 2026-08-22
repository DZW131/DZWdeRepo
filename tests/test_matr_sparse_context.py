"""Unit tests for zero-anchored pure-PyTorch SACR."""

import torch

from network.matr_hfrm28 import MATR_HFRM28_1
from network.matr_sparse_context import SparseAdaptiveContext
from network.resnet38_cls import HFRM


def test_sacr_initial_delta_is_exact_zero():
    torch.manual_seed(5)
    module = SparseAdaptiveContext(4)
    features = torch.randn(2, 4, 9, 9)
    delta, diagnostics = module(features)
    assert torch.count_nonzero(delta).item() == 0
    assert diagnostics["mean_abs_offset"].item() == 0.0
    assert diagnostics["mean_modulation"].item() == 1.0
    assert 0.018 < module.gamma_adapt.item() < 0.019
    weights = torch.softmax(module.a_logits, dim=1)
    assert torch.equal(weights, torch.full_like(weights, 1.0 / 9.0))


def test_sacr_final_predictor_gradient_is_active_and_beta_finite():
    torch.manual_seed(6)
    module = SparseAdaptiveContext(4)
    features = torch.randn(2, 4, 9, 9, requires_grad=True)
    delta, _ = module(features)
    probe = torch.randn_like(delta)
    loss = (module.gamma_adapt * delta * probe).mean()
    loss.backward()
    final_conv = module.predictor[-1]
    assert final_conv.weight.grad is not None
    assert final_conv.weight.grad.abs().sum() > 0
    assert module.beta_adapt.grad is not None
    assert torch.isfinite(module.beta_adapt.grad).all()


def test_matr_hfrm_exactly_matches_original_at_sacr_anchor():
    torch.manual_seed(7)
    original = HFRM(512, 4096, 15).eval()
    matr = MATR_HFRM28_1().eval()
    incompat = matr.load_state_dict(original.state_dict(), strict=False)
    assert all(name.startswith("sacr.") for name in incompat.missing_keys)
    feature = torch.randn(1, 512, 8, 8)
    deep = torch.randn(1, 4096, 8, 8)
    expected = original(feature, deep)
    observed = matr(feature, deep)
    assert torch.equal(observed, expected)
