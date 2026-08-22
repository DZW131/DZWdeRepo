import math

import torch

from network.tcrd_dynamics import TCRDDynamics


def test_frozen_diffusion_initialization_and_range():
    module = TCRDDynamics("D")
    assert module.steps == 3
    assert torch.allclose(module.kappa, torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(module.eta_d, torch.tensor(0.10), atol=1e-6)
    assert 0.05 < module.eta_d.item() < 0.50


def test_conductance_is_finite_normalized_and_structure_sensitive():
    module = TCRDDynamics("D")
    feature = torch.zeros(1, 3, 5, 5)
    feature[:, 0, :, :3] = 1.0
    feature[:, 1, :, 3:] = 1.0
    weights = module.conductance(feature)
    assert weights.shape == (1, 8, 5, 5)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(1), torch.ones(1, 5, 5), atol=1e-6)
    center_left = weights[0, :, 2, 2]
    same_direction = center_left[3]  # (0,-1)
    cross_direction = center_left[4]  # (0,+1)
    assert same_direction > cross_direction


def test_diffusion_is_nonzero_and_differentiable():
    module = TCRDDynamics("D")
    evidence = torch.randn(2, 4, 7, 7, requires_grad=True)
    feature = torch.randn(2, 6, 7, 7, requires_grad=True)
    output, diagnostics = module(evidence, feature, return_diagnostics=True)
    assert output.shape == evidence.shape
    assert torch.isfinite(output).all()
    assert diagnostics["diffusion_update"].square().mean().sqrt() > 0
    output.square().mean().backward()
    assert module.theta_kappa.grad is not None
    assert module.beta_d.grad is not None
    assert feature.grad is not None


def test_c0_is_exact_identity():
    module = TCRDDynamics("C0")
    evidence = torch.randn(2, 4, 5, 5)
    feature = torch.randn(2, 8, 5, 5)
    assert torch.equal(module(evidence, feature), evidence)
