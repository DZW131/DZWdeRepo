import torch

from network.tcrd_dynamics import TCRDDynamics


def test_frozen_reaction_initialization_and_matrix_contract():
    module = TCRDDynamics("R")
    matrix = module.competition_matrix()
    assert module.steps == 3
    assert torch.allclose(module.eta_r, torch.tensor(0.10), atol=1e-6)
    assert torch.equal(matrix, matrix.T)
    assert torch.equal(matrix.diag(), torch.zeros(4))
    assert torch.allclose(matrix[matrix > 0], torch.ones(12), atol=1e-6)


def test_reaction_is_zero_sum_and_absent_classes_are_unchanged():
    module = TCRDDynamics("R")
    evidence = torch.randn(2, 4, 6, 6)
    active = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)
    output, diagnostics = module(
        evidence, torch.randn(2, 5, 6, 6), active, return_diagnostics=True
    )
    update = diagnostics["reaction_update"]
    assert torch.allclose(update[0, :2].sum(0), torch.zeros(6, 6), atol=1e-6)
    assert torch.allclose(update[1, 1:].sum(0), torch.zeros(6, 6), atol=1e-6)
    assert torch.equal(output[0, 2:], evidence[0, 2:])
    assert torch.equal(output[1, 0], evidence[1, 0])


def test_less_than_two_active_classes_is_exact_identity():
    module = TCRDDynamics("R")
    evidence = torch.randn(3, 4, 4, 4)
    active = torch.tensor(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.bool
    )
    output = module(evidence, torch.randn(3, 7, 4, 4), active)
    assert torch.equal(output, evidence)


def test_pairwise_parameters_receive_task_gradient():
    module = TCRDDynamics("R")
    evidence = torch.randn(2, 4, 5, 5, requires_grad=True)
    active = torch.ones(2, 4, dtype=torch.bool)
    output = module(evidence, torch.randn(2, 3, 5, 5), active)
    output.square().mean().backward()
    assert module.pair_raw.grad is not None
    assert torch.isfinite(module.pair_raw.grad).all()
    assert module.beta_r.grad is not None
