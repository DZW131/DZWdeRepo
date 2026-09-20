import torch

from network.hqmr import residual_logits
from network.racc import LocalPresenceRescue, RACCController, ReliabilityArbitrator


def test_arbitrator_identity_initialization():
    torch.manual_seed(1)
    coarse, direct = torch.randn(2, 7, 5, 5), torch.randn(2, 7, 9, 9)
    module = ReliabilityArbitrator()
    actual, alpha = module(coarse, direct)
    expected = residual_logits(coarse, direct)
    assert torch.equal(alpha, torch.ones_like(alpha))
    assert torch.equal(actual, expected)


def test_presence_identity_gate_and_shapes():
    module = LocalPresenceRescue()
    result = module(torch.rand(2, 4, 8, 8), torch.rand(2, 4, 16, 16))
    assert result["features"].shape == (2, 4, 5)
    assert torch.allclose(result["probability"], torch.full((2, 4), torch.sigmoid(torch.tensor(-4.0))))
    deep = torch.tensor([[.9, .1, .8, .2]])
    threshold = torch.tensor([.8, .9, .8, .6])
    gate = RACCController.rescued_gate(deep, result["probability"][:1], threshold)
    assert torch.equal(gate, deep > threshold)


def test_new_parameters_receive_gradients():
    arbitrator, presence = ReliabilityArbitrator(), LocalPresenceRescue()
    coarse, direct = torch.randn(1, 3, 4, 4), torch.randn(1, 3, 6, 6)
    output, _ = arbitrator(coarse, direct)
    local = presence(torch.rand(1, 4, 5, 5), torch.rand(1, 4, 7, 7))
    (output.square().mean() + local["logits"].square().mean()).backward()
    assert all(parameter.grad is not None for parameter in arbitrator.parameters())
    assert all(parameter.grad is not None for parameter in presence.parameters())
