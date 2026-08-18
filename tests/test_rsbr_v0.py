import torch

from network.rsbr_v0 import RSBRRefinement


def synthetic_inputs():
    generator = torch.Generator().manual_seed(42)
    feature = torch.randn(2, 512, 8, 8, generator=generator)
    cam_56 = torch.randn(2, 4, 16, 16, generator=generator)
    cam_28_1 = torch.randn(2, 4, 8, 8, generator=generator)
    cam_28_2 = torch.randn(2, 4, 8, 8, generator=generator)
    cam_deep = torch.randn(2, 4, 8, 8, generator=generator)
    presence = torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]])
    return feature, cam_56, cam_28_1, cam_28_2, cam_deep, presence


def test_zero_initialization_is_exact_parity():
    module = RSBRRefinement()
    inputs = synthetic_inputs()
    result = module(*inputs)
    assert torch.equal(result.refined_cam, inputs[2])
    assert torch.count_nonzero(result.delta_core) == 0
    assert torch.count_nonzero(result.delta_transition) == 0


def test_region_extraction_is_deterministic():
    module = RSBRRefinement()
    inputs = synthetic_inputs()
    first = module(*inputs, collect_structures=True)
    second = module(*inputs, collect_structures=True)
    assert first.structures == second.structures


def test_both_heads_receive_gradient_after_output_head_opens():
    module = RSBRRefinement()
    inputs = synthetic_inputs()
    with torch.no_grad():
        module.region_semantic_head.weight.fill_(1e-3)
        module.transition_head[-1].weight.fill_(1e-3)
    result = module(*inputs)
    loss = result.refined_cam.square().mean()
    loss.backward()
    assert module.region_semantic_head.weight.grad is not None
    assert torch.count_nonzero(module.region_semantic_head.weight.grad) > 0
    assert module.transition_head[0].weight.grad is not None
    assert torch.count_nonzero(module.transition_head[0].weight.grad) > 0
    assert module.transition_head[-1].weight.grad is not None
    assert torch.count_nonzero(module.transition_head[-1].weight.grad) > 0


def test_parameter_budget_and_frozen_constants():
    module = RSBRRefinement()
    assert module.trainable_parameter_count() == 199_944
    assert module.region_semantic_head.in_features == 512
    assert module.region_semantic_head.out_features == 4
    assert module.transition_head[0].in_channels == 1541
    assert module.transition_head[0].out_channels == 128
    assert module.transition_head[-1].out_channels == 4

