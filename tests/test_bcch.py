import pytest
import torch

from network import resnet38_cls
from network.resnet38_bcch import Net as BCCHNet
from research.wdch import BoundaryAwareContext, HFRMBCCH


def test_boundary_map_matches_confirmed_contract():
    torch.manual_seed(41)
    operator = BoundaryAwareContext(3)
    value = torch.randn(2, 3, 28, 28, requires_grad=True)
    _, lh, hl, hh = operator.haar.dwt(value.detach())
    energy = torch.sqrt(lh.square() + hl.square() + hh.square()).mean(1, keepdim=True)
    expected = (energy - energy.amin((-2, -1), keepdim=True)) / (
        energy.amax((-2, -1), keepdim=True)
        - energy.amin((-2, -1), keepdim=True)
        + 1.0e-6
    )
    expected = torch.nn.functional.interpolate(
        expected, size=(28, 28), mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    actual = operator.boundary_map(value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (2, 1, 28, 28)
    assert not actual.requires_grad


def test_alpha_is_one_minus_boundary_and_output_is_exact_mixture():
    torch.manual_seed(42)
    operator = BoundaryAwareContext(2)
    value = torch.randn(1, 2, 28, 28)
    output, context, boundary, alpha = operator.forward_with_maps(value)
    torch.testing.assert_close(alpha, 1.0 - boundary, rtol=0, atol=0)
    torch.testing.assert_close(
        output, alpha * context + (1.0 - alpha) * value, rtol=0, atol=0
    )
    assert float(alpha.min()) >= 0.0
    assert float(alpha.max()) <= 1.0


def test_constant_input_recovers_exact_original_ch():
    operator = BoundaryAwareContext(2)
    value = torch.ones(1, 2, 28, 28)
    output, context, boundary, alpha = operator.forward_with_maps(value)
    torch.testing.assert_close(boundary, torch.zeros_like(boundary), rtol=0, atol=0)
    torch.testing.assert_close(alpha, torch.ones_like(alpha), rtol=0, atol=0)
    torch.testing.assert_close(output, context, rtol=0, atol=0)


def test_hf_energy_detects_a_synthetic_transition():
    operator = BoundaryAwareContext(3)
    value = torch.zeros(1, 3, 28, 28)
    value[:, :, :, 13:] = 1.0
    boundary = operator.boundary_map(value)
    transition = float(boundary[:, :, :, 12:14].mean())
    homogeneous = float(boundary[:, :, :, :8].mean())
    assert transition > homogeneous + 0.5


def test_only_hfrm28_1_changes_and_trainable_count_equals_c0():
    model = BCCHNet(4)
    c0 = resnet38_cls.Net(4)
    assert isinstance(model.hfrm_56, resnet38_cls.HFRM)
    assert isinstance(model.hfrm_28_1, HFRMBCCH)
    assert isinstance(model.hfrm_28_2, resnet38_cls.HFRM)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in c0.parameters())
    state = model.state_dict()
    assert "hfrm_28_1.context_conv.weight" in state
    assert not any(key.startswith("hfrm_28_1.bcch") for key in state)
    assert set(key for key in state if key.startswith("hfrm_28_1.haar")) == {
        "hfrm_28_1.haar.analysis_filters",
        "hfrm_28_1.haar.synthesis_filters",
    }


def test_optimizer_parameter_coverage_is_exactly_once():
    model = BCCHNet(4)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    assert sum(parameter is model.hfrm_28_1.context_conv.weight for parameter in grouped) == 1
    assert sum(parameter is model.hfrm_28_1.gamma_context for parameter in grouped) == 1


def test_non_ch15_configuration_is_rejected():
    with pytest.raises(ValueError, match="freezes the original CH15"):
        BoundaryAwareContext(2, context_kernel=7)
