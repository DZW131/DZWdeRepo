import pytest
import torch
import torch.nn.functional as F

from network import resnet38_cls
from network.resnet38_bcpch import Net as BCPCHNet
from research.wdch import (
    FixedHaarDWT2D,
    HFRMBCPCH,
    LowFrequencyPrototypeRecovery,
    low_frequency_reconstruction,
)
from research.wdch.bcch import _detached_boundary_map


def test_ll_reconstruction_is_exact_idwt_with_zero_high_bands():
    torch.manual_seed(51)
    haar = FixedHaarDWT2D()
    value = torch.randn(2, 5, 8, 8)
    ll, _, _, _ = haar.dwt(value)
    zero = torch.zeros_like(ll)
    expected = haar.idwt(ll, zero, zero, zero)
    actual = low_frequency_reconstruction(haar, value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, lh, hl, hh = haar.dwt(actual)
    torch.testing.assert_close(lh, torch.zeros_like(lh), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(hl, torch.zeros_like(hl), atol=1.0e-6, rtol=0)
    torch.testing.assert_close(hh, torch.zeros_like(hh), atol=1.0e-6, rtol=0)


def test_prototype_dimensions_cam_mask_and_detachment():
    torch.manual_seed(52)
    operator = LowFrequencyPrototypeRecovery()
    feature = torch.randn(2, 6, 8, 8, requires_grad=True)
    cam = torch.randn(2, 4, 8, 8, requires_grad=True)
    deep = torch.full((2, 4, 2, 2), 10.0)
    fallback = torch.randn_like(feature)
    output, state = operator(feature, cam, deep, FixedHaarDWT2D(), fallback)
    assert output.shape == feature.shape
    assert state["prototype"].shape == (2, 4, 6)
    assert state["prototype_weight"].shape == (2, 4, 8, 8)
    assert state["confidence_mask"].shape == (2, 4, 8, 8)
    assert not state["confidence_mask"].requires_grad
    assert torch.equal(
        state["confidence_mask"],
        (state["normalized_cam"] > 0.7) & state["presence"][:, :, None, None],
    )
    torch.testing.assert_close(
        state["prototype_weight"].sum(1),
        torch.ones_like(state["prototype_weight"][:, 0]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_no_valid_prototype_falls_back_exactly_to_affinity():
    operator = LowFrequencyPrototypeRecovery()
    feature = torch.randn(2, 3, 8, 8)
    constant_cam = torch.zeros(2, 4, 8, 8)
    deep = torch.full((2, 4, 2, 2), 10.0)
    fallback = torch.randn_like(feature)
    output, state = operator(
        feature, constant_cam, deep, FixedHaarDWT2D(), fallback
    )
    assert state["fallback"].all()
    torch.testing.assert_close(output, fallback, rtol=0, atol=0)


def test_presence_uses_official_thresholds_and_argmax_fallback():
    logits = torch.tensor(
        [
            [[[2.0]], [[3.0]], [[2.0]], [[1.0]]],
            [[[-4.0]], [[-3.0]], [[-2.0]], [[-1.0]]],
        ]
    )
    presence = LowFrequencyPrototypeRecovery.predicted_presence(logits)
    assert presence[0].tolist() == [True, True, True, True]
    assert presence[1].sum() == 1
    assert presence[1, 3]
    assert not presence.requires_grad


def test_final_context_matches_frozen_bcpch_equation():
    torch.manual_seed(53)
    module = HFRMBCPCH(4, deep_channels=8)
    semantic_probe = torch.nn.Conv2d(4, 4, 1)
    presence_probe = torch.nn.Conv2d(8, 4, 1)
    with torch.no_grad():
        presence_probe.weight.zero_()
        presence_probe.bias.fill_(10.0)
    module.set_shared_probes(semantic_probe, presence_probe)
    feature = torch.randn(1, 4, 8, 8)
    deep = torch.randn(1, 8, 2, 2)
    context, affinity, prototype, boundary, _ = module._context_components(
        feature, deep
    )
    expected = (1.0 - boundary) * (0.5 * affinity + 0.5 * prototype) + boundary * feature
    torch.testing.assert_close(context, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        boundary, _detached_boundary_map(module.haar, feature), rtol=0, atol=0
    )
    assert not boundary.requires_grad


def test_bcpch_reuses_ic1_fc8_without_new_parameters_or_duplicate_state():
    model = BCPCHNet(4)
    c0 = resnet38_cls.Net(4)
    assert isinstance(model.hfrm_28_1, HFRMBCPCH)
    assert model.hfrm_28_1._semantic_probe is model.ic1
    assert model.hfrm_28_1._presence_probe is model.fc8
    assert sum(parameter.numel() for parameter in model.parameters()) == sum(
        parameter.numel() for parameter in c0.parameters()
    )
    state = model.state_dict()
    assert "hfrm_28_1.context_conv.weight" in state
    assert not any("semantic_probe" in key or "presence_probe" in key for key in state)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }


def test_frozen_threshold_and_kernel_are_rejected():
    with pytest.raises(ValueError, match="threshold=0.70"):
        LowFrequencyPrototypeRecovery(cam_threshold=0.6)
    with pytest.raises(ValueError, match="support=15"):
        HFRMBCPCH(4, deep_channels=8, context_kernel=7)
