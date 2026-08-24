import numpy as np
import pytest
import torch

from network import resnet38_cls
from network.resnet38_scwdch import Net as SCWDCHNet
from network.resnet38_wdch import Net as WDCHNet
from research.wdch import (
    FixedHaarDWT2D,
    HFRMSCWDCH,
    HFRMWDCH,
    StrengthCalibratedWaveletContext,
    WaveletDecoupledContext,
)
from tools.wdch_common import (
    OfficialMetricAccumulator,
    PairedZoneAccumulator,
    foreground_boundary_distance,
)
from tools.wdch_evaluation import forward_cam_compatible


@pytest.mark.parametrize("shape", [(2, 3, 28, 28), (1, 5, 56, 56)])
def test_haar_fp32_reconstruction_and_shape(shape):
    torch.manual_seed(3)
    transform = FixedHaarDWT2D()
    value = torch.randn(shape)
    reconstructed = transform.reconstruct(value)
    relative = (reconstructed - value).square().mean().sqrt() / value.square().mean().sqrt()
    assert reconstructed.shape == value.shape
    assert float(relative) < 1.0e-6


def test_haar_orthonormal_energy_and_fixed_buffers():
    torch.manual_seed(4)
    transform = FixedHaarDWT2D()
    value = torch.randn(2, 4, 28, 28)
    bands = transform.dwt(value)
    input_energy = value.square().sum()
    band_energy = sum(band.square().sum() for band in bands)
    assert torch.allclose(input_energy, band_energy, rtol=1.0e-6, atol=1.0e-5)
    assert not list(transform.parameters())
    assert set(dict(transform.named_buffers())) == {
        "analysis_filters",
        "synthesis_filters",
    }


def test_haar_rejects_odd_shape_without_resize():
    with pytest.raises(ValueError, match="even spatial"):
        FixedHaarDWT2D().dwt(torch.randn(1, 2, 27, 28))


@pytest.mark.parametrize("kernel", [5, 7, 9])
def test_wdch_identity_when_ll_context_is_identity(kernel):
    torch.manual_seed(kernel)
    operator = WaveletDecoupledContext(3, kernel)
    with torch.no_grad():
        operator.ll_context.weight.zero_()
        operator.ll_context.weight[:, 0, kernel // 2, kernel // 2] = 1.0
    value = torch.randn(2, 3, 28, 28)
    output = operator(value)
    assert torch.allclose(output, value, rtol=1.0e-6, atol=1.0e-6)


def test_wdch_band_ablation_is_explicit_and_bounded_to_hf_names():
    operator = WaveletDecoupledContext(2, 7)
    operator.set_ablation(("LH", "HH"))
    assert operator.ablated_bands == ("HH", "LH")
    output = operator(torch.randn(1, 2, 28, 28))
    assert output.shape == (1, 2, 28, 28)
    with pytest.raises(ValueError, match="Unknown Haar bands"):
        operator.set_ablation(("LL",))


def test_only_hfrm28_1_uses_wdch_and_parameter_coverage_is_exact():
    model = WDCHNet(4, wdch_kernel_size=7)
    assert isinstance(model.hfrm_56, resnet38_cls.HFRM)
    assert isinstance(model.hfrm_28_1, HFRMWDCH)
    assert isinstance(model.hfrm_28_2, resnet38_cls.HFRM)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}
    assert sum(parameter is model.hfrm_28_1.gamma_veto for parameter in grouped) == 1
    assert sum(parameter is model.hfrm_28_1.gamma_context for parameter in grouped) == 1
    assert sum(parameter is model.hfrm_28_1.wdch.ll_context.weight for parameter in grouped) == 1


def test_official_metric_preserves_background_overwrite():
    truth = np.asarray([[0, 4], [1, 4]], dtype=np.uint8)
    prediction = np.asarray([[0, 0], [0, 2]], dtype=np.uint8)
    metric = OfficialMetricAccumulator()
    metric.update(truth, prediction)
    result = metric.result()
    assert result["histogram"][4][4] == 0
    assert result["histogram"][4][0] == 0
    assert result["histogram"][4][2] == 0


def test_hma_boundary_definition_is_reused_exactly():
    truth = np.full((20, 20), 4, dtype=np.uint8)
    truth[2:18, 2:10] = 0
    truth[2:18, 10:18] = 1
    bins = foreground_boundary_distance(truth)
    assert not np.any(bins["boundary_le_7"] & bins["interior_ge_8"])
    assert np.array_equal(
        bins["boundary_le_7"] | bins["interior_ge_8"], truth < 4
    )
    accumulator = PairedZoneAccumulator()
    accumulator.update(truth, truth.copy(), truth.copy())
    result = accumulator.result()
    assert result["boundary_le_7"]["delta_pp"] == 0.0
    assert result["interior_ge_8"]["delta_pp"] == 0.0


def test_training_net_uses_released_forward_cam_equation():
    torch.manual_seed(19)
    reference = resnet38_cls.Net_CAM(4)
    training_net = resnet38_cls.Net(4)
    training_net.load_state_dict(reference.state_dict(), strict=True)
    reference.eval()
    training_net.eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        expected = reference.forward_cam(image)
        actual = forward_cam_compatible(training_net, image)
    for expected_tensor, actual_tensor in zip(expected, actual):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_strength_calibration_equation_is_exact():
    torch.manual_seed(23)
    scale = 1.75
    operator = StrengthCalibratedWaveletContext(3, 7, scale)
    value = torch.randn(2, 3, 28, 28)
    wd_output, _, _ = operator.unscaled_forward_with_bands(value)
    actual = operator(value)
    expected = value + scale * (wd_output - value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_scale_one_recovers_wdch_and_scale_is_fixed_buffer():
    torch.manual_seed(29)
    wdch = WaveletDecoupledContext(2, 7)
    scwdch = StrengthCalibratedWaveletContext(2, 7, 1.0)
    scwdch.load_state_dict({**wdch.state_dict(), "scale": scwdch.scale}, strict=True)
    value = torch.randn(1, 2, 28, 28)
    torch.testing.assert_close(scwdch(value), wdch(value), rtol=0, atol=1.0e-6)
    assert "scale" in dict(scwdch.named_buffers())
    assert "scale" not in dict(scwdch.named_parameters())
    assert scwdch.scale.requires_grad is False


def test_scwdch_modifies_only_hfrm28_1_and_optimizer_coverage_is_exact():
    model = SCWDCHNet(4, wdch_kernel_size=7, scwdch_scale=1.5)
    assert isinstance(model.hfrm_56, resnet38_cls.HFRM)
    assert isinstance(model.hfrm_28_1, HFRMSCWDCH)
    assert isinstance(model.hfrm_28_2, resnet38_cls.HFRM)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in trainable
    }
    assert model.hfrm_28_1.wdch.scale.requires_grad is False
    assert all(
        parameter is not model.hfrm_28_1.wdch.scale for parameter in grouped
    )
