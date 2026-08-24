import numpy as np
import pytest
import torch

from network import resnet38_cls
from network.resnet38_wdch import Net as WDCHNet
from research.wdch import FixedHaarDWT2D, HFRMWDCH, WaveletDecoupledContext
from tools.wdch_common import (
    OfficialMetricAccumulator,
    PairedZoneAccumulator,
    foreground_boundary_distance,
)


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
