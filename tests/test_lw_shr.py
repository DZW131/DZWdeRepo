import math

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from network.resnet38_cls import HFRM, Net
from network.wavelet_gate import (
    GroupedDWT2D,
    SharedLearnableWaveletBank,
    SubbandStructuralGate,
    haar_analysis_filters,
)
from tools.lw_shr_common import (
    official_histogram,
    paired_image_bootstrap_miou,
    scores_from_histogram,
)


@pytest.mark.parametrize("channels", [256, 512, 1024])
@pytest.mark.parametrize("shape", [(8, 10), (7, 9), (7, 10), (8, 9)])
def test_grouped_dwt_shapes_and_finiteness(channels, shape):
    bank = SharedLearnableWaveletBank(trainable=True)
    x = torch.randn(1, channels, *shape)
    bands = GroupedDWT2D()(x, bank)
    expected = (math.ceil(shape[0] / 2), math.ceil(shape[1] / 2))
    assert len(bands) == 4
    for band in bands:
        assert band.shape == (1, channels, *expected)
        assert torch.isfinite(band).all()


def test_haar_initialization_matches_lwtformer_convolution_orientation():
    bank = SharedLearnableWaveletBank(trainable=True)
    dec_lo, dec_hi = haar_analysis_filters()
    torch.testing.assert_close(bank.dec_lo, dec_lo)
    torch.testing.assert_close(bank.dec_hi, dec_hi)
    assert bank.diagnostics()["dec_lo"]["l2_drift"] == 0.0
    assert bank.diagnostics()["dec_hi"]["l2_drift"] == 0.0


@pytest.mark.parametrize("mode", ["fixed", "learnable", "joint"])
def test_zero_output_projection_is_identity_gate(mode):
    bank = SharedLearnableWaveletBank(trainable=mode != "fixed")
    hfrm = HFRM(32, deep_channels=64, context_kernel=3, wavelet_mode=mode)
    hfrm.gamma_veto.data.fill_(0.37)
    hfrm.gamma_context.data.fill_(-0.21)
    feature = torch.randn(2, 32, 9, 11)
    deep = torch.randn(2, 64, 3, 3)

    actual, details = hfrm(
        feature, deep, wavelet_bank=bank, return_diagnostics=True
    )
    pooled = F.adaptive_avg_pool2d(deep, 1).view(2, -1)
    veto = hfrm.veto_mlp(pooled).view(2, 32, 1, 1)
    expected = (
        feature
        + hfrm.gamma_veto * feature * veto
        + hfrm.gamma_context * hfrm.context_conv(feature)
    )
    torch.testing.assert_close(details["wavelet_gate"], torch.ones_like(feature))
    assert float((actual - expected).detach().abs().max()) < 1.0e-5


def test_a2_filter_gradient_opens_on_second_step_without_breaking_identity_init():
    torch.manual_seed(7)
    bank = SharedLearnableWaveletBank(trainable=True)
    gate = SubbandStructuralGate(32)
    optimizer = torch.optim.SGD(
        list(gate.parameters()) + list(bank.parameters()), lr=0.1
    )
    x = torch.randn(2, 32, 8, 8)

    first = 2.0 * torch.sigmoid(gate(x, bank))
    torch.testing.assert_close(first, torch.ones_like(first), rtol=0.0, atol=0.0)
    first.square().mean().backward()
    assert bank.dec_lo.grad is not None and bank.dec_hi.grad is not None
    assert float(bank.dec_lo.grad.norm()) == 0.0
    assert float(bank.dec_hi.grad.norm()) == 0.0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second = 2.0 * torch.sigmoid(gate(x, bank))
    second.square().mean().backward()
    assert float(bank.dec_lo.grad.norm()) > 0.0
    assert float(bank.dec_hi.grad.norm()) > 0.0


@pytest.mark.parametrize(
    "mode,expected_direct",
    [("fixed", 0), ("learnable", 2), ("joint", 3)],
)
def test_new_direct_parameters_are_covered_exactly_once(mode, expected_direct):
    model = Net(4, wavelet_hfrm_mode=mode)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    assert len(grouped) == len({id(parameter) for parameter in grouped})

    expected = []
    if mode in ("learnable", "joint"):
        expected.extend((model.wavelet_bank.dec_lo, model.wavelet_bank.dec_hi))
    if mode == "joint":
        expected.append(model.hfrm_28_1.lambda_sf)
    assert len(expected) == expected_direct
    for parameter in expected:
        assert sum(candidate is parameter for candidate in grouped) == 1


def test_uniform_mode_preserves_original_state_keys_and_outputs():
    torch.manual_seed(42)
    baseline = Net(4)
    torch.manual_seed(42)
    explicit = Net(4, wavelet_hfrm_mode="none")
    assert baseline.state_dict().keys() == explicit.state_dict().keys()
    baseline.eval()
    explicit.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        left = baseline(x)
        right = explicit(x)
    for expected, actual in zip(left, right):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_original_checkpoint_load_only_misses_lw_shr_keys():
    baseline = Net(4)
    proposed = Net(4, wavelet_hfrm_mode="joint")
    incompatible = proposed.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    assert all(
        key.startswith("wavelet_bank.")
        or key.startswith("hfrm_28_1.wavelet_gate.")
        or key == "hfrm_28_1.lambda_sf"
        for key in incompatible.missing_keys
    )


def test_official_histogram_reproduces_background_overwrite():
    truth = np.asarray([[0, 1, 4], [2, 3, 4]], dtype=np.uint8)
    prediction = np.asarray([[0, 0, 0], [2, 1, 3]], dtype=np.uint8)
    histogram = official_histogram(truth, prediction)
    assert histogram[4, 4] == 2
    assert histogram[4, 0] == 0
    score = scores_from_histogram(histogram)
    assert score["histogram"][4][4] == 0
    assert 0.0 <= score["mIoU"] <= 1.0


def test_paired_image_bootstrap_is_reproducible_and_recomputes_global_miou():
    truth_a = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
    truth_b = np.asarray([[2, 2], [3, 3]], dtype=np.uint8)
    base = np.stack(
        [
            official_histogram(truth_a, np.asarray([[0, 1], [1, 1]])),
            official_histogram(truth_b, np.asarray([[2, 2], [2, 3]])),
        ]
    )
    candidate = np.stack(
        [official_histogram(truth_a, truth_a), official_histogram(truth_b, truth_b)]
    )
    left = paired_image_bootstrap_miou(base, candidate, resamples=100, seed=42)
    right = paired_image_bootstrap_miou(base, candidate, resamples=100, seed=42)
    assert left == right
    assert left["ci95_low_pp"] >= 0.0
