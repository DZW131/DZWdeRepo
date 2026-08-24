import pytest
import torch

from network import resnet38_cls
from network.resnet38_fdhr import Net as FDHRNet
from network.resnet38_wdch import Net as WDCHNet
from research.wdch import CrossBandWaveletContext, HFRMFDHR


def _identity_ll(operator):
    with torch.no_grad():
        operator.ll_context.weight.zero_()
        center = operator.kernel_size // 2
        operator.ll_context.weight[:, 0, center, center] = 1.0


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_fdhr_equations_match_frozen_spec(variant):
    torch.manual_seed(31)
    value = torch.randn(2, 3, 28, 28)
    operator = CrossBandWaveletContext(3, kernel_size=7, variant=variant)
    _identity_ll(operator)
    ll, lh, hl, hh = operator.haar.dwt(value)
    if variant == "A":
        expected = operator.haar.idwt(ll, 1.1 * lh, 1.1 * hl, 1.1 * hh)
    elif variant == "B":
        magnitude = lh.abs() + hl.abs() + hh.abs()
        expected = operator.haar.idwt(ll * (1.0 + 0.1 * magnitude), lh, hl, hh)
    else:
        hf_down = torch.stack((lh, hl, hh), dim=0).mean(dim=0)
        expected = operator.haar.idwt(ll + 0.1 * hf_down, lh, hl, hh)
    torch.testing.assert_close(operator(value), expected, rtol=1.0e-6, atol=1.0e-6)


def test_variant_c_pool_is_band_mean_without_spatial_downsampling():
    operator = CrossBandWaveletContext(2, variant="C")
    value = torch.randn(1, 2, 28, 28)
    ll, lh, hl, hh = operator.haar.dwt(value)
    hf_down = torch.stack((lh, hl, hh), dim=0).mean(dim=0)
    assert hf_down.shape == ll.shape == (1, 2, 14, 14)


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_fdhr_has_fixed_strength_and_no_extra_trainable_parameters(variant):
    fdhr = FDHRNet(4, fdhr_variant=variant, wdch_kernel_size=7)
    w1 = WDCHNet(4, wdch_kernel_size=7)
    assert fdhr.hfrm_28_1.wdch.strength.item() == pytest.approx(0.1)
    assert "strength" in dict(fdhr.hfrm_28_1.wdch.named_buffers())
    assert "strength" not in dict(fdhr.hfrm_28_1.wdch.named_parameters())
    assert sum(p.numel() for p in fdhr.parameters()) == sum(p.numel() for p in w1.parameters())


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_only_hfrm28_1_changes_and_optimizer_coverage_is_exact(variant):
    model = FDHRNet(4, fdhr_variant=variant, wdch_kernel_size=7)
    assert isinstance(model.hfrm_56, resnet38_cls.HFRM)
    assert isinstance(model.hfrm_28_1, HFRMFDHR)
    assert isinstance(model.hfrm_28_2, resnet38_cls.HFRM)
    groups = model.get_parameter_groups()
    grouped = [parameter for group in groups for parameter in group]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in trainable}


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_fdhr_feature_diagnostics_are_finite(variant):
    operator = CrossBandWaveletContext(3, variant=variant)
    output, diagnostics = operator.forward_with_diagnostics(
        torch.randn(2, 3, 28, 28)
    )
    assert torch.isfinite(output).all()
    assert set(diagnostics) == {
        "E_LL",
        "E_HF",
        "interaction_rms",
        "interaction_input_rms",
    }
    assert all(torch.isfinite(value) for value in diagnostics.values())


def test_unknown_or_tuned_fdhr_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown FDHR variant"):
        CrossBandWaveletContext(2, variant="D")
    with pytest.raises(ValueError, match="freezes interaction strength"):
        CrossBandWaveletContext(2, variant="A", strength=0.2)
