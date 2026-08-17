import math
from types import SimpleNamespace

import pytest
import torch

from network.cdsr import (
    AnalyticalRectificationNeed,
    SelectiveRectificationGate,
)
from network.cdsr.disagreement import normalized_entropy, normalized_jsd
from network.resnet38_cls import HFRM, Net, Net_CAM
from train_sshr import get_model_kwargs
from tools.analyze_cdsr_need_signal import analytical_need as phase0_need


def _confident_logits(class_id, shape=(1, 4, 3, 3)):
    logits = torch.full(shape, -12.0)
    logits[:, class_id] = 12.0
    return logits


def _cdsr_parameter_names(model):
    return [
        name
        for name, _ in model.named_parameters()
        if ".selective_gate." in name
    ]


def test_jsd_is_symmetric_and_normalized():
    first = torch.softmax(torch.randn(2, 4, 7, 5), dim=1)
    second = torch.softmax(torch.randn(2, 4, 7, 5), dim=1)
    forward = normalized_jsd(first, second)
    reverse = normalized_jsd(second, first)

    torch.testing.assert_close(forward, reverse)
    assert forward.shape == (2, 1, 7, 5)
    assert torch.all((forward >= 0.0) & (forward <= 1.0))


def test_entropy_and_need_components_are_normalized():
    stage = torch.randn(2, 4, 8, 6)
    deep = torch.randn(2, 4, 4, 3)
    result = AnalyticalRectificationNeed()(stage, deep)

    entropy = normalized_entropy(torch.softmax(stage, dim=1))
    assert torch.all((entropy >= 0.0) & (entropy <= 1.0))
    for value in result.values():
        assert value.shape == (2, 1, 8, 6)
        assert torch.all((value >= 0.0) & (value <= 1.0))


def test_identical_confident_views_have_low_need():
    logits = _confident_logits(0)
    result = AnalyticalRectificationNeed()(logits, logits)
    assert result["need_map"].max().item() < 1e-5


def test_reliable_deep_disagreement_has_high_need():
    result = AnalyticalRectificationNeed()(
        _confident_logits(0), _confident_logits(1)
    )
    assert result["need_map"].min().item() > 0.95


def test_unreliable_deep_suppresses_need():
    result = AnalyticalRectificationNeed()(
        _confident_logits(0), torch.zeros(1, 4, 3, 3)
    )
    assert result["deep_reliability"].abs().max().item() < 1e-6
    assert result["need_map"].abs().max().item() < 1e-6


def test_need_module_has_no_parameters_and_stops_gradients():
    module = AnalyticalRectificationNeed()
    stage = torch.randn(2, 4, 5, 5, requires_grad=True)
    deep = torch.randn(2, 4, 3, 3, requires_grad=True)
    result = module(stage, deep)

    assert list(module.parameters()) == []
    assert all(not value.requires_grad for value in result.values())


def test_implementation_exactly_matches_frozen_phase0_formula():
    stage = torch.randn(2, 4, 9, 7)
    deep = torch.randn(2, 4, 5, 4)

    frozen = phase0_need(stage, deep)
    implemented = AnalyticalRectificationNeed()(stage, deep)

    torch.testing.assert_close(
        implemented["disagreement"][:, 0], frozen["disagreement"]
    )
    torch.testing.assert_close(
        implemented["stage_uncertainty"][:, 0], frozen["uncertainty"]
    )
    torch.testing.assert_close(
        implemented["deep_reliability"][:, 0], frozen["deep_reliability"]
    )
    torch.testing.assert_close(
        implemented["need_map"][:, 0], frozen["need"]
    )


def test_gate_initialization_and_range():
    gate = SelectiveRectificationGate(alpha_init=0.10)
    need = torch.linspace(0.0, 1.0, 20).reshape(1, 1, 4, 5)
    semantic, context = gate(need)

    assert gate.alpha_sem.item() == pytest.approx(0.10, abs=1e-7)
    assert gate.alpha_ctx.item() == pytest.approx(0.10, abs=1e-7)
    for value in (semantic, context):
        assert value.min().item() >= 0.90 - 1e-7
        assert value.max().item() <= 1.0


def test_cdsr_has_exactly_six_new_learnable_scalars():
    uniform = Net(n_class=4)
    cdsr = Net(n_class=4, rectification_mode="cdsr")
    new_names = sorted(set(dict(cdsr.named_parameters())) - set(dict(uniform.named_parameters())))

    assert new_names == sorted(_cdsr_parameter_names(cdsr))
    assert len(new_names) == 6
    assert all(dict(cdsr.named_parameters())[name].numel() == 1 for name in new_names)


def test_uniform_state_loads_into_cdsr_with_only_six_expected_missing_keys():
    uniform = Net(n_class=4)
    cdsr = Net(n_class=4, rectification_mode="cdsr")
    incompatible = cdsr.load_state_dict(uniform.state_dict(), strict=False)

    assert sorted(incompatible.missing_keys) == sorted(_cdsr_parameter_names(cdsr))
    assert incompatible.unexpected_keys == []


def test_default_uniform_model_has_no_cdsr_state_and_preserves_ch15():
    model = Net(n_class=4)
    assert model.rectification_mode == "uniform"
    assert not hasattr(model, "cdsr_need")
    assert _cdsr_parameter_names(model) == []
    for hfrm in (model.hfrm_56, model.hfrm_28_1, model.hfrm_28_2):
        assert hfrm.context_mode == "ch"
        assert hfrm.context_conv.kernel_size == (15, 15)


def test_cdsr_initial_forward_is_identity_centered_like_uniform():
    torch.manual_seed(7)
    uniform = Net(n_class=4)
    cdsr = Net(n_class=4, rectification_mode="cdsr")
    uniform.eval()
    cdsr.eval()
    cdsr.load_state_dict(uniform.state_dict(), strict=False)
    image = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        uniform_outputs = uniform(image)
        cdsr_outputs = cdsr(image)

    for uniform_value, cdsr_value in zip(uniform_outputs, cdsr_outputs):
        assert torch.equal(uniform_value, cdsr_value)


def test_alpha_zero_exactly_degenerates_to_original_hfrm():
    uniform = HFRM(in_channels=8, deep_channels=16).eval()
    cdsr = HFRM(
        in_channels=8,
        deep_channels=16,
        rectification_mode="cdsr",
    ).eval()
    cdsr.load_state_dict(uniform.state_dict(), strict=False)
    with torch.no_grad():
        uniform.gamma_veto.fill_(0.31)
        uniform.gamma_context.fill_(-0.17)
        cdsr.gamma_veto.copy_(uniform.gamma_veto)
        cdsr.gamma_context.copy_(uniform.gamma_context)
        cdsr.selective_gate.alpha_sem_logit.fill_(-torch.inf)
        cdsr.selective_gate.alpha_ctx_logit.fill_(-torch.inf)
    feature = torch.randn(2, 8, 9, 9)
    deep = torch.randn(2, 16, 5, 5)
    need_signal = {
        "need_map": torch.rand(2, 1, 9, 9),
        "disagreement": torch.rand(2, 1, 9, 9),
        "stage_uncertainty": torch.rand(2, 1, 9, 9),
        "deep_reliability": torch.rand(2, 1, 9, 9),
    }

    reference = uniform(feature, deep)
    candidate = cdsr(feature, deep, need_signal=need_signal)
    assert torch.equal(reference, candidate)


def test_existing_cam_heads_are_reused_as_detached_raw_probes():
    model = Net(n_class=4, rectification_mode="cdsr")
    model.eval()
    calls = {"stage1": 0, "stage2": 0, "stage3": 0, "deep": 0}
    handles = []
    for name, head in (
        ("stage1", model.ic_56),
        ("stage2", model.ic1),
        ("stage3", model.ic2),
        ("deep", model.fc8),
    ):
        handles.append(
            head.register_forward_hook(
                lambda _module, _inputs, _output, key=name: calls.__setitem__(
                    key, calls[key] + 1
                )
            )
        )

    image = torch.randn(1, 3, 64, 64, requires_grad=True)
    outputs, diagnostics = model.forward_with_diagnostics(image)
    for handle in handles:
        handle.remove()

    assert calls == {"stage1": 2, "stage2": 2, "stage3": 2, "deep": 2}
    assert set(diagnostics["cdsr"]) == {"stage1", "stage2", "stage3"}
    assert all(
        not diagnostics["cdsr"][stage]["need_map"].requires_grad
        for stage in diagnostics["cdsr"]
    )
    assert all(torch.isfinite(value).all() for value in outputs)


def test_stage_shapes_and_forward_cam_are_finite():
    model = Net(n_class=4, rectification_mode="cdsr")
    cam_model = Net_CAM(n_class=4, rectification_mode="cdsr")
    model.eval()
    cam_model.eval()
    cam_model.load_state_dict(model.state_dict())
    image = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        outputs, diagnostics = model.forward_with_diagnostics(image)
        cams = cam_model.forward_cam(image)

    expected = {"stage1": (16, 16), "stage2": (8, 8), "stage3": (8, 8)}
    for stage, spatial in expected.items():
        assert diagnostics["cdsr"][stage]["need_map"].shape[-2:] == spatial
    assert all(torch.isfinite(value).all() for value in outputs)
    assert all(torch.isfinite(value).all() for value in cams)


def test_optimizer_groups_cover_alpha_logits_exactly_once():
    model = Net(n_class=4, rectification_mode="cdsr")
    groups = model.get_parameter_groups()
    grouped_ids = [id(parameter) for group in groups for parameter in group]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    alpha_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if ".selective_gate." in name
    }

    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == set(trainable_ids)
    assert alpha_ids.issubset({id(parameter) for parameter in groups[2]})


def test_cdsr_rejects_archived_innovation_combinations():
    with pytest.raises(ValueError, match="FA-MPR"):
        Net(n_class=4, context_mode="fampr", rectification_mode="cdsr")
    with pytest.raises(ValueError, match="CDSR requires"):
        Net(n_class=4, rectifier_type="hst", rectification_mode="cdsr")


def test_cli_defaults_to_uniform_and_resolves_full_cdsr():
    default = SimpleNamespace(rectifier="hfrm", context_mode="ch")
    cdsr = SimpleNamespace(
        rectifier="hfrm",
        context_mode="ch",
        rectification_mode="cdsr",
    )

    assert get_model_kwargs(default) == {
        "rectifier_type": "hfrm",
        "context_mode": "ch",
        "rectification_mode": "uniform",
    }
    assert get_model_kwargs(cdsr) == {
        "rectifier_type": "hfrm",
        "context_mode": "ch",
        "rectification_mode": "cdsr",
    }
