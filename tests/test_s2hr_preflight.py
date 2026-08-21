"""Frozen minimal preflight tests for S²HR-v1."""

import inspect

import torch
import torch.nn as nn

from network.resnet38_cls import HFRM, Net as SSHRNet
from network.resnet38_cls_s2hr import Net
from network.s2hfrm28_1 import S2HFRM28_1


def _inputs(batch=2, height=8, width=8):
    torch.manual_seed(17)
    feature = torch.randn(batch, 512, height, width)
    deep_feature = torch.randn(batch, 4096, height, width)
    deep_cam = torch.randn(batch, 4, height, width)
    raw_cam = torch.randn(batch, 4, height, width)
    presence = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)[:batch]
    classifier = torch.randn(4, 512, 1, 1)
    return feature, deep_feature, deep_cam, raw_cam, presence, classifier


def test_zero_initialized_residual_is_bit_exact_identity():
    module = S2HFRM28_1().eval()
    values = _inputs()
    output, _ = module(*values)
    assert torch.equal(output, values[0])
    assert module.gamma_veto.item() == 0.0
    assert module.gamma_context.item() == 0.0
    assert module.gamma_spatial.item() == 0.0


def test_presence_logic_train_and_inference_with_fallback():
    probability = torch.tensor([[0.81, 0.91, 0.79, 0.61], [0.1, 0.2, 0.3, 0.4]])
    logits = torch.logit(probability).view(2, 4, 1, 1).expand(2, 4, 3, 3)
    labels = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)
    training = S2HFRM28_1.training_presence(labels, logits)
    inference = S2HFRM28_1.inference_presence(logits)
    assert training[0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert training[1].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert inference[0].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert inference[1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_boundary_band_single_class_and_two_region_interface():
    single = torch.zeros(1, 8, 8, dtype=torch.long)
    assert S2HFRM28_1.semantic_boundary_band(single).count_nonzero() == 0
    two_region = single.clone()
    two_region[:, :, 4:] = 1
    band = S2HFRM28_1.semantic_boundary_band(two_region)
    assert torch.all(band[:, :, :, 2:6] == 1)
    assert torch.all(band[:, :, :, :2] == 0)
    assert torch.all(band[:, :, :, 6:] == 0)


def test_gate_range_and_frozen_boundary_width():
    module = S2HFRM28_1().eval()
    values = list(_inputs())
    values[2][:, 0, :, :4] = 10
    values[2][:, 1, :, 4:] = 10
    _, diagnostics = module(*values)
    rho = torch.sigmoid(module.rho_boundary_raw).item()
    assert 0.0 <= diagnostics["ch_gate_boundary_mean"].item() <= 1.0
    assert abs(diagnostics["ch_gate_boundary_mean"].item() - (1.0 - rho)) < 1.0e-7
    assert diagnostics["ch_gate_interior_mean"].item() == 1.0


def test_deep_teacher_and_classifier_directions_are_detached():
    module = S2HFRM28_1().eval()
    module.gamma_spatial.data.fill_(1.0)
    feature, deep_feature, deep_cam, raw_cam, presence, classifier = _inputs(batch=1)
    deep_cam.requires_grad_()
    raw_cam.requires_grad_()
    classifier.requires_grad_()
    output, _ = module(
        feature, deep_feature, deep_cam, raw_cam, presence, classifier
    )
    output.sum().backward()
    assert deep_cam.grad is None
    assert raw_cam.grad is not None and raw_cam.grad.abs().sum() > 0
    assert classifier.grad is None


def test_network_replaces_only_hfrm28_1_and_registers_two_scalars():
    model = Net(4)
    assert type(model.hfrm_56) is HFRM
    assert type(model.hfrm_28_2) is HFRM
    assert type(model.hfrm_28_1) is S2HFRM28_1
    groups = model.get_parameter_groups()
    scratch_ids = [id(parameter) for parameter in groups[2]]
    assert scratch_ids.count(id(model.hfrm_28_1.gamma_spatial)) == 1
    assert scratch_ids.count(id(model.hfrm_28_1.rho_boundary_raw)) == 1
    classifier_heads = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
        and module.kernel_size == (1, 1)
        and module.out_channels == 4
    ]
    assert classifier_heads == ["ic_56", "ic1", "ic2", "fc8"]


def test_all_shared_parameters_match_same_seed_a0_initialization():
    torch.manual_seed(42)
    baseline = SSHRNet(4)
    torch.manual_seed(42)
    s2hr = Net(4)
    baseline_state = baseline.state_dict()
    s2hr_state = s2hr.state_dict()
    new_keys = {"hfrm_28_1.gamma_spatial", "hfrm_28_1.rho_boundary_raw"}
    assert set(s2hr_state) - set(baseline_state) == new_keys
    assert set(baseline_state) - set(s2hr_state) == set()
    for name, value in baseline_state.items():
        assert torch.equal(value, s2hr_state[name]), name


def test_forward_api_has_no_dense_target_argument():
    parameters = inspect.signature(Net.forward).parameters
    assert set(parameters) == {"self", "x", "image_label", "mode", "present_mask"}
    assert not any("seg" in name or "boundary" in name or "mask_gt" in name for name in parameters)
