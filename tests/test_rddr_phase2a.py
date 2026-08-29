from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from network.rddr_context import (
    compute_rddr_dross_score,
    context_reliability,
)
from network.resnet38_cls import Net
from train_sshr import validate_rddr_phase2a_training_contract


def _legacy_forward(model, x):
    x = model.conv1a(x)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)
    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    feat_56 = x
    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x); x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    feat_28_1 = F.relu(model.bn45(x))
    x, _ = model.b5(x, get_x_bn_relu=True); x = model.b5_1(x); x = model.b5_2(x)
    feat_28_2 = F.relu(model.bn52(x))
    x, _ = model.b6(x, get_x_bn_relu=True); x = model.b7(x)
    feat_deep = F.relu(model.bn7(x))
    feat_56 = model.hfrm_56(feat_56, feat_deep)
    feat_28_1 = model.hfrm_28_1(feat_28_1, feat_deep)
    feat_28_2 = model.hfrm_28_2(feat_28_2, feat_deep)
    cam_56 = model.ic_56(feat_56)
    cam_28_1 = model.ic1(feat_28_1)
    cam_28_2 = model.ic2(feat_28_2)
    cam_deep = model.fc8(model.dropout7(feat_deep))
    outputs = [
        F.avg_pool2d(cam, kernel_size=cam.shape[-2:]).view(cam.size(0), -1)
        for cam in (cam_56, cam_28_1, cam_28_2, cam_deep)
    ]
    return (
        *outputs,
        torch.sigmoid(outputs[-1]),
        cam_56,
        cam_28_1,
        cam_28_2,
        cam_deep,
        feat_56,
    )


def test_none_mode_equivalence():
    torch.manual_seed(7)
    model = Net(4, rddr_context_mode="none")
    model.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = _legacy_forward(model, image)
        actual = model(image)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)


def test_js_matches_phase0_formula():
    shallow = torch.randn(2, 4, 7, 7)
    deep = torch.randn(2, 4, 7, 7)
    p_shallow = torch.softmax(shallow.float(), dim=1)
    p_deep = torch.softmax(deep.float(), dim=1)
    midpoint = 0.5 * (p_shallow + p_deep)
    expected_js = 0.5 * (
        p_shallow
        * ((p_shallow + 1.0e-8).log() - (midpoint + 1.0e-8).log())
    ).sum(1, keepdim=True)
    expected_js += 0.5 * (
        p_deep * ((p_deep + 1.0e-8).log() - (midpoint + 1.0e-8).log())
    ).sum(1, keepdim=True)
    expected = (expected_js / torch.log(torch.tensor(2.0))).clamp(0.0, 1.0)
    torch.testing.assert_close(
        compute_rddr_dross_score(shallow, deep), expected, rtol=0.0, atol=1.0e-7
    )


def test_q_detached_and_range():
    shallow = torch.randn(3, 4, 8, 8, requires_grad=True)
    deep = torch.randn(3, 4, 8, 8, requires_grad=True)
    q = compute_rddr_dross_score(shallow, deep)
    assert q.shape == (3, 1, 8, 8)
    assert not q.requires_grad
    assert q.grad_fn is None
    assert torch.isfinite(q).all()
    assert float(q.min()) >= 0.0
    assert float(q.max()) <= 1.0


def test_receiver_gate_shape_and_direction():
    q = torch.rand(2, 1, 7, 7)
    gate = context_reliability(q, "receiver")
    assert gate.shape == q.shape
    torch.testing.assert_close(gate, 1.0 - q)


def test_global_gate_is_spatial_constant_and_mean_matched():
    q = torch.rand(2, 1, 7, 7)
    receiver = context_reliability(q, "receiver")
    gate = context_reliability(q, "global")
    assert gate.shape == (2, 1, 1, 1)
    torch.testing.assert_close(gate.flatten(), receiver.mean((-2, -1)).flatten())
    expanded = gate.expand_as(q)
    assert torch.equal(expanded, expanded[..., :1, :1].expand_as(expanded))


def test_no_new_trainable_params_and_optimizer_unchanged():
    models = {mode: Net(4, rddr_context_mode=mode) for mode in ("none", "global", "receiver")}
    reference_names = list(models["none"].state_dict())
    reference_parameters = sum(p.numel() for p in models["none"].parameters())
    reference_groups = [len(group) for group in models["none"].get_parameter_groups()]
    for model in models.values():
        assert list(model.state_dict()) == reference_names
        assert sum(p.numel() for p in model.parameters()) == reference_parameters
        assert [len(group) for group in model.get_parameter_groups()] == reference_groups


def test_semantic_feature_is_not_modified_before_hfrm():
    torch.manual_seed(42)
    baseline = Net(4, rddr_context_mode="none")
    receiver = Net(4, rddr_context_mode="receiver")
    baseline.eval()
    receiver.eval()
    receiver.load_state_dict(baseline.state_dict(), strict=True)
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        base_diag = baseline.forward_rddr_context_diagnostics(image)
        receiver_diag = receiver.forward_rddr_context_diagnostics(image)
    assert torch.equal(base_diag["F28_raw"], receiver_diag["F28_raw"])
    assert torch.equal(base_diag["context_before"], receiver_diag["context_before"])
    cosine = F.cosine_similarity(
        base_diag["F28_raw"].flatten(), receiver_diag["F28_raw"].flatten(), dim=0
    )
    torch.testing.assert_close(cosine, torch.ones_like(cosine))


def test_no_test_or_luad_access():
    valid = SimpleNamespace(
        rddr_context_mode="receiver",
        dataset="bcss",
        eval_every=0,
        seed=42,
        max_epoches=25,
    )
    validate_rddr_phase2a_training_contract(valid)
    for change in (
        {"dataset": "luad"},
        {"eval_every": 1},
        {"seed": 11},
        {"max_epoches": 5},
    ):
        values = vars(valid).copy()
        values.update(change)
        with pytest.raises(AssertionError):
            validate_rddr_phase2a_training_contract(SimpleNamespace(**values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mode", ["global", "receiver"])
def test_bf16_forward_backward(mode):
    model = Net(4, rddr_context_mode=mode).cuda()
    model.train()
    image = torch.randn(2, 3, 64, 64, device="cuda")
    label = torch.randint(0, 2, (2, 4), device="cuda").float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(image)
        loss = sum(
            weight * F.multilabel_soft_margin_loss(output, label)
            for weight, output in zip((0.10, 0.15, 0.25, 0.50), outputs[:4])
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.hfrm_28_1.gamma_context.grad is not None
    assert torch.isfinite(model.hfrm_28_1.gamma_context.grad).all()
