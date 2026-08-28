from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from network.dross_disposal import compute_rddr_dross_score
from network.resnet38_cls import Net
from train_sshr import validate_rddr_phase1_training_contract


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
        F.avg_pool2d(
            cam, kernel_size=(cam.size(2), cam.size(3)), padding=0
        ).view(cam.size(0), -1)
        for cam in (cam_56, cam_28_1, cam_28_2, cam_deep)
    ]
    return (*outputs, torch.sigmoid(outputs[-1]), cam_56, cam_28_1, cam_28_2, cam_deep, feat_56)


def _identity_pair(mode):
    torch.manual_seed(42)
    baseline = Net(4, rddr_phase1_mode="none")
    candidate = Net(4, rddr_phase1_mode=mode)
    baseline.eval()
    candidate.eval()
    incompatible = candidate.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = baseline(x)
        actual = candidate(x)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1.0e-6)


def test_default_forward_equivalence():
    torch.manual_seed(7)
    model = Net(4, rddr_phase1_mode="none")
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = _legacy_forward(model, x)
        actual = model(x)
    for left, right in zip(expected, actual):
        assert torch.equal(left, right)


def test_dd_identity_init():
    _identity_pair("dd")


def test_uc_identity_init():
    _identity_pair("uc")


def test_q_detached():
    shallow = torch.randn(2, 4, 5, 5, requires_grad=True)
    deep = torch.randn(2, 4, 5, 5, requires_grad=True)
    q = compute_rddr_dross_score(shallow, deep)
    assert not q.requires_grad
    assert q.grad_fn is None


def test_js_finite():
    q = compute_rddr_dross_score(
        torch.tensor([[[[1000.0]], [[-1000.0]], [[0.0]], [[1.0]]]]),
        torch.tensor([[[[-1000.0]], [[1000.0]], [[0.0]], [[1.0]]]]),
    )
    assert torch.isfinite(q).all()


def test_q_range_0_1():
    q = compute_rddr_dross_score(
        torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8)
    )
    assert q.shape == (3, 1, 8, 8)
    assert float(q.min()) >= 0.0
    assert float(q.max()) <= 1.0


def test_optimizer_membership_once():
    model = Net(4, rddr_phase1_mode="dd")
    groups = model.get_parameter_groups()
    all_grouped = [parameter for group in groups for parameter in group]
    for name, parameter in model.named_parameters():
        if name.startswith("dross_disposal."):
            assert sum(candidate is parameter for candidate in all_grouped) == 1
            assert any(candidate is parameter for candidate in groups[2] + groups[3])


def test_no_test_or_luad_access():
    valid = SimpleNamespace(
        rddr_phase1_mode="dd", dataset="bcss", eval_every=0, seed=42, max_epoches=25
    )
    validate_rddr_phase1_training_contract(valid)
    for change in (
        {"dataset": "luad"},
        {"eval_every": 1},
        {"seed": 11},
        {"max_epoches": 5},
    ):
        values = vars(valid).copy()
        values.update(change)
        with pytest.raises(AssertionError):
            validate_rddr_phase1_training_contract(SimpleNamespace(**values))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bf16_forward_backward():
    model = Net(4, rddr_phase1_mode="dd").cuda()
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
    assert model.dross_disposal.expand.weight.grad is not None
    assert torch.isfinite(model.dross_disposal.expand.weight.grad).all()
