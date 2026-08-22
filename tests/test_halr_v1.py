"""CPU unit and protocol-contract tests for frozen HALR-v1."""

from pathlib import Path

import torch

from network.resnet38_cls import Net
from tools.halr_objectives import apply_pair_transform, epoch_alpha, halr_terms


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_epoch_ramp():
    assert [epoch_alpha(epoch) for epoch in range(1, 7)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert 0.05 * epoch_alpha(1) == 0.0
    assert 0.05 * epoch_alpha(5) == 0.05


def test_per_sample_flip_is_exactly_self_inverse():
    tensor = torch.arange(4 * 3 * 5 * 7).reshape(4, 3, 5, 7)
    codes = torch.tensor([0, 1, 0, 1])
    transformed = apply_pair_transform(tensor, codes)
    assert torch.equal(apply_pair_transform(transformed, codes), tensor)
    assert torch.equal(transformed[0], torch.flip(tensor[0], dims=(-1,)))
    assert torch.equal(transformed[1], torch.flip(tensor[1], dims=(-2,)))


def test_exact_paired_cams_have_zero_cvle():
    torch.manual_seed(3)
    cam28 = torch.randn(3, 4, 6, 6, requires_grad=True)
    deep = torch.randn(3, 4, 6, 6, requires_grad=True)
    codes = torch.tensor([0, 1, 0])
    labels = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.], [1., 0., 0., 1.]])
    terms = halr_terms(
        cam28, deep, apply_pair_transform(cam28, codes),
        apply_pair_transform(deep, codes), labels, codes,
    )
    assert abs(terms["cvle_loss"].item()) < 1.0e-8
    assert abs(terms["jsd28"].item()) < 1.0e-8
    assert abs(terms["jsddeep"].item()) < 1.0e-8


def test_reliability_is_finite_normalized_and_detached():
    torch.manual_seed(4)
    cams = [torch.randn(3, 4, 5, 5, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.], [1., 0., 1., 1.]])
    terms = halr_terms(*cams, labels, torch.tensor([0, 1, 0]))
    weight28 = terms["weight28_per_sample"]
    weightdeep = terms["weightdeep_per_sample"]
    assert torch.isfinite(weight28).all() and torch.isfinite(weightdeep).all()
    assert torch.allclose(weight28 + weightdeep, torch.ones_like(weight28), atol=1.0e-6, rtol=0.0)
    assert not weight28.requires_grad and not weightdeep.requires_grad
    assert not terms["teacher_view1"].requires_grad
    assert not terms["teacher_view2"].requires_grad


def test_cvle_and_rahd_reach_both_hierarchies():
    torch.manual_seed(5)
    cams = [torch.randn(2, 4, 5, 5, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 1.]])
    terms = halr_terms(*cams, labels, torch.tensor([0, 1]))
    gradients = torch.autograd.grad(terms["cvle_loss"] + terms["rahd_loss"], cams)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum().item() > 0.0 for gradient in gradients)


def test_single_present_class_zeroes_both_objectives():
    cams = [torch.randn(2, 4, 5, 5, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    terms = halr_terms(*cams, labels, torch.tensor([0, 1]))
    assert terms["cvle_loss"].item() == 0.0
    assert terms["rahd_loss"].item() == 0.0
    assert not terms["valid_samples"].any()


def test_epoch1_weighted_localization_is_exact_zero():
    cams = [torch.randn(1, 4, 4, 4, requires_grad=True) for _ in range(4)]
    labels = torch.tensor([[1., 1., 0., 0.]])
    terms = halr_terms(*cams, labels, torch.tensor([0]))
    weighted = 0.05 * epoch_alpha(1) * terms["cvle_loss"] + 0.05 * epoch_alpha(1) * terms["rahd_loss"]
    assert weighted.item() == 0.0


def test_halr_uses_exact_official_model_with_no_new_parameters():
    model = Net(4)
    assert sum(parameter.numel() for parameter in model.parameters()) == 112709714
    assert not any("halr" in name or "teacher" in name for name, _ in model.named_parameters())
    training_source = (ROOT / "tools" / "train_halr_v1_25ep.py").read_text(encoding="utf-8")
    assert "from network.resnet38_cls import Net" in training_source
    assert "network.resnet38_cls (clean official A0)" in training_source


def test_training_only_and_no_dense_gt_contract():
    sources = "\n".join(
        (ROOT / name).read_text(encoding="utf-8").lower()
        for name in (
            "tools/halr_objectives.py", "tools/train_halr_v1_25ep.py",
            "tool/infer_halr.py",
        )
    )
    for forbidden in ("spsr", "ptcr", "pcsd", "bps", "pseudo_mask"):
        assert forbidden not in sources
    assert '"segmentation_gt_used_in_training": false' in sources
    inference_source = (ROOT / "tool" / "infer_halr.py").read_text(encoding="utf-8")
    assert '"double_view_inference": False' in inference_source
    assert '"extra_inference_parameters": 0' in inference_source


def test_checkpoint_schedule_and_no_validation_selection():
    source = (ROOT / "tools" / "train_halr_v1_25ep.py").read_text(encoding="utf-8")
    assert '"checkpoint_epochs": (5, 10, 15, 20, 25)' in source
    assert '"validation_during_training": False' in source
    assert "best_val" not in source
    assert "epoch25 FINAL only" in source
