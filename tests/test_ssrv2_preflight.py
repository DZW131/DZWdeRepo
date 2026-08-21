"""CPU unit and contract tests for the frozen SSR-v2 design."""

from pathlib import Path

import torch
import torch.nn.functional as F

from network.hfrm28_1_ssrv2 import SSRv2HFRM28_1, epoch_alpha
from network.resnet38_cls import HFRM, Net as SSHRNet
from network.resnet38_cls_ssrv2 import Net as SSRv2Net


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_epoch_ramp():
    assert [epoch_alpha(epoch) for epoch in range(1, 7)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    assert 0.05 * epoch_alpha(1) == 0.0
    assert 0.05 * epoch_alpha(5) == 0.05


def test_alpha_zero_exactly_matches_original_hfrm():
    torch.manual_seed(7)
    original = HFRM(512, 4096, 15).eval()
    ssrv2 = SSRv2HFRM28_1().eval()
    incompat = ssrv2.load_state_dict(original.state_dict(), strict=False)
    assert set(incompat.missing_keys) == {"beta_spatial"}
    feature = torch.randn(2, 512, 6, 6)
    deep = torch.randn(2, 4096, 6, 6)
    deep_logits = torch.randn(2, 4, 6, 6)
    raw_logits = torch.randn(2, 4, 6, 6)
    presence = torch.tensor([[1., 1., 0., 0.], [1., 0., 1., 0.]])
    classifier = torch.randn(4, 512, 1, 1)
    expected = original(feature, deep)
    observed, _ = ssrv2(
        feature, deep, deep_logits, raw_logits, presence, classifier, alpha=0.0
    )
    assert torch.equal(observed, expected)


def test_pcsd_teacher_detached_student_active_and_ptcr_detached():
    torch.manual_seed(8)
    deep = torch.randn(2, 4, 5, 5, requires_grad=True)
    raw = torch.randn(2, 4, 5, 5, requires_grad=True)
    classifier = torch.randn(4, 512, 1, 1, requires_grad=True)
    presence = torch.tensor([[1., 1., 0., 0.], [0., 1., 1., 0.]])
    terms = SSRv2HFRM28_1.spatial_terms(deep, raw, presence, classifier)
    teacher_grad, student_grad = torch.autograd.grad(
        terms["pcsd_loss"], (deep, raw), allow_unused=True
    )
    assert teacher_grad is None
    assert student_grad is not None and student_grad.abs().sum() > 0
    assert not terms["teacher_residual"].requires_grad
    assert not terms["discrepancy_detached"].requires_grad


def test_single_present_class_zeroes_both_mechanisms():
    deep = torch.randn(1, 4, 5, 5, requires_grad=True)
    raw = torch.randn(1, 4, 5, 5, requires_grad=True)
    classifier = torch.randn(4, 512, 1, 1)
    terms = SSRv2HFRM28_1.spatial_terms(
        deep, raw, torch.tensor([[1., 0., 0., 0.]]), classifier
    )
    assert terms["pcsd_loss"].item() == 0.0
    assert terms["teacher_residual"].count_nonzero().item() == 0
    assert not terms["valid_samples"].item()


def test_gamma_spatial_is_positive_only():
    module = SSRv2HFRM28_1()
    assert module.beta_spatial.item() == -4.0
    assert 0.018 < module.gamma_spatial.item() < 0.019
    for beta in (-100.0, -4.0, 0.0, 10.0):
        assert F.softplus(torch.tensor(beta)).item() > 0.0


def test_exactly_one_parameter_added_and_optimizer_coverage():
    baseline = SSHRNet(4)
    ssrv2 = SSRv2Net(4)
    baseline_names = set(baseline.state_dict())
    ssrv2_names = set(ssrv2.state_dict())
    assert ssrv2_names - baseline_names == {"hfrm_28_1.beta_spatial"}
    assert baseline_names - ssrv2_names == set()
    assert sum(p.numel() for p in ssrv2.parameters()) - sum(p.numel() for p in baseline.parameters()) == 1
    groups = ssrv2.get_parameter_groups()
    beta_id = id(ssrv2.hfrm_28_1.beta_spatial)
    assert sum(id(parameter) == beta_id for group in groups for parameter in group) == 1
    assert sum(id(parameter) == beta_id for parameter in groups[2]) == 1


def test_original_ch15_and_gsr_are_restored_without_bps():
    torch.manual_seed(9)
    baseline = SSHRNet(4)
    ssrv2 = SSRv2Net(4)
    ssrv2.hfrm_28_1.load_state_dict(baseline.hfrm_28_1.state_dict(), strict=False)
    assert torch.equal(ssrv2.hfrm_28_1.context_conv.weight, baseline.hfrm_28_1.context_conv.weight)
    assert ssrv2.hfrm_28_1.context_conv.kernel_size == (15, 15)
    names = set(dict(ssrv2.hfrm_28_1.named_parameters()))
    assert "beta_spatial" in names
    assert not any("rho" in name or "boundary" in name for name in names)
    assert not hasattr(ssrv2.hfrm_28_1, "teacher_head")
    assert not hasattr(ssrv2.hfrm_28_1, "student_head")


def test_model_and_loss_sources_have_no_dense_gt_or_bps():
    sources = "\n".join(
        (ROOT / name).read_text(encoding="utf-8").lower()
        for name in (
            "network/hfrm28_1_ssrv2.py",
            "network/resnet38_cls_ssrv2.py",
            "tools/train_ssrv2_25ep.py",
        )
    )
    for forbidden in (
        "segmentation_gt", "boundary_gt", "pseudo_mask", "rho_boundary",
        "semantic_boundary", "bps_gate",
    ):
        assert forbidden not in sources


def test_checkpoint_schedule_and_no_validation_selection():
    source = (ROOT / "tools" / "train_ssrv2_25ep.py").read_text(encoding="utf-8")
    assert '"checkpoint_epochs": (5, 10, 15, 20, 25)' in source
    assert '"validation_during_training": False' in source
    assert "best_val" not in source
    assert "epoch25 FINAL only" in source
