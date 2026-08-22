"""Architecture and frozen-protocol contracts for MATR-v1."""

from pathlib import Path

from network.resnet38_cls import Net as SSHRNet
from network.resnet38_cls_matr import Net as MATRNet


ROOT = Path(__file__).resolve().parents[1]


def test_parameter_budget_and_optimizer_raw_parameter_coverage():
    baseline = SSHRNet(4)
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    del baseline
    matr = MATRNet(4)
    matr_count = sum(parameter.numel() for parameter in matr.parameters())
    overhead = matr_count - baseline_count
    assert overhead > 0
    assert overhead / baseline_count < 0.001
    groups = matr.get_parameter_groups()
    for parameter in (
        matr.ic1.d_raw,
        matr.hfrm_28_1.sacr.a_logits,
        matr.hfrm_28_1.sacr.beta_adapt,
    ):
        assert sum(id(parameter) == id(item) for group in groups for item in group) == 1
        assert any(id(parameter) == id(item) for item in groups[2])


def test_only_two_frozen_innovations_and_no_dense_gt_routes():
    sources = "\n".join(
        (ROOT / path).read_text(encoding="utf-8").lower()
        for path in (
            "network/matr_multiprototype_head.py", "network/matr_sparse_context.py",
            "network/matr_hfrm28.py", "network/resnet38_cls_matr.py",
            "tools/matr_objectives.py",
        )
    )
    for forbidden in (
        "teacher", "cross-view", "pseudo_mask", "boundary detector",
        "contrastive", "orthogonality", "mamba", "kan",
    ):
        assert forbidden not in sources


def test_checkpoint_schedule_and_no_validation_selection():
    source = (ROOT / "tools" / "train_matr_25ep.py").read_text(encoding="utf-8")
    assert '"checkpoint_epochs": (5, 10, 15, 20, 25)' in source
    assert '"validation_during_training": False' in source
    assert '"segmentation_gt_used_in_training": False' in source
    assert "best_val" not in source
    assert "epoch25 FINAL only" in source


def test_official_inference_contract_is_frozen():
    source = (ROOT / "tool" / "infer_matr.py").read_text(encoding="utf-8")
    assert "[0.8, 0.9, 0.8, 0.6]" in source
    assert "0.6 * cams[\"28_1\"] + 0.2 * cams[\"28_2\"] + 0.2 * cams[\"deep\"]" in source
    assert "TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))" in source
