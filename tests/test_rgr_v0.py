from pathlib import Path
from types import MethodType

import numpy as np
import pytest
import torch

from network.resnet38_cls_rgr import Net
from network.rgr_v0 import RGRRefinement, _edge_geometry
from tool.infer_rgr_v0_paired import _node_group, _validate_scope
from tools.run_rgr_v0 import (
    BATCH_SIZE,
    IMAGE_SIZE,
    PILOT_EPOCHS,
    READINESS_STEPS,
    build_optimizer,
    frozen_mode_ok,
    set_frozen_training_mode,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixed_two_region_proposal(self, cam_56, cam_28_1, cam_28_2, cam_deep, presence):
    batch, _, height, width = cam_28_1.shape
    labels = torch.zeros((batch, height, width), dtype=torch.long, device=cam_28_1.device)
    labels[:, :, width // 2:] = 1
    return labels


def _inputs(batch=2, height=8, width=8):
    torch.manual_seed(7)
    feature = torch.randn(batch, 512, height, width)
    cams = [torch.randn(batch, 4, height, width) for _ in range(4)]
    presence = torch.ones(batch, 4)
    return feature, cams, presence


def test_zero_init_exact_identity_and_deterministic_structures():
    module = RGRRefinement()
    module.proposal_labels = MethodType(_fixed_two_region_proposal, module)
    feature, cams, presence = _inputs()
    first = module(feature, *cams, presence, collect_structures=True)
    second = module(feature, *cams, presence, collect_structures=True)
    assert torch.equal(first.refined_cam, cams[1])
    assert torch.count_nonzero(first.delta_iso).item() == 0
    assert torch.count_nonzero(first.delta_graph).item() == 0
    assert first.structures == second.structures
    assert first.per_image_region_counts == [2, 2]


def test_complete_directed_graph_has_no_self_edge_and_finite_features():
    masks = []
    for column in (0, 2, 4):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:4, column:column + 2] = 1
        masks.append(mask)
    tokens = torch.randn(3, 512)
    centroids = torch.tensor([[2.0, 0.5], [2.0, 2.5], [2.0, 4.5]])
    classes = torch.tensor([0, 1, 1])
    source, target, features = _edge_geometry(
        masks, centroids, tokens, classes, 6, 6
    )
    assert source.numel() == 3 * 2
    assert torch.all(source != target)
    assert torch.isfinite(features).all()
    assert features.shape == (6, 4)
    assert torch.all((features[:, 1] >= 0) & (features[:, 1] <= 1))
    assert set(features[:, 2].tolist()).issubset({0.0, 1.0})
    assert set(features[:, 3].tolist()).issubset({0.0, 1.0})


def test_touch_flag_detects_dilated_contact():
    left = np.zeros((5, 5), dtype=np.uint8)
    right = np.zeros((5, 5), dtype=np.uint8)
    far = np.zeros((5, 5), dtype=np.uint8)
    left[2, 1] = 1
    right[2, 2] = 1
    far[2, 4] = 1
    tokens = torch.randn(3, 512)
    centroids = torch.tensor([[2.0, 1.0], [2.0, 2.0], [2.0, 4.0]])
    source, target, features = _edge_geometry(
        [left, right, far], centroids, tokens, torch.tensor([0, 1, 2]), 5, 5
    )
    edge_touch = {
        (int(target[index]), int(source[index])): int(features[index, 2].item())
        for index in range(source.numel())
    }
    assert edge_touch[(0, 1)] == 1
    assert edge_touch[(0, 2)] == 0


def test_single_node_has_exact_zero_message_and_graph_residual():
    module = RGRRefinement()
    token = torch.randn(1, 512)
    base = torch.randn(1, 4)
    mask = np.ones((8, 8), dtype=np.uint8)
    iso, graph, _, source, target, stats = module._graph_reasoning(
        token, base, [mask], torch.tensor([[3.5, 3.5]]), 8, 8
    )
    assert iso.shape == graph.shape == (1, 4)
    assert torch.count_nonzero(graph).item() == 0
    assert source.numel() == target.numel() == 0
    assert stats["message_norm"] == 0.0


def test_output_heads_zero_but_internal_graph_is_normally_initialized():
    module = RGRRefinement()
    assert torch.count_nonzero(module.isolated_head.weight).item() == 0
    assert torch.count_nonzero(module.isolated_head.bias).item() == 0
    assert torch.count_nonzero(module.graph_head.weight).item() == 0
    assert torch.count_nonzero(module.graph_head.bias).item() == 0
    assert torch.count_nonzero(module.node_projection.weight).item() > 0
    assert torch.count_nonzero(module.value_projection.weight).item() > 0
    assert torch.count_nonzero(module.message_projection.weight).item() > 0


def test_graph_upstream_opens_after_zero_output_head_moves():
    module = RGRRefinement()
    module.proposal_labels = MethodType(_fixed_two_region_proposal, module)
    feature, cams, presence = _inputs(batch=1)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    upstream_seen = {name: False for name in (
        "node_projection", "edge_gate", "value_projection", "message_projection"
    )}
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        result = module(feature, *cams, presence)
        target = torch.randn_like(result.refined_cam)
        loss = (result.refined_cam - target).square().mean()
        loss.backward()
        if step == 0:
            assert module.isolated_head.weight.grad.norm().item() > 0
            assert module.graph_head.weight.grad.norm().item() > 0
        for name in upstream_seen:
            gradients = [
                parameter.grad for parameter in getattr(module, name).parameters()
                if parameter.grad is not None
            ]
            upstream_seen[name] |= bool(
                gradients and sum(gradient.abs().sum().item() for gradient in gradients) > 0
            )
        optimizer.step()
    assert all(upstream_seen.values())
    assert result.statistics["message_norm"] > 0
    assert torch.isfinite(result.refined_cam).all()


def test_parameter_overhead_is_below_one_percent():
    model = Net(n_class=4)
    base = sum(
        parameter.numel() for name, parameter in model.named_parameters()
        if not name.startswith("rgr.")
    )
    assert model.rgr.trainable_parameter_count() / base < 0.01


def test_frozen_mode_and_optimizer_cover_exactly_rgr():
    model = Net(n_class=4)
    set_frozen_training_mode(model)
    assert all(frozen_mode_ok(model).values())
    optimizer = build_optimizer(model, max_steps=32)
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in model.rgr.parameters()}


def test_hard_stage_limits_and_forbidden_scope(tmp_path):
    assert READINESS_STEPS == 32
    assert PILOT_EPOCHS == 3
    assert BATCH_SIZE == 20
    assert IMAGE_SIZE == 224
    source = (ROOT / "tools" / "run_rgr_v0.py").read_text(encoding="utf-8")
    assert 'add_argument("--epochs"' not in source
    assert '"test_accessed": False' in source
    assert '"luad_accessed": False' in source
    assert '"auto_25epoch": False' in source
    with pytest.raises(ValueError):
        _validate_scope(tmp_path / "test", ("base", "full"))
    with pytest.raises(ValueError):
        _validate_scope(tmp_path / "LUAD" / "val", ("base", "full"))


def test_four_way_paired_evaluator_and_official_fusion_are_locked():
    source = (ROOT / "tool" / "infer_rgr_v0_paired.py").read_text(encoding="utf-8")
    assert 'VARIANTS = ("base", "isolated", "graph_only", "full")' in source
    assert "0.6 * normalized_28_1 + 0.2 * normalized_28_2 + 0.2 * normalized_deep" in source
    assert "(((), ()), ((3,), (2,)), ((2,), (1,)))" in source
    assert '"optimizer_updates_during_evaluation": 0' in source


def test_node_count_groups_are_frozen():
    assert _node_group(1) == "N=1"
    assert _node_group(2) == "N=2"
    assert _node_group(3) == "N=3-4"
    assert _node_group(4) == "N=3-4"
    assert _node_group(5) == "N>=5"


def test_no_rsbr_transition_or_extended_graph_modules_exist():
    model_source = (ROOT / "network" / "rgr_v0.py").read_text(encoding="utf-8")
    assert "transition_head" not in model_source
    assert "Transformer" not in model_source
    assert "GAT" not in model_source
    assert model_source.count("message_projection") >= 1


def test_disposable_preflight_is_three_step_bf16_real_batch_contract():
    source = (ROOT / "tools" / "preflight_rgr_v0.py").read_text(encoding="utf-8")
    assert "PREFLIGHT_STEPS = 3" in source
    assert "sys.path.insert(0, str(ROOT))" in source
    assert '"batch_size_20"' in source
    assert '"upstream_gradients_by_step3"' in source
    assert '"frozen_base_unchanged"' in source
    assert "RGR_V0_PREFLIGHT_PASS" in source
