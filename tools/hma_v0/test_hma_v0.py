"""Focused unit tests for HMA-v0 audit primitives."""

import numpy as np
import torch

from network.resnet38_cls import HFRM
from tool import iouutils
from tools.hma_v0.instrumentation import HMAAuditNet, presence_from_probability
from tools.hma_v0.gradient_audit import parameter_inventory
from tools.hma_v0.kernels import channel_kernel_metrics
from tools.hma_v0.metrics import (
    OfficialMetricAccumulator,
    foreground_boundary_distance,
    prediction_from_fusion,
)


def test_hfrm_full_variant_is_exact_module_equation():
    torch.manual_seed(7)
    module = HFRM(8, deep_channels=16, context_kernel=3).eval()
    module.gamma_veto.data.fill_(0.17)
    module.gamma_context.data.fill_(-0.08)
    feature = torch.randn(2, 8, 9, 9)
    deep = torch.randn(2, 16, 3, 3)
    _, _, _, variants = HMAAuditNet._stage_variants(module, feature, deep)
    assert torch.equal(module(feature, deep), variants["full"])


def test_zero_gamma_variants_recover_raw():
    module = HFRM(8, deep_channels=16, context_kernel=3).eval()
    feature = torch.randn(2, 8, 9, 9)
    deep = torch.randn(2, 16, 3, 3)
    _, _, _, variants = HMAAuditNet._stage_variants(module, feature, deep)
    for variant in ("raw", "gsr", "ch", "full"):
        assert torch.equal(feature, variants[variant])


def test_presence_threshold_and_fallback():
    probability = torch.tensor([[0.81, 0.91, 0.79, 0.61], [0.1, 0.2, 0.3, 0.4]])
    result = presence_from_probability(probability)
    assert result[0].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert result[1].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_fusion_uses_frozen_weights_and_gate():
    base = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
    prediction, response = prediction_from_fusion(base, base[::-1].copy(), base, [1, 0, 1, 0])
    assert response.shape == (4, 2, 2)
    assert set(np.unique(prediction)).issubset({0, 2})


def test_streaming_metric_matches_released_iouutils():
    truth = [
        np.asarray([[0, 1, 4], [2, 3, 4]], dtype=np.uint8),
        np.asarray([[3, 2, 1], [0, 4, 4]], dtype=np.uint8),
    ]
    prediction = [
        np.asarray([[0, 2, 1], [2, 0, 1]], dtype=np.uint8),
        np.asarray([[3, 1, 1], [2, 2, 0]], dtype=np.uint8),
    ]
    released = iouutils.scores(truth, [item.copy() for item in prediction], n_class=4)
    accumulator = OfficialMetricAccumulator()
    for gt, pred in zip(truth, prediction):
        accumulator.update(gt, pred)
    observed = accumulator.scores()
    assert observed["mean_iou"] == released["Mean IoU"]
    assert observed["mean_dice"] == released["Mean Dice"]
    assert np.allclose(list(observed["class_iou"].values()), list(released["Class IoU"].values()))


def test_boundary_bins_partition_foreground():
    truth = np.asarray([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 4, 4]], dtype=np.uint8)
    bins = foreground_boundary_distance(truth)
    counts = sum(bins[name].astype(np.uint8) for name in ("B0_le_2", "B1_3_7", "B2_ge_8"))
    assert np.array_equal(counts > 0, truth < 4)
    assert counts.max() == 1


def test_uniform_kernel_is_homogenizer_like():
    kernels = np.full((3, 1, 15, 15), 1.0 / 225.0, dtype=np.float32)
    rows = channel_kernel_metrics("56", kernels)
    assert all(row["uniform_cosine"] > 0.999999 for row in rows)
    assert all(row["negative_fraction"] == 0.0 for row in rows)
    assert all(abs(row["dc_gain"] - 1.0) < 1e-6 for row in rows)


def test_gradient_inventory_excludes_officially_frozen_parameters():
    model = HMAAuditNet(4)
    model.eval()
    items, groups = parameter_inventory(model)
    assert items
    assert all(parameter.requires_grad for _, parameter, _ in items)
    assert all(groups[name] for name in groups)
