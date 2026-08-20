from __future__ import annotations

import numpy as np
import torch

from tools.crra_v0 import REPRESENTATIONS
from tools.crra_v0.extractor import presence_from_probability, rgr_coarse_proposal
from tools.crra_v0.probes import decide_crra, make_fold_assignments, run_oof_probe
from tools.crra_v0.regions import (
    component_records,
    core_rim_indices,
    deterministic_top_fraction,
    extract_image_regions,
    resize_gt_nearest,
    slide_id_from_image,
)


def test_exactly_three_frozen_representations():
    assert REPRESENTATIONS == ("whole", "core", "core_rim")


def test_slide_parser():
    assert slide_id_from_image("TCGA-A2-A0CV_xmin1_ymin2.png") == "TCGA-A2-A0CV"


def test_connected_components_are_classwise_and_8_connected():
    labels = np.array([[0, 1, 1], [2, 0, 1], [2, 2, 3]], dtype=np.uint8)
    records = component_records(labels)
    by_class = {item[0]: item[2] for item in records}
    assert set(by_class) == {0, 1, 2, 3}
    assert len(by_class[0]) == 2


def test_top_fraction_has_deterministic_index_tie_break():
    indices = np.asarray([8, 3, 5, 1], dtype=np.int64)
    values = np.ones(4)
    assert deterministic_top_fraction(indices, values).tolist() == [1]


def test_core_rim_is_partition_and_uses_outer_boundary():
    height = width = 7
    mask = np.zeros((height, width), dtype=bool)
    mask[1:6, 1:6] = True
    indices = np.flatnonzero(mask)
    feature = torch.ones(3, 25)
    token = feature.mean(1)
    core, rim = core_rim_indices(indices, feature, token, height, width)
    assert not np.intersect1d(core, rim).size
    assert np.array_equal(np.union1d(core, rim), indices)
    assert len(rim) >= 16
    assert len(core) >= 1


def test_nearest_gt_mapping_preserves_labels():
    gt = np.repeat(np.repeat(np.array([[0, 1], [2, 4]], dtype=np.uint8), 2, 0), 2, 1)
    assert np.array_equal(resize_gt_nearest(gt, (2, 2)), np.array([[0, 1], [2, 4]]))


def test_presence_fallback_and_gate_boundaries():
    probability = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.9, 0.95, 0.9, 0.7]])
    presence = presence_from_probability(probability)
    assert presence[0].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert presence[1].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_coarse_proposal_has_h28_shape_and_is_detached():
    base = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    proposal, presence = rgr_coarse_proposal(base, base + 1, base + 2, base + 3, torch.ones(1, 4))
    assert proposal.shape == (1, 4, 4)
    assert presence.shape == (1, 4)
    assert not proposal.requires_grad


def test_region_dump_aligns_three_tokens_on_common_support():
    proposal = np.zeros((5, 5), dtype=np.uint8)
    gt = np.zeros((10, 10), dtype=np.uint8)
    feature = torch.arange(4 * 25, dtype=torch.float32).reshape(4, 5, 5)
    rows, arrays, stats = extract_image_regions(
        proposal, gt, feature, "TCGA-A2-A0CV_xmin1_ymin2", 0
    )
    assert stats["proposed_regions"] == 1
    assert len(rows) == 1 and rows[0]["common_support"]
    assert rows[0]["taxonomy"] == "Type-A"
    assert arrays["z_whole"].shape == (1, 4)
    assert arrays["z_core"].shape == arrays["z_rim"].shape == (1, 4)


def test_groupkfold_has_no_slide_leakage_and_is_shared():
    labels = np.tile(np.arange(5), 10)
    groups = np.repeat([f"slide-{index}" for index in range(10)], 5)
    folds, manifest = make_fold_assignments(labels, groups)
    assert set(folds.tolist()) == set(range(5))
    assert len(manifest) == 5
    for item in manifest:
        assert not set(item["train_slides"]) & set(item["held_out_slides"])


def test_fixed_probe_runs_with_prespecified_configuration():
    labels = np.tile(np.arange(5), 10)
    groups = np.repeat([f"slide-{index}" for index in range(10)], 5)
    features = np.eye(5, dtype=np.float32)[labels]
    folds, _ = make_fold_assignments(labels, groups)
    predictions, metrics, fold_results = run_oof_probe(features, labels, groups, folds)
    assert np.array_equal(predictions, labels)
    assert metrics["macro_f1"] == 1.0
    assert len(fold_results) == 5


def _decision_inputs(core_f1=0.56, dual_f1=0.55, type_b_core=0.50, type_b_dual=0.49):
    def metric(value):
        return {
            "macro_f1": value,
            "per_class_f1": {str(index): value for index in range(5)},
        }
    metrics = {"whole": metric(0.50), "core": metric(core_f1), "core_rim": metric(dual_f1)}
    type_b = {"whole": 0.40, "core": type_b_core, "core_rim": type_b_dual}
    type_a = {"whole": 0.90, "core": 0.89, "core_rim": 0.89}
    folds = {
        "whole": [{"macro_f1": 0.50} for _ in range(5)],
        "core": [{"macro_f1": core_f1} for _ in range(5)],
        "core_rim": [{"macro_f1": dual_f1} for _ in range(5)],
    }
    bootstrap = {
        "core_minus_whole": {"ci95_low": 0.02, "ci95_high": 0.10},
        "core_rim_minus_whole": {"ci95_low": 0.01, "ci95_high": 0.09},
    }
    return metrics, type_b, type_a, folds, bootstrap


def test_strong_go_and_core_only_flag_are_threshold_driven():
    metrics, type_b, type_a, folds, bootstrap = _decision_inputs()
    decision = decide_crra(metrics, type_b, type_a, folds, 0.9, bootstrap)
    assert decision["decision"] == "CRRA_V0_STRONG_GO"
    assert decision["representation_flag"] == "CORE_ONLY_PREFERRED"


def test_hard_nogo_closes_route():
    metrics, type_b, type_a, folds, bootstrap = _decision_inputs(
        core_f1=0.495, dual_f1=0.49, type_b_core=0.39, type_b_dual=0.38
    )
    decision = decide_crra(metrics, type_b, type_a, folds, 0.9, bootstrap)
    assert decision["decision"] == "CRRA_V0_NOGO"
    assert decision["representation_flag"] == "REGION_REPRESENTATION_ROUTE_CLOSED"
