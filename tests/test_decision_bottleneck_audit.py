import ast
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
import torch

from tool import iouutils
from tools.audit_decision_bottleneck import frozen_decision
from tools.decision_audit import BCSS_THRESHOLDS, OFFICIAL_FUSION
from tools.decision_audit.cam_cache import (
    EXPECTED_CHECKPOINT_SHA256,
    source_group,
)
from tools.decision_audit.class_probe import ClassConditionedLinearProbe
from tools.decision_audit.fusion import (
    normalize_cam,
    official_score_from_hist,
    prediction_from_scores,
    score_predictions,
    simplex_weights,
)
from tools.decision_audit.oracle import (
    _binary_class_iou,
    _present_class_mean_iou,
)


ROOT = Path(__file__).resolve().parents[1]


def cli_options():
    tree = ast.parse((ROOT / "tools" / "audit_decision_bottleneck.py").read_text())
    options = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument" and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    options.append(value.value)
    return options


def test_cli_has_only_frozen_non_test_options():
    assert set(cli_options()) == {
        "--val-root",
        "--checkpoint",
        "--output-dir",
        "--num-workers",
    }


def test_frozen_checkpoint_fusion_and_threshold_constants():
    assert EXPECTED_CHECKPOINT_SHA256 == (
        "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
    )
    assert OFFICIAL_FUSION == (0.0, 0.6, 0.2, 0.2)
    assert BCSS_THRESHOLDS == (0.8, 0.9, 0.8, 0.6)


def test_normalized_cams_are_finite_and_bounded():
    cam = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 5.0], [5.0, 5.0]],
            [[-3.0, 0.0], [1.0, 9.0]],
            [[0.2, 0.3], [0.4, 0.5]],
        ],
        dtype=np.float32,
    )
    normalized = normalize_cam(cam)
    assert np.isfinite(normalized).all()
    assert normalized.min() >= 0
    assert normalized.max() <= 1


def test_presence_gating_and_foreground_argmax_match_contract():
    scores = np.zeros((4, 2, 2), dtype=np.float32)
    scores[0] = 0.9
    scores[1] = 1.0
    scores[2] = 0.8
    prediction = prediction_from_scores(scores, np.array([1, 0, 1, 0]))
    assert np.all(prediction == 0)
    assert prediction.dtype == np.uint8


def test_streaming_official_metric_matches_released_iouutils():
    ground_truth = np.asarray(
        [
            [[0, 0, 1], [2, 3, 4]],
            [[3, 2, 1], [0, 4, 4]],
        ],
        dtype=np.uint8,
    )
    prediction = np.asarray(
        [
            [[0, 1, 1], [2, 0, 3]],
            [[3, 0, 1], [0, 2, 1]],
        ],
        dtype=np.uint8,
    )
    released = iouutils.scores(
        [value.copy() for value in ground_truth],
        [value.copy() for value in prediction],
        n_class=4,
    )
    audit = score_predictions(ground_truth, prediction)
    assert audit["Mean IoU"] == released["Mean IoU"]
    assert audit["Mean Dice"] == released["Mean Dice"]
    assert audit["Class IoU"] == released["Class IoU"]
    assert audit["Dice Coefficients"] == released["Dice Coefficients"]


def test_official_hist_background_overwrite_algebra():
    hist = np.zeros((5, 5), dtype=np.float64)
    hist[0, 0] = 2
    hist[1, 1] = 3
    hist[4, 4] = 100
    score = official_score_from_hist(hist)
    assert score["Pixel Accuracy"] == 1.0


def test_frozen_simplex_grid_is_complete_and_valid():
    weights = simplex_weights(0.05)
    assert weights.shape == (1771, 4)
    assert np.all(weights >= 0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.sum(np.all(np.isclose(weights, OFFICIAL_FUSION), axis=1)) == 1


def test_source_group_is_recovered_without_patch_leakage():
    first = "TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0"
    second = "TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+101"
    assert source_group(first) == "TCGA-EW-A1PB-DX1"
    assert source_group(second) == "TCGA-EW-A1PB-DX1"


def test_group_kfold_is_deterministic_and_has_no_group_leakage():
    groups = np.asarray([f"slide{i // 3}" for i in range(30)])
    indices = np.arange(len(groups))
    first = list(GroupKFold(5).split(indices, groups=groups))
    second = list(GroupKFold(5).split(indices, groups=groups))
    for (train_a, held_a), (train_b, held_b) in zip(first, second):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(held_a, held_b)
        assert not (set(groups[train_a]) & set(groups[held_a]))


def test_probe_has_exactly_16_scalars_and_normalized_class_weights():
    probe = ClassConditionedLinearProbe()
    assert sum(parameter.numel() for parameter in probe.parameters()) == 16
    weights = probe.weights().detach()
    assert torch.allclose(weights.sum(dim=0), torch.ones(4))
    assert torch.all(weights >= 0)


def test_probe_forward_has_no_cross_class_mixing():
    probe = ClassConditionedLinearProbe()
    cams = torch.zeros(1, 4, 4, 2, 2)
    cams[:, :, 2] = 1.0
    output = probe(cams)
    assert torch.equal(output[:, 0], torch.zeros_like(output[:, 0]))
    assert torch.equal(output[:, 1], torch.zeros_like(output[:, 1]))
    assert torch.allclose(output[:, 2], torch.ones_like(output[:, 2]))
    assert torch.equal(output[:, 3], torch.zeros_like(output[:, 3]))


def test_image_oracle_metrics_select_only_existing_branch_predictions():
    truth = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
    bad = np.asarray([[1, 1], [0, 0]], dtype=np.uint8)
    good = truth.copy()
    scores = [_present_class_mean_iou(value, truth) for value in (bad, good)]
    assert int(np.argmax(scores)) == 1
    assert _binary_class_iou(good, truth, 0) == 1.0


def test_frozen_decision_precedence_and_grey_zone():
    assert frozen_decision(0.5, 1.5, 2.0, True)[0] == (
        "DECISION_BOTTLENECK_STRONG_GO"
    )
    assert frozen_decision(0.3, 1.0, 1.0, False)[0] == "DECISION_BOTTLENECK_GO"
    assert frozen_decision(0.1, 1.5, 2.0, False)[0] == "NONLINEAR_ROUTING_REVIEW"
    assert frozen_decision(0.0, 0.4, 1.5, False)[0] == "SPATIAL_ROUTING_SIGNAL"
    assert frozen_decision(0.0, 0.4, 1.4, False)[0] == "DECISION_BOTTLENECK_NOGO"
    assert frozen_decision(0.2, 0.8, 1.0, False)[0] == "NONLINEAR_ROUTING_REVIEW"


def test_analysis_source_does_not_import_training_entrypoint():
    source = (ROOT / "tools" / "audit_decision_bottleneck.py").read_text()
    assert "train_sshr" not in source
    assert "--test-root" not in cli_options()
