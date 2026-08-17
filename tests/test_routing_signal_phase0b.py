"""Frozen-contract tests for Phase-0B routing-signal learnability."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from tool.infer_fun import _tta_transforms
from tools.routing_signal_audit import (
    BASELINE_COMMIT,
    BCSS_THRESHOLDS,
    BOOTSTRAP_REPLICATES,
    CHECKPOINT_SHA256,
    FOLD_SEED,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_DIM,
    MLP_LR,
    MLP_WEIGHT_DECAY,
    OFFICIAL_FUSION,
    PCA_DIMENSIONS,
    PHASE0_PARENT_COMMIT,
    RIDGE_ALPHA,
    ROUTING_THRESHOLD,
    SAFE_CANDIDATES,
)
from tools.routing_signal_audit.cache_validation import load_phase0_assignment
from tools.routing_signal_audit.metrics import (
    candidate_utilities,
    frozen_primary_decision,
    present_class_mean_iou,
    route_from_relative_predictions,
    routing_diagnostics,
)
from tools.routing_signal_audit.oracle_image_fusion import image_fusion_grid
from tools.routing_signal_audit.oracle_local_class import local_class_combinations
from tools.routing_signal_audit.probe_mlp import TinyRelativeUtilityMLP
from tools.routing_signal_audit.signal_cam import (
    _morphology,
    _softmax,
    _spatial_features,
    extract_cam_signals,
)
from tools.routing_signal_audit.signal_feature import (
    compose_signal_c,
    fit_fold_pca_context,
)
from tools.routing_signal_audit.signal_tta import (
    STAGE_NAMES,
    extract_tta_and_feature_signals,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE0_RESULTS = ROOT / "audit" / "results" / "decision_bottleneck_phase0"


def _cli_options():
    tree = ast.parse((ROOT / "tools" / "audit_routing_signal_learnability.py").read_text(encoding="utf-8"))
    return [
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]


def test_01_cli_is_frozen_and_contains_only_audit_paths():
    assert _cli_options() == [
        "--phase0-dir",
        "--val-root",
        "--checkpoint",
        "--output-dir",
        "--num-workers",
    ]


def test_02_cli_exposes_no_test_or_luad_path():
    assert not any("test" in option or "luad" in option for option in _cli_options())


def test_03_baseline_commit_is_frozen():
    assert BASELINE_COMMIT == "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"


def test_04_phase0_parent_commit_is_frozen():
    assert PHASE0_PARENT_COMMIT == "f1a95059cd7914e9d6b72e08ec135c4c8ea32c06"


def test_05_checkpoint_sha_is_frozen():
    assert CHECKPOINT_SHA256 == "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"


def test_06_official_fusion_and_thresholds_are_unchanged():
    assert OFFICIAL_FUSION == (0.0, 0.6, 0.2, 0.2)
    assert BCSS_THRESHOLDS == (0.8, 0.9, 0.8, 0.6)


def test_07_official_tta_is_unchanged():
    assert _tta_transforms() == (((), ()), ((3,), (2,)), ((2,), (1,)))


def test_08_safe_candidates_include_official_first():
    assert SAFE_CANDIDATES == (
        "official_fusion",
        "cam56",
        "cam28_1",
        "cam28_2",
        "camdeep",
    )


def test_09_image_fusion_grid_has_exactly_286_candidates():
    assert image_fusion_grid().shape == (286, 4)


def test_10_image_fusion_grid_is_a_nonnegative_simplex():
    weights = image_fusion_grid()
    assert np.all(weights >= 0)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_11_image_fusion_tie_order_puts_official_first():
    assert np.array_equal(
        image_fusion_grid()[0], np.asarray(OFFICIAL_FUSION, dtype=np.float32)
    )


def test_12_local_imageclass_enumeration_has_625_candidates():
    assert local_class_combinations().shape == (625, 4)


def test_13_local_imageclass_tie_order_puts_all_official_first():
    assert np.array_equal(local_class_combinations()[0], np.zeros(4, dtype=np.int8))


def test_14_phase0_assignment_is_reused_exactly():
    folds, rows, digest = load_phase0_assignment(PHASE0_RESULTS)
    assert len(rows) == 3418 and folds.shape == (3418,)
    assert len(digest) == 64


def test_15_phase0_assignment_has_22_nonleaking_slides():
    _, rows, _ = load_phase0_assignment(PHASE0_RESULTS)
    group_folds = {}
    for row in rows:
        group_folds.setdefault(row["source_group"], set()).add(row["fold"])
    assert len(group_folds) == 22
    assert all(len(folds) == 1 for folds in group_folds.values())


def test_16_relative_routing_uses_strictly_positive_threshold():
    values = np.asarray([[0.0, -1.0, -2.0, -3.0], [0.01, 0.0, 0.0, 0.0]])
    assert route_from_relative_predictions(values).tolist() == [-1, 0]


def test_17_relative_routing_tie_order_is_branch_order():
    values = np.asarray([[0.2, 0.2, 0.1, 0.0]])
    assert route_from_relative_predictions(values).tolist() == [0]


def test_18_present_class_utility_ignores_absent_gt_classes():
    truth = np.zeros((2, 2), dtype=np.uint8)
    prediction = np.asarray([[0, 0], [0, 1]], dtype=np.uint8)
    assert present_class_mean_iou(prediction, truth) == pytest.approx(0.75)


def test_19_candidate_utilities_include_official_and_four_branches():
    truth = np.zeros((2, 2), dtype=np.uint8)
    official = np.zeros_like(truth)
    branches = np.stack([official] * 4)
    assert candidate_utilities(truth, official, branches).shape == (5,)


def test_20_signal_a_extractor_has_no_gt_argument():
    assert "gt" not in inspect.signature(extract_cam_signals).parameters
    assert "truth" not in inspect.signature(extract_cam_signals).parameters


def test_21_signal_bc_extractor_has_no_gt_argument():
    parameters = inspect.signature(extract_tta_and_feature_signals).parameters
    assert "gt" not in parameters and "truth" not in parameters


def test_22_signal_sources_do_not_use_slide_id_as_feature():
    source = "\n".join(
        (ROOT / "tools" / "routing_signal_audit" / name).read_text(encoding="utf-8")
        for name in ("signal_cam.py", "signal_tta.py", "signal_feature.py")
    )
    assert "slide_id" not in source and "patient_id" not in source


def test_23_signal_softmax_is_finite_and_normalized():
    probability = _softmax(np.zeros((4, 3, 3), dtype=np.float32))
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=0), 1.0)


def test_24_spatial_concentration_is_finite_for_zero_cam():
    assert _spatial_features(np.zeros((4, 4), dtype=np.float32)) == (0.0, 0.0, 0.0)


def test_25_morphology_uses_zero_plus_presence_for_absent_class():
    assert _morphology(np.zeros((4, 4), dtype=bool)) == [0.0] * 6


def test_26_feature_context_has_four_frozen_stages():
    assert STAGE_NAMES == ("h56", "h28_1", "h28_2", "fdeep")


def test_27_pca_dimension_is_fixed_at_16():
    assert PCA_DIMENSIONS == 16


def test_28_fold_pca_fits_only_train_rows():
    rng = np.random.default_rng(7)
    gaps = {stage: rng.normal(size=(12, 20)).astype(np.float32) for stage in STAGE_NAMES}
    train = np.arange(8)
    heldout = np.arange(8, 12)
    train_context, heldout_context, _, rows = fit_fold_pca_context(gaps, train, heldout)
    assert train_context.shape == (8, 64)
    assert heldout_context.shape == (4, 64)
    assert all(row["fit_images"] == 8 and row["fit_scope"] == "train_fold_only" for row in rows)


def test_29_signal_c_composition_repeats_only_image_context():
    signal_b = np.zeros((3, 4, 2), dtype=np.float32)
    scalar = np.arange(9, dtype=np.float32).reshape(3, 3)
    pca = np.arange(6, dtype=np.float32).reshape(3, 2)
    result = compose_signal_c(signal_b, scalar, pca, np.arange(3))
    assert result.shape == (3, 4, 7)
    assert np.array_equal(result[:, 0, 2:], result[:, 3, 2:])


def test_30_ridge_alpha_is_fixed():
    assert RIDGE_ALPHA == 1.0


def test_31_mlp_contract_is_fixed():
    assert (MLP_HIDDEN_DIM, MLP_EPOCHS, MLP_LR, MLP_WEIGHT_DECAY, MLP_BATCH_SIZE) == (
        32,
        200,
        1e-3,
        0.0,
        256,
    )


def test_32_mlp_has_exactly_two_linear_layers_and_one_relu():
    model = TinyRelativeUtilityMLP(10)
    assert [type(module) for module in model.network] == [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]


def test_33_mlp_output_is_finite_scalar_per_candidate():
    output = TinyRelativeUtilityMLP(5)(torch.zeros(7, 5))
    assert output.shape == (7,) and torch.isfinite(output).all()


def test_34_seed_threshold_and_bootstrap_are_frozen():
    assert FOLD_SEED == 20260817
    assert ROUTING_THRESHOLD == 0.0
    assert BOOTSTRAP_REPLICATES == 2000


@pytest.mark.parametrize(
    "inputs,expected",
    [
        ((0.50, 0.25, 4, 0.01), "ROUTING_SIGNAL_STRONG_GO"),
        ((0.30, 0.15, 3, -0.01), "ROUTING_SIGNAL_GO"),
        ((0.20, 0.50, 5, 0.10), "ROUTING_SIGNAL_WEAK_REVIEW"),
        ((0.40, 0.05, 2, -0.10), "ROUTING_SIGNAL_WEAK_REVIEW"),
        ((0.09, 1.00, 5, 1.00), "ROUTING_SIGNAL_NOGO"),
        ((-0.10, -0.05, 0, -1.00), "ROUTING_SIGNAL_NOGO"),
    ],
)
def test_35_to_40_frozen_decision_hierarchy(inputs, expected):
    assert frozen_primary_decision(*inputs)[0] == expected


def test_41_routing_diagnostics_are_finite():
    predicted = np.asarray([[0.1, -0.1, 0.0, 0.0], [-0.1, -0.2, -0.3, -0.4]])
    truth = np.asarray([[0.2, -0.2, 0.0, 0.0], [0.0, -0.1, -0.2, -0.3]])
    diagnostics = routing_diagnostics(predicted, truth, np.asarray([0, -1]))
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_42_no_forbidden_baseline_source_diff():
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            PHASE0_PARENT_COMMIT,
            "--",
            "network",
            "train_sshr.py",
            "tool/infer_fun.py",
            "tool/iouutils.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


def test_43_no_formal_training_entrypoint_is_called():
    source = (ROOT / "tools" / "audit_routing_signal_learnability.py").read_text(encoding="utf-8")
    assert "train_sshr" not in source


def test_44_no_threshold_sweep_is_implemented():
    source = (ROOT / "tools" / "audit_routing_signal_learnability.py").read_text(encoding="utf-8")
    assert "threshold_sweep" in source
    assert "ROUTING_THRESHOLD" in source
    assert "--threshold" not in source


def test_45_report_uses_exact_local_diagnostic_wording():
    source = (ROOT / "tools" / "routing_signal_audit" / "report.py").read_text(encoding="utf-8")
    assert "Exact Local Image×Class Oracle" in source
    assert "strict global upper bound" not in source
