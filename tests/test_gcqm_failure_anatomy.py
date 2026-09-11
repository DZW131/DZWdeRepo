import numpy as np
from pathlib import Path

from tools.audit_gcqm_full25_failure_anatomy import (
    BANDS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    TOPK,
    binary_metrics,
    boundary_band,
    fp_fn_decision,
    mask_morphology,
    weight_metrics,
)


def test_frozen_anatomy_protocol():
    assert BANDS == (1, 3, 5)
    assert TOPK == (1, 3, 5, 10, 20, 50, 100, 196)
    assert BOOTSTRAP_SEED == 20260911 and BOOTSTRAP_RESAMPLES == 10_000


def test_binary_error_decomposition():
    truth = np.asarray([[1, 1], [0, 0]], dtype=bool)
    prediction = np.asarray([[1, 0], [1, 0]], dtype=bool)
    result = binary_metrics(truth, prediction, np.ones_like(truth))
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)


def test_boundary_and_morphology_are_deterministic():
    mask = np.zeros((9, 9), dtype=bool); mask[2:7, 2:7] = True; mask[4, 4] = False
    assert boundary_band(mask, 3).dtype == bool
    result = mask_morphology(mask, 25)
    assert result["components"] == 1 and result["hole_count"] == 1
    assert result["hole_area_fraction"] > 0


def test_diffuse_uniform_weights():
    result = weight_metrics(np.ones(196))
    assert abs(result["effective_query_count"] - 196) < 1e-4
    assert result["diffuseness_bin"] == "very_diffuse"
    assert abs(result["top5_mass"] - 5 / 196) < 1e-6


def test_audit_contains_no_training_path():
    source = (Path(__file__).resolve().parents[1] / "tools/audit_gcqm_full25_failure_anatomy.py").read_text(encoding="utf-8")
    assert ".backward(" not in source and "torch.optim" not in source
    assert "gcqm_full25_epoch25_final.pth" not in source


def test_fp_reduction_is_not_mislabeled_as_fp_failure():
    assert fp_fn_decision(-2.8, .02, [-7.8, .2], [.01, .03]) == "FN_DOMINANT"
    assert fp_fn_decision(-.2, -.1, [-.4, .1], [-.2, -.01]) == "MIXED"
