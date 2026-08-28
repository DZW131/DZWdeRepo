import numpy as np

from tools.analyze_rddr_phase1 import TransitionAccumulator
from tools.rddr_phase1_analysis_common import (
    foreground_boundary_distance,
    official_histogram,
    paired_bootstrap_miou,
    scores_from_histogram,
)


def test_official_background_overwrite():
    truth = np.asarray([[0, 1, 4]], dtype=np.uint8)
    prediction = np.asarray([[0, 0, 2]], dtype=np.uint8)
    histogram = official_histogram(truth, prediction)
    assert histogram[0, 0] == 1
    assert histogram[1, 0] == 1
    assert histogram[4, 4] == 1
    score = scores_from_histogram(histogram)
    assert score["histogram"][4][4] == 0


def test_boundary_interior_partition_foreground():
    truth = np.zeros((32, 32), dtype=np.uint8)
    truth[:, 16:] = 1
    truth[:2, :2] = 4
    zones = foreground_boundary_distance(truth)
    boundary = zones["boundary_le_7"]
    interior = zones["interior_gt_7"]
    assert not np.any(boundary & interior)
    assert np.array_equal(boundary | interior, truth < 4)


def test_transition_repair_harm():
    truth = np.asarray([[0, 0, 1, 1]], dtype=np.uint8)
    baseline = np.asarray([[1, 0, 1, 1]], dtype=np.uint8)
    candidate = np.asarray([[0, 1, 1, 0]], dtype=np.uint8)
    accumulator = TransitionAccumulator(("Top20",))
    accumulator.update(
        "Top20", np.ones_like(truth, dtype=bool), truth, baseline, candidate
    )
    row = accumulator.rows("DD", "fixed")[0]
    assert row["repair"] == 1
    assert row["harm"] == 2
    assert row["net_repair"] == -0.25


def test_paired_bootstrap_reproducible():
    base = np.zeros((4, 5, 5), dtype=np.int64)
    candidate = np.zeros((4, 5, 5), dtype=np.int64)
    for index in range(4):
        base[index, 0, 0] = 5
        base[index, 1, 0] = 2
        candidate[index, 0, 0] = 5
        candidate[index, 1, 1] = 2
    first, first_values = paired_bootstrap_miou(
        base, candidate, resamples=100, seed=42
    )
    second, second_values = paired_bootstrap_miou(
        base, candidate, resamples=100, seed=42
    )
    assert first == second
    np.testing.assert_array_equal(first_values, second_values)
    assert first["observed_delta_mIoU"] > 0
