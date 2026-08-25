import numpy as np
import torch

from tools.run_wsa_ch_exp001 import (
    AssignmentAccumulator,
    assignment_arrays,
    build_cam_groups,
    build_oracle_groups,
    paired_bootstrap_ci,
)


def test_cam_groups_are_normalized_and_finite():
    feature = torch.tensor(
        [[
            [[2.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 2.0]],
        ]]
    )
    classifier = torch.nn.Conv2d(2, 2, 1, bias=False)
    with torch.no_grad():
        classifier.weight.zero_()
        classifier.weight[0, 0, 0, 0] = 1.0
        classifier.weight[1, 1, 0, 0] = 1.0
    groups, diagnostics = build_cam_groups(feature, classifier, [0, 1])
    assert groups.shape == (2, 2)
    assert torch.allclose(groups.norm(dim=1), torch.ones(2))
    assert diagnostics["normalized_spatial_entropy"] > 0.0
    assert np.isfinite(diagnostics["prototype_interclass_cosine"])


def test_oracle_groups_use_only_classes_with_interior_pixels():
    feature = torch.tensor(
        [[
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]]
    )
    truth = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
    interior = np.asarray([[True, True], [False, False]])
    groups, classes = build_oracle_groups(feature, truth, interior, [0, 1])
    assert classes == [0]
    assert groups.shape == (1, 2)


def test_assignment_uses_hardest_wrong_group():
    query = torch.tensor(
        [[
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 0.0]],
        ]]
    )
    prototypes = torch.eye(3)
    truth = np.asarray([[0, 1]], dtype=np.uint8)
    boundary = np.ones_like(truth, dtype=bool)
    result = assignment_arrays(query, prototypes, [0, 1, 2], truth, boundary)
    assert result["correct"].tolist() == [True, True]
    assert np.allclose(result["margin"], [1.0, 1.0])
    assert np.allclose(result["wrong_similarity"], [0.0, 0.0])


def test_difficulty_accounting_is_directional():
    raw = {
        "margin": np.asarray([1.0, -1.0]),
        "correct": np.asarray([True, False]),
        "truth_class": np.asarray([0, 1]),
        "same_similarity": np.asarray([1.0, 0.0]),
        "wrong_similarity": np.asarray([0.0, 1.0]),
    }
    improved = {
        "margin": np.asarray([-1.0, 1.0]),
        "correct": np.asarray([False, True]),
        "truth_class": np.asarray([0, 1]),
        "same_similarity": np.asarray([0.0, 1.0]),
        "wrong_similarity": np.asarray([1.0, 0.0]),
    }
    accumulator = AssignmentAccumulator()
    accumulator.update(
        {"raw_F": raw, "CH_F": improved, "CBCCH_Fb": improved}, chance=0.5
    )
    result = accumulator.result()["difficulty"]["CBCCH_Fb"]
    assert result["hard_correction_rate"] == 1.0
    assert result["easy_harm_rate"] == 1.0


def test_bootstrap_is_reproducible():
    values = np.linspace(0.1, 0.2, 25)
    first = paired_bootstrap_ci(values, resamples=500, seed=42)
    second = paired_bootstrap_ci(values, resamples=500, seed=42)
    assert first == second
    assert first["ci95_low"] > 0.0
