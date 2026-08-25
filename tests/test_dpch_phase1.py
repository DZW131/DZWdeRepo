import numpy as np
import torch

from tools.run_dpch_phase1 import (
    binary_auroc,
    cohen_d,
    paired_bootstrap_ci,
    semantic_concentration,
)


def test_binary_auroc_and_effect_direction():
    positive = np.asarray([0.8, 0.9, 1.0])
    negative = np.asarray([0.0, 0.1, 0.2])
    assert binary_auroc(positive, negative) == 1.0
    assert cohen_d(positive, negative) > 0.0


def test_binary_auroc_uses_average_ranks_for_ties():
    positive = np.asarray([0.0, 1.0])
    negative = np.asarray([0.0, 1.0])
    assert binary_auroc(positive, negative) == 0.5


def test_paired_bootstrap_is_reproducible_and_positive():
    values = np.linspace(0.1, 0.2, 20)
    first = paired_bootstrap_ci(values, resamples=500, seed=42)
    second = paired_bootstrap_ci(values, resamples=500, seed=42)
    assert first == second
    assert first["ci95_low"] > 0.0


def test_semantic_concentration_reports_expected_regions():
    raw = torch.tensor(
        [[
            [[1.0, 0.8], [0.0, 0.2]],
            [[0.0, 0.2], [1.0, 0.8]],
        ]]
    )
    semantic = torch.tensor(
        [[
            [[1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0]],
        ]]
    )
    truth = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
    zones = {
        "boundary_le_7": np.asarray([[False, True], [False, True]]),
        "interior_ge_8": np.asarray([[True, False], [True, False]]),
    }
    image, classes = semantic_concentration(raw, semantic, truth, zones)
    assert len(classes) == 2
    assert image["delta_boundary"] > 0.0
    assert image["semantic_boundary_pixels"] == 2
    assert np.isfinite(image["interclass_semantic"])
