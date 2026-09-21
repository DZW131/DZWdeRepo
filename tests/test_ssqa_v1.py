"""Synthetic protocol checks; no foundation-model weights or BCSS GT required."""
from __future__ import annotations

import numpy as np
import pandas as pd

from tools.ssqa_v1.evaluate import bootstrap_patient, match_controls, metric_row
from tools.ssqa_v1.prepare import bbox15_square, freeze_split


def test_bbox15_preserves_edge_component() -> None:
    mask = np.zeros((224, 224), bool)
    mask[0:3, 220:224] = True
    x0, y0, x1, y1 = bbox15_square(mask)
    assert (x1 - x0) == (y1 - y0)
    assert 0 <= x0 < x1 <= 224 and 0 <= y0 < y1 <= 224
    assert np.all(mask[y0:y1, x0:x1][mask[y0:y1, x0:x1]])
    assert mask.sum() == mask[y0:y1, x0:x1].sum()


def test_patient_split_uses_gt_free_fields_only() -> None:
    frame = pd.DataFrame({"patient_id": [f"TCGA-X-{i:02d}" for i in range(10)],
                          "baseline_class": [i % 4 for i in range(10)], "area": np.arange(1, 11)})
    first = freeze_split(frame); second = freeze_split(frame)
    assert first == second
    assert len(first["dev_patient_ids"]) == 6
    assert not set(first["dev_patient_ids"]) & set(first["holdout_patient_ids"])


def test_metrics_and_patient_bootstrap() -> None:
    hard = pd.DataFrame({"row_index": [0, 1, 2, 3], "patient_id": ["A", "A", "B", "B"],
                         "area": [1, 2, 3, 4], "true_class": [0, 1, 2, 3],
                         "sequential5_pred": [1, 0, 3, 2]})
    controls = pd.DataFrame({"row_index": [0, 1], "patient_id": ["A", "B"], "true_class": [0, 1]})
    scores = np.eye(4, dtype=np.float32)
    plip = np.roll(scores, 1, axis=1)
    result = metric_row(hard, scores, plip, controls)
    assert result["HTRP_area"] == 1 and result["HTop1"] == 1
    assert result["NRR_over_PLIP"] == 1 and result["ControlTop1"] == 1
    ci = bootstrap_patient(hard, controls, scores, plip, n=20)
    assert ci["HTRP_area"] == [1, 1]


def test_matching_is_within_exact_strata_without_replacement() -> None:
    frame = pd.DataFrame({"row_index": [0, 1, 2], "true_class": [0, 0, 1],
        "area_quartile": [0, 0, 0], "confidence_quartile": [1, 1, 1],
        "baseline_correct": [True, True, True], "evaluable": [True, True, True]})
    hard = pd.DataFrame({"true_class": [0, 0, 0, 1], "area_quartile": [0] * 4,
                         "confidence_quartile": [1] * 4})
    matched = match_controls(frame, hard, 42)
    assert len(matched) == 3
    assert matched.row_index.is_unique
