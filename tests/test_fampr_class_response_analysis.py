import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from tool import iouutils
from tools.analyze_fampr_class_response import (
    OfficialMetricAccumulator,
    VectorStats,
    boundary_mask,
    component_size,
    discover_samples,
    normalize_cams,
    predict_from_cams,
    presence_label,
    transition_group_masks,
)


class FrozenAnalysisPrimitiveTest(unittest.TestCase):
    def test_streaming_official_metric_matches_iouutils(self):
        targets = [
            np.asarray([[0, 0, 1, 4], [2, 3, 1, 4]], dtype=np.uint8),
            np.asarray([[3, 2, 2, 4], [0, 1, 3, 4]], dtype=np.uint8),
        ]
        predictions = [
            np.asarray([[0, 1, 1, 0], [2, 0, 1, 3]], dtype=np.uint8),
            np.asarray([[3, 0, 2, 2], [1, 1, 3, 0]], dtype=np.uint8),
        ]
        accumulator = OfficialMetricAccumulator()
        for target, prediction in zip(targets, predictions):
            accumulator.update(target, prediction)
        streaming = accumulator.result()
        official = iouutils.scores(
            [target.copy() for target in targets],
            [prediction.copy() for prediction in predictions],
            n_class=4,
        )
        self.assertAlmostEqual(streaming["Mean IoU"], official["Mean IoU"])
        self.assertAlmostEqual(streaming["Mean Dice"], official["Mean Dice"])
        for class_id in range(4):
            self.assertAlmostEqual(
                streaming["Class IoU"][class_id],
                official["Class IoU"][class_id],
            )

    def test_transition_groups_are_exhaustive_and_disjoint(self):
        target = np.asarray([[0, 1], [2, 3]])
        a0 = np.asarray([[0, 0], [2, 0]])
        fampr = np.asarray([[0, 1], [0, 1]])
        groups = transition_group_masks(target, a0, fampr)
        total = sum(mask.astype(np.uint8) for mask in groups.values())
        self.assertTrue(np.array_equal(total, np.ones_like(target)))
        self.assertEqual(groups["BOTH_CORRECT"].sum(), 1)
        self.assertEqual(groups["CORRECTED_BY_FAMPR"].sum(), 1)
        self.assertEqual(groups["HARMED_BY_FAMPR"].sum(), 1)
        self.assertEqual(groups["BOTH_WRONG"].sum(), 1)

    def test_boundary_band_expands_a_class_transition(self):
        target = np.zeros((15, 15), dtype=np.uint8)
        target[:, 8:] = 1
        boundary = boundary_mask(target, radius=3)
        self.assertTrue(boundary[:, 7].all())
        self.assertTrue(boundary[:, 8].all())
        self.assertFalse(boundary[:, 0].any())
        self.assertFalse(boundary[:, -1].any())

    def test_presence_and_fusion_match_zero_gated_argmax_semantics(self):
        probability = np.asarray([0.7, 0.95, 0.2, 0.7], dtype=np.float32)
        label = presence_label(probability)
        self.assertTrue(np.array_equal(label, np.asarray([0, 1, 0, 1])))
        raw = {
            stage: np.stack(
                [np.zeros((2, 2)), np.ones((2, 2)), 2 * np.ones((2, 2)), 3 * np.ones((2, 2))]
            ).astype(np.float32)
            for stage in ("cam56", "cam28_1", "cam28_2", "camdeep")
        }
        normalized = normalize_cams(raw)
        prediction, fusion = predict_from_cams(normalized, label)
        self.assertEqual(prediction.shape, (2, 2))
        self.assertEqual(fusion.shape, (4, 2, 2))

    def test_vector_stats_has_exact_moments(self):
        accumulator = VectorStats(("x", "y"), reservoir_size=10, sample_stride=2, seed=1)
        values = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        accumulator.update(values)
        rows = {row["metric"]: row for row in accumulator.rows({})}
        self.assertEqual(rows["x"]["count"], 3)
        self.assertAlmostEqual(rows["x"]["mean"], 3.0)
        self.assertAlmostEqual(rows["y"]["mean"], 4.0)
        self.assertAlmostEqual(accumulator.correlation("x", "y"), 1.0)

    def test_component_size_uses_frozen_quartiles(self):
        threshold = {"q25": 10.0, "q75": 100.0}
        self.assertEqual(component_size(10, threshold), "small")
        self.assertEqual(component_size(11, threshold), "medium")
        self.assertEqual(component_size(100, threshold), "medium")
        self.assertEqual(component_size(101, threshold), "large")

    def test_sample_discovery_requires_paired_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "img").mkdir()
            (root / "mask").mkdir()
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(root / "img" / "b.png")
            Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(root / "mask" / "b.png")
            samples = discover_samples(root, max_images=None)
            self.assertEqual(samples[0][0], "b")


if __name__ == "__main__":
    unittest.main()
