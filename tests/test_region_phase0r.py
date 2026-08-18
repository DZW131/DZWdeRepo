import unittest

import numpy as np
import pandas as pd
import torch

from network.resnet38_cls import Net_CAM
from tools.region_phase0r.extractor import RegionAuditExtractor
from tools.region_phase0r.probes import decide_phase0r
from tools.region_phase0r.regions import (
    connected_components,
    extract_image_regions,
    relabel_predictions,
    slide_id_from_image,
)


class RegionPhase0RTests(unittest.TestCase):
    def test_slide_parser(self):
        self.assertEqual(
            slide_id_from_image("TCGA-EW-A1PB-DX1_xmin000_ymin001"),
            "TCGA-EW-A1PB-DX1",
        )

    def test_classwise_eight_connectivity_and_no_background(self):
        prediction = np.full((4, 4), 4, dtype=np.uint8)
        prediction[0, 0] = 0
        prediction[1, 1] = 0
        prediction[2, 2] = 1
        components = connected_components(prediction)
        self.assertEqual(components[0][1], 1)
        self.assertEqual(components[1][1], 1)
        self.assertEqual(sum(item[1] for item in components.values()), 2)

    def test_region_diagnostic_and_tokens(self):
        prediction = np.full((4, 4), 4, dtype=np.uint8)
        prediction[:2, :2] = 0
        gt = np.full((4, 4), 4, dtype=np.uint8)
        gt[:2, :2] = 1
        feature = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        rows, region, bbox, centroid, geometry = extract_image_regions(
            prediction, gt, feature, "TCGA-AA-0001-DX1_xmin0", 0
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["taxonomy"], "B_misclassified_pure")
        self.assertEqual(rows[0]["majority_gt"], 1)
        np.testing.assert_allclose(region, bbox)
        self.assertEqual(region.shape, (1, 1))
        self.assertEqual(centroid.shape, (1, 1))
        self.assertEqual(geometry.shape, (1, 8))

    def test_fixed_support_relabel_and_background_removal(self):
        prediction = np.full((1, 3, 3), 4, dtype=np.uint8)
        prediction[0, :2, :2] = 0
        frame = pd.DataFrame([{
            "image_index": 0, "predicted_class": 0, "component_label": 1,
        }])
        relabeled = relabel_predictions(prediction, frame, np.array([4]))
        self.assertTrue(np.all(relabeled == 4))

    def test_extractor_is_parameter_identical(self):
        released = Net_CAM(n_class=4)
        extractor = RegionAuditExtractor(n_class=4)
        self.assertEqual(list(released.state_dict()), list(extractor.state_dict()))
        for key in released.state_dict():
            self.assertEqual(released.state_dict()[key].shape, extractor.state_dict()[key].shape)

    def test_preregistered_decisions(self):
        self.assertEqual(
            decide_phase0r(2.1, 0.04, 0.06, 0.6, 0.3, 4, 0.2, False),
            "REGION_REP_STRONG_GO",
        )
        self.assertEqual(
            decide_phase0r(0.4, 0.02, 0.02, 0.2, 0.2, 2, 0.2, False),
            "REGION_REP_NOGO",
        )


if __name__ == "__main__":
    unittest.main()
