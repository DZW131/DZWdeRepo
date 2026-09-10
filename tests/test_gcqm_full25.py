from pathlib import Path

import numpy as np

from tools.eval_gcqm_full25_bcss_seed42 import performance_decision, prediction_from_cam
from tools.run_gcqm_full25_bcss_seed42 import CONFIG, EPOCHS, MILESTONES, TOTAL_STEPS


ROOT = Path(__file__).resolve().parents[1]


def test_full25_frozen_training_contract():
    assert EPOCHS == 25 and TOTAL_STEPS == 29275 and MILESTONES == {5, 10, 15, 20, 25}
    assert CONFIG["effective_batch_size"] == 20 and CONFIG["locality_denominator"] == 29275
    assert CONFIG["checkpoint_selection"] == "fixed Epoch25 FINAL only"
    assert CONFIG["new_parameters"] == 0 and CONFIG["new_auxiliary_loss"] == 0


def test_performance_decision_boundaries():
    healthy = {str(i): 0.0 for i in range(4)}
    assert performance_decision(.50, .01, healthy) == "GCQM_FULL25_STRONG_GO"
    assert performance_decision(.30, .01, healthy) == "GCQM_FULL25_GO"
    assert performance_decision(.30, 0.0, healthy) == "GCQM_FULL25_POSITIVE_BUT_UNCERTAIN"
    assert performance_decision(.299, .01, healthy) == "GCQM_FULL25_NEUTRAL"
    assert performance_decision(-.30, .01, healthy) == "GCQM_FULL25_NOGO"
    regressed = {**healthy, "2": -3.0}
    assert performance_decision(.40, .01, regressed) == "GCQM_FULL25_NOGO"
    assert performance_decision(.80, .10, healthy, comparable=False) == "GCQM_FULL25_NOT_COMPARABLE"


def test_evaluation_is_e25_only_and_bootstrap_is_frozen():
    source = (ROOT / "tools/eval_gcqm_full25_bcss_seed42.py").read_text(encoding="utf-8")
    assert "BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260910, 10_000" in source
    assert "gcqm_full25_epoch25_final.json" in source
    assert "baseline_metric_reproduction" in source and "baseline_per_image_reproduction" in source
    assert '"params": complexity["gcqm_parameters"]' in source


def test_gcqm_inference_does_not_use_segmentation_gt_as_model_input():
    source = (ROOT / "tools/eval_gcqm_full25_bcss_seed42.py").read_text(encoding="utf-8")
    prediction = source[source.index("def _predict_gcqm"):source.index("def _infer")]
    assert "dummy = torch.ones" in prediction and "truth" not in prediction


def test_prediction_argmax_uses_only_present_classes():
    cam = np.zeros((4, 2, 2), dtype=np.float32); cam[1] = .4; cam[2] = .9
    prediction = prediction_from_cam(cam, np.asarray([0, 1, 0, 0]), np.zeros((2, 2, 3), dtype=np.uint8))
    assert np.array_equal(prediction, np.ones((2, 2), dtype=np.int64))


def test_training_has_no_validation_import_or_path():
    source = (ROOT / "tools/run_gcqm_full25_bcss_seed42.py").read_text(encoding="utf-8")
    assert "Stage1_TrainDataset" in source and "Stage1_InferDataset" not in source
    assert '"validation_during_training": False' in source
