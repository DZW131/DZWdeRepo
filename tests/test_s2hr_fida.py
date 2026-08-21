"""Unit tests for the frozen S²HR-v1 FIDA-v0 audit."""

import inspect

import numpy as np

from tool import iouutils
from tool.infer_s2hr_fida import FIDAInstrumentor, PRIMARY_VARIANTS, masked_argmax
from tools.s2hr_fida_metrics import (
    BoundaryQualityAccumulator,
    OfficialMetricAccumulator,
    TeacherReliabilityAccumulator,
    foreground_boundary_bins,
    semantic_transition_band,
)


def test_primary_factorial_is_exactly_preregistered():
    assert PRIMARY_VARIANTS == {
        "V00": {"bps": False, "spatial": "zero"},
        "V10": {"bps": True, "spatial": "zero"},
        "V01": {"bps": False, "spatial": "learned"},
        "V11": {"bps": True, "spatial": "learned"},
    }


def test_masked_argmax_never_selects_absent_class():
    logits = np.zeros((4, 3, 3), dtype=np.float32)
    logits[1] = 100
    logits[3] = 2
    prediction = masked_argmax(logits, [True, False, False, True])
    assert np.all(prediction == 3)


def test_instrumentor_keeps_model_when_released_eval_returns_none():
    class ReleasedStyleModel:
        def cuda(self):
            return self

        def eval(self):
            return None

    model = ReleasedStyleModel()
    assert FIDAInstrumentor(model).model is model


def test_streaming_metric_matches_released_scores():
    truth = [
        np.asarray([[0, 0, 1], [2, 3, 4]], dtype=np.uint8),
        np.asarray([[0, 1, 1], [2, 3, 4]], dtype=np.uint8),
    ]
    prediction = [
        np.asarray([[0, 1, 1], [2, 0, 2]], dtype=np.uint8),
        np.asarray([[0, 1, 2], [3, 3, 1]], dtype=np.uint8),
    ]
    accumulator = OfficialMetricAccumulator()
    for target, candidate in zip(truth, prediction):
        accumulator.update(target, candidate)
    observed = accumulator.scores()
    released = iouutils.scores(
        [item.copy() for item in truth], [item.copy() for item in prediction], n_class=4
    )
    assert observed["mIoU"] == released["Mean IoU"]
    assert observed["mDice"] == released["Mean Dice"]
    for class_index in range(4):
        assert observed["class_iou"][str(class_index)] == released["Class IoU"][class_index]


def test_hma_boundary_bins_partition_only_foreground():
    truth = np.full((24, 24), 4, dtype=np.uint8)
    truth[2:22, 2:12] = 0
    truth[2:22, 12:22] = 1
    bins = foreground_boundary_bins(truth)
    count = sum(
        bins[name].astype(np.uint8)
        for name in ("B0_le_2", "B1_3_7", "B2_ge_8")
    )
    assert np.array_equal(count, bins["foreground"].astype(np.uint8))
    assert not bins["foreground"][0, 0]


def test_gt_semantic_boundary_excludes_foreground_background_transition():
    truth = np.full((8, 8), 4, dtype=np.uint8)
    truth[:, :4] = 0
    no_tissue_transition = semantic_transition_band(truth, foreground_only=True)
    assert no_tissue_transition.sum() == 0
    truth[:, 2:4] = 1
    tissue_transition = semantic_transition_band(truth, foreground_only=True)
    assert tissue_transition.sum() > 0


def test_teacher_help_harm_accounting():
    truth = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
    deep = np.asarray([[0, 1], [1, 0]], dtype=np.uint8)
    shallow = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
    bins = foreground_boundary_bins(truth)
    accumulator = TeacherReliabilityAccumulator()
    accumulator.update("oracle", truth, deep, shallow, bins, [True, True, False, False])
    rows, _ = accumulator.summary()
    overall = next(
        row for row in rows
        if row["presence"] == "oracle" and row["region"] == "overall" and row["class"] == "overall"
    )
    assert overall["deep_help"] == 1
    assert overall["deep_harm"] == 1
    assert overall["teacher_net"] == 0


def test_boundary_quality_reports_interior_contamination():
    truth = np.zeros((56, 56), dtype=np.uint8)
    truth[:, 28:] = 1
    bins = foreground_boundary_bins(truth)
    predicted = np.zeros((28, 28), dtype=bool)
    predicted[:, :2] = True
    accumulator = BoundaryQualityAccumulator()
    accumulator.update(predicted, truth, bins)
    result = accumulator.summary()
    assert 0.0 <= result["precision"] <= 1.0
    assert result["b2_interior_contamination"] > 0


def test_audit_source_has_no_optimizer_or_checkpoint_write():
    import tools.audit_s2hr_fida as audit

    audit_source = inspect.getsource(audit)
    infer_source = inspect.getsource(FIDAInstrumentor)
    assert "torch.optim" not in audit_source
    assert "torchutils" not in audit_source
    assert "torch.save" not in audit_source
    assert "load_state_dict" not in infer_source
