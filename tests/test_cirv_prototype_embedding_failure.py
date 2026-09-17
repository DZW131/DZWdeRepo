import numpy as np
import pandas as pd

from tools.audit_cirv_prototype_embedding_failure import (
    classification_metrics,
    classify,
    decide,
    gate_strength,
    gt_anatomy,
    purity_stratum,
    similarity_anatomy,
)


def test_purity_strata_boundaries():
    assert [purity_stratum(x) for x in (.49, .50, .699, .70, .899, .90, 1.0)] == [
        "P0", "P1", "P1", "P2", "P2", "P3", "P3"
    ]


def test_gate_strength_boundaries():
    assert gate_strength(.0699) == "WEAK"
    assert gate_strength(.07) == "MODERATE"
    assert gate_strength(.15) == "STRONG"


def test_gt_anatomy_counts_ignore_in_purity_denominator():
    truth = np.array([[0, 0], [1, 4]], dtype=np.uint8)
    result = gt_anatomy(np.ones((2, 2), bool), truth)
    assert result["gt_majority_class"] == 0
    assert result["gt_majority_purity"] == .5
    assert result["ignore_fraction"] == .25


def test_classification_metrics_excludes_p0_and_weights_area():
    frame = pd.DataFrame({
        "purity": [.49, .5, .9], "prediction": [0, 0, 1],
        "gt_majority_class": [0, 0, 0], "area": [100, 10, 30],
    })
    value = classification_metrics(frame, "prediction")
    assert value["components"] == 2
    assert value["component_accuracy"] == .5
    assert value["area_weighted_accuracy"] == .25


def test_similarity_anatomy_uses_true_class_margin():
    bank = np.zeros((4, 4, 4), np.float32)
    for cls in range(4):
        bank[cls, :, cls] = 1.0
    result = similarity_anatomy(np.array([.8, .9, 0, 0], np.float32), bank, true_class=0)
    assert result["nearest_class"] == 1
    assert result["strongest_rival_class"] == 1
    assert result["prototype_margin"] < 0


def test_classify_optional_prior_reproduces_cirv_semantics():
    bank = np.zeros((4, 4, 4), np.float32)
    for cls in range(4):
        bank[cls, :, cls] = 1.0
    embedding = np.array([.9, .8, 0, 0], np.float32)
    assert classify(embedding, bank)[0] == 0
    assert classify(embedding, bank, np.array([.01, .99, .01, .01], np.float32))[0] == 1


def test_exact_decision_branches():
    reps_capable = {"R1_K4": .9, "R2_H5": .8, "R3_query_conditioned_H4": .8, "R4_deep_backbone": .8}
    assert decide(.16, .01, .9, reps_capable, .1, .8)[0] == "SOURCE_LABEL_CONTAMINATION"
    assert decide(.01, .16, .9, reps_capable, .1, .8)[0] == "SOURCE_REGION_CONTAMINATION"
    reps_k4 = {**reps_capable, "R1_K4": .6, "R2_H5": .86}
    assert decide(.01, .01, .6, reps_k4, .1, .8)[0] == "K4_SPECIFIC_REPRESENTATION_LIMIT"
    reps_weak = {key: .6 for key in reps_capable}
    assert decide(.01, .01, .6, reps_weak, .1, .6)[0] == "EXISTING_REPRESENTATIONS_INSUFFICIENT"
    assert decide(.01, .01, .9, reps_capable, .6, .8)[0] == "TARGET_REGION_IMPURITY"
