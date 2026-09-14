import inspect

import numpy as np
import pandas as pd
import pytest

from tools.audit_hqmr_m1_false_component_separability import (
    _decision,
    exact_shapley_3,
    extract_component_features_no_gt,
    label_components_with_gt,
    operating_point,
    percentile_scores,
    reliable_positive_from_scores,
)


def observable_fixture():
    prediction = np.zeros((8, 8), np.uint8)
    prediction[:, 5:] = 1
    prediction[1:3, 1:3] = 1
    scores = np.zeros((4, 8, 8), np.float32)
    for cls in range(4):
        scores[cls] = .1 + .05 * cls
    scores[0, prediction == 0] = .8
    scores[1, prediction == 1] = .9
    anchors = np.zeros_like(scores, bool)
    anchors[0, 6, 2] = True
    anchors[1, 2, 6] = True
    basis = np.linspace(0, 1, 6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4)
    weights = np.ones((6, 4), np.float32) / 6
    return prediction, scores, anchors, basis, weights


def test_no_gt_extractor_signature_and_execution_without_mask_files(tmp_path):
    assert "truth" not in inspect.signature(extract_component_features_no_gt).parameters
    prediction, scores, anchors, basis, weights = observable_fixture()
    # No GT directory or mask exists under tmp_path; extraction still succeeds.
    assert not (tmp_path / "mask").exists()
    rows = extract_component_features_no_gt("case", prediction, scores, anchors, basis, weights)
    assert rows
    assert all("is_false_component" not in row for row in rows)
    assert {row["component_uid"] for row in rows} == {row["component_uid"] for row in rows}


def test_component_label_is_exact_zero_same_class_overlap():
    prediction, scores, anchors, basis, weights = observable_fixture()
    features = pd.DataFrame(extract_component_features_no_gt(
        "case", prediction, scores, anchors, basis, weights))
    truth = np.zeros((8, 8), np.uint8)
    truth[:, 5:] = 1
    labels = pd.DataFrame(label_components_with_gt(features, prediction, truth, "case", "hqmr"))
    class1 = labels[labels.class_id == 1].sort_values("component_id")
    # Stable scipy scan order sees the valid right-hand component first.
    assert class1.is_false_component.tolist() == [0, 1]
    assert class1.gt_same_class_overlap.tolist() == [24, 0]


def test_query_features_are_finite_and_geometry_uses_eight_connectivity():
    prediction, scores, anchors, basis, weights = observable_fixture()
    prediction[3, 3] = 1  # diagonally attached to the 2x2 class-1 island under 8-connectivity
    rows = extract_component_features_no_gt("case", prediction, scores, anchors, basis, weights)
    class1 = [row for row in rows if row["class_id"] == 1]
    assert len(class1) == 2
    assert all(np.isfinite(row["weighted_query_support"]) for row in class1)
    assert all(row["number_of_same_class_components_in_image"] == 2 for row in class1)


def test_reliable_positive_contract_obeys_presence_and_top_fraction():
    scores = np.zeros((4, 10, 10), np.float32)
    scores[0, :2] = .9
    scores[1, 5:7] = .9
    labels = np.array([1, 0, 0, 0], np.uint8)
    anchors = reliable_positive_from_scores(scores, labels)
    assert anchors[0].any()
    assert not anchors[1:].any()
    assert anchors[0].sum() <= 15


def test_percentiles_use_classwise_unlabelled_rank_only():
    frame = pd.DataFrame({"class_id": [0, 0, 0, 1, 1], "anchor_density": [0., .5, 1., .2, .8],
                          "is_false_component": [0, 1, 0, 1, 0]})
    ranked = percentile_scores(frame, ["anchor_density"])
    assert ranked.loc[0, "false_rank_anchor_density"] == 1
    assert ranked.loc[2, "false_rank_anchor_density"] == 0
    changed_labels = frame.copy(); changed_labels["is_false_component"] = 1 - changed_labels["is_false_component"]
    reranked = percentile_scores(changed_labels, ["anchor_density"])
    assert np.array_equal(ranked.false_rank_anchor_density, reranked.false_rank_anchor_density)


def test_operating_point_respects_ties_and_valid_area_budget():
    y = np.array([1, 0, 1, 0])
    score = np.array([.9, .8, .7, .1])
    area = np.array([5, 1, 5, 99])
    result = operating_point(y, score, area, .05, weighted=True)
    assert result["valid_suppression"] <= .05
    assert result["false_recall"] == 1.0


def test_exact_shapley_three_recovers_additive_utilities():
    contributions = [.1, .2, .3]
    values = {subset: .5 + sum(value for index, value in enumerate(contributions)
                               if subset & (1 << index)) for subset in range(8)}
    assert list(exact_shapley_3(values).values()) == pytest.approx(contributions)


@pytest.mark.parametrize(
    "inputs,expected",
    [
        ((.90, .91, .70, {"0": .9, "1": .88, "2": .82, "3": .75},
          {"A": .08, "M": .07, "Q": .25}, .70), "FALSE_COMPONENT_SEPARABLE"),
        ((.80, .80, .40, {"0": .8, "1": .76, "2": .72, "3": .70},
          {"A": .02, "M": .10, "Q": .18}, .70), "PARTIAL_FALSE_COMPONENT_SEPARABILITY"),
        ((.70, .72, .20, {"0": .7, "1": .7, "2": .7, "3": .7},
          {"A": .01, "M": .02, "Q": .03}, .90), "GEOMETRY_SHORTCUT_ONLY"),
        ((.70, .72, .20, {"0": .7, "1": .7, "2": .7, "3": .7},
          {"A": .01, "M": .02, "Q": .03}, .70), "FALSE_COMPONENT_NOT_SEPARABLE"),
    ],
)
def test_frozen_decision_rules(inputs, expected):
    assert _decision(*inputs)[0] == expected
