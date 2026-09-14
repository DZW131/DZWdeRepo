import numpy as np
import pytest

from tools.hqmr_morphology_oracle_audit import (
    apply_operator,
    exact_shapley,
    m1_fp_island_removal,
    m2_enclosed_hole_fill,
    m3_fragment_bridge,
    m4_protrusion_trim,
    m5_indentation_fill,
    subset_states,
)


def test_m1_removes_only_zero_overlap_fp_island():
    truth = np.zeros((7, 7), np.uint8)
    truth[:, 4:] = 1
    pred = truth.copy()
    pred[1:3, 1:3] = 1  # class-1 island with no class-1 GT overlap
    pred[4:6, 3:5] = 1  # component overlaps class-1 GT and must survive intact

    out, changed, stats = m1_fp_island_removal(pred, truth)

    assert np.all(out[1:3, 1:3] == 0)
    assert np.all(out[4:6, 3:5] == 1)
    assert changed.sum() == 4
    assert stats["by_class"]["1"] == {"count": 1, "area": 4}


def test_m2_fills_enclosed_hole_but_not_open_background():
    truth = np.ones((7, 7), np.uint8)
    pred = np.ones((7, 7), np.uint8)
    pred[3, 3] = 0
    pred[0, 0] = 0

    out, changed, _ = m2_enclosed_hole_fill(pred, truth)

    assert out[3, 3] == 1 and changed[3, 3]
    assert out[0, 0] == 0 and not changed[0, 0]


def test_m3_bridges_fragments_inside_gt_with_eight_connectivity():
    truth = np.zeros((7, 7), np.uint8)
    truth[1:6, 1:6] = 1
    pred = np.zeros_like(truth)
    pred[1:3, 1:3] = 1
    pred[5, 5] = 1

    out, changed, stats = m3_fragment_bridge(pred, truth)

    labels = __import__("scipy").ndimage.label((out == 1) & (truth == 1), np.ones((3, 3)))[1]
    assert labels == 1
    assert not np.any(changed & (truth != 1))
    assert stats["bridges"] == 1


def test_m3_does_nothing_without_two_predicted_fragments():
    truth = np.ones((5, 5), np.uint8)
    pred = np.zeros_like(truth)
    out, changed, stats = m3_fragment_bridge(pred, truth)
    assert np.array_equal(out, pred)
    assert not changed.any()
    assert stats["count"] == 0


def test_m4_trims_attached_protrusion_but_leaves_remote_island():
    truth = np.zeros((9, 9), np.uint8)
    truth[3:6, 3:6] = 1
    pred = truth.copy()
    pred[2, 4] = 1  # attached outer protrusion
    pred[0, 0] = 1  # disconnected island, owned by M1 rather than M4

    out, changed, _ = m4_protrusion_trim(pred, truth, radius=1)

    assert out[2, 4] == 0 and changed[2, 4]
    assert out[0, 0] == 1 and not changed[0, 0]


def test_m5_fills_inner_boundary_only_in_seeded_gt_component():
    truth = np.zeros((10, 10), np.uint8)
    truth[1:5, 1:5] = 1
    truth[6:9, 6:9] = 1
    pred = np.zeros_like(truth)
    pred[2:4, 2:4] = 1

    out, changed, _ = m5_indentation_fill(pred, truth, radius=1)

    assert changed[1, 2] and out[1, 2] == 1
    assert not np.any(changed[6:9, 6:9])


def test_subset_enumeration_is_fixed_and_operator_dispatch_rejects_invalid_index():
    truth = np.zeros((4, 4), np.uint8)
    pred = truth.copy()
    states, changes, stats = subset_states(pred, truth)
    assert list(states) == list(range(32))
    assert len(changes) == len(stats) == 31
    with pytest.raises(ValueError):
        apply_operator(5, pred, truth)


def test_exact_shapley_recovers_additive_contributions():
    contributions = [0.2, -0.1, 0.7, 0.0, 1.1]
    values = {
        subset: 3.0 + sum(value for index, value in enumerate(contributions) if subset & (1 << index))
        for subset in range(32)
    }
    result = exact_shapley(values)
    assert list(result) == ["M1", "M2", "M3", "M4", "M5"]
    assert list(result.values()) == pytest.approx(contributions)
