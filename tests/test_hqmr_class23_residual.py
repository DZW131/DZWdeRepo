import numpy as np

from tools.audit_hqmr_class23_residual import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, confusion5, js_divergence,
    normalized_matrix, ratio,
)


def test_audit_is_frozen_and_zero_tuning():
    assert BOOTSTRAP_SEED == 20260913 and BOOTSTRAP_RESAMPLES == 10_000


def test_confusion5_has_background_row_and_column():
    truth = np.array([[0, 2, 3, 4]]); pred = np.array([[0, 3, 2, 1]])
    matrix = confusion5(truth, pred)
    assert matrix.shape == (5, 5) and matrix[2, 3] == 1 and matrix[3, 2] == 1 and matrix[4, 1] == 1


def test_frozen_foreground_protocol_has_zero_predicted_background_column():
    truth = np.array([[2, 3, 4]]); pred = np.array([[3, 2, 0]])
    assert confusion5(truth, pred)[:, 4].sum() == 0


def test_row_normalization_closes():
    value = np.array([[1, 1], [0, 2]])
    assert np.allclose(normalized_matrix(value, 1).sum(1), 1)


def test_column_normalization_closes():
    value = np.array([[1, 1], [1, 1]])
    assert np.allclose(normalized_matrix(value, 0).sum(0), 1)


def test_js_is_zero_for_identical_weights():
    value = np.array([.2, .3, .5])
    assert abs(js_divergence(value, value)) < 1e-12


def test_js_is_symmetric_and_positive():
    left, right = np.array([.9, .1]), np.array([.1, .9])
    assert js_divergence(left, right) > 0 and np.isclose(js_divergence(left, right), js_divergence(right, left))


def test_ratio_zero_denominator_is_explicit_zero():
    assert ratio(3, 0) == 0.0
