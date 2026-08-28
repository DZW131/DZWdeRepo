import numpy as np
import pytest
import torch

from network.resnet38_cls import Net
from tools.rddr_phase0_common import (
    bootstrap_indices,
    bootstrap_mean,
    dataset_quantile_threshold,
    diagnostic_forward,
    eligible_error,
    official_histogram,
    probability_scores,
)


@pytest.fixture(scope="module")
def forward_pair():
    torch.manual_seed(42)
    model = Net(4)
    model.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        expected = model(image)
        actual, diagnostics = diagnostic_forward(model, image)
    return expected, actual, diagnostics


def test_phase0_tensor_contract(forward_pair):
    _, _, diagnostics = forward_pair
    assert diagnostics["F28_raw"].shape[1] == 512
    assert diagnostics["F28_rect"].shape == diagnostics["F28_raw"].shape
    assert diagnostics["Ddeep"].shape[1] == 4096
    assert diagnostics["CAM28_raw_logits"].shape[1] == 4
    assert diagnostics["CAM28_rect_logits"].shape[1] == 4
    assert diagnostics["CAMdeep_logits"].shape[1] == 4


def test_default_forward_equivalence(forward_pair):
    expected, actual, _ = forward_pair
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.fixture
def score_fixture():
    torch.manual_seed(7)
    shallow = torch.randn(2, 4, 5, 6)
    deep = torch.randn(2, 4, 5, 6)
    return probability_scores(shallow, deep)


def test_probability_sum_to_one(score_fixture):
    p_shallow, p_deep, _ = score_fixture
    torch.testing.assert_close(p_shallow.sum(1), torch.ones(2, 5, 6))
    torch.testing.assert_close(p_deep.sum(1), torch.ones(2, 5, 6))


def test_js_nonnegative_finite(score_fixture):
    _, _, scores = score_fixture
    assert torch.isfinite(scores["S_JS"]).all()
    assert float(scores["S_JS"].min()) >= -1.0e-7


def test_error_label_matches_official_foreground_evaluator():
    truth = np.asarray([[0, 1, 4], [2, 3, 4]], dtype=np.uint8)
    prediction = np.asarray([[0, 2, 1], [1, 3, 0]], dtype=np.uint8)
    valid, error = eligible_error(prediction, truth)
    assert valid.tolist() == [[True, True, False], [True, True, False]]
    assert error.tolist() == [[False, True, False], [True, False, False]]
    histogram = official_histogram(truth, prediction)
    assert histogram.sum() == valid.sum()
    assert histogram.trace() == int((valid & ~error).sum())


def test_quantile_mask_is_deterministic():
    values = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    first = dataset_quantile_threshold(values, 0.20)
    second = dataset_quantile_threshold(values.copy(), 0.20)
    assert first == second
    assert np.isclose(first, 0.4)
    assert np.array_equal(values >= first, np.asarray([False, False, False, False, True]))


def test_bootstrap_reproducibility():
    values = np.asarray([0.55, 0.61, 0.72, 0.65])
    first_indices = bootstrap_indices(4, resamples=100, seed=42)
    second_indices = bootstrap_indices(4, resamples=100, seed=42)
    assert np.array_equal(first_indices, second_indices)
    first, first_summary = bootstrap_mean(values, first_indices)
    second, second_summary = bootstrap_mean(values, second_indices)
    assert np.array_equal(first, second)
    assert first_summary == second_summary


def test_bootstrap_empty_stratum_returns_nan_without_failure():
    indices = bootstrap_indices(3, resamples=10, seed=42)
    values, summary = bootstrap_mean(np.asarray([np.nan, np.nan, np.nan]), indices)
    assert np.isnan(values).all()
    assert np.isnan(summary["observed"])
    assert np.isnan(summary["ci95_low"])
