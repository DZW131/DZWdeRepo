import torch

from tools.analyze_cdsr_need_signal import (
    analytical_need,
    normalized_entropy,
    normalized_jsd,
)


def test_normalized_jsd_is_symmetric_and_bounded():
    first = torch.softmax(torch.randn(2, 4, 5, 5), dim=1)
    second = torch.softmax(torch.randn(2, 4, 5, 5), dim=1)

    forward = normalized_jsd(first, second)
    reverse = normalized_jsd(second, first)

    torch.testing.assert_close(forward, reverse)
    assert torch.all(forward >= 0.0)
    assert torch.all(forward <= 1.0)


def test_normalized_entropy_is_bounded():
    probabilities = torch.softmax(torch.randn(2, 4, 5, 5), dim=1)
    entropy = normalized_entropy(probabilities)

    assert torch.all(entropy >= 0.0)
    assert torch.all(entropy <= 1.0)


def test_identical_confident_views_have_low_need():
    logits = torch.full((1, 4, 2, 2), -12.0)
    logits[:, 0] = 12.0

    result = analytical_need(logits, logits)

    assert result["need"].max().item() < 1e-5


def test_reliable_deep_disagreement_has_high_need():
    stage = torch.full((1, 4, 2, 2), -12.0)
    deep = torch.full((1, 4, 2, 2), -12.0)
    stage[:, 0] = 12.0
    deep[:, 1] = 12.0

    result = analytical_need(stage, deep)

    assert result["need"].min().item() > 0.95


def test_unreliable_deep_suppresses_need():
    stage = torch.full((1, 4, 2, 2), -12.0)
    stage[:, 0] = 12.0
    deep = torch.zeros((1, 4, 2, 2))

    result = analytical_need(stage, deep)

    assert result["deep_reliability"].abs().max().item() < 1e-6
    assert result["need"].abs().max().item() < 1e-6
