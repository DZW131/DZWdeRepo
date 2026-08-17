import torch
import torch.nn.functional as F

from tool import iouutils
from tools.audit_clrr_v2_phase0 import OfficialMetricAccumulator
from tools.clrr_v2_phase0_core import (
    analytical_virtual_correction,
    classifier_backprojection,
    leave_one_out_consensus,
    normalized_entropy_reliability,
)


def probabilities(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return {
        "stage1": torch.softmax(torch.randn(2, 4, 7, 7, generator=generator), 1),
        "stage2": torch.softmax(torch.randn(2, 4, 5, 5, generator=generator), 1),
        "stage3": torch.softmax(torch.randn(2, 4, 5, 5, generator=generator), 1),
        "deep": torch.softmax(torch.randn(2, 4, 5, 5, generator=generator), 1),
    }


def test_reliability_is_bounded():
    value = normalized_entropy_reliability(probabilities()["stage1"])
    assert torch.all(value >= 0)
    assert torch.all(value <= 1)


def test_leave_one_out_excludes_target_and_normalizes_consensus():
    values = probabilities()
    result = leave_one_out_consensus(values, "stage1", (7, 7))
    assert "stage1" not in result["source_names"]
    assert set(result["source_names"]) == {"stage2", "stage3", "deep"}
    assert torch.allclose(
        result["consensus"].sum(dim=1),
        torch.ones(2, 7, 7),
        atol=1e-6,
    )
    assert torch.all(result["consensus_reliability"] >= 0)
    assert torch.all(result["consensus_reliability"] <= 1)


def test_backprojection_matches_negative_consensus_ce_gradient():
    generator = torch.Generator().manual_seed(4)
    feature = torch.randn(2, 6, 5, 5, generator=generator, requires_grad=True)
    weight = torch.randn(4, 6, 1, 1, generator=generator)
    bias = torch.randn(4, generator=generator)
    target = torch.softmax(torch.randn(2, 4, 5, 5, generator=generator), 1)
    logits = F.conv2d(feature, weight, bias)
    probability = torch.softmax(logits, dim=1)
    loss = -(target * probability.clamp_min(1e-8).log()).sum()
    gradient = torch.autograd.grad(loss, feature)[0]
    analytical = classifier_backprojection(target - probability, weight)
    cosine = F.cosine_similarity(
        analytical.flatten(), (-gradient).flatten(), dim=0
    )
    assert cosine.item() > 0.999999


def test_fixed_point_and_zero_maturity_close_the_loop():
    generator = torch.Generator().manual_seed(8)
    feature = torch.randn(1, 6, 5, 5, generator=generator)
    probability = torch.softmax(torch.randn(1, 4, 5, 5, generator=generator), 1)
    weight = torch.randn(4, 6, 1, 1, generator=generator)
    state = {
        "consensus": probability.clone(),
        "consensus_reliability": torch.ones(1, 1, 5, 5),
    }
    fixed = analytical_virtual_correction(
        feature,
        probability,
        state,
        weight,
        torch.tensor([1.0]),
        torch.tensor([1.0]),
    )
    assert torch.equal(fixed["update"], torch.zeros_like(fixed["update"]))

    other = probabilities(seed=9)
    consensus = leave_one_out_consensus(other, "stage2", (5, 5))
    immature = analytical_virtual_correction(
        torch.randn(2, 6, 5, 5, generator=generator),
        other["stage2"],
        consensus,
        weight,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )
    assert torch.equal(
        immature["update"], torch.zeros_like(immature["update"])
    )


def test_error_mismatch_detach_and_update_bound():
    generator = torch.Generator().manual_seed(12)
    values = probabilities(seed=12)
    consensus = leave_one_out_consensus(values, "stage2", (5, 5))
    feature = torch.randn(2, 6, 5, 5, generator=generator, requires_grad=True)
    weight = torch.randn(4, 6, 1, 1, generator=generator, requires_grad=True)
    result = analytical_virtual_correction(
        feature,
        values["stage2"],
        consensus,
        weight,
        torch.tensor([0.6], requires_grad=True),
        torch.tensor([0.7], requires_grad=True),
    )
    assert torch.allclose(
        result["semantic_error"].sum(dim=1),
        torch.zeros(2, 5, 5),
        atol=1e-6,
    )
    assert torch.all(result["mismatch"] >= 0)
    assert torch.all(result["mismatch"] <= 1 + 1e-6)
    assert result["update_ratio"].max().item() <= 0.05 + 1e-6
    for key in (
        "consensus",
        "consensus_reliability",
        "semantic_error",
        "mismatch",
        "backprojection",
        "normalized_direction",
        "feature_scale",
        "maturity",
        "delta",
        "update",
        "updated_feature",
        "update_ratio",
    ):
        assert not result[key].requires_grad


def test_streaming_metric_is_exact_official_iouutils():
    ground_truth = [
        torch.tensor(
            [[0, 0, 1, 4], [0, 2, 1, 4], [3, 2, 1, 4]], dtype=torch.uint8
        ).numpy(),
        torch.tensor(
            [[3, 3, 2, 4], [0, 1, 2, 4], [0, 1, 2, 4]], dtype=torch.uint8
        ).numpy(),
    ]
    predictions = [
        torch.tensor(
            [[0, 1, 1, 0], [0, 2, 3, 1], [3, 2, 1, 2]], dtype=torch.uint8
        ).numpy(),
        torch.tensor(
            [[3, 1, 2, 0], [0, 1, 0, 1], [2, 1, 2, 3]], dtype=torch.uint8
        ).numpy(),
    ]
    official = iouutils.scores(
        [value.copy() for value in ground_truth],
        [value.copy() for value in predictions],
        n_class=4,
    )
    accumulator = OfficialMetricAccumulator()
    for truth, prediction in zip(ground_truth, predictions):
        accumulator.update(truth, prediction)
    streamed = accumulator.score()
    for key in ("Pixel Accuracy", "Mean Accuracy", "Frequency Weighted IoU", "Mean IoU", "Mean Dice"):
        assert streamed[key] == official[key]
    assert streamed["Class IoU"] == official["Class IoU"]
    assert streamed["Dice Coefficients"] == official["Dice Coefficients"]
