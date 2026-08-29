import numpy as np
import torch

from tools.analyze_rddr_phase2a import (
    ContextAccumulator,
    QuintileAccumulator,
    checkpoint_for_epoch,
    phase2a_decision,
)


def _gates(a, b, c, d):
    return {
        name: {"pass": value}
        for name, value in zip(("A", "B", "C", "D"), (a, b, c, d))
    }


def test_preregistered_decision_priority():
    assert phase2a_decision(_gates(True, True, True, True)) == "RDDR_PHASE2A_GO"
    assert (
        phase2a_decision(_gates(True, False, True, True))
        == "CONTEXT_REDUCTION_WORKS_SPATIAL_SPECIFICITY_FAIL"
    )
    assert (
        phase2a_decision(_gates(True, True, False, True))
        == "CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE"
    )
    assert (
        phase2a_decision(_gates(False, False, True, True))
        == "LOCAL_CH_HARM_REDUCED_NO_GLOBAL_GAIN"
    )
    assert phase2a_decision(_gates(False, False, True, False)) == "RDDR_PHASE2A_NOGO"
    assert (
        phase2a_decision(_gates(True, True, True, True), full=False)
        == "RDDR_PHASE2A_SMOKE_ONLY"
    )


def test_context_strength_accumulator():
    accumulator = ContextAccumulator("RCS")
    before = torch.full((1, 3, 2, 2), 2.0)
    reliability = torch.tensor([[[[1.0, 0.5], [0.25, 0.0]]]])
    accumulator.update(
        {
            "q": 1.0 - reliability,
            "reliability": reliability,
            "context_before": before,
            "context_after": before * reliability,
        }
    )
    result = accumulator.result()
    assert result["pixels"] == 4
    assert result["mean_reliability"] == 0.4375
    assert result["mean_suppression"] == 0.5625
    assert 0.0 < result["context_rms_ratio"] < 1.0


def test_fixed_c0_quintile_analysis():
    accumulator = QuintileAccumulator()
    truth = np.zeros((10, 10), dtype=np.uint8)
    c0 = np.zeros_like(truth)
    gs = c0.copy()
    rcs = c0.copy()
    gs.ravel()[::7] = 1
    rcs.ravel()[::11] = 1
    predictions = {
        "C0": {"Final": c0},
        "GS": {"Final": gs},
        "RCS": {"Final": rcs},
    }
    q_feature = torch.linspace(0.0, 1.0, 25).reshape(1, 1, 5, 5)
    q_full = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    canonical = {}
    for name, reliability in (
        ("C0", torch.ones(1, 1, 1, 1)),
        ("GS", torch.full((1, 1, 1, 1), 0.5)),
        ("RCS", 1.0 - q_feature),
    ):
        before = torch.ones(1, 3, 5, 5)
        values = {
            "q": q_feature,
            "reliability": reliability,
            "context_before": before,
            "context_after": before * reliability,
        }
        canonical[name] = (values, q_full, c0, c0)
    accumulator.update(truth, predictions, canonical)
    result = accumulator.result()
    assert len(result["rows"]) == 10
    assert len(result["thresholds_full"]) == 4
    assert len(result["thresholds_feature"]) == 4
    assert all(np.isfinite(row["context_rms_ratio"]) for row in result["rows"])
    rcs_rows = [row for row in result["rows"] if row["variant"] == "RCS"]
    assert rcs_rows[0]["mean_reliability"] > rcs_rows[-1]["mean_reliability"]


def test_checkpoint_contract(tmp_path):
    assert checkpoint_for_epoch(tmp_path, 25).name == "stage1_last.pth"
    assert checkpoint_for_epoch(tmp_path, 5).name == "stage1_epoch_0005.pth"
