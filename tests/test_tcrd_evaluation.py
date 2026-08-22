import numpy as np
import json
from types import SimpleNamespace

from tools.eval_tcrd_utility import compare_boundary, present_confusion, run
from tools.tcrd_common import BRANCH_DIRS


def test_present_confusion_accepts_official_background_overwrite():
    truth = np.asarray([[[0, 1], [4, 4]]], dtype=np.uint8)
    prediction = np.asarray([[[1, 1], [4, 4]]], dtype=np.uint8)
    result = present_confusion(prediction, truth)
    assert result["wrong_pixels"] == 1
    assert result["wrong_pixels_by_true_class"] == [1, 0, 0, 0]


def test_boundary_comparison_reports_recovered_minus_harmed_net():
    truth = np.zeros((1, 28, 28), dtype=np.uint8)
    base = truth.copy()
    candidate = truth.copy()
    base[0, 0, 0] = 1
    candidate[0, 0, 1] = 1
    bins = compare_boundary(base, candidate, truth)
    totals = [values["net"] for values in bins.values()]
    assert sum(totals) == 0


def test_finalizer_emits_machine_readable_route_and_report(tmp_path):
    truth = np.zeros((2, 28, 28), dtype=np.uint8)
    truth[:, 14:] = 1
    prediction = truth.copy()
    prediction[:, :2, :2] = 4
    image_ids = np.asarray(["a", "b"])
    matrix = [[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0],
              [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
    diagnostics = {
        "z0_rms": 1.0,
        "diffusion_update_rms": 0.01,
        "diffusion_update_ratio": 0.01,
        "reaction_update_rms": 0.01,
        "reaction_update_ratio": 0.01,
        "conductance_same_mean": 0.11,
        "conductance_cross_mean": 0.10,
        "conductance_same_cross_ratio": 1.1,
        "present_entropy_z0": 0.6,
        "present_entropy_zt": 0.5,
        "present_top1_top2_margin_z0": 0.2,
        "present_top1_top2_margin_zt": 0.3,
    }
    scores = {
        "mIoU": 0.6, "mDice": 0.7,
        "class_iou": {str(index): 0.6 for index in range(4)},
    }
    for branch in ("C0", "D", "R", "DR"):
        branch_dir = tmp_path / BRANCH_DIRS[branch]
        (branch_dir / "validation").mkdir(parents=True)
        (branch_dir / "predictions").mkdir()
        history = []
        for epoch, point in enumerate(("step0", "epoch1", "epoch2", "epoch3", "epoch4", "epoch5")):
            history.append({
                "point": point, "epoch": epoch,
                "scores": scores, "standalone_cam28_1": scores,
                "diagnostics": diagnostics,
                "mechanism_parameters": {
                    "eta_r": 0.1,
                    "competition_matrix": matrix if branch in ("R", "DR") else None,
                },
            })
        (branch_dir / "validation" / "history.json").write_text(
            json.dumps(history), encoding="utf-8"
        )
        (branch_dir / "complete.json").write_text(
            json.dumps({"status": "complete"}), encoding="utf-8"
        )
        np.savez_compressed(
            branch_dir / "predictions" / "epoch5_validation.npz",
            image_ids=image_ids, predictions=prediction, truths=truth,
        )
    run(SimpleNamespace(experiment_dir=str(tmp_path)))
    decision = json.loads(
        (tmp_path / "comparison" / "route_decision.json").read_text(encoding="utf-8")
    )
    assert decision["route"] == "ROUTE_E_CLOSE"
    assert (tmp_path / "docs" / "tcrd_v0_utility_gate_report.md").is_file()
