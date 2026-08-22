#!/usr/bin/env python3
"""Evaluate only the epoch-25 FINAL TCER-R checkpoint on BCSS validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls_tcrd_gate import Net
from tool.infer_tcrd import infer_bcss
from tools.eval_tcrd_utility import compare_present
from tools.tcer_r_full25_common import (
    A0_REFERENCE, EXPECTED_A0_SHA256, compact_scores,
    exploratory_decision, sha256_file, write_json,
)


def load_model(branch, checkpoint):
    model = Net(4, branch=branch)
    state = torch_load(checkpoint)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(incompat)
    return model.cuda()


def torch_load(path):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def run(args):
    if args.num_workers != 4:
        raise AssertionError("Frozen exploratory protocol requires num_workers=4")
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("Validation-only path guard failed")
    if sha256_file(args.a0_checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("A0 checkpoint SHA mismatch")
    run_dir = Path(args.run_dir)
    completion = json.loads(
        (run_dir / "training_complete.json").read_text(encoding="utf-8")
    )
    checkpoint = run_dir / "stage1_last.pth"
    if completion["checkpoint_sha256"] != sha256_file(checkpoint):
        raise AssertionError("Final checkpoint SHA mismatch")

    predictions = run_dir / "predictions"
    a0_model = load_model("C0", args.a0_checkpoint)
    a0_fused, a0_cam, a0_runtime, a0_diag = infer_bcss(
        a0_model, args.val_root, "bf16", args.num_workers,
        predictions / "a0_validation.npz",
    )
    del a0_model
    import torch
    torch.cuda.empty_cache()
    r_model = load_model("R", checkpoint)
    r_fused, r_cam, r_runtime, r_diag = infer_bcss(
        r_model, args.val_root, "bf16", args.num_workers,
        predictions / "tcer_r25_validation.npz",
    )

    with np.load(predictions / "a0_validation.npz") as data:
        a0_prediction = data["predictions"].copy()
        truths = data["truths"].copy()
        ids = data["image_ids"].copy()
    with np.load(predictions / "tcer_r25_validation.npz") as data:
        r_prediction = data["predictions"].copy()
        if not np.array_equal(ids, data["image_ids"]):
            raise AssertionError("Validation order mismatch")
        if not np.array_equal(truths, data["truths"]):
            raise AssertionError("Validation truth mismatch")
    confusion = compare_present(a0_prediction, r_prediction, truths)

    a0_scores = compact_scores(a0_fused)
    r_scores = compact_scores(r_fused)
    a0_cam_scores = compact_scores(a0_cam)
    r_cam_scores = compact_scores(r_cam)
    final_delta = 100 * (r_scores["mIoU"] - a0_scores["mIoU"])
    dice_delta = 100 * (r_scores["mDice"] - a0_scores["mDice"])
    cam_delta = 100 * (r_cam_scores["mIoU"] - a0_cam_scores["mIoU"])
    decision = exploratory_decision(
        final_delta, cam_delta, confusion["relative_reduction"]
    )
    history = json.loads(
        (run_dir / "training_history.json").read_text(encoding="utf-8")
    )
    result = {
        "decision": decision,
        "scientific_status": "post-gate exploratory; not preregistered confirmation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "a0": {"scores": a0_scores, "cam28_1": a0_cam_scores,
               "runtime": a0_runtime, "diagnostics": a0_diag},
        "tcer_r25": {"scores": r_scores, "cam28_1": r_cam_scores,
                     "runtime": r_runtime, "diagnostics": r_diag},
        "delta_mIoU_pp": final_delta,
        "delta_mDice_pp": dice_delta,
        "delta_cam28_1_mIoU_pp": cam_delta,
        "present_confusion": confusion,
        "criteria": {
            "final_mIoU_delta_ge_0_15": final_delta >= 0.15,
            "cam28_1_delta_ge_0_20": cam_delta >= 0.20,
            "present_confusion_reduction_ge_0_5pct": confusion["relative_reduction"] >= 0.005,
        },
        "reference_consistency": {
            "expected": A0_REFERENCE,
            "observed": {
                "mIoU": a0_scores["mIoU"], "mDice": a0_scores["mDice"],
                "cam28_1_mIoU": a0_cam_scores["mIoU"],
            },
        },
        "test_used": False, "luad_used": False,
        "checkpoint_selection": "none; epoch25 FINAL only",
    }
    write_json(run_dir / "validation_result.json", result)

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["classification_loss"] for row in history], marker="o")
    axes[0].set(xlabel="epoch", ylabel="classification loss", title="TCER-R training")
    axes[1].plot(epochs, [row["mechanism"]["eta_r"] for row in history], marker="o")
    axes[1].set(xlabel="epoch", ylabel="eta_R", title="Reaction strength")
    fig.tight_layout(); fig.savefig(figures / "training_dynamics.png", dpi=180); plt.close(fig)

    lines = [
        "# TCER R-only Fresh-25 Exploratory Report", "",
        "## Executive conclusion", "",
        f"- Decision: **{decision}**",
        "- Scientific status: post-gate exploratory run; it does not replace the preregistered ROUTE_E_CLOSE result.",
        "- Checkpoint rule: epoch-25 FINAL only; no validation checkpoint selection.",
        "- Test and LUAD were not run.", "",
        "## Final validation comparison", "",
        "| Model | mIoU | mDice | CAM28_1 mIoU |",
        "|---|---:|---:|---:|",
        f"| A0 | {100*a0_scores['mIoU']:.4f} | {100*a0_scores['mDice']:.4f} | {100*a0_cam_scores['mIoU']:.4f} |",
        f"| TCER R-only | {100*r_scores['mIoU']:.4f} | {100*r_scores['mDice']:.4f} | {100*r_cam_scores['mIoU']:.4f} |",
        f"| Delta | {final_delta:+.4f} pp | {dice_delta:+.4f} pp | {cam_delta:+.4f} pp |", "",
        "## Per-class IoU delta", "",
    ]
    for index in range(4):
        delta = 100 * (
            r_scores["class_iou"][str(index)] - a0_scores["class_iou"][str(index)]
        )
        lines.append(f"- Class {index}: {delta:+.4f} pp")
    lines += [
        "", "## Mechanism", "",
        f"- Present-confusion relative reduction: {100*confusion['relative_reduction']:+.4f}%.",
        f"- Reaction update RMS/Z0 RMS: {r_diag['reaction_update_ratio']:.6f}.",
        f"- Entropy Z0→ZT: {r_diag['present_entropy_z0']:.6f}→{r_diag['present_entropy_zt']:.6f}.",
        f"- Margin Z0→ZT: {r_diag['present_top1_top2_margin_z0']:.6f}→{r_diag['present_top1_top2_margin_zt']:.6f}.",
        f"- Final eta_R: {history[-1]['mechanism']['eta_r']:.6f}.", "",
        "## Frozen exploratory gate", "",
        f"- Final mIoU delta ≥ +0.15 pp: {final_delta >= 0.15}",
        f"- CAM28_1 delta ≥ +0.20 pp: {cam_delta >= 0.20}",
        f"- Present-confusion reduction ≥ 0.5%: {confusion['relative_reduction'] >= 0.005}", "",
        "## Provenance", "",
        f"- Final checkpoint SHA256: `{completion['checkpoint_sha256']}`",
        "- BCSS seed42, 25 epochs, batch20, BF16, official SSHR preprocessing/loss/optimizer/schedule.",
        "- Only the frozen R/TCER path was added.", "",
        "STOP. No automatic test, other seed, LUAD, or tuning follows.",
    ]
    report = run_dir / "tcer_r25_validation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result, flush=True)
    print(decision, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
