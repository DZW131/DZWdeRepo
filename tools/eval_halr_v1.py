#!/usr/bin/env python3
"""Epoch25 FINAL BCSS validation comparison for HALR-v1 and SSHR A0."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net_CAM
from tool.infer_halr import diagnose_gt_present_hierarchy, infer_bcss
from tools.train_halr_v1_25ep import sha256_file, write_json


EXPECTED_A0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def compact(scores):
    return {
        "mIoU": float(scores["Mean IoU"]), "mDice": float(scores["Mean Dice"]),
        "class_iou": {str(i): float(scores["Class IoU"][i]) for i in range(4)},
        "class_dice": {str(i): float(scores["Dice Coefficients"][i]) for i in range(4)},
        "pixel_accuracy": float(scores["Pixel Accuracy"]),
        "mean_accuracy": float(scores["Mean Accuracy"]),
        "frequency_weighted_iou": float(scores["Frequency Weighted IoU"]),
    }


def plot_history(history, output):
    output.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(epochs, [row["total_loss"] for row in history], label="total")
    ax.plot(epochs, [row["classification_loss"] for row in history], label="classification")
    ax.plot(epochs, [row["cvle_weighted"] for row in history], label="weighted CVLE")
    ax.plot(epochs, [row["rahd_weighted"] for row in history], label="weighted RAHD")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("HALR-v1 training losses")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "training_losses.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(epochs, [row["jsd28"] for row in history], label="JSD28")
    axes[0].plot(epochs, [row["jsddeep"] for row in history], label="JSDdeep")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-view JSD")
    axes[0].set_title("Cross-view stability"); axes[0].legend()
    axes[1].plot(epochs, [row["weight28"] for row in history], label="w28")
    axes[1].plot(epochs, [row["weightdeep"] for row in history], label="wdeep")
    axes[1].plot(
        epochs, [row["fraction_weight28_gt"] for row in history],
        label="fraction w28>wdeep", linestyle="--",
    )
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("weight / fraction")
    axes[1].set_title("Reliability-adaptive teacher"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "teacher_dynamics.png", dpi=180); plt.close(fig)


def render_report(result, history, output):
    a0 = result["models"]["sshr_a0"]; halr = result["models"]["halr_v1"]
    selected = {row["epoch"]: row for row in history}
    diagnosis = result["teacher_diagnosis"]
    lines = [
        "# HALR-v1 Full Model — BCSS Seed42 Epoch25 FINAL", "",
        "## 1. Executive result", "",
        f"- Decision: **{result['decision']}**",
        f"- ΔmIoU: **{result['delta']['mIoU_pp']:+.4f} pp**",
        f"- ΔmDice: **{result['delta']['mDice_pp']:+.4f} pp**",
        "- Epoch25 FINAL is the only primary checkpoint; no validation selection occurred.",
        "", "## 2. Frozen experimental control", "",
        "- Fresh ImageNet-pretrained clean official SSHR; no trained checkpoint was loaded.",
        "- BCSS seed42, 25 epochs, effective base batch20, 224×224, BF16, released optimizer and augmentation.",
        "- HALR adds zero model parameters and zero inference operations.",
        "- CVLE and RAHD use only CAM28_1/CAMdeep plus image-level present labels during training.",
        "- λCV=λHD=0.05 and the epoch1–5 ramp were frozen; no tuning occurred.",
        "", "## 3. Epoch25 validation", "",
        "| Model | Epoch | mIoU | mDice | C0 | C1 | C2 | C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (("SSHR A0", a0), ("HALR-v1", halr)):
        lines.append(
            f"| {name} | 25 | {100*value['mIoU']:.4f} | {100*value['mDice']:.4f} | "
            + " | ".join(f"{100*value['class_iou'][str(i)]:.4f}" for i in range(4)) + " |"
        )
    lines += ["", "| Quantity | Delta (pp) |", "|---|---:|",
              f"| mIoU | {result['delta']['mIoU_pp']:+.4f} |",
              f"| mDice | {result['delta']['mDice_pp']:+.4f} |"]
    for index in range(4):
        lines.append(f"| C{index} IoU | {result['delta']['class_iou_pp'][str(index)]:+.4f} |")
    lines += [
        "", "## 4. Teacher dynamics", "",
        "| Epoch | JSD28 | JSDdeep | w28 | wdeep | w28>wdeep fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in (5, 10, 15, 20, 25):
        row = selected[epoch]
        lines.append(
            f"| {epoch} | {row['jsd28']:.8f} | {row['jsddeep']:.8f} | "
            f"{row['weight28']:.8f} | {row['weightdeep']:.8f} | "
            f"{row['fraction_weight28_gt']:.8f} |"
        )
    lines += ["", "## 5. Epoch25 mechanism diagnosis", ""]
    for name in ("foreground", "boundary", "interior"):
        row = diagnosis["regions"][name]
        lines.append(
            f"- {name}: Deep {100*row['deep_accuracy']:.4f}%, "
            f"CAM28_1 {100*row['raw28_1_accuracy']:.4f}%, "
            f"Deep advantage {row['deep_advantage_pp']:+.4f} pp."
        )
    lines += [
        f"- Boundary definition: {diagnosis['boundary_definition']}.",
        "- GT masks were applied only after network forward and never entered training or inference decisions.",
        "", "## 6. Runtime and resources", "",
        f"- SSHR / HALR parameters: {result['parameters']['sshr_total']:,} / {result['parameters']['halr_total']:,}",
        f"- Added parameters: {result['parameters']['new_parameters']}",
        f"- Mean seconds/epoch: {result['runtime']['mean_training_seconds_per_epoch']:.2f}",
        f"- Peak training CUDA memory: {result['runtime']['training_peak_cuda_memory_gib']:.3f} GiB",
        f"- A0 / HALR inference seconds per image: {result['runtime']['a0_seconds_per_image']:.6f} / {result['runtime']['halr_seconds_per_image']:.6f}",
        "", "## 7. Provenance", "",
        f"- Training source commit: `{result['provenance']['training_source_commit']}`",
        f"- Evaluation source commit: `{result['provenance']['evaluation_source_commit']}`",
        f"- A0 checkpoint SHA256: `{result['provenance']['a0_checkpoint_sha256']}`",
        f"- HALR checkpoint SHA256: `{result['provenance']['halr_checkpoint_sha256']}`",
        f"- Training config SHA256: `{result['provenance']['training_config_sha256']}`",
        "- Validation pairs: 3,418; BF16; official 3-way TTA, thresholds, class gate, min-max, 0/0.6/0.2/0.2 fusion and released metric.",
        "", "## 8. Figures", "",
        "![Training losses](../figures/training_losses.png)",
        "![Teacher dynamics](../figures/teacher_dynamics.png)",
        "", "## 9. Stop boundary", "",
        "No test, LUAD, seeds 11/17, ablation, lambda/ramp sweep or HALR-v2 was run.",
        "", f"**{result['decision']}**", "", "STOP.",
    ]
    report = output / "docs" / "halr_v1_full_25ep_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--halr-checkpoint", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("HALR-v1 formal evaluation is BCSS validation only")
    val_root = Path(args.val_root)
    if len(list((val_root / "img").glob("*.png"))) != 3418 or len(list((val_root / "mask").glob("*.png"))) != 3418:
        raise AssertionError("Expected exactly 3418 BCSS validation pairs")
    output = Path(args.experiment_dir)
    if (output / "validation" / "final_comparison.json").exists():
        raise FileExistsError("Formal HALR-v1 evaluation already exists")
    for directory in ("validation/sshr_a0", "validation/halr_v1_epoch25", "figures", "docs"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    a0_sha = sha256_file(args.a0_checkpoint)
    if a0_sha != EXPECTED_A0_SHA256:
        raise AssertionError(f"A0 SHA mismatch: {a0_sha}")
    halr_sha = sha256_file(args.halr_checkpoint)
    manifest = json.loads((output / "provenance" / "manifest.json").read_text(encoding="utf-8"))
    history = json.loads((output / "train" / "history.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "train" / "training_complete.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((output / "checkpoints" / "manifest.json").read_text(encoding="utf-8"))
    final_meta = next(item for item in checkpoints if item["epoch"] == 25)
    if final_meta["sha256"] != halr_sha or not final_meta["primary_final"]:
        raise AssertionError("Epoch25 primary checkpoint provenance mismatch")

    a0_model = Net_CAM(4); a0_model.load_state_dict(load_state(args.a0_checkpoint), strict=True)
    a0_raw, a0_runtime = infer_bcss(a0_model, str(val_root), "bf16", args.num_workers)
    a0 = compact(a0_raw)
    write_json(output / "validation" / "sshr_a0" / "metrics.json", {"scores": a0, "runtime": a0_runtime})
    del a0_model; torch.cuda.empty_cache()

    halr_model = Net_CAM(4); halr_model.load_state_dict(load_state(args.halr_checkpoint), strict=True)
    halr_raw, halr_runtime = infer_bcss(halr_model, str(val_root), "bf16", args.num_workers)
    halr = compact(halr_raw)
    diagnosis = diagnose_gt_present_hierarchy(halr_model, str(val_root), "bf16", args.num_workers)
    write_json(output / "validation" / "halr_v1_epoch25" / "metrics.json", {
        "scores": halr, "runtime": halr_runtime, "teacher_diagnosis": diagnosis,
    })

    delta_miou = 100 * (halr["mIoU"] - a0["mIoU"])
    delta_mdice = 100 * (halr["mDice"] - a0["mDice"])
    class_delta = {
        str(i): 100 * (halr["class_iou"][str(i)] - a0["class_iou"][str(i)])
        for i in range(4)
    }
    if delta_miou >= 0.50:
        decision = "HALR_V1_MAJOR_SUCCESS"
    elif delta_miou >= 0.30:
        decision = "HALR_V1_STRONG_SUCCESS"
    elif delta_miou >= 0.15:
        decision = "HALR_V1_CLEAR_SUCCESS"
    elif delta_miou >= 0.05:
        decision = "HALR_V1_POSITIVE"
    else:
        decision = "HALR_V1_NO_CLEAR_GAIN"
    evaluation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    result = {
        "decision": decision,
        "models": {"sshr_a0": a0, "halr_v1": halr},
        "delta": {"mIoU_pp": delta_miou, "mDice_pp": delta_mdice, "class_iou_pp": class_delta},
        "teacher_diagnosis": diagnosis, "parameters": manifest["parameters"],
        "runtime": {
            "mean_training_seconds_per_epoch": completion["mean_epoch_seconds"],
            "training_peak_cuda_memory_gib": completion["peak_cuda_memory_gib"],
            "a0_seconds_per_image": a0_runtime["seconds_per_image"],
            "halr_seconds_per_image": halr_runtime["seconds_per_image"],
        },
        "training_manifest": manifest,
        "provenance": {
            "training_source_commit": manifest["halr_source_commit"],
            "evaluation_source_commit": evaluation_commit,
            "training_config_sha256": manifest["config_sha256"],
            "a0_checkpoint_sha256": a0_sha, "halr_checkpoint_sha256": halr_sha,
            "validation_images": 3418, "precision": "bf16",
            "test_used": False, "luad_used": False,
            "checkpoint_selection": "epoch25 FINAL only",
            "evaluation_command": " ".join(sys.argv),
        },
    }
    plot_history(history, output / "figures")
    write_json(output / "validation" / "final_comparison.json", result)
    report = render_report(result, history, output)
    print(json.dumps({"decision": decision, "delta": result["delta"], "report": str(report)}, indent=2, sort_keys=True), flush=True)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
