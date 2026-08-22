#!/usr/bin/env python3
"""Epoch25 FINAL BCSS validation comparison for MATR-v1 and SSHR A0."""

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

from network.resnet38_cls import Net_CAM as A0Net
from network.resnet38_cls_matr import Net_CAM as MATRNet
from tool.infer_matr import infer_bcss
from tools.train_matr_25ep import sha256_file, write_json


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
    ax.plot(epochs, [row["ot_weighted"] for row in history], label="weighted OT")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("MATR-v1 training losses")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "training_losses.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for index in range(4):
        axes[0].plot(epochs, [row["mode_cosine"][index] for row in history], label=f"C{index}")
        axes[1].plot(epochs, [row["mode_activation_ratio"][index] for row in history], label=f"C{index}")
    axes[0].set_title("Mode cosine"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].set_title("Mode-0 activation ratio"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output / "prototype_dynamics.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    axes[0].plot(epochs, [row["gamma_adapt"] for row in history])
    axes[0].set_title("gamma_adapt")
    axes[1].plot(epochs, [row["mean_abs_offset"] for row in history], label="mean")
    axes[1].plot(epochs, [row["p95_abs_offset"] for row in history], label="p95")
    axes[1].set_title("Absolute offset"); axes[1].legend()
    axes[2].plot(epochs, [row["delta_context15_ratio"] for row in history])
    axes[2].set_title("Delta_C / C15 RMS")
    for ax in axes:
        ax.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(output / "sacr_dynamics.png", dpi=180); plt.close(fig)


def render_report(result, history, output):
    a0 = result["models"]["sshr_a0"]; matr = result["models"]["matr_v1"]
    final = history[-1]
    lines = [
        "# MATR-v1 Full Model — BCSS Seed42 Epoch25 FINAL", "",
        "## 1. Executive result", "",
        f"- Decision: **{result['decision']}**",
        f"- Class safety: **{result['class_safety']}**",
        f"- ΔmIoU: **{result['delta']['mIoU_pp']:+.4f} pp**",
        f"- ΔmDice: **{result['delta']['mDice_pp']:+.4f} pp**",
        "- Epoch25 FINAL is the only primary checkpoint; no validation selection occurred.",
        "", "## 2. Frozen experimental control", "",
        "- Fresh ImageNet-pretrained clean official SSHR; no trained checkpoint was loaded.",
        "- BCSS seed42, 25 epochs, batch20, 224×224, BF16, released optimizer and augmentation.",
        "- Only OT-MTR and SACR alter HFRM28_1/CAM28_1; all other SSHR branches and official inference are frozen.",
        "- OT λ=0.05, two modes, epsilon=0.1, 20 Sinkhorn iterations and the epoch1–5 ramp were not tuned.",
        "", "## 3. Epoch25 validation", "",
        "| Model | Epoch | mIoU | mDice | C0 | C1 | C2 | C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (("SSHR A0", a0), ("MATR-v1", matr)):
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
        "", "## 4. Standalone CAM28_1", "",
        "| Model | CAM28_1 mIoU | CAM28_1 mDice |",
        "|---|---:|---:|",
        f"| SSHR A0 | {100*result['standalone_cam28_1']['sshr_a0']['mIoU']:.4f} | {100*result['standalone_cam28_1']['sshr_a0']['mDice']:.4f} |",
        f"| MATR-v1 | {100*result['standalone_cam28_1']['matr_v1']['mIoU']:.4f} | {100*result['standalone_cam28_1']['matr_v1']['mDice']:.4f} |",
        "", "## 5. OT-MTR mechanism", "",
        "| Class | Mode cosine | Mode-0 activation | Mean seeds | Transport mass mode0/mode1 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for index in range(4):
        mass = final["ot_transport_mass_by_class"][index]
        lines.append(
            f"| C{index} | {final['mode_cosine'][index]:.8f} | "
            f"{final['mode_activation_ratio'][index]:.8f} | "
            f"{final['ot_mean_seeds_by_class'][index]:.4f} | "
            f"{mass[0]:.6f}/{mass[1]:.6f} |"
        )
    lines += [
        f"", f"- Epoch25 valid OT image-class pairs: {final['ot_valid_pairs']:.0f}.",
        f"- Epoch25 raw/weighted OT loss: {final['ot_loss']:.8f} / {final['ot_weighted']:.8f}.",
        "", "## 6. SACR mechanism", "",
        f"- gamma_adapt: {final['gamma_adapt']:.8f}",
        f"- Mean / p95 absolute offset: {final['mean_abs_offset']:.8f} / {final['p95_abs_offset']:.8f}",
        f"- Mean modulation: {final['mean_modulation']:.8f}",
        f"- Delta_C RMS / C15 RMS: {final['delta_context_rms']:.8f} / {final['context15_rms']:.8f}",
        f"- Delta_C/C15 RMS: {final['delta_context15_ratio']:.8f}",
        "", "## 7. Runtime and parameters", "",
        f"- SSHR / MATR parameters: {result['parameters']['sshr_total']:,} / {result['parameters']['matr_total']:,}",
        f"- Added parameters: {result['parameters']['new_parameters']:,} ({result['parameters']['overhead_percent']:.6f}%)",
        f"- Mean seconds/epoch: {result['runtime']['mean_training_seconds_per_epoch']:.2f}",
        f"- Peak training CUDA memory: {result['runtime']['training_peak_cuda_memory_gib']:.3f} GiB",
        f"- A0 / MATR inference seconds per image: {result['runtime']['a0_seconds_per_image']:.6f} / {result['runtime']['matr_seconds_per_image']:.6f}",
        "", "## 8. Provenance", "",
        f"- Training source commit: `{result['provenance']['training_source_commit']}`",
        f"- Evaluation source commit: `{result['provenance']['evaluation_source_commit']}`",
        f"- A0 checkpoint SHA256: `{result['provenance']['a0_checkpoint_sha256']}`",
        f"- MATR checkpoint SHA256: `{result['provenance']['matr_checkpoint_sha256']}`",
        f"- Training config SHA256: `{result['provenance']['training_config_sha256']}`",
        "- Validation pairs: 3,418; BF16; official 3-way TTA, thresholds, class gate, min-max, 0/0.6/0.2/0.2 fusion and released metric.",
        "", "## 9. Figures", "",
        "![Training losses](../figures/training_losses.png)",
        "![Prototype dynamics](../figures/prototype_dynamics.png)",
        "![SACR dynamics](../figures/sacr_dynamics.png)",
        "", "## 10. Stop boundary", "",
        "No test, LUAD, seeds 11/17, ablation, mode/lambda/epsilon/offset sweep, diversity loss or MATR-v2 was run.",
        "", f"**{result['decision']}**", "", "STOP.",
    ]
    report = output / "docs" / "matr_v1_full_25ep_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--matr-checkpoint", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("MATR formal evaluation is BCSS validation only")
    val_root = Path(args.val_root)
    if len(list((val_root / "img").glob("*.png"))) != 3418 or len(list((val_root / "mask").glob("*.png"))) != 3418:
        raise AssertionError("Expected exactly 3418 BCSS validation pairs")
    output = Path(args.experiment_dir)
    if (output / "validation" / "final_comparison.json").exists():
        raise FileExistsError("Formal MATR evaluation already exists")
    for directory in ("validation/sshr_a0", "validation/matr_v1_epoch25", "figures", "docs"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    a0_sha = sha256_file(args.a0_checkpoint)
    if a0_sha != EXPECTED_A0_SHA256:
        raise AssertionError(f"A0 SHA mismatch: {a0_sha}")
    matr_sha = sha256_file(args.matr_checkpoint)
    manifest = json.loads((output / "provenance" / "manifest.json").read_text(encoding="utf-8"))
    history = json.loads((output / "train" / "history.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "train" / "training_complete.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((output / "checkpoints" / "manifest.json").read_text(encoding="utf-8"))
    final_meta = next(item for item in checkpoints if item["epoch"] == 25)
    if final_meta["sha256"] != matr_sha or not final_meta["primary_final"]:
        raise AssertionError("Epoch25 primary checkpoint provenance mismatch")

    a0_model = A0Net(4); a0_model.load_state_dict(load_state(args.a0_checkpoint), strict=True)
    a0_raw, a0_standalone_raw, a0_runtime = infer_bcss(a0_model, str(val_root), "bf16", args.num_workers)
    a0, a0_standalone = compact(a0_raw), compact(a0_standalone_raw)
    write_json(output / "validation" / "sshr_a0" / "metrics.json", {
        "scores": a0, "standalone_cam28_1": a0_standalone, "runtime": a0_runtime,
    })
    del a0_model; torch.cuda.empty_cache()

    matr_model = MATRNet(4); matr_model.load_state_dict(load_state(args.matr_checkpoint), strict=True)
    matr_raw, matr_standalone_raw, matr_runtime = infer_bcss(matr_model, str(val_root), "bf16", args.num_workers)
    matr, matr_standalone = compact(matr_raw), compact(matr_standalone_raw)
    write_json(output / "validation" / "matr_v1_epoch25" / "metrics.json", {
        "scores": matr, "standalone_cam28_1": matr_standalone, "runtime": matr_runtime,
    })

    delta_miou = 100 * (matr["mIoU"] - a0["mIoU"])
    delta_mdice = 100 * (matr["mDice"] - a0["mDice"])
    class_delta = {
        str(i): 100 * (matr["class_iou"][str(i)] - a0["class_iou"][str(i)])
        for i in range(4)
    }
    if delta_miou >= 1.00:
        decision = "MATR_V1_BREAKTHROUGH"
    elif delta_miou >= 0.50:
        decision = "MATR_V1_MAJOR_SUCCESS"
    elif delta_miou >= 0.30:
        decision = "MATR_V1_STRONG_SUCCESS"
    elif delta_miou >= 0.15:
        decision = "MATR_V1_CLEAR_SUCCESS"
    elif delta_miou >= 0.05:
        decision = "MATR_V1_POSITIVE"
    else:
        decision = "MATR_V1_NO_CLEAR_GAIN"
    class_safety = (
        "MATR_CLASS_REGRESSION_REVIEW"
        if delta_miou >= 0.05 and any(value <= -0.50 for value in class_delta.values())
        else "MATR_CLASS_SAFETY_NOT_TRIGGERED"
    )
    evaluation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    result = {
        "decision": decision, "class_safety": class_safety,
        "models": {"sshr_a0": a0, "matr_v1": matr},
        "standalone_cam28_1": {"sshr_a0": a0_standalone, "matr_v1": matr_standalone},
        "delta": {"mIoU_pp": delta_miou, "mDice_pp": delta_mdice, "class_iou_pp": class_delta},
        "parameters": manifest["parameters"],
        "runtime": {
            "mean_training_seconds_per_epoch": completion["mean_epoch_seconds"],
            "training_peak_cuda_memory_gib": completion["peak_cuda_memory_gib"],
            "a0_seconds_per_image": a0_runtime["seconds_per_image"],
            "matr_seconds_per_image": matr_runtime["seconds_per_image"],
        },
        "training_manifest": manifest, "final_mechanism": completion["final_mechanism"],
        "provenance": {
            "training_source_commit": manifest["matr_source_commit"],
            "evaluation_source_commit": evaluation_commit,
            "training_config_sha256": manifest["config_sha256"],
            "a0_checkpoint_sha256": a0_sha, "matr_checkpoint_sha256": matr_sha,
            "validation_images": 3418, "precision": "bf16",
            "test_used": False, "luad_used": False,
            "checkpoint_selection": "epoch25 FINAL only",
            "evaluation_command": " ".join(sys.argv),
        },
    }
    plot_history(history, output / "figures")
    write_json(output / "validation" / "final_comparison.json", result)
    report = render_report(result, history, output)
    print(json.dumps({
        "decision": decision, "class_safety": class_safety,
        "delta": result["delta"], "report": str(report),
    }, indent=2, sort_keys=True), flush=True)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
