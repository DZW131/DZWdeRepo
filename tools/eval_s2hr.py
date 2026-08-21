#!/usr/bin/env python3
"""Final-only BCSS validation comparison for SSHR A0 and S²HR-v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net_CAM as SSHRNet
from network.resnet38_cls_s2hr import Net_CAM as S2HRNet
from tool.infer_s2hr import infer_bcss


EXPECTED_A0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def compact_scores(scores):
    return {
        "mIoU": float(scores["Mean IoU"]),
        "mDice": float(scores["Mean Dice"]),
        "class_iou": {str(index): float(scores["Class IoU"][index]) for index in range(4)},
        "class_dice": {
            str(index): float(scores["Dice Coefficients"][index]) for index in range(4)
        },
        "pixel_accuracy": float(scores["Pixel Accuracy"]),
        "mean_accuracy": float(scores["Mean Accuracy"]),
        "frequency_weighted_iou": float(scores["Frequency Weighted IoU"]),
    }


def _plot_history(history, figures):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key in ("total_loss", "loss_56", "loss_28_1", "loss_28_2", "loss_deep"):
        ax.plot(epochs, [row[key] for row in history], label=key)
    ax.set_xlabel("epoch"); ax.set_ylabel("classification loss"); ax.legend(ncol=2)
    fig.tight_layout(); fig.savefig(figures / "training_losses.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for key in ("gamma_global_28_1", "gamma_context_28_1", "gamma_spatial", "rho_boundary"):
        ax.plot(epochs, [row[key] for row in history], label=key)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("epoch"); ax.set_ylabel("learned mechanism value"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "mechanism_trajectory.png", dpi=180); plt.close(fig)


def render_report(result, history, output, commands):
    a0, s2 = result["models"]["sshr_a0"], result["models"]["s2hr_v1"]
    delta = result["delta"]
    final = history[-1]
    classes_a0 = [100 * a0["class_iou"][str(index)] for index in range(4)]
    classes_s2 = [100 * s2["class_iou"][str(index)] for index in range(4)]
    lines = [
        "# S²HR-v1 Full Model — BCSS Seed42 Final Report",
        "",
        "## 1. Experimental control",
        "",
        "- Base: frozen official SSHR A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.",
        f"- S²HR source commit: `{result['provenance']['source_commit']}`.",
        "- Only HFRM28_1 is reconstructed; HFRM56/HFRM28_2, backbone, heads, loss, optimizer, schedule, augmentation and released inference/metric are unchanged.",
        "- Fresh ImageNet-pretrained start, BCSS, seed42, batch20, 224×224, BF16, 25 epochs.",
        "- Primary checkpoint: epoch25 FINAL; no validation selection, early stop, test, LUAD, ablation or tuning.",
        "",
        "## 2. Exact commands",
        "",
        "```bash",
        commands["train"],
        "```",
        "",
        "```bash",
        commands["evaluation"],
        "```",
        "",
        "## 3. Final validation result",
        "",
        "| Model | Epoch | mIoU | mDice | C0 | C1 | C2 | C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| SSHR A0 seed42 | 25 | {100*a0['mIoU']:.4f} | {100*a0['mDice']:.4f} | "
        + " | ".join(f"{value:.4f}" for value in classes_a0) + " |",
        f"| S²HR-v1 Full | 25 | {100*s2['mIoU']:.4f} | {100*s2['mDice']:.4f} | "
        + " | ".join(f"{value:.4f}" for value in classes_s2) + " |",
        "",
        f"- ΔmIoU: **{delta['mIoU_pp']:+.4f} pp**",
        f"- ΔmDice: **{delta['mDice_pp']:+.4f} pp**",
        "- Per-class ΔIoU: " + ", ".join(
            f"C{index}={delta['class_iou_pp'][str(index)]:+.4f} pp" for index in range(4)
        ),
        f"- Primary decision: **{result['decision']}**",
        f"- Per-class safety: **{result['class_safety']}**",
        "",
        "## 4. Learned mechanism state",
        "",
        "| Parameter | Init | Epoch25 |",
        "|---|---:|---:|",
        f"| gamma_global_28_1 | 0 | {final['gamma_global_28_1']:.8f} |",
        f"| gamma_context_28_1 | 0 | {final['gamma_context_28_1']:.8f} |",
        f"| gamma_spatial | 0 | {final['gamma_spatial']:.8f} |",
        f"| rho_boundary | {torch.sigmoid(torch.tensor(-4.0)).item():.8f} | {final['rho_boundary']:.8f} |",
        "",
        f"- mean |Pd-Ps|: {final['semantic_discrepancy_abs_mean']:.8f}",
        f"- mean boundary fraction: {final['boundary_fraction']:.8f}",
        f"- mean CH gate(boundary): {final['ch_gate_boundary_mean']:.8f}",
        f"- mean CH gate(interior): {final['ch_gate_interior_mean']:.8f}",
        "",
        "## 5. Runtime and resources",
        "",
        f"- SSHR parameters: {result['parameters']['sshr_total']:,}",
        f"- S²HR parameters: {result['parameters']['s2hr_total']:,}",
        f"- New parameters: {result['parameters']['new_parameters']:,} ({100*result['parameters']['overhead_fraction']:.6f}%)",
        f"- Mean training seconds/epoch: {result['runtime']['mean_training_seconds_per_epoch']:.2f}",
        f"- Training peak CUDA memory: {result['runtime']['training_peak_cuda_memory_gib']:.3f} GiB",
        f"- A0 inference seconds/image: {result['runtime']['a0_seconds_per_image']:.6f}",
        f"- S²HR inference seconds/image: {result['runtime']['s2hr_seconds_per_image']:.6f}",
        "",
        "## 6. Provenance and safety",
        "",
        f"- A0 checkpoint SHA256: `{result['provenance']['a0_checkpoint_sha256']}`",
        f"- S²HR checkpoint SHA256: `{result['provenance']['s2hr_checkpoint_sha256']}`",
        f"- Training config SHA256: `{result['provenance']['training_config_sha256']}`",
        "- Validation contains exactly 3418 BCSS validation images.",
        "- Both checkpoints use this evaluation script and identical TTA, thresholds, class gate, min-max, fusion and released `iouutils.scores()`.",
        "- S²HR uses a first TTA pass only to obtain the same averaged official deep-presence mask required internally; final postprocessing is unchanged.",
        "",
        "## 7. Deferred by protocol",
        "",
        "No BPS-CH/SPSR ablation, seeds 11/17, LUAD, BCSS test, mechanism follow-up or hyperparameter change was run.",
        "",
        f"**{result['decision']}**",
        "",
        "STOP.",
    ]
    report = output / "docs" / "s2hr_v1_fullmodel_25ep_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--s2hr-checkpoint", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("Final S²HR-v1 execution is BCSS validation only")
    val_root = Path(args.val_root)
    images = list((val_root / "img").glob("*.png"))
    masks = list((val_root / "mask").glob("*.png"))
    if len(images) != 3418 or len(masks) != 3418:
        raise AssertionError(f"Expected 3418 BCSS val pairs, got {len(images)}/{len(masks)}")
    output = Path(args.experiment_dir)
    for directory in (
        "validation/sshr_a0", "validation/s2hr_epoch25", "figures", "docs"
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    a0_sha = sha256_file(args.a0_checkpoint)
    if a0_sha != EXPECTED_A0_SHA256:
        raise AssertionError(f"A0 checkpoint SHA mismatch: {a0_sha}")
    s2hr_sha = sha256_file(args.s2hr_checkpoint)
    training_manifest = json.loads(
        (output / "provenance" / "manifest.json").read_text(encoding="utf-8")
    )
    training_config = json.loads(
        (output / "provenance" / "training_config.json").read_text(encoding="utf-8")
    )
    history = json.loads((output / "train" / "history.json").read_text(encoding="utf-8"))
    completion = json.loads(
        (output / "train" / "training_complete.json").read_text(encoding="utf-8")
    )

    a0_model = SSHRNet(n_class=4)
    incompat = a0_model.load_state_dict(load_state(args.a0_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(incompat)
    a0_scores_raw, a0_runtime = infer_bcss(
        a0_model, str(val_root), amp_dtype="bf16", num_workers=args.num_workers
    )
    a0_scores = compact_scores(a0_scores_raw)
    write_json(output / "validation" / "sshr_a0" / "metrics.json", {
        "scores": a0_scores, "runtime": a0_runtime,
    })
    del a0_model
    torch.cuda.empty_cache()

    s2hr_model = S2HRNet(n_class=4)
    incompat = s2hr_model.load_state_dict(load_state(args.s2hr_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(incompat)
    s2hr_scores_raw, s2hr_runtime = infer_bcss(
        s2hr_model, str(val_root), amp_dtype="bf16", num_workers=args.num_workers
    )
    s2hr_scores = compact_scores(s2hr_scores_raw)
    write_json(output / "validation" / "s2hr_epoch25" / "metrics.json", {
        "scores": s2hr_scores, "runtime": s2hr_runtime,
    })

    delta_miou = 100 * (s2hr_scores["mIoU"] - a0_scores["mIoU"])
    delta_mdice = 100 * (s2hr_scores["mDice"] - a0_scores["mDice"])
    class_delta = {
        str(index): 100 * (
            s2hr_scores["class_iou"][str(index)] - a0_scores["class_iou"][str(index)]
        )
        for index in range(4)
    }
    if delta_miou >= 0.15:
        decision = "S2HR_FULLMODEL_CLEAR_SUCCESS"
    elif delta_miou >= 0.05:
        decision = "S2HR_FULLMODEL_POSITIVE"
    else:
        decision = "S2HR_FULLMODEL_NO_CLEAR_GAIN"
    class_safety = (
        "CLASS_REGRESSION_REVIEW"
        if any(value < -0.50 for value in class_delta.values())
        else "NO_CLASS_REGRESSION_OVER_0.50PP"
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    parameters = training_manifest["parameters"]
    result = {
        "decision": decision,
        "class_safety": class_safety,
        "models": {"sshr_a0": a0_scores, "s2hr_v1": s2hr_scores},
        "delta": {
            "mIoU_pp": delta_miou,
            "mDice_pp": delta_mdice,
            "class_iou_pp": class_delta,
        },
        "parameters": parameters,
        "runtime": {
            "mean_training_seconds_per_epoch": completion["mean_epoch_seconds"],
            "training_peak_cuda_memory_gib": completion["peak_cuda_memory_gib"],
            "a0_seconds_per_image": a0_runtime["seconds_per_image"],
            "s2hr_seconds_per_image": s2hr_runtime["seconds_per_image"],
        },
        "provenance": {
            "source_commit": source_commit,
            "training_source_commit": training_manifest["s2hr_source_commit"],
            "training_config_sha256": training_manifest["config_sha256"],
            "a0_checkpoint_sha256": a0_sha,
            "s2hr_checkpoint_sha256": s2hr_sha,
            "validation_images": len(images),
            "precision": "bf16",
        },
    }
    commands = {
        "train": training_manifest["command"],
        "evaluation": " ".join(sys.argv),
    }
    _plot_history(history, output / "figures")
    write_json(output / "validation" / "final_comparison.json", result)
    report = render_report(result, history, output, commands)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(report, flush=True)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
