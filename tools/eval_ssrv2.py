#!/usr/bin/env python3
"""Epoch-25 FINAL BCSS validation comparison for SSR-v2 and SSHR A0."""

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
from network.resnet38_cls_ssrv2 import Net_CAM as SSRv2Net
from tool.infer_ssrv2 import diagnose_gt_present_teacher, infer_bcss
from tools.train_ssrv2_25ep import sha256_file, write_json


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
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(epochs, [row["total_loss"] for row in history], label="total")
    ax.plot(epochs, [row["classification_loss"] for row in history], label="classification")
    ax.plot(epochs, [row["pcsd_loss_weighted"] for row in history], label="weighted PCSD")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("SSR-v2 training losses")
    ax.legend(); fig.tight_layout(); fig.savefig(output / "training_losses.png", dpi=180); plt.close(fig)

    fig, left = plt.subplots(figsize=(8.0, 4.6))
    left.plot(epochs, [row["gamma_spatial"] for row in history], label="gamma_spatial")
    left.plot(epochs, [row["effective_gamma"] for row in history], label="effective_gamma")
    left.set_xlabel("epoch"); left.set_ylabel("positive spatial scale")
    right = left.twinx()
    right.plot(epochs, [row["mean_abs_pd_minus_ps"] for row in history], color="tab:red", label="mean |Pd-Ps|")
    right.set_ylabel("mean |Pd-Ps|")
    lines = left.lines + right.lines
    left.legend(lines, [line.get_label() for line in lines], loc="best")
    left.set_title("SSR-v2 mechanism trajectory")
    fig.tight_layout(); fig.savefig(output / "mechanism_trajectory.png", dpi=180); plt.close(fig)


def mechanism_rows(manifest, history):
    selected = {row["epoch"]: row for row in history}
    init = manifest["initial_mechanism"]
    columns = ["init", "epoch05", "epoch10", "epoch15", "epoch20", "epoch25"]
    return columns, {
        "gamma_spatial": [init["gamma_spatial"]] + [selected[e]["gamma_spatial"] for e in (5, 10, 15, 20, 25)],
        "beta_spatial": [init["beta_spatial"]] + [selected[e]["beta_spatial"] for e in (5, 10, 15, 20, 25)],
        "gamma_global": [init["gamma_global"]] + [selected[e]["gamma_global_28_1"] for e in (5, 10, 15, 20, 25)],
        "gamma_context": [init["gamma_context"]] + [selected[e]["gamma_context_28_1"] for e in (5, 10, 15, 20, 25)],
        "mean_pcsd_kl": [None] + [selected[e]["pcsd_loss_raw"] for e in (5, 10, 15, 20, 25)],
        "mean_abs_pd_minus_ps": [None] + [selected[e]["mean_abs_pd_minus_ps"] for e in (5, 10, 15, 20, 25)],
    }


def render_report(result, history, output):
    a0 = result["models"]["sshr_a0"]; ssr = result["models"]["ssrv2"]
    columns, mechanisms = mechanism_rows(result["training_manifest"], history)
    lines = [
        "# SSR-v2 Full Model — BCSS Seed42 Epoch25 FINAL",
        "", "## 1. Executive result", "",
        f"- Decision: **{result['decision']}**",
        f"- Class safety: **{result['class_safety']}**",
        f"- ΔmIoU: **{result['delta']['mIoU_pp']:+.4f} pp**",
        f"- ΔmDice: **{result['delta']['mDice_pp']:+.4f} pp**",
        "- Epoch25 FINAL is the only primary checkpoint; no validation selection occurred.",
        "", "## 2. Frozen experimental control", "",
        "- Fresh ImageNet-pretrained ResNet38; no trained SSHR/S²HR checkpoint was loaded.",
        "- BCSS seed42, 25 epochs, batch20, 224×224, BF16, base LR 0.01, released PolyOptimizer and augmentation.",
        "- Original HFRM56/HFRM28_2, HFRM28_1 GSR/CH15, CAM heads, loss weights and official inference remain frozen.",
        "- SSR-v2 adds only beta_spatial (+1 scalar), PCSD (λmax=0.05) and positive PTCR with the fixed epoch1–5 ramp.",
        "", "## 3. Epoch25 validation", "",
        "| Model | Epoch | mIoU | mDice | C0 | C1 | C2 | C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in (("SSHR A0", a0), ("SSR-v2 Full", ssr)):
        lines.append(
            f"| {name} | 25 | {100*value['mIoU']:.4f} | {100*value['mDice']:.4f} | "
            + " | ".join(f"{100*value['class_iou'][str(i)]:.4f}" for i in range(4)) + " |"
        )
    lines += [
        "", "| Quantity | Delta (pp) |", "|---|---:|",
        f"| mIoU | {result['delta']['mIoU_pp']:+.4f} |",
        f"| mDice | {result['delta']['mDice_pp']:+.4f} |",
    ]
    for index in range(4):
        lines.append(f"| C{index} IoU | {result['delta']['class_iou_pp'][str(index)]:+.4f} |")
    lines += [
        "", "## 4. Mechanism trajectory", "",
        "| Quantity | Init | Epoch5 | Epoch10 | Epoch15 | Epoch20 | Epoch25 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in mechanisms.items():
        rendered = ["—" if value is None else f"{value:.8f}" for value in values]
        lines.append(f"| {name} | " + " | ".join(rendered) + " |")
    teacher = result["teacher_diagnosis"]
    lines += [
        "", "## 5. Validation-only teacher diagnosis", "",
        f"- GT-present deep spatial accuracy: {100*teacher['deep_accuracy']:.4f}%",
        f"- GT-present raw CAM28_1 accuracy: {100*teacher['raw28_1_accuracy']:.4f}%",
        f"- Deep advantage: {teacher['teacher_advantage_pp']:+.4f} pp",
        "- GT masks were applied only after network forward and never entered training or inference decisions.",
        "", "## 6. Runtime and resources", "",
        f"- SSHR / SSR-v2 parameters: {result['parameters']['sshr_total']:,} / {result['parameters']['ssrv2_total']:,}",
        f"- Added parameters: {result['parameters']['new_parameters']}",
        f"- Mean seconds/epoch: {result['runtime']['mean_training_seconds_per_epoch']:.2f}",
        f"- Peak training CUDA memory: {result['runtime']['training_peak_cuda_memory_gib']:.3f} GiB",
        f"- A0 / SSR-v2 inference seconds per image: {result['runtime']['a0_seconds_per_image']:.6f} / {result['runtime']['ssrv2_seconds_per_image']:.6f}",
        "", "## 7. Provenance", "",
        f"- Training source commit: `{result['provenance']['training_source_commit']}`",
        f"- Evaluation source commit: `{result['provenance']['evaluation_source_commit']}`",
        f"- A0 checkpoint SHA256: `{result['provenance']['a0_checkpoint_sha256']}`",
        f"- SSR-v2 checkpoint SHA256: `{result['provenance']['ssrv2_checkpoint_sha256']}`",
        f"- Training config SHA256: `{result['provenance']['training_config_sha256']}`",
        "- Validation pairs: 3,418; precision BF16; official 3-way TTA, thresholds, class gate, min-max, 0/0.6/0.2/0.2 fusion and released metric.",
        "", "## 8. Figures", "",
        "![Training losses](../figures/training_losses.png)",
        "![Mechanism trajectory](../figures/mechanism_trajectory.png)",
        "", "## 9. Stop boundary", "",
        "No test, LUAD, seeds 11/17, ablation, lambda/gamma/ramp sweep or SSR-v3 was run.",
        "", f"**{result['decision']}**", "", "STOP.",
    ]
    report = output / "docs" / "ssrv2_full_25ep_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--ssrv2-checkpoint", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("SSR-v2 formal evaluation is BCSS validation only")
    val_root = Path(args.val_root)
    if len(list((val_root / "img").glob("*.png"))) != 3418 or len(list((val_root / "mask").glob("*.png"))) != 3418:
        raise AssertionError("Expected exactly 3418 BCSS validation pairs")
    output = Path(args.experiment_dir)
    if (output / "validation" / "final_comparison.json").exists():
        raise FileExistsError("Formal SSR-v2 evaluation already exists")
    for directory in ("validation/sshr_a0", "validation/ssrv2_epoch25", "figures", "docs"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    a0_sha = sha256_file(args.a0_checkpoint)
    if a0_sha != EXPECTED_A0_SHA256:
        raise AssertionError(f"A0 SHA mismatch: {a0_sha}")
    ssrv2_sha = sha256_file(args.ssrv2_checkpoint)
    manifest = json.loads((output / "provenance" / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((output / "provenance" / "training_config.json").read_text(encoding="utf-8"))
    history = json.loads((output / "train" / "history.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "train" / "training_complete.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((output / "checkpoints" / "manifest.json").read_text(encoding="utf-8"))
    final_meta = next(item for item in checkpoints if item["epoch"] == 25)
    if final_meta["sha256"] != ssrv2_sha or not final_meta["primary_final"]:
        raise AssertionError("Epoch25 primary checkpoint provenance mismatch")

    a0_model = A0Net(4); a0_model.load_state_dict(load_state(args.a0_checkpoint), strict=True)
    a0_raw, a0_runtime = infer_bcss(a0_model, str(val_root), "bf16", args.num_workers)
    a0 = compact(a0_raw)
    write_json(output / "validation" / "sshr_a0" / "metrics.json", {"scores": a0, "runtime": a0_runtime})
    del a0_model; torch.cuda.empty_cache()

    ssrv2_model = SSRv2Net(4); ssrv2_model.load_state_dict(load_state(args.ssrv2_checkpoint), strict=True)
    ssrv2_raw, ssrv2_runtime = infer_bcss(ssrv2_model, str(val_root), "bf16", args.num_workers)
    ssrv2 = compact(ssrv2_raw)
    teacher = diagnose_gt_present_teacher(ssrv2_model, str(val_root), "bf16", args.num_workers)
    write_json(output / "validation" / "ssrv2_epoch25" / "metrics.json", {
        "scores": ssrv2, "runtime": ssrv2_runtime, "teacher_diagnosis": teacher,
    })

    delta_miou = 100 * (ssrv2["mIoU"] - a0["mIoU"])
    delta_mdice = 100 * (ssrv2["mDice"] - a0["mDice"])
    class_delta = {
        str(i): 100 * (ssrv2["class_iou"][str(i)] - a0["class_iou"][str(i)])
        for i in range(4)
    }
    if delta_miou >= 0.30:
        decision = "SSRV2_FULLMODEL_STRONG_SUCCESS"
    elif delta_miou >= 0.15:
        decision = "SSRV2_FULLMODEL_CLEAR_SUCCESS"
    elif delta_miou >= 0.05:
        decision = "SSRV2_FULLMODEL_POSITIVE"
    else:
        decision = "SSRV2_FULLMODEL_NO_CLEAR_GAIN"
    class_safety = (
        "SSRV2_CLASS_REGRESSION_REVIEW"
        if any(value <= -0.50 for value in class_delta.values())
        else "NO_CLASS_REGRESSION_AT_OR_BELOW_MINUS_0.50PP"
    )
    evaluation_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    result = {
        "decision": decision, "class_safety": class_safety,
        "models": {"sshr_a0": a0, "ssrv2": ssrv2},
        "delta": {"mIoU_pp": delta_miou, "mDice_pp": delta_mdice, "class_iou_pp": class_delta},
        "teacher_diagnosis": teacher,
        "parameters": manifest["parameters"],
        "runtime": {
            "mean_training_seconds_per_epoch": completion["mean_epoch_seconds"],
            "training_peak_cuda_memory_gib": completion["peak_cuda_memory_gib"],
            "a0_seconds_per_image": a0_runtime["seconds_per_image"],
            "ssrv2_seconds_per_image": ssrv2_runtime["seconds_per_image"],
        },
        "training_manifest": manifest,
        "provenance": {
            "training_source_commit": manifest["ssrv2_source_commit"],
            "evaluation_source_commit": evaluation_commit,
            "training_config_sha256": manifest["config_sha256"],
            "a0_checkpoint_sha256": a0_sha, "ssrv2_checkpoint_sha256": ssrv2_sha,
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
