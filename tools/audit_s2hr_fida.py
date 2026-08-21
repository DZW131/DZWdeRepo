#!/usr/bin/env python3
"""S²HR-v1 FIDA-v0 frozen innovation decomposition on BCSS validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net_CAM as A0Net
from network.resnet38_cls_s2hr import Net_CAM as S2HRNet
from tool.GenDataset import Stage1_InferDataset
from tool.infer_s2hr import infer_bcss
from tool.infer_s2hr_fida import FIDAInstrumentor, PRIMARY_VARIANTS, masked_argmax
from tools.s2hr_fida_metrics import (
    BoundaryQualityAccumulator,
    ErrorTaxonomyAccumulator,
    OfficialMetricAccumulator,
    ResidualUtilityAccumulator,
    SpatialTransitionAccumulator,
    TeacherReliabilityAccumulator,
    foreground_boundary_bins,
    image_presence,
)


EXPECTED_A0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
EXPECTED_S2HR_SHA256 = "129ad097ad73f9f564d8778baa8e914f92c8200ed90dc1f8763677dffe91b9ac"
REFERENCE_FULL_MIOU = 0.6704998279876101
REFERENCE_A0_MIOU = 0.6732834150110829
ATTRIBUTION_THRESHOLD_PP = 0.05


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def compact_a0_scores(scores):
    return {
        "mIoU": float(scores["Mean IoU"]),
        "mDice": float(scores["Mean Dice"]),
        "class_iou": {str(i): float(scores["Class IoU"][i]) for i in range(4)},
        "class_dice": {str(i): float(scores["Dice Coefficients"][i]) for i in range(4)},
        "pixel_accuracy": float(scores["Pixel Accuracy"]),
        "mean_accuracy": float(scores["Mean Accuracy"]),
        "frequency_weighted_iou": float(scores["Frequency Weighted IoU"]),
        "images": 3418,
    }


def attribution_label(prefix, effect_pp):
    if effect_pp >= ATTRIBUTION_THRESHOLD_PP:
        return f"{prefix}_POSITIVE"
    if effect_pp <= -ATTRIBUTION_THRESHOLD_PP:
        return f"{prefix}_HARMFUL"
    return f"{prefix}_NEUTRAL"


def flatten_spatial(summary):
    return [
        {"comparison": name, "region": region, **values}
        for name, regions in summary.items()
        for region, values in regions.items()
    ]


def flatten_taxonomy(summary):
    return [
        {"variant": variant, "category": category, **values}
        for variant, categories in summary.items()
        for category, values in categories.items()
    ]


def plot_results(output, metrics, effects, sign_metrics, teacher_rows, boundary, spatial):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    labels = ["A0", "V00", "V10", "V01", "V11"]
    values = [100 * metrics[name]["mIoU"] for name in labels]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(labels, values, color=["#777777", "#4c78a8", "#59a14f", "#f28e2b", "#e15759"])
    ax.set_ylabel("validation mIoU (%)")
    ax.set_title("Frozen 2×2 innovation decomposition")
    for index, value in enumerate(values):
        ax.text(index, value + 0.015, f"{value:.4f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(figures / "factorial_miou.png", dpi=180); plt.close(fig)

    sign_names = ["negative", "zero", "positive"]
    sign_values = [100 * sign_metrics[name]["mIoU"] for name in sign_names]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(sign_names, sign_values, color=["#4c78a8", "#999999", "#f28e2b"])
    ax.set_ylabel("validation mIoU (%)"); ax.set_title("SPSR sign-direction probe")
    for index, value in enumerate(sign_values):
        ax.text(index, value + 0.015, f"{value:.4f}", ha="center")
    fig.tight_layout(); fig.savefig(figures / "spsr_sign_probe.png", dpi=180); plt.close(fig)

    overall = [row for row in teacher_rows if row["region"] == "overall" and row["class"] == "overall"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(overall)); width = 0.34
    ax.bar(x - width / 2, [row["deep_help"] for row in overall], width, label="deep help")
    ax.bar(x + width / 2, [row["deep_harm"] for row in overall], width, label="deep harm")
    ax.set_xticks(x, [row["presence"] for row in overall]); ax.set_ylabel("foreground pixels")
    ax.set_title("Deep teacher help and harm opportunities"); ax.legend()
    fig.tight_layout(); fig.savefig(figures / "teacher_help_harm.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    names = ["precision", "recall", "f1", "iou"]
    ax.bar(names, [boundary[name] for name in names], color="#4c78a8")
    ax.set_ylim(0, 1); ax.set_ylabel("score"); ax.set_title("BPS boundary quality")
    fig.tight_layout(); fig.savefig(figures / "boundary_precision_recall.png", dpi=180); plt.close(fig)

    bps = spatial["BPS_V10_minus_V00"]
    regions = ["B0_le_2", "B1_3_7", "B2_ge_8"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(regions, [100 * bps[region]["accuracy_delta"] for region in regions], color="#59a14f")
    ax.axhline(0, color="black", linewidth=0.7); ax.set_ylabel("accuracy delta (pp)")
    ax.set_title("BPS direct spatial causal effect")
    fig.tight_layout(); fig.savefig(figures / "bps_boundary_interior_effect.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    x = np.arange(4); width = 0.18
    for index, name in enumerate(("V00", "V10", "V01", "V11")):
        ax.bar(
            x + (index - 1.5) * width,
            [100 * metrics[name]["class_iou"][str(c)] for c in range(4)],
            width,
            label=name,
        )
    ax.set_xticks(x, [f"C{c}" for c in range(4)]); ax.set_ylabel("IoU (%)")
    ax.set_title("Per-class frozen factorial"); ax.legend(ncol=4)
    fig.tight_layout(); fig.savefig(figures / "per_class_factorial.png", dpi=180); plt.close(fig)


def render_report(output, result):
    metrics = result["metrics"]
    effects = result["effects_pp"]
    labels = result["attribution_labels"]
    teacher = result["teacher_key_rows"]
    boundary = result["boundary_quality"]
    bps_spatial = result["spatial_causal"]["BPS_V10_minus_V00"]
    sign = result["sign_probe"]
    class_decomp = result["class_decomposition_pp"]
    taxonomy = result["error_taxonomy"]
    residual = result["residual_utility"]
    bps_boundary_net = (
        bps_spatial["B0_le_2"]["net"] + bps_spatial["B1_3_7"]["net"]
    )
    lines = [
        "# S²HR-v1 FIDA-v0 — Frozen Innovation Decomposition Audit",
        "",
        "## 1. Executive attribution",
        "",
        f"- Final route: **{result['route']}**",
        f"- Labels: {', '.join(labels)}",
        f"- Present-confusion teacher finding: **{result['teacher_present_confusion_finding']}**",
        "- Frozen-checkpoint deployment effects do not establish fresh-training causality.",
        "- No optimizer, update, retraining, test, LUAD, sweep or checkpoint mutation occurred.",
        "",
        "## 2. Provenance and instrumentation parity",
        "",
        f"- Source commit: `{result['provenance']['source_commit']}`",
        f"- A0 checkpoint SHA256: `{result['provenance']['a0_sha256']}`",
        f"- S²HR checkpoint SHA256 before/after: `{result['provenance']['s2hr_sha256_before']}` / `{result['provenance']['s2hr_sha256_after']}`",
        f"- In-memory state_dict SHA256 before/after: `{result['provenance']['state_dict_sha256_before']}` / `{result['provenance']['state_dict_sha256_after']}`",
        f"- Frozen learned gamma_spatial / rho_boundary: `{result['provenance']['learned_gamma_spatial']:+.8f}` / `{result['provenance']['learned_rho_boundary']:.8f}`",
        f"- Prior-run reference drift (A0 / V11): `{result['provenance']['reference_delta_pp']['a0']:+.6f}` / `{result['provenance']['reference_delta_pp']['s2hr_full']:+.6f}` pp (recorded only; not a selection or stopping rule)",
        f"- 32-image parity max CAM difference: `{result['parity']['max_cam_abs_difference']}`",
        f"- Parity differing final pixels: `{result['parity']['differing_final_pixels']}`",
        f"- Parity mIoU delta: `{result['parity']['miou_delta']}`",
        "",
        "## 3. Frozen 2×2 factorial",
        "",
        "| Variant | mIoU | Δ vs V00 | mDice | C0 | C1 | C2 | C3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A0", "V00", "V10", "V01", "V11"):
        value = metrics[name]
        delta = 100 * (value["mIoU"] - metrics["V00"]["mIoU"])
        lines.append(
            f"| {name} | {100*value['mIoU']:.4f} | {delta:+.4f} | {100*value['mDice']:.4f} | "
            + " | ".join(f"{100*value['class_iou'][str(c)]:.4f}" for c in range(4))
            + " |"
        )
    lines += [
        "",
        "## 4. Factor effects",
        "",
        "| Effect | ΔmIoU (pp) |",
        "|---|---:|",
    ]
    for name, value in effects.items():
        lines.append(f"| {name} | {value:+.6f} |")
    lines += [
        "",
        "## 5. SPSR sign-direction audit",
        "",
        "| State | gamma | mIoU | Δ vs zero | Boundary Δacc | Interior Δacc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("negative", "Learned negative"), ("zero", "Zero"), ("positive", "Positive flip")):
        value = sign[name]
        lines.append(
            f"| {label} | {value['gamma']:+.8f} | {100*value['mIoU']:.4f} | {value['delta_vs_zero_pp']:+.4f} | {value['boundary_delta_pp']:+.4f} | {value['interior_delta_pp']:+.4f} |"
        )
    lines += [
        "",
        "## 6. Deep spatial teacher reliability",
        "",
        "| Presence | Region | Deep Acc | Raw28_1 Acc | Deep-help | Deep-harm | Net rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in teacher:
        lines.append(
            f"| {row['presence']} | {row['region']} | {100*row['deep_accuracy']:.4f} | {100*row['raw28_1_accuracy']:.4f} | {row['deep_help']:,} | {row['deep_harm']:,} | {100*row['teacher_net_rate']:+.4f}% |"
        )
    lines += [
        "",
        "## 7. BPS boundary quality and direct effect",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Boundary precision | {boundary['precision']:.6f} |",
        f"| Boundary recall | {boundary['recall']:.6f} |",
        f"| Boundary F1 | {boundary['f1']:.6f} |",
        f"| Boundary IoU | {boundary['iou']:.6f} |",
        f"| Predicted boundary fraction | {boundary['predicted_boundary_fraction']:.6f} |",
        f"| GT boundary fraction | {boundary['gt_boundary_fraction']:.6f} |",
        f"| B2 interior contamination | {boundary['b2_interior_contamination']:.6f} |",
        f"| Outside-foreground fraction | {boundary['outside_foreground_fraction']:.6f} |",
        f"| BPS overall ΔmIoU | {effects['BPS given SPSR off']:+.6f} pp |",
        f"| BPS boundary (B0+B1) net recovery | {bps_boundary_net:,} |",
        f"| BPS B0 net recovery | {bps_spatial['B0_le_2']['net']:,} |",
        f"| BPS B2 net recovery | {bps_spatial['B2_ge_8']['net']:,} |",
        "",
        "The boundary-quality estimate pools the three actual unflipped 28×28 TTA controller maps. Teacher logits are TTA-averaged before the GT/deployed-presence diagnostic.",
        "",
        "## 8. Counterfactual error taxonomy",
        "",
        "| Variant | Absent error | Present confusion | Boundary error | Interior error |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("V00", "V10", "V01", "V11"):
        lines.append(
            f"| {name} | {taxonomy[name]['absent_class']['errors']:,} | "
            f"{taxonomy[name]['present_confusion']['errors']:,} | "
            f"{taxonomy[name]['boundary']['errors']:,} | "
            f"{taxonomy[name]['interior']['errors']:,} |"
        )
    lines += [
        "",
        f"Positive-sign SPSR residual utility: recovered={residual['recovered']:,}, harmed={residual['harmed']:,}, net={residual['net']:,}; recovered-in-deep-help fraction={residual['recovered_fraction_in_deep_help']:.6f}.",
        "",
        "## 9. C1/C3 decomposition",
        "",
        "| Class | Trajectory | BPS main | SPSR main | Interaction | Total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for class_index in (1, 3):
        row = class_decomp[str(class_index)]
        lines.append(
            f"| C{class_index} | {row['trajectory']:+.4f} | {row['bps_main']:+.4f} | {row['spsr_main']:+.4f} | {row['interaction']:+.4f} | {row['total']:+.4f} |"
        )
    answers = result["answers"]
    lines += ["", "## 10. Required answers", ""]
    for index, answer in enumerate(answers, start=1):
        lines.append(f"{index}. {answer}")
    lines += [
        "",
        "## 11. Figures",
        "",
        "![Factorial mIoU](../figures/factorial_miou.png)",
        "![SPSR sign probe](../figures/spsr_sign_probe.png)",
        "![Teacher help/harm](../figures/teacher_help_harm.png)",
        "![Boundary precision/recall](../figures/boundary_precision_recall.png)",
        "![BPS boundary/interior effect](../figures/bps_boundary_interior_effect.png)",
        "![Per-class factorial](../figures/per_class_factorial.png)",
        "",
        "## 12. Stop boundary",
        "",
        "No automatic retraining, redesign, other seed, LUAD, test or tuning was run.",
        "",
        f"**{result['route']}**",
        "",
        "STOP.",
    ]
    report = output / "docs" / "s2hr_v1_frozen_innovation_decomposition.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--s2hr-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--parity-images", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("FIDA-v0 is BCSS validation only")
    if args.parity_images != 32:
        raise AssertionError("Instrumentation parity is frozen at 32 images")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty FIDA output directory: {output}")
    for directory in (
        "provenance", "parity", "factorial", "spsr_direction",
        "teacher_reliability", "bps_boundary", "error_taxonomy", "figures", "docs",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)

    a0_sha = sha256_file(args.a0_checkpoint)
    s2hr_sha_before = sha256_file(args.s2hr_checkpoint)
    if a0_sha != EXPECTED_A0_SHA256 or s2hr_sha_before != EXPECTED_S2HR_SHA256:
        raise AssertionError(f"Checkpoint SHA mismatch: A0={a0_sha}, S2HR={s2hr_sha_before}")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    source_files = (
        "tool/infer_s2hr_fida.py", "tools/s2hr_fida_metrics.py",
        "tools/audit_s2hr_fida.py", "tests/test_s2hr_fida.py",
        "network/resnet38_cls_s2hr.py", "network/s2hfrm28_1.py",
        "tool/infer_s2hr.py", "tool/iouutils.py",
    )
    write_json(output / "provenance" / "source_hashes.json", {
        name: sha256_file(REPO_ROOT / name) for name in source_files
    })
    write_json(output / "provenance" / "checkpoint_hashes.json", {
        "a0": a0_sha, "s2hr_before": s2hr_sha_before,
    })

    val_root = Path(args.val_root)
    image_files = list((val_root / "img").glob("*.png"))
    mask_files = list((val_root / "mask").glob("*.png"))
    if len(image_files) != 3418 or len(mask_files) != 3418:
        raise AssertionError(f"Expected 3418 validation pairs, got {len(image_files)}/{len(mask_files)}")
    dataset = Stage1_InferDataset(str(val_root / "img"), img_size=224)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = S2HRNet(4)
    incompat = model.load_state_dict(load_state(args.s2hr_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(incompat)
    instrumentor = FIDAInstrumentor(model, amp_dtype="bf16")
    state_dict_sha_before = sha256_state_dict(model)
    learned_gamma = float(model.hfrm_28_1.gamma_spatial.detach().cpu())
    learned_rho = float(torch.sigmoid(model.hfrm_28_1.rho_boundary_raw.detach()).cpu())
    if abs(learned_gamma - (-1.0526387691497803)) > 1.0e-7:
        raise AssertionError(f"Unexpected learned gamma_spatial: {learned_gamma}")
    if abs(learned_rho - 0.12448688) > 1.0e-7:
        raise AssertionError(f"Unexpected learned rho_boundary: {learned_rho}")

    parity_audit = OfficialMetricAccumulator()
    parity_release = OfficialMetricAccumulator()
    max_cam_difference = 0.0
    differing_pixels = 0
    parity_names = []
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader):
            if image_index >= args.parity_images:
                break
            image_id = name_tuple[0]
            truth = np.asarray(Image.open(val_root / "mask" / f"{image_id}.png"))
            audit, _ = instrumentor.audit_image(image, truth.shape)
            released = instrumentor.released_image(image, truth.shape)
            for key in ("cam_56", "cam_28_1", "cam_28_2", "cam_deep"):
                difference = np.max(np.abs(audit["V11"]["cams"][key] - released["cams"][key]))
                max_cam_difference = max(max_cam_difference, float(difference))
            differing_pixels += int(np.count_nonzero(
                audit["V11"]["prediction"] != released["prediction"]
            ))
            parity_audit.update(truth, audit["V11"]["prediction"])
            parity_release.update(truth, released["prediction"])
            parity_names.append(image_id)
    parity_audit_score = parity_audit.scores()
    parity_release_score = parity_release.scores()
    parity = {
        "decision": "S2HR_FIDA_INSTRUMENTATION_PASS",
        "images": args.parity_images,
        "image_ids": parity_names,
        "max_cam_abs_difference": max_cam_difference,
        "differing_final_pixels": differing_pixels,
        "audit_mIoU": parity_audit_score["mIoU"],
        "released_mIoU": parity_release_score["mIoU"],
        "miou_delta": parity_audit_score["mIoU"] - parity_release_score["mIoU"],
    }
    if max_cam_difference != 0.0 or differing_pixels != 0 or parity["miou_delta"] != 0.0:
        parity["decision"] = "S2HR_FIDA_INSTRUMENTATION_NOGO"
        write_json(output / "parity" / "instrumentation.json", parity)
        raise RuntimeError("S2HR_FIDA_INSTRUMENTATION_NOGO")
    write_json(output / "parity" / "instrumentation.json", parity)

    metric_acc = {name: OfficialMetricAccumulator() for name in (*PRIMARY_VARIANTS, "Splus")}
    teacher_acc = TeacherReliabilityAccumulator()
    boundary_acc = BoundaryQualityAccumulator()
    spatial_acc = SpatialTransitionAccumulator((
        "BPS_V10_minus_V00", "SPSR_negative_minus_zero", "SPSR_positive_minus_zero",
    ))
    taxonomy_acc = ErrorTaxonomyAccumulator(tuple(PRIMARY_VARIANTS))
    residual_acc = ResidualUtilityAccumulator()
    started = time.time()
    base_forward_start = instrumentor.backbone_forwards
    with torch.no_grad():
        for image_index, (name_tuple, image) in enumerate(loader, start=1):
            image_id = name_tuple[0]
            truth = np.asarray(Image.open(val_root / "mask" / f"{image_id}.png"))
            variants, diagnostics = instrumentor.audit_image(image, truth.shape)
            predictions = {name: variants[name]["prediction"] for name in PRIMARY_VARIANTS}
            for name in metric_acc:
                metric_acc[name].update(truth, variants[name]["prediction"])

            gt_presence = image_presence(truth)
            deployed_presence = diagnostics["deployed_presence"].astype(bool)
            bins = foreground_boundary_bins(truth)
            teacher_predictions = {}
            for condition, presence in (("oracle", gt_presence), ("deployed", deployed_presence)):
                deep_prediction = masked_argmax(diagnostics["deep_logits"], presence)
                shallow_prediction = masked_argmax(diagnostics["raw_28_1_logits"], presence)
                teacher_predictions[condition] = (deep_prediction, shallow_prediction)
                teacher_acc.update(
                    condition, truth, deep_prediction, shallow_prediction, bins, gt_presence
                )

            for boundary_28 in diagnostics["tta_boundaries_28"]:
                boundary_acc.update(boundary_28, truth, bins)
            spatial_acc.update(
                "BPS_V10_minus_V00", truth, predictions["V00"], predictions["V10"], bins
            )
            spatial_acc.update(
                "SPSR_negative_minus_zero", truth, predictions["V00"], predictions["V01"], bins
            )
            spatial_acc.update(
                "SPSR_positive_minus_zero", truth, predictions["V00"], variants["Splus"]["prediction"], bins
            )
            taxonomy_acc.update(truth, predictions, gt_presence, bins)
            residual_acc.update(
                truth,
                predictions["V00"],
                variants["Splus"]["prediction"],
                teacher_predictions["deployed"][0],
                teacher_predictions["deployed"][1],
                bins,
            )
            if image_index % 200 == 0:
                print(f"FIDA_PROGRESS {image_index}/{len(dataset)}", flush=True)

    audit_elapsed = time.time() - started
    full_base_forwards = instrumentor.backbone_forwards - base_forward_start
    if full_base_forwards != 3 * len(dataset):
        raise AssertionError(f"Variants did not reuse exactly 3 bases/image: {full_base_forwards}")
    variant_metrics = {name: accumulator.scores() for name, accumulator in metric_acc.items()}
    # The preregistered hard gate is exact 32-image released-inference parity.
    # Prior full-run values are provenance references, not an additional
    # floating-point equality gate across process/restart states.
    full_reference_delta_pp = 100 * (
        variant_metrics["V11"]["mIoU"] - REFERENCE_FULL_MIOU
    )

    state_dict_sha_after = sha256_state_dict(model)
    if state_dict_sha_after != state_dict_sha_before:
        raise AssertionError("In-memory S2HR state_dict changed during frozen audit")

    del instrumentor, model
    torch.cuda.empty_cache()
    a0_model = A0Net(4)
    incompat = a0_model.load_state_dict(load_state(args.a0_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(incompat)
    a0_raw, a0_runtime = infer_bcss(
        a0_model, str(val_root), amp_dtype="bf16", num_workers=args.num_workers
    )
    variant_metrics["A0"] = compact_a0_scores(a0_raw)
    a0_reference_delta_pp = 100 * (
        variant_metrics["A0"]["mIoU"] - REFERENCE_A0_MIOU
    )

    m = {name: 100 * value["mIoU"] for name, value in variant_metrics.items()}
    effects = {
        "Training trajectory: V00 - A0": m["V00"] - m["A0"],
        "BPS given SPSR off": m["V10"] - m["V00"],
        "BPS given SPSR on": m["V11"] - m["V01"],
        "SPSR given BPS off": m["V01"] - m["V00"],
        "SPSR given BPS on": m["V11"] - m["V10"],
        "Interaction": m["V11"] - m["V10"] - m["V01"] + m["V00"],
        "Full - V00 direct innovation effect": m["V11"] - m["V00"],
        "Full - A0 total effect": m["V11"] - m["A0"],
    }
    effects["BPS main effect"] = 0.5 * (
        effects["BPS given SPSR off"] + effects["BPS given SPSR on"]
    )
    effects["SPSR main effect"] = 0.5 * (
        effects["SPSR given BPS off"] + effects["SPSR given BPS on"]
    )
    effects["Decomposition identity residual"] = (
        effects["Full - A0 total effect"]
        - effects["Training trajectory: V00 - A0"]
        - effects["Full - V00 direct innovation effect"]
    )
    if abs(effects["Decomposition identity residual"]) > 1.0e-10:
        raise AssertionError(effects)

    spatial = spatial_acc.summary()
    sign_metrics = {
        "negative": {**variant_metrics["V01"], "gamma": learned_gamma},
        "zero": {**variant_metrics["V00"], "gamma": 0.0},
        "positive": {**variant_metrics["Splus"], "gamma": abs(learned_gamma)},
    }
    for name, comparison in (
        ("negative", "SPSR_negative_minus_zero"),
        ("positive", "SPSR_positive_minus_zero"),
    ):
        sign_metrics[name]["delta_vs_zero_pp"] = 100 * (
            sign_metrics[name]["mIoU"] - sign_metrics["zero"]["mIoU"]
        )
        sign_metrics[name]["boundary_delta_pp"] = 100 * (
            (
                spatial[comparison]["B0_le_2"]["candidate_correct"]
                + spatial[comparison]["B1_3_7"]["candidate_correct"]
            )
            / max(
                spatial[comparison]["B0_le_2"]["pixels"]
                + spatial[comparison]["B1_3_7"]["pixels"], 1
            )
            - (
                spatial[comparison]["B0_le_2"]["baseline_correct"]
                + spatial[comparison]["B1_3_7"]["baseline_correct"]
            )
            / max(
                spatial[comparison]["B0_le_2"]["pixels"]
                + spatial[comparison]["B1_3_7"]["pixels"], 1
            )
        )
        sign_metrics[name]["interior_delta_pp"] = 100 * spatial[comparison]["B2_ge_8"]["accuracy_delta"]
    sign_metrics["zero"].update({
        "delta_vs_zero_pp": 0.0, "boundary_delta_pp": 0.0, "interior_delta_pp": 0.0,
    })

    teacher_rows, present_confusion = teacher_acc.summary()
    teacher_key_rows = [
        row for row in teacher_rows
        if row["class"] == "overall" and row["region"] in ("overall", "boundary", "interior")
    ]
    boundary_quality = boundary_acc.summary()
    taxonomy = taxonomy_acc.summary()
    residual_utility = residual_acc.summary()

    bps_label = attribution_label("BPS_DIRECT", effects["BPS main effect"])
    spsr_label = attribution_label("SPSR_DIRECT", effects["SPSR main effect"])
    trajectory_label = attribution_label(
        "TRAINING_TRAJECTORY", effects["Training trajectory: V00 - A0"]
    )
    interaction_label = attribution_label("INNOVATION_INTERACTION", effects["Interaction"])
    labels = [bps_label, spsr_label, trajectory_label, interaction_label]
    if (
        (
            sign_metrics["negative"]["mIoU"] > sign_metrics["zero"]["mIoU"]
            > sign_metrics["positive"]["mIoU"]
        )
        or 100 * (sign_metrics["negative"]["mIoU"] - sign_metrics["positive"]["mIoU"])
        >= ATTRIBUTION_THRESHOLD_PP
    ):
        labels.append("SPSR_DIRECTION_REJECTED")
    oracle_overall = next(
        row for row in teacher_rows
        if row["presence"] == "oracle" and row["region"] == "overall" and row["class"] == "overall"
    )
    oracle_boundary = next(
        row for row in teacher_rows
        if row["presence"] == "oracle" and row["region"] == "boundary" and row["class"] == "overall"
    )
    if oracle_overall["teacher_net_rate"] <= 0:
        labels.append("DEEP_SPATIAL_TEACHER_UNRELIABLE")
    if oracle_boundary["deep_accuracy"] < oracle_boundary["raw28_1_accuracy"]:
        labels.append("DEEP_LOCAL_BOUNDARY_UNRELIABLE")

    if bps_label == "BPS_DIRECT_POSITIVE" and (
        spsr_label != "SPSR_DIRECT_POSITIVE" or "SPSR_DIRECTION_REJECTED" in labels
    ):
        route = "ROUTE A — KEEP BPS, REPLACE SPSR"
    elif (
        bps_label == "BPS_DIRECT_POSITIVE"
        and spsr_label == "SPSR_DIRECT_POSITIVE"
        and interaction_label == "INNOVATION_INTERACTION_HARMFUL"
    ):
        route = "ROUTE B — KEEP BOTH, REDESIGN COUPLING"
    elif (
        trajectory_label == "TRAINING_TRAJECTORY_HARMFUL"
        and bps_label != "BPS_DIRECT_HARMFUL"
        and spsr_label != "SPSR_DIRECT_HARMFUL"
    ):
        route = "ROUTE C — TRAINING-TRAJECTORY PROBLEM"
    else:
        route = "ROUTE D — CLOSE S²HR-v1 DUAL-INNOVATION DESIGN"

    class_decomposition = {}
    for class_index in range(4):
        key = str(class_index)
        c = {name: 100 * variant_metrics[name]["class_iou"][key] for name in ("A0", "V00", "V10", "V01", "V11")}
        class_decomposition[key] = {
            "trajectory": c["V00"] - c["A0"],
            "bps_main": 0.5 * ((c["V10"] - c["V00"]) + (c["V11"] - c["V01"])),
            "spsr_main": 0.5 * ((c["V01"] - c["V00"]) + (c["V11"] - c["V10"])),
            "interaction": c["V11"] - c["V10"] - c["V01"] + c["V00"],
            "total": c["V11"] - c["A0"],
        }

    negative_better = sign_metrics["negative"]["mIoU"] > sign_metrics["zero"]["mIoU"]
    positive_worse = sign_metrics["positive"]["mIoU"] < sign_metrics["negative"]["mIoU"]
    present_lookup = {
        (row["presence"], row["method"]): row for row in present_confusion
    }
    oracle_deep_confusion = present_lookup[("oracle", "deep")]
    oracle_raw_confusion = present_lookup[("oracle", "raw28_1")]
    teacher_present_confusion_finding = (
        "DEEP_PRESENT_CLASS_TEACHER_SUPPORTED"
        if oracle_deep_confusion["present_confusion_error_rate"]
        < oracle_raw_confusion["present_confusion_error_rate"]
        else "DEEP_PRESENT_CLASS_TEACHER_NOT_SUPPORTED"
    )
    answers = [
        f"V00-A0 is {effects['Training trajectory: V00 - A0']:+.4f} pp ({trajectory_label}); this is the frozen joint-training trajectory effect.",
        f"BPS-only V10-V00 is {effects['BPS given SPSR off']:+.4f} pp ({bps_label}).",
        f"SPSR-only V01-V00 is {effects['SPSR given BPS off']:+.4f} pp ({spsr_label}).",
        f"The factorial interaction is {effects['Interaction']:+.4f} pp ({interaction_label}).",
        f"Learned negative gamma is {'better' if negative_better else 'not better'} than zero by {sign_metrics['negative']['delta_vs_zero_pp']:+.4f} pp.",
        f"Positive sign flip is {'worse' if positive_worse else 'not worse'} than learned negative by {100*(sign_metrics['positive']['mIoU']-sign_metrics['negative']['mIoU']):+.4f} pp.",
        f"With GT-present classes, deep local accuracy is {100*oracle_overall['deep_accuracy']:.4f}% versus raw28_1 {100*oracle_overall['raw28_1_accuracy']:.4f}%; present-confusion error is {100*oracle_deep_confusion['present_confusion_error_rate']:.4f}% versus {100*oracle_raw_confusion['present_confusion_error_rate']:.4f}% ({teacher_present_confusion_finding}).",
        f"Deep teacher opportunities are help={oracle_overall['deep_help']:,}, harm={oracle_overall['deep_harm']:,}, net={oracle_overall['teacher_net']:,}.",
        f"At GT boundary bins, deep accuracy is {100*oracle_boundary['deep_accuracy']:.4f}% versus raw28_1 {100*oracle_boundary['raw28_1_accuracy']:.4f}% ({'unreliable' if oracle_boundary['deep_accuracy'] < oracle_boundary['raw28_1_accuracy'] else 'not worse'}).",
        f"BPS boundary precision/recall/F1 are {boundary_quality['precision']:.4f}/{boundary_quality['recall']:.4f}/{boundary_quality['f1']:.4f}; B2 contamination is {100*boundary_quality['b2_interior_contamination']:.2f}%.",
        f"BPS B0/B1/B2 net recoveries are {spatial['BPS_V10_minus_V00']['B0_le_2']['net']:,}/{spatial['BPS_V10_minus_V00']['B1_3_7']['net']:,}/{spatial['BPS_V10_minus_V00']['B2_ge_8']['net']:,} pixels.",
        f"C1 decomposition is trajectory={class_decomposition['1']['trajectory']:+.4f}, BPS={class_decomposition['1']['bps_main']:+.4f}, SPSR={class_decomposition['1']['spsr_main']:+.4f}, interaction={class_decomposition['1']['interaction']:+.4f} pp; C3 is trajectory={class_decomposition['3']['trajectory']:+.4f}, BPS={class_decomposition['3']['bps_main']:+.4f}, SPSR={class_decomposition['3']['spsr_main']:+.4f}, interaction={class_decomposition['3']['interaction']:+.4f} pp.",
        f"The total {effects['Full - A0 total effect']:+.4f} pp splits into trajectory {effects['Training trajectory: V00 - A0']:+.4f} pp and direct deployment {effects['Full - V00 direct innovation effect']:+.4f} pp.",
        f"BPS-CH retention decision: {bps_label}.",
        f"SPSR decision: {spsr_label}; sign/teacher evidence labels are {', '.join(label for label in labels if 'SPSR' in label or 'TEACHER' in label)}.",
        f"The only selected next route is {route}.",
    ]

    s2hr_sha_after = sha256_file(args.s2hr_checkpoint)
    if s2hr_sha_after != s2hr_sha_before:
        raise AssertionError("S2HR checkpoint changed during frozen audit")
    provenance = {
        "source_commit": source_commit,
        "a0_sha256": a0_sha,
        "s2hr_sha256_before": s2hr_sha_before,
        "s2hr_sha256_after": s2hr_sha_after,
        "state_dict_sha256_before": state_dict_sha_before,
        "state_dict_sha256_after": state_dict_sha_after,
        "validation_images": len(dataset),
        "precision": "bf16",
        "optimizer_constructed": False,
        "optimizer_step": False,
        "checkpoint_mutated": False,
        "test_used": False,
        "luad_used": False,
        "full_audit_base_forwards": full_base_forwards,
        "counterfactual_variants_per_base": 5,
        "learned_gamma_spatial": learned_gamma,
        "learned_rho_boundary": learned_rho,
        "boundary_quality_protocol": "all three actual controller maps, unflipped at native 28x28 and pooled",
        "teacher_protocol": "TTA-averaged deep/raw28_1 logits with GT/deployed presence applied only after forward",
        "reference_mIoU": {
            "a0": REFERENCE_A0_MIOU,
            "s2hr_full": REFERENCE_FULL_MIOU,
        },
        "reference_delta_pp": {
            "a0": a0_reference_delta_pp,
            "s2hr_full": full_reference_delta_pp,
        },
        "audit_seconds": audit_elapsed,
        "a0_runtime": a0_runtime,
        "command": " ".join(sys.argv),
    }
    result = {
        "route": route,
        "attribution_labels": labels,
        "metrics": variant_metrics,
        "effects_pp": effects,
        "sign_probe": sign_metrics,
        "teacher_key_rows": teacher_key_rows,
        "present_confusion": present_confusion,
        "teacher_present_confusion_finding": teacher_present_confusion_finding,
        "boundary_quality": boundary_quality,
        "spatial_causal": spatial,
        "error_taxonomy": taxonomy,
        "residual_utility": residual_utility,
        "class_decomposition_pp": class_decomposition,
        "answers": answers,
        "parity": parity,
        "provenance": provenance,
    }

    write_json(output / "provenance" / "manifest.json", provenance)
    checkpoint_hashes = {
        "a0": a0_sha,
        "s2hr_before": s2hr_sha_before,
        "s2hr_after": s2hr_sha_after,
        "state_dict_before": state_dict_sha_before,
        "state_dict_after": state_dict_sha_after,
    }
    write_json(output / "provenance" / "checkpoint_hashes.json", checkpoint_hashes)
    write_json(output / "factorial" / "variant_metrics.json", variant_metrics)
    write_json(output / "factorial" / "factorial_effects.json", effects)
    write_csv(output / "factorial" / "per_class.csv", [
        {"variant": name, "class": class_index, "iou": metrics["class_iou"][str(class_index)]}
        for name, metrics in variant_metrics.items() if name != "Splus"
        for class_index in range(4)
    ])
    write_json(output / "spsr_direction" / "sign_probe.json", sign_metrics)
    write_csv(output / "spsr_direction" / "per_class.csv", [
        {"state": name, "class": class_index, "iou": metrics["class_iou"][str(class_index)]}
        for name, metrics in sign_metrics.items() for class_index in range(4)
    ])
    write_csv(output / "teacher_reliability" / "oracle_presence.csv", [
        row for row in teacher_rows if row["presence"] == "oracle"
    ])
    write_csv(output / "teacher_reliability" / "deployed_presence.csv", [
        row for row in teacher_rows if row["presence"] == "deployed"
    ])
    write_csv(output / "teacher_reliability" / "boundary_bins.csv", [
        row for row in teacher_rows
        if row["class"] == "overall" and row["region"] in ("B0_le_2", "B1_3_7", "B2_ge_8")
    ])
    write_csv(output / "teacher_reliability" / "present_confusion.csv", present_confusion)
    write_json(output / "spsr_direction" / "residual_utility.json", residual_utility)
    write_json(output / "bps_boundary" / "boundary_quality.json", boundary_quality)
    write_csv(output / "bps_boundary" / "spatial_causal.csv", flatten_spatial({
        "BPS_V10_minus_V00": spatial["BPS_V10_minus_V00"]
    }))
    write_csv(output / "error_taxonomy" / "taxonomy.csv", flatten_taxonomy(taxonomy))
    write_json(output / "fida_summary.json", result)
    plot_results(output, variant_metrics, effects, sign_metrics, teacher_rows, boundary_quality, spatial)
    report = render_report(output, result)
    print(json.dumps({
        "route": route, "labels": labels, "effects_pp": effects,
        "variant_mIoU": {name: 100 * value["mIoU"] for name, value in variant_metrics.items()},
        "report": str(report),
    }, indent=2, sort_keys=True), flush=True)
    print("S2HR_FIDA_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
