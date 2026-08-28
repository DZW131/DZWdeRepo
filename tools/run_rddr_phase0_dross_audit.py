#!/usr/bin/env python3
"""Run the frozen RDDR Phase-0 spatial-semantic dross feasibility audit."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls import Net  # noqa: E402
from tool.GenDataset import Stage1_InferDataset  # noqa: E402
from tools.rddr_phase0_common import (  # noqa: E402
    A0_COMMIT,
    CHECKPOINT_SHA256,
    N_CLASS,
    SCORE_NAMES,
    TOP_FRACTIONS,
    BinaryHistogram,
    bootstrap_indices,
    bootstrap_mean,
    bootstrap_ratio,
    canonical_predictions,
    dataset_quantile_threshold,
    diagnostic_forward,
    eligible_error,
    foreground_boundary_distance,
    official_histogram,
    official_scores,
    probability_scores,
    sha256_file,
    write_csv,
    write_json,
)


def safe_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_ap(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.size == 0 or labels.sum() == 0:
        return float("nan")
    return float(average_precision_score(labels, scores))


def source_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_clean():
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip() == ""


def tensor_to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()[0]


def score_maps(diag, truth_shape):
    predictions = canonical_predictions(diag, truth_shape)
    _, _, scores = probability_scores(
        predictions["raw_logits"], predictions["deep_logits"]
    )
    return predictions, {name: tensor_to_numpy(value) for name, value in scores.items()}


def prediction_state(truth, predictions):
    raw = tensor_to_numpy(predictions["raw"]).astype(np.uint8)
    rect = tensor_to_numpy(predictions["rect"]).astype(np.uint8)
    deep = tensor_to_numpy(predictions["deep"]).astype(np.uint8)
    final = tensor_to_numpy(predictions["final"]).astype(np.uint8)
    valid, error = eligible_error(raw, truth)
    return {
        "raw": raw,
        "rect": rect,
        "deep": deep,
        "final": final,
        "valid": valid,
        "error": error,
        "rect_error": (rect != truth) & valid,
        "final_error": (final != truth) & valid,
        "zones": foreground_boundary_distance(truth),
    }


def image_metric_record(image_id, truth, predictions, scores):
    state = prediction_state(truth, predictions)
    raw, rect, deep, final = (state[name] for name in ("raw", "rect", "deep", "final"))
    valid, error = state["valid"], state["error"]
    rect_error, final_error = state["rect_error"], state["final_error"]
    primary = scores["S_JS"]
    row = {
        "image_id": image_id,
        "eligible_pixels": int(valid.sum()),
        "excluded_background_pixels": int((truth == 4).sum()),
        "excluded_ignore_pixels": int(((truth < 0) | (truth > 4)).sum()),
        "error_pixels": int(error.sum()),
        "error_prevalence": float(error.sum() / max(valid.sum(), 1)),
        "S_JS_AUROC": safe_auc(error[valid], primary[valid]),
        "S_JS_AUPRC": safe_ap(error[valid], primary[valid]),
        "S_JS_rect_error_AUROC": safe_auc(rect_error[valid], primary[valid]),
        "S_JS_final_error_AUROC": safe_auc(final_error[valid], primary[valid]),
        "raw_correct": int((valid & ~error).sum()),
        "raw_error_score_sum": float(primary[error].sum(dtype=np.float64)),
        "raw_error_score_sq_sum": float(np.square(primary[error], dtype=np.float64).sum()),
        "raw_correct_score_sum": float(primary[valid & ~error].sum(dtype=np.float64)),
        "raw_correct_score_sq_sum": float(np.square(primary[valid & ~error], dtype=np.float64).sum()),
    }
    zones = state["zones"]
    for zone_name, zone in zones.items():
        mask = zone & valid
        row[f"{zone_name}_pixels"] = int(mask.sum())
        row[f"{zone_name}_errors"] = int((error & mask).sum())
        row[f"{zone_name}_S_JS_AUROC"] = safe_auc(error[mask], primary[mask])
        row[f"{zone_name}_S_JS_AUPRC"] = safe_ap(error[mask], primary[mask])
    for class_id in range(N_CLASS):
        mask = valid & (truth == class_id)
        row[f"class{class_id}_pixels"] = int(mask.sum())
        row[f"class{class_id}_errors"] = int((error & mask).sum())
        row[f"class{class_id}_S_JS_AUROC"] = safe_auc(error[mask], primary[mask])
        row[f"class{class_id}_S_JS_AUPRC"] = safe_ap(error[mask], primary[mask])

    groups = {
        "Corrected_by_CH": error & ~rect_error,
        "Still_Wrong": error & rect_error,
        "Harmed_by_CH": ~error & rect_error & valid,
        "Stable_Correct": ~error & ~rect_error & valid,
    }
    for group, mask in groups.items():
        row[f"ch_{group}_count"] = int(mask.sum())
        row[f"ch_{group}_score_sum"] = float(primary[mask].sum(dtype=np.float64))
        row[f"ch_{group}_score_sq_sum"] = float(np.square(primary[mask], dtype=np.float64).sum())
    harmed_or_stable = groups["Harmed_by_CH"] | groups["Stable_Correct"]
    harmed_label = groups["Harmed_by_CH"][harmed_or_stable]
    row["ch_harmed_vs_stable_AUROC"] = safe_auc(
        harmed_label, primary[harmed_or_stable]
    )
    return row, state


def aggregate_ratio(rows, numerator, denominator):
    top = sum(float(row.get(numerator, 0)) for row in rows)
    bottom = sum(float(row.get(denominator, 0)) for row in rows)
    return float(top / bottom) if bottom > 0 else float("nan")


def add_threshold_counts(row, truth, state, scores, thresholds):
    valid, error, deep = state["valid"], state["error"], state["deep"]
    raw = state["raw"]
    for score_name in SCORE_NAMES:
        for fraction in TOP_FRACTIONS:
            label = f"top{int(100 * fraction):02d}"
            flagged = valid & (scores[score_name] >= thresholds[score_name][fraction])
            correctable = flagged & error & (deep == truth)
            harmful = flagged & ~error & (deep != truth)
            prefix = f"{score_name}_{label}"
            row[f"{prefix}_flagged"] = int(flagged.sum())
            row[f"{prefix}_errors"] = int((flagged & error).sum())
            row[f"{prefix}_correctable"] = int(correctable.sum())
            row[f"{prefix}_harmful"] = int(harmful.sum())
            if score_name == "S_JS":
                hybrid = raw.copy()
                hybrid[flagged] = deep[flagged]
                histogram = official_histogram(truth, hybrid)
                for r in range(5):
                    for c in range(5):
                        row[f"S_JS_{label}_hybrid_hist_{r}_{c}"] = int(histogram[r, c])

    flagged20 = valid & (scores["S_JS"] >= thresholds["S_JS"][0.20])
    for zone_name, zone in state["zones"].items():
        mask = zone & valid
        prefix = f"{zone_name}_top20"
        row[f"{prefix}_flagged"] = int((flagged20 & mask).sum())
        row[f"{prefix}_errors"] = int((flagged20 & mask & error).sum())
        row[f"{prefix}_correctable"] = int((flagged20 & mask & error & (deep == truth)).sum())
        row[f"{prefix}_harmful"] = int((flagged20 & mask & ~error & (deep != truth)).sum())
    for class_id in range(N_CLASS):
        mask = valid & (truth == class_id)
        prefix = f"class{class_id}_top20"
        row[f"{prefix}_flagged"] = int((flagged20 & mask).sum())
        row[f"{prefix}_errors"] = int((flagged20 & mask & error).sum())
        row[f"{prefix}_correctable"] = int((flagged20 & mask & error & (deep == truth)).sum())
        row[f"{prefix}_harmful"] = int((flagged20 & mask & ~error & (deep != truth)).sum())


def ratio_ci(rows, numerator, denominator, indices):
    return bootstrap_ratio(
        [row.get(numerator, 0) for row in rows],
        [row.get(denominator, 0) for row in rows],
        indices,
    )


def enrichment_ci(rows, prefix, population_prefix, indices):
    flagged_rate, _ = ratio_ci(rows, f"{prefix}_errors", f"{prefix}_flagged", indices)
    if population_prefix:
        error_key, pixel_key = f"{population_prefix}errors", f"{population_prefix}pixels"
    else:
        error_key, pixel_key = "error_pixels", "eligible_pixels"
    prevalence, _ = ratio_ci(rows, error_key, pixel_key, indices)
    values = flagged_rate / prevalence
    finite = values[np.isfinite(values)]
    observed = aggregate_ratio(rows, f"{prefix}_errors", f"{prefix}_flagged") / aggregate_ratio(
        rows, error_key, pixel_key
    )
    return values, {
        "observed": float(observed),
        "bootstrap_mean": float(finite.mean()),
        "ci95_low": float(np.quantile(finite, 0.025)),
        "ci95_high": float(np.quantile(finite, 0.975)),
    }


def net_ci(rows, prefix, indices):
    numerator = [row.get(f"{prefix}_correctable", 0) - row.get(f"{prefix}_harmful", 0) for row in rows]
    return bootstrap_ratio(numerator, [row.get(f"{prefix}_flagged", 0) for row in rows], indices)


def summarize_score_bins(rows, thresholds, histogram_results):
    output = []
    prevalence = aggregate_ratio(rows, "error_pixels", "eligible_pixels")
    for score_name in SCORE_NAMES:
        for fraction in TOP_FRACTIONS:
            label = f"top{int(100 * fraction):02d}"
            prefix = f"{score_name}_{label}"
            flagged = sum(row[f"{prefix}_flagged"] for row in rows)
            errors = sum(row[f"{prefix}_errors"] for row in rows)
            correctable = sum(row[f"{prefix}_correctable"] for row in rows)
            harmful = sum(row[f"{prefix}_harmful"] for row in rows)
            output.append({
                "score": score_name,
                "top_fraction": fraction,
                "threshold": thresholds[score_name][fraction],
                "flagged_pixels": flagged,
                "coverage": flagged / sum(row["eligible_pixels"] for row in rows),
                "error_rate": errors / max(flagged, 1),
                "error_prevalence": prevalence,
                "enrichment": (errors / max(flagged, 1)) / prevalence,
                "correctable": correctable,
                "harmful": harmful,
                "net_correction_per_flagged": (correctable - harmful) / max(flagged, 1),
                "correction_rate_among_flagged_shallow_wrong": correctable / max(errors, 1),
                "harm_rate_among_flagged_shallow_correct": harmful / max(flagged - errors, 1),
                **histogram_results[score_name],
            })
    return output


def aggregate_ch(rows, indices):
    groups = ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct")
    result, pairwise = {}, []
    boot_means = {}
    for group in groups:
        sums = [row[f"ch_{group}_score_sum"] for row in rows]
        counts = [row[f"ch_{group}_count"] for row in rows]
        estimates, ci = bootstrap_ratio(sums, counts, indices)
        total_sum, total_count = sum(sums), sum(counts)
        sq_sum = sum(row[f"ch_{group}_score_sq_sum"] for row in rows)
        mean = total_sum / max(total_count, 1)
        result[group] = {
            "count": total_count,
            "mean_S_JS": mean,
            "std_S_JS": float(np.sqrt(max(sq_sum / max(total_count, 1) - mean * mean, 0.0))),
            "bootstrap": ci,
        }
        boot_means[group] = estimates
    for left_index, left in enumerate(groups):
        for right in groups[left_index + 1:]:
            difference = boot_means[left] - boot_means[right]
            finite = difference[np.isfinite(difference)]
            pairwise.append({
                "left": left,
                "right": right,
                "observed_difference": result[left]["mean_S_JS"] - result[right]["mean_S_JS"],
                "ci95_low": float(np.quantile(finite, 0.025)),
                "ci95_high": float(np.quantile(finite, 0.975)),
            })
    auc_values, auc_ci = bootstrap_mean(
        [row["ch_harmed_vs_stable_AUROC"] for row in rows], indices
    )
    return result, pairwise, auc_values, auc_ci


def confusion_from_rows(rows, prefix=None):
    histogram = np.zeros((5, 5), dtype=np.int64)
    if prefix is None:
        return histogram
    for row in rows:
        for r in range(5):
            for c in range(5):
                histogram[r, c] += row[f"{prefix}_hist_{r}_{c}"]
    return histogram


def make_report(summary, tensor_contract):
    primary = summary["primary"]
    gates = summary["gates"]
    lines = [
        "# RDDR Phase-0 Spatial-Semantic Dross Feasibility Report",
        "",
        "## 1. Commit and frozen provenance",
        "",
        f"- Audit commit: `{summary['commit']}`",
        f"- Pure A0 commit: `{summary['a0_commit']}`",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        "- Model: evaluation-only; no optimizer, gradient, training, or new checkpoint.",
        "",
        "## 2. Exact command",
        "",
        "```bash",
        " ".join(summary["command"]),
        "```",
        "",
        "## 3. Tensor contract",
        "",
        tensor_contract.strip(),
        "",
        "## 4. Eligible population",
        "",
        f"- Images: `{summary['population']['images']}`",
        f"- Eligible foreground pixels: `{summary['population']['eligible_pixels']}`",
        f"- Excluded evaluator-background pixels: `{summary['population']['excluded_background_pixels']}`",
        f"- Excluded invalid/ignore pixels: `{summary['population']['excluded_ignore_pixels']}`",
        "",
        "## 5. Frozen Primary S_JS",
        "",
        "`S_JS = JS(softmax(ic1(F28_raw)), softmax(upsample(fc8(Ddeep))))`, using natural log, epsilon 1e-8 and temperature 1.0. No GT, boundary, edge, threshold, or learned parameter enters the score.",
        "",
        "## 6. Primary error prevalence",
        "",
        f"Raw shallow foreground error prevalence: `{100 * primary['error_prevalence']:.4f}%`.",
        "",
        "## 7. AUROC / AUPRC",
        "",
        f"- Image-balanced AUROC: `{primary['image_balanced_AUROC']:.6f}`; 95% CI `[{primary['image_balanced_AUROC_CI']['ci95_low']:.6f}, {primary['image_balanced_AUROC_CI']['ci95_high']:.6f}]`.",
        f"- Pixel-weighted AUROC/AUPRC: `{primary['pixel_weighted_AUROC']:.6f}` / `{primary['pixel_weighted_AUPRC']:.6f}`.",
        f"- Image-balanced AUPRC: `{primary['image_balanced_AUPRC']:.6f}`; 95% CI `[{primary['image_balanced_AUPRC_CI']['ci95_low']:.6f}, {primary['image_balanced_AUPRC_CI']['ci95_high']:.6f}]`.",
        f"- Pixel AUPRC/prevalence ratio: `{primary['pixel_AUPRC_over_prevalence']:.6f}`.",
        "",
        "## 8. Fixed Top-k enrichment",
        "",
        "| Quantile | Coverage | Enrichment | Net correction |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary["score_bins"]:
        if row["score"] == "S_JS":
            lines.append(f"| Top {int(100 * row['top_fraction'])}% | {100 * row['coverage']:.3f}% | {row['enrichment']:.4f} | {100 * row['net_correction_per_flagged']:+.4f} pp |")
    lines += [
        "",
        f"Primary Top20 enrichment bootstrap 95% CI: `[{primary['enrichment20_CI']['ci95_low']:.4f}, {primary['enrichment20_CI']['ci95_high']:.4f}]`.",
        "",
        "## 9. Deep correction potential",
        "",
        f"- Top20 correction rate among shallow-wrong pixels: `{100 * primary['correction_rate']:.4f}%`.",
        f"- Top20 harm rate among shallow-correct pixels: `{100 * primary['harm_rate']:.4f}%`.",
        f"- NetCorrection20: `{100 * primary['net_correction20']:+.4f} pp`; 95% CI `[{100 * primary['net_correction20_CI']['ci95_low']:+.4f}, {100 * primary['net_correction20_CI']['ci95_high']:+.4f}] pp`.",
        "",
        "## 10. Oracle Top20 swap diagnostic",
        "",
        f"Raw shallow mIoU/accuracy: `{100 * summary['oracle']['raw']['mIoU']:.4f}` / `{100 * summary['oracle']['raw']['pixel_accuracy']:.4f}`. Hybrid-Top20: `{100 * summary['oracle']['top20']['mIoU']:.4f}` / `{100 * summary['oracle']['top20']['pixel_accuracy']:.4f}`.",
        "",
        "## 11. Boundary / interior",
        "",
        "| Stratum | AUROC [95% CI] | Enrichment20 [95% CI] | NetCorrection20 [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for name in ("boundary", "interior"):
        item = summary["strata"][name]
        lines.append(
            f"| {name} | {item['image_balanced_AUROC']:.6f} "
            f"[{item['AUROC_CI']['ci95_low']:.6f}, {item['AUROC_CI']['ci95_high']:.6f}] | "
            f"{item['enrichment20']:.4f} [{item['enrichment20_CI']['ci95_low']:.4f}, "
            f"{item['enrichment20_CI']['ci95_high']:.4f}] | "
            f"{100 * item['net_correction20']:+.4f} "
            f"[{100 * item['net_correction20_CI']['ci95_low']:+.4f}, "
            f"{100 * item['net_correction20_CI']['ci95_high']:+.4f}] pp |"
        )
    lines += [
        "",
        "## 12. Per-class analysis",
        "",
        "| Class | Error prevalence | AUROC [95% CI] | AUPRC | Enrichment20 [95% CI] | NetCorrection20 [95% CI] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["per_class"]:
        lines.append(
            f"| {item['class']} | {100 * item['error_prevalence']:.4f}% | "
            f"{item['image_balanced_AUROC']:.6f} [{item['AUROC_CI']['ci95_low']:.6f}, "
            f"{item['AUROC_CI']['ci95_high']:.6f}] | {item['image_balanced_AUPRC']:.6f} | "
            f"{item['enrichment20']:.4f} [{item['enrichment20_CI']['ci95_low']:.4f}, "
            f"{item['enrichment20_CI']['ci95_high']:.4f}] | "
            f"{100 * item['net_correction20']:+.4f} "
            f"[{100 * item['net_correction20_CI']['ci95_low']:+.4f}, "
            f"{100 * item['net_correction20_CI']['ci95_high']:+.4f}] pp |"
        )
    lines += [
        "",
        f"Macro AUROC `{summary['per_class_macro_AUROC']:.6f}`; minimum class AUROC `{summary['per_class_min_AUROC']:.6f}`.",
        "",
        "## 13. Correct/error score distributions",
        "",
        f"- Correct S_JS: mean ± std `{summary['distribution']['correct']['mean']:.6f} ± {summary['distribution']['correct']['std']:.6f}`, median `{summary['distribution']['correct']['p50']:.6f}`, p25/p75 `{summary['distribution']['correct']['p25']:.6f}/{summary['distribution']['correct']['p75']:.6f}`.",
        f"- Error S_JS: mean ± std `{summary['distribution']['error']['mean']:.6f} ± {summary['distribution']['error']['std']:.6f}`, median `{summary['distribution']['error']['p50']:.6f}`, p25/p75 `{summary['distribution']['error']['p25']:.6f}/{summary['distribution']['error']['p75']:.6f}`.",
        f"- Error-minus-correct mean difference `{summary['distribution']['mean_difference']:.6f}`; 95% CI `[{summary['distribution']['mean_difference_CI']['ci95_low']:.6f}, {summary['distribution']['mean_difference_CI']['ci95_high']:.6f}]`.",
        f"- Cohen's d `{summary['distribution']['cohen_d']:.6f}`; Cliff's delta `{summary['distribution']['cliffs_delta']:.6f}`.",
        "",
        "## 14. CH transition analysis",
        "",
        "| Group | Pixels | Mean S_JS |",
        "|---|---:|---:|",
    ]
    for name, item in summary["ch_transition"].items():
        lines.append(f"| {name} | {item['count']} | {item['mean_S_JS']:.6f} [{item['bootstrap']['ci95_low']:.6f}, {item['bootstrap']['ci95_high']:.6f}] |")
    lines += [
        "",
        f"Harmed-by-CH vs Stable-Correct image-balanced AUROC: `{summary['ch_harmed_vs_stable_AUROC']:.6f}`.",
        "",
        "Pairwise image-bootstrap mean differences:",
        "",
        "| Left | Right | Difference [95% CI] |",
        "|---|---|---:|",
    ]
    for item in summary["ch_pairwise_bootstrap"]:
        lines.append(
            f"| {item['left']} | {item['right']} | {item['observed_difference']:+.6f} "
            f"[{item['ci95_low']:+.6f}, {item['ci95_high']:+.6f}] |"
        )
    lines += [
        "",
        "## 15. Uncertainty baseline comparison",
        "",
        "| Score | Pixel AUROC | Pixel AUPRC | Enrichment20 | NetCorrection20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["score_comparison"]:
        lines.append(f"| {item['score']} | {item['pixel_weighted_AUROC']:.6f} | {item['pixel_weighted_AUPRC']:.6f} | {item['enrichment20']:.4f} | {100 * item['net_correction20']:+.4f} pp |")
    lines += [
        "",
        "## 16. Secondary error targets",
        "",
        f"S_JS image-balanced AUROC for rectified CAM28_1 error: `{summary['secondary_targets']['rect_AUROC']:.6f}`; for canonical final error: `{summary['secondary_targets']['final_AUROC']:.6f}`.",
        "",
        "## 17. Bootstrap contract",
        "",
        "All reported CIs use 10,000 image-level resamples with seed 42. Dataset-level Top20 thresholds remain fixed during resampling.",
        "",
        "## 18. Four preregistered gates",
        "",
        "| Gate | Requirement | Result | Pass |",
        "|---|---|---:|---:|",
    ]
    for name in ("A", "B", "C", "D"):
        gate = gates[name]
        lines.append(f"| {name} | {gate['requirement']} | {gate['result']} | {gate['pass']} |")
    lines += [
        "",
        "## 19. Scientific interpretation",
        "",
        summary["interpretation"],
        "",
        "## 20. Final decision",
        "",
        f"Decision: `{summary['decision']}`.",
        "",
        f"DECISION = {summary['decision']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--smoke-images", type=int, default=0)
    args = parser.parse_args()
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("RDDR Phase-0 is BCSS validation-only")
    if sha256_file(args.checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("Frozen A0 checkpoint SHA256 mismatch")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not git_clean():
        raise AssertionError("Audit must run from a clean committed worktree")

    dataset = Stage1_InferDataset(str(Path(args.val_root) / "img"), img_size=224)
    dataset.object = sorted(dataset.object)
    if args.smoke_images:
        dataset.object = dataset.object[: args.smoke_images]
    elif len(dataset) != 3418 or args.bootstrap_resamples != 10000:
        raise AssertionError("Formal audit is exactly 3418 images / 10,000 bootstrap resamples")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = Net(4).cuda()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(str(incompatible))
    model.eval()
    if any(parameter.requires_grad and parameter.grad is not None for parameter in model.parameters()):
        raise AssertionError("Unexpected pre-existing gradients")

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    maximum_pixels = len(dataset) * 224 * 224
    image_rows = []
    binary = {name: BinaryHistogram(name) for name in SCORE_NAMES}
    target_binary = {name: BinaryHistogram("S_JS") for name in ("rect", "final")}
    raw_histograms = []
    tensor_shapes = None

    with tempfile.TemporaryDirectory(prefix="rddr_phase0_", dir=output) as cache_dir:
        cache_dir = Path(cache_dir)
        score_cache = {
            name: np.memmap(cache_dir / f"{name}.f32", dtype=np.float32, mode="w+", shape=(maximum_pixels,))
            for name in SCORE_NAMES
        }
        cursor = 0
        with torch.no_grad():
            for index, (name_tuple, image) in enumerate(loader, 1):
                image_id = name_tuple[0]
                truth = np.asarray(Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"), dtype=np.uint8)
                image = image.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, diag = diagnostic_forward(model, image)
                    predictions, scores = score_maps(diag, truth.shape)
                if tensor_shapes is None:
                    tensor_shapes = {name: list(value.shape) for name, value in diag.items()}
                row, state = image_metric_record(image_id, truth, predictions, scores)
                image_rows.append(row)
                raw_histograms.append(official_histogram(truth, state["raw"]))
                valid = state["valid"]
                count = int(valid.sum())
                for score_name in SCORE_NAMES:
                    values = scores[score_name][valid].astype(np.float32)
                    score_cache[score_name][cursor:cursor + count] = values
                    binary[score_name].update(values, state["error"][valid])
                target_binary["rect"].update(scores["S_JS"][valid], state["rect_error"][valid])
                target_binary["final"].update(scores["S_JS"][valid], state["final_error"][valid])
                cursor += count
                if index % 100 == 0 or index == len(dataset):
                    print(f"RDDR_PHASE0_PASS1 {index}/{len(dataset)}", flush=True)

        thresholds = {}
        for score_name in SCORE_NAMES:
            score_cache[score_name].flush()
            values = score_cache[score_name][:cursor]
            thresholds[score_name] = {
                fraction: dataset_quantile_threshold(values, fraction)
                for fraction in TOP_FRACTIONS
            }
        del score_cache

        with torch.no_grad():
            for index, (name_tuple, image) in enumerate(loader, 1):
                image_id = name_tuple[0]
                truth = np.asarray(Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"), dtype=np.uint8)
                image = image.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, diag = diagnostic_forward(model, image)
                    predictions, scores = score_maps(diag, truth.shape)
                state = prediction_state(truth, predictions)
                add_threshold_counts(image_rows[index - 1], truth, state, scores, thresholds)
                if index % 100 == 0 or index == len(dataset):
                    print(f"RDDR_PHASE0_PASS2 {index}/{len(dataset)}", flush=True)

    histogram_results = {name: accumulator.result() for name, accumulator in binary.items()}
    score_bins = summarize_score_bins(image_rows, thresholds, histogram_results)
    indices = bootstrap_indices(len(image_rows), args.bootstrap_resamples, 42)
    primary_auc_samples, primary_auc_ci = bootstrap_mean([row["S_JS_AUROC"] for row in image_rows], indices)
    primary_ap_samples, primary_ap_ci = bootstrap_mean([row["S_JS_AUPRC"] for row in image_rows], indices)
    enrichment_samples, enrichment_ci20 = enrichment_ci(image_rows, "S_JS_top20", "", indices)
    net_samples, net_ci20 = net_ci(image_rows, "S_JS_top20", indices)

    strata = {}
    bootstrap_columns = {
        "S_JS_image_balanced_AUROC": primary_auc_samples,
        "S_JS_image_balanced_AUPRC": primary_ap_samples,
        "S_JS_enrichment20": enrichment_samples,
        "S_JS_net_correction20": net_samples,
    }
    for zone in ("boundary", "interior"):
        auc_samples, auc_ci = bootstrap_mean([row[f"{zone}_S_JS_AUROC"] for row in image_rows], indices)
        enrich_samples, enrich_ci = enrichment_ci(image_rows, f"{zone}_top20", f"{zone}_", indices)
        zone_net_samples, zone_net_ci = net_ci(image_rows, f"{zone}_top20", indices)
        strata[zone] = {
            "image_balanced_AUROC": float(np.nanmean([row[f"{zone}_S_JS_AUROC"] for row in image_rows])),
            "image_balanced_AUPRC": float(np.nanmean([row[f"{zone}_S_JS_AUPRC"] for row in image_rows])),
            "error_prevalence": aggregate_ratio(image_rows, f"{zone}_errors", f"{zone}_pixels"),
            "enrichment20": enrich_ci["observed"],
            "net_correction20": zone_net_ci["observed"],
            "AUROC_CI": auc_ci,
            "enrichment20_CI": enrich_ci,
            "net_correction20_CI": zone_net_ci,
        }
        bootstrap_columns[f"{zone}_AUROC"] = auc_samples
        bootstrap_columns[f"{zone}_enrichment20"] = enrich_samples
        bootstrap_columns[f"{zone}_net20"] = zone_net_samples

    per_class = []
    for class_id in range(N_CLASS):
        auc_samples, auc_ci = bootstrap_mean([row[f"class{class_id}_S_JS_AUROC"] for row in image_rows], indices)
        enrich_samples, enrich_ci = enrichment_ci(image_rows, f"class{class_id}_top20", f"class{class_id}_", indices)
        class_net_samples, class_net_ci = net_ci(image_rows, f"class{class_id}_top20", indices)
        item = {
            "class": class_id,
            "pixels": sum(row[f"class{class_id}_pixels"] for row in image_rows),
            "error_prevalence": aggregate_ratio(image_rows, f"class{class_id}_errors", f"class{class_id}_pixels"),
            "image_balanced_AUROC": float(np.nanmean([row[f"class{class_id}_S_JS_AUROC"] for row in image_rows])),
            "image_balanced_AUPRC": float(np.nanmean([row[f"class{class_id}_S_JS_AUPRC"] for row in image_rows])),
            "enrichment20": enrich_ci["observed"],
            "net_correction20": class_net_ci["observed"],
            "AUROC_CI": auc_ci,
            "enrichment20_CI": enrich_ci,
            "net_correction20_CI": class_net_ci,
        }
        per_class.append(item)
        bootstrap_columns[f"class{class_id}_AUROC"] = auc_samples
        bootstrap_columns[f"class{class_id}_enrichment20"] = enrich_samples
        bootstrap_columns[f"class{class_id}_net20"] = class_net_samples

    ch_transition, ch_pairwise, ch_auc_samples, ch_auc_ci = aggregate_ch(image_rows, indices)
    bootstrap_columns["ch_harmed_vs_stable_AUROC"] = ch_auc_samples

    error_count = sum(row["error_pixels"] for row in image_rows)
    correct_count = sum(row["raw_correct"] for row in image_rows)
    error_sum = sum(row["raw_error_score_sum"] for row in image_rows)
    correct_sum = sum(row["raw_correct_score_sum"] for row in image_rows)
    error_sq = sum(row["raw_error_score_sq_sum"] for row in image_rows)
    correct_sq = sum(row["raw_correct_score_sq_sum"] for row in image_rows)
    error_mean, correct_mean = error_sum / error_count, correct_sum / correct_count
    error_std = np.sqrt(max(error_sq / error_count - error_mean**2, 0.0))
    correct_std = np.sqrt(max(correct_sq / correct_count - correct_mean**2, 0.0))
    pooled = np.sqrt(((error_count - 1) * error_std**2 + (correct_count - 1) * correct_std**2) / (error_count + correct_count - 2))
    error_mean_samples, _ = bootstrap_ratio(
        [row["raw_error_score_sum"] for row in image_rows],
        [row["error_pixels"] for row in image_rows],
        indices,
    )
    correct_mean_samples, _ = bootstrap_ratio(
        [row["raw_correct_score_sum"] for row in image_rows],
        [row["raw_correct"] for row in image_rows],
        indices,
    )
    mean_difference_samples = error_mean_samples - correct_mean_samples
    finite_difference = mean_difference_samples[np.isfinite(mean_difference_samples)]
    mean_difference_ci = {
        "observed": error_mean - correct_mean,
        "bootstrap_mean": float(finite_difference.mean()),
        "ci95_low": float(np.quantile(finite_difference, 0.025)),
        "ci95_high": float(np.quantile(finite_difference, 0.975)),
    }
    bootstrap_columns["S_JS_error_minus_correct_mean"] = mean_difference_samples
    distribution = {
        "error": {"count": error_count, "mean": error_mean, "std": error_std,
                  **histogram_results["S_JS"]["positive_quantiles"]},
        "correct": {"count": correct_count, "mean": correct_mean, "std": correct_std,
                    **histogram_results["S_JS"]["negative_quantiles"]},
        "cohen_d": (error_mean - correct_mean) / pooled,
        "cliffs_delta": 2.0 * histogram_results["S_JS"]["pixel_weighted_AUROC"] - 1.0,
        "mean_difference": error_mean - correct_mean,
        "mean_difference_CI": mean_difference_ci,
    }

    raw_hist = np.sum(raw_histograms, axis=0)
    oracle = {"raw": official_scores(raw_hist)}
    for fraction in TOP_FRACTIONS:
        label = f"top{int(100 * fraction):02d}"
        oracle[label] = official_scores(confusion_from_rows(image_rows, f"S_JS_{label}_hybrid"))

    score_comparison = []
    for score_name in SCORE_NAMES:
        row20 = next(row for row in score_bins if row["score"] == score_name and row["top_fraction"] == 0.20)
        score_comparison.append({
            "score": score_name,
            "pixel_weighted_AUROC": histogram_results[score_name]["pixel_weighted_AUROC"],
            "pixel_weighted_AUPRC": histogram_results[score_name]["pixel_weighted_AUPRC"],
            "enrichment20": row20["enrichment"],
            "net_correction20": row20["net_correction_per_flagged"],
        })

    primary_row20 = next(row for row in score_bins if row["score"] == "S_JS" and row["top_fraction"] == 0.20)
    primary = {
        "error_prevalence": error_count / (error_count + correct_count),
        "image_balanced_AUROC": primary_auc_ci["observed"],
        "image_balanced_AUROC_CI": primary_auc_ci,
        "image_balanced_AUPRC": primary_ap_ci["observed"],
        "image_balanced_AUPRC_CI": primary_ap_ci,
        **histogram_results["S_JS"],
        "pixel_AUPRC_over_prevalence": histogram_results["S_JS"]["pixel_weighted_AUPRC"] / (error_count / (error_count + correct_count)),
        "enrichment20": primary_row20["enrichment"],
        "enrichment20_CI": enrichment_ci20,
        "net_correction20": primary_row20["net_correction_per_flagged"],
        "net_correction20_CI": net_ci20,
        "correction_rate": primary_row20["correction_rate_among_flagged_shallow_wrong"],
        "harm_rate": primary_row20["harm_rate_among_flagged_shallow_correct"],
    }

    gates = {
        "A": {
            "requirement": "AUROC >= 0.58 and bootstrap lower > 0.50",
            "result": f"AUROC={primary['image_balanced_AUROC']:.4f}, low={primary_auc_ci['ci95_low']:.4f}",
            "pass": bool(primary["image_balanced_AUROC"] >= 0.58 and primary_auc_ci["ci95_low"] > 0.50),
        },
        "B": {
            "requirement": "Enrichment20 >= 1.40 and bootstrap lower > 1.20",
            "result": f"enrichment={primary['enrichment20']:.4f}, low={enrichment_ci20['ci95_low']:.4f}",
            "pass": bool(primary["enrichment20"] >= 1.40 and enrichment_ci20["ci95_low"] > 1.20),
        },
        "C": {
            "requirement": "NetCorrection20 > 0 and bootstrap lower > 0",
            "result": f"net={primary['net_correction20']:.6f}, low={net_ci20['ci95_low']:.6f}",
            "pass": bool(primary["net_correction20"] > 0 and net_ci20["ci95_low"] > 0),
        },
        "D": {
            "requirement": "Interior AUROC > 0.52 and Enrichment20 > 1.10",
            "result": f"AUROC={strata['interior']['image_balanced_AUROC']:.4f}, enrichment={strata['interior']['enrichment20']:.4f}",
            "pass": bool(strata["interior"]["image_balanced_AUROC"] > 0.52 and strata["interior"]["enrichment20"] > 1.10),
        },
    }
    if all(item["pass"] for item in gates.values()):
        decision = "RDDR_PHASE0_GO"
        interpretation = "All preregistered links are supported: hierarchical disagreement detects local errors, high-disagreement pixels retain positive deep-repair potential, and the signal remains informative in interiors."
    elif gates["A"]["pass"] and gates["B"]["pass"] and not gates["C"]["pass"]:
        decision = "DROSS_EXISTS_DEEP_REPAIR_FAIL"
        interpretation = "Hierarchical disagreement identifies dross, but replacing flagged shallow evidence with deep semantics lacks preregistered positive repair potential."
    elif gates["A"]["pass"] and gates["B"]["pass"] and gates["C"]["pass"] and not gates["D"]["pass"]:
        decision = "DROSS_IS_BOUNDARY_DOMINATED"
        interpretation = "Detection and repair potential exist, but the frozen score does not pass the interior guard and is therefore boundary-dominated."
    else:
        decision = "RDDR_PHASE0_NOGO"
        interpretation = "At least one preregistered dross detection/enrichment gate fails; the RDDR dross line must stop under the frozen contract."

    bootstrap_rows = []
    for sample in range(args.bootstrap_resamples):
        bootstrap_rows.append({"replicate": sample, **{name: values[sample] for name, values in bootstrap_columns.items()}})

    tensor_contract_source = (ROOT / "docs" / "phase0_tensor_contract.md").read_text(encoding="utf-8")
    shutil.copyfile(ROOT / "docs" / "phase0_tensor_contract.md", output / "phase0_tensor_contract.md")
    summary = {
        "status": "SMOKE" if args.smoke_images else "COMPLETE",
        "decision": decision,
        "commit": source_commit(),
        "a0_commit": A0_COMMIT,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "command": [sys.executable, *sys.argv],
        "view": "canonical unflipped 224x224, BF16",
        "temperature": 1.0,
        "epsilon": 1.0e-8,
        "tensor_shapes": tensor_shapes,
        "population": {
            "images": len(image_rows),
            "eligible_pixels": sum(row["eligible_pixels"] for row in image_rows),
            "excluded_background_pixels": sum(row["excluded_background_pixels"] for row in image_rows),
            "excluded_ignore_pixels": sum(row["excluded_ignore_pixels"] for row in image_rows),
        },
        "primary": primary,
        "thresholds": thresholds,
        "score_bins": score_bins,
        "score_comparison": score_comparison,
        "strata": strata,
        "per_class": per_class,
        "per_class_macro_AUROC": float(np.nanmean([row["image_balanced_AUROC"] for row in per_class])),
        "per_class_min_AUROC": float(np.nanmin([row["image_balanced_AUROC"] for row in per_class])),
        "distribution": distribution,
        "ch_transition": ch_transition,
        "ch_pairwise_bootstrap": ch_pairwise,
        "ch_harmed_vs_stable_AUROC": ch_auc_ci["observed"],
        "ch_harmed_vs_stable_AUROC_CI": ch_auc_ci,
        "secondary_targets": {
            "rect_AUROC": float(np.nanmean([row["S_JS_rect_error_AUROC"] for row in image_rows])),
            "final_AUROC": float(np.nanmean([row["S_JS_final_error_AUROC"] for row in image_rows])),
            "rect_pixel_metrics": target_binary["rect"].result(),
            "final_pixel_metrics": target_binary["final"].result(),
        },
        "oracle": oracle,
        "gates": gates,
        "interpretation": interpretation,
        "bootstrap": {"resamples": args.bootstrap_resamples, "seed": 42, "unit": "image"},
        "engineering": {
            "model_eval": True,
            "no_grad": True,
            "optimizer": None,
            "checkpoint_written": False,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "runtime_seconds": time.time() - started,
            "images_per_second_two_pass": 2 * len(image_rows) / (time.time() - started),
            "test_used": False,
            "luad_used": False,
        },
    }
    write_json(output / "rddr_phase0_summary.json", summary)
    write_csv(output / "rddr_phase0_per_image.csv", image_rows)
    write_csv(output / "rddr_phase0_per_class.csv", per_class)
    write_csv(output / "rddr_phase0_score_bins.csv", score_bins)
    write_csv(output / "rddr_phase0_ch_transition.csv", [
        {"row_type": "group", "group": name, **item} for name, item in ch_transition.items()
    ] + [{"row_type": "pairwise", **item} for item in ch_pairwise])
    write_csv(output / "rddr_phase0_bootstrap.csv", bootstrap_rows)
    report = make_report(summary, tensor_contract_source)
    (output / "rddr_phase0_spatial_semantic_dross_feasibility_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"decision": decision, "primary": primary, "gates": gates}, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


if __name__ == "__main__":
    main()
