#!/usr/bin/env python3
"""CRRA-v0 frozen BCSS validation-only region representation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shlex
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tool.GenDataset import Stage1_InferDataset
from tools.crra_v0 import (
    A0_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_IMAGES,
    EXPECTED_SLIDES,
    IMAGE_SIZE,
    REPRESENTATION_DIMS,
    REPRESENTATIONS,
    SEED,
)
from tools.crra_v0.extractor import CRRAFeatureExtractor, extract_batch
from tools.crra_v0.probes import (
    decide_crra,
    discrepancy_rank_test,
    make_fold_assignments,
    run_oof_probe,
    select_candidate,
    slide_bootstrap,
    subset_accuracy,
)
from tools.crra_v0.regions import extract_image_regions
from tools.crra_v0.report import write_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=("bf16",), default="bf16")
    parser.add_argument(
        "--expected-checkpoint-sha256", default=EXPECTED_CHECKPOINT_SHA256
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def validate_scope(args):
    val_root = Path(args.val_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    if val_root.name.lower() != "val" or val_root.parent.name != "BCSS-WSSS":
        raise ValueError("CRRA-v0 requires the BCSS-WSSS/val split exactly")
    if not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise FileNotFoundError("BCSS validation img/ and mask/ are required")
    forbidden = f"{val_root} {output}".lower()
    if "luad" in forbidden or f"{os.sep}test" in forbidden:
        raise ValueError("CRRA-v0 forbids LUAD and test paths")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    for directory in (
        "provenance", "regions", "folds", "diagnostics", "figures", "docs",
        "probes/whole", "probes/core", "probes/core_rim",
    ):
        (output / directory).mkdir(parents=True, exist_ok=True)
    return val_root, checkpoint, output


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = CRRAFeatureExtractor(n_class=4)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(f"Checkpoint mismatch: {incompat}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def extract_features(model, val_root, batch_size, num_workers, amp_dtype):
    dataset = Stage1_InferDataset(str(val_root / "img"), img_size=IMAGE_SIZE)
    dataset.object = sorted(dataset.object)
    if len(dataset) != EXPECTED_IMAGES:
        raise AssertionError(f"Expected {EXPECTED_IMAGES} validation images, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    model = model.cuda()
    rows, whole_parts, core_parts, rim_parts = [], [], [], []
    image_rows, presence_rows = [], []
    counters = {"raw_components": 0, "tiny_components": 0, "proposed_regions": 0}
    image_index = 0
    for names, images in loader:
        proposals, features, presence = extract_batch(
            model, images.cuda(non_blocking=True), amp_dtype=amp_dtype
        )
        for batch_index, image_id in enumerate(names):
            gt_path = val_root / "mask" / f"{image_id}.png"
            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)
            ground_truth = np.asarray(Image.open(gt_path), dtype=np.uint8)
            image_region_rows, arrays, stats = extract_image_regions(
                proposals[batch_index],
                ground_truth,
                features[batch_index],
                image_id,
                image_index,
            )
            rows.extend(image_region_rows)
            whole_parts.append(arrays["z_whole"])
            core_parts.append(arrays["z_core"])
            rim_parts.append(arrays["z_rim"])
            for key in counters:
                counters[key] += int(stats[key])
            image_rows.append({
                "image_index": image_index,
                "image_id": image_id,
                "slide_id": image_id.split("_xmin", 1)[0],
                "region_count": int(stats["proposed_regions"]),
            })
            presence_rows.append({
                "image_index": image_index,
                "image_id": image_id,
                **{f"presence_c{c}": int(presence[batch_index, c]) for c in range(4)},
            })
            image_index += 1
        if image_index % 200 < len(names):
            print(f"EXTRACT_PROGRESS {image_index}/{len(dataset)}", flush=True)
    frame = pd.DataFrame(rows)
    frame.insert(0, "token_index", np.arange(len(frame), dtype=np.int64))
    arrays = {
        "z_whole": np.concatenate(whole_parts, axis=0),
        "z_core": np.concatenate(core_parts, axis=0),
        "z_rim": np.concatenate(rim_parts, axis=0),
    }
    if not all(len(value) == len(frame) for value in arrays.values()):
        raise AssertionError("Metadata/features alignment failure")
    if counters["proposed_regions"] != len(frame):
        raise AssertionError("Region counter mismatch")
    return frame, arrays, pd.DataFrame(image_rows), pd.DataFrame(presence_rows), counters


def exclusion_table(frame, column):
    output = []
    for value, subset in frame.groupby(column, sort=True):
        excluded = int((~subset.common_support).sum())
        output.append({
            column: int(value),
            "proposed_regions": int(len(subset)),
            "excluded_regions": excluded,
            "exclusion_fraction": float(excluded / max(len(subset), 1)),
        })
    return pd.DataFrame(output)


def diagnostic_groups(frame):
    result = {}
    for taxonomy in ("Type-A", "Type-B", "Mixed"):
        subset = frame[frame.taxonomy == taxonomy]
        result[taxonomy] = {
            "n_regions": int(len(subset)),
            "whole_dispersion_mean": float(subset.whole_dispersion.mean()),
            "whole_dispersion_median": float(subset.whole_dispersion.median()),
            "core_dispersion_mean": float(subset.core_dispersion.mean()),
            "core_dispersion_median": float(subset.core_dispersion.median()),
            "rim_dispersion_mean": float(subset.rim_dispersion.mean()),
            "rim_dispersion_median": float(subset.rim_dispersion.median()),
            "core_rim_discrepancy_mean": float(subset.core_rim_discrepancy.mean()),
            "core_rim_discrepancy_median": float(subset.core_rim_discrepancy.median()),
        }
    return result


def save_figures(summary, output):
    reps = summary["representations"]
    labels = ["Whole", "Core", "Core+Rim"]
    names = ["whole", "core", "core_rim"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, [reps[name]["macro_f1"] for name in names])
    axes[0].set_ylabel("OOF Macro-F1")
    axes[0].set_title("Pure common-support regions")
    axes[1].bar(labels, [reps[name]["type_b_accuracy"] for name in names])
    axes[1].set_ylabel("Type-B accuracy")
    axes[1].set_title("Frozen difficult subset")
    figure.tight_layout()
    figure.savefig(output / "figures" / "representation_metrics.png", dpi=180)
    plt.close(figure)

    folds = summary["fold_comparison"]
    x = np.arange(1, 6)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(x, [row["core_delta"] for row in folds], marker="o", label="Core-Whole")
    axis.plot(x, [row["core_rim_delta"] for row in folds], marker="o", label="Core+Rim-Whole")
    axis.set_xlabel("GroupKFold fold")
    axis.set_ylabel("Macro-F1 delta")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "figures" / "fold_deltas.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    seed_everything()
    val_root, checkpoint, output = validate_scope(args)
    started = time.perf_counter()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise AssertionError(
            f"Checkpoint SHA256 mismatch: {checkpoint_sha} != {args.expected_checkpoint_sha256}"
        )
    image_files = sorted((val_root / "img").glob("*.png"))
    mask_files = sorted((val_root / "mask").glob("*.png"))
    if len(image_files) != EXPECTED_IMAGES or len(mask_files) != EXPECTED_IMAGES:
        raise AssertionError("BCSS validation image/mask count mismatch")
    if {path.stem for path in image_files} != {path.stem for path in mask_files}:
        raise AssertionError("BCSS validation image/mask filename mismatch")

    torch.cuda.reset_peak_memory_stats()
    model = load_model(checkpoint)
    frame, arrays, images, presence, counters = extract_features(
        model, val_root, args.batch_size, args.num_workers, args.amp_dtype
    )
    del model
    torch.cuda.empty_cache()
    slides = sorted(frame.slide_id.unique().tolist())
    if len(slides) != EXPECTED_SLIDES:
        raise AssertionError(f"Expected {EXPECTED_SLIDES} slides, found {len(slides)}")

    frame.to_csv(output / "regions" / "metadata.csv", index=False)
    images.to_csv(output / "regions" / "images.csv", index=False)
    presence.to_csv(output / "regions" / "image_presence.csv", index=False)
    np.savez_compressed(
        output / "regions" / "features.npz",
        z_whole=arrays["z_whole"],
        z_core=arrays["z_core"],
        z_rim=arrays["z_rim"],
        z_core_rim=np.concatenate((arrays["z_core"], arrays["z_rim"]), axis=1),
    )

    total = len(frame)
    common_mask = frame.common_support.to_numpy(dtype=bool)
    common_count = int(common_mask.sum())
    coverage = {
        "raw_components": int(counters["raw_components"]),
        "tiny_components": int(counters["tiny_components"]),
        "total_proposed_regions": total,
        "empty_core_regions": int(frame.empty_core.sum()),
        "empty_rim_regions": int(frame.empty_rim.sum()),
        "excluded_regions": int((~common_mask).sum()),
        "common_support_regions": common_count,
        "common_support_fraction": float(common_count / max(total, 1)),
        "coverage_review": (
            "CORE_MASK_COVERAGE_REVIEW" if common_count / max(total, 1) < 0.70 else "COVERAGE_PASS"
        ),
    }
    write_json(output / "diagnostics" / "coverage.json", coverage)
    exclusion_table(frame, "base_predicted_class").to_csv(
        output / "diagnostics" / "exclusion_by_predicted_class.csv", index=False
    )
    exclusion_table(frame, "gt_majority_class").to_csv(
        output / "diagnostics" / "exclusion_by_gt_majority_class.csv", index=False
    )

    pure_mask = common_mask & (frame.purity.to_numpy() >= 0.80)
    pure = frame.loc[pure_mask].copy().reset_index(drop=True)
    if pure.slide_id.nunique() != EXPECTED_SLIDES:
        raise AssertionError("Pure common-support set does not cover all 22 slides")
    labels = pure.gt_majority_class.to_numpy(dtype=np.int64)
    groups = pure.slide_id.to_numpy()
    fold_ids, fold_manifest = make_fold_assignments(labels, groups)
    pure["fold"] = fold_ids
    pure[["token_index", "region_id", "image_id", "slide_id", "fold"]].to_csv(
        output / "folds" / "fold_assignments.csv", index=False
    )
    write_json(output / "folds" / "fold_manifest.json", fold_manifest)

    indices = pure.token_index.to_numpy(dtype=np.int64)
    feature_map = {
        "whole": arrays["z_whole"][indices],
        "core": arrays["z_core"][indices],
        "core_rim": np.concatenate((arrays["z_core"][indices], arrays["z_rim"][indices]), axis=1),
    }
    predictions, metrics, fold_results = {}, {}, {}
    for name in REPRESENTATIONS:
        print(f"PROBE_START {name} n={len(labels)} d={feature_map[name].shape[1]}", flush=True)
        prediction, metric, folds = run_oof_probe(
            feature_map[name], labels, groups, fold_ids
        )
        predictions[name], metrics[name], fold_results[name] = prediction, metric, folds
        pd.DataFrame({
            "token_index": indices,
            "region_id": pure.region_id,
            "slide_id": groups,
            "fold": fold_ids,
            "gt_label": labels,
            "base_predicted_class": pure.base_predicted_class,
            "taxonomy": pure.taxonomy,
            "oof_prediction": prediction,
        }).to_csv(output / "probes" / name / "oof_predictions.csv", index=False)
        write_json(output / "probes" / name / "metrics.json", metric)
        write_json(output / "probes" / name / "fold_metrics.json", folds)
        print(f"PROBE_DONE {name} macro_f1={metric['macro_f1']:.6f}", flush=True)

    type_a = pure.taxonomy.to_numpy() == "Type-A"
    type_b = pure.taxonomy.to_numpy() == "Type-B"
    type_a_accuracy = {
        name: subset_accuracy(labels, predictions[name], type_a) for name in REPRESENTATIONS
    }
    type_b_accuracy = {
        name: subset_accuracy(labels, predictions[name], type_b) for name in REPRESENTATIONS
    }
    bootstrap = slide_bootstrap(labels, predictions, groups)
    decision = decide_crra(
        metrics,
        type_b_accuracy,
        type_a_accuracy,
        fold_results,
        coverage["common_support_fraction"],
        bootstrap,
    )

    representation_summary = {}
    whole_f1 = metrics["whole"]["macro_f1"]
    whole_b = type_b_accuracy["whole"]
    for name in REPRESENTATIONS:
        representation_summary[name] = {
            "dim": REPRESENTATION_DIMS[name],
            **metrics[name],
            "type_a_accuracy": type_a_accuracy[name],
            "type_b_accuracy": type_b_accuracy[name],
            "delta_macro_f1_vs_whole": float(metrics[name]["macro_f1"] - whole_f1),
            "delta_type_b_accuracy_vs_whole": float(type_b_accuracy[name] - whole_b),
        }

    fold_comparison = []
    for fold in range(5):
        whole = fold_results["whole"][fold]["macro_f1"]
        core = fold_results["core"][fold]["macro_f1"]
        dual = fold_results["core_rim"][fold]["macro_f1"]
        fold_comparison.append({
            "fold": fold,
            "whole_f1": whole,
            "core_f1": core,
            "core_delta": float(core - whole),
            "core_rim_f1": dual,
            "core_rim_delta": float(dual - whole),
        })
    fold_stability = {}
    for name, key in (("core", "core_delta"), ("core_rim", "core_rim_delta")):
        values = np.asarray([row[key] for row in fold_comparison])
        fold_stability[name] = {
            "positive_folds": int((values > 0).sum()),
            "mean_delta": float(values.mean()),
            "min_delta": float(values.min()),
            "max_delta": float(values.max()),
        }

    per_class = []
    for class_index in range(4):
        whole = metrics["whole"]["per_class_f1"][str(class_index)]
        core = metrics["core"]["per_class_f1"][str(class_index)]
        dual = metrics["core_rim"]["per_class_f1"][str(class_index)]
        per_class.append({
            "class": class_index,
            "whole_f1": whole,
            "core_f1": core,
            "core_delta": float(core - whole),
            "core_rim_f1": dual,
            "core_rim_delta": float(dual - whole),
        })
    pd.DataFrame(fold_comparison).to_csv(output / "diagnostics" / "fold_comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "diagnostics" / "per_class_comparison.csv", index=False)

    common = frame.loc[common_mask].copy()
    groups_diagnostic = diagnostic_groups(common)
    rank = discrepancy_rank_test(
        common.loc[common.taxonomy == "Type-A", "core_rim_discrepancy"],
        common.loc[common.taxonomy == "Type-B", "core_rim_discrepancy"],
    )
    pd.DataFrame([
        {"group": group, **values} for group, values in groups_diagnostic.items()
    ]).to_csv(output / "diagnostics" / "dispersion_discrepancy.csv", index=False)
    write_json(output / "diagnostics" / "rank_test.json", rank)
    write_json(output / "diagnostics" / "bootstrap.json", bootstrap)

    command = shlex.join([sys.executable, *sys.argv])
    runtime = {
        "elapsed_seconds": float(time.perf_counter() - started),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu": torch.cuda.get_device_name(0),
    }
    summary = {
        "decision": decision,
        "provenance": {
            "a0_commit": A0_COMMIT,
            "audit_commit": args.audit_commit,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "command": command,
            "amp_dtype": args.amp_dtype,
            "seed": SEED,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "dataset": {
            "split": "BCSS validation only",
            "images": int(len(images)),
            "slides": int(len(slides)),
            "slide_ids": slides,
            "pure_common_support_regions": int(len(pure)),
            "type_a_regions": int(type_a.sum()),
            "type_b_regions": int(type_b.sum()),
        },
        "coverage": coverage,
        "representations": representation_summary,
        "fold_comparison": fold_comparison,
        "fold_stability": fold_stability,
        "per_class_comparison": per_class,
        "bootstrap": bootstrap,
        "diagnostics": {
            "groups": groups_diagnostic,
            "type_b_vs_type_a_rank_test": rank,
        },
        "runtime": runtime,
        "stop_boundary": {
            "sshr_training": False,
            "test": False,
            "luad": False,
            "crsr": False,
            "extra_representation": False,
        },
    }
    write_json(output / "provenance" / "run.json", summary["provenance"])
    write_json(output / "summary.json", summary)
    save_figures(summary, output)
    write_report(summary, output / "docs" / "crra_v0_region_representation_audit.md")
    print(f"FINAL_DECISION {decision['decision']}", flush=True)
    print(f"REPRESENTATION_FLAG {decision['representation_flag']}", flush=True)
    print(f"REPORT {output / 'docs' / 'crra_v0_region_representation_audit.md'}", flush=True)


if __name__ == "__main__":
    main()
