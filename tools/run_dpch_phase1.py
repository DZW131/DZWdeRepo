#!/usr/bin/env python3
"""Run EXP-DPCH-001 without training or test-set access.

The primary analysis keeps all compared representations in the CBCCH feature
coordinate system. The locked C0 CH15 operator is applied to the CBCCH
HFRM28_1 input, while the boundary representation is the CBCCH local affinity
output before routing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cbcch, resnet38_cls
from tool.GenDataset import Stage1_InferDataset
from tools.wdch_common import (
    EXPECTED_VAL,
    foreground_boundary_distance,
    read_state,
    set_seed,
    sha256_file,
    verify_validation_root,
    write_json,
)
from research.wdch.bcch import _detached_boundary_map


EXPERIMENT_ID = "EXP-DPCH-001"
EXPECTED_SHA256 = {
    "C0": "44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8",
    "BC-CH": "959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad",
    "CBCCH-A3": "2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e",
}
CLASS_IDS = tuple(range(4))
BOOTSTRAP_SEED = 42
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def checkpoint_path(directory: Path) -> Path:
    return directory / "checkpoints" / "epoch25_final.pth"


def prediction_path(directory: Path) -> Path:
    return directory / "predictions" / "epoch25_validation.npz"


def validate_locked_artifacts(c0_dir: Path, bcch_dir: Path, cbcch_dir: Path):
    directories = {"C0": c0_dir, "BC-CH": bcch_dir, "CBCCH-A3": cbcch_dir}
    completions = {
        name: read_json(directory / "complete.json")
        for name, directory in directories.items()
    }
    if (
        completions["C0"].get("status") != "WDCH_MATCHED_BRANCH_COMPLETE"
        or completions["C0"].get("branch") != "C0"
    ):
        raise AssertionError("Invalid locked C0 completion")
    if completions["BC-CH"].get("status") != "BCCH_MATCHED_COMPLETE":
        raise AssertionError("Invalid locked BC-CH completion")
    if (
        completions["CBCCH-A3"].get("status") != "CBCCH_MATCHED_COMPLETE"
        or completions["CBCCH-A3"].get("variant") != "A3"
    ):
        raise AssertionError("Invalid locked CBCCH-A3 completion")

    digests = {}
    for name, directory in directories.items():
        completion = completions[name]
        if completion.get("epochs") != [21, 22, 23, 24, 25]:
            raise AssertionError(f"{name}: continuation epochs differ")
        if completion.get("test_used") or completion["final_validation"].get("test_used"):
            raise AssertionError(f"{name}: test-set use detected")
        if completion["final_validation"].get("epoch") != 25:
            raise AssertionError(f"{name}: final-checkpoint rule differs")
        path = checkpoint_path(directory)
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256[name]:
            raise AssertionError(f"{name}: checkpoint SHA256 changed: {digest}")
        if completion.get("checkpoint_sha256") != digest:
            raise AssertionError(f"{name}: completion/checkpoint digest mismatch")
        digests[name] = digest
    return completions, digests


def load_predictions(path: Path, include_truth: bool):
    with np.load(path, allow_pickle=False) as data:
        result = {
            "image_ids": data["image_ids"].astype(str),
            "predictions": data["predictions"].copy(),
        }
        if include_truth:
            result["truths"] = data["truths"].copy()
    return result


def load_models(c0_checkpoint: Path, cbcch_checkpoint: Path, device: str):
    c0 = resnet38_cls.Net(4)
    incompat = c0.load_state_dict(read_state(c0_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))

    cbcch = resnet38_cbcch.Net(4, variant="A3")
    incompat = cbcch.load_state_dict(read_state(cbcch_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    cbcch.hfrm_28_1.set_semantic_probe(cbcch.ic1)
    # SSHR overrides ``train`` without returning ``self``; consequently its
    # inherited ``eval`` also returns None and must not be call-chained.
    c0 = c0.to(device)
    cbcch = cbcch.to(device)
    c0.eval()
    cbcch.eval()
    return c0, cbcch


def extract_hfrm28_1_input(model, image: torch.Tensor) -> torch.Tensor:
    x = model.conv1a(image)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)
    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x)
    x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    return F.relu(model.bn45(x))


def cosine_map(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left.float(), dim=1, eps=1.0e-8)
    right = F.normalize(right.float(), dim=1, eps=1.0e-8)
    return (left * right).sum(dim=1, keepdim=True)


def resize_scalar(value: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    value = F.interpolate(
        value.float(), size=shape, mode="bilinear", align_corners=False
    )
    return value[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)


def weighted_centroid(feature: torch.Tensor, weight: torch.Tensor):
    denominator = weight.sum()
    if float(denominator) <= 0.0:
        return None
    centroid = (feature * weight).sum(dim=(2, 3)) / denominator
    return F.normalize(centroid, dim=1, eps=1.0e-8)


def semantic_concentration(
    raw_feature: torch.Tensor,
    semantic_feature: torch.Tensor,
    truth: np.ndarray,
    zones: dict[str, np.ndarray],
):
    raw = F.normalize(raw_feature.float(), dim=1, eps=1.0e-8)
    semantic = F.normalize(semantic_feature.float(), dim=1, eps=1.0e-8)
    shape = tuple(int(value) for value in truth.shape)
    centroids_raw = []
    centroids_semantic = []
    class_rows = []
    totals = {
        "boundary": {"pixels": 0, "raw_sum": 0.0, "semantic_sum": 0.0},
        "interior": {"pixels": 0, "raw_sum": 0.0, "semantic_sum": 0.0},
    }
    for class_id in CLASS_IDS:
        full_class = truth == class_id
        class_weight = torch.from_numpy(full_class.astype(np.float32))[None, None]
        class_weight = class_weight.to(device=raw.device)
        class_weight = F.interpolate(
            class_weight,
            size=raw.shape[-2:],
            mode="area",
        )
        raw_centroid = weighted_centroid(raw, class_weight)
        semantic_centroid = weighted_centroid(semantic, class_weight)
        if raw_centroid is None or semantic_centroid is None:
            continue
        centroids_raw.append(raw_centroid)
        centroids_semantic.append(semantic_centroid)
        raw_similarity = (raw * raw_centroid[:, :, None, None]).sum(1, keepdim=True)
        semantic_similarity = (
            semantic * semantic_centroid[:, :, None, None]
        ).sum(1, keepdim=True)
        raw_similarity = resize_scalar(raw_similarity, shape)
        semantic_similarity = resize_scalar(semantic_similarity, shape)
        row = {"class_id": class_id}
        for output_name, zone_name in (
            ("boundary", "boundary_le_7"),
            ("interior", "interior_ge_8"),
        ):
            mask = full_class & zones[zone_name]
            pixels = int(mask.sum())
            row[f"{output_name}_pixels"] = pixels
            if pixels:
                raw_mean = float(raw_similarity[mask].mean())
                semantic_mean = float(semantic_similarity[mask].mean())
                totals[output_name]["pixels"] += pixels
                totals[output_name]["raw_sum"] += float(raw_similarity[mask].sum())
                totals[output_name]["semantic_sum"] += float(
                    semantic_similarity[mask].sum()
                )
            else:
                raw_mean = semantic_mean = float("nan")
            row[f"raw_{output_name}"] = raw_mean
            row[f"semantic_{output_name}"] = semantic_mean
            row[f"delta_{output_name}"] = semantic_mean - raw_mean
        class_rows.append(row)

    def interclass(centroids):
        if len(centroids) < 2:
            return float("nan")
        matrix = torch.cat(centroids, dim=0)
        similarity = matrix @ matrix.T
        upper = torch.triu(
            torch.ones_like(similarity, dtype=torch.bool), diagonal=1
        )
        return float(similarity[upper].mean())

    image_row = {
        "interclass_raw": interclass(centroids_raw),
        "interclass_semantic": interclass(centroids_semantic),
    }
    image_row["interclass_delta"] = (
        image_row["interclass_semantic"] - image_row["interclass_raw"]
    )
    for zone in ("boundary", "interior"):
        pixels = totals[zone]["pixels"]
        image_row[f"semantic_{zone}_pixels"] = pixels
        image_row[f"raw_{zone}"] = (
            totals[zone]["raw_sum"] / pixels if pixels else float("nan")
        )
        image_row[f"semantic_{zone}"] = (
            totals[zone]["semantic_sum"] / pixels if pixels else float("nan")
        )
        image_row[f"delta_{zone}"] = (
            image_row[f"semantic_{zone}"] - image_row[f"raw_{zone}"]
        )
    return image_row, class_rows


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def finite_std(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.std(ddof=1)) if array.size > 1 else float("nan")


def paired_bootstrap_ci(values, resamples: int, seed: int = BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite values for paired bootstrap")
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(resamples, dtype=np.float64)
    chunk = 256
    for start in range(0, resamples, chunk):
        stop = min(start + chunk, resamples)
        index = rng.integers(0, values.size, size=(stop - start, values.size))
        sampled_means[start:stop] = values[index].mean(axis=1)
    low, high = np.quantile(sampled_means, (0.025, 0.975))
    return {
        "images": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def binary_auroc(positive, negative) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if not positive.size or not negative.size:
        return float("nan")
    values = np.concatenate((positive, negative))
    ranks = stats.rankdata(values, method="average")
    positive_rank_sum = ranks[: positive.size].sum()
    mann_whitney = positive_rank_sum - positive.size * (positive.size + 1) / 2
    return float(mann_whitney / (positive.size * negative.size))


def cohen_d(positive, negative) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size < 2 or negative.size < 2:
        return float("nan")
    variance = (
        (positive.size - 1) * positive.var(ddof=1)
        + (negative.size - 1) * negative.var(ddof=1)
    ) / (positive.size + negative.size - 2)
    if variance <= 0.0:
        return float("nan")
    return float((positive.mean() - negative.mean()) / math.sqrt(variance))


def effect_summary(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    return {
        "positive_count": int(positive.size),
        "negative_count": int(negative.size),
        "positive_mean": float(positive.mean()) if positive.size else float("nan"),
        "negative_mean": float(negative.mean()) if negative.size else float("nan"),
        "mean_delta": (
            float(positive.mean() - negative.mean())
            if positive.size and negative.size
            else float("nan")
        ),
        "auroc": binary_auroc(positive, negative),
        "cohen_d": cohen_d(positive, negative),
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_class_rows(rows: list[dict]):
    output = []
    for class_id in CLASS_IDS:
        selected = [row for row in rows if row["class_id"] == class_id]
        row = {"class_id": class_id, "images": len(selected)}
        for zone in ("boundary", "interior"):
            for metric in ("raw", "semantic", "delta"):
                key = f"{metric}_{zone}"
                row[key] = finite_mean(item[key] for item in selected)
                row[f"{key}_std"] = finite_std(item[key] for item in selected)
            row[f"{zone}_pixels"] = int(
                sum(item[f"{zone}_pixels"] for item in selected)
            )
        row["boundary_fb_fs"] = finite_mean(
            item["boundary_fb_fs"] for item in selected
        )
        row["interior_f_fs"] = finite_mean(
            item["interior_f_fs"] for item in selected
        )
        output.append(row)
    return output


def official_score(completion: dict):
    value = completion["final_validation"]["scores"]["final"]
    return {"mIoU": float(value["mIoU"]), "mDice": float(value["mDice"])}


def render_report(summary: dict) -> str:
    primary = summary["primary"]
    bootstrap = primary["semantic_concentration"]["interior_delta_bootstrap"]
    compatibility = primary["boundary_compatibility"]
    utility = primary["guidance_feasibility_vs_c0"]["pixel_level"]
    gates = summary["gates"]
    locked = summary["locked_validation_reference"]
    class_rows = summary["per_class"]
    direct = summary["cross_checkpoint_sensitivity"]

    def f(value, digits=6):
        return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"

    lines = [
        "# DPCH Phase-1 CH-as-Semantic-Anchor Validation",
        "",
        "## 1. Executive conclusion",
        "",
        f"**Decision: `{summary['decision']}`.**",
        "",
        summary["interpretation"],
        "",
        "This is a validation-only representation audit. It does not train or evaluate a dual-path model, so the guidance result is a feasibility signal rather than causal evidence of benefit.",
        "",
        "## 2. Frozen contract and provenance",
        "",
        f"- Experiment: `{EXPERIMENT_ID}`; implementation commit: `{summary['source_commit']}`.",
        "- BCSS validation only; 3,418 canonical, unflipped 224×224 views; BF16 inference.",
        "- Primary coordinate system: CBCCH HFRM28_1 input `F`; `F_s=CH_C0(F)`; `F_b=P_affinity(F)`.",
        "- Routed CBCCH context and independently forwarded C0 features are secondary controls.",
        "- GT boundary/interior: foreground-class transition distance `≤7 px` / `>7 px`.",
        "- Paired image bootstrap: 10,000 resamples, seed 42.",
        "- No training, test, LUAD, alternate seed, checkpoint selection, or tuning.",
        f"- Exact command: `{summary['command']}`",
        "",
        "| Artifact | SHA256 | Final val mIoU | Final val mDice |",
        "|---|---|---:|---:|",
    ]
    for name in ("C0", "BC-CH", "CBCCH-A3"):
        row = locked[name]
        lines.append(
            f"| {name} | `{summary['checkpoint_sha256'][name]}` | "
            f"{100*row['mIoU']:.4f} | {100*row['mDice']:.4f} |"
        )
    lines += [
        "",
        "## 3. Semantic concentration",
        "",
        "Values are image-balanced pixel-to-own-GT-class-centroid cosine similarities.",
        "",
        "| Region | Raw F | CH anchor F_s | Δ(F_s−F) |",
        "|---|---:|---:|---:|",
    ]
    for zone in ("interior", "boundary"):
        row = primary["semantic_concentration"][zone]
        lines.append(
            f"| {zone} | {f(row['raw'])} | {f(row['semantic'])} | {f(row['delta'])} |"
        )
    lines += [
        "",
        f"- Interior Δ bootstrap 95% CI: **{f(bootstrap['mean'])} [{f(bootstrap['ci95_low'])}, {f(bootstrap['ci95_high'])}]** across {bootstrap['images']} images.",
        f"- Inter-class centroid cosine changes from {f(primary['semantic_concentration']['interclass_raw'])} to {f(primary['semantic_concentration']['interclass_semantic'])} (Δ={f(primary['semantic_concentration']['interclass_delta'])}). Lower is more separated; this is a collapse guardrail, not a gate.",
        "",
        "### Per-class concentration and compatibility",
        "",
        "| Class | Interior raw | Interior F_s | Δ | Boundary raw | Boundary F_s | Δ | Boundary cos(F_b,F_s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(
            f"| {row['class_id']} | {f(row['raw_interior'])} | {f(row['semantic_interior'])} | "
            f"{f(row['delta_interior'])} | {f(row['raw_boundary'])} | "
            f"{f(row['semantic_boundary'])} | {f(row['delta_boundary'])} | "
            f"{f(row['boundary_fb_fs'])} |"
        )
    lines += [
        "",
        "## 4. Boundary-to-CH compatibility",
        "",
        f"- Boundary mean `cos(F_b,F_s)`: **{f(compatibility['boundary_fb_fs'])}**.",
        f"- Interior mean `cos(F,F_s)`: **{f(compatibility['interior_f_fs'])}**.",
        f"- Boundary/interior ratio: **{f(compatibility['ratio'])}**.",
        f"- Routed CBCCH-context-to-CH boundary cosine: {f(compatibility['boundary_routed_fs'])} (secondary control).",
        "",
        "## 5. Guidance-feasibility proxy",
        "",
        "Positive samples are GT-boundary pixels where CBCCH-A3 corrects C0; negative samples are those where it harms C0.",
        "",
        "| Aggregation | Corrected n | Harmed n | Corrected mean | Harmed mean | AUROC | Cohen's d |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Pixel-level (gate) | {utility['positive_count']} | {utility['negative_count']} | {f(utility['positive_mean'])} | {f(utility['negative_mean'])} | {f(utility['auroc'])} | {f(utility['cohen_d'])} |",
    ]
    image_utility = primary["guidance_feasibility_vs_c0"]["image_balanced"]
    lines.append(
        f"| Image-balanced control | {image_utility['positive_count']} | {image_utility['negative_count']} | {f(image_utility['positive_mean'])} | {f(image_utility['negative_mean'])} | {f(image_utility['auroc'])} | {f(image_utility['cohen_d'])} |"
    )
    bcch_utility = primary["guidance_feasibility_vs_bcch"]["pixel_level"]
    lines += [
        "",
        f"Against BC-CH rather than C0 (secondary): AUROC={f(bcch_utility['auroc'])}, Cohen's d={f(bcch_utility['cohen_d'])}.",
        "",
        "## 6. Cross-checkpoint sensitivity",
        "",
        "The following literal comparison forwards C0 and CBCCH independently. It is not used for GO/NO-GO because independent continuation can rotate or rescale channels.",
        "",
        f"- Boundary `cos(F_b^CBCCH,F_s^C0)`: {f(direct['boundary_fb_fs_c0'])}.",
        f"- Interior `cos(F^C0,F_s^C0)`: {f(direct['interior_f_fs_c0'])}.",
        f"- Ratio: {f(direct['ratio'])}.",
        "",
        "## 7. Preregistered gates",
        "",
        "| Gate | Criterion | Observed | Result |",
        "|---|---|---:|:---:|",
        f"| Semantic concentration | Δ>0 and bootstrap CI low>0 | Δ={f(bootstrap['mean'])}; low={f(bootstrap['ci95_low'])} | {'PASS' if gates['semantic_concentration']['pass'] else 'FAIL'} |",
        f"| Boundary compatibility | cosine≥0.50 and ratio≥0.80 | cosine={f(compatibility['boundary_fb_fs'])}; ratio={f(compatibility['ratio'])} | {'PASS' if gates['boundary_compatibility']['pass'] else 'FAIL'} |",
        f"| Guidance utility | AUROC>0.55 and d>0.20 | AUROC={f(utility['auroc'])}; d={f(utility['cohen_d'])} | {'PASS' if gates['guidance_utility']['pass'] else 'FAIL'} |",
        "",
        "## 8. Scientific interpretation",
        "",
        summary["scientific_interpretation"],
        "",
        "## 9. Runtime and outputs",
        "",
        f"- Runtime: {summary['runtime']['seconds']/60:.2f} min ({summary['runtime']['seconds_per_image']:.4f} s/image).",
        f"- Peak CUDA allocated memory: {summary['runtime']['peak_cuda_memory_bytes']/2**30:.3f} GiB.",
        "- Machine-readable outputs: `dpch_phase1_summary.json`, `dpch_per_image.csv`, `dpch_per_class_image.csv`, and `dpch_per_class_summary.csv`.",
        "",
        "STOP. No dual-path training was started.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    if "test" in str(args.output_dir).lower():
        raise AssertionError("Output path must not imply test-set evaluation")
    verify_validation_root(args.val_root)
    c0_dir = Path(args.c0_dir)
    bcch_dir = Path(args.bcch_dir)
    cbcch_dir = Path(args.cbcch_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completions, checkpoint_digests = validate_locked_artifacts(
        c0_dir, bcch_dir, cbcch_dir
    )

    c0_predictions = load_predictions(prediction_path(c0_dir), include_truth=True)
    bcch_predictions = load_predictions(prediction_path(bcch_dir), include_truth=False)
    cbcch_predictions = load_predictions(prediction_path(cbcch_dir), include_truth=False)
    for name, values in (
        ("BC-CH", bcch_predictions),
        ("CBCCH-A3", cbcch_predictions),
    ):
        if not np.array_equal(c0_predictions["image_ids"], values["image_ids"]):
            raise AssertionError(f"{name}: stored prediction order differs")
    prediction_index = {
        image_id: index for index, image_id in enumerate(c0_predictions["image_ids"])
    }

    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-DPCH-001 requires CUDA BF16 inference")
    set_seed(42, deterministic=True)
    c0, cbcch = load_models(
        checkpoint_path(c0_dir), checkpoint_path(cbcch_dir), device
    )
    dataset = Stage1_InferDataset(
        os.path.join(args.val_root, "img"), img_size=224
    )
    if len(dataset) != EXPECTED_VAL:
        raise AssertionError(f"Expected {EXPECTED_VAL} images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    expected_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))

    image_rows = []
    class_rows = []
    compatibility_sums = {
        key: [0.0, 0]
        for key in (
            "boundary_fb_fs",
            "interior_f_fs",
            "boundary_routed_fs",
            "boundary_fb_fs_c0",
            "interior_f_fs_c0",
        )
    }
    corrected_c0_values, harmed_c0_values = [], []
    corrected_bcch_values, harmed_bcch_values = [], []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()

    with torch.inference_mode():
        for count, (names, image) in enumerate(loader, start=1):
            if count > expected_images:
                break
            image_id = names[0]
            index = prediction_index[image_id]
            truth = c0_predictions["truths"][index]
            disk_truth = np.asarray(
                Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"),
                dtype=np.uint8,
            )
            if not np.array_equal(truth, disk_truth):
                raise AssertionError(f"Stored/disk truth differs: {image_id}")
            zones = foreground_boundary_distance(truth)
            boundary_mask = zones["boundary_le_7"]
            interior_mask = zones["interior_ge_8"]
            image = image.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw_cbcch = extract_hfrm28_1_input(cbcch, image)
                raw_c0 = extract_hfrm28_1_input(c0, image)
                semantic_primary = c0.hfrm_28_1.context_conv(raw_cbcch)
                semantic_c0 = c0.hfrm_28_1.context_conv(raw_c0)
                semantic_probe = F.normalize(
                    F.relu(cbcch.ic1(raw_cbcch)), dim=1, eps=1.0e-6
                )
                boundary_feature = cbcch.hfrm_28_1.affinity(
                    raw_cbcch, semantic_probe
                )
                boundary_router = _detached_boundary_map(
                    cbcch.hfrm_28_1.haar, raw_cbcch
                )
                routed_context = (
                    (1.0 - boundary_router) * boundary_feature
                    + boundary_router * raw_cbcch
                )

            maps = {
                "boundary_fb_fs": resize_scalar(
                    cosine_map(boundary_feature, semantic_primary), truth.shape
                ),
                "interior_f_fs": resize_scalar(
                    cosine_map(raw_cbcch, semantic_primary), truth.shape
                ),
                "boundary_routed_fs": resize_scalar(
                    cosine_map(routed_context, semantic_primary), truth.shape
                ),
                "boundary_fb_fs_c0": resize_scalar(
                    cosine_map(boundary_feature, semantic_c0), truth.shape
                ),
                "interior_f_fs_c0": resize_scalar(
                    cosine_map(raw_c0, semantic_c0), truth.shape
                ),
            }
            masks = {
                "boundary_fb_fs": boundary_mask,
                "interior_f_fs": interior_mask,
                "boundary_routed_fs": boundary_mask,
                "boundary_fb_fs_c0": boundary_mask,
                "interior_f_fs_c0": interior_mask,
            }
            row = {"image_id": image_id}
            for key, value in maps.items():
                mask = masks[key]
                selected = value[mask]
                row[key] = float(selected.mean()) if selected.size else float("nan")
                compatibility_sums[key][0] += float(selected.sum())
                compatibility_sums[key][1] += int(selected.size)

            semantic_row, semantic_classes = semantic_concentration(
                raw_cbcch, semantic_primary, truth, zones
            )
            row.update(semantic_row)

            c0_prediction = c0_predictions["predictions"][index]
            bcch_prediction = bcch_predictions["predictions"][index]
            cbcch_prediction = cbcch_predictions["predictions"][index]
            corrected_c0 = boundary_mask & (c0_prediction != truth) & (cbcch_prediction == truth)
            harmed_c0 = boundary_mask & (c0_prediction == truth) & (cbcch_prediction != truth)
            corrected_bcch = boundary_mask & (bcch_prediction != truth) & (cbcch_prediction == truth)
            harmed_bcch = boundary_mask & (bcch_prediction == truth) & (cbcch_prediction != truth)
            similarity = maps["boundary_fb_fs"]
            for label, mask, storage in (
                ("corrected_c0", corrected_c0, corrected_c0_values),
                ("harmed_c0", harmed_c0, harmed_c0_values),
                ("corrected_bcch", corrected_bcch, corrected_bcch_values),
                ("harmed_bcch", harmed_bcch, harmed_bcch_values),
            ):
                selected = similarity[mask].astype(np.float32, copy=False)
                row[f"{label}_pixels"] = int(selected.size)
                row[f"{label}_similarity"] = (
                    float(selected.mean()) if selected.size else float("nan")
                )
                if selected.size:
                    storage.append(selected.copy())

            for class_row in semantic_classes:
                class_id = class_row["class_id"]
                class_boundary = boundary_mask & (truth == class_id)
                class_interior = interior_mask & (truth == class_id)
                class_row.update(
                    {
                        "image_id": image_id,
                        "boundary_fb_fs": (
                            float(similarity[class_boundary].mean())
                            if class_boundary.any()
                            else float("nan")
                        ),
                        "interior_f_fs": (
                            float(maps["interior_f_fs"][class_interior].mean())
                            if class_interior.any()
                            else float("nan")
                        ),
                    }
                )
                class_rows.append(class_row)
            image_rows.append(row)
            if count % 100 == 0 or count == expected_images:
                print(
                    f"DPCH_PHASE1_PROGRESS {count}/{expected_images}", flush=True
                )

    torch.cuda.synchronize()
    elapsed = time.time() - started
    write_csv(output / "dpch_per_image.csv", image_rows)
    write_csv(output / "dpch_per_class_image.csv", class_rows)
    per_class = aggregate_class_rows(class_rows)
    write_csv(output / "dpch_per_class_summary.csv", per_class)

    pooled = {
        key: total / count if count else float("nan")
        for key, (total, count) in compatibility_sums.items()
    }
    corrected_c0_values = (
        np.concatenate(corrected_c0_values) if corrected_c0_values else np.empty(0)
    )
    harmed_c0_values = (
        np.concatenate(harmed_c0_values) if harmed_c0_values else np.empty(0)
    )
    corrected_bcch_values = (
        np.concatenate(corrected_bcch_values) if corrected_bcch_values else np.empty(0)
    )
    harmed_bcch_values = (
        np.concatenate(harmed_bcch_values) if harmed_bcch_values else np.empty(0)
    )
    image_corrected_c0 = [row["corrected_c0_similarity"] for row in image_rows]
    image_harmed_c0 = [row["harmed_c0_similarity"] for row in image_rows]
    bootstrap = paired_bootstrap_ci(
        [row["delta_interior"] for row in image_rows],
        resamples=args.bootstrap_resamples,
    )
    semantic_summary = {}
    for zone in ("boundary", "interior"):
        semantic_summary[zone] = {
            "raw": finite_mean(row[f"raw_{zone}"] for row in image_rows),
            "semantic": finite_mean(
                row[f"semantic_{zone}"] for row in image_rows
            ),
            "delta": finite_mean(row[f"delta_{zone}"] for row in image_rows),
        }
    semantic_summary.update(
        {
            "interclass_raw": finite_mean(row["interclass_raw"] for row in image_rows),
            "interclass_semantic": finite_mean(
                row["interclass_semantic"] for row in image_rows
            ),
            "interclass_delta": finite_mean(
                row["interclass_delta"] for row in image_rows
            ),
            "interior_delta_bootstrap": bootstrap,
        }
    )
    boundary_compatibility = {
        "boundary_fb_fs": pooled["boundary_fb_fs"],
        "interior_f_fs": pooled["interior_f_fs"],
        "ratio": pooled["boundary_fb_fs"] / pooled["interior_f_fs"],
        "boundary_routed_fs": pooled["boundary_routed_fs"],
        "aggregation": "foreground-pixel pooled",
    }
    guidance_c0 = {
        "pixel_level": effect_summary(corrected_c0_values, harmed_c0_values),
        "image_balanced": effect_summary(image_corrected_c0, image_harmed_c0),
    }
    guidance_bcch = {
        "pixel_level": effect_summary(corrected_bcch_values, harmed_bcch_values),
        "image_balanced": effect_summary(
            [row["corrected_bcch_similarity"] for row in image_rows],
            [row["harmed_bcch_similarity"] for row in image_rows],
        ),
    }
    gates = {
        "semantic_concentration": {
            "pass": bootstrap["mean"] > 0.0 and bootstrap["ci95_low"] > 0.0,
            "criterion": "interior delta > 0 and paired-bootstrap 95% CI low > 0",
        },
        "boundary_compatibility": {
            "pass": (
                boundary_compatibility["boundary_fb_fs"] >= 0.50
                and boundary_compatibility["ratio"] >= 0.80
            ),
            "criterion": "boundary cosine >= 0.50 and boundary/interior ratio >= 0.80",
        },
        "guidance_utility": {
            "pass": (
                guidance_c0["pixel_level"]["auroc"] > 0.55
                and guidance_c0["pixel_level"]["cohen_d"] > 0.20
            ),
            "criterion": "corrected-vs-harmed AUROC > 0.55 and Cohen d > 0.20",
        },
    }
    is_full = expected_images == EXPECTED_VAL
    if not is_full:
        decision = "DPCH_PHASE1_SMOKE_ONLY"
    else:
        decision = "DPCH_PHASE1_GO" if all(row["pass"] for row in gates.values()) else "DPCH_PHASE1_NOGO"
    if decision == "DPCH_PHASE1_GO":
        interpretation = (
            "All preregistered gates pass: CH improves interior class concentration, "
            "the affinity representation remains compatible with the CH semantic direction, "
            "and higher compatibility predicts CBCCH corrections rather than harms."
        )
        scientific = (
            "The locked evidence supports CH as a candidate semantic anchor for a separately "
            "preregistered dual-path training experiment. It does not prove that the proposed "
            "fusion will improve segmentation; that claim requires a matched causal run."
        )
    elif decision == "DPCH_PHASE1_NOGO":
        failed = [name for name, row in gates.items() if not row["pass"]]
        interpretation = "The frozen CH-anchor hypothesis fails: " + ", ".join(failed) + "."
        scientific = (
            "Under the preregistered definitions, the existing checkpoints do not justify "
            "starting Dual-path CH training. The result should be treated as a mechanism no-go, "
            "not repaired by post-hoc threshold or representation changes."
        )
    else:
        interpretation = "Smoke execution only; no scientific decision is permitted."
        scientific = "Run all 3,418 validation images before interpreting the gates."

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "interpretation": interpretation,
        "scientific_interpretation": scientific,
        "source_commit": source_commit(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "images": expected_images,
        "full_validation": is_full,
        "checkpoint_sha256": checkpoint_digests,
        "locked_validation_reference": {
            name: official_score(completions[name])
            for name in ("C0", "BC-CH", "CBCCH-A3")
        },
        "primary": {
            "coordinate_system": "CBCCH HFRM28_1 input feature",
            "semantic_anchor": "locked C0 CH15 applied to CBCCH F",
            "boundary_feature": "CBCCH P_affinity before routing",
            "semantic_concentration": semantic_summary,
            "boundary_compatibility": boundary_compatibility,
            "guidance_feasibility_vs_c0": guidance_c0,
            "guidance_feasibility_vs_bcch": guidance_bcch,
        },
        "cross_checkpoint_sensitivity": {
            "boundary_fb_fs_c0": pooled["boundary_fb_fs_c0"],
            "interior_f_fs_c0": pooled["interior_f_fs_c0"],
            "ratio": pooled["boundary_fb_fs_c0"] / pooled["interior_f_fs_c0"],
            "used_for_decision": False,
        },
        "per_class": per_class,
        "gates": gates,
        "runtime": {
            "seconds": elapsed,
            "seconds_per_image": elapsed / expected_images,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "precision": "BF16 autocast",
            "view": "canonical unflipped 224x224",
        },
        "test_used": False,
        "training_performed": False,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "resamples": args.bootstrap_resamples,
        },
    }
    write_json(output / "dpch_phase1_summary.json", summary)
    (output / "dpch_phase1_ch_anchor_validation_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"DPCH_PHASE1_COMPLETE decision={decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--bcch-dir", required=True)
    parser.add_argument("--cbcch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
