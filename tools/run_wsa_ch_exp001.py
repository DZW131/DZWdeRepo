#!/usr/bin/env python3
"""Run the frozen WSA-CH EXP001 validation-only assignment audit."""

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


EXPERIMENT_ID = "WSA-CH-EXP001"
REPRESENTATIONS = ("raw_F", "CH_F", "CBCCH_Fb")
CLASS_IDS = tuple(range(4))
EXPECTED_SHA256 = {
    "C0": "44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8",
    "CBCCH-A3": "2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e",
}
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 10_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def checkpoint_path(directory: Path) -> Path:
    return directory / "checkpoints" / "epoch25_final.pth"


def validate_locked_artifacts(c0_dir: Path, cbcch_dir: Path):
    directories = {"C0": c0_dir, "CBCCH-A3": cbcch_dir}
    completions = {
        name: read_json(directory / "complete.json")
        for name, directory in directories.items()
    }
    if (
        completions["C0"].get("status") != "WDCH_MATCHED_BRANCH_COMPLETE"
        or completions["C0"].get("branch") != "C0"
    ):
        raise AssertionError("Invalid locked C0 completion")
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
        digest = sha256_file(checkpoint_path(directory))
        if digest != EXPECTED_SHA256[name]:
            raise AssertionError(f"{name}: checkpoint SHA256 changed: {digest}")
        if completion.get("checkpoint_sha256") != digest:
            raise AssertionError(f"{name}: completion/checkpoint digest mismatch")
        digests[name] = digest
    return completions, digests


def load_models(c0_checkpoint: Path, cbcch_checkpoint: Path):
    c0 = resnet38_cls.Net(4)
    incompat = c0.load_state_dict(read_state(c0_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    cbcch = resnet38_cbcch.Net(4, variant="A3")
    incompat = cbcch.load_state_dict(read_state(cbcch_checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    cbcch.hfrm_28_1.set_semantic_probe(cbcch.ic1)
    c0 = c0.cuda()
    cbcch = cbcch.cuda()
    # SSHR's overridden train/eval methods do not return self.
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


def normalized_pairwise_cosine(prototypes: torch.Tensor) -> float:
    if prototypes.shape[0] < 2:
        return float("nan")
    prototypes = F.normalize(prototypes.float(), dim=1, eps=1.0e-8)
    matrix = prototypes @ prototypes.T
    upper = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    return float(matrix[upper].mean().detach())


def build_cam_groups(
    ch_feature: torch.Tensor,
    classifier,
    present_classes: list[int],
):
    """Create deployment-shaped CH groups without segmentation GT."""

    logits = F.relu(classifier(ch_feature)).float()[0, present_classes]
    height, width = logits.shape[-2:]
    weights = torch.softmax(logits.flatten(1), dim=1).view(-1, height, width)
    values = ch_feature.float()[0]
    prototypes = torch.einsum("khw,chw->kc", weights, values)
    prototypes = F.normalize(prototypes, dim=1, eps=1.0e-8)
    entropy = -(weights * weights.clamp_min(1.0e-12).log()).sum(dim=(1, 2))
    normalized_entropy = entropy / math.log(height * width)
    diagnostics = {
        "normalized_spatial_entropy": float(normalized_entropy.mean().detach()),
        "max_spatial_weight": float(weights.amax(dim=(1, 2)).mean().detach()),
        "effective_locations": float(entropy.exp().mean().detach()),
        "prototype_interclass_cosine": normalized_pairwise_cosine(prototypes),
    }
    return prototypes, diagnostics


def build_oracle_groups(
    ch_feature: torch.Tensor,
    truth: np.ndarray,
    interior: np.ndarray,
    present_classes: list[int],
):
    """Create a GT-interior upper-bound prototype set."""

    values = ch_feature.float()[0]
    prototypes = []
    valid_classes = []
    for class_id in present_classes:
        mask = ((truth == class_id) & interior).astype(np.float32)
        if not mask.any():
            continue
        weight = torch.from_numpy(mask)[None, None].to(values.device)
        weight = F.interpolate(weight, size=values.shape[-2:], mode="area")[0, 0]
        denominator = weight.sum()
        if float(denominator) <= 0.0:
            continue
        prototype = (values * weight).sum(dim=(1, 2)) / denominator
        prototypes.append(F.normalize(prototype, dim=0, eps=1.0e-8))
        valid_classes.append(class_id)
    if not prototypes:
        return None, []
    return torch.stack(prototypes), valid_classes


def assignment_arrays(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_classes: list[int],
    truth: np.ndarray,
    boundary: np.ndarray,
):
    """Return hardest-wrong margin and assignment correctness at GT boundary."""

    if prototypes.shape[0] < 2:
        return None
    query = F.normalize(query.float(), dim=1, eps=1.0e-8)[0]
    similarities = torch.einsum("kc,chw->khw", prototypes.float(), query)[None]
    similarities = F.interpolate(
        similarities,
        size=truth.shape,
        mode="bilinear",
        align_corners=False,
    )[0].cpu().numpy()
    class_to_group = np.full(5, -1, dtype=np.int64)
    for group_index, class_id in enumerate(prototype_classes):
        class_to_group[class_id] = group_index
    true_group = class_to_group[truth]
    eligible = boundary & (true_group >= 0)
    y, x = np.nonzero(eligible)
    if y.size == 0:
        return None
    true_index = true_group[y, x]
    selected = similarities[:, y, x].T
    same = selected[np.arange(y.size), true_index]
    wrong = selected.copy()
    wrong[np.arange(y.size), true_index] = -np.inf
    hardest_wrong = wrong.max(axis=1)
    predicted = selected.argmax(axis=1)
    return {
        "margin": (same - hardest_wrong).astype(np.float32),
        "correct": predicted == true_index,
        "truth_class": truth[y, x].astype(np.uint8),
        "same_similarity": same.astype(np.float32),
        "wrong_similarity": hardest_wrong.astype(np.float32),
    }


def summarize_arrays(values: dict):
    pixels = int(values["margin"].size)
    return {
        "pixels": pixels,
        "margin": float(values["margin"].mean()),
        "accuracy": float(values["correct"].mean()),
        "same_similarity": float(values["same_similarity"].mean()),
        "wrong_similarity": float(values["wrong_similarity"].mean()),
    }


class AssignmentAccumulator:
    def __init__(self):
        self.data = {
            representation: {
                "pixels": 0,
                "margin_sum": 0.0,
                "correct": 0,
                "same_sum": 0.0,
                "wrong_sum": 0.0,
            }
            for representation in REPRESENTATIONS
        }
        self.classes = {
            (representation, class_id): {
                "pixels": 0,
                "margin_sum": 0.0,
                "correct": 0,
            }
            for representation in REPRESENTATIONS
            for class_id in CLASS_IDS
        }
        self.difficulty = {
            representation: {
                "raw_easy": 0,
                "raw_hard": 0,
                "easy_harmed": 0,
                "hard_corrected": 0,
                "easy_rep_correct": 0,
                "hard_rep_correct": 0,
            }
            for representation in ("CH_F", "CBCCH_Fb")
        }
        self.chance_weighted_sum = 0.0
        self.chance_pixels = 0

    def update(self, results: dict, chance: float):
        reference_size = results["raw_F"]["margin"].size
        if any(row["margin"].size != reference_size for row in results.values()):
            raise AssertionError("Representations use different eligible boundary pixels")
        self.chance_weighted_sum += chance * reference_size
        self.chance_pixels += reference_size
        for representation, values in results.items():
            row = self.data[representation]
            row["pixels"] += int(values["margin"].size)
            row["margin_sum"] += float(values["margin"].sum())
            row["correct"] += int(values["correct"].sum())
            row["same_sum"] += float(values["same_similarity"].sum())
            row["wrong_sum"] += float(values["wrong_similarity"].sum())
            for class_id in CLASS_IDS:
                mask = values["truth_class"] == class_id
                target = self.classes[(representation, class_id)]
                target["pixels"] += int(mask.sum())
                target["margin_sum"] += float(values["margin"][mask].sum())
                target["correct"] += int(values["correct"][mask].sum())

        raw_correct = results["raw_F"]["correct"]
        raw_easy = raw_correct
        raw_hard = ~raw_correct
        for representation in ("CH_F", "CBCCH_Fb"):
            rep_correct = results[representation]["correct"]
            row = self.difficulty[representation]
            row["raw_easy"] += int(raw_easy.sum())
            row["raw_hard"] += int(raw_hard.sum())
            row["easy_harmed"] += int((raw_easy & ~rep_correct).sum())
            row["hard_corrected"] += int((raw_hard & rep_correct).sum())
            row["easy_rep_correct"] += int(rep_correct[raw_easy].sum())
            row["hard_rep_correct"] += int(rep_correct[raw_hard].sum())

    def result(self):
        overall = {}
        for representation, row in self.data.items():
            pixels = max(row["pixels"], 1)
            overall[representation] = {
                **row,
                "margin": row["margin_sum"] / pixels,
                "accuracy": row["correct"] / pixels,
                "same_similarity": row["same_sum"] / pixels,
                "wrong_similarity": row["wrong_sum"] / pixels,
            }
        per_class = []
        for representation in REPRESENTATIONS:
            for class_id in CLASS_IDS:
                row = self.classes[(representation, class_id)]
                pixels = max(row["pixels"], 1)
                per_class.append(
                    {
                        "representation": representation,
                        "class_id": class_id,
                        **row,
                        "margin": row["margin_sum"] / pixels,
                        "accuracy": row["correct"] / pixels,
                    }
                )
        difficulty = {}
        for representation, row in self.difficulty.items():
            difficulty[representation] = {
                **row,
                "hard_correction_rate": row["hard_corrected"] / max(row["raw_hard"], 1),
                "easy_harm_rate": row["easy_harmed"] / max(row["raw_easy"], 1),
                "accuracy_on_raw_easy": row["easy_rep_correct"] / max(row["raw_easy"], 1),
                "accuracy_on_raw_hard": row["hard_rep_correct"] / max(row["raw_hard"], 1),
            }
        return {
            "overall": overall,
            "per_class": per_class,
            "difficulty": difficulty,
            "chance_accuracy": self.chance_weighted_sum / max(self.chance_pixels, 1),
        }


def finite_values(values):
    output = np.asarray(list(values), dtype=np.float64)
    return output[np.isfinite(output)]


def finite_mean(values) -> float:
    values = finite_values(values)
    return float(values.mean()) if values.size else float("nan")


def paired_bootstrap_ci(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    values = finite_values(values)
    if not values.size:
        raise ValueError("No finite values for bootstrap")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        index = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[index].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return {
        "images": int(values.size),
        "mean": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": int(resamples),
        "seed": int(seed),
    }


def image_balanced_summary(rows: list[dict], prefix: str):
    output = {}
    for representation in REPRESENTATIONS:
        output[representation] = {
            "images": sum(
                np.isfinite(row[f"{prefix}_{representation}_margin"])
                for row in rows
            ),
            "margin": finite_mean(
                row[f"{prefix}_{representation}_margin"] for row in rows
            ),
            "accuracy": finite_mean(
                row[f"{prefix}_{representation}_accuracy"] for row in rows
            ),
            "same_similarity": finite_mean(
                row[f"{prefix}_{representation}_same_similarity"] for row in rows
            ),
            "wrong_similarity": finite_mean(
                row[f"{prefix}_{representation}_wrong_similarity"] for row in rows
            ),
        }
    return output


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def official_score(completion: dict):
    value = completion["final_validation"]["scores"]["final"]
    return {"mIoU": float(value["mIoU"]), "mDice": float(value["mDice"])}


def render_report(summary: dict) -> str:
    primary = summary["primary"]
    image = primary["image_balanced"]
    pixel = primary["pixel_pooled"]
    oracle = summary["oracle_ceiling"]
    gates = summary["gates"]
    diagnostics = summary["group_diagnostics"]

    def fmt(value, digits=6):
        return "nan" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

    lines = [
        "# WSA-CH EXP001 Semantic Assignment Feasibility Audit",
        "",
        "## 1. Executive conclusion",
        "",
        f"**Decision: `{summary['decision']}`.**",
        "",
        summary["interpretation"],
        "",
        "This is a validation-only diagnostic. No WSA-CH model was trained and the oracle ceiling cannot unlock a future run.",
        "",
        "## 2. Frozen protocol and provenance",
        "",
        f"- Implementation commit: `{summary['source_commit']}`.",
        "- BCSS validation only; canonical unflipped 224×224 view; BF16 inference.",
        "- Same-space HFRM28_1 representations: `F`, `F_CH=CH_C0(F)`, and `F_b=P_affinity(F)`.",
        "- Primary `G_c`: normalized CH feature weighted by `softmax(ReLU(ic1_C0(F_CH)))`, restricted only by image-level foreground labels.",
        "- Oracle `G_c`: GT-interior-weighted CH feature; observation only.",
        "- Boundary: foreground-class transition distance ≤7 px; hardest wrong group defines the margin.",
        "- No training, test, LUAD, threshold, temperature, checkpoint selection, or parameter tuning.",
        f"- Exact command: `{summary['command']}`",
        "",
        "| Artifact | SHA256 | Locked val mIoU | Locked val mDice |",
        "|---|---|---:|---:|",
    ]
    for name in ("C0", "CBCCH-A3"):
        reference = summary["locked_validation_reference"][name]
        lines.append(
            f"| {name} | `{summary['checkpoint_sha256'][name]}` | "
            f"{100*reference['mIoU']:.4f} | {100*reference['mDice']:.4f} |"
        )
    lines += [
        "",
        "## 3. Primary automatic-group assignment",
        "",
        "| Query | Image-balanced margin | Image-balanced acc. | Pixel margin | Pixel acc. | Same sim. | Hardest-wrong sim. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {"raw_F": "Raw F", "CH_F": "CH(F)", "CBCCH_Fb": "CBCCH F_b"}
    for representation in REPRESENTATIONS:
        image_row = image[representation]
        pixel_row = pixel["overall"][representation]
        lines.append(
            f"| {labels[representation]} | {fmt(image_row['margin'])} | "
            f"{fmt(image_row['accuracy'])} | {fmt(pixel_row['margin'])} | "
            f"{fmt(pixel_row['accuracy'])} | {fmt(pixel_row['same_similarity'])} | "
            f"{fmt(pixel_row['wrong_similarity'])} |"
        )
    lines += [
        "",
        f"- Image-balanced chance accuracy: {fmt(primary['image_balanced_chance'])}; pixel-weighted chance: {fmt(pixel['chance_accuracy'])}.",
        f"- Eligible images/pixels: {summary['eligibility']['primary_images']} / {pixel['overall']['raw_F']['pixels']}.",
        f"- Excluded: {summary['eligibility']['single_class_images']} single-class images and {summary['eligibility']['no_boundary_images']} multi-class images without eligible GT boundary pixels.",
        "",
        "## 4. Preregistered gates",
        "",
        "| Gate | Bootstrap estimate [95% CI] | Result |",
        "|---|---:|:---:|",
    ]
    for key, title in (
        ("semantic_margin", "F_b margin > 0"),
        ("above_chance", "F_b accuracy − 1/K > 0"),
        ("refinement_gain", "F_b accuracy − Raw F accuracy > 0"),
    ):
        value = gates[key]
        ci = value["bootstrap"]
        lines.append(
            f"| {title} | {fmt(ci['mean'])} [{fmt(ci['ci95_low'])}, {fmt(ci['ci95_high'])}] | "
            f"{'PASS' if value['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## 5. Boundary difficulty analysis",
        "",
        "Easy/hard is frozen by the raw-F assignment, not by post-hoc margin quantiles.",
        "",
        "| Query | Raw-easy pixels | Raw-hard pixels | Hard correction rate | Easy harm rate | Acc. on raw-hard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for representation in ("CH_F", "CBCCH_Fb"):
        row = pixel["difficulty"][representation]
        lines.append(
            f"| {labels[representation]} | {row['raw_easy']} | {row['raw_hard']} | "
            f"{fmt(row['hard_correction_rate'])} | {fmt(row['easy_harm_rate'])} | "
            f"{fmt(row['accuracy_on_raw_hard'])} |"
        )
    lines += [
        "",
        "## 6. Per-class primary assignment",
        "",
        "| Class | Raw margin / acc. | CH margin / acc. | F_b margin / acc. |",
        "|---:|---:|---:|---:|",
    ]
    for class_id in CLASS_IDS:
        selected = {
            row["representation"]: row
            for row in pixel["per_class"]
            if row["class_id"] == class_id
        }
        lines.append(
            f"| {class_id} | {fmt(selected['raw_F']['margin'])} / {fmt(selected['raw_F']['accuracy'])} | "
            f"{fmt(selected['CH_F']['margin'])} / {fmt(selected['CH_F']['accuracy'])} | "
            f"{fmt(selected['CBCCH_Fb']['margin'])} / {fmt(selected['CBCCH_Fb']['accuracy'])} |"
        )
    lines += [
        "",
        "## 7. GT-interior oracle ceiling",
        "",
        f"Eligible oracle images/pixels: {summary['eligibility']['oracle_images']} / {oracle['pixel_pooled']['overall']['raw_F']['pixels']}; excluded for insufficient interior prototypes or boundary pixels: {summary['eligibility']['oracle_excluded_images']}.",
        "",
        "| Query | Image-balanced margin | Image-balanced acc. | Pixel margin | Pixel acc. |",
        "|---|---:|---:|---:|---:|",
    ]
    for representation in REPRESENTATIONS:
        irow = oracle["image_balanced"][representation]
        prow = oracle["pixel_pooled"]["overall"][representation]
        lines.append(
            f"| {labels[representation]} | {fmt(irow['margin'])} | {fmt(irow['accuracy'])} | "
            f"{fmt(prow['margin'])} | {fmt(prow['accuracy'])} |"
        )
    lines += [
        "",
        "The oracle uses segmentation GT and therefore measures representational capacity only; it is not a deployable assignment result.",
        "",
        "## 8. Semantic-group diagnostics",
        "",
        f"- CAM spatial entropy: {fmt(diagnostics['normalized_spatial_entropy']['mean'])} ± {fmt(diagnostics['normalized_spatial_entropy']['std'])}.",
        f"- Mean maximum spatial weight: {fmt(diagnostics['max_spatial_weight']['mean'])}.",
        f"- Effective weighted locations: {fmt(diagnostics['effective_locations']['mean'])}.",
        f"- Inter-group prototype cosine: {fmt(diagnostics['prototype_interclass_cosine']['mean'])} ± {fmt(diagnostics['prototype_interclass_cosine']['std'])}.",
        "",
        "## 9. Scientific interpretation",
        "",
        summary["scientific_interpretation"],
        "",
        "## 10. Validation evidence and artifacts",
        "",
        f"- Processed {summary['images']} images in {summary['runtime']['seconds']:.2f} s ({summary['runtime']['seconds_per_image']:.4f} s/image).",
        f"- Peak CUDA allocated memory: {summary['runtime']['peak_cuda_memory_bytes']/2**30:.3f} GiB.",
        "- Tests: numerical margin/assignment, CAM prototype construction, oracle construction, bootstrap reproducibility, and difficulty accounting.",
        "- Machine-readable outputs: `wsa_ch_exp001_summary.json`, `wsa_ch_exp001_per_image.csv`, and `wsa_ch_exp001_per_class.csv`.",
        "",
        "STOP. No full WSA-CH implementation or training was started.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    verify_validation_root(args.val_root)
    c0_dir = Path(args.c0_dir)
    cbcch_dir = Path(args.cbcch_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completions, digests = validate_locked_artifacts(c0_dir, cbcch_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("WSA-CH EXP001 requires CUDA BF16 inference")
    set_seed(42, deterministic=True)
    c0, cbcch = load_models(checkpoint_path(c0_dir), checkpoint_path(cbcch_dir))
    dataset = Stage1_InferDataset(os.path.join(args.val_root, "img"), img_size=224)
    if len(dataset) != EXPECTED_VAL:
        raise AssertionError(f"Expected {EXPECTED_VAL}, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    expected_images = len(dataset) if args.max_images <= 0 else min(args.max_images, len(dataset))

    primary_accumulator = AssignmentAccumulator()
    oracle_accumulator = AssignmentAccumulator()
    image_rows = []
    diagnostic_rows = []
    eligibility = {
        "single_class_images": 0,
        "no_boundary_images": 0,
        "primary_images": 0,
        "oracle_images": 0,
        "oracle_excluded_images": 0,
    }
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    with torch.inference_mode():
        for index, (names, image) in enumerate(loader, start=1):
            if index > expected_images:
                break
            image_id = names[0]
            truth = np.asarray(
                Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"),
                dtype=np.uint8,
            )
            present_classes = [
                class_id for class_id in CLASS_IDS if np.any(truth == class_id)
            ]
            base_row = {
                "image_id": image_id,
                "candidate_classes": "".join(str(value) for value in present_classes),
                "candidate_count": len(present_classes),
            }
            if len(present_classes) < 2:
                eligibility["single_class_images"] += 1
                image_rows.append(base_row)
                continue
            zones = foreground_boundary_distance(truth)
            boundary = zones["boundary_le_7"]
            interior = zones["interior_ge_8"]
            if not boundary.any():
                eligibility["no_boundary_images"] += 1
                image_rows.append(base_row)
                continue

            image = image.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw_feature = extract_hfrm28_1_input(cbcch, image)
                ch_feature = c0.hfrm_28_1.context_conv(raw_feature)
                boundary_semantic = F.normalize(
                    F.relu(cbcch.ic1(raw_feature)), dim=1, eps=1.0e-6
                )
                boundary_feature = cbcch.hfrm_28_1.affinity(
                    raw_feature, boundary_semantic
                )
                primary_groups, group_diagnostics = build_cam_groups(
                    ch_feature, c0.ic1, present_classes
                )
            representations = {
                "raw_F": raw_feature,
                "CH_F": ch_feature,
                "CBCCH_Fb": boundary_feature,
            }
            primary_results = {
                name: assignment_arrays(
                    value, primary_groups, present_classes, truth, boundary
                )
                for name, value in representations.items()
            }
            if any(value is None for value in primary_results.values()):
                eligibility["no_boundary_images"] += 1
                image_rows.append(base_row)
                continue
            chance = 1.0 / len(present_classes)
            primary_accumulator.update(primary_results, chance)
            eligibility["primary_images"] += 1
            base_row.update({"primary_chance": chance})
            for name, values in primary_results.items():
                for key, value in summarize_arrays(values).items():
                    base_row[f"primary_{name}_{key}"] = value

            oracle_groups, oracle_classes = build_oracle_groups(
                ch_feature, truth, interior, present_classes
            )
            if oracle_groups is not None and len(oracle_classes) >= 2:
                oracle_results = {
                    name: assignment_arrays(
                        value, oracle_groups, oracle_classes, truth, boundary
                    )
                    for name, value in representations.items()
                }
                if all(value is not None for value in oracle_results.values()):
                    oracle_chance = 1.0 / len(oracle_classes)
                    oracle_accumulator.update(oracle_results, oracle_chance)
                    eligibility["oracle_images"] += 1
                    base_row["oracle_chance"] = oracle_chance
                    base_row["oracle_candidate_count"] = len(oracle_classes)
                    for name, values in oracle_results.items():
                        for key, value in summarize_arrays(values).items():
                            base_row[f"oracle_{name}_{key}"] = value
                else:
                    eligibility["oracle_excluded_images"] += 1
            else:
                eligibility["oracle_excluded_images"] += 1
            image_rows.append(base_row)
            diagnostic_rows.append({"image_id": image_id, **group_diagnostics})
            if index % 100 == 0 or index == expected_images:
                print(f"WSA_CH_EXP001_PROGRESS {index}/{expected_images}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.time() - started
    primary_pixel = primary_accumulator.result()
    oracle_pixel = oracle_accumulator.result()
    primary_image_rows = [
        row for row in image_rows if "primary_CBCCH_Fb_margin" in row
    ]
    oracle_image_rows = [
        row for row in image_rows if "oracle_CBCCH_Fb_margin" in row
    ]
    primary_image = image_balanced_summary(primary_image_rows, "primary")
    oracle_image = image_balanced_summary(oracle_image_rows, "oracle")
    image_chance = finite_mean(row["primary_chance"] for row in primary_image_rows)

    margin_bootstrap = paired_bootstrap_ci(
        (row["primary_CBCCH_Fb_margin"] for row in primary_image_rows),
        resamples=args.bootstrap_resamples,
    )
    chance_bootstrap = paired_bootstrap_ci(
        (
            row["primary_CBCCH_Fb_accuracy"] - row["primary_chance"]
            for row in primary_image_rows
        ),
        resamples=args.bootstrap_resamples,
    )
    gain_bootstrap = paired_bootstrap_ci(
        (
            row["primary_CBCCH_Fb_accuracy"] - row["primary_raw_F_accuracy"]
            for row in primary_image_rows
        ),
        resamples=args.bootstrap_resamples,
    )
    gates = {
        "semantic_margin": {
            "pass": margin_bootstrap["mean"] > 0.0 and margin_bootstrap["ci95_low"] > 0.0,
            "bootstrap": margin_bootstrap,
        },
        "above_chance": {
            "pass": chance_bootstrap["mean"] > 0.0 and chance_bootstrap["ci95_low"] > 0.0,
            "bootstrap": chance_bootstrap,
        },
        "refinement_gain": {
            "pass": gain_bootstrap["mean"] > 0.0 and gain_bootstrap["ci95_low"] > 0.0,
            "bootstrap": gain_bootstrap,
        },
    }
    is_full = expected_images == EXPECTED_VAL
    if not is_full:
        decision = "WSA_CH_EXP001_SMOKE_ONLY"
    elif all(value["pass"] for value in gates.values()):
        decision = "WSA_CH_EXP001_GO"
    elif gates["semantic_margin"]["pass"] and gates["above_chance"]["pass"]:
        decision = "WSA_CH_ASSIGNMENT_EXISTS_NO_REFINEMENT_GAIN"
    else:
        decision = "WSA_CH_EXP001_NOGO"

    if decision == "WSA_CH_EXP001_GO":
        interpretation = "Automatic CH groups are assignable and CBCCH boundary refinement adds significant assignment utility."
        scientific = (
            "All frozen gates pass. This supports a separately preregistered WSA-CH "
            "implementation, but the present audit itself is not training evidence."
        )
    elif decision == "WSA_CH_ASSIGNMENT_EXISTS_NO_REFINEMENT_GAIN":
        interpretation = "Semantic assignment exists, but CBCCH boundary refinement does not improve it over raw F."
        scientific = (
            f"CBCCH F_b has positive semantic margin ({margin_bootstrap['mean']:.6f}) "
            f"and exceeds chance ({chance_bootstrap['mean']:.6f}), but its paired "
            f"accuracy gain over raw F is {gain_bootstrap['mean']:.6f} with 95% CI "
            f"[{gain_bootstrap['ci95_low']:.6f}, {gain_bootstrap['ci95_high']:.6f}]. "
            "Thus the CH groups contain assignment structure, while the frozen CBCCH "
            "refinement supplies no validated incremental assignment signal. Full WSA-CH "
            "training is not unlocked under this contract."
        )
    elif decision == "WSA_CH_EXP001_NOGO":
        failed = [name for name, value in gates.items() if not value["pass"]]
        interpretation = "The semantic-assignment feasibility hypothesis fails: " + ", ".join(failed) + "."
        scientific = (
            "The primary automatic CH groups do not satisfy the frozen assignment gates. "
            "Oracle behavior is only an upper bound and cannot override this no-go."
        )
    else:
        interpretation = "Smoke execution only; no scientific decision is permitted."
        scientific = "Run all 3,418 validation images before interpretation."

    diagnostic_summary = {}
    for key in (
        "normalized_spatial_entropy",
        "max_spatial_weight",
        "effective_locations",
        "prototype_interclass_cosine",
    ):
        values = finite_values(row[key] for row in diagnostic_rows)
        diagnostic_summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    summary = {
        "experiment_id": EXPERIMENT_ID,
        "decision": decision,
        "interpretation": interpretation,
        "scientific_interpretation": scientific,
        "source_commit": source_commit(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "images": expected_images,
        "full_validation": is_full,
        "checkpoint_sha256": digests,
        "locked_validation_reference": {
            name: official_score(completions[name])
            for name in ("C0", "CBCCH-A3")
        },
        "eligibility": eligibility,
        "primary": {
            "group_definition": "softmax(ReLU(ic1_C0(F_CH))) weighted F_CH",
            "pixel_pooled": primary_pixel,
            "image_balanced": primary_image,
            "image_balanced_chance": image_chance,
        },
        "oracle_ceiling": {
            "group_definition": "GT-interior weighted F_CH",
            "used_for_decision": False,
            "pixel_pooled": oracle_pixel,
            "image_balanced": oracle_image,
        },
        "group_diagnostics": diagnostic_summary,
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
    }
    write_csv(output / "wsa_ch_exp001_per_image.csv", image_rows)
    per_class_rows = []
    for scope, value in (("primary", primary_pixel), ("oracle", oracle_pixel)):
        for row in value["per_class"]:
            per_class_rows.append({"scope": scope, **row})
    write_csv(output / "wsa_ch_exp001_per_class.csv", per_class_rows)
    write_json(output / "wsa_ch_exp001_summary.json", summary)
    (output / "wsa_ch_exp001_semantic_assignment_feasibility_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"WSA_CH_EXP001_COMPLETE decision={decision}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--c0-dir", required=True)
    parser.add_argument("--cbcch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
