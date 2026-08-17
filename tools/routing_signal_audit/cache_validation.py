"""Validate and reuse the immutable Phase-0 CAM and fold assets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tool.GenDataset import Stage1_InferDataset
from tools.decision_audit.fusion import prediction_from_scores, score_predictions
from tools.routing_signal_audit import (
    BASELINE_COMMIT,
    BCSS_THRESHOLDS,
    BRANCH_NAMES,
    CHECKPOINT_SHA256,
    EXPECTED_IMAGES,
    EXPECTED_SLIDES,
    OFFICIAL_FUSION,
    PHASE0_PARENT_COMMIT,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    return Path(path).read_text(encoding="utf-8").splitlines()


def load_phase0_assignment(phase0_dir: Path) -> tuple[np.ndarray, list[dict], str]:
    assignment_path = (
        Path(phase0_dir) / "tables" / "class_probe_fold_assignments.csv"
    )
    rows = list(csv.DictReader(assignment_path.open(encoding="utf-8", newline="")))
    if len(rows) != EXPECTED_IMAGES:
        raise RuntimeError(
            f"Phase-0 assignment has {len(rows)} rows, expected {EXPECTED_IMAGES}"
        )
    fold_by_index = np.full(EXPECTED_IMAGES, -1, dtype=np.int8)
    seen = set()
    group_folds: dict[str, set[int]] = {}
    for row in rows:
        index = int(row["index"])
        fold = int(row["fold"])
        if index in seen or not 0 <= index < EXPECTED_IMAGES or not 0 <= fold < 5:
            raise RuntimeError("Phase-0 fold assignment is duplicated or out of range")
        seen.add(index)
        fold_by_index[index] = fold
        group_folds.setdefault(row["source_group"], set()).add(fold)
    if seen != set(range(EXPECTED_IMAGES)) or np.any(fold_by_index < 0):
        raise RuntimeError("Every validation image must have exactly one Phase-0 fold")
    if len(group_folds) != EXPECTED_SLIDES or any(
        len(folds) != 1 for folds in group_folds.values()
    ):
        raise RuntimeError("Phase-0 fold assignment leaks or omits source slides")
    return fold_by_index, rows, sha256_file(assignment_path)


def validate_phase0_assets(
    phase0_dir: Path,
    validation_root: Path,
    checkpoint: Path,
    phase0b_parent_commit: str,
) -> dict:
    phase0_dir = Path(phase0_dir).resolve()
    validation_root = Path(validation_root).resolve()
    checkpoint = Path(checkpoint).resolve()
    if phase0b_parent_commit != PHASE0_PARENT_COMMIT:
        raise RuntimeError(
            f"Phase-0B parent mismatch: {phase0b_parent_commit} != {PHASE0_PARENT_COMMIT}"
        )
    phase0_config = _read_json(phase0_dir / "config.json")
    manifest = _read_json(phase0_dir / "cache" / "cache_manifest.json")
    parity = _read_json(phase0_dir / "parity.json")
    expected = {
        "base_commit": (phase0_config.get("base_commit"), BASELINE_COMMIT),
        "checkpoint_sha256": (manifest.get("checkpoint_sha256"), CHECKPOINT_SHA256),
        "dataset": (manifest.get("dataset"), "BCSS"),
        "split": (manifest.get("split"), "val"),
        "num_images": (manifest.get("num_images"), EXPECTED_IMAGES),
        "num_masks": (manifest.get("num_masks"), EXPECTED_IMAGES),
        "num_source_groups": (manifest.get("num_source_groups"), EXPECTED_SLIDES),
        "fusion_official": (manifest.get("fusion_official"), list(OFFICIAL_FUSION)),
        "class_presence_thresholds": (
            manifest.get("class_presence_thresholds"),
            list(BCSS_THRESHOLDS),
        ),
        "tta": (
            manifest.get("tta"),
            "official three-way identity/horizontal/vertical",
        ),
        "cache_complete": (manifest.get("complete"), True),
        "cache_finite": (manifest.get("all_normalized_cams_finite"), True),
        "phase0_test_evaluated": (phase0_config.get("test_evaluated"), False),
        "phase0_training_performed": (
            phase0_config.get("sshr_training_performed"),
            False,
        ),
        "phase0_parity": (parity.get("pass"), True),
    }
    mismatches = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    }
    if mismatches:
        raise RuntimeError(f"Phase-0 cache contract mismatch; STOP: {mismatches}")
    actual_checkpoint_sha = sha256_file(checkpoint)
    if actual_checkpoint_sha != CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA256 mismatch; STOP")

    cache_dir = phase0_dir / "cache"
    names = _read_lines(cache_dir / "image_paths.txt")
    groups = _read_lines(cache_dir / "source_groups.txt")
    if len(names) != EXPECTED_IMAGES or len(groups) != EXPECTED_IMAGES:
        raise RuntimeError("Phase-0 image/source ordering has the wrong length")
    if len(set(groups)) != EXPECTED_SLIDES:
        raise RuntimeError("Phase-0 source slide count mismatch")
    dataset = Stage1_InferDataset(
        data_path=str(validation_root / "img"), img_size=224
    )
    live_names = [Path(path).stem for path in dataset.object]
    if live_names != names:
        raise RuntimeError("BCSS validation image ordering differs from Phase-0; STOP")
    mask_names = {path.stem for path in (validation_root / "mask").glob("*.png")}
    if len(mask_names) != EXPECTED_IMAGES or mask_names != set(names):
        raise RuntimeError("BCSS validation image/mask identity mismatch; STOP")
    for name in BRANCH_NAMES:
        array = np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        if array.shape != (EXPECTED_IMAGES, 4, 224, 224) or array.dtype != np.float32:
            raise RuntimeError(f"Invalid Phase-0 CAM cache for {name}; STOP")
    for filename, shape, dtype in (
        ("gt.npy", (EXPECTED_IMAGES, 224, 224), np.uint8),
        ("class_presence.npy", (EXPECTED_IMAGES, 4), np.uint8),
        ("official_predictions.npy", (EXPECTED_IMAGES, 224, 224), np.uint8),
        ("branch_predictions.npy", (EXPECTED_IMAGES, 4, 224, 224), np.uint8),
    ):
        array = np.load(cache_dir / filename, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype:
            raise RuntimeError(f"Invalid Phase-0 cache array {filename}; STOP")
    fold_by_index, assignment_rows, assignment_hash = load_phase0_assignment(
        phase0_dir
    )
    for row in assignment_rows:
        index = int(row["index"])
        if row["image_name"] != names[index] or row["source_group"] != groups[index]:
            raise RuntimeError("Phase-0 fold assignment identity mismatch; STOP")
    return {
        "phase0_dir": str(phase0_dir),
        "phase0b_parent_commit": phase0b_parent_commit,
        "baseline_commit": BASELINE_COMMIT,
        "checkpoint_sha256": actual_checkpoint_sha,
        "image_paths_match": True,
        "image_order_match": True,
        "slide_ids_match": True,
        "num_images": len(names),
        "num_slides": len(set(groups)),
        "fold_assignment_hash": assignment_hash,
        "fold_by_index": fold_by_index,
        "assignment_rows": assignment_rows,
        "image_names": names,
        "source_groups": groups,
        "phase0_parity": parity,
    }


def recheck_exact_parity(phase0_dir: Path, reconstruction_path: Path) -> dict:
    cache_dir = Path(phase0_dir) / "cache"
    cams = [
        np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in BRANCH_NAMES
    ]
    truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    official = np.load(cache_dir / "official_predictions.npy", mmap_mode="r")
    differing = 0
    reconstructed = np.lib.format.open_memmap(
        reconstruction_path,
        mode="w+",
        dtype=np.uint8,
        shape=official.shape,
    )
    for index in range(EXPECTED_IMAGES):
        score = sum(
            weight * cam[index] for weight, cam in zip(OFFICIAL_FUSION, cams)
        )
        prediction = prediction_from_scores(score, presence[index])
        reconstructed[index] = prediction
        differing += int(np.count_nonzero(prediction != official[index]))
    reconstructed.flush()
    released_parity = _read_json(Path(phase0_dir) / "parity.json")
    audit_score = score_predictions(truth, reconstructed)
    released_score = released_parity["released_score"]
    miou_difference = abs(audit_score["Mean IoU"] - released_score["Mean IoU"])
    mdice_difference = abs(audit_score["Mean Dice"] - released_score["Mean Dice"])
    result = {
        "official_mIoU": 100 * audit_score["Mean IoU"],
        "official_mDice": 100 * audit_score["Mean Dice"],
        "released_vs_phase0b_differing_pixels": differing,
        "mIoU_absolute_difference": miou_difference,
        "mDice_absolute_difference": mdice_difference,
        "phase0_released_differing_pixels": released_parity[
            "differing_prediction_pixels"
        ],
    }
    result["pass"] = bool(
        differing == 0
        and released_parity["differing_prediction_pixels"] == 0
        and miou_difference < 1e-7
        and mdice_difference < 1e-7
    )
    if not result["pass"]:
        raise RuntimeError(f"Phase-0B exact parity failed; STOP: {result}")
    return result
