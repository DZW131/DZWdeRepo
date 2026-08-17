"""Frozen A0 CAM extraction, cache manifest, and released-inference parity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tool import iouutils
from tool.GenDataset import Stage1_InferDataset
from tool.infer_fun import _tta_transforms, infer
from tools.decision_audit import (
    BCSS_THRESHOLDS,
    BRANCH_NAMES,
    OFFICIAL_FUSION,
)
from tools.decision_audit.fusion import (
    normalize_cam,
    prediction_from_scores,
    score_predictions,
)


EXPECTED_CHECKPOINT_SHA256 = (
    "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
)
EXPECTED_SAMPLES = 3418
IMAGE_SIZE = 224


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_group(image_name: str) -> str:
    if "_xmin" not in image_name:
        raise ValueError(f"BCSS source/slide ID is not recoverable: {image_name}")
    return image_name.split("_xmin", 1)[0]


def inference_args(num_workers: int) -> argparse.Namespace:
    return argparse.Namespace(
        dataset="bcss",
        img_size=IMAGE_SIZE,
        num_workers=num_workers,
        amp_dtype="bf16",
    )


def capture_released_inference(model, validation_root: Path, num_workers: int):
    """Call released ``infer`` while capturing its exact prediction masks."""
    captured = {}
    original_scores = iouutils.scores

    def capture_scores(label_trues, label_preds, n_class):
        captured["ground_truth"] = [np.array(value, copy=True) for value in label_trues]
        captured["predictions"] = [np.array(value, copy=True) for value in label_preds]
        return original_scores(label_trues, label_preds, n_class)

    iouutils.scores = capture_scores
    try:
        started = time.perf_counter()
        score = infer(
            model,
            str(validation_root),
            4,
            inference_args(num_workers),
            thr=None,
            cam_weights=(0.6, 0.2, 0.2),
        )
        seconds = time.perf_counter() - started
    finally:
        iouutils.scores = original_scores
    if score is None or "predictions" not in captured:
        raise RuntimeError("Released official inference did not complete")
    return score, captured, seconds


def _open_cache_arrays(cache_dir: Path, sample_count: int):
    shape = (sample_count, 4, IMAGE_SIZE, IMAGE_SIZE)
    cams = {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy", mode="w+", dtype=np.float32, shape=shape
        )
        for name in BRANCH_NAMES
    }
    ground_truth = np.lib.format.open_memmap(
        cache_dir / "gt.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, IMAGE_SIZE, IMAGE_SIZE),
    )
    presence = np.lib.format.open_memmap(
        cache_dir / "class_presence.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(sample_count, 4),
    )
    return cams, ground_truth, presence


def cache_frozen_cams(
    model,
    validation_root: Path,
    cache_dir: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    git_commit: str,
    num_workers: int,
) -> dict:
    """Cache exact TTA-aggregated, official-normalized CAMs in FP32."""
    validation_root = Path(validation_root)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Stage1_InferDataset(
        data_path=str(validation_root / "img"), img_size=IMAGE_SIZE
    )
    if len(dataset) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} images, found {len(dataset)}")
    mask_count = len(list((validation_root / "mask").glob("*.png")))
    if mask_count != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} masks, found {mask_count}")
    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    cams, ground_truth_cache, presence_cache = _open_cache_arrays(
        cache_dir, len(dataset)
    )
    raw_minimum = np.zeros((len(dataset), 4, 4), dtype=np.float32)
    raw_maximum = np.zeros_like(raw_minimum)
    raw_mean = np.zeros_like(raw_minimum)
    image_names = []
    source_groups = []
    finite = True
    started = time.perf_counter()

    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        for index, (image_name_tuple, image_tensor) in enumerate(loader):
            image_name = image_name_tuple[0]
            image_names.append(image_name)
            source_groups.append(source_group(image_name))
            image = np.asarray(
                Image.open(validation_root / "img" / f"{image_name}.png").convert("RGB")
            )
            if image.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Unexpected BCSS image size for {image_name}: {image.shape}")
            ground_truth = np.asarray(
                Image.open(validation_root / "mask" / f"{image_name}.png"),
                dtype=np.uint8,
            )
            if ground_truth.shape != (IMAGE_SIZE, IMAGE_SIZE):
                raise ValueError(f"Unexpected mask size for {image_name}: {ground_truth.shape}")
            image_tensor = image_tensor.cuda(non_blocking=True)
            tta_cams = {name: [] for name in BRANCH_NAMES}
            probabilities = []
            for input_flip_dims, cam_flip_dims in _tta_transforms():
                transformed = (
                    torch.flip(image_tensor, dims=input_flip_dims)
                    if input_flip_dims
                    else image_tensor
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    outputs = model.forward_cam(transformed)
                    resized_outputs = [
                        F.interpolate(
                            cam,
                            (IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        for cam in outputs[:4]
                    ]
                for name, cam in zip(BRANCH_NAMES, resized_outputs):
                    if cam_flip_dims:
                        cam = torch.flip(cam, dims=cam_flip_dims)
                    tta_cams[name].append(cam)
                probabilities.append(outputs[4])

            for branch_index, name in enumerate(BRANCH_NAMES):
                raw = (
                    torch.stack(tta_cams[name]).mean(dim=0).detach().float().cpu().numpy()
                )
                raw_minimum[index, branch_index] = raw.min(axis=(1, 2))
                raw_maximum[index, branch_index] = raw.max(axis=(1, 2))
                raw_mean[index, branch_index] = raw.mean(axis=(1, 2))
                normalized = normalize_cam(raw)
                finite = finite and bool(np.isfinite(normalized).all())
                cams[name][index] = normalized
            probability = (
                torch.stack(probabilities).mean(dim=0).detach().float().cpu().numpy()[0]
            )
            presence = (probability > np.asarray(BCSS_THRESHOLDS)).astype(np.uint8)
            if presence.sum() == 0:
                presence[int(np.argmax(probability))] = 1
            ground_truth_cache[index] = ground_truth
            presence_cache[index] = presence

    for array in (*cams.values(), ground_truth_cache, presence_cache):
        array.flush()
    np.savez_compressed(
        cache_dir / "raw_pre_normalization_summary.npz",
        minimum=raw_minimum,
        maximum=raw_maximum,
        mean=raw_mean,
    )
    (cache_dir / "image_paths.txt").write_text(
        "\n".join(image_names) + "\n", encoding="utf-8"
    )
    (cache_dir / "source_groups.txt").write_text(
        "\n".join(source_groups) + "\n", encoding="utf-8"
    )
    manifest = {
        "complete": True,
        "git_commit": git_commit,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": "BCSS",
        "split": "val",
        "num_images": len(dataset),
        "num_masks": mask_count,
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE],
        "cam_shape": [len(dataset), 4, IMAGE_SIZE, IMAGE_SIZE],
        "cam_dtype": "float32",
        "tta": "official three-way identity/horizontal/vertical",
        "cam_names": list(BRANCH_NAMES),
        "fusion_official": list(OFFICIAL_FUSION),
        "class_presence_thresholds": list(BCSS_THRESHOLDS),
        "source_group_rule": "filename prefix before _xmin",
        "num_source_groups": len(set(source_groups)),
        "model_eval": not model.training,
        "model_requires_grad_parameters": sum(
            int(parameter.requires_grad) for parameter in model.parameters()
        ),
        "torch_no_grad": True,
        "all_normalized_cams_finite": finite,
        "test_evaluated": False,
        "seconds": time.perf_counter() - started,
    }
    with (cache_dir / "cache_manifest.json").open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, sort_keys=True)
        output.write("\n")
    return manifest


def reconstruct_official_from_cache(cache_dir: Path):
    cache_dir = Path(cache_dir)
    cams = [
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in BRANCH_NAMES
    ]
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    predictions = np.lib.format.open_memmap(
        cache_dir / "official_predictions.npy",
        mode="w+",
        dtype=np.uint8,
        shape=ground_truth.shape,
    )
    for index in range(len(ground_truth)):
        fused = sum(
            weight * cam[index]
            for weight, cam in zip(OFFICIAL_FUSION, cams)
        )
        predictions[index] = prediction_from_scores(fused, presence[index])
    predictions.flush()
    return score_predictions(ground_truth, predictions), predictions


def exact_official_parity(
    released_score: dict,
    released_capture: dict,
    cache_dir: Path,
) -> dict:
    audit_score, audit_predictions = reconstruct_official_from_cache(cache_dir)
    released_predictions = released_capture["predictions"]
    released_ground_truth = released_capture["ground_truth"]
    ground_truth = np.load(Path(cache_dir) / "gt.npy", mmap_mode="r")
    if len(released_predictions) != len(audit_predictions):
        raise RuntimeError("Released and audit prediction counts differ")
    differing_prediction_pixels = 0
    differing_ground_truth_pixels = 0
    for index in range(len(audit_predictions)):
        differing_prediction_pixels += int(
            np.count_nonzero(released_predictions[index] != audit_predictions[index])
        )
        differing_ground_truth_pixels += int(
            np.count_nonzero(released_ground_truth[index] != ground_truth[index])
        )
    parity = {
        "released_score": released_score,
        "audit_score": audit_score,
        "miou_absolute_difference": abs(
            released_score["Mean IoU"] - audit_score["Mean IoU"]
        ),
        "mdice_absolute_difference": abs(
            released_score["Mean Dice"] - audit_score["Mean Dice"]
        ),
        "differing_prediction_pixels": differing_prediction_pixels,
        "differing_ground_truth_pixels": differing_ground_truth_pixels,
    }
    parity["pass"] = (
        parity["miou_absolute_difference"] < 1e-7
        and parity["mdice_absolute_difference"] < 1e-7
        and differing_prediction_pixels == 0
        and differing_ground_truth_pixels == 0
    )
    return parity

