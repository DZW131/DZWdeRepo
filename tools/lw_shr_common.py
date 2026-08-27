"""Protocol-locked utilities for the LW-SHR matched continuation audit."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from torch.backends import cudnn
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode

from network import resnet38_cls
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
COMMON_EPOCH20_SHA256 = "2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb"
C0_FINAL_SHA256 = "44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8"
SCHEDULE_SHA256 = "fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea"
DATASET_FINGERPRINT = "c0817cf906f36370f3109ed24dac2cfa6f4b7b434bb95bf1728570021c5c5841"
EXPECTED_TRAIN = 23422
EXPECTED_VAL = 3418
N_CLASS = 4
BACKGROUND = 4
BCSS_THRESHOLDS = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
CAM_WEIGHTS = {"56": 0.0, "28_1": 0.6, "28_2": 0.2, "deep": 0.2}
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))
POLY_POWER = 0.9
VARIANT_TO_MODE = {"A1": "fixed", "A2": "learnable", "A3": "joint"}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(item):
        if isinstance(item, dict):
            return {str(key): convert(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(val) for val in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.integer):
            return int(item)
        if isinstance(item, np.floating):
            return float(item)
        return item

    path.write_text(
        json.dumps(convert(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = not deterministic
    cudnn.deterministic = deterministic
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def verify_validation_root(root: str) -> None:
    lowered = str(root).lower()
    if "test" in lowered or "luad" in lowered:
        raise AssertionError("LW-SHR Phase-1 is BCSS validation-only")
    images = sorted((Path(root) / "img").glob("*.png"))
    masks = sorted((Path(root) / "mask").glob("*.png"))
    if len(images) != EXPECTED_VAL or len(masks) != EXPECTED_VAL:
        raise AssertionError(
            f"Expected {EXPECTED_VAL} pairs, got {len(images)} images/{len(masks)} masks"
        )
    if [path.stem for path in images] != [path.stem for path in masks]:
        raise AssertionError("Validation image/mask names differ")


def load_common_checkpoint(path):
    if sha256_file(path) != COMMON_EPOCH20_SHA256:
        raise AssertionError("Common Epoch-20 checkpoint SHA256 mismatch")
    common = torch.load(path, map_location="cpu", weights_only=False)
    if common.get("format") != "WDCH_MATCHED_COMMON_V1":
        raise AssertionError("Unexpected common checkpoint format")
    if common.get("epoch") != 20 or common.get("optimizer_global_step") != 23420:
        raise AssertionError("Common checkpoint is not the frozen Epoch-20 state")
    if common.get("a0_commit") != A0_COMMIT:
        raise AssertionError("Common checkpoint A0 commit mismatch")
    if common.get("dataset_fingerprint") != DATASET_FINGERPRINT:
        raise AssertionError("Common checkpoint dataset fingerprint mismatch")
    return common


def load_schedule(path):
    if sha256_file(path) != SCHEDULE_SHA256:
        raise AssertionError("Matched schedule SHA256 mismatch")
    with np.load(path, allow_pickle=False) as data:
        schedule = {name: data[name].copy() for name in data.files}
    if schedule["indices"].shape != (25, 1171, 20):
        raise AssertionError(f"Unexpected schedule shape: {schedule['indices'].shape}")
    return schedule


class MatchedAugmentationDataset(Dataset):
    """Released resize/flip/normalize algorithm with schedule-keyed randomness."""

    def __init__(self, train_root: str, image_size: int = 224):
        self.base = Stage1_TrainDataset(
            data_path=train_root, dataset="bcss", img_size=image_size
        )
        self.image_size = int(image_size)
        if len(self.base) != EXPECTED_TRAIN:
            raise AssertionError(f"Expected {EXPECTED_TRAIN} samples, got {len(self.base)}")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, request):
        index, augmentation_seed = int(request[0]), int(request[1])
        path, label = self.base.object[index]
        image = Image.open(path).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
        generator = random.Random(augmentation_seed)
        if generator.random() > 0.5:
            image = TF.hflip(image)
        if generator.random() > 0.5:
            image = TF.vflip(image)
        image = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return Path(path).stem, image, label.clone()


class ScheduleBatchSampler(Sampler):
    def __init__(self, schedule, epoch: int):
        self.indices = schedule["indices"]
        self.augmentation_seeds = schedule["augmentation_seeds"]
        self.epoch = int(epoch)

    def __iter__(self):
        for step in range(self.indices.shape[1]):
            yield [
                (int(index), int(seed))
                for index, seed in zip(
                    self.indices[self.epoch, step],
                    self.augmentation_seeds[self.epoch, step],
                )
            ]

    def __len__(self):
        return int(self.indices.shape[1])


def build_model(variant: str, device="cuda"):
    if variant not in VARIANT_TO_MODE:
        raise ValueError(f"Unknown LW-SHR variant: {variant}")
    # All three variants receive identical subband-processor initialization.
    set_seed(42, deterministic=True)
    model = resnet38_cls.Net(
        N_CLASS,
        wavelet_hfrm_mode=VARIANT_TO_MODE[variant],
        wavelet_hfrm_stages="28_1",
    )
    return model.to(device)


def load_variant_from_common(variant: str, common, device="cuda"):
    model = build_model(variant, device="cpu")
    incompatible = model.load_state_dict(common["model_state"], strict=False)
    allowed_prefixes = ("wavelet_bank.", "hfrm_28_1.wavelet_gate.")
    allowed_exact = {"hfrm_28_1.lambda_sf"} if variant == "A3" else set()
    forbidden = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_prefixes) and key not in allowed_exact
    ]
    if forbidden or incompatible.unexpected_keys:
        raise AssertionError(
            f"Checkpoint compatibility failed: forbidden missing={forbidden}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if not incompatible.missing_keys:
        raise AssertionError("LW-SHR checkpoint audit expected new missing keys")
    return model.to(device), {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def build_optimizer(model, max_step: int = 25 * 1171):
    groups = model.get_parameter_groups()
    optimizer_groups = [
        {"params": groups[0], "lr": 0.01, "weight_decay": 5.0e-4},
        {"params": groups[1], "lr": 0.02, "weight_decay": 0.0},
        {"params": groups[2], "lr": 0.10, "weight_decay": 5.0e-4},
        {"params": groups[3], "lr": 0.20, "weight_decay": 0.0},
    ]
    return torchutils.PolyOptimizer(
        optimizer_groups,
        lr=0.01,
        weight_decay=5.0e-4,
        max_step=max_step,
        lr_power=POLY_POWER,
    )


def optimizer_summary(optimizer):
    return [
        {
            "index": index,
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "momentum": float(group["momentum"]),
            "parameters": int(sum(parameter.numel() for parameter in group["params"])),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def restore_named_optimizer_state(model, optimizer, saved):
    restored, skipped = [], []
    parameters = dict(model.named_parameters())
    for name, values in saved["state"].items():
        parameter = parameters.get(name)
        if parameter is None:
            skipped.append({"name": name, "reason": "parameter absent"})
            continue
        compatible = all(
            not torch.is_tensor(value)
            or not value.ndim
            or value.shape == parameter.shape
            for value in values.values()
        )
        if not compatible:
            skipped.append({"name": name, "reason": "state shape differs"})
            continue
        optimizer.state[parameter] = {
            key: value.to(device=parameter.device, dtype=parameter.dtype)
            if torch.is_tensor(value)
            else value
            for key, value in values.items()
        }
        restored.append(name)
    for target, source in zip(optimizer.param_groups, saved["groups"]):
        for key, value in source.items():
            target[key] = value
    return {"restored": restored, "skipped": skipped}


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def minmax_normalize(cam):
    cam = np.asarray(cam, dtype=np.float32)
    minimum = cam.min(axis=(1, 2), keepdims=True)
    maximum = cam.max(axis=(1, 2), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1.0e-8)


def presence_from_probability(probability):
    presence = (
        np.asarray(probability, dtype=np.float32) > BCSS_THRESHOLDS
    ).astype(np.float32)
    if presence.sum() == 0:
        presence[int(np.argmax(probability))] = 1.0
    return presence


def official_histogram(truth, prediction):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64).copy()
    prediction[truth == BACKGROUND] = BACKGROUND
    valid = (truth >= 0) & (truth <= BACKGROUND)
    encoded = 5 * truth[valid] + prediction[valid]
    return np.bincount(encoded, minlength=25).reshape(5, 5).astype(np.int64)


def scores_from_histogram(histogram):
    hist = np.asarray(histogram, dtype=np.float64).copy()
    hist[4, 4] = 0.0
    diagonal = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - diagonal
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diagonal[:4] / union[:4]
    dice = []
    for index in range(4):
        tp = diagonal[index]
        fp = hist[:, index].sum() - tp
        fn = hist[index, :].sum() - tp
        denominator = 2 * tp + fp + fn
        dice.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return {
        "mIoU": float(np.nanmean(iou)),
        "mDice": float(np.mean(dice)),
        "class_iou": {str(i): float(iou[i]) for i in range(4)},
        "class_dice": {str(i): float(dice[i]) for i in range(4)},
        "histogram": hist.astype(np.int64).tolist(),
    }


class OfficialMetricAccumulator:
    def __init__(self):
        self.hist = np.zeros((5, 5), dtype=np.int64)
        self.images = 0

    def update(self, truth, prediction):
        self.hist += official_histogram(truth, prediction)
        self.images += 1

    def result(self):
        result = scores_from_histogram(self.hist)
        result["images"] = int(self.images)
        return result


def foreground_boundary_distance(truth):
    """Exact historical HMA/WD-CH boundary-interior definition."""

    truth = np.asarray(truth, dtype=np.uint8)
    foreground = truth < BACKGROUND
    boundary = np.zeros_like(foreground, dtype=bool)
    height, width = truth.shape
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        left = truth[y0:y1, x0:x1]
        right = truth[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        transition = (left < BACKGROUND) & (right < BACKGROUND) & (left != right)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= transition
    distance = (
        ndimage.distance_transform_edt(~boundary)
        if boundary.any()
        else np.full(truth.shape, np.inf)
    )
    return {
        "boundary_le_7": foreground & (distance <= 7.0),
        "interior_ge_8": foreground & (distance > 7.0),
    }


def restricted_histogram(truth, prediction, mask):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    valid = np.asarray(mask, dtype=bool) & (truth >= 0) & (truth <= BACKGROUND)
    encoded = 5 * truth[valid] + prediction[valid]
    return np.bincount(encoded, minlength=25).reshape(5, 5).astype(np.int64)


class ZoneMetricAccumulator:
    def __init__(self):
        self.data = {
            name: {"pixels": 0, "correct": 0, "histogram": np.zeros((5, 5), dtype=np.int64)}
            for name in ("boundary_le_7", "interior_ge_8")
        }

    def update(self, truth, prediction):
        for name, mask in foreground_boundary_distance(truth).items():
            row = self.data[name]
            row["pixels"] += int(mask.sum())
            row["correct"] += int((mask & (prediction == truth)).sum())
            row["histogram"] += restricted_histogram(truth, prediction, mask)

    def result(self):
        output = {}
        for name, row in self.data.items():
            score = scores_from_histogram(row["histogram"])
            output[name] = {
                "pixels": row["pixels"],
                "correct": row["correct"],
                "accuracy": row["correct"] / max(row["pixels"], 1),
                "restricted_mIoU": score["mIoU"],
                "class_iou": score["class_iou"],
            }
        return output


def component_thresholds(val_root: str):
    areas = {class_id: [] for class_id in range(4)}
    structure = np.ones((3, 3), dtype=np.uint8)
    for mask_path in sorted((Path(val_root) / "mask").glob("*.png")):
        truth = np.asarray(Image.open(mask_path), dtype=np.uint8)
        for class_id in range(4):
            labels, count = ndimage.label(truth == class_id, structure=structure)
            if count:
                areas[class_id].extend(np.bincount(labels.ravel())[1:].tolist())
    return {
        class_id: {
            "component_count": len(values),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
        for class_id, values in areas.items()
    }


class ComponentMetricAccumulator:
    """Historical component recall plus diagnostic size-restricted mIoU."""

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.structure = np.ones((3, 3), dtype=np.uint8)
        self.rows = {}
        self.size_histograms = {
            size: np.zeros((5, 5), dtype=np.int64)
            for size in ("small", "medium", "large")
        }

    def update(self, truth, prediction):
        size_masks = {
            size: np.zeros_like(truth, dtype=bool)
            for size in ("small", "medium", "large")
        }
        for class_id in range(4):
            labels, count = ndimage.label(
                truth == class_id, structure=self.structure
            )
            threshold = self.thresholds[class_id]
            for component_id in range(1, count + 1):
                mask = labels == component_id
                area = int(mask.sum())
                size = (
                    "small"
                    if area <= threshold["q25"]
                    else "medium"
                    if area <= threshold["q75"]
                    else "large"
                )
                row = self.rows.setdefault(
                    (class_id, size),
                    {"components": 0, "pixels": 0, "correct": 0},
                )
                row["components"] += 1
                row["pixels"] += area
                row["correct"] += int((prediction[mask] == class_id).sum())
                size_masks[size] |= mask
        for size, mask in size_masks.items():
            self.size_histograms[size] += restricted_histogram(truth, prediction, mask)

    def result(self):
        per_class = []
        aggregate = {}
        for size in ("small", "medium", "large"):
            pixels = correct = components = 0
            for class_id in range(4):
                row = self.rows.get(
                    (class_id, size),
                    {"components": 0, "pixels": 0, "correct": 0},
                )
                pixels += row["pixels"]
                correct += row["correct"]
                components += row["components"]
                per_class.append(
                    {
                        "class": class_id,
                        "size": size,
                        **self.thresholds[class_id],
                        **row,
                        "recall": row["correct"] / max(row["pixels"], 1),
                    }
                )
            score = scores_from_histogram(self.size_histograms[size])
            aggregate[size] = {
                "components": components,
                "pixels": pixels,
                "correct": correct,
                "historical_component_recall": correct / max(pixels, 1),
                "diagnostic_size_restricted_mIoU": score["mIoU"],
                "diagnostic_class_iou": score["class_iou"],
            }
        return {"aggregate": aggregate, "per_class": per_class}


def paired_image_bootstrap_miou(base_histograms, candidate_histograms, resamples=10000, seed=42):
    base = np.asarray(base_histograms, dtype=np.int64)
    candidate = np.asarray(candidate_histograms, dtype=np.int64)
    if base.shape != candidate.shape or base.ndim != 3 or base.shape[1:] != (5, 5):
        raise ValueError("Paired bootstrap expects matching [N,5,5] histograms")
    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 100):
        count = min(100, resamples - start)
        indices = generator.integers(0, base.shape[0], size=(count, base.shape[0]))
        for offset, sample_indices in enumerate(indices):
            base_score = scores_from_histogram(base[sample_indices].sum(axis=0))["mIoU"]
            candidate_score = scores_from_histogram(
                candidate[sample_indices].sum(axis=0)
            )["mIoU"]
            differences[start + offset] = 100.0 * (candidate_score - base_score)
    return {
        "resamples": int(resamples),
        "seed": int(seed),
        "mean_delta_pp": float(differences.mean()),
        "ci95_low_pp": float(np.quantile(differences, 0.025)),
        "ci95_high_pp": float(np.quantile(differences, 0.975)),
    }
