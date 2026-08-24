"""Protocol-locked utilities shared by WD-CH audits and matched training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage
from torch.backends import cudnn
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode

from network import resnet38_cls
from network import resnet38_wdch
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
EXPECTED_A0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
EXPECTED_TRAIN = 23422
EXPECTED_VAL = 3418
N_CLASS = 4
BACKGROUND = 4
BCSS_THRESHOLDS = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
CAM_WEIGHTS = {"56": 0.0, "28_1": 0.6, "28_2": 0.2, "deep": 0.2}
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))
GROUP_MULTIPLIERS = (1.0, 2.0, 10.0, 20.0)
POLY_POWER = 0.9


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return float(item)
        return item

    path.write_text(
        json.dumps(convert(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_state(path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


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
        raise AssertionError("WD-CH utility gate is BCSS validation-only")
    image_paths = sorted((Path(root) / "img").glob("*.png"))
    mask_paths = sorted((Path(root) / "mask").glob("*.png"))
    if len(image_paths) != EXPECTED_VAL or len(mask_paths) != EXPECTED_VAL:
        raise AssertionError(
            f"Expected {EXPECTED_VAL} validation pairs, found "
            f"{len(image_paths)} images and {len(mask_paths)} masks"
        )
    if [p.stem for p in image_paths] != [p.stem for p in mask_paths]:
        raise AssertionError("Validation image/mask names differ")


def load_a0_model(checkpoint: str, cam: bool = True, device: str = "cuda"):
    if sha256_file(checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("Frozen A0 checkpoint SHA256 mismatch")
    model_class = resnet38_cls.Net_CAM if cam else resnet38_cls.Net
    model = model_class(N_CLASS)
    incompat = model.load_state_dict(read_state(checkpoint), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    model = model.to(device)
    return model


def load_wdch_from_a0(
    checkpoint: str,
    kernel_size: int,
    cam: bool = True,
    device: str = "cuda",
):
    if sha256_file(checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("Frozen A0 checkpoint SHA256 mismatch")
    model_class = resnet38_wdch.Net_CAM if cam else resnet38_wdch.Net
    model = model_class(N_CLASS, wdch_kernel_size=kernel_size)
    state = read_state(checkpoint)
    removed = state.pop("hfrm_28_1.context_conv.weight")
    incompat = model.load_state_dict(state, strict=False)
    expected = {
        "hfrm_28_1.wdch.haar.analysis_filters",
        "hfrm_28_1.wdch.haar.synthesis_filters",
        "hfrm_28_1.wdch.ll_context.weight",
    }
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise AssertionError(
            f"Unexpected WD-CH load result: missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    audit = {
        "expected_missing_keys": sorted(expected),
        "unexpected_keys": list(incompat.unexpected_keys),
        "replaced_a0_key": "hfrm_28_1.context_conv.weight",
        "replaced_a0_shape": list(removed.shape),
        "new_key": "hfrm_28_1.wdch.ll_context.weight",
        "new_shape": list(model.hfrm_28_1.wdch.ll_context.weight.shape),
        "initialization": f"uniform 1/{kernel_size**2}",
    }
    return model.to(device), audit


def dataset_fingerprint(dataset: Stage1_TrainDataset, root: str) -> str:
    root = os.path.abspath(root)
    rows = []
    for path, label in dataset.object:
        relative = os.path.relpath(os.path.abspath(path), root).replace("\\", "/")
        rows.append(relative + "\t" + "".join(str(int(v)) for v in label.tolist()))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


class MatchedAugmentationDataset(Dataset):
    """Released resize/flip/normalize algorithm with schedule-keyed randomness."""

    def __init__(self, train_root: str, image_size: int = 224) -> None:
        self.base = Stage1_TrainDataset(
            data_path=train_root, dataset="bcss", img_size=image_size
        )
        self.image_size = int(image_size)

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


class ScheduleBatchSampler(Sampler[List[Tuple[int, int]]]):
    def __init__(self, schedule: Mapping[str, np.ndarray], epoch: int):
        self.indices = schedule["indices"]
        self.augmentation_seeds = schedule["augmentation_seeds"]
        self.epoch = int(epoch)
        if self.indices.shape != self.augmentation_seeds.shape:
            raise ValueError("Schedule indices and augmentation seeds differ")

    def __iter__(self) -> Iterator[List[Tuple[int, int]]]:
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


def build_schedule(
    train_root: str,
    output: str,
    seed: int = 42,
    epochs: int = 25,
    batch_size: int = 20,
):
    dataset = Stage1_TrainDataset(train_root, dataset="bcss", img_size=224)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(f"Expected {EXPECTED_TRAIN} samples, found {len(dataset)}")
    steps = len(dataset) // batch_size
    generator = np.random.default_rng(seed)
    indices = np.empty((epochs, steps, batch_size), dtype=np.int32)
    augmentation_seeds = np.empty_like(indices, dtype=np.int64)
    model_seeds = np.empty((epochs, steps), dtype=np.int64)
    for epoch in range(epochs):
        permutation = generator.permutation(len(dataset))[: steps * batch_size]
        indices[epoch] = permutation.reshape(steps, batch_size)
        augmentation_seeds[epoch] = generator.integers(
            0,
            np.iinfo(np.int64).max,
            size=(steps, batch_size),
            dtype=np.int64,
        )
        model_seeds[epoch] = generator.integers(
            0, np.iinfo(np.int64).max, size=steps, dtype=np.int64
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(
        output,
        indices=indices,
        augmentation_seeds=augmentation_seeds,
        model_seeds=model_seeds,
    )
    metadata = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps,
        "dataset_samples": len(dataset),
        "scheduled_samples_per_epoch": steps * batch_size,
        "dropped_samples_per_epoch": len(dataset) - steps * batch_size,
        "dataset_order_sha256": dataset_fingerprint(dataset, train_root),
        "schedule_sha256": sha256_file(output),
        "worker_independent_released_flip_algorithm": True,
        "common_model_seed_per_step": True,
    }
    write_json(output.with_suffix(".json"), metadata)
    return metadata


def load_schedule(path: str):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


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


def named_optimizer_state(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        name = names[id(parameter)]
        state[name] = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in values.items()
        }
    groups = []
    for group in optimizer.param_groups:
        groups.append(
            {
                key: value
                for key, value in group.items()
                if key != "params"
            }
        )
    return {"state": state, "groups": groups}


def restore_named_optimizer_state(model, optimizer, saved):
    restored, skipped = [], []
    parameters = dict(model.named_parameters())
    for name, values in saved["state"].items():
        parameter = parameters.get(name)
        if parameter is None:
            skipped.append({"name": name, "reason": "parameter absent"})
            continue
        compatible = True
        for value in values.values():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                compatible = False
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


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


class OfficialMetricAccumulator:
    """Streaming equivalent of released ``iouutils.scores``."""

    def __init__(self):
        self.hist = np.zeros((5, 5), dtype=np.float64)
        self.images = 0

    def update(self, truth, prediction):
        truth = np.asarray(truth, dtype=np.int64)
        prediction = np.asarray(prediction, dtype=np.int64).copy()
        prediction[truth == BACKGROUND] = BACKGROUND
        valid = (truth >= 0) & (truth <= BACKGROUND)
        encoded = 5 * truth[valid] + prediction[valid]
        self.hist += np.bincount(encoded, minlength=25).reshape(5, 5)
        self.images += 1

    def result(self):
        hist = self.hist.copy()
        hist[4, 4] = 0.0
        diagonal = np.diag(hist)
        union = hist.sum(1) + hist.sum(0) - diagonal
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
            "images": int(self.images),
        }


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


def foreground_boundary_distance(truth):
    """Exact HMA-v0 boundary/interior definition."""

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


class PairedZoneAccumulator:
    def __init__(self):
        self.data = {
            "boundary_le_7": {"pixels": 0, "base_correct": 0, "candidate_correct": 0},
            "interior_ge_8": {"pixels": 0, "base_correct": 0, "candidate_correct": 0},
        }

    def update(self, truth, base, candidate):
        bins = foreground_boundary_distance(truth)
        for name, mask in bins.items():
            row = self.data[name]
            row["pixels"] += int(mask.sum())
            row["base_correct"] += int((mask & (base == truth)).sum())
            row["candidate_correct"] += int((mask & (candidate == truth)).sum())

    def result(self):
        result = {}
        for name, row in self.data.items():
            pixels = max(row["pixels"], 1)
            base = row["base_correct"] / pixels
            candidate = row["candidate_correct"] / pixels
            result[name] = {
                **row,
                "base_accuracy": base,
                "candidate_accuracy": candidate,
                "delta_pp": 100.0 * (candidate - base),
            }
        return result


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


class PairedComponentAccumulator:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.data = {}
        self.structure = np.ones((3, 3), dtype=np.uint8)

    def update(self, truth, base, candidate):
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
                row = self.data.setdefault(
                    (class_id, size),
                    {"components": 0, "pixels": 0, "base_correct": 0, "candidate_correct": 0},
                )
                row["components"] += 1
                row["pixels"] += area
                row["base_correct"] += int((base[mask] == class_id).sum())
                row["candidate_correct"] += int((candidate[mask] == class_id).sum())

    def result(self):
        output = []
        for class_id in range(4):
            for size in ("small", "medium", "large"):
                row = self.data.get(
                    (class_id, size),
                    {"components": 0, "pixels": 0, "base_correct": 0, "candidate_correct": 0},
                )
                pixels = max(row["pixels"], 1)
                base = row["base_correct"] / pixels
                candidate = row["candidate_correct"] / pixels
                output.append(
                    {
                        "class": class_id,
                        "size": size,
                        **self.thresholds[class_id],
                        **row,
                        "base_recall": base,
                        "candidate_recall": candidate,
                        "delta_pp": 100.0 * (candidate - base),
                    }
                )
        return output
