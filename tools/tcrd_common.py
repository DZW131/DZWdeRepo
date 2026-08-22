"""Shared, protocol-locked utilities for the TCRD-v0 matched utility gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset, Sampler
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from network.resnet38_cls_tcrd_gate import Net
from tool.GenDataset import Stage1_TrainDataset
from tool import torchutils


EXPECTED_A0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
FROZEN_A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
BRANCH_DIRS = {
    "C0": "C0_control", "D": "D_diffusion",
    "R": "R_reaction", "DR": "DR_full",
}
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
GROUP_MULTIPLIERS = (1.0, 2.0, 10.0, 20.0)
POLY_POWER = 0.9


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state(path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def dataset_fingerprint(dataset: Stage1_TrainDataset, root: str) -> str:
    root = os.path.abspath(root)
    lines = []
    for path, label in dataset.object:
        relative = os.path.relpath(os.path.abspath(path), root).replace("\\", "/")
        lines.append(relative + "\t" + "".join(str(int(value)) for value in label.tolist()))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def derive_tail_replay_lrs(
    base_lr: float,
    steps_per_epoch: int,
    original_epochs: int = 25,
    tail_start_epoch: int = 20,
    power: float = POLY_POWER,
) -> Tuple[float, List[float]]:
    original_max_step = steps_per_epoch * original_epochs
    tail_start_step = steps_per_epoch * tail_start_epoch
    multiplier = (1.0 - tail_start_step / original_max_step) ** power
    tail_base_lr = base_lr * multiplier
    return tail_base_lr, [tail_base_lr * value for value in GROUP_MULTIPLIERS]


def build_optimizer(model: Net, steps_per_epoch: int, epochs: int = 5):
    tail_base_lr, group_lrs = derive_tail_replay_lrs(0.01, steps_per_epoch)
    groups = model.get_parameter_groups()
    optimizer_groups = [
        {"params": groups[0], "lr": group_lrs[0], "weight_decay": 5.0e-4},
        {"params": groups[1], "lr": group_lrs[1], "weight_decay": 0.0},
        {"params": groups[2], "lr": group_lrs[2], "weight_decay": 5.0e-4},
        {"params": groups[3], "lr": group_lrs[3], "weight_decay": 0.0},
    ]
    optimizer = torchutils.PolyOptimizer(
        optimizer_groups,
        lr=tail_base_lr,
        weight_decay=5.0e-4,
        max_step=steps_per_epoch * epochs,
        lr_power=POLY_POWER,
    )
    return optimizer, tail_base_lr, group_lrs


def load_branch_model(branch: str, checkpoint: str, device="cpu"):
    if sha256_file(checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("Frozen A0 checkpoint SHA256 mismatch")
    model = Net(4, branch=branch)
    state = load_state(checkpoint)
    incompat = model.load_state_dict(state, strict=False)
    expected_missing = set()
    if model.tcrd is not None:
        expected_missing = {"tcrd." + name for name, _ in model.tcrd.named_parameters()}
    if set(incompat.missing_keys) != expected_missing or incompat.unexpected_keys:
        raise AssertionError(
            f"Unexpected load result for {branch}: missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    return model.to(device), incompat


class MatchedAugmentationDataset(Dataset):
    """BCSS image-level dataset whose random flips are keyed by schedule seeds."""

    def __init__(self, train_root: str, image_size: int = 224):
        self.base = Stage1_TrainDataset(
            data_path=train_root, dataset="bcss", img_size=image_size
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, request):
        index, augmentation_seed = int(request[0]), int(request[1])
        path, label = self.base.object[index]
        image = Image.open(path).convert("RGB")
        if image.size != (self.image_size, self.image_size):
            image = TF.resize(
                image, [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
        generator = random.Random(augmentation_seed)
        if generator.random() > 0.5:
            image = TF.hflip(image)
        if generator.random() > 0.5:
            image = TF.vflip(image)
        image = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        return Path(path).stem, image, label.clone()


class ScheduleBatchSampler(Sampler[List[Tuple[int, int]]]):
    def __init__(self, indices: np.ndarray, augmentation_seeds: np.ndarray, epoch: int):
        if indices.shape != augmentation_seeds.shape:
            raise ValueError("indices and augmentation seeds must have identical shapes")
        self.indices = indices
        self.augmentation_seeds = augmentation_seeds
        self.epoch = int(epoch)

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
        return self.indices.shape[1]


def load_schedule(path: str):
    with np.load(path, allow_pickle=False) as data:
        return {
            "indices": data["indices"].copy(),
            "augmentation_seeds": data["augmentation_seeds"].copy(),
            "model_seeds": data["model_seeds"].copy(),
        }


def compact_scores(scores):
    return {
        "mIoU": float(scores["Mean IoU"]),
        "mDice": float(scores["Mean Dice"]),
        "class_iou": {str(i): float(scores["Class IoU"][i]) for i in range(4)},
        "class_dice": {
            str(i): float(scores["Dice Coefficients"][i]) for i in range(4)
        },
        "pixel_accuracy": float(scores["Pixel Accuracy"]),
        "mean_accuracy": float(scores["Mean Accuracy"]),
        "frequency_weighted_iou": float(scores["Frequency Weighted IoU"]),
    }
