"""Frozen controls shared by the exploratory TCER-R fresh-25 experiment."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.backends import cudnn

from tool import torchutils


EXPERIMENT_NAME = "TCER_R_V0_BCSS_SEED42_25EP_EXPLORATORY"
EXPECTED_PRETRAINED_SHA256 = (
    "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
)
EXPECTED_A0_SHA256 = (
    "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
)
EPOCHS = 25
BATCH_SIZE = 20
STEPS_PER_EPOCH = 1171
MAX_STEPS = EPOCHS * STEPS_PER_EPOCH
SEED = 42
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
A0_REFERENCE = {"mIoU": 0.673283, "mDice": 0.802683, "cam28_1_mIoU": 0.670272}


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def set_official_seed(seed: int = SEED) -> None:
    """Mirror the released SSHR seed/TF32/nondeterminism policy exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_official_optimizer(model):
    groups = model.get_parameter_groups()
    parameters = [
        {"params": groups[0], "lr": 0.01, "weight_decay": 5.0e-4},
        {"params": groups[1], "lr": 0.02, "weight_decay": 0.0},
        {"params": groups[2], "lr": 0.10, "weight_decay": 5.0e-4},
        {"params": groups[3], "lr": 0.20, "weight_decay": 0.0},
    ]
    optimizer = torchutils.PolyOptimizer(
        parameters, lr=0.01, weight_decay=5.0e-4,
        max_step=MAX_STEPS, lr_power=0.9,
    )
    return optimizer


def official_classification_loss(outputs, labels):
    losses = [
        F.multilabel_soft_margin_loss(output, labels) for output in outputs[:4]
    ]
    return sum(weight * loss for weight, loss in zip(LOSS_WEIGHTS, losses))


def compact_scores(scores):
    return {
        "mIoU": float(scores["Mean IoU"]),
        "mDice": float(scores["Mean Dice"]),
        "class_iou": {str(i): float(scores["Class IoU"][i]) for i in range(4)},
        "class_dice": {
            str(i): float(scores["Dice Coefficients"][i]) for i in range(4)
        },
    }


def exploratory_decision(final_delta_pp, cam_delta_pp, confusion_reduction):
    passed = (
        final_delta_pp >= 0.15
        and cam_delta_pp >= 0.20
        and confusion_reduction >= 0.005
    )
    return "TCER_R25_EXPLORATORY_GO" if passed else "TCER_R25_EXPLORATORY_CLOSE"
