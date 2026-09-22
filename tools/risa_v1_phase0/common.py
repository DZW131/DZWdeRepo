from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
REFERENCE_MIOU = 0.6557244403737567
REFERENCE_MDICE = 0.7896084511076717
REFERENCE_CLASS_IOU = {
    "0": 0.7556121475562709,
    "1": 0.6919999153520463,
    "2": 0.5581297246446123,
    "3": 0.617155973942097,
}
EPOCHS = 5
MICRO_BATCH = 5
ACCUMULATION = 4
STEPS_PER_EPOCH = 1171
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH
CLASSES = ("Tumor", "Stroma", "Inflammation", "Necrosis")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positive, negative = int(target.sum()), int((~target).sum())
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(score):
        end = start + 1
        while end < len(score) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = .5 * (start + end - 1) + 1.
        start = end
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))
