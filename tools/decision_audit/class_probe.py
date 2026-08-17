"""The frozen 16-scalar diagnostic class-conditioned linear probe."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.decision_audit import BRANCH_NAMES, FOLD_SEED, OFFICIAL_FUSION
from tools.decision_audit.fusion import prediction_from_scores, score_predictions


PROBE_LR = 0.05
PROBE_STEPS = 500
PROBE_BATCH_SIZE = 16
PROBE_FOLDS = 5
PROBE_WEIGHT_DECAY = 0.0
PROBE_EPSILON = 1e-4


class ClassConditionedLinearProbe(nn.Module):
    """Four scale logits per class: exactly 4x4 trainable scalars."""

    def __init__(self):
        super().__init__()
        initial = np.asarray(OFFICIAL_FUSION, dtype=np.float64)
        initial[0] = PROBE_EPSILON
        initial /= initial.sum()
        logits = np.log(initial)[:, None].repeat(4, axis=1)
        self.scale_class_logits = nn.Parameter(
            torch.tensor(logits, dtype=torch.float32)
        )

    def weights(self):
        return torch.softmax(self.scale_class_logits, dim=0)

    def forward(self, cams):
        return torch.einsum("sc,bschw->bchw", self.weights(), cams)


def _load_names(cache_dir: Path):
    names = (cache_dir / "image_paths.txt").read_text(encoding="utf-8").splitlines()
    groups = (cache_dir / "source_groups.txt").read_text(encoding="utf-8").splitlines()
    if len(names) != len(groups):
        raise RuntimeError("Image/source-group manifests have different lengths")
    return names, np.asarray(groups)


def _next_batch(train_indices, rng, state, batch_size):
    order, position = state
    if position + batch_size > len(order):
        order = rng.permutation(train_indices)
        position = 0
    batch = order[position : position + batch_size]
    return batch, (order, position + batch_size)


def _cam_batch(cam_arrays, indices):
    return np.stack([array[indices] for array in cam_arrays], axis=1).astype(
        np.float32, copy=False
    )


def run_class_probe(
    cache_dir: Path,
    output_dir: Path,
    device: str = "cuda",
) -> dict:
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    cam_arrays = [
        np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in BRANCH_NAMES
    ]
    ground_truth = np.load(cache_dir / "gt.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    official_predictions = np.load(
        cache_dir / "official_predictions.npy", mmap_mode="r"
    )
    names, groups = _load_names(cache_dir)
    indices = np.arange(len(ground_truth))
    splitter = GroupKFold(n_splits=PROBE_FOLDS)
    splits = list(splitter.split(indices, groups=groups))
    oof_predictions = np.lib.format.open_memmap(
        output_dir / "class_probe_oof_predictions.npy",
        mode="w+",
        dtype=np.uint8,
        shape=ground_truth.shape,
    )
    assignment_count = np.zeros(len(ground_truth), dtype=np.uint8)
    fold_rows = []
    weight_rows = []
    training_rows = []
    assignment_rows = []

    for fold, (train_indices, heldout_indices) in enumerate(splits):
        train_groups = set(groups[train_indices])
        heldout_groups = set(groups[heldout_indices])
        overlap = train_groups & heldout_groups
        if overlap:
            raise RuntimeError(f"Fold {fold} has source leakage: {sorted(overlap)}")
        torch.manual_seed(FOLD_SEED + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FOLD_SEED + fold)
        probe = ClassConditionedLinearProbe().to(device)
        if sum(parameter.numel() for parameter in probe.parameters()) != 16:
            raise RuntimeError("Class probe must have exactly 16 trainable scalars")
        optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=PROBE_LR,
            weight_decay=PROBE_WEIGHT_DECAY,
        )
        rng = np.random.default_rng(FOLD_SEED + fold)
        state = (rng.permutation(train_indices), 0)
        probe.train()
        for step in range(1, PROBE_STEPS + 1):
            batch_indices, state = _next_batch(
                train_indices, rng, state, PROBE_BATCH_SIZE
            )
            cams = torch.from_numpy(_cam_batch(cam_arrays, batch_indices)).to(device)
            target = torch.from_numpy(
                np.asarray(ground_truth[batch_indices], dtype=np.int64)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = probe(cams)
            loss = F.cross_entropy(logits, target, ignore_index=4)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite probe loss at fold={fold}, step={step}; STOP FOR REVIEW"
                )
            loss.backward()
            optimizer.step()
            if step == 1 or step % 50 == 0 or step == PROBE_STEPS:
                training_rows.append(
                    {"fold": fold, "step": step, "loss": float(loss.item())}
                )

        probe.eval()
        learned_weights = probe.weights().detach().cpu().numpy()
        for class_id in range(4):
            for branch_index, branch_name in enumerate(BRANCH_NAMES):
                weight_rows.append(
                    {
                        "fold": fold,
                        "class_id": class_id,
                        "branch": branch_name,
                        "weight": float(learned_weights[branch_index, class_id]),
                    }
                )
        with torch.no_grad():
            for start in range(0, len(heldout_indices), PROBE_BATCH_SIZE):
                heldout_batch = heldout_indices[start : start + PROBE_BATCH_SIZE]
                cams = torch.from_numpy(_cam_batch(cam_arrays, heldout_batch)).to(device)
                scores = probe(cams).cpu().numpy()
                batch_prediction = prediction_from_scores(
                    scores,
                    np.asarray(presence[heldout_batch], dtype=np.float32),
                )
                oof_predictions[heldout_batch] = batch_prediction
                assignment_count[heldout_batch] += 1

        probe_score = score_predictions(
            ground_truth[heldout_indices], oof_predictions[heldout_indices]
        )
        official_score = score_predictions(
            ground_truth[heldout_indices], official_predictions[heldout_indices]
        )
        fold_rows.append(
            {
                "fold": fold,
                "train_images": len(train_indices),
                "heldout_images": len(heldout_indices),
                "train_groups": len(train_groups),
                "heldout_groups": len(heldout_groups),
                "group_overlap": 0,
                "official_mIoU": 100 * official_score["Mean IoU"],
                "probe_mIoU": 100 * probe_score["Mean IoU"],
                "delta_mIoU": 100
                * (probe_score["Mean IoU"] - official_score["Mean IoU"]),
                "official_mDice": 100 * official_score["Mean Dice"],
                "probe_mDice": 100 * probe_score["Mean Dice"],
            }
        )
        heldout_set = set(int(index) for index in heldout_indices)
        if heldout_set & set(int(index) for index in train_indices):
            raise RuntimeError("Held-out images participated in probe fitting")
        for index in heldout_indices:
            assignment_rows.append(
                {
                    "index": int(index),
                    "image_name": names[index],
                    "source_group": groups[index],
                    "fold": fold,
                }
            )

    oof_predictions.flush()
    if not np.all(assignment_count == 1):
        raise RuntimeError("Every validation image must have exactly one OOF prediction")
    oof_score = score_predictions(ground_truth, oof_predictions)
    official_score = score_predictions(ground_truth, official_predictions)
    summary = {
        "fold_method": "GroupKFold",
        "fold_seed": FOLD_SEED,
        "num_groups": len(set(groups)),
        "num_folds": PROBE_FOLDS,
        "trainable_scalars": 16,
        "optimizer": "Adam",
        "learning_rate": PROBE_LR,
        "steps": PROBE_STEPS,
        "batch_size": PROBE_BATCH_SIZE,
        "weight_decay": PROBE_WEIGHT_DECAY,
        "official_mIoU": 100 * official_score["Mean IoU"],
        "official_mDice": 100 * official_score["Mean Dice"],
        "oof_mIoU": 100 * oof_score["Mean IoU"],
        "oof_mDice": 100 * oof_score["Mean Dice"],
        "delta_mIoU": 100 * (oof_score["Mean IoU"] - official_score["Mean IoU"]),
        "delta_mDice": 100
        * (oof_score["Mean Dice"] - official_score["Mean Dice"]),
        "oof_assignment_min": int(assignment_count.min()),
        "oof_assignment_max": int(assignment_count.max()),
        "test_evaluated": False,
    }
    summary.update(
        {
            f"class{class_id}_iou": 100 * oof_score["Class IoU"][class_id]
            for class_id in range(4)
        }
    )
    return {
        "summary": summary,
        "fold_rows": fold_rows,
        "weight_rows": weight_rows,
        "training_rows": training_rows,
        "assignment_rows": assignment_rows,
    }
