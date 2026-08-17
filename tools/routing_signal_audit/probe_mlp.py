"""Fixed one-hidden-layer MLP relative-utility probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

from tools.routing_signal_audit import (
    FOLD_SEED,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_DIM,
    MLP_LR,
    MLP_WEIGHT_DECAY,
)
from tools.routing_signal_audit.signal_feature import prepare_fold_signal_set


class TinyRelativeUtilityMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(MLP_HIDDEN_DIM, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def run_mlp_probe(
    signal_set: str,
    cache_dir: Path,
    utilities: np.ndarray,
    fold_by_index: np.ndarray,
    output_path: Path,
    device: str = "cuda",
) -> dict:
    true_relative = utilities[:, 1:] - utilities[:, [0]]
    predicted = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=true_relative.shape,
    )
    assignment = np.zeros(len(utilities), dtype=np.uint8)
    fit_rows = []
    training_rows = []
    feature_names = None
    for fold in range(5):
        train = np.flatnonzero(fold_by_index != fold)
        heldout = np.flatnonzero(fold_by_index == fold)
        train_features, heldout_features, names, pca_rows = prepare_fold_signal_set(
            signal_set, cache_dir, train, heldout
        )
        feature_names = names
        train_flat = train_features.reshape(-1, train_features.shape[-1])
        heldout_flat = heldout_features.reshape(-1, heldout_features.shape[-1])
        target_flat = true_relative[train].reshape(-1).astype(np.float32)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_flat).astype(np.float32)
        heldout_scaled = scaler.transform(heldout_flat).astype(np.float32)
        torch.manual_seed(FOLD_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(FOLD_SEED)
        model = TinyRelativeUtilityMLP(train_scaled.shape[1]).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=MLP_LR, weight_decay=MLP_WEIGHT_DECAY
        )
        features_tensor = torch.from_numpy(train_scaled).to(device)
        target_tensor = torch.from_numpy(target_flat).to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(FOLD_SEED)
        model.train()
        for epoch in range(1, MLP_EPOCHS + 1):
            order = torch.randperm(
                len(features_tensor), device=device, generator=generator
            )
            epoch_loss = 0.0
            seen = 0
            for start in range(0, len(order), MLP_BATCH_SIZE):
                indices = order[start : start + MLP_BATCH_SIZE]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(features_tensor[indices])
                loss = torch.mean((prediction - target_tensor[indices]) ** 2)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite MLP loss: set={signal_set}, fold={fold}, epoch={epoch}"
                    )
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(indices)
                seen += len(indices)
            if epoch == 1 or epoch % 20 == 0 or epoch == MLP_EPOCHS:
                training_rows.append(
                    {
                        "probe": f"MLP-{signal_set.upper()}",
                        "fold": fold,
                        "epoch": epoch,
                        "mse": epoch_loss / max(seen, 1),
                    }
                )
        model.eval()
        with torch.no_grad():
            heldout_prediction = []
            for start in range(0, len(heldout_scaled), MLP_BATCH_SIZE):
                batch = torch.from_numpy(
                    heldout_scaled[start : start + MLP_BATCH_SIZE]
                ).to(device)
                heldout_prediction.append(model(batch).cpu().numpy())
        fold_prediction = np.concatenate(heldout_prediction).reshape(len(heldout), 4)
        predicted[heldout] = fold_prediction.astype(np.float32)
        assignment[heldout] += 1
        fit_rows.append(
            {
                "probe": f"MLP-{signal_set.upper()}",
                "fold": fold,
                "train_images": len(train),
                "heldout_images": len(heldout),
                "scaler_fit_candidates": len(train_flat),
                "scaler_fit_scope": "train_fold_only",
                "hidden_dim": MLP_HIDDEN_DIM,
                "epochs": MLP_EPOCHS,
                "learning_rate": MLP_LR,
                "weight_decay": MLP_WEIGHT_DECAY,
                "batch_size": MLP_BATCH_SIZE,
                "seed": FOLD_SEED,
                "heldout_used_for_fit": False,
                "heldout_gt_used_for_fit": False,
            }
        )
        for row in pca_rows:
            row.update({"probe": f"MLP-{signal_set.upper()}", "fold": fold})
            fit_rows.append(row)
    predicted.flush()
    if not np.all(assignment == 1) or not np.isfinite(predicted).all():
        raise RuntimeError("MLP probe OOF assignment/output contract failed")
    return {
        "predicted_relative": predicted,
        "fit_rows": fit_rows,
        "training_rows": training_rows,
        "feature_names": feature_names,
        "assignment_min": int(assignment.min()),
        "assignment_max": int(assignment.max()),
    }
