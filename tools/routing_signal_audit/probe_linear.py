"""Fixed train-fold-scaled Ridge relative-utility probes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from tools.routing_signal_audit import RIDGE_ALPHA
from tools.routing_signal_audit.signal_feature import prepare_fold_signal_set


def run_linear_probe(
    signal_set: str,
    cache_dir: Path,
    utilities: np.ndarray,
    fold_by_index: np.ndarray,
    output_path: Path,
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
    coefficient_rows = []
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
        target_flat = true_relative[train].reshape(-1)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_flat)
        scaled_heldout = scaler.transform(heldout_flat)
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(scaled_train, target_flat)
        fold_prediction = model.predict(scaled_heldout).reshape(len(heldout), 4)
        predicted[heldout] = fold_prediction.astype(np.float32)
        assignment[heldout] += 1
        fit_rows.append(
            {
                "probe": f"Linear-{signal_set.upper()}",
                "fold": fold,
                "train_images": len(train),
                "heldout_images": len(heldout),
                "scaler_fit_candidates": len(train_flat),
                "scaler_fit_scope": "train_fold_only",
                "ridge_alpha": RIDGE_ALPHA,
                "heldout_used_for_fit": False,
                "heldout_gt_used_for_fit": False,
            }
        )
        for row in pca_rows:
            row.update({"probe": f"Linear-{signal_set.upper()}", "fold": fold})
            fit_rows.append(row)
        for feature_name, coefficient in zip(names, model.coef_):
            coefficient_rows.append(
                {
                    "probe": f"Linear-{signal_set.upper()}",
                    "fold": fold,
                    "feature": feature_name,
                    "coefficient": float(coefficient),
                }
            )
    predicted.flush()
    if not np.all(assignment == 1) or not np.isfinite(predicted).all():
        raise RuntimeError("Linear probe OOF assignment/output contract failed")
    return {
        "predicted_relative": predicted,
        "fit_rows": fit_rows,
        "coefficient_rows": coefficient_rows,
        "feature_names": feature_names,
        "assignment_min": int(assignment.min()),
        "assignment_max": int(assignment.max()),
    }

