"""Fold-safe frozen semantic feature context and PCA composition."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from tools.routing_signal_audit import FOLD_SEED, PCA_DIMENSIONS
from tools.routing_signal_audit.signal_tta import STAGE_NAMES


def load_base_signal_sets(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    cache_dir = Path(cache_dir)
    signal_a = np.load(cache_dir / "cam_signal_features.npy", mmap_mode="r")
    tta_increment = np.load(cache_dir / "tta_signal_features.npy", mmap_mode="r")
    names_a = json.loads(
        (cache_dir / "cam_signal_features.names.json").read_text(encoding="utf-8")
    )
    names_tta = json.loads(
        (cache_dir / "tta_signal_features.names.json").read_text(encoding="utf-8")
    )
    signal_b = np.concatenate([signal_a, tta_increment], axis=2)
    names_b = names_a + names_tta
    return signal_a, signal_b, names_a, names_b


def load_feature_context(cache_dir: Path):
    cache_dir = Path(cache_dir)
    gaps = {
        stage: np.load(cache_dir / "gap_features" / f"{stage}.npy", mmap_mode="r")
        for stage in STAGE_NAMES
    }
    scalar = np.load(cache_dir / "feature_scalar_context.npy", mmap_mode="r")
    scalar_names = json.loads(
        (cache_dir / "feature_scalar_context.names.json").read_text(
            encoding="utf-8"
        )
    )
    return gaps, scalar, scalar_names


def fit_fold_pca_context(
    gaps: dict[str, np.ndarray],
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    train_parts = []
    heldout_parts = []
    names = []
    rows = []
    for stage in STAGE_NAMES:
        train_values = np.asarray(gaps[stage][train_indices], dtype=np.float32)
        heldout_values = np.asarray(gaps[stage][heldout_indices], dtype=np.float32)
        components = min(PCA_DIMENSIONS, train_values.shape[0], train_values.shape[1])
        pca = PCA(
            n_components=components,
            svd_solver="randomized",
            random_state=FOLD_SEED,
        )
        train_transformed = pca.fit_transform(train_values).astype(np.float32)
        heldout_transformed = pca.transform(heldout_values).astype(np.float32)
        effective = int(np.sum(pca.explained_variance_ > np.finfo(np.float32).eps))
        effective = max(effective, 1)
        train_transformed[:, effective:] = 0
        heldout_transformed[:, effective:] = 0
        if components < PCA_DIMENSIONS:
            train_transformed = np.pad(
                train_transformed, ((0, 0), (0, PCA_DIMENSIONS - components))
            )
            heldout_transformed = np.pad(
                heldout_transformed, ((0, 0), (0, PCA_DIMENSIONS - components))
            )
        train_parts.append(train_transformed)
        heldout_parts.append(heldout_transformed)
        names.extend(f"{stage}_pca_{index:02d}" for index in range(PCA_DIMENSIONS))
        rows.append(
            {
                "stage": stage,
                "fit_images": len(train_indices),
                "heldout_images": len(heldout_indices),
                "requested_components": PCA_DIMENSIONS,
                "effective_rank": effective,
                "explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_[:effective].sum()
                ),
                "fit_scope": "train_fold_only",
            }
        )
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(heldout_parts, axis=1),
        names,
        rows,
    )


def compose_signal_c(
    signal_b: np.ndarray,
    feature_scalar: np.ndarray,
    pca_context: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    common = np.concatenate(
        [
            np.asarray(feature_scalar[indices], dtype=np.float32),
            np.asarray(pca_context, dtype=np.float32),
        ],
        axis=1,
    )
    repeated = np.repeat(common[:, None, :], 4, axis=1)
    return np.concatenate(
        [np.asarray(signal_b[indices], dtype=np.float32), repeated], axis=2
    )


def build_oof_pca_context(
    cache_dir: Path,
    fold_by_index: np.ndarray,
    output_path: Path,
) -> tuple[np.ndarray, list[str], list[dict]]:
    gaps, _, _ = load_feature_context(cache_dir)
    context = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(fold_by_index), 4 * PCA_DIMENSIONS),
    )
    all_rows = []
    names = None
    for fold in range(5):
        heldout = np.flatnonzero(fold_by_index == fold)
        train = np.flatnonzero(fold_by_index != fold)
        _, heldout_context, fold_names, rows = fit_fold_pca_context(
            gaps, train, heldout
        )
        context[heldout] = heldout_context
        names = fold_names
        for row in rows:
            row["fold"] = fold
            all_rows.append(row)
    context.flush()
    if not np.isfinite(context).all():
        raise RuntimeError("OOF PCA context contains non-finite values")
    return context, names, all_rows


def prepare_fold_signal_set(
    signal_set: str,
    cache_dir: Path,
    train_indices: np.ndarray,
    heldout_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    signal_a, signal_b, names_a, names_b = load_base_signal_sets(cache_dir)
    signal_set = signal_set.upper()
    if signal_set == "A":
        return (
            np.asarray(signal_a[train_indices], dtype=np.float32),
            np.asarray(signal_a[heldout_indices], dtype=np.float32),
            names_a,
            [],
        )
    if signal_set == "B":
        return (
            np.asarray(signal_b[train_indices], dtype=np.float32),
            np.asarray(signal_b[heldout_indices], dtype=np.float32),
            names_b,
            [],
        )
    if signal_set != "C":
        raise ValueError(f"Unknown preregistered signal set: {signal_set}")
    gaps, scalar, scalar_names = load_feature_context(cache_dir)
    train_pca, heldout_pca, pca_names, pca_rows = fit_fold_pca_context(
        gaps, train_indices, heldout_indices
    )
    train = compose_signal_c(
        signal_b, scalar, train_pca, train_indices
    )
    heldout = compose_signal_c(
        signal_b, scalar, heldout_pca, heldout_indices
    )
    return train, heldout, names_b + scalar_names + pca_names, pca_rows
