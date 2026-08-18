"""Frozen GroupKFold probes and diagnostic metrics for Phase-0R."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    davies_bouldin_score,
    f1_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tool import iouutils
from tools.region_phase0r import MAX_CLUSTER_SAMPLE, N_LABELS, N_SPLITS, SEED


def official_scores(ground_truth, prediction):
    """Run the released mutating metric without changing caller-owned arrays."""
    return iouutils.scores(
        [item.copy() for item in ground_truth],
        [item.copy() for item in prediction],
        n_class=4,
    )


def classification_metrics(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=range(N_LABELS), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        **{
            f"f1_class_{index}": float(value)
            for index, value in enumerate(
                f1_score(labels, predictions, labels=range(N_LABELS), average=None, zero_division=0)
            )
        },
    }


def run_oof_probe(features, labels, groups):
    """Exactly specified slide-held-out multinomial logistic probe."""
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    splitter = GroupKFold(n_splits=N_SPLITS)
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    fold_ids = np.full(labels.shape, -1, dtype=np.int64)
    folds = []
    for fold, (train, held_out) in enumerate(splitter.split(features, labels, groups)):
        train_groups = set(groups[train].tolist())
        held_out_groups = set(groups[held_out].tolist())
        if train_groups & held_out_groups:
            raise AssertionError("Slide leakage detected in GroupKFold")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
                multi_class="multinomial",
            ),
        )
        model.fit(features[train], labels[train])
        predictions[held_out] = model.predict(features[held_out])
        fold_ids[held_out] = fold
        folds.append(
            {
                "fold": fold,
                "train_slides": sorted(train_groups),
                "held_out_slides": sorted(held_out_groups),
                "n_train": int(len(train)),
                "n_held_out": int(len(held_out)),
                **classification_metrics(labels[held_out], predictions[held_out]),
            }
        )
    if np.any(predictions < 0) or np.any(fold_ids < 0):
        raise AssertionError("Incomplete out-of-fold prediction coverage")
    return predictions, fold_ids, classification_metrics(labels, predictions), folds


def representation_cluster_metrics(features, labels):
    """Deterministic label-geometry diagnostics with bounded silhouette cost."""
    features = StandardScaler().fit_transform(np.asarray(features, dtype=np.float32))
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(SEED)
    if len(labels) > MAX_CLUSTER_SAMPLE:
        indices = np.sort(rng.choice(len(labels), MAX_CLUSTER_SAMPLE, replace=False))
    else:
        indices = np.arange(len(labels))
    sampled_x, sampled_y = features[indices], labels[indices]
    means = []
    within_sum = 0.0
    for class_index in np.unique(labels):
        subset = features[labels == class_index]
        mean = subset.mean(0)
        means.append(mean)
        within_sum += float(np.square(subset - mean).sum())
    means = np.stack(means)
    global_mean = features.mean(0)
    between_sum = 0.0
    for class_index, mean in zip(np.unique(labels), means):
        between_sum += int(np.sum(labels == class_index)) * float(np.square(mean - global_mean).sum())
    return {
        "n": int(len(labels)),
        "silhouette_sample_n": int(len(indices)),
        "silhouette": float(silhouette_score(sampled_x, sampled_y, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "between_within_scatter_ratio": float(between_sum / (within_sum + 1e-12)),
    }


def decide_phase0r(oracle_gain_pp, region_bbox_gain, region_centroid_gain,
                   region_seg_gain_pp, recovery_fraction, positive_folds,
                   mixed_error_fraction, geometry_adds_value):
    """Apply the preregistered decision thresholds without discretionary tuning."""
    values = [oracle_gain_pp, region_bbox_gain, region_centroid_gain,
              region_seg_gain_pp, recovery_fraction, mixed_error_fraction,
              float(geometry_adds_value)]
    if not np.all(np.isfinite(values)):
        raise ValueError("Decision inputs must be finite")
    if (oracle_gain_pp >= 2.0 and region_bbox_gain >= 0.03
            and region_centroid_gain >= 0.05 and region_seg_gain_pp >= 0.5
            and recovery_fraction >= 0.25 and positive_folds >= 4):
        return "REGION_REP_STRONG_GO"
    if (oracle_gain_pp >= 1.0 and region_bbox_gain >= 0.02
            and region_seg_gain_pp >= 0.3 and recovery_fraction >= 0.15
            and positive_folds >= 3):
        return "REGION_REP_GO"
    if oracle_gain_pp >= 1.0 and region_seg_gain_pp < 0.3 and geometry_adds_value:
        return "REGION_GEOMETRY_REVIEW"
    if oracle_gain_pp < 0.5 or (region_seg_gain_pp < 0.1 and region_bbox_gain < 0.01):
        return "REGION_REP_NOGO"
    if oracle_gain_pp < 1.0 and mixed_error_fraction > 0.5:
        return "REGION_SHAPE_BOTTLENECK"
    return "REGION_REP_REVIEW"
