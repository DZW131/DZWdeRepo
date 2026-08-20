"""Frozen slide-held-out linear probes and CRRA-v0 decision logic."""

from __future__ import annotations

import inspect
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tools.crra_v0 import (
    BOOTSTRAP_SAMPLES,
    FOREGROUND_LABELS,
    N_LABELS,
    N_SPLITS,
    SEED,
)


def foreground_macro_f1(labels, predictions) -> float:
    return float(
        f1_score(
            labels,
            predictions,
            labels=FOREGROUND_LABELS,
            average="macro",
            zero_division=0,
        )
    )


def classification_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": foreground_macro_f1(labels, predictions),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=range(N_LABELS)
        ).astype(int).tolist(),
    }
    per_class = f1_score(
        labels,
        predictions,
        labels=range(N_LABELS),
        average=None,
        zero_division=0,
    )
    result["per_class_f1"] = {
        str(index): float(value) for index, value in enumerate(per_class)
    }
    return result


def make_fold_assignments(labels, groups):
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    folds = np.full(labels.shape, -1, dtype=np.int64)
    splitter = GroupKFold(n_splits=N_SPLITS)
    dummy = np.zeros((len(labels), 1), dtype=np.float32)
    fold_manifest = []
    for fold, (train, held_out) in enumerate(splitter.split(dummy, labels, groups)):
        train_groups = set(groups[train].tolist())
        held_out_groups = set(groups[held_out].tolist())
        if train_groups & held_out_groups:
            raise AssertionError("Slide leakage detected")
        folds[held_out] = fold
        fold_manifest.append({
            "fold": int(fold),
            "train_slides": sorted(train_groups),
            "held_out_slides": sorted(held_out_groups),
            "n_train": int(len(train)),
            "n_held_out": int(len(held_out)),
        })
    if np.any(folds < 0):
        raise AssertionError("Incomplete GroupKFold assignment")
    return folds, fold_manifest


def run_oof_probe(features, labels, groups, fold_ids):
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    fold_ids = np.asarray(fold_ids, dtype=np.int64)
    if not np.isfinite(features).all():
        raise ValueError("Probe features contain non-finite values")
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    fold_results = []
    for fold in range(N_SPLITS):
        train = np.flatnonzero(fold_ids != fold)
        held_out = np.flatnonzero(fold_ids == fold)
        train_groups = set(groups[train].tolist())
        held_out_groups = set(groups[held_out].tolist())
        if train_groups & held_out_groups:
            raise AssertionError("Slide leakage detected during probing")
        logistic_kwargs = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 5000,
            "class_weight": "balanced",
            "random_state": SEED,
        }
        # Older sklearn requires this explicit flag; newer sklearn makes every
        # >=3-class lbfgs fit multinomial and removes the parameter.
        if "multi_class" in inspect.signature(LogisticRegression).parameters:
            logistic_kwargs["multi_class"] = "multinomial"
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(**logistic_kwargs),
        )
        pipeline.fit(features[train], labels[train])
        predictions[held_out] = pipeline.predict(features[held_out])
        fold_results.append({
            "fold": int(fold),
            "train_slides": sorted(train_groups),
            "held_out_slides": sorted(held_out_groups),
            "n_train": int(len(train)),
            "n_held_out": int(len(held_out)),
            **classification_metrics(labels[held_out], predictions[held_out]),
        })
    if np.any(predictions < 0):
        raise AssertionError("Incomplete OOF predictions")
    return predictions, classification_metrics(labels, predictions), fold_results


def subset_accuracy(labels, predictions, mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(accuracy_score(np.asarray(labels)[mask], np.asarray(predictions)[mask]))


def _macro_f1_from_confusion(matrix: np.ndarray) -> float:
    values = []
    for class_index in FOREGROUND_LABELS:
        tp = float(matrix[class_index, class_index])
        fp = float(matrix[:, class_index].sum() - tp)
        fn = float(matrix[class_index, :].sum() - tp)
        denominator = 2.0 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
    return float(np.mean(values))


def slide_bootstrap(labels, predictions_by_representation, groups):
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    slides = np.asarray(sorted(set(groups.tolist())))
    matrices = {}
    for name, predictions in predictions_by_representation.items():
        predictions = np.asarray(predictions, dtype=np.int64)
        matrices[name] = np.stack([
            confusion_matrix(
                labels[groups == slide], predictions[groups == slide], labels=range(N_LABELS)
            )
            for slide in slides
        ])
    rng = np.random.default_rng(SEED)
    deltas = {"core_minus_whole": [], "core_rim_minus_whole": []}
    for _ in range(BOOTSTRAP_SAMPLES):
        draw = rng.integers(0, len(slides), size=len(slides))
        scores = {
            name: _macro_f1_from_confusion(values[draw].sum(axis=0))
            for name, values in matrices.items()
        }
        deltas["core_minus_whole"].append(scores["core"] - scores["whole"])
        deltas["core_rim_minus_whole"].append(scores["core_rim"] - scores["whole"])
    return {
        name: {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for name, values in deltas.items()
    }


def discrepancy_rank_test(type_a, type_b):
    type_a = np.asarray(type_a, dtype=np.float64)
    type_b = np.asarray(type_b, dtype=np.float64)
    type_a = type_a[np.isfinite(type_a)]
    type_b = type_b[np.isfinite(type_b)]
    if not len(type_a) or not len(type_b):
        return {
            "u": float("nan"),
            "p_two_sided": float("nan"),
            "rank_biserial_type_b_vs_type_a": float("nan"),
        }
    test = mannwhitneyu(type_b, type_a, alternative="two-sided")
    rank_biserial = 2.0 * float(test.statistic) / (len(type_b) * len(type_a)) - 1.0
    return {
        "u": float(test.statistic),
        "p_two_sided": float(test.pvalue),
        "rank_biserial_type_b_vs_type_a": float(rank_biserial),
        "n_type_a": int(len(type_a)),
        "n_type_b": int(len(type_b)),
    }


def select_candidate(metrics):
    return "core" if metrics["core"]["macro_f1"] >= metrics["core_rim"]["macro_f1"] else "core_rim"


def decide_crra(metrics, type_b_accuracy, type_a_accuracy, fold_results, coverage, bootstrap):
    best = select_candidate(metrics)
    whole_f1 = metrics["whole"]["macro_f1"]
    delta = metrics[best]["macro_f1"] - whole_f1
    delta_b = type_b_accuracy[best] - type_b_accuracy["whole"]
    type_a_drop = type_a_accuracy["whole"] - type_a_accuracy[best]
    fold_delta = [
        fold_results[best][index]["macro_f1"] - fold_results["whole"][index]["macro_f1"]
        for index in range(N_SPLITS)
    ]
    positive_folds = int(np.sum(np.asarray(fold_delta) > 0.0))
    class_delta = [
        metrics[best]["per_class_f1"][str(index)]
        - metrics["whole"]["per_class_f1"][str(index)]
        for index in FOREGROUND_LABELS
    ]
    nonnegative_classes = int(np.sum(np.asarray(class_delta) >= 0.0))
    positive_classes = int(np.sum(np.asarray(class_delta) > 0.0))

    strong = (
        delta >= 0.05 and delta_b >= 0.08 and positive_folds >= 4
        and nonnegative_classes >= 3 and coverage >= 0.70 and type_a_drop <= 0.02
    )
    go = (
        delta >= 0.03 and delta_b >= 0.05 and positive_folds >= 4
        and positive_classes >= 2 and coverage >= 0.70 and type_a_drop <= 0.02
    )
    both_worse = (
        metrics["core"]["macro_f1"] < whole_f1
        and metrics["core_rim"]["macro_f1"] < whole_f1
    )
    bootstrap_key = f"{best}_minus_whole"
    interval = bootstrap[bootstrap_key]
    bootstrap_includes_zero = interval["ci95_low"] <= 0.0 <= interval["ci95_high"]
    core_delta = metrics["core"]["macro_f1"] - whole_f1
    dual_delta = metrics["core_rim"]["macro_f1"] - whole_f1
    candidates_conflict = core_delta * dual_delta < 0.0
    review_barrier = (
        bootstrap_includes_zero or candidates_conflict
        or coverage < 0.70 or type_a_drop > 0.02
    )
    hard_nogo = delta < 0.01 or delta_b <= 0.0 or positive_folds <= 2 or both_worse
    if strong and not review_barrier:
        decision = "CRRA_V0_STRONG_GO"
    elif go and not review_barrier:
        decision = "CRRA_V0_GO"
    elif hard_nogo:
        decision = "CRRA_V0_NOGO"
    else:
        decision = "CRRA_V0_REVIEW"

    core_gain = metrics["core"]["macro_f1"] - whole_f1
    dual_over_core = metrics["core_rim"]["macro_f1"] - metrics["core"]["macro_f1"]
    dual_b_over_core = type_b_accuracy["core_rim"] - type_b_accuracy["core"]
    if decision == "CRRA_V0_NOGO":
        flag = "REGION_REPRESENTATION_ROUTE_CLOSED"
    elif core_gain >= 0.03 and dual_over_core < 0.01:
        flag = "CORE_ONLY_PREFERRED"
    elif dual_over_core >= 0.02 and dual_b_over_core >= 0.03:
        flag = "DUAL_TOKEN_PREFERRED"
    else:
        flag = "CORE_SIGNAL_WEAK"

    return {
        "decision": decision,
        "representation_flag": flag,
        "best_candidate": best,
        "best_delta_macro_f1": float(delta),
        "best_delta_type_b_accuracy": float(delta_b),
        "best_type_a_drop": float(type_a_drop),
        "positive_folds": positive_folds,
        "fold_deltas": [float(item) for item in fold_delta],
        "nonnegative_classes": nonnegative_classes,
        "positive_classes": positive_classes,
        "per_class_deltas": [float(item) for item in class_delta],
        "common_support_fraction": float(coverage),
        "hard_nogo_conditions": {
            "best_delta_below_0.01": bool(delta < 0.01),
            "type_b_does_not_improve": bool(delta_b <= 0.0),
            "at_most_2_positive_folds": bool(positive_folds <= 2),
            "both_candidates_worse_than_whole": bool(both_worse),
        },
        "review_conditions": {
            "best_bootstrap_ci_includes_zero": bool(bootstrap_includes_zero),
            "core_and_dual_sign_conflict": bool(candidates_conflict),
            "common_support_below_0.70": bool(coverage < 0.70),
            "type_a_drop_above_0.02": bool(type_a_drop > 0.02),
        },
    }
