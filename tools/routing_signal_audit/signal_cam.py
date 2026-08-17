"""GT-free aggregated-CAM routing signals (Signal Set A)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from tools.routing_signal_audit import BRANCH_NAMES


EPSILON = 1e-8


def _softmax(values: np.ndarray, axis: int = 0) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted).astype(np.float32, copy=False)
    return exponential / (np.sum(exponential, axis=axis, keepdims=True) + EPSILON)


def _stats(values: np.ndarray, quantiles=()) -> list[float]:
    output = [float(np.mean(values)), float(np.std(values))]
    output.extend(float(np.quantile(values, quantile)) for quantile in quantiles)
    return output


def _spatial_features(cam: np.ndarray) -> tuple[float, float, float]:
    flattened = np.maximum(cam.reshape(-1).astype(np.float64), 0)
    total = flattened.sum()
    if total <= EPSILON:
        return 0.0, 0.0, 0.0
    distribution = flattened / total
    entropy = -np.sum(distribution * np.log(distribution + EPSILON)) / np.log(
        len(distribution)
    )
    ordered = np.sort(flattened)
    top10 = ordered[-max(1, int(np.ceil(0.10 * len(ordered)))) :].sum() / total
    top20 = ordered[-max(1, int(np.ceil(0.20 * len(ordered)))) :].sum() / total
    return float(entropy), float(top10), float(top20)


def _morphology(mask: np.ndarray) -> list[float]:
    area = int(mask.sum())
    if area == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    components, count = ndimage.label(
        mask, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    sizes = np.bincount(components.reshape(-1))[1:]
    horizontal = np.mean(mask[:, 1:] != mask[:, :-1])
    vertical = np.mean(mask[1:, :] != mask[:-1, :])
    return [
        area / mask.size,
        float(count),
        float(sizes.max() / area),
        float(sizes.mean()),
        float((horizontal + vertical) / 2),
        1.0,
    ]


def _jsd(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (first + second)
    return 0.5 * np.sum(
        first * (np.log(first + EPSILON) - np.log(midpoint + EPSILON))
        + second * (np.log(second + EPSILON) - np.log(midpoint + EPSILON)),
        axis=0,
    )


def _pair_features(
    cams: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[str], list[float], dict[tuple[int, int], dict]]:
    names = []
    values = []
    pair_summary = {}
    for first in range(4):
        for second in range(first + 1, 4):
            prefix = f"pair_{BRANCH_NAMES[first]}_{BRANCH_NAMES[second]}"
            jsd = _jsd(probabilities[first], probabilities[second])
            agreement = float(
                np.mean(
                    np.argmax(probabilities[first], axis=0)
                    == np.argmax(probabilities[second], axis=0)
                )
            )
            cosine_values = []
            for class_id in range(4):
                left = cams[first, class_id].reshape(-1).astype(np.float64)
                right = cams[second, class_id].reshape(-1).astype(np.float64)
                cosine_values.append(
                    float(
                        np.dot(left, right)
                        / (np.linalg.norm(left) * np.linalg.norm(right) + EPSILON)
                    )
                )
            l1 = float(
                np.mean(np.abs(probabilities[first] - probabilities[second]))
            )
            pair_values = {
                "jsd_mean": float(jsd.mean()),
                "jsd_std": float(jsd.std()),
                "jsd_p90": float(np.quantile(jsd, 0.90)),
                "argmax_agreement": agreement,
                "cam_cosine_mean": float(np.mean(cosine_values)),
                "cam_cosine_min": float(np.min(cosine_values)),
                "probability_l1": l1,
            }
            pair_summary[(first, second)] = pair_values
            for suffix, value in pair_values.items():
                names.append(f"{prefix}_{suffix}")
                values.append(value)
    return names, values, pair_summary


def _branch_features(
    branch_index: int,
    cam: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    predicted_presence: np.ndarray,
    pair_summary: dict,
    probabilities: np.ndarray,
) -> tuple[list[str], list[float]]:
    names = []
    values = []
    entropy = -np.sum(probability * np.log(probability + EPSILON), axis=0)
    ordered_probability = np.sort(probability, axis=0)
    top1 = ordered_probability[-1]
    margin = ordered_probability[-1] - ordered_probability[-2]
    for suffix, value in zip(
        (
            "entropy_mean",
            "entropy_std",
            "entropy_p90",
            "top1_mean",
            "top1_std",
            "margin_mean",
            "margin_std",
            "margin_p10",
            "margin_p50",
        ),
        [
            *_stats(entropy, (0.90,)),
            *_stats(top1),
            *_stats(margin, (0.10, 0.50)),
        ],
    ):
        names.append(suffix)
        values.append(value)

    activation_metric_names = ("mean", "std", "max", "p90", "p95", "p99")
    activation_by_class = np.zeros((4, len(activation_metric_names)), dtype=np.float64)
    spatial_by_class = np.zeros((4, 3), dtype=np.float64)
    for class_id in range(4):
        class_map = cam[class_id]
        class_values = (
            float(class_map.mean()),
            float(class_map.std()),
            float(class_map.max()),
            float(np.quantile(class_map, 0.90)),
            float(np.quantile(class_map, 0.95)),
            float(np.quantile(class_map, 0.99)),
        )
        activation_by_class[class_id] = class_values
        for metric_name, value in zip(activation_metric_names, class_values):
            names.append(f"cam_class{class_id}_{metric_name}")
            values.append(value)
        spatial_values = _spatial_features(class_map)
        spatial_by_class[class_id] = spatial_values
        for metric_name, value in zip(
            ("spatial_entropy", "top10_mass_ratio", "top20_mass_ratio"),
            spatial_values,
        ):
            names.append(f"cam_class{class_id}_{metric_name}")
            values.append(value)
        for metric_name, value in zip(
            (
                "area_fraction",
                "component_count",
                "largest_component_ratio",
                "mean_component_size",
                "boundary_density",
                "argmax_presence",
            ),
            _morphology(prediction == class_id),
        ):
            names.append(f"pred_class{class_id}_{metric_name}")
            values.append(value)

    present = np.flatnonzero(predicted_presence)
    if not len(present):
        present = np.asarray([int(np.argmax(cam.mean(axis=(1, 2))))])
    for metric_index, metric_name in enumerate(activation_metric_names):
        selected = activation_by_class[present, metric_index]
        for aggregate_name, value in zip(
            ("mean", "std", "max"),
            (selected.mean(), selected.std(), selected.max()),
        ):
            names.append(f"present_cam_{metric_name}_{aggregate_name}")
            values.append(float(value))
    for metric_index, metric_name in enumerate(
        ("spatial_entropy", "top10_mass_ratio", "top20_mass_ratio")
    ):
        selected = spatial_by_class[present, metric_index]
        for aggregate_name, value in zip(
            ("mean", "std", "max"),
            (selected.mean(), selected.std(), selected.max()),
        ):
            names.append(f"present_cam_{metric_name}_{aggregate_name}")
            values.append(float(value))

    related = []
    for pair, pair_values in pair_summary.items():
        if branch_index in pair:
            related.append(pair_values)
    for metric, reduce_name, reducer in (
        ("jsd_mean", "mean", np.mean),
        ("jsd_mean", "max", np.max),
        ("probability_l1", "mean", np.mean),
        ("probability_l1", "max", np.max),
        ("argmax_agreement", "mean", np.mean),
        ("argmax_agreement", "min", np.min),
    ):
        names.append(f"to_others_{metric}_{reduce_name}")
        values.append(float(reducer([item[metric] for item in related])))
    argmax_maps = np.argmax(probabilities, axis=1)
    class_votes = np.stack(
        [(argmax_maps == class_id).sum(axis=0) for class_id in range(4)], axis=0
    )
    majority = np.argmax(class_votes, axis=0)
    names.append("agreement_with_majority_branch")
    values.append(float(np.mean(argmax_maps[branch_index] == majority)))
    for class_id in range(4):
        names.append(f"predicted_class_presence_{class_id}")
        values.append(float(predicted_presence[class_id]))
    for identity in range(4):
        names.append(f"branch_id_{identity}")
        values.append(float(identity == branch_index))
    return names, values


def extract_cam_signals(cache_dir: Path, output_path: Path) -> dict:
    cache_dir = Path(cache_dir)
    cams = [np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in BRANCH_NAMES]
    predictions = np.load(cache_dir / "branch_predictions.npy", mmap_mode="r")
    presence = np.load(cache_dir / "class_presence.npy", mmap_mode="r")
    first_names = None
    feature_memmap = None
    for index in range(len(presence)):
        image_cams = np.stack([cam[index] for cam in cams], axis=0)
        probabilities = np.stack(
            [_softmax(image_cams[branch], axis=0) for branch in range(4)], axis=0
        )
        pair_names, pair_values, pair_summary = _pair_features(
            image_cams, probabilities
        )
        rows = []
        for branch_index in range(4):
            branch_names, branch_values = _branch_features(
                branch_index,
                image_cams[branch_index],
                probabilities[branch_index],
                predictions[index, branch_index],
                presence[index],
                pair_summary,
                probabilities,
            )
            names = branch_names + pair_names
            rows.append(branch_values + pair_values)
            if first_names is None:
                first_names = names
            elif names != first_names:
                raise RuntimeError("Signal A feature ordering changed across candidates")
        if feature_memmap is None:
            feature_memmap = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(presence), 4, len(first_names)),
            )
        feature_memmap[index] = np.asarray(rows, dtype=np.float32)
    feature_memmap.flush()
    if not np.isfinite(feature_memmap).all():
        raise RuntimeError("Signal Set A contains non-finite values")
    names_path = Path(output_path).with_suffix(".names.json")
    names_path.write_text(json.dumps(first_names, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(output_path),
        "names_path": str(names_path),
        "shape": list(feature_memmap.shape),
        "feature_count": len(first_names),
        "contains_gt": False,
        "finite": True,
    }
