"""Streaming metrics for the frozen S²HR-v1 FIDA-v0 audit."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from PIL import Image
from scipy import ndimage


N_CLASS = 4
BACKGROUND = 4
REGIONS = ("overall", "B0_le_2", "B1_3_7", "B2_ge_8", "boundary", "interior")


class OfficialMetricAccumulator:
    """Streaming equivalent of released ``iouutils.scores``."""

    def __init__(self):
        self.hist = np.zeros((5, 5), dtype=np.float64)
        self.images = 0

    def update(self, ground_truth, prediction):
        truth = np.asarray(ground_truth, dtype=np.int64)
        pred = np.asarray(prediction, dtype=np.int64).copy()
        pred[truth == BACKGROUND] = BACKGROUND
        valid = (truth >= 0) & (truth <= BACKGROUND)
        values = 5 * truth[valid] + pred[valid]
        self.hist += np.bincount(values, minlength=25).reshape(5, 5)
        self.images += 1

    def scores(self):
        hist = self.hist.copy()
        hist[BACKGROUND, BACKGROUND] = 0.0
        diagonal = np.diag(hist)
        total = hist.sum()
        class_accuracy = diagonal[:4] / hist.sum(axis=1)[:4]
        union = hist.sum(axis=1) + hist.sum(axis=0) - diagonal
        iou = diagonal[:4] / union[:4]
        frequency = hist.sum(axis=1)[:4] / total
        dice = []
        for class_index in range(4):
            tp = diagonal[class_index]
            denominator = 2 * tp + hist[:, class_index].sum() - tp + hist[class_index, :].sum() - tp
            dice.append(0.0 if denominator == 0 else 2 * tp / denominator)
        return {
            "mIoU": float(np.nanmean(iou)),
            "mDice": float(np.mean(dice)),
            "class_iou": {str(i): float(v) for i, v in enumerate(iou)},
            "class_dice": {str(i): float(v) for i, v in enumerate(dice)},
            "pixel_accuracy": float(diagonal.sum() / total),
            "mean_accuracy": float(np.nanmean(class_accuracy)),
            "frequency_weighted_iou": float(
                (frequency[frequency > 0] * iou[frequency > 0]).sum()
            ),
            "images": self.images,
            "histogram": hist.astype(np.int64).tolist(),
        }


def image_presence(ground_truth):
    values = np.unique(np.asarray(ground_truth))
    return np.asarray([index in values for index in range(4)], dtype=bool)


def foreground_boundary_bins(ground_truth):
    truth = np.asarray(ground_truth, dtype=np.uint8)
    foreground = truth < BACKGROUND
    boundary = np.zeros_like(foreground, dtype=bool)
    height, width = truth.shape
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        left = truth[y0:y1, x0:x1]
        right = truth[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        transition = (left < 4) & (right < 4) & (left != right)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= transition
    distance = (
        ndimage.distance_transform_edt(~boundary)
        if boundary.any()
        else np.full(truth.shape, np.inf, dtype=np.float64)
    )
    return {
        "B0_le_2": foreground & (distance <= 2.0),
        "B1_3_7": foreground & (distance > 2.0) & (distance <= 7.0),
        "B2_ge_8": foreground & (distance > 7.0),
        "boundary": foreground & (distance <= 7.0),
        "interior": foreground & (distance > 7.0),
        "foreground": foreground,
        "distance": distance,
    }


def semantic_transition_band(class_map, foreground_only):
    values = np.asarray(class_map, dtype=np.uint8)
    boundary = np.zeros_like(values, dtype=bool)
    height, width = values.shape
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        left = values[y0:y1, x0:x1]
        right = values[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        transition = left != right
        if foreground_only:
            transition &= (left < 4) & (right < 4)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= transition
    return ndimage.maximum_filter(
        boundary.astype(np.uint8), size=3, mode="constant", cval=0
    ).astype(bool)


def resize_nearest(array, shape):
    height, width = shape
    return np.asarray(
        Image.fromarray(np.asarray(array)).resize((width, height), Image.Resampling.NEAREST)
    )


class SpatialTransitionAccumulator:
    def __init__(self, names):
        self.data = {
            name: {region: defaultdict(int) for region in ("B0_le_2", "B1_3_7", "B2_ge_8")}
            for name in names
        }

    def update(self, name, truth, baseline, candidate, bins):
        base_correct = baseline == truth
        candidate_correct = candidate == truth
        for region in self.data[name]:
            mask = bins[region]
            target = self.data[name][region]
            target["pixels"] += int(mask.sum())
            target["baseline_correct"] += int((mask & base_correct).sum())
            target["candidate_correct"] += int((mask & candidate_correct).sum())
            target["recovered"] += int((mask & ~base_correct & candidate_correct).sum())
            target["harmed"] += int((mask & base_correct & ~candidate_correct).sum())

    def summary(self):
        output = {}
        for name, regions in self.data.items():
            output[name] = {}
            for region, values in regions.items():
                pixels = values["pixels"]
                output[name][region] = {
                    **{key: int(value) for key, value in values.items()},
                    "net": int(values["recovered"] - values["harmed"]),
                    "accuracy_delta": float(
                        (values["candidate_correct"] - values["baseline_correct"])
                        / max(pixels, 1)
                    ),
                }
        return output


class TeacherReliabilityAccumulator:
    def __init__(self):
        self.data = {
            presence: {
                region: {
                    class_name: defaultdict(int)
                    for class_name in ("overall", "0", "1", "2", "3")
                }
                for region in REGIONS
            }
            for presence in ("oracle", "deployed")
        }
        self.present_confusion = {
            presence: {method: defaultdict(int) for method in ("deep", "raw28_1")}
            for presence in ("oracle", "deployed")
        }

    def update(self, presence_name, truth, deep, shallow, bins, gt_presence):
        foreground = bins["foreground"]
        deep_correct = deep == truth
        shallow_correct = shallow == truth
        region_masks = {
            "overall": foreground,
            "B0_le_2": bins["B0_le_2"],
            "B1_3_7": bins["B1_3_7"],
            "B2_ge_8": bins["B2_ge_8"],
            "boundary": bins["boundary"],
            "interior": bins["interior"],
        }
        for region, region_mask in region_masks.items():
            for class_name in ("overall", "0", "1", "2", "3"):
                mask = region_mask if class_name == "overall" else region_mask & (truth == int(class_name))
                target = self.data[presence_name][region][class_name]
                target["pixels"] += int(mask.sum())
                target["deep_correct"] += int((mask & deep_correct).sum())
                target["shallow_correct"] += int((mask & shallow_correct).sum())
                target["deep_help"] += int((mask & deep_correct & ~shallow_correct).sum())
                target["deep_harm"] += int((mask & ~deep_correct & shallow_correct).sum())
                target["both_correct"] += int((mask & deep_correct & shallow_correct).sum())
                target["both_wrong"] += int((mask & ~deep_correct & ~shallow_correct).sum())

        for method, prediction in (("deep", deep), ("raw28_1", shallow)):
            predicted_present = np.asarray(gt_presence, dtype=bool)[prediction]
            errors = foreground & (prediction != truth) & predicted_present
            target = self.present_confusion[presence_name][method]
            target["foreground_pixels"] += int(foreground.sum())
            target["errors"] += int(errors.sum())

    def summary(self):
        rows = []
        for presence, regions in self.data.items():
            for region, classes in regions.items():
                for class_name, values in classes.items():
                    pixels = values["pixels"]
                    net = values["deep_help"] - values["deep_harm"]
                    rows.append({
                        "presence": presence,
                        "region": region,
                        "class": class_name,
                        **{key: int(value) for key, value in values.items()},
                        "deep_accuracy": values["deep_correct"] / max(pixels, 1),
                        "raw28_1_accuracy": values["shallow_correct"] / max(pixels, 1),
                        "teacher_net": int(net),
                        "teacher_net_rate": net / max(pixels, 1),
                    })
        confusion = []
        for presence, methods in self.present_confusion.items():
            for method, values in methods.items():
                confusion.append({
                    "presence": presence,
                    "method": method,
                    "foreground_pixels": int(values["foreground_pixels"]),
                    "present_confusion_errors": int(values["errors"]),
                    "present_confusion_error_rate": values["errors"] / max(values["foreground_pixels"], 1),
                })
        return rows, confusion


class BoundaryQualityAccumulator:
    def __init__(self):
        self.counts = defaultdict(int)

    def update(self, predicted_boundary_28, truth, bins):
        predicted = np.asarray(predicted_boundary_28, dtype=bool)
        truth_28 = resize_nearest(np.asarray(truth, dtype=np.uint8), predicted.shape)
        gt_boundary = semantic_transition_band(truth_28, foreground_only=True)
        self.counts["pixels"] += int(predicted.size)
        self.counts["predicted"] += int(predicted.sum())
        self.counts["ground_truth"] += int(gt_boundary.sum())
        self.counts["intersection"] += int((predicted & gt_boundary).sum())
        self.counts["union"] += int((predicted | gt_boundary).sum())
        sampled_foreground = np.zeros_like(predicted, dtype=bool)
        for region in ("B0_le_2", "B1_3_7", "B2_ge_8"):
            sampled_region = resize_nearest(bins[region].astype(np.uint8), predicted.shape).astype(bool)
            sampled_foreground |= sampled_region
            self.counts[f"predicted_in_{region}"] += int((predicted & sampled_region).sum())
        self.counts["predicted_outside_foreground"] += int(
            (predicted & ~sampled_foreground).sum()
        )

    def summary(self):
        predicted = self.counts["predicted"]
        truth = self.counts["ground_truth"]
        intersection = self.counts["intersection"]
        return {
            **{key: int(value) for key, value in self.counts.items()},
            "precision": intersection / max(predicted, 1),
            "recall": intersection / max(truth, 1),
            "f1": 2 * intersection / max(predicted + truth, 1),
            "iou": intersection / max(self.counts["union"], 1),
            "predicted_boundary_fraction": predicted / max(self.counts["pixels"], 1),
            "gt_boundary_fraction": truth / max(self.counts["pixels"], 1),
            "predicted_boundary_region_fraction": {
                region: self.counts[f"predicted_in_{region}"] / max(predicted, 1)
                for region in ("B0_le_2", "B1_3_7", "B2_ge_8")
            },
            "b2_interior_contamination": self.counts["predicted_in_B2_ge_8"] / max(predicted, 1),
            "outside_foreground_fraction": self.counts["predicted_outside_foreground"] / max(predicted, 1),
        }


class ErrorTaxonomyAccumulator:
    CATEGORIES = ("absent_class", "present_confusion", "boundary", "interior")

    def __init__(self, variants):
        self.data = {
            variant: {category: defaultdict(int) for category in self.CATEGORIES}
            for variant in variants
        }

    @staticmethod
    def masks(truth, prediction, presence, bins):
        foreground = bins["foreground"]
        wrong = foreground & (prediction != truth)
        predicted_present = np.asarray(presence, dtype=bool)[prediction]
        return {
            "absent_class": wrong & ~predicted_present,
            "present_confusion": wrong & predicted_present,
            "boundary": wrong & bins["boundary"],
            "interior": wrong & bins["interior"],
        }

    def update(self, truth, predictions, gt_presence, bins):
        baseline = predictions["V00"]
        baseline_correct = bins["foreground"] & (baseline == truth)
        baseline_masks = self.masks(truth, baseline, gt_presence, bins)
        for variant, prediction in predictions.items():
            candidate_correct = bins["foreground"] & (prediction == truth)
            candidate_masks = self.masks(truth, prediction, gt_presence, bins)
            for category in self.CATEGORIES:
                target = self.data[variant][category]
                target["errors"] += int(candidate_masks[category].sum())
                target["v00_errors"] += int(baseline_masks[category].sum())
                target["recovered"] += int((baseline_masks[category] & candidate_correct).sum())
                target["harmed"] += int((baseline_correct & candidate_masks[category]).sum())

    def summary(self):
        output = {}
        for variant, categories in self.data.items():
            output[variant] = {}
            for category, values in categories.items():
                output[variant][category] = {
                    **{key: int(value) for key, value in values.items()},
                    "net": int(values["recovered"] - values["harmed"]),
                }
        return output


class ResidualUtilityAccumulator:
    def __init__(self):
        self.counts = defaultdict(int)

    def update(self, truth, zero, positive, deep, shallow, bins):
        foreground = bins["foreground"]
        zero_correct = zero == truth
        positive_correct = positive == truth
        recovered = foreground & ~zero_correct & positive_correct
        harmed = foreground & zero_correct & ~positive_correct
        unchanged = foreground & (zero_correct == positive_correct)
        help_opportunity = foreground & (deep == truth) & (shallow != truth)
        harm_opportunity = foreground & (deep != truth) & (shallow == truth)
        for name, mask in (
            ("foreground", foreground), ("recovered", recovered),
            ("harmed", harmed), ("unchanged", unchanged),
            ("deep_help_opportunity", help_opportunity),
            ("deep_harm_opportunity", harm_opportunity),
            ("recovered_in_deep_help", recovered & help_opportunity),
            ("recovered_in_deep_harm", recovered & harm_opportunity),
            ("harmed_in_deep_help", harmed & help_opportunity),
            ("harmed_in_deep_harm", harmed & harm_opportunity),
        ):
            self.counts[name] += int(mask.sum())

    def summary(self):
        return {
            **{key: int(value) for key, value in self.counts.items()},
            "net": int(self.counts["recovered"] - self.counts["harmed"]),
            "recovered_fraction_in_deep_help": self.counts["recovered_in_deep_help"] / max(self.counts["recovered"], 1),
            "harmed_fraction_in_deep_harm": self.counts["harmed_in_deep_harm"] / max(self.counts["harmed"], 1),
        }
