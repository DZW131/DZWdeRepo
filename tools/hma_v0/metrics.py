"""Streaming official metrics and pixel-level mechanism diagnostics."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import ndimage

from tools.hma_v0 import BACKGROUND, CAM_WEIGHTS, N_CLASS


def minmax_normalize(cam):
    cam = np.asarray(cam, dtype=np.float32)
    minimum = cam.min(axis=(1, 2), keepdims=True)
    maximum = cam.max(axis=(1, 2), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def fusion_response(cam_28_1, cam_28_2, cam_deep):
    return (
        CAM_WEIGHTS["28_1"] * minmax_normalize(cam_28_1)
        + CAM_WEIGHTS["28_2"] * minmax_normalize(cam_28_2)
        + CAM_WEIGHTS["deep"] * minmax_normalize(cam_deep)
    )


def prediction_from_fusion(cam_28_1, cam_28_2, cam_deep, presence=None):
    response = fusion_response(cam_28_1, cam_28_2, cam_deep)
    if presence is not None:
        response = response * np.asarray(presence, dtype=np.float32).reshape(N_CLASS, 1, 1)
    return response.argmax(axis=0).astype(np.uint8), response


def prediction_from_standalone(cam, presence):
    response = minmax_normalize(cam)
    response *= np.asarray(presence, dtype=np.float32).reshape(N_CLASS, 1, 1)
    return response.argmax(axis=0).astype(np.uint8)


class OfficialMetricAccumulator:
    """Streaming equivalent of released iouutils.scores()."""

    def __init__(self):
        self.hist = np.zeros((N_CLASS + 1, N_CLASS + 1), dtype=np.float64)
        self.images = 0

    def update(self, ground_truth, prediction):
        ground_truth = np.asarray(ground_truth, dtype=np.int64)
        prediction = np.asarray(prediction, dtype=np.int64).copy()
        prediction[ground_truth == BACKGROUND] = BACKGROUND
        mask = (ground_truth >= 0) & (ground_truth <= BACKGROUND)
        values = (N_CLASS + 1) * ground_truth[mask] + prediction[mask]
        self.hist += np.bincount(
            values, minlength=(N_CLASS + 1) ** 2
        ).reshape(N_CLASS + 1, N_CLASS + 1)
        self.images += 1

    def scores(self):
        hist = self.hist.copy()
        hist[BACKGROUND, BACKGROUND] = 0.0
        diagonal = np.diag(hist)
        total = hist.sum()
        accuracy = diagonal.sum() / total
        class_accuracy = diagonal[:N_CLASS] / hist.sum(axis=1)[:N_CLASS]
        union = hist.sum(axis=1) + hist.sum(axis=0) - diagonal
        iou = diagonal[:N_CLASS] / union[:N_CLASS]
        frequency = hist.sum(axis=1)[:N_CLASS] / total
        dice = []
        for class_index in range(N_CLASS):
            tp = diagonal[class_index]
            fp = hist[:, class_index].sum() - tp
            fn = hist[class_index, :].sum() - tp
            denominator = 2 * tp + fp + fn
            dice.append(0.0 if denominator == 0 else 2 * tp / denominator)
        return {
            "pixel_accuracy": float(accuracy),
            "mean_accuracy": float(np.nanmean(class_accuracy)),
            "frequency_weighted_iou": float(
                (frequency[frequency > 0] * iou[frequency > 0]).sum()
            ),
            "mean_iou": float(np.nanmean(iou)),
            "class_iou": {str(index): float(value) for index, value in enumerate(iou)},
            "class_dice": {str(index): float(value) for index, value in enumerate(dice)},
            "mean_dice": float(np.mean(dice)),
            "images": int(self.images),
            "histogram": hist.astype(np.int64).tolist(),
        }


def foreground_boundary_distance(ground_truth):
    ground_truth = np.asarray(ground_truth, dtype=np.uint8)
    foreground = ground_truth < BACKGROUND
    boundary = np.zeros_like(foreground, dtype=bool)
    height, width = ground_truth.shape
    for dy, dx in (
        (-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1),
    ):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        left = ground_truth[y0:y1, x0:x1]
        right = ground_truth[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        transition = (left < BACKGROUND) & (right < BACKGROUND) & (left != right)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= transition
    if boundary.any():
        distance = ndimage.distance_transform_edt(~boundary)
    else:
        distance = np.full(ground_truth.shape, np.inf, dtype=np.float64)
    return {
        "B0_le_2": foreground & (distance <= 2.0),
        "B1_3_7": foreground & (distance > 2.0) & (distance <= 7.0),
        "B2_ge_8": foreground & (distance > 7.0),
        "boundary_le_7": foreground & (distance <= 7.0),
        "interior_ge_8": foreground & (distance > 7.0),
    }


class SpatialTransitionAccumulator:
    def __init__(self, names):
        self.data = {
            name: {
                bin_name: defaultdict(int)
                for bin_name in ("B0_le_2", "B1_3_7", "B2_ge_8")
            }
            for name in names
        }

    def update(self, name, ground_truth, baseline, candidate, bins):
        for bin_name in ("B0_le_2", "B1_3_7", "B2_ge_8"):
            mask = bins[bin_name]
            base_correct = baseline == ground_truth
            candidate_correct = candidate == ground_truth
            target = self.data[name][bin_name]
            target["pixels"] += int(mask.sum())
            target["baseline_correct"] += int((mask & base_correct).sum())
            target["candidate_correct"] += int((mask & candidate_correct).sum())
            target["recovered"] += int((mask & ~base_correct & candidate_correct).sum())
            target["harmed"] += int((mask & base_correct & ~candidate_correct).sum())

    def summary(self):
        output = {}
        for name, bins in self.data.items():
            output[name] = {}
            for bin_name, values in bins.items():
                pixels = values["pixels"]
                recovered, harmed = values["recovered"], values["harmed"]
                output[name][bin_name] = {
                    **{key: int(value) for key, value in values.items()},
                    "net": int(recovered - harmed),
                    "accuracy_delta": float(
                        (values["candidate_correct"] - values["baseline_correct"])
                        / max(pixels, 1)
                    ),
                }
        return output


def _predicted_present(prediction, presence):
    lookup = np.asarray(presence, dtype=bool)
    return lookup[np.asarray(prediction, dtype=np.int64)]


class ErrorTaxonomyAccumulator:
    """Raw-to-candidate error transitions on GT foreground pixels."""

    ERROR_TYPES = ("absent_class", "present_confusion", "boundary", "interior")

    def __init__(self, candidate_names):
        self.data = {
            name: {error_type: defaultdict(int) for error_type in self.ERROR_TYPES}
            for name in candidate_names
        }

    @staticmethod
    def masks(ground_truth, prediction, gt_presence, bins):
        foreground = ground_truth < BACKGROUND
        wrong = foreground & (prediction != ground_truth)
        predicted_present = _predicted_present(prediction, gt_presence)
        return {
            "absent_class": wrong & ~predicted_present,
            "present_confusion": wrong & predicted_present,
            "boundary": wrong & bins["boundary_le_7"],
            "interior": wrong & bins["interior_ge_8"],
        }

    def update(self, name, ground_truth, baseline, candidate, gt_presence, bins):
        base_masks = self.masks(ground_truth, baseline, gt_presence, bins)
        candidate_masks = self.masks(ground_truth, candidate, gt_presence, bins)
        foreground = ground_truth < BACKGROUND
        base_correct = foreground & (baseline == ground_truth)
        candidate_correct = foreground & (candidate == ground_truth)
        for error_type in self.ERROR_TYPES:
            target = self.data[name][error_type]
            target["raw_wrong"] += int(base_masks[error_type].sum())
            target["candidate_wrong"] += int(candidate_masks[error_type].sum())
            target["recovered"] += int((base_masks[error_type] & candidate_correct).sum())
            target["harmed"] += int((base_correct & candidate_masks[error_type]).sum())

    def summary(self):
        result = {}
        for name, categories in self.data.items():
            result[name] = {}
            for category, values in categories.items():
                result[name][category] = {
                    **{key: int(value) for key, value in values.items()},
                    "net": int(values["recovered"] - values["harmed"]),
                }
        return result


class ComplementarityAccumulator:
    def __init__(self):
        self.counts = defaultdict(int)

    def update(self, ground_truth, raw, gsr, ch, full):
        mask = ground_truth < BACKGROUND
        raw_correct = raw == ground_truth
        gsr_correct = gsr == ground_truth
        ch_correct = ch == ground_truth
        full_correct = full == ground_truth
        g_recover = mask & ~raw_correct & gsr_correct
        c_recover = mask & ~raw_correct & ch_correct
        f_recover = mask & ~raw_correct & full_correct
        g_harm = mask & raw_correct & ~gsr_correct
        c_harm = mask & raw_correct & ~ch_correct
        f_harm = mask & raw_correct & ~full_correct
        pairs = {
            "foreground_pixels": mask,
            "gsr_recover": g_recover,
            "ch_recover": c_recover,
            "full_recover": f_recover,
            "gsr_unique_recover": g_recover & ~c_recover,
            "ch_unique_recover": c_recover & ~g_recover,
            "both_recover": g_recover & c_recover,
            "gsr_harm": g_harm,
            "ch_harm": c_harm,
            "full_harm": f_harm,
            "gsr_unique_harm": g_harm & ~c_harm,
            "ch_unique_harm": c_harm & ~g_harm,
            "both_harm": g_harm & c_harm,
            "gsr_correct_ch_wrong": mask & gsr_correct & ~ch_correct,
            "ch_correct_gsr_wrong": mask & ch_correct & ~gsr_correct,
        }
        for name, values in pairs.items():
            self.counts[name] += int(values.sum())

    def summary(self):
        recovery_union = (
            self.counts["gsr_recover"] + self.counts["ch_recover"]
            - self.counts["both_recover"]
        )
        harm_union = (
            self.counts["gsr_harm"] + self.counts["ch_harm"]
            - self.counts["both_harm"]
        )
        return {
            **{key: int(value) for key, value in self.counts.items()},
            "recovery_set_jaccard": float(
                self.counts["both_recover"] / max(recovery_union, 1)
            ),
            "harm_set_jaccard": float(
                self.counts["both_harm"] / max(harm_union, 1)
            ),
        }
