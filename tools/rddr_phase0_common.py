"""Frozen, diagnostic-only utilities for the RDDR Phase-0 audit."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
CHECKPOINT_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
N_CLASS = 4
BACKGROUND = 4
EPSILON = 1.0e-8
BCSS_THRESHOLDS = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
SCORE_NAMES = ("S_JS", "S_entropy", "S_lowconf", "S_cos", "S_hard")
TOP_FRACTIONS = (0.05, 0.10, 0.20, 0.30)
SCORE_RANGES = {
    "S_JS": (0.0, float(np.log(2.0))),
    "S_entropy": (0.0, float(np.log(4.0))),
    "S_lowconf": (0.0, 0.75),
    "S_cos": (0.0, 1.0),
    "S_hard": (0.0, 1.0),
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_forward(model, image):
    """Reproduce A0 forward exactly while exposing frozen intermediate tensors."""

    x = model.conv1a(image)
    x = model.b2(x)
    x = model.b2_1(x)
    x = model.b2_2(x)
    x = model.b3(x)
    x = model.b3_1(x)
    x = model.b3_2(x)
    feat_56 = x
    x = model.b4(x)
    x = model.b4_1(x)
    x = model.b4_2(x)
    x = model.b4_3(x)
    x = model.b4_4(x)
    x = model.b4_5(x)
    feat_28_raw = F.relu(model.bn45(x))
    x, _ = model.b5(x, get_x_bn_relu=True)
    x = model.b5_1(x)
    x = model.b5_2(x)
    feat_28_2 = F.relu(model.bn52(x))
    x, _ = model.b6(x, get_x_bn_relu=True)
    x = model.b7(x)
    feat_deep = F.relu(model.bn7(x))

    feat_56_rect = model.hfrm_56(feat_56, feat_deep)
    feat_28_rect = model.hfrm_28_1(feat_28_raw, feat_deep)
    feat_28_2_rect = model.hfrm_28_2(feat_28_2, feat_deep)
    cam_56_logits = model.ic_56(feat_56_rect)
    cam_28_raw_logits = model.ic1(feat_28_raw)
    cam_28_rect_logits = model.ic1(feat_28_rect)
    cam_28_2_logits = model.ic2(feat_28_2_rect)
    feat_deep_drop = model.dropout7(feat_deep)
    cam_deep_logits = model.fc8(feat_deep_drop)

    def pool(value):
        return F.avg_pool2d(
            value, kernel_size=(value.size(2), value.size(3)), padding=0
        ).view(value.size(0), -1)
    outputs = (
        pool(cam_56_logits),
        pool(cam_28_rect_logits),
        pool(cam_28_2_logits),
        pool(cam_deep_logits),
    )
    outputs = (*outputs, torch.sigmoid(outputs[3]), cam_56_logits,
               cam_28_rect_logits, cam_28_2_logits, cam_deep_logits, feat_56_rect)
    diagnostics = {
        "F28_raw": feat_28_raw,
        "F28_rect": feat_28_rect,
        "Ddeep": feat_deep,
        "CAM28_raw_logits": cam_28_raw_logits,
        "CAM28_rect_logits": cam_28_rect_logits,
        "CAM28_2_logits": cam_28_2_logits,
        "CAMdeep_logits": model.fc8(feat_deep),
    }
    return outputs, diagnostics


def probability_scores(shallow_logits, deep_logits):
    p_shallow = torch.softmax(shallow_logits.float(), dim=1)
    p_deep = torch.softmax(deep_logits.float(), dim=1)
    midpoint = 0.5 * (p_shallow + p_deep)
    js = 0.5 * (
        (p_shallow * ((p_shallow + EPSILON).log() - (midpoint + EPSILON).log())).sum(1)
        + (p_deep * ((p_deep + EPSILON).log() - (midpoint + EPSILON).log())).sum(1)
    )
    entropy = -(p_shallow * (p_shallow + EPSILON).log()).sum(1)
    low_confidence = 1.0 - p_shallow.max(1).values
    cosine = 1.0 - F.cosine_similarity(p_shallow, p_deep, dim=1, eps=EPSILON)
    hard = (p_shallow.argmax(1) != p_deep.argmax(1)).float()
    scores = {
        "S_JS": js,
        "S_entropy": entropy,
        "S_lowconf": low_confidence,
        "S_cos": cosine,
        "S_hard": hard,
    }
    if not all(torch.isfinite(value).all() for value in scores.values()):
        raise FloatingPointError("Non-finite RDDR Phase-0 score")
    return p_shallow, p_deep, scores


def minmax_cam(cam):
    minimum = cam.amin(dim=(-2, -1), keepdim=True)
    maximum = cam.amax(dim=(-2, -1), keepdim=True)
    return (cam - minimum) / (maximum - minimum + EPSILON)


def canonical_predictions(diagnostics, output_size):
    def resize(value):
        return F.interpolate(
            value, size=output_size, mode="bilinear", align_corners=False
        )
    raw_logits = resize(diagnostics["CAM28_raw_logits"])
    rect_logits = resize(diagnostics["CAM28_rect_logits"])
    deep_logits = resize(diagnostics["CAMdeep_logits"])
    cam_28_2 = resize(F.relu(diagnostics["CAM28_2_logits"]))
    raw_prediction = raw_logits.argmax(1)
    rect_prediction = F.relu(rect_logits).argmax(1)
    deep_prediction = deep_logits.argmax(1)

    rect_cam = minmax_cam(F.relu(rect_logits))
    cam_28_2 = minmax_cam(cam_28_2)
    deep_cam = minmax_cam(F.relu(deep_logits))
    deep_probability = torch.sigmoid(
        F.adaptive_avg_pool2d(diagnostics["CAMdeep_logits"], 1).flatten(1)
    )
    presence = (deep_probability > torch.as_tensor(
        BCSS_THRESHOLDS, device=deep_probability.device
    )).float()
    empty = presence.sum(1) == 0
    if empty.any():
        presence[empty, deep_probability[empty].argmax(1)] = 1.0
    fused = (0.6 * rect_cam + 0.2 * cam_28_2 + 0.2 * deep_cam)
    fused = fused * presence[:, :, None, None]
    return {
        "raw": raw_prediction,
        "rect": rect_prediction,
        "deep": deep_prediction,
        "final": fused.argmax(1),
        "raw_logits": raw_logits,
        "deep_logits": deep_logits,
    }


def eligible_error(prediction, truth):
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    valid = (truth >= 0) & (truth < N_CLASS)
    return valid, (prediction != truth) & valid


def foreground_boundary_distance(truth):
    truth = np.asarray(truth, dtype=np.uint8)
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
        transition = (left < BACKGROUND) & (right < BACKGROUND) & (left != right)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0 + dy:y1 + dy, x0 + dx:x1 + dx] |= transition
    distance = (
        ndimage.distance_transform_edt(~boundary)
        if boundary.any() else np.full(truth.shape, np.inf)
    )
    return {
        "boundary": foreground & (distance <= 7.0),
        "interior": foreground & (distance > 7.0),
    }


def dataset_quantile_threshold(values, top_fraction):
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("Cannot threshold an empty score array")
    return float(np.quantile(values, 1.0 - top_fraction, method="higher"))


def official_histogram(truth, prediction):
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64).copy()
    prediction[truth == BACKGROUND] = BACKGROUND
    valid = (truth >= 0) & (truth <= BACKGROUND)
    encoded = 5 * truth[valid] + prediction[valid]
    histogram = np.bincount(encoded, minlength=25).reshape(5, 5)
    histogram[4, 4] = 0
    return histogram.astype(np.int64)


def official_scores(histogram):
    histogram = np.asarray(histogram, dtype=np.float64)
    diagonal = np.diag(histogram)
    union = histogram.sum(1) + histogram.sum(0) - diagonal
    iou = np.divide(
        diagonal[:4], union[:4], out=np.full(4, np.nan), where=union[:4] > 0
    )
    return {
        "pixel_accuracy": float(diagonal.sum() / max(histogram.sum(), 1.0)),
        "mIoU": float(np.nanmean(iou)),
        "class_iou": iou.tolist(),
    }


def bootstrap_indices(n_images, resamples=10000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_images, size=(resamples, n_images), dtype=np.int32)


def bootstrap_mean(values, indices):
    values = np.asarray(values, dtype=np.float64)
    sampled = values[indices]
    valid = np.isfinite(sampled)
    estimates = np.divide(
        np.where(valid, sampled, 0.0).sum(axis=1),
        valid.sum(axis=1),
        out=np.full(sampled.shape[0], np.nan, dtype=np.float64),
        where=valid.sum(axis=1) > 0,
    )
    finite = estimates[np.isfinite(estimates)]
    return estimates, {
        "observed": float(np.nanmean(values)) if np.isfinite(values).any() else float("nan"),
        "bootstrap_mean": float(finite.mean()) if finite.size else float("nan"),
        "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
    }


def bootstrap_ratio(numerator, denominator, indices):
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    sampled_numerator = numerator[indices].sum(axis=1)
    sampled_denominator = denominator[indices].sum(axis=1)
    estimates = np.divide(
        sampled_numerator,
        sampled_denominator,
        out=np.full_like(sampled_numerator, np.nan),
        where=sampled_denominator > 0,
    )
    finite = estimates[np.isfinite(estimates)]
    denominator_total = denominator.sum()
    return estimates, {
        "observed": float(numerator.sum() / denominator_total) if denominator_total > 0 else float("nan"),
        "bootstrap_mean": float(finite.mean()) if finite.size else float("nan"),
        "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
    }


class BinaryHistogram:
    """High-resolution streaming binary metrics with deterministic ties."""

    def __init__(self, score_name, bins=65536):
        self.score_name = score_name
        self.minimum, self.maximum = SCORE_RANGES[score_name]
        self.bins = 2 if score_name == "S_hard" else int(bins)
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)

    def _indices(self, scores):
        scores = np.clip(np.asarray(scores, dtype=np.float64), self.minimum, self.maximum)
        scale = self.bins / max(self.maximum - self.minimum, EPSILON)
        indices = ((scores - self.minimum) * scale).astype(np.int64)
        return np.clip(indices, 0, self.bins - 1)

    def update(self, scores, labels):
        labels = np.asarray(labels, dtype=bool)
        indices = self._indices(scores)
        self.positive += np.bincount(indices[labels], minlength=self.bins)
        self.negative += np.bincount(indices[~labels], minlength=self.bins)

    @staticmethod
    def _auc(positive, negative):
        n_positive = positive.sum()
        n_negative = negative.sum()
        if n_positive == 0 or n_negative == 0:
            return float("nan")
        negative_below = np.cumsum(negative) - negative
        wins = (positive * (negative_below + 0.5 * negative)).sum(dtype=np.float64)
        return float(wins / (n_positive * n_negative))

    @staticmethod
    def _average_precision(positive, negative):
        n_positive = positive.sum()
        if n_positive == 0:
            return float("nan")
        tp = np.cumsum(positive[::-1], dtype=np.float64)
        fp = np.cumsum(negative[::-1], dtype=np.float64)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        return float((precision * positive[::-1]).sum() / n_positive)

    def _quantiles(self, counts):
        total = int(counts.sum())
        if total == 0:
            return {"p25": float("nan"), "p50": float("nan"), "p75": float("nan")}
        cumulative = np.cumsum(counts)
        width = (self.maximum - self.minimum) / self.bins
        result = {}
        for name, probability in (("p25", 0.25), ("p50", 0.50), ("p75", 0.75)):
            index = int(np.searchsorted(cumulative, probability * (total - 1), side="right"))
            index = min(max(index, 0), self.bins - 1)
            result[name] = float(self.minimum + (index + 0.5) * width)
        return result

    def result(self):
        return {
            "pixel_weighted_AUROC": self._auc(self.positive, self.negative),
            "pixel_weighted_AUPRC": self._average_precision(self.positive, self.negative),
            "positive": int(self.positive.sum()),
            "negative": int(self.negative.sum()),
            "histogram_bins": self.bins,
            "score_range": [self.minimum, self.maximum],
            "positive_quantiles": self._quantiles(self.positive),
            "negative_quantiles": self._quantiles(self.negative),
        }
