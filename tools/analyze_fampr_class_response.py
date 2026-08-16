"""Frozen A0 versus FA-MPR class-response diagnosis on BCSS validation.

This is an analysis-only tool.  It never constructs an optimizer, never calls
``train()``, and never mutates either checkpoint.  The final predictions use
the released SSHR preprocessing, three-view TTA, BCSS class-presence
thresholds, CAM normalization/fusion, and official background-overwrite
metric convention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, stats
from skimage.morphology import dilation as morphology_dilation, disk
import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net_CAM


CLASS_NAMES = ("tumor", "stroma", "normal", "necrosis")
STAGES = ("cam56", "cam28_1", "cam28_2", "camdeep")
FAMPR_STAGES = ("stage1", "stage2", "stage3")
FAMPR_TO_CAM = {
    "stage1": "cam56",
    "stage2": "cam28_1",
    "stage3": "cam28_2",
}
TRANSITION_GROUPS = (
    "BOTH_CORRECT",
    "CORRECTED_BY_FAMPR",
    "HARMED_BY_FAMPR",
    "BOTH_WRONG",
)
DIAGNOSTIC_METRICS = (
    "morphology",
    "dilation",
    "band_weight_0",
    "band_weight_1",
    "band_weight_2",
    "band_weight_3",
    "high_frequency_ratio",
    "g_low",
    "g_high",
    "g_high_over_low",
    "g_low_channel_std",
    "g_high_channel_std",
    "anchor_lambda",
    "gamma_context",
    "r_fa",
    "r_mpr",
)
KEY_METRICS = (
    "morphology",
    "dilation",
    "high_frequency_ratio",
    "r_mpr",
)
BCSS_THRESHOLDS = np.asarray((0.8, 0.9, 0.8, 0.6), dtype=np.float32)
OFFICIAL_CAM_WEIGHTS = np.asarray((0.0, 0.6, 0.2, 0.2), dtype=np.float32)
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--a0-checkpoint", type=Path, required=True)
    parser.add_argument("--fampr-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("audit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument(
        "--amp-dtype", choices=("none", "bf16", "fp16"), default="bf16"
    )
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--reservoir-size", type=int, default=20_000)
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--visual-top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Analysis smoke only. Omit for the complete validation set.",
    )
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().float().item()
        return value.detach().float().cpu().tolist()
    return value


class VectorStats:
    """Exact moments plus an explicitly approximate quantile reservoir."""

    def __init__(
        self,
        names: Sequence[str],
        reservoir_size: int,
        sample_stride: int,
        seed: int,
    ) -> None:
        self.names = tuple(names)
        self.count = 0
        self.total = np.zeros(len(self.names), dtype=np.float64)
        self.total_sq = np.zeros(len(self.names), dtype=np.float64)
        self.cross_total = np.zeros(
            (len(self.names), len(self.names)), dtype=np.float64
        )
        self.minimum = np.full(len(self.names), np.inf, dtype=np.float64)
        self.maximum = np.full(len(self.names), -np.inf, dtype=np.float64)
        self.reservoir_size = int(reservoir_size)
        self.sample_stride = max(1, int(sample_stride))
        self.rng = np.random.default_rng(seed)
        self._blocks = []
        self._sample_count = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if values.size == 0:
            return
        if values.shape[1] != len(self.names):
            raise ValueError(
                f"Expected {len(self.names)} columns, got {values.shape[1]}"
            )
        finite = np.isfinite(values).all(axis=1)
        values = values[finite]
        if not len(values):
            return
        self.count += len(values)
        values64 = values.astype(np.float64)
        self.total += values64.sum(axis=0)
        self.total_sq += np.square(values64).sum(axis=0)
        self.cross_total += values64.T @ values64
        self.minimum = np.minimum(self.minimum, values64.min(axis=0))
        self.maximum = np.maximum(self.maximum, values64.max(axis=0))

        take = min(len(values), max(1, math.ceil(len(values) / self.sample_stride)))
        indices = self.rng.choice(len(values), size=take, replace=False)
        self._blocks.append(values[indices].copy())
        self._sample_count += take
        if self._sample_count > 4 * self.reservoir_size:
            self._compact()

    def _compact(self) -> None:
        if not self._blocks:
            return
        sample = np.concatenate(self._blocks, axis=0)
        if len(sample) > self.reservoir_size:
            indices = self.rng.choice(
                len(sample), size=self.reservoir_size, replace=False
            )
            sample = sample[indices]
        self._blocks = [sample]
        self._sample_count = len(sample)

    def sample(self) -> np.ndarray:
        self._compact()
        if not self._blocks:
            return np.empty((0, len(self.names)), dtype=np.float32)
        return self._blocks[0]

    def rows(self, base: Mapping[str, object]) -> list:
        sample = self.sample()
        rows = []
        for index, name in enumerate(self.names):
            if self.count:
                mean = self.total[index] / self.count
                variance = max(
                    self.total_sq[index] / self.count - mean * mean, 0.0
                )
                column = sample[:, index]
                quantiles = np.quantile(
                    column, (0.10, 0.25, 0.50, 0.75, 0.90)
                )
                minimum = self.minimum[index]
                maximum = self.maximum[index]
            else:
                mean = variance = minimum = maximum = np.nan
                quantiles = np.full(5, np.nan)
            rows.append(
                {
                    **base,
                    "metric": name,
                    "count": self.count,
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "min": minimum,
                    "p10": quantiles[0],
                    "p25": quantiles[1],
                    "p50": quantiles[2],
                    "p75": quantiles[3],
                    "p90": quantiles[4],
                    "max": maximum,
                    "quantile_method": (
                        f"deterministic sampled reservoir, max={self.reservoir_size}, "
                        f"stride={self.sample_stride}"
                    ),
                }
            )
        return rows

    def exact_moments(self, metric: str) -> Tuple[int, float, float]:
        index = self.names.index(metric)
        if not self.count:
            return 0, np.nan, np.nan
        mean = self.total[index] / self.count
        variance = max(self.total_sq[index] / self.count - mean * mean, 0.0)
        return self.count, mean, math.sqrt(variance)

    def correlation(self, metric_a: str, metric_b: str) -> float:
        if not self.count:
            return np.nan
        index_a = self.names.index(metric_a)
        index_b = self.names.index(metric_b)
        mean_a = self.total[index_a] / self.count
        mean_b = self.total[index_b] / self.count
        variance_a = max(
            self.total_sq[index_a] / self.count - mean_a * mean_a, 0.0
        )
        variance_b = max(
            self.total_sq[index_b] / self.count - mean_b * mean_b, 0.0
        )
        denominator = math.sqrt(variance_a * variance_b)
        if denominator == 0:
            return np.nan
        covariance = (
            self.cross_total[index_a, index_b] / self.count - mean_a * mean_b
        )
        return covariance / denominator


def new_stats(
    names: Sequence[str], args: argparse.Namespace, salt: int
) -> VectorStats:
    return VectorStats(
        names,
        reservoir_size=args.reservoir_size,
        sample_stride=args.sample_stride,
        seed=args.seed + salt,
    )


class OfficialMetricAccumulator:
    """Streaming implementation of tool.iouutils.scores for BCSS."""

    def __init__(self) -> None:
        self.hist = np.zeros((5, 5), dtype=np.float64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.int64)
        prediction = np.asarray(prediction, dtype=np.int64).copy()
        prediction[target == 4] = 4
        valid = (target >= 0) & (target < 5)
        encoded = 5 * target[valid] + prediction[valid]
        self.hist += np.bincount(encoded, minlength=25).reshape(5, 5)

    def result(self) -> dict:
        hist = self.hist.copy()
        hist[4, 4] = 0
        diagonal = np.diag(hist)
        row = hist.sum(axis=1)
        column = hist.sum(axis=0)
        denom = row + column - diagonal
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = diagonal[:4] / denom[:4]
            dice = 2.0 * diagonal[:4] / (row[:4] + column[:4])
            accuracy = diagonal.sum() / hist.sum()
            class_accuracy = np.nanmean(diagonal[:4] / row[:4])
            frequency = row[:4] / hist.sum()
        present = frequency > 0
        return {
            "Pixel Accuracy": accuracy,
            "Mean Accuracy": class_accuracy,
            "Frequency Weighted IoU": np.sum(frequency[present] * iou[present]),
            "Mean IoU": np.nanmean(iou),
            "Class IoU": {index: value for index, value in enumerate(iou)},
            "Dice Coefficients": {
                index: (value if np.isfinite(value) else 0.0)
                for index, value in enumerate(dice)
            },
            "Mean Dice": np.mean(np.nan_to_num(dice, nan=0.0)),
            "confusion_matrix_after_background_overwrite": hist,
        }


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(to_builtin(dict(row)))


def discover_samples(data_root: Path, max_images: int | None) -> list:
    image_dir = data_root / "img"
    mask_dir = data_root / "mask"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected validation img/ and mask/ under {data_root}"
        )
    extensions = {".png", ".jpg", ".jpeg"}
    samples = []
    for image_path in sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in extensions
    ):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        samples.append((image_path.stem, image_path, mask_path))
    if max_images is not None:
        samples = samples[:max_images]
    if not samples:
        raise RuntimeError(f"No validation samples found under {data_root}")
    return samples


def load_input(image_path: Path, img_size: int) -> Tuple[np.ndarray, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)
    if image.size != (img_size, img_size):
        image = TF.resize(
            image,
            [img_size, img_size],
            interpolation=InterpolationMode.BILINEAR,
        )
    tensor = TF.to_tensor(image)
    tensor = TF.normalize(
        tensor,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return original, tensor.unsqueeze(0)


def unwrap_state_dict(checkpoint) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                return checkpoint[key]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint)!r}")
    return checkpoint


def load_model(path: Path, context_mode: str, device: torch.device) -> Net_CAM:
    model = Net_CAM(n_class=4, rectifier_type="hfrm", context_mode=context_mode)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(checkpoint)
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict load unexpectedly incompatible: {incompatible}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def tensor_parameter_summary(tensor: torch.Tensor) -> dict:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "l2_norm": tensor.norm().item(),
        "mean": tensor.mean().item(),
        "std": tensor.std(unbiased=False).item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
        "nonzero_count": int(torch.count_nonzero(tensor).item()),
        "element_count": tensor.numel(),
    }


def fampr_parameter_audit(model: Net_CAM) -> dict:
    stages = {
        "stage1": model.hfrm_56,
        "stage2": model.hfrm_28_1,
        "stage3": model.hfrm_28_2,
    }
    output = {}
    for stage, hfrm in stages.items():
        context = hfrm.fampr_context
        output[stage] = {
            "band_predictor_final_weight": tensor_parameter_summary(
                context.frequency_selector.band_weight_network[-1].weight
            ),
            "band_predictor_final_bias": tensor_parameter_summary(
                context.frequency_selector.band_weight_network[-1].bias
            ),
            "kernel_gate_final_weight": tensor_parameter_summary(
                context.adaptive_kernel.gate_network[-1].weight
            ),
            "kernel_gate_final_bias": tensor_parameter_summary(
                context.adaptive_kernel.gate_network[-1].bias
            ),
            "base_kernel": tensor_parameter_summary(
                context.adaptive_kernel.base_kernel
            ),
            "anchor_logit": context.anchor_logit.detach().float().item(),
            "anchor_lambda_fp32": context.anchor_lambda.detach().float().item(),
            "gamma_context_fp32": hfrm.gamma_context.detach().float().item(),
        }
    return output


def amp_dtype_from_name(name: str):
    return {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def normalize_cams(cams: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    normalized = {}
    for stage, cam in cams.items():
        minimum = cam.min(axis=(1, 2), keepdims=True)
        maximum = cam.max(axis=(1, 2), keepdims=True)
        normalized[stage] = (cam - minimum) / (maximum - minimum + EPS)
    return normalized


def presence_label(probability: np.ndarray) -> np.ndarray:
    label = (probability > BCSS_THRESHOLDS).astype(np.float32)
    if label.sum() == 0:
        label[int(np.argmax(probability))] = 1.0
    return label


def predict_from_cams(
    normalized_cams: Mapping[str, np.ndarray], label: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    fusion = sum(
        OFFICIAL_CAM_WEIGHTS[index] * normalized_cams[stage]
        for index, stage in enumerate(STAGES)
    )
    gated = fusion * label.reshape(4, 1, 1)
    return np.argmax(gated, axis=0).astype(np.uint8), fusion


def stage_predictions(
    normalized_cams: Mapping[str, np.ndarray], label: np.ndarray
) -> Dict[str, np.ndarray]:
    return {
        stage: np.argmax(
            normalized_cams[stage] * label.reshape(4, 1, 1), axis=0
        ).astype(np.uint8)
        for stage in STAGES
    }


def resize_and_unflip(
    tensor: torch.Tensor, size: Tuple[int, int], flip_dims: Tuple[int, ...]
) -> np.ndarray:
    tensor = F.interpolate(
        tensor, size=size, mode="bilinear", align_corners=False
    )[0]
    if flip_dims:
        tensor = torch.flip(tensor, dims=flip_dims)
    return tensor.detach().float().cpu().numpy()


def diagnostic_maps(
    diagnostics: Mapping[str, object],
    size: Tuple[int, int],
    flip_dims: Tuple[int, ...],
) -> Dict[str, Dict[str, np.ndarray]]:
    output = {}
    for stage in FAMPR_STAGES:
        hfrm = diagnostics["hfrm_stages"][stage]
        diag = diagnostics["fampr"][stage]
        band_energy = diag["band_energy"]
        high_frequency_ratio = (
            band_energy[:, 0:1] + band_energy[:, 1:2]
        ) / (band_energy.sum(dim=1, keepdim=True) + EPS)
        original_ch = diag["original_ch"]
        denominator = torch.linalg.vector_norm(
            original_ch.float(), dim=1, keepdim=True
        ).clamp_min(EPS)
        r_fa = torch.linalg.vector_norm(
            (diag["adaptive_context"] - original_ch).float(),
            dim=1,
            keepdim=True,
        ) / denominator
        r_mpr = torch.linalg.vector_norm(
            (diag["fampr_context"] - original_ch).float(),
            dim=1,
            keepdim=True,
        ) / denominator
        gate_low_values = diag["kernel_gate_low"].float().reshape(-1)
        gate_high_values = diag["kernel_gate_high"].float().reshape(-1)
        gate_low = gate_low_values.mean().item()
        gate_high = gate_high_values.mean().item()
        gate_low_std = gate_low_values.std(unbiased=False).item()
        gate_high_std = gate_high_values.std(unbiased=False).item()
        scalar_maps = {
            "g_low": torch.full_like(diag["morphology_map"], gate_low),
            "g_high": torch.full_like(diag["morphology_map"], gate_high),
            "g_high_over_low": torch.full_like(
                diag["morphology_map"], gate_high / (gate_low + EPS)
            ),
            "g_low_channel_std": torch.full_like(
                diag["morphology_map"], gate_low_std
            ),
            "g_high_channel_std": torch.full_like(
                diag["morphology_map"], gate_high_std
            ),
            "anchor_lambda": torch.full_like(
                diag["morphology_map"], diag["anchor_lambda"].float().item()
            ),
            "gamma_context": torch.full_like(
                diag["morphology_map"], hfrm["gamma_context"].float().item()
            ),
        }
        maps = {
            "morphology": diag["morphology_map"],
            "dilation": diag["dilation_map"],
            "high_frequency_ratio": high_frequency_ratio,
            "r_fa": r_fa,
            "r_mpr": r_mpr,
            **scalar_maps,
        }
        for band in range(4):
            maps[f"band_weight_{band}"] = diag["band_weights"][:, band : band + 1]
        output[stage] = {
            name: resize_and_unflip(value, size, flip_dims)[0]
            for name, value in maps.items()
        }
    return output


@torch.no_grad()
def run_model(
    model: Net_CAM,
    image_tensor: torch.Tensor,
    original_size: Tuple[int, int],
    device: torch.device,
    amp_dtype,
    include_diagnostics: bool,
) -> dict:
    image_tensor = image_tensor.to(device, non_blocking=True)
    tta_cams = {stage: [] for stage in STAGES}
    probabilities = []
    tta_diagnostics = {
        stage: {metric: [] for metric in DIAGNOSTIC_METRICS}
        for stage in FAMPR_STAGES
    }
    for input_flip_dims, cam_flip_dims in TTA_TRANSFORMS:
        tta_input = (
            torch.flip(image_tensor, dims=input_flip_dims)
            if input_flip_dims
            else image_tensor
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            if include_diagnostics:
                outputs, diagnostics = model.forward_with_diagnostics(tta_input)
                raw_cams = outputs[5:9]
                probability = outputs[4]
            else:
                *raw_cams, probability = model.forward_cam(tta_input)
                diagnostics = None
            cams = [F.relu(cam) for cam in raw_cams]
            for stage, cam in zip(STAGES, cams):
                tta_cams[stage].append(
                    resize_and_unflip(cam, original_size, cam_flip_dims)
                )
            probabilities.append(probability.detach().float().cpu().numpy()[0])
            if include_diagnostics:
                maps = diagnostic_maps(diagnostics, original_size, cam_flip_dims)
                for stage in FAMPR_STAGES:
                    for metric in DIAGNOSTIC_METRICS:
                        tta_diagnostics[stage][metric].append(maps[stage][metric])

    cams = {
        stage: np.mean(np.stack(values, axis=0), axis=0)
        for stage, values in tta_cams.items()
    }
    normalized = normalize_cams(cams)
    probability = np.mean(np.stack(probabilities, axis=0), axis=0)
    label = presence_label(probability)
    prediction, fusion = predict_from_cams(normalized, label)
    result = {
        "cams": cams,
        "normalized_cams": normalized,
        "probability": probability,
        "presence_label": label,
        "prediction": prediction,
        "fusion": fusion,
        "stage_predictions": stage_predictions(normalized, label),
    }
    if include_diagnostics:
        result["diagnostics"] = {
            stage: {
                metric: np.mean(np.stack(values, axis=0), axis=0)
                for metric, values in metrics.items()
            }
            for stage, metrics in tta_diagnostics.items()
        }
    return result


@torch.no_grad()
def verify_fampr_diagnostic_interface(
    model: Net_CAM,
    image_tensor: torch.Tensor,
    device: torch.device,
    amp_dtype,
) -> dict:
    image_tensor = image_tensor.to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_dtype is not None,
    ):
        official = model.forward_cam(image_tensor)
        outputs, diagnostics = model.forward_with_diagnostics(image_tensor)
        diagnostic = tuple(F.relu(cam) for cam in outputs[5:9]) + (outputs[4],)
    differences = [
        (left.detach().float() - right.detach().float()).abs().max().item()
        for left, right in zip(official, diagnostic)
    ]
    return {
        "official_interface": "Net_CAM.forward_cam",
        "analysis_interface": "Net_CAM.forward_with_diagnostics",
        "tensor_order": ["cam56", "cam28_1", "cam28_2", "camdeep", "classification_probability"],
        "max_abs_differences": differences,
        "allclose_atol_1e-6_rtol_1e-6": all(
            torch.allclose(left.float(), right.float(), atol=1e-6, rtol=1e-6)
            for left, right in zip(official, diagnostic)
        ),
        "diagnostic_stages": sorted(diagnostics["fampr"].keys()),
    }


def transition_group_masks(
    target: np.ndarray, a0: np.ndarray, fampr: np.ndarray
) -> Dict[str, np.ndarray]:
    a0_correct = a0 == target
    fampr_correct = fampr == target
    return {
        "BOTH_CORRECT": a0_correct & fampr_correct,
        "CORRECTED_BY_FAMPR": (~a0_correct) & fampr_correct,
        "HARMED_BY_FAMPR": a0_correct & (~fampr_correct),
        "BOTH_WRONG": (~a0_correct) & (~fampr_correct),
    }


def boundary_mask(target: np.ndarray, radius: int) -> np.ndarray:
    boundary = np.zeros_like(target, dtype=bool)
    boundary[1:, :] |= target[1:, :] != target[:-1, :]
    boundary[:-1, :] |= target[:-1, :] != target[1:, :]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, :-1] != target[:, 1:]
    if radius > 0:
        boundary = morphology_dilation(boundary, footprint=disk(radius))
    return np.asarray(boundary, dtype=bool)


def precompute_component_thresholds(samples: Sequence[tuple]) -> dict:
    areas = {class_id: [] for class_id in range(4)}
    structure = np.ones((3, 3), dtype=np.uint8)
    for _, _, mask_path in samples:
        target = np.asarray(Image.open(mask_path), dtype=np.uint8)
        for class_id in range(4):
            labels, count = ndimage.label(target == class_id, structure=structure)
            if count:
                component_areas = np.bincount(labels.ravel())[1:]
                areas[class_id].extend(component_areas.tolist())
    thresholds = {}
    for class_id, values in areas.items():
        array = np.asarray(values, dtype=np.float64)
        thresholds[class_id] = {
            "component_count": len(array),
            "q25": float(np.quantile(array, 0.25)) if len(array) else np.nan,
            "q75": float(np.quantile(array, 0.75)) if len(array) else np.nan,
        }
    return thresholds


def component_size(area: int, threshold: Mapping[str, float]) -> str:
    if area <= threshold["q25"]:
        return "small"
    if area <= threshold["q75"]:
        return "medium"
    return "large"


def update_components(
    target: np.ndarray,
    a0: np.ndarray,
    fampr: np.ndarray,
    thresholds: Mapping[int, Mapping[str, float]],
    aggregate: MutableMapping[tuple, dict],
) -> None:
    structure = np.ones((3, 3), dtype=np.uint8)
    for class_id in range(4):
        labels, count = ndimage.label(target == class_id, structure=structure)
        for component_id in range(1, count + 1):
            mask = labels == component_id
            area = int(mask.sum())
            size = component_size(area, thresholds[class_id])
            record = aggregate.setdefault(
                (class_id, size),
                {
                    "component_count": 0,
                    "pixel_count": 0,
                    "a0_correct": 0,
                    "fampr_correct": 0,
                    "corrected": 0,
                    "harmed": 0,
                },
            )
            a0_correct = a0[mask] == class_id
            fampr_correct = fampr[mask] == class_id
            record["component_count"] += 1
            record["pixel_count"] += area
            record["a0_correct"] += int(a0_correct.sum())
            record["fampr_correct"] += int(fampr_correct.sum())
            record["corrected"] += int(((~a0_correct) & fampr_correct).sum())
            record["harmed"] += int((a0_correct & (~fampr_correct)).sum())


def diagnostic_matrix(maps: Mapping[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    return np.column_stack([maps[name][mask] for name in DIAGNOSTIC_METRICS])


def effect_statistics(
    corrected: VectorStats, harmed: VectorStats, metric: str
) -> dict:
    n1, mean1, std1 = corrected.exact_moments(metric)
    n2, mean2, std2 = harmed.exact_moments(metric)
    denominator = max(n1 + n2 - 2, 1)
    pooled = math.sqrt(
        max(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / denominator, 0.0)
    )
    cohen_d = (mean1 - mean2) / pooled if pooled > 0 else np.nan
    index = corrected.names.index(metric)
    sample1 = corrected.sample()[:, index]
    sample2 = harmed.sample()[:, index]
    if len(sample1) and len(sample2):
        welch = stats.ttest_ind(sample1, sample2, equal_var=False)
        mann = stats.mannwhitneyu(sample1, sample2, alternative="two-sided")
        p_welch = float(welch.pvalue)
        p_mann = float(mann.pvalue)
    else:
        p_welch = p_mann = np.nan
    return {
        "corrected_minus_harmed_mean": mean1 - mean2,
        "cohen_d_exact_moments": cohen_d,
        "welch_p_on_reservoir": p_welch,
        "mann_whitney_p_on_reservoir": p_mann,
        "statistical_warning": (
            "exploratory only; reservoir approximation and spatially correlated pixels"
        ),
    }


def categorical_rgb(mask: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        (
            (213, 62, 79),
            (27, 158, 119),
            (55, 126, 184),
            (230, 159, 0),
            (245, 245, 245),
        ),
        dtype=np.uint8,
    )
    clipped = np.clip(mask.astype(np.int64), 0, 4)
    return palette[clipped]


def save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def render_case(
    name: str,
    original: np.ndarray,
    target: np.ndarray,
    a0_result: Mapping[str, object],
    fampr_result: Mapping[str, object],
    output_path: Path,
) -> None:
    a0 = a0_result["prediction"]
    fampr = fampr_result["prediction"]
    groups = transition_group_masks(target, a0, fampr)
    corrected = groups["CORRECTED_BY_FAMPR"] & (target < 4)
    harmed = groups["HARMED_BY_FAMPR"] & (target < 4)
    fig, axes = plt.subplots(4, 6, figsize=(15, 10), constrained_layout=True)
    row0 = (
        (original, "Input"),
        (categorical_rgb(target), "GT"),
        (categorical_rgb(a0), "A0 prediction"),
        (categorical_rgb(fampr), "FA-MPR prediction"),
        (corrected, "Corrected pixels"),
        (harmed, "Harmed pixels"),
    )
    for axis, (image, title) in zip(axes[0], row0):
        axis.imshow(image, cmap="Greens" if title.startswith("Corrected") else "Reds")
        axis.set_title(title)
        axis.axis("off")
    for row, stage in enumerate(FAMPR_STAGES, start=1):
        maps = fampr_result["diagnostics"][stage]
        cam_stage = FAMPR_TO_CAM[stage]
        cam_difference = np.abs(
            fampr_result["normalized_cams"][cam_stage]
            - a0_result["normalized_cams"][cam_stage]
        ).mean(axis=0)
        panels = (
            (maps["morphology"], "M", "viridis", 0.0, 1.0),
            (maps["dilation"], "D", "cividis", 1.0, 7.0),
            (maps["high_frequency_ratio"], "HF ratio", "magma", 0.0, 1.0),
            (maps["r_fa"], "R_FA", "magma", 0.0, None),
            (maps["r_mpr"], "R_MPR", "magma", 0.0, None),
            (cam_difference, "|CAM_FA-CAM_A0|", "magma", 0.0, None),
        )
        for axis, (image, title, cmap, vmin, vmax) in zip(axes[row], panels):
            artist = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(f"{stage}: {title}")
            axis.axis("off")
            fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.02)
    fig.suptitle(name, fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_options = {"dpi": 120, "bbox_inches": "tight"}
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_options["pil_kwargs"] = {"quality": 88, "optimize": True}
    fig.savefig(output_path, **save_options)
    plt.close(fig)


def plot_aggregates(
    visual_dir: Path,
    transition_matrix: np.ndarray,
    transition_group_counts: np.ndarray,
    class_totals: np.ndarray,
    corrected_harmed: Mapping[tuple, VectorStats],
    boundary_counts: Mapping[tuple, dict],
    component_rows: Sequence[Mapping[str, object]],
    stage_results: Mapping[tuple, dict],
) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    for class_id, axis in enumerate(axes.ravel()):
        matrix = transition_matrix[class_id].astype(float)
        matrix /= max(matrix.sum(), 1.0)
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), EPS))
        for row in range(4):
            for column in range(4):
                axis.text(column, row, f"{100 * matrix[row, column]:.1f}%", ha="center", va="center", fontsize=8)
        axis.set_title(f"GT {class_id}: {CLASS_NAMES[class_id]}")
        axis.set_xlabel("FA-MPR prediction")
        axis.set_ylabel("A0 prediction")
        axis.set_xticks(range(4))
        axis.set_yticks(range(4))
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_figure(fig, visual_dir / "prediction_transition_by_gt_class")

    corrected = transition_group_counts[:, 1] / np.maximum(class_totals, 1)
    harmed = transition_group_counts[:, 2] / np.maximum(class_totals, 1)
    net = corrected - harmed
    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    colors = np.where(net >= 0, "#1b9e77", "#d95f02")
    axis.bar(np.arange(4), 100 * net, color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(4), [f"{i}: {name}" for i, name in enumerate(CLASS_NAMES)])
    axis.set_ylabel("Net correction rate (percentage points)")
    axis.set_title("FA-MPR pixel correction minus harm by GT class")
    save_figure(fig, visual_dir / "net_correction_by_class")

    columns = [(stage, class_id) for stage in FAMPR_STAGES for class_id in range(4)]
    effect = np.full((len(KEY_METRICS), len(columns)), np.nan)
    for column, (stage, class_id) in enumerate(columns):
        corrected_stats = corrected_harmed.get((stage, class_id, "CORRECTED_BY_FAMPR"))
        harmed_stats = corrected_harmed.get((stage, class_id, "HARMED_BY_FAMPR"))
        if corrected_stats is None or harmed_stats is None:
            continue
        for row, metric in enumerate(KEY_METRICS):
            effect[row, column] = effect_statistics(corrected_stats, harmed_stats, metric)["cohen_d_exact_moments"]
    fig, axis = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
    bound = np.nanmax(np.abs(effect)) if np.isfinite(effect).any() else 1.0
    bound = max(bound, 0.1)
    image = axis.imshow(effect, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_yticks(range(len(KEY_METRICS)), KEY_METRICS)
    axis.set_xticks(range(len(columns)), [f"{stage}\nC{class_id}" for stage, class_id in columns], rotation=45, ha="right")
    axis.set_title("Corrected minus harmed standardized effect (Cohen d)")
    fig.colorbar(image, ax=axis, label="Cohen d")
    save_figure(fig, visual_dir / "corrected_vs_harmed_effects")

    zones = ("boundary", "interior")
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    x = np.arange(4)
    width = 0.36
    for zone_index, zone in enumerate(zones):
        values = []
        for class_id in range(4):
            record = boundary_counts.get((class_id, zone), {})
            pixels = max(record.get("pixels", 0), 1)
            values.append(100 * (record.get("corrected", 0) - record.get("harmed", 0)) / pixels)
        axis.bar(x + (zone_index - 0.5) * width, values, width, label=zone)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, [f"C{i}: {name}" for i, name in enumerate(CLASS_NAMES)])
    axis.set_ylabel("Net correction rate (percentage points)")
    axis.legend(frameon=False)
    axis.set_title("Boundary versus interior response")
    save_figure(fig, visual_dir / "boundary_interior_net_correction")

    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    markers = {"small": "o", "medium": "s", "large": "^"}
    for size, marker in markers.items():
        rows = {int(row["gt_class"]): row for row in component_rows if row["size_group"] == size}
        values = [100 * float(rows[class_id]["net_correction_rate"]) if class_id in rows else np.nan for class_id in range(4)]
        axis.plot(range(4), values, marker=marker, label=size)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(4), [f"C{i}: {name}" for i, name in enumerate(CLASS_NAMES)])
    axis.set_ylabel("Net correction rate (percentage points)")
    axis.legend(frameon=False)
    axis.set_title("GT connected-component size response")
    save_figure(fig, visual_dir / "component_size_net_correction")

    delta = np.full((4, 4), np.nan)
    for stage_index, stage in enumerate(STAGES):
        a0 = stage_results[("a0", stage)]["Class IoU"]
        fampr = stage_results[("fampr", stage)]["Class IoU"]
        for class_id in range(4):
            delta[class_id, stage_index] = 100 * (fampr[class_id] - a0[class_id])
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bound = max(np.nanmax(np.abs(delta)), 0.1)
    image = axis.imshow(delta, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for row in range(4):
        for column in range(4):
            axis.text(column, row, f"{delta[row, column]:+.2f}", ha="center", va="center")
    axis.set_xticks(range(4), STAGES)
    axis.set_yticks(range(4), [f"C{i}: {name}" for i, name in enumerate(CLASS_NAMES)])
    axis.set_title("FA-MPR minus A0 stagewise IoU (percentage points)")
    fig.colorbar(image, ax=axis, label="IoU delta (pp)")
    save_figure(fig, visual_dir / "stagewise_iou_delta")


def main() -> None:
    args = parse_args()
    if args.boundary_radius not in (2, 3):
        raise ValueError("Boundary radius is frozen to approximately 2-3 pixels")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    if device.type != "cuda" and amp_dtype is not None:
        raise ValueError("AMP diagnosis is only supported on CUDA")

    output_dir = args.output_dir.resolve()
    visual_dir = output_dir / "fampr_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.data_root, args.max_images)
    component_thresholds = precompute_component_thresholds(samples)

    print(f"Loading frozen A0: {args.a0_checkpoint}", flush=True)
    a0_model = load_model(args.a0_checkpoint, "ch", device)
    print(f"Loading frozen FA-MPR: {args.fampr_checkpoint}", flush=True)
    fampr_model = load_model(args.fampr_checkpoint, "fampr", device)
    if a0_model.training or fampr_model.training:
        raise RuntimeError("Both frozen models must remain in eval mode")
    _, verification_tensor = load_input(samples[0][1], args.img_size)
    interface_verification = verify_fampr_diagnostic_interface(
        fampr_model, verification_tensor, device, amp_dtype
    )
    if not interface_verification["allclose_atol_1e-6_rtol_1e-6"]:
        raise RuntimeError(
            "forward_with_diagnostics does not reproduce official forward_cam"
        )
    parameter_audit = fampr_parameter_audit(fampr_model)

    final_metrics = {name: OfficialMetricAccumulator() for name in ("a0", "fampr")}
    stage_metrics = {
        (model, stage): OfficialMetricAccumulator()
        for model in ("a0", "fampr")
        for stage in STAGES
    }
    transition_matrix = np.zeros((4, 4, 4), dtype=np.int64)
    transition_group_counts = np.zeros((4, 4), dtype=np.int64)
    class_totals = np.zeros(4, dtype=np.int64)
    class_stats = {}
    corrected_harmed_stats = {}
    boundary_diagnostic_stats = {}
    boundary_counts = {}
    component_aggregate = {}
    cam_difference_stats = {}
    image_records = []
    salt = 0

    for index, (name, image_path, mask_path) in enumerate(samples, start=1):
        original, image_tensor = load_input(image_path, args.img_size)
        target = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if target.shape != original.shape[:2]:
            raise ValueError(f"Image/mask shape mismatch for {name}")
        original_size = target.shape
        a0 = run_model(a0_model, image_tensor, original_size, device, amp_dtype, False)
        fampr = run_model(fampr_model, image_tensor, original_size, device, amp_dtype, True)
        if not all(np.isfinite(cam).all() for cam in fampr["cams"].values()):
            raise RuntimeError(f"Non-finite CAM detected for {name}")

        final_metrics["a0"].update(target, a0["prediction"])
        final_metrics["fampr"].update(target, fampr["prediction"])
        for stage in STAGES:
            stage_metrics[("a0", stage)].update(target, a0["stage_predictions"][stage])
            stage_metrics[("fampr", stage)].update(target, fampr["stage_predictions"][stage])

        foreground = target < 4
        groups = transition_group_masks(target, a0["prediction"], fampr["prediction"])
        image_record = {
            "image": name,
            "foreground_pixels": int(foreground.sum()),
            "corrected": int((groups["CORRECTED_BY_FAMPR"] & foreground).sum()),
            "harmed": int((groups["HARMED_BY_FAMPR"] & foreground).sum()),
        }
        image_record["net_correction"] = image_record["corrected"] - image_record["harmed"]
        for class_id in range(4):
            class_mask = target == class_id
            class_totals[class_id] += int(class_mask.sum())
            for group_index, group in enumerate(TRANSITION_GROUPS):
                transition_group_counts[class_id, group_index] += int((class_mask & groups[group]).sum())
            encoded = 4 * a0["prediction"][class_mask].astype(np.int64) + fampr["prediction"][class_mask].astype(np.int64)
            transition_matrix[class_id] += np.bincount(encoded, minlength=16).reshape(4, 4)
            corrected_count = int((class_mask & groups["CORRECTED_BY_FAMPR"]).sum())
            harmed_count = int((class_mask & groups["HARMED_BY_FAMPR"]).sum())
            image_record[f"class{class_id}_corrected"] = corrected_count
            image_record[f"class{class_id}_harmed"] = harmed_count
            image_record[f"class{class_id}_net"] = corrected_count - harmed_count

            for stage in FAMPR_STAGES:
                maps = fampr["diagnostics"][stage]
                key = (stage, class_id)
                if key not in class_stats:
                    salt += 1
                    class_stats[key] = new_stats(DIAGNOSTIC_METRICS, args, salt)
                class_stats[key].update(diagnostic_matrix(maps, class_mask))
                for group in ("CORRECTED_BY_FAMPR", "HARMED_BY_FAMPR"):
                    group_mask = class_mask & groups[group]
                    group_key = (stage, class_id, group)
                    if group_key not in corrected_harmed_stats:
                        salt += 1
                        corrected_harmed_stats[group_key] = new_stats(DIAGNOSTIC_METRICS, args, salt)
                    corrected_harmed_stats[group_key].update(diagnostic_matrix(maps, group_mask))

            for stage in STAGES:
                difference = np.abs(fampr["normalized_cams"][stage] - a0["normalized_cams"][stage]).mean(axis=0)
                key = (stage, class_id)
                if key not in cam_difference_stats:
                    salt += 1
                    cam_difference_stats[key] = new_stats(("normalized_cam_abs_difference",), args, salt)
                cam_difference_stats[key].update(difference[class_mask])

        boundary = boundary_mask(target, args.boundary_radius)
        for class_id in range(4):
            class_mask = target == class_id
            for zone, zone_mask in (("boundary", boundary), ("interior", ~boundary)):
                mask = class_mask & zone_mask
                key = (class_id, zone)
                record = boundary_counts.setdefault(
                    key,
                    {"pixels": 0, "a0_correct": 0, "fampr_correct": 0, "corrected": 0, "harmed": 0},
                )
                record["pixels"] += int(mask.sum())
                record["a0_correct"] += int(((a0["prediction"] == class_id) & mask).sum())
                record["fampr_correct"] += int(((fampr["prediction"] == class_id) & mask).sum())
                record["corrected"] += int((groups["CORRECTED_BY_FAMPR"] & mask).sum())
                record["harmed"] += int((groups["HARMED_BY_FAMPR"] & mask).sum())
                for stage in FAMPR_STAGES:
                    diag_key = (stage, class_id, zone)
                    if diag_key not in boundary_diagnostic_stats:
                        salt += 1
                        boundary_diagnostic_stats[diag_key] = new_stats(KEY_METRICS, args, salt)
                    maps = fampr["diagnostics"][stage]
                    values = np.column_stack([maps[metric][mask] for metric in KEY_METRICS])
                    boundary_diagnostic_stats[diag_key].update(values)

        update_components(target, a0["prediction"], fampr["prediction"], component_thresholds, component_aggregate)
        image_records.append(image_record)
        if index % args.progress_every == 0 or index == len(samples):
            print(f"Analyzed {index}/{len(samples)} validation images", flush=True)

    final_results = {name: metric.result() for name, metric in final_metrics.items()}
    stage_results = {key: metric.result() for key, metric in stage_metrics.items()}

    transition_rows = []
    for class_id in range(4):
        total = max(int(class_totals[class_id]), 1)
        corrected = int(transition_group_counts[class_id, 1])
        harmed = int(transition_group_counts[class_id, 2])
        for group_index, group in enumerate(TRANSITION_GROUPS):
            count = int(transition_group_counts[class_id, group_index])
            transition_rows.append(
                {
                    "record_type": "correctness_group",
                    "gt_class": class_id,
                    "gt_class_name": CLASS_NAMES[class_id],
                    "group": group,
                    "a0_prediction": "",
                    "fampr_prediction": "",
                    "count": count,
                    "fraction_of_gt_class": count / total,
                    "corrected_rate": corrected / total,
                    "harmed_rate": harmed / total,
                    "net_correction_rate": (corrected - harmed) / total,
                }
            )
        for a0_prediction in range(4):
            for fampr_prediction in range(4):
                count = int(transition_matrix[class_id, a0_prediction, fampr_prediction])
                transition_rows.append(
                    {
                        "record_type": "prediction_transition",
                        "gt_class": class_id,
                        "gt_class_name": CLASS_NAMES[class_id],
                        "group": "",
                        "a0_prediction": a0_prediction,
                        "fampr_prediction": fampr_prediction,
                        "count": count,
                        "fraction_of_gt_class": count / total,
                        "corrected_rate": corrected / total,
                        "harmed_rate": harmed / total,
                        "net_correction_rate": (corrected - harmed) / total,
                    }
                )
    write_csv(output_dir / "fampr_pixel_transition_stats.csv", transition_rows)

    class_rows = []
    for (stage, class_id), accumulator in sorted(class_stats.items()):
        correlations = {
            "correlation_with_morphology": {
                metric: accumulator.correlation("morphology", metric)
                for metric in DIAGNOSTIC_METRICS
            },
            "correlation_with_high_frequency_ratio": {
                metric: accumulator.correlation("high_frequency_ratio", metric)
                for metric in DIAGNOSTIC_METRICS
            },
        }
        for row in accumulator.rows(
            {
                "stage": stage,
                "gt_class": class_id,
                "gt_class_name": CLASS_NAMES[class_id],
            }
        ):
            row["correlation_with_morphology"] = correlations[
                "correlation_with_morphology"
            ][row["metric"]]
            row["correlation_with_high_frequency_ratio"] = correlations[
                "correlation_with_high_frequency_ratio"
            ][row["metric"]]
            class_rows.append(row)
    write_csv(output_dir / "fampr_class_conditioned_stats.csv", class_rows)

    corrected_harmed_rows = []
    for stage in FAMPR_STAGES:
        for class_id in range(4):
            corrected_stats = corrected_harmed_stats[(stage, class_id, "CORRECTED_BY_FAMPR")]
            harmed_stats = corrected_harmed_stats[(stage, class_id, "HARMED_BY_FAMPR")]
            effects = {metric: effect_statistics(corrected_stats, harmed_stats, metric) for metric in DIAGNOSTIC_METRICS}
            for group, accumulator in (("CORRECTED_BY_FAMPR", corrected_stats), ("HARMED_BY_FAMPR", harmed_stats)):
                for row in accumulator.rows({"stage": stage, "gt_class": class_id, "gt_class_name": CLASS_NAMES[class_id], "group": group}):
                    row.update(effects[row["metric"]])
                    corrected_harmed_rows.append(row)
    write_csv(output_dir / "fampr_corrected_vs_harmed_stats.csv", corrected_harmed_rows)

    boundary_rows = []
    for class_id in range(4):
        for zone in ("boundary", "interior"):
            record = boundary_counts[(class_id, zone)]
            pixels = max(record["pixels"], 1)
            for stage in FAMPR_STAGES:
                for row in boundary_diagnostic_stats[(stage, class_id, zone)].rows(
                    {"stage": stage, "gt_class": class_id, "gt_class_name": CLASS_NAMES[class_id], "zone": zone}
                ):
                    row.update(
                        {
                            "zone_pixels": record["pixels"],
                            "a0_accuracy": record["a0_correct"] / pixels,
                            "fampr_accuracy": record["fampr_correct"] / pixels,
                            "corrected_rate": record["corrected"] / pixels,
                            "harmed_rate": record["harmed"] / pixels,
                            "net_correction_rate": (record["corrected"] - record["harmed"]) / pixels,
                        }
                    )
                    boundary_rows.append(row)
    write_csv(output_dir / "fampr_boundary_interior_stats.csv", boundary_rows)

    component_rows = []
    for class_id in range(4):
        for size in ("small", "medium", "large"):
            record = component_aggregate.get((class_id, size), {})
            pixels = max(record.get("pixel_count", 0), 1)
            component_rows.append(
                {
                    "gt_class": class_id,
                    "gt_class_name": CLASS_NAMES[class_id],
                    "size_group": size,
                    "q25_area": component_thresholds[class_id]["q25"],
                    "q75_area": component_thresholds[class_id]["q75"],
                    "component_count": record.get("component_count", 0),
                    "pixel_count": record.get("pixel_count", 0),
                    "a0_recall": record.get("a0_correct", 0) / pixels,
                    "fampr_recall": record.get("fampr_correct", 0) / pixels,
                    "corrected_rate": record.get("corrected", 0) / pixels,
                    "harmed_rate": record.get("harmed", 0) / pixels,
                    "net_correction_rate": (record.get("corrected", 0) - record.get("harmed", 0)) / pixels,
                }
            )
    write_csv(output_dir / "fampr_component_size_stats.csv", component_rows)

    stage_rows = []
    for stage in STAGES:
        for model_name in ("a0", "fampr"):
            result = stage_results[(model_name, stage)]
            stage_rows.append(
                {
                    "record_type": "overall",
                    "model": model_name,
                    "stage": stage,
                    "gt_class": "",
                    "gt_class_name": "",
                    "mIoU": result["Mean IoU"],
                    "mDice": result["Mean Dice"],
                    "class_IoU": "",
                    "class_Dice": "",
                    "normalized_cam_abs_difference_mean": "",
                }
            )
            for class_id in range(4):
                difference_row = cam_difference_stats[(stage, class_id)].rows({})[0]
                stage_rows.append(
                    {
                        "record_type": "per_class",
                        "model": model_name,
                        "stage": stage,
                        "gt_class": class_id,
                        "gt_class_name": CLASS_NAMES[class_id],
                        "mIoU": result["Mean IoU"],
                        "mDice": result["Mean Dice"],
                        "class_IoU": result["Class IoU"][class_id],
                        "class_Dice": result["Dice Coefficients"][class_id],
                        "normalized_cam_abs_difference_mean": difference_row["mean"],
                    }
                )
    write_csv(output_dir / "fampr_stagewise_cam_stats.csv", stage_rows)

    image_records = sorted(image_records, key=lambda row: row["image"])
    write_csv(output_dir / "fampr_image_level_selection_scores.csv", image_records)
    rankings = {
        "most_improved": sorted(image_records, key=lambda row: (-row["net_correction"], row["image"]))[: args.visual_top_k],
        "most_harmed": sorted(image_records, key=lambda row: (row["net_correction"], row["image"]))[: args.visual_top_k],
        "class3_improvement": sorted(image_records, key=lambda row: (-row["class3_net"], row["image"]))[: args.visual_top_k],
        "class01_degradation": sorted(image_records, key=lambda row: (row["class0_net"] + row["class1_net"], row["image"]))[: args.visual_top_k],
    }
    ranking_rows = [
        {"ranking": ranking, "rank": rank, **record}
        for ranking, records in rankings.items()
        for rank, record in enumerate(records, start=1)
    ]
    write_csv(output_dir / "fampr_visualization_selection.csv", ranking_rows)

    if not args.skip_visualizations:
        plot_aggregates(
            visual_dir,
            transition_matrix,
            transition_group_counts,
            class_totals,
            corrected_harmed_stats,
            boundary_counts,
            component_rows,
            stage_results,
        )
        sample_by_name = {name: (image_path, mask_path) for name, image_path, mask_path in samples}
        unique_dir = visual_dir / "selected_cases"
        rendered = {}
        for ranking, records in rankings.items():
            for rank, record in enumerate(records, start=1):
                name = record["image"]
                if name not in rendered:
                    image_path, mask_path = sample_by_name[name]
                    original, image_tensor = load_input(image_path, args.img_size)
                    target = np.asarray(Image.open(mask_path), dtype=np.uint8)
                    a0 = run_model(a0_model, image_tensor, target.shape, device, amp_dtype, False)
                    fampr = run_model(fampr_model, image_tensor, target.shape, device, amp_dtype, True)
                    unique_path = unique_dir / f"{name}.png"
                    render_case(name, original, target, a0, fampr, unique_path)
                    rendered[name] = unique_path

    summary = {
        "analysis_scope": {
            "dataset": "BCSS validation",
            "sample_count": len(samples),
            "test_set_used": False,
            "training_performed": False,
            "model_or_hyperparameter_changes": False,
            "img_size": args.img_size,
            "tta": "original + horizontal flip + vertical flip",
            "amp_dtype": args.amp_dtype,
            "presence_thresholds": BCSS_THRESHOLDS,
            "cam_weights": OFFICIAL_CAM_WEIGHTS,
            "boundary_radius_pixels": args.boundary_radius,
            "component_size_definition": "per-class validation q25/q75 of 8-connected GT component area",
        },
        "checkpoints": {
            "a0": {
                "path": args.a0_checkpoint,
                "size_bytes": args.a0_checkpoint.stat().st_size,
                "sha256": sha256_file(args.a0_checkpoint),
                "context_mode": "ch",
            },
            "fampr": {
                "path": args.fampr_checkpoint,
                "size_bytes": args.fampr_checkpoint.stat().st_size,
                "sha256": sha256_file(args.fampr_checkpoint),
                "context_mode": "fampr",
                "source_commit": "e4b7b6cb0d9354afc07f9d0348f801340043ffd1",
            },
        },
        "diagnostic_interface_verification": interface_verification,
        "fampr_parameter_audit": parameter_audit,
        "official_validation_metrics": final_results,
        "official_metric_delta_percentage_points": {
            "mIoU": 100 * (final_results["fampr"]["Mean IoU"] - final_results["a0"]["Mean IoU"]),
            "mDice": 100 * (final_results["fampr"]["Mean Dice"] - final_results["a0"]["Mean Dice"]),
            "per_class_IoU": {
                class_id: 100 * (final_results["fampr"]["Class IoU"][class_id] - final_results["a0"]["Class IoU"][class_id])
                for class_id in range(4)
            },
        },
        "transition_group_counts": {
            class_id: {
                group: int(transition_group_counts[class_id, group_index])
                for group_index, group in enumerate(TRANSITION_GROUPS)
            }
            for class_id in range(4)
        },
        "prediction_transition_matrices": transition_matrix,
        "component_thresholds": component_thresholds,
        "stagewise_metrics": {
            f"{model}_{stage}": result for (model, stage), result in stage_results.items()
        },
        "outputs": {
            "pixel_transition_stats": output_dir / "fampr_pixel_transition_stats.csv",
            "class_conditioned_stats": output_dir / "fampr_class_conditioned_stats.csv",
            "corrected_vs_harmed_stats": output_dir / "fampr_corrected_vs_harmed_stats.csv",
            "boundary_interior_stats": output_dir / "fampr_boundary_interior_stats.csv",
            "component_size_stats": output_dir / "fampr_component_size_stats.csv",
            "stagewise_cam_stats": output_dir / "fampr_stagewise_cam_stats.csv",
            "visualizations": visual_dir,
        },
        "limitations": [
            "GT is used for offline validation diagnosis only, never for inference.",
            "Pixel-level p-values are exploratory because neighboring pixels are spatially correlated.",
            "Quantiles and rank tests use deterministic sampled reservoirs; means, standard deviations, counts, minima, and maxima are streaming exact.",
            "AdaKern gates and anchor/gamma are image-level or stage-level values broadcast to pixels for class conditioning; they are not spatial gates.",
            "Deep CAM can differ because A0 and FA-MPR are independently trained checkpoints even though FA-MPR does not structurally modify the deep head.",
        ],
    }
    summary_path = output_dir / "fampr_class_response_summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(to_builtin({"summary": summary_path, "metrics": final_results, "delta_pp": summary["official_metric_delta_percentage_points"]}), indent=2), flush=True)


if __name__ == "__main__":
    main()
