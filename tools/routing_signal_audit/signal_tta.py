"""GT-free aligned TTA reliability and frozen feature extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

from tool.infer_fun import _tta_transforms
from tools.decision_audit.fusion import normalize_cam
from tools.routing_signal_audit import BRANCH_NAMES, EXPECTED_IMAGES


EPSILON = 1e-8
STAGE_NAMES = ("h56", "h28_1", "h28_2", "fdeep")


class OrderedInferenceDataset(Dataset):
    def __init__(self, validation_root: Path, image_names: list[str]):
        self.validation_root = Path(validation_root)
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]
        image = Image.open(self.validation_root / "img" / f"{name}.png").convert("RGB")
        if image.size != (224, 224):
            image = TF.resize(
                image, [224, 224], interpolation=InterpolationMode.BILINEAR
            )
        tensor = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        return index, tensor


def _probability(cam: np.ndarray) -> np.ndarray:
    normalized = normalize_cam(cam)
    shifted = normalized - normalized.max(axis=0, keepdims=True)
    exponential = np.exp(shifted).astype(np.float32, copy=False)
    return exponential / (exponential.sum(axis=0, keepdims=True) + EPSILON)


def _jsd(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (first + second)
    return 0.5 * np.sum(
        first * (np.log(first + EPSILON) - np.log(midpoint + EPSILON))
        + second * (np.log(second + EPSILON) - np.log(midpoint + EPSILON)),
        axis=0,
    )


def _tta_branch_features(view_cams: list[np.ndarray]) -> list[float]:
    probabilities = [_probability(cam) for cam in view_cams]
    pairs = ((0, 1), (0, 2), (1, 2))
    jsd_values = [float(_jsd(probabilities[a], probabilities[b]).mean()) for a, b in pairs]
    l1_values = [float(np.abs(probabilities[a] - probabilities[b]).mean()) for a, b in pairs]
    argmax = [np.argmax(probability, axis=0) for probability in probabilities]
    consistency = float(np.mean((argmax[0] == argmax[1]) & (argmax[0] == argmax[2])))
    variance = np.var(np.stack(probabilities, axis=0), axis=0)
    return [
        float(np.mean(jsd_values)),
        float(np.max(jsd_values)),
        float(np.mean(l1_values)),
        float(np.max(l1_values)),
        consistency,
        float(variance.mean()),
        float(np.quantile(variance, 0.90)),
    ]


def extract_tta_and_feature_signals(
    model,
    validation_root: Path,
    image_names: list[str],
    phase0_cache_dir: Path,
    output_cache_dir: Path,
    num_workers: int,
) -> dict:
    if len(image_names) != EXPECTED_IMAGES:
        raise RuntimeError("Feature extraction must preserve all 3418 Phase-0 images")
    output_cache_dir = Path(output_cache_dir)
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    gap_dir = output_cache_dir / "gap_features"
    gap_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        OrderedInferenceDataset(validation_root, image_names),
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    tta_names = [
        "tta_jsd_mean",
        "tta_jsd_max",
        "tta_l1_mean",
        "tta_l1_max",
        "tta_three_view_argmax_consistency",
        "tta_probability_variance_mean",
        "tta_probability_variance_p90",
    ]
    tta_features = np.lib.format.open_memmap(
        output_cache_dir / "tta_signal_features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_IMAGES, 4, len(tta_names)),
    )
    stage_channels = {
        "h56": 256,
        "h28_1": 512,
        "h28_2": 1024,
        "fdeep": 4096,
    }
    gap_arrays = {
        stage: np.lib.format.open_memmap(
            gap_dir / f"{stage}.npy",
            mode="w+",
            dtype=np.float32,
            shape=(EXPECTED_IMAGES, channels),
        )
        for stage, channels in stage_channels.items()
    }
    feature_scalars = np.lib.format.open_memmap(
        output_cache_dir / "feature_scalar_context.npy",
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_IMAGES, 12),
    )
    phase0_cams = {
        name: np.load(Path(phase0_cache_dir) / f"{name}.npy", mmap_mode="r")
        for name in BRANCH_NAMES
    }
    captures: dict[str, list[torch.Tensor]] = {stage: [] for stage in STAGE_NAMES}
    handles = [
        model.hfrm_56.register_forward_hook(
            lambda module, inputs, output: captures["h56"].append(output.detach())
        ),
        model.hfrm_28_1.register_forward_hook(
            lambda module, inputs, output: captures["h28_1"].append(output.detach())
        ),
        model.hfrm_28_2.register_forward_hook(
            lambda module, inputs, output: captures["h28_2"].append(output.detach())
        ),
        model.fc8.register_forward_pre_hook(
            lambda module, inputs: captures["fdeep"].append(inputs[0].detach())
        ),
    ]
    max_aggregate_difference = 0.0
    model.eval()
    model.requires_grad_(False)
    try:
        with torch.no_grad():
            for expected_index, (index_tensor, image_tensor) in enumerate(loader):
                index = int(index_tensor.item())
                if index != expected_index:
                    raise RuntimeError("Feature loader changed the Phase-0 image order")
                for values in captures.values():
                    values.clear()
                image_tensor = image_tensor.cuda(non_blocking=True)
                view_outputs = {name: [] for name in BRANCH_NAMES}
                identity_features = None
                for view_index, (input_flip_dims, cam_flip_dims) in enumerate(_tta_transforms()):
                    transformed = (
                        torch.flip(image_tensor, dims=input_flip_dims)
                        if input_flip_dims
                        else image_tensor
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        outputs = model.forward_cam(transformed)
                        resized = [
                            F.interpolate(
                                cam,
                                (224, 224),
                                mode="bilinear",
                                align_corners=False,
                            )[0]
                            for cam in outputs[:4]
                        ]
                    if view_index == 0:
                        identity_features = {
                            "h56": captures["h56"][0],
                            "h28_1": captures["h28_1"][0],
                            "h28_2": captures["h28_2"][0],
                            "fdeep": captures["fdeep"][0],
                        }
                    for name, cam in zip(BRANCH_NAMES, resized):
                        if cam_flip_dims:
                            cam = torch.flip(cam, dims=cam_flip_dims)
                        view_outputs[name].append(cam.detach())
                scalar_values = []
                for stage in STAGE_NAMES:
                    feature = identity_features[stage].float()
                    gap_arrays[stage][index] = (
                        F.adaptive_avg_pool2d(feature, 1)
                        .reshape(-1)
                        .cpu()
                        .numpy()
                    )
                    scalar_values.extend(
                        [
                            float(feature.mean().item()),
                            float(feature.std(unbiased=False).item()),
                            float(torch.sqrt(torch.mean(feature.square())).item()),
                        ]
                    )
                feature_scalars[index] = np.asarray(scalar_values, dtype=np.float32)
                for branch_index, name in enumerate(BRANCH_NAMES):
                    views = [
                        value.float().cpu().numpy() for value in view_outputs[name]
                    ]
                    tta_features[index, branch_index] = _tta_branch_features(views)
                    aggregated = normalize_cam(
                        torch.stack(view_outputs[name])
                        .mean(dim=0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                    max_aggregate_difference = max(
                        max_aggregate_difference,
                        float(np.max(np.abs(aggregated - phase0_cams[name][index]))),
                    )
    finally:
        for handle in handles:
            handle.remove()
    tta_features.flush()
    feature_scalars.flush()
    for array in gap_arrays.values():
        array.flush()
    arrays = [tta_features, feature_scalars, *gap_arrays.values()]
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("Signal Set B/C extraction produced non-finite values")
    if max_aggregate_difference > 1e-6:
        raise RuntimeError(
            "Fresh TTA aggregation does not match the frozen Phase-0 cache; STOP: "
            f"max_abs_diff={max_aggregate_difference}"
        )
    (output_cache_dir / "tta_signal_features.names.json").write_text(
        json.dumps(tta_names, indent=2) + "\n", encoding="utf-8"
    )
    scalar_names = [
        f"{stage}_{metric}"
        for stage in STAGE_NAMES
        for metric in ("feature_mean", "feature_std", "feature_rms")
    ]
    (output_cache_dir / "feature_scalar_context.names.json").write_text(
        json.dumps(scalar_names, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "tta_shape": list(tta_features.shape),
        "tta_feature_count": len(tta_names),
        "gap_shapes": {stage: list(array.shape) for stage, array in gap_arrays.items()},
        "feature_scalar_shape": list(feature_scalars.shape),
        "fresh_vs_phase0_aggregate_max_abs_difference": max_aggregate_difference,
        "contains_gt": False,
        "finite": True,
    }
