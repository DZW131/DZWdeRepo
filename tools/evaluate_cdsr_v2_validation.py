"""Evaluate a frozen CDSR-v2 FINAL checkpoint on BCSS validation only.

The fused result delegates to the released SSHR inference function.  A second
observation-only pass reproduces the same TTA, class-presence thresholds,
normalization, and mask conversion while exposing each CAM scale separately.
No test-set path is accepted by this program.
"""

import argparse
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tool import infer_utils, iouutils
from tool.GenDataset import Stage1_InferDataset
from tool.infer_fun import (
    _get_class_thresholds,
    _tta_transforms,
    infer,
)


STAGE_NAMES = ("cam56", "cam28_1", "cam28_2", "camdeep")
HFRM_STAGES = {
    "stage1": "hfrm_56",
    "stage2": "hfrm_28_1",
    "stage3": "hfrm_28_2",
}
OFFICIAL_FUSION = (0.0, 0.6, 0.2, 0.2)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_cam(cam):
    minimum = np.min(cam, axis=(1, 2), keepdims=True)
    maximum = np.max(cam, axis=(1, 2), keepdims=True)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def score_from_cam(cam, label, original_image):
    selected = cam * label.reshape(label.shape[0], 1, 1)
    cam_dict = infer_utils.cam_npy_to_cam_dict(selected, label)
    cam_score, _ = infer_utils.dict2npy(cam_dict, label, original_image)
    return infer_utils.cam_npy_to_label_map(cam_score)


class RunningMoments:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_square = 0.0
        self.finite = True

    def update(self, tensor):
        values = tensor.detach().float()
        self.finite = self.finite and bool(torch.isfinite(values).all().item())
        self.count += values.numel()
        self.total += values.double().sum().item()
        self.total_square += values.double().square().sum().item()

    def summary(self):
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "mean": mean,
            "std": variance**0.5,
            "finite": self.finite,
            "count": self.count,
        }


class RunningRMS:
    def __init__(self):
        self.count = 0
        self.total_square = 0.0
        self.finite = True

    def update(self, tensor):
        values = tensor.detach().float()
        self.finite = self.finite and bool(torch.isfinite(values).all().item())
        self.count += values.numel()
        self.total_square += values.double().square().sum().item()

    def summary(self):
        return {
            "rms": (self.total_square / self.count) ** 0.5,
            "finite": self.finite,
            "count": self.count,
        }


def official_args(args):
    return argparse.Namespace(
        dataset="bcss",
        img_size=224,
        num_workers=args.num_workers,
        amp_dtype="bf16",
    )


def evaluate_stage_cams(model, validation_root, args):
    """Expose individual scales using an exact copy of official inference."""
    dataset = Stage1_InferDataset(
        data_path=str(validation_root / "img"),
        img_size=224,
    )
    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    thresholds = _get_class_thresholds(official_args(args), None, 4)
    predictions = {name: [] for name in (*STAGE_NAMES, "official_fused")}
    ground_truth = []
    start = time.perf_counter()

    with torch.no_grad():
        for image_names, image_tensor in loader:
            image_name = image_names[0]
            image_path = validation_root / "img" / f"{image_name}.png"
            original_image = np.asarray(Image.open(image_path).convert("RGB"))
            original_size = original_image.shape[:2]
            image_tensor = image_tensor.cuda(non_blocking=True)
            tta_cams = {name: [] for name in STAGE_NAMES}
            probabilities = []

            for input_flip_dims, cam_flip_dims in _tta_transforms():
                transformed = (
                    torch.flip(image_tensor, dims=input_flip_dims)
                    if input_flip_dims else image_tensor
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    outputs = model.forward_cam(transformed)
                    cams = outputs[:4]
                    probability = outputs[4]
                    resized = [
                        F.interpolate(
                            cam,
                            original_size,
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        for cam in cams
                    ]
                if cam_flip_dims:
                    resized = [
                        torch.flip(cam, dims=cam_flip_dims) for cam in resized
                    ]
                for name, cam in zip(STAGE_NAMES, resized):
                    tta_cams[name].append(cam)
                probabilities.append(probability)

            probability = (
                torch.stack(probabilities).mean(dim=0).detach().float()
                .cpu().numpy()[0]
            )
            label = (probability > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(probability))] = 1.0

            normalized = {}
            for name in STAGE_NAMES:
                averaged = torch.stack(tta_cams[name]).mean(dim=0)
                normalized[name] = normalize_cam(
                    averaged.detach().float().cpu().numpy()
                )
                predictions[name].append(
                    score_from_cam(normalized[name], label, original_image)
                )

            fused = (
                OFFICIAL_FUSION[1] * normalized["cam28_1"]
                + OFFICIAL_FUSION[2] * normalized["cam28_2"]
                + OFFICIAL_FUSION[3] * normalized["camdeep"]
            )
            predictions["official_fused"].append(
                score_from_cam(fused, label, original_image)
            )
            ground_truth.append(
                np.asarray(
                    Image.open(validation_root / "mask" / f"{image_name}.png")
                )
            )

    scores = {
        name: iouutils.scores(ground_truth, values, n_class=4)
        for name, values in predictions.items()
    }
    return {
        "sample_count": len(dataset),
        "seconds": time.perf_counter() - start,
        "scores": scores,
    }


def collect_mechanism_state(model, validation_root, args):
    dataset = Stage1_InferDataset(
        data_path=str(validation_root / "img"),
        img_size=224,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.mechanism_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    accumulators = {
        stage: {
            "need": RunningMoments(),
            "semantic_gate": RunningMoments(),
            "context_gate": RunningMoments(),
            "effective_semantic": RunningRMS(),
            "effective_context": RunningRMS(),
            "need_values": [],
        }
        for stage in HFRM_STAGES
    }
    start = time.perf_counter()
    with torch.no_grad():
        for _, image_tensor in loader:
            image_tensor = image_tensor.cuda(non_blocking=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                _, diagnostics = model.forward_with_diagnostics(image_tensor)
            for stage in HFRM_STAGES:
                values = diagnostics["cdsr"][stage]
                need = values["need_map"].detach().float()
                accumulators[stage]["need"].update(need)
                accumulators[stage]["semantic_gate"].update(
                    values["semantic_gate"]
                )
                accumulators[stage]["context_gate"].update(
                    values["context_gate"]
                )
                accumulators[stage]["effective_semantic"].update(
                    values["effective_semantic"]
                )
                accumulators[stage]["effective_context"].update(
                    values["effective_context"]
                )
                accumulators[stage]["need_values"].append(
                    need.cpu().numpy().reshape(-1)
                )

    stage_results = {}
    for stage, hfrm_name in HFRM_STAGES.items():
        hfrm = getattr(model, hfrm_name)
        need_values = np.concatenate(accumulators[stage].pop("need_values"))
        need_summary = accumulators[stage]["need"].summary()
        quantiles = np.quantile(need_values, [0.10, 0.50, 0.90])
        need_summary.update(
            {"p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2]}
        )
        stage_results[stage] = {
            "need": need_summary,
            "semantic_gate": accumulators[stage]["semantic_gate"].summary(),
            "context_gate": accumulators[stage]["context_gate"].summary(),
            "gamma_sem": hfrm.gamma_veto.detach().float().item(),
            "gamma_ctx": hfrm.gamma_context.detach().float().item(),
            "effective_semantic_residual": accumulators[stage][
                "effective_semantic"
            ].summary(),
            "effective_context_residual": accumulators[stage][
                "effective_context"
            ].summary(),
        }
    return {
        "sample_count": len(dataset),
        "seconds": time.perf_counter() - start,
        "alpha_sem": model.cdsr_selective_gate.alpha_sem.detach().float().item(),
        "alpha_ctx": model.cdsr_selective_gate.alpha_ctx.detach().float().item(),
        "stages": stage_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--network", default="network.resnet38_cls")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mechanism-batch-size", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Official BF16 validation requires CUDA")
    expected_count = 3418
    image_count = len(list((args.val_root / "img").glob("*.png")))
    mask_count = len(list((args.val_root / "mask").glob("*.png")))
    if image_count != expected_count or mask_count != expected_count:
        raise ValueError(
            "BCSS validation must contain exactly 3418 images and masks; "
            f"found images={image_count}, masks={mask_count}"
        )

    state_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = getattr(importlib.import_module(args.network), "Net_CAM")(
        n_class=4,
        rectifier_type="hfrm",
        context_mode="ch",
        rectification_mode="cdsr",
    )
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()
    torch.cuda.reset_peak_memory_stats()

    official_start = time.perf_counter()
    official_score = infer(
        model,
        str(args.val_root),
        4,
        official_args(args),
        thr=None,
        cam_weights=(0.6, 0.2, 0.2),
    )
    official_seconds = time.perf_counter() - official_start
    if official_score is None:
        raise RuntimeError("Released SSHR inference returned no score")

    cam_audit = evaluate_stage_cams(model, args.val_root, args)
    reproduced_fused = cam_audit["scores"]["official_fused"]
    parity = {
        "mean_iou_absolute_difference": abs(
            float(official_score["Mean IoU"])
            - float(reproduced_fused["Mean IoU"])
        ),
        "mean_dice_absolute_difference": abs(
            float(official_score["Mean Dice"])
            - float(reproduced_fused["Mean Dice"])
        ),
    }
    parity["pass"] = (
        parity["mean_iou_absolute_difference"] <= 1e-12
        and parity["mean_dice_absolute_difference"] <= 1e-12
    )
    if not parity["pass"]:
        raise RuntimeError(f"Official inference parity failed: {parity}")

    mechanism = collect_mechanism_state(model, args.val_root, args)
    result = {
        "scope": "BCSS validation only",
        "test_evaluated": False,
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        },
        "validation": {
            "root": str(args.val_root.resolve()),
            "image_count": image_count,
            "mask_count": mask_count,
            "official_fused_score": official_score,
            "stage_cam_scores": {
                name: cam_audit["scores"][name] for name in STAGE_NAMES
            },
            "independent_official_fusion_score": reproduced_fused,
            "official_inference_parity": parity,
        },
        "mechanism": mechanism,
        "runtime": {
            "official_fused_seconds": official_seconds,
            "stage_cam_audit_seconds": cam_audit["seconds"],
            "mechanism_audit_seconds": mechanism["seconds"],
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "precision": "BF16 autocast",
        },
        "protocol": {
            "tta": "identity + horizontal flip + vertical flip",
            "class_presence_thresholds": [0.8, 0.9, 0.8, 0.6],
            "official_fusion_cam56_cam28_1_cam28_2_camdeep": list(
                OFFICIAL_FUSION
            ),
            "metric": "tool.iouutils.scores",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(result), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(json_ready(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
