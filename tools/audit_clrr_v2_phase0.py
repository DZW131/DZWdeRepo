"""Run the preregistered frozen-A0 CLRR-v2 Phase-0 validation audit.

This program accepts a BCSS validation root only. It performs no training and
has no test-root argument. CLRR-v2 eta is fixed at 0.05.
"""

import argparse
import hashlib
import importlib
import json
import math
import platform
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from tool.infer_fun import _get_class_thresholds, _tta_transforms, infer
from tools.clrr_v2_phase0_core import (
    analytical_virtual_correction,
    leave_one_out_consensus,
)


ETA = 0.05
STAGES = ("stage1", "stage2", "stage3")
CAM_NAMES = ("cam56", "cam28_1", "cam28_2", "camdeep")
STAGE_TO_CAM = {
    "stage1": "cam56",
    "stage2": "cam28_1",
    "stage3": "cam28_2",
}
STAGE_TO_HEAD = {
    "stage1": "ic_56",
    "stage2": "ic1",
    "stage3": "ic2",
}
STAGE_TO_HFRM = {
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


def label_map_from_cam(cam, label, original_image):
    selected = cam * label.reshape(label.shape[0], 1, 1)
    cam_dict = infer_utils.cam_npy_to_cam_dict(selected, label)
    cam_score, _ = infer_utils.dict2npy(cam_dict, label, original_image)
    return infer_utils.cam_npy_to_label_map(cam_score).astype(np.uint8)


def fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    return np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class**2,
    ).reshape(n_class, n_class)


def official_score_from_hist(histogram):
    """Exact algebra from tool.iouutils.scores after streaming accumulation."""
    hist = histogram.copy()
    hist[4, 4] = 0
    accuracy = np.diag(hist).sum() / hist.sum()
    class_accuracy = np.diag(hist)[:4] / hist.sum(axis=1)[:4]
    class_accuracy = np.nanmean(class_accuracy)
    iou = np.diag(hist)[:4] / (
        hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
    )[:4]
    mean_iou = np.nanmean(iou)
    frequency = hist.sum(axis=1)[:4] / hist.sum()
    frequency_weighted = (
        frequency[frequency > 0] * iou[frequency > 0]
    ).sum()
    dice = {}
    for class_id in range(4):
        true_positive = np.diag(hist)[class_id]
        false_positive = hist[:, class_id].sum() - true_positive
        false_negative = hist[class_id, :].sum() - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        dice[class_id] = (
            2 * true_positive / denominator if denominator > 0 else 0.0
        )
    return {
        "Pixel Accuracy": accuracy,
        "Mean Accuracy": class_accuracy,
        "Frequency Weighted IoU": frequency_weighted,
        "Mean IoU": mean_iou,
        "Class IoU": dict(zip(range(4), iou)),
        "Dice Coefficients": dice,
        "Mean Dice": np.mean(list(dice.values())),
    }


class OfficialMetricAccumulator:
    def __init__(self):
        self.histogram = np.zeros((5, 5), dtype=np.float64)

    def update(self, ground_truth, prediction):
        prediction = np.array(prediction, copy=True)
        prediction[ground_truth == 4] = 4
        self.histogram += fast_hist(
            ground_truth.flatten(), prediction.flatten(), 5
        )

    def score(self):
        return official_score_from_hist(self.histogram)


class ChangeAccumulator:
    def __init__(self):
        self.total_foreground = 0
        self.corrected = 0
        self.harmed = 0
        self.changed = 0
        self.per_class = {
            class_id: {"total": 0, "corrected": 0, "harmed": 0, "changed": 0}
            for class_id in range(4)
        }

    def update(self, ground_truth, before, after):
        foreground = ground_truth < 4
        before_correct = before == ground_truth
        after_correct = after == ground_truth
        corrected = foreground & ~before_correct & after_correct
        harmed = foreground & before_correct & ~after_correct
        changed = foreground & (before != after)
        self.total_foreground += int(foreground.sum())
        self.corrected += int(corrected.sum())
        self.harmed += int(harmed.sum())
        self.changed += int(changed.sum())
        for class_id in range(4):
            class_mask = ground_truth == class_id
            values = self.per_class[class_id]
            values["total"] += int(class_mask.sum())
            values["corrected"] += int((corrected & class_mask).sum())
            values["harmed"] += int((harmed & class_mask).sum())
            values["changed"] += int((changed & class_mask).sum())

    def summary(self):
        denominator = max(self.total_foreground, 1)
        return {
            "foreground_pixels": self.total_foreground,
            "corrected": self.corrected,
            "harmed": self.harmed,
            "net_corrected": self.corrected - self.harmed,
            "corrected_rate": self.corrected / denominator,
            "harmed_rate": self.harmed / denominator,
            "net_rate": (self.corrected - self.harmed) / denominator,
            "prediction_change_rate": self.changed / denominator,
            "per_class": self.per_class,
        }


class ConsensusUtilityAccumulator:
    def __init__(self):
        self.total = 0
        self.recoverable = 0
        self.harmful = 0
        self.stage_hist = np.zeros((4, 4), dtype=np.float64)
        self.consensus_hist = np.zeros((4, 4), dtype=np.float64)
        self.per_class = {
            class_id: {"total": 0, "recoverable": 0, "harmful": 0}
            for class_id in range(4)
        }

    def update(self, ground_truth, stage_prediction, consensus_prediction):
        foreground = ground_truth < 4
        truth = ground_truth[foreground]
        stage = stage_prediction[foreground]
        consensus = consensus_prediction[foreground]
        stage_correct = stage == truth
        consensus_correct = consensus == truth
        recoverable = ~stage_correct & consensus_correct
        harmful = stage_correct & ~consensus_correct
        self.total += int(foreground.sum())
        self.recoverable += int(recoverable.sum())
        self.harmful += int(harmful.sum())
        self.stage_hist += fast_hist(truth, stage, 4)
        self.consensus_hist += fast_hist(truth, consensus, 4)
        for class_id in range(4):
            class_mask = truth == class_id
            values = self.per_class[class_id]
            values["total"] += int(class_mask.sum())
            values["recoverable"] += int((recoverable & class_mask).sum())
            values["harmful"] += int((harmful & class_mask).sum())

    @staticmethod
    def mean_iou(hist):
        denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
        iou = np.divide(
            np.diag(hist),
            denominator,
            out=np.full(4, np.nan),
            where=denominator > 0,
        )
        return np.nanmean(iou), iou

    def summary(self):
        denominator = max(self.total, 1)
        stage_mean, stage_iou = self.mean_iou(self.stage_hist)
        consensus_mean, consensus_iou = self.mean_iou(self.consensus_hist)
        return {
            "foreground_pixels": self.total,
            "recoverable": self.recoverable,
            "harmful": self.harmful,
            "net_recoverability": self.recoverable - self.harmful,
            "recoverable_rate": self.recoverable / denominator,
            "harmful_rate": self.harmful / denominator,
            "net_rate": (self.recoverable - self.harmful) / denominator,
            "stage_foreground_miou": stage_mean,
            "consensus_foreground_miou": consensus_mean,
            "stage_per_class_iou": dict(zip(range(4), stage_iou)),
            "consensus_per_class_iou": dict(zip(range(4), consensus_iou)),
            "per_class": self.per_class,
        }


class MechanismAccumulator:
    def __init__(self):
        self.ce_deltas = []
        self.ce_decrease = 0
        self.nontrivial = 0
        self.update_ratios = []
        self.probability_change_sum = 0.0
        self.probability_values = 0
        self.argmax_changed = 0
        self.argmax_values = 0
        self.mismatch_sum = 0.0
        self.mismatch_values = 0
        self.finite = True

    def update(self, probability, virtual_probability, consensus, correction):
        ce_before = -(
            consensus * probability.clamp_min(1e-8).log()
        ).sum(dim=1)
        ce_after = -(
            consensus * virtual_probability.clamp_min(1e-8).log()
        ).sum(dim=1)
        ce_delta = ce_after - ce_before
        mismatch = correction["mismatch"][:, 0]
        nontrivial = mismatch > 1e-6
        selected = ce_delta[nontrivial].detach().cpu().numpy().astype(np.float32)
        self.ce_deltas.append(selected)
        self.ce_decrease += int((ce_delta[nontrivial] < 0).sum().item())
        self.nontrivial += int(nontrivial.sum().item())
        ratios = correction["update_ratio"].detach().cpu().numpy().astype(np.float32)
        self.update_ratios.append(ratios.reshape(-1))
        difference = (virtual_probability - probability).abs()
        self.probability_change_sum += difference.double().sum().item()
        self.probability_values += difference.numel()
        self.argmax_changed += int(
            (virtual_probability.argmax(1) != probability.argmax(1)).sum().item()
        )
        self.argmax_values += probability.shape[0] * probability.shape[2] * probability.shape[3]
        self.mismatch_sum += mismatch.double().sum().item()
        self.mismatch_values += mismatch.numel()
        tensors = [
            probability,
            virtual_probability,
            consensus,
            *[value for value in correction.values() if torch.is_tensor(value)],
        ]
        self.finite = self.finite and all(
            bool(torch.isfinite(value).all().item()) for value in tensors
        )

    def summary(self):
        ce_delta = np.concatenate(self.ce_deltas)
        update_ratio = np.concatenate(self.update_ratios)
        return {
            "nontrivial_mismatch_pixels": self.nontrivial,
            "mean_consensus_ce_delta": float(ce_delta.mean()),
            "median_consensus_ce_delta": float(np.median(ce_delta)),
            "ce_decrease_pixels": self.ce_decrease,
            "ce_decrease_fraction": self.ce_decrease / max(self.nontrivial, 1),
            "mean_mismatch": self.mismatch_sum / self.mismatch_values,
            "mean_update_rms_ratio": float(update_ratio.mean()),
            "p99_update_rms_ratio": float(np.quantile(update_ratio, 0.99)),
            "max_update_rms_ratio": float(update_ratio.max()),
            "mean_absolute_probability_change": (
                self.probability_change_sum / self.probability_values
            ),
            "argmax_change_fraction": self.argmax_changed / self.argmax_values,
            "all_finite": self.finite,
        }


def classifier_manifest(model):
    result = {}
    for stage, head_name in STAGE_TO_HEAD.items():
        head = getattr(model, head_name)
        if not isinstance(head, nn.Conv2d) or head.kernel_size != (1, 1):
            raise RuntimeError(f"{head_name} is not a linear 1x1 Conv2d")
        result[stage] = {
            "module": head_name,
            "weight_shape": list(head.weight.shape),
            "bias_shape": list(head.bias.shape) if head.bias is not None else None,
            "direct_input": "corresponding HFRM rectified feature",
            "intervening_trainable_nonlinearity": False,
        }
    return result


def extract_pass0(model, image):
    """Execute the official A0 backbone and HFRM exactly once."""
    x = model.conv1a(image)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)
    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    feature_56 = x
    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x)
    x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    feature_28_1 = F.relu(model.bn45(x))
    x, _ = model.b5(x, get_x_bn_relu=True)
    x = model.b5_1(x); x = model.b5_2(x)
    feature_28_2 = F.relu(model.bn52(x))
    x, _ = model.b6(x, get_x_bn_relu=True); x = model.b7(x)
    feature_deep = F.relu(model.bn7(x))
    features = {
        "stage1": model.hfrm_56(feature_56, feature_deep),
        "stage2": model.hfrm_28_1(feature_28_1, feature_deep),
        "stage3": model.hfrm_28_2(feature_28_2, feature_deep),
        "deep": feature_deep,
    }
    logits = {
        "stage1": model.ic_56(features["stage1"]),
        "stage2": model.ic1(features["stage2"]),
        "stage3": model.ic2(features["stage3"]),
        "deep": model.fc8(feature_deep),
    }
    return features, logits


def virtual_pass(model, features, logits):
    probabilities = {
        name: torch.softmax(value.detach().float(), dim=1)
        for name, value in logits.items()
    }
    virtual_logits = {}
    mechanism = {}
    for stage in STAGES:
        head = getattr(model, STAGE_TO_HEAD[stage])
        hfrm = getattr(model, STAGE_TO_HFRM[stage])
        consensus = leave_one_out_consensus(
            probabilities, stage, features[stage].shape[-2:]
        )
        correction = analytical_virtual_correction(
            features[stage],
            probabilities[stage],
            consensus,
            head.weight,
            hfrm.gamma_veto,
            hfrm.gamma_context,
            eta=ETA,
        )
        bias = head.bias.detach().float() if head.bias is not None else None
        # Phase-0 feedback and directional probe are explicitly FP32.
        virtual_logits[stage] = F.conv2d(
            correction["updated_feature"],
            head.weight.detach().float(),
            bias,
        )
        mechanism[stage] = {
            "probability": probabilities[stage],
            "virtual_probability": torch.softmax(
                virtual_logits[stage], dim=1
            ).detach(),
            "consensus_state": consensus,
            "correction": correction,
        }
    virtual_logits["deep"] = logits["deep"].detach().float()
    return probabilities, virtual_logits, mechanism


def resize_ground_truth(ground_truth, size):
    tensor = torch.from_numpy(ground_truth.astype(np.float32))[None, None]
    return F.interpolate(tensor, size=size, mode="nearest")[0, 0].numpy().astype(np.uint8)


def official_infer_args(num_workers):
    return argparse.Namespace(
        dataset="bcss",
        img_size=224,
        num_workers=num_workers,
        amp_dtype="bf16",
    )


def run_phase0(model, validation_root, num_workers):
    dataset = Stage1_InferDataset(
        data_path=str(validation_root / "img"), img_size=224
    )
    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    thresholds = _get_class_thresholds(
        official_infer_args(num_workers), None, 4
    )
    metrics = {
        mode: {name: OfficialMetricAccumulator() for name in (*CAM_NAMES, "fused")}
        for mode in ("pass0", "virtual")
    }
    changes = {
        name: ChangeAccumulator() for name in (*CAM_NAMES[:3], "fused")
    }
    consensus_utility = {
        stage: ConsensusUtilityAccumulator() for stage in STAGES
    }
    mechanism_accumulators = {
        stage: MechanismAccumulator() for stage in STAGES
    }
    start = time.perf_counter()

    with torch.no_grad():
        for image_names, image_tensor in loader:
            image_name = image_names[0]
            original_image = np.asarray(
                Image.open(validation_root / "img" / f"{image_name}.png").convert("RGB")
            )
            ground_truth = np.asarray(
                Image.open(validation_root / "mask" / f"{image_name}.png")
            )
            original_size = original_image.shape[:2]
            image_tensor = image_tensor.cuda(non_blocking=True)
            tta_cams = {
                mode: {name: [] for name in CAM_NAMES}
                for mode in ("pass0", "virtual")
            }
            deep_probabilities = []

            for input_flip_dims, cam_flip_dims in _tta_transforms():
                transformed = (
                    torch.flip(image_tensor, dims=input_flip_dims)
                    if input_flip_dims else image_tensor
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    features, logits = extract_pass0(model, transformed)
                probabilities, virtual_logits, mechanism = virtual_pass(
                    model, features, logits
                )
                pass0_cams = {
                    "cam56": F.relu(logits["stage1"]),
                    "cam28_1": F.relu(logits["stage2"]),
                    "cam28_2": F.relu(logits["stage3"]),
                    "camdeep": F.relu(logits["deep"]),
                }
                virtual_cams = {
                    "cam56": F.relu(virtual_logits["stage1"]),
                    "cam28_1": F.relu(virtual_logits["stage2"]),
                    "cam28_2": F.relu(virtual_logits["stage3"]),
                    "camdeep": F.relu(virtual_logits["deep"]),
                }
                for mode, cam_values in (
                    ("pass0", pass0_cams), ("virtual", virtual_cams)
                ):
                    for name, cam in cam_values.items():
                        resized = F.interpolate(
                            cam,
                            original_size,
                            mode="bilinear",
                            align_corners=False,
                        )[0]
                        if cam_flip_dims:
                            resized = torch.flip(resized, dims=cam_flip_dims)
                        tta_cams[mode][name].append(resized)
                out_deep = F.adaptive_avg_pool2d(
                    logits["deep"], 1
                ).view(1, -1)
                deep_probabilities.append(torch.sigmoid(out_deep))

                if not input_flip_dims:
                    for stage in STAGES:
                        state = mechanism[stage]
                        mechanism_accumulators[stage].update(
                            state["probability"],
                            state["virtual_probability"],
                            state["consensus_state"]["consensus"],
                            state["correction"],
                        )
                        native_ground_truth = resize_ground_truth(
                            ground_truth,
                            state["probability"].shape[-2:],
                        )
                        consensus_utility[stage].update(
                            native_ground_truth,
                            state["probability"][0].argmax(0).cpu().numpy().astype(np.uint8),
                            state["consensus_state"]["consensus"][0]
                            .argmax(0).cpu().numpy().astype(np.uint8),
                        )

            deep_probability = (
                torch.stack(deep_probabilities).mean(0).float().cpu().numpy()[0]
            )
            label = (deep_probability > thresholds).astype(np.float32)
            if label.sum() == 0:
                label[int(np.argmax(deep_probability))] = 1.0

            predictions = {"pass0": {}, "virtual": {}}
            normalized = {"pass0": {}, "virtual": {}}
            for mode in ("pass0", "virtual"):
                for name in CAM_NAMES:
                    averaged = torch.stack(tta_cams[mode][name]).mean(0)
                    normalized[mode][name] = normalize_cam(
                        averaged.cpu().numpy()
                    )
                    predictions[mode][name] = label_map_from_cam(
                        normalized[mode][name], label, original_image
                    )
                    metrics[mode][name].update(
                        ground_truth, predictions[mode][name]
                    )
                fused = (
                    OFFICIAL_FUSION[1] * normalized[mode]["cam28_1"]
                    + OFFICIAL_FUSION[2] * normalized[mode]["cam28_2"]
                    + OFFICIAL_FUSION[3] * normalized[mode]["camdeep"]
                )
                predictions[mode]["fused"] = label_map_from_cam(
                    fused, label, original_image
                )
                metrics[mode]["fused"].update(
                    ground_truth, predictions[mode]["fused"]
                )

            for name, accumulator in changes.items():
                accumulator.update(
                    ground_truth,
                    predictions["pass0"][name],
                    predictions["virtual"][name],
                )

    return {
        "sample_count": len(dataset),
        "seconds": time.perf_counter() - start,
        "scores": {
            mode: {name: accumulator.score() for name, accumulator in values.items()}
            for mode, values in metrics.items()
        },
        "prediction_changes": {
            name: accumulator.summary() for name, accumulator in changes.items()
        },
        "consensus_utility": {
            stage: accumulator.summary()
            for stage, accumulator in consensus_utility.items()
        },
        "mechanism": {
            stage: accumulator.summary()
            for stage, accumulator in mechanism_accumulators.items()
        },
    }


def phase0_decision(result):
    recoverability_positive = sum(
        result["consensus_utility"][stage]["net_recoverability"] > 0
        for stage in STAGES
    )
    directional_all = all(
        result["mechanism"][stage]["mean_consensus_ce_delta"] < 0
        for stage in STAGES
    )
    directional_fraction_stages = sum(
        result["mechanism"][stage]["ce_decrease_fraction"] >= 0.70
        for stage in STAGES
    )
    nondecreasing_stage_cams = sum(
        result["scores"]["virtual"][STAGE_TO_CAM[stage]]["Mean IoU"]
        >= result["scores"]["pass0"][STAGE_TO_CAM[stage]]["Mean IoU"]
        for stage in STAGES
    )
    positive_net_correction = sum(
        result["prediction_changes"][STAGE_TO_CAM[stage]]["net_corrected"] > 0
        for stage in STAGES
    )
    fused_delta_pp = 100 * (
        result["scores"]["virtual"]["fused"]["Mean IoU"]
        - result["scores"]["pass0"]["fused"]["Mean IoU"]
    )
    all_finite = all(
        result["mechanism"][stage]["all_finite"] for stage in STAGES
    )
    go = (
        recoverability_positive >= 2
        and directional_all
        and directional_fraction_stages >= 2
        and nondecreasing_stage_cams >= 2
        and positive_net_correction >= 2
        and fused_delta_pp > -0.05
        and all_finite
    )
    strong_go = (
        go
        and fused_delta_pp >= 0.20
        and sum(
            result["scores"]["virtual"][STAGE_TO_CAM[stage]]["Mean IoU"]
            > result["scores"]["pass0"][STAGE_TO_CAM[stage]]["Mean IoU"]
            for stage in STAGES
        ) >= 2
    )
    return {
        "recoverability_positive_stages": recoverability_positive,
        "directional_mean_ce_pass_all_stages": directional_all,
        "ce_decrease_fraction_ge_70_percent_stages": directional_fraction_stages,
        "nondecreasing_stage_cam_miou_stages": nondecreasing_stage_cams,
        "positive_net_correction_stages": positive_net_correction,
        "fused_miou_delta_percentage_points": fused_delta_pp,
        "all_finite": all_finite,
        "strong_go": strong_go,
        "signal": "CLRR_V2_SIGNAL_GO" if go else "CLRR_V2_SIGNAL_NOGO",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen BF16 Phase-0 audit requires CUDA")

    image_count = len(list((args.val_root / "img").glob("*.png")))
    mask_count = len(list((args.val_root / "mask").glob("*.png")))
    if image_count != 3418 or mask_count != 3418:
        raise ValueError(
            "BCSS validation must contain exactly 3418 images and masks; "
            f"found images={image_count}, masks={mask_count}"
        )

    state_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = importlib.import_module("network.resnet38_cls").Net_CAM(n_class=4)
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()
    manifest = classifier_manifest(model)
    torch.cuda.reset_peak_memory_stats()

    official_start = time.perf_counter()
    official_score = infer(
        model,
        str(args.val_root),
        4,
        official_infer_args(args.num_workers),
        thr=None,
        cam_weights=(0.6, 0.2, 0.2),
    )
    official_seconds = time.perf_counter() - official_start
    if official_score is None:
        raise RuntimeError("Released official inference failed")

    phase0 = run_phase0(model, args.val_root, args.num_workers)
    pass0_fused = phase0["scores"]["pass0"]["fused"]
    parity = {
        "mean_iou_absolute_difference": abs(
            float(official_score["Mean IoU"])
            - float(pass0_fused["Mean IoU"])
        ),
        "mean_dice_absolute_difference": abs(
            float(official_score["Mean Dice"])
            - float(pass0_fused["Mean Dice"])
        ),
    }
    parity["pass"] = (
        parity["mean_iou_absolute_difference"] <= 1e-12
        and parity["mean_dice_absolute_difference"] <= 1e-12
    )
    if not parity["pass"]:
        raise RuntimeError(f"Pass0 official inference parity failed: {parity}")
    decision = phase0_decision(phase0)

    result = {
        "scope": "BCSS validation only; frozen A0; no training",
        "test_evaluated": False,
        "eta": ETA,
        "base_commit": "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9",
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "size_bytes": args.checkpoint.stat().st_size,
            "sha256": sha256_file(args.checkpoint),
        },
        "validation": {
            "root": str(args.val_root.resolve()),
            "image_count": image_count,
            "mask_count": mask_count,
        },
        "classifier_structure": manifest,
        "official_pass0_score": official_score,
        "official_inference_parity": parity,
        "phase0": phase0,
        "decision": decision,
        "runtime": {
            "official_inference_seconds": official_seconds,
            "virtual_audit_seconds": phase0["seconds"],
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "precision": "official BF16 inference; detached FP32 feedback",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(result), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(json_ready(result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
