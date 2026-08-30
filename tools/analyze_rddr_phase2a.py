"""Evaluate the frozen RDDR Phase-2A C0/GS/RCS utility gate on BCSS validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.rddr_context import compute_rddr_dross_score, context_reliability  # noqa: E402
from network.resnet38_cls import Net_CAM  # noqa: E402
from tools.smoke_rddr_phase2a import load_pretrained  # noqa: E402
from tool import infer_fun, iouutils  # noqa: E402
from tool.GenDataset import Stage1_InferDataset  # noqa: E402
from tools.rddr_phase2a_analysis_common import (  # noqa: E402
    BACKGROUND,
    CAM_WEIGHTS,
    EXPECTED_VAL,
    TTA_TRANSFORMS,
    ComponentMetricAccumulator,
    SortedValidationDataset,
    ZoneMetricAccumulator,
    component_thresholds,
    minmax_normalize,
    official_histogram,
    paired_bootstrap_miou,
    presence_from_probability,
    scores_from_histogram,
    set_seed,
    sha256_file,
    write_csv,
    write_json,
)


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
IMPLEMENTATION_COMMIT = "6f45ac7676b2e7bd7ae21c23db3303de95e02c6c"
C0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
CONTEXT_HELPER_SHA256 = "1142ff8e8f95d3447012af9c4eb8f91eb923a48d5e8f840ea42098cc2f1de58b"
MODEL_SOURCE_SHA256 = "a6f6cf3a82c23d5a7a99c41c6f1348c118428aa6a508ee0dc71d7f44ac9f1f3d"
MILESTONES = (1, 5, 10, 15, 20, 25)


def load_model(mode, checkpoint, device="cuda"):
    model = Net_CAM(4, rddr_context_mode=mode)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError(
            f"{mode} checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model


def checkpoint_for_epoch(directory, epoch):
    directory = Path(directory)
    return (
        directory / "stage1_last.pth"
        if epoch == 25
        else directory / f"stage1_epoch_{epoch:04d}.pth"
    )


def cam_prediction(normalized_cam, presence):
    return np.argmax(
        normalized_cam * presence.reshape(4, 1, 1), axis=0
    ).astype(np.uint8)


def official_tta_predictions(model, image, output_size):
    cam_lists = {name: [] for name in ("CAM56", "CAM28_1", "CAM28_2", "CAMdeep")}
    probabilities = []
    for input_flip_dims, cam_flip_dims in TTA_TRANSFORMS:
        tta_image = torch.flip(image, dims=input_flip_dims) if input_flip_dims else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cam56, cam28_1, cam28_2, camdeep, probability = model.forward_cam(tta_image)
            upsampled = tuple(
                F.interpolate(
                    cam, output_size, mode="bilinear", align_corners=False
                )[0]
                for cam in (cam56, cam28_1, cam28_2, camdeep)
            )
        for name, cam in zip(cam_lists, upsampled):
            if cam_flip_dims:
                cam = torch.flip(cam, dims=cam_flip_dims)
            cam_lists[name].append(cam)
        probabilities.append(probability)
    averaged = {
        name: minmax_normalize(torch.stack(values).mean(0).float().cpu().numpy())
        for name, values in cam_lists.items()
    }
    presence = presence_from_probability(
        torch.stack(probabilities).mean(0).float().cpu().numpy()[0]
    )
    predictions = {
        name: cam_prediction(cam, presence) for name, cam in averaged.items()
    }
    fused = (
        CAM_WEIGHTS[0] * averaged["CAM28_1"]
        + CAM_WEIGHTS[1] * averaged["CAM28_2"]
        + CAM_WEIGHTS[2] * averaged["CAMdeep"]
    )
    predictions["Final"] = cam_prediction(fused, presence)
    return predictions


def official_inference_parity(models, dataset, val_root, count=8):
    """Call unchanged official infer() on the same small, sorted val subset."""
    count = min(count, len(dataset))
    native = Stage1_InferDataset(str(Path(val_root) / "img"), img_size=224)
    native.object = [str(path) for path in dataset.images[:count]]
    args = SimpleNamespace(dataset="bcss", img_size=224, num_workers=0, amp_dtype="bf16")
    records = {}
    official_scores = iouutils.scores
    for variant, model in models.items():
        captured = []

        def capture_scores(truths, predictions, n_class):
            captured.extend(np.array(prediction, copy=True) for prediction in predictions)
            return official_scores(truths, predictions, n_class)

        with patch.object(infer_fun, "Stage1_InferDataset", return_value=native), patch.object(
            iouutils, "scores", side_effect=capture_scores
        ):
            result = infer_fun.infer(model, str(val_root), 4, args)
        if result is None or len(captured) != count:
            raise AssertionError("Official inference parity invocation failed")
        mismatched = 0
        with torch.inference_mode():
            for index, expected in enumerate(captured):
                _, image, _, truth = dataset[index]
                actual = official_tta_predictions(model, image[None].cuda(), truth.shape)["Final"]
                mismatched += int((actual != expected).sum())
        records[variant] = {"images": count, "mismatched_prediction_pixels": mismatched}
        if mismatched:
            raise AssertionError(f"Official inference pixel parity failed: {variant}: {mismatched}")
    print(f"RDDR_OFFICIAL_INFERENCE_PARITY {records}", flush=True)
    return records


def canonical_diagnostics(model, image, output_size):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = model.forward_rddr_context_diagnostics(image)
        raw_logits = model.ic1(values["F28_raw"])
        rect_logits = model.ic1(values["F28_rect"])
        deep_logits = model.fc8(values["Ddeep"])
        raw_up = F.interpolate(
            raw_logits, output_size, mode="bilinear", align_corners=False
        )
        rect_up = F.interpolate(
            rect_logits, output_size, mode="bilinear", align_corners=False
        )
        deep_up = F.interpolate(
            deep_logits, output_size, mode="bilinear", align_corners=False
        )
    phase0_q = compute_rddr_dross_score(raw_up, deep_up)[0, 0].cpu().numpy()
    return (
        values,
        phase0_q,
        raw_up.argmax(1)[0].cpu().numpy(),
        F.relu(rect_up).argmax(1)[0].cpu().numpy(),
    )


class StageAccumulator:
    def __init__(self):
        self.histograms = {
            name: np.zeros((5, 5), dtype=np.int64)
            for name in ("CAM56", "CAM28_1", "CAM28_2", "CAMdeep", "Final")
        }
        self.per_image_final = []

    def update(self, truth, predictions):
        for name, prediction in predictions.items():
            histogram = official_histogram(truth, prediction)
            self.histograms[name] += histogram
            if name == "Final":
                self.per_image_final.append(histogram)

    def result(self):
        return {
            name: scores_from_histogram(histogram)
            for name, histogram in self.histograms.items()
        }


class TransitionAccumulator:
    def __init__(self, labels):
        self.data = {
            label: {"pixels": 0, "repair": 0, "harm": 0}
            for label in labels
        }

    def update(self, label, mask, truth, base_prediction, candidate_prediction):
        mask = np.asarray(mask, dtype=bool)
        row = self.data[label]
        row["pixels"] += int(mask.sum())
        row["repair"] += int(
            (mask & (base_prediction != truth) & (candidate_prediction == truth)).sum()
        )
        row["harm"] += int(
            (mask & (base_prediction == truth) & (candidate_prediction != truth)).sum()
        )

    def rows(self, variant, kind):
        output = []
        for label, row in self.data.items():
            output.append(
                {
                    "variant": variant,
                    "analysis": kind,
                    "group": label,
                    **row,
                    "repair_rate": row["repair"] / max(row["pixels"], 1),
                    "harm_rate": row["harm"] / max(row["pixels"], 1),
                    "net_repair": (row["repair"] - row["harm"])
                    / max(row["pixels"], 1),
                }
            )
        return output


class ContextAccumulator:
    def __init__(self, variant):
        self.variant = variant
        self.before_sq = 0.0
        self.after_sq = 0.0
        self.context_count = 0
        self.q_values = []
        self.reliability_values = []

    def update(self, values):
        before = values["context_before"].float()
        after = values["context_after"].float()
        q = values["q"].float()
        reliability = values["reliability"].float().expand_as(q)
        self.before_sq += float(before.square().sum().item())
        self.after_sq += float(after.square().sum().item())
        self.context_count += before.numel()
        self.q_values.append(q.flatten().cpu().numpy())
        self.reliability_values.append(reliability.flatten().cpu().numpy())

    def result(self):
        q = np.concatenate(self.q_values)
        reliability = np.concatenate(self.reliability_values)
        return {
            "variant": self.variant,
            "pixels": int(q.size),
            "q_mean": float(q.mean()),
            "q_std": float(q.std()),
            "q_min": float(q.min()),
            "q_max": float(q.max()),
            "mean_reliability": float(reliability.mean()),
            "mean_suppression": float(1.0 - reliability.mean()),
            "reliability_std": float(reliability.std()),
            "reliability_min": float(reliability.min()),
            "reliability_p05": float(np.quantile(reliability, 0.05)),
            "reliability_p25": float(np.quantile(reliability, 0.25)),
            "reliability_p50": float(np.quantile(reliability, 0.50)),
            "reliability_p75": float(np.quantile(reliability, 0.75)),
            "reliability_p95": float(np.quantile(reliability, 0.95)),
            "reliability_max": float(reliability.max()),
            "context_before_rms": math.sqrt(
                self.before_sq / max(self.context_count, 1)
            ),
            "context_after_rms": math.sqrt(
                self.after_sq / max(self.context_count, 1)
            ),
            "context_rms_ratio": math.sqrt(
                self.after_sq / max(self.before_sq, 1.0e-30)
            ),
        }


class QuintileAccumulator:
    """Cache fixed-C0 populations for the preregistered quintile audit."""

    def __init__(self):
        self.q_full = []
        self.q_feature = []
        self.accuracy_delta = {name: [] for name in ("GS", "RCS")}
        self.context = {
            name: {key: [] for key in ("reliability", "before_power", "after_power")}
            for name in ("GS", "RCS")
        }

    def update(self, truth, predictions, canonical):
        foreground = truth < BACKGROUND
        q_full = canonical["C0"][1]
        self.q_full.append(q_full[foreground].astype(np.float32, copy=True))
        c0_correct = predictions["C0"]["Final"] == truth
        for variant in ("GS", "RCS"):
            candidate_correct = predictions[variant]["Final"] == truth
            delta = candidate_correct.astype(np.int8) - c0_correct.astype(np.int8)
            self.accuracy_delta[variant].append(delta[foreground].copy())

        q_feature = canonical["C0"][0]["q"].float()[0, 0].cpu().numpy()
        foreground_feature = F.interpolate(
            torch.from_numpy(foreground.astype(np.float32))[None, None],
            size=q_feature.shape,
            mode="nearest",
        )[0, 0].bool().numpy()
        self.q_feature.append(q_feature[foreground_feature].astype(np.float32, copy=True))
        for variant in ("GS", "RCS"):
            values = canonical[variant][0]
            q_shape = values["q"].shape
            reliability = values["reliability"].float().expand(q_shape)[0, 0]
            before_power = values["context_before"].float().square().mean(1)[0]
            after_power = values["context_after"].float().square().mean(1)[0]
            for key, tensor in (
                ("reliability", reliability),
                ("before_power", before_power),
                ("after_power", after_power),
            ):
                array = tensor.cpu().numpy()
                self.context[variant][key].append(
                    array[foreground_feature].astype(np.float32, copy=True)
                )

    def result(self):
        q_full = np.concatenate(self.q_full)
        q_feature = np.concatenate(self.q_feature)
        thresholds_full = np.quantile(q_full, (0.2, 0.4, 0.6, 0.8))
        thresholds_feature = np.quantile(q_feature, (0.2, 0.4, 0.6, 0.8))
        bins_full = np.digitize(q_full, thresholds_full, right=True)
        bins_feature = np.digitize(q_feature, thresholds_feature, right=True)
        rows = []
        for variant in ("GS", "RCS"):
            accuracy_delta = np.concatenate(self.accuracy_delta[variant])
            context = {
                key: np.concatenate(values)
                for key, values in self.context[variant].items()
            }
            for index, quintile in enumerate(("Q1", "Q2", "Q3", "Q4", "Q5")):
                full_mask = bins_full == index
                feature_mask = bins_feature == index
                before_rms = float(
                    np.sqrt(context["before_power"][feature_mask].mean())
                )
                after_rms = float(
                    np.sqrt(context["after_power"][feature_mask].mean())
                )
                rows.append(
                    {
                        "variant": variant,
                        "quintile": quintile,
                        "prediction_pixels": int(full_mask.sum()),
                        "context_pixels": int(feature_mask.sum()),
                        "frozen_c0_q_mean_full": float(q_full[full_mask].mean()),
                        "frozen_c0_q_mean_feature": float(
                            q_feature[feature_mask].mean()
                        ),
                        "mean_reliability": float(
                            context["reliability"][feature_mask].mean()
                        ),
                        "context_before_rms": before_rms,
                        "context_after_rms": after_rms,
                        "context_rms_ratio": after_rms / max(before_rms, 1.0e-30),
                        "prediction_accuracy_delta_vs_c0": float(
                            accuracy_delta[full_mask].mean()
                        ),
                    }
                )
        return {
            "thresholds_full": thresholds_full.tolist(),
            "thresholds_feature": thresholds_feature.tolist(),
            "rows": rows,
        }


def distribution(values, epoch, source):
    values = np.concatenate(values).astype(np.float64, copy=False)
    return {
        "source": source,
        "epoch": epoch,
        "pixels": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def compute_dynamics(gs_dir, rcs_dir, c0_checkpoint, phase1_dir, loader):
    rows = []
    gamma_rows = []
    checkpoints = [("Phase0-C0", 25, "none", Path(c0_checkpoint))]
    for source, mode, directory in (
        ("GS", "global", gs_dir),
        ("RCS", "receiver", rcs_dir),
    ):
        checkpoints += [
            (source, epoch, mode, checkpoint_for_epoch(directory, epoch))
            for epoch in MILESTONES
        ]
    for source, epoch, mode, checkpoint in checkpoints:
        model = load_model(mode, checkpoint)
        q_values = []
        reliability_values = []
        with torch.inference_mode():
            for _, image, _, _ in loader:
                image = image.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = model.forward_rddr_context_diagnostics(image)
                    q = values["q"]
                    reliability = values["reliability"].expand_as(q)
                q_values.append(q.float().flatten().cpu().numpy())
                reliability_values.append(
                    reliability.float().flatten().cpu().numpy()
                )
        row = distribution(q_values, epoch, source)
        reliability = np.concatenate(reliability_values)
        row["mean_reliability"] = float(reliability.mean())
        row["mean_suppression"] = float(1.0 - reliability.mean())
        rows.append(row)
        gamma_context = float(model.hfrm_28_1.gamma_context.detach().item())
        gamma_veto = float(model.hfrm_28_1.gamma_veto.detach().item())
        gamma_rows.append(
            {
                "variant": source,
                "epoch": epoch,
                "gamma_context": gamma_context,
                "gamma_veto": gamma_veto,
                "mean_reliability": row["mean_reliability"],
                "effective_context_scale": abs(gamma_context)
                * row["mean_reliability"],
            }
        )
        del model
        torch.cuda.empty_cache()
        print(f"RDDR_Q_DYNAMICS source={source} epoch={epoch}", flush=True)
    phase1_path = Path(phase1_dir) / "rddr_phase1_q_dynamics.csv"
    if not phase1_path.is_file():
        raise FileNotFoundError(phase1_path)
    with phase1_path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if raw["source"] != "DD":
                continue
            rows.append(
                {
                    key: (
                        "Phase1-DD"
                        if key == "source"
                        else int(value)
                        if key in {"epoch", "pixels"}
                        else float(value)
                    )
                    for key, value in raw.items()
                }
            )
    return rows, gamma_rows


def audit_epoch0(pretrained, loader):
    """Reconstruct seed42 initialization; observation only, no optimizer steps.

    This is a retrospective eval-mode validation probe, not a saved epoch0
    training measurement. Net_CAM adds no initialization beyond Net.
    """
    set_seed(42)
    model = Net_CAM(4, rddr_context_mode="none").cuda()
    pretrained_audit = load_pretrained(model, pretrained)
    model.eval()
    accumulators = {name: ContextAccumulator(name) for name in ("GS", "RCS")}
    q_values = []
    with torch.inference_mode():
        for _, image, _, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = model.forward_rddr_context_diagnostics(image.cuda())
                q, before = values["q"], values["context_before"]
                q_values.append(q.float().flatten().cpu().numpy())
                for name, mode in (("GS", "global"), ("RCS", "receiver")):
                    reliability = context_reliability(q, mode)
                    after = reliability.to(before.dtype) * before
                    accumulators[name].update({
                        "q": q, "reliability": reliability,
                        "context_before": before, "context_after": after,
                    })
    result = {
        "method": "Retrospective seed42/pretrained reconstruction, eval mode, validation images, batch20 BF16; zero training steps",
        "pretrained": pretrained_audit,
        "q": distribution(q_values, 0, "Reconstructed-init"),
        "context_strength": {name: acc.result() for name, acc in accumulators.items()},
        "gamma_context": float(model.hfrm_28_1.gamma_context.item()),
        "gamma_veto": float(model.hfrm_28_1.gamma_veto.item()),
    }
    del model
    torch.cuda.empty_cache()
    print("RDDR_EPOCH0_AUDIT completed (no training)", flush=True)
    return result


def load_training_curves(gs_dir, rcs_dir):
    rows = []
    for variant, directory in (("GS", gs_dir), ("RCS", rcs_dir)):
        with (Path(directory) / "training_curve.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"variant": variant, **row})
    return rows


def load_optimizer_audits(gs_dir, rcs_dir):
    return {
        variant: json.loads((Path(directory) / "optimizer_audit.json").read_text())
        for variant, directory in (("GS", gs_dir), ("RCS", rcs_dir))
    }


def training_runtime(directory):
    text = (Path(directory) / "train.log").read_text(errors="replace")
    matches = re.findall(r"Total Training Time:\s*([0-9.]+)s", text)
    return float(matches[-1]) if matches else float("nan")


def phase2a_decision(gates, full=True):
    if not full:
        return "RDDR_PHASE2A_SMOKE_ONLY"
    gate_a = gates["A"]["pass"]
    gate_b = gates["B"]["pass"]
    gate_c = gates["C"]["pass"]
    gate_d = gates["D"]["pass"]
    if gate_a and gate_b and gate_c and gate_d:
        return "RDDR_PHASE2A_GO"
    if gate_a and gate_c and gate_d and not gate_b:
        return "CONTEXT_REDUCTION_WORKS_SPATIAL_SPECIFICITY_FAIL"
    if not gate_c:
        return "CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE"
    if gate_d and not gate_a:
        return "LOCAL_CH_HARM_REDUCED_NO_GLOBAL_GAIN"
    return "RDDR_PHASE2A_NOGO"


def render_report(summary):
    metric = summary["metrics"]
    zones = summary["zones"]
    objects = summary["object_size"]
    fixed = summary["fixed_strata"]
    gates = summary["gates"]
    bootstrap = summary["bootstrap"]
    lines = [
        "# RDDR Phase-2A Dross-Aware Context Suppression Report",
        "",
        "## 1. Frozen provenance and commands",
        "",
        f"- Implementation commit: `{summary['implementation_commit']}`",
        f"- Evaluation commit: `{summary['evaluation_commit']}`",
        f"- Pure A0 commit: `{summary['a0_commit']}`",
        f"- C0 checkpoint SHA256: `{summary['checkpoint_sha256']['C0']}`",
        f"- GS checkpoint SHA256: `{summary['checkpoint_sha256']['GS']}`",
        f"- RCS checkpoint SHA256: `{summary['checkpoint_sha256']['RCS']}`",
        f"- Locked JSD helper SHA256: `{summary['engineering']['context_helper_sha256']}`",
        f"- Locked model source SHA256: `{summary['engineering']['model_source_sha256']}`",
        f"- Dataset/split: {summary['images']} BCSS validation images; no test or LUAD access.",
        "",
        "```bash",
        summary["commands"]["training"],
        summary["commands"]["analysis"],
        "```",
        "",
        "## 2. Architecture, capacity, and semantic-preservation contract",
        "",
        "Only the HFRM28_1 context residual is scaled. The original feature F and "
        "semantic veto residual remain untouched.",
        "",
        "```text",
        "C0: F' = F + gamma_sem*R_sem + gamma_ctx*R_ctx",
        "GS: F' = F + gamma_sem*R_sem + gamma_ctx*mean(1-q)*R_ctx",
        "RCS: F' = F + gamma_sem*R_sem + gamma_ctx*(1-q_i)*R_ctx",
        "```",
        "",
        f"- Total parameters (all variants): {summary['engineering']['parameters']['C0']}",
        "- Additional trainable parameters: 0",
        f"- Initial zero-gamma max absolute difference: {summary['engineering']['identity_max_abs_diff']:.8g}",
        f"- Same-checkpoint pre-HFRM feature max difference: {summary['engineering']['semantic_preservation']['max_abs_diff']:.8g}",
        f"- Same-checkpoint pre-HFRM feature cosine: {summary['engineering']['semantic_preservation']['cosine']:.9f}",
        "",
        "## 3. Training equivalence",
        "",
        "GS and RCS use seed42, batch20, BF16, epoch0→25, official pretrained "
        "weights, released augmentation, loss 0.10/0.15/0.25/0.50, released "
        "PolyOptimizer/LR schedule, and Epoch-25 FINAL checkpoints. Training never "
        "evaluated validation or test.",
        "",
        "## 4. Overall metrics and CAM hierarchy",
        "",
        "Official three-view TTA is averaged in the native output dtype before FP32 "
        "normalization. BCSS presence thresholds are [0.8,0.9,0.8,0.6], with argmax "
        "fallback when none pass; final fusion is 0.6/0.2/0.2 (CAM56 diagnostic only). "
        "The initial copied audit helper averaged after FP32 conversion; this was "
        "corrected before this evaluation. Original infer()/metric/model files were not changed.",
        "",
        f"Direct pixel parity against unchanged official infer(): {summary['engineering']['official_inference_parity']}",
        "",
        "Metric retains official GT-background overwrite; foreground classes 0–3 "
        "enter the mean. Absent-class IoU is NaN/excluded; absent-class Dice is 0. "
        "Boundary masks include foreground-to-foreground transitions only. Size "
        "bins use per-class 8-connected GT-component area q25/q75; recall is "
        "pixel-weighted and size mIoU is mask-restricted, not instance IoU.",
        "",
        "| Variant | CAM56 mIoU | CAM28_1 mIoU | CAM28_2 mIoU | CAMdeep mIoU | Final mIoU | Final mDice |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("C0", "GS", "RCS"):
        row = metric[variant]
        lines.append(
            f"| {variant} | {100*row['CAM56']['mIoU']:.4f} | "
            f"{100*row['CAM28_1']['mIoU']:.4f} | {100*row['CAM28_2']['mIoU']:.4f} | "
            f"{100*row['CAMdeep']['mIoU']:.4f} | {100*row['Final']['mIoU']:.4f} | "
            f"{100*row['Final']['mDice']:.4f} |"
        )
    lines += [
        "",
        "## 5. Boundary, interior, and object size",
        "",
        "| Variant | Boundary acc | Boundary mIoU | Interior acc | Interior mIoU | Small recall/mIoU | Medium recall/mIoU | Large recall/mIoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("C0", "GS", "RCS"):
        boundary = zones[variant]["boundary_le_7"]
        interior = zones[variant]["interior_gt_7"]
        size = objects[variant]
        lines.append(
            f"| {variant} | {100*boundary['accuracy']:.4f} | {100*boundary['restricted_mIoU']:.4f} | "
            f"{100*interior['accuracy']:.4f} | {100*interior['restricted_mIoU']:.4f} | "
            f"{100*size['small']['historical_component_recall']:.4f}/{100*size['small']['diagnostic_size_restricted_mIoU']:.4f} | "
            f"{100*size['medium']['historical_component_recall']:.4f}/{100*size['medium']['diagnostic_size_restricted_mIoU']:.4f} | "
            f"{100*size['large']['historical_component_recall']:.4f}/{100*size['large']['diagnostic_size_restricted_mIoU']:.4f} |"
        )
    lines += [
        "",
        "## 6. Per-class IoU",
        "",
        "| Variant | Class 0 | Class 1 | Class 2 | Class 3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("C0", "GS", "RCS"):
        values = metric[variant]["Final"]["class_iou"]
        lines.append(
            f"| {variant} | {100*values['0']:.4f} | {100*values['1']:.4f} | "
            f"{100*values['2']:.4f} | {100*values['3']:.4f} |"
        )
    lines += [
        "",
        "## 7. q dynamics",
        "",
        "q is JS/ln(2), computed at 28x28; these dynamics include all grid positions. "
        "Phase1-DD rows are imported observations, not re-trained models.",
        "",
        "| Source | Epoch | Mean | Std | Min | p05 | p25 | p50 | p75 | p95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["q_dynamics"]:
        lines.append(
            f"| {row['source']} | {row['epoch']} | {row['mean']:.6f} | {row['std']:.6f} | "
            f"{row['min']:.6f} | {row['p05']:.6f} | {row['p25']:.6f} | {row['p50']:.6f} | "
            f"{row['p75']:.6f} | {row['p95']:.6f} | {row['max']:.6f} |"
        )
    lines += [
        "",
        "## 8. Effective context strength",
        "",
        "| Variant | Mean reliability | Mean suppression | r p05/p25/p50/p75/p95 | Context RMS before | after | ratio |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for variant in ("GS", "RCS"):
        row = summary["context_strength"][variant]
        lines.append(
            f"| {variant} | {row['mean_reliability']:.6f} | {row['mean_suppression']:.6f} | "
            f"{row['reliability_p05']:.6f}/{row['reliability_p25']:.6f}/{row['reliability_p50']:.6f}/"
            f"{row['reliability_p75']:.6f}/{row['reliability_p95']:.6f} | "
            f"{row['context_before_rms']:.6f} | {row['context_after_rms']:.6f} | "
            f"{row['context_rms_ratio']:.6f} |"
        )
    lines += [
        "",
        "## 9. gamma dynamics and compensation",
        "",
        "| Variant | Epoch | gamma_context | gamma_veto | Mean r | EffectiveContextScale |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["gamma_dynamics"]:
        lines.append(
            f"| {row['variant']} | {row['epoch']} | {row['gamma_context']:+.6f} | "
            f"{row['gamma_veto']:+.6f} | {row['mean_reliability']:.6f} | "
            f"{row['effective_context_scale']:.6f} |"
        )
    lines += [
        "",
        "## 10. Frozen Phase-0 Top20 / Bottom80",
        "",
        "| Variant | Top20 repair/harm/net | Bottom80 repair/harm/net |",
        "|---|---:|---:|",
    ]
    for variant in ("GS", "RCS"):
        top = fixed[variant]["Top20"]
        bottom = fixed[variant]["Bottom80"]
        lines.append(
            f"| {variant} | {100*top['repair_rate']:.4f}/{100*top['harm_rate']:.4f}/{100*top['net_repair']:+.4f} pp | "
            f"{100*bottom['repair_rate']:.4f}/{100*bottom['harm_rate']:.4f}/{100*bottom['net_repair']:+.4f} pp |"
        )
    lines += [
        "",
        "## 11. Frozen C0 CH-transition groups",
        "",
        "| Variant/group | Repair | Harm | Net change |",
        "|---|---:|---:|---:|",
    ]
    for variant in ("GS", "RCS"):
        for row in summary["ch_transition"][variant]:
            lines.append(
                f"| {variant}/{row['group']} | {100*row['repair_rate']:.4f} pp | "
                f"{100*row['harm_rate']:.4f} pp | {100*row['net_repair']:+.4f} pp |"
            )
    lines += [
        "",
        "## 12. Frozen-C0 q-quintile selectivity",
        "",
        "All bins are defined from the locked C0, never from candidate q. "
        "Prediction bins use full-resolution foreground q; context bins use "
        "28x28 C0 q with nearest-resized foreground masks and separately computed "
        "quintiles. They are resolution-specific populations, not identical pixels. "
        "Exact thresholds and counts are in the JSON/CSV.",
        "",
        "| Variant | Quintile | Mean r | Context RMS before | after | ratio | Accuracy delta vs C0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["quintile_analysis"]:
        lines.append(
            f"| {row['variant']} | {row['quintile']} | {row['mean_reliability']:.6f} | "
            f"{row['context_before_rms']:.6f} | {row['context_after_rms']:.6f} | "
            f"{row['context_rms_ratio']:.6f} | {100*row['prediction_accuracy_delta_vs_c0']:+.4f} pp |"
        )
    lines += [
        "",
        "## 13. Paired image-level bootstrap",
        "",
        "| Comparison | Observed delta mIoU | Bootstrap mean | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for name in ("RCS-C0", "RCS-GS", "GS-C0"):
        row = bootstrap[name]
        lines.append(
            f"| {name} | {100*row['observed_delta_mIoU']:+.4f} pp | "
            f"{100*row['bootstrap_mean']:+.4f} pp | "
            f"[{100*row['ci95_low']:+.4f}, {100*row['ci95_high']:+.4f}] pp |"
        )
    lines += [
        "",
        "## 14. Preregistered gates",
        "",
        "| Gate | Requirement | Result | Pass |",
        "|---|---|---|:---:|",
    ]
    for name in ("A", "B", "C", "D"):
        row = gates[name]
        lines.append(
            f"| {name} | {row['requirement']} | {row['result']} | {row['pass']} |"
        )
    lines += [
        "",
        "## 15. Scientific interpretation",
        "",
        summary["scientific_interpretation"],
        "",
        "## 16. Engineering and artifact record",
        "",
        f"- Main final-checkpoint evaluation: {summary['runtime']['seconds']/60:.2f} min; "
        f"complete evaluation including dynamics/bootstrap: {summary['runtime']['total_seconds']/60:.2f} min; peak CUDA memory "
        f"{summary['runtime']['peak_cuda_memory_bytes']/2**30:.3f} GiB.",
        f"- GS/RCS training runtime: {summary['runtime']['training_seconds']['GS']/60:.2f} / "
        f"{summary['runtime']['training_seconds']['RCS']/60:.2f} min.",
        "- All required curves, q/context/gamma, fixed-strata, CH, quintile, per-class, "
        "bootstrap, optimizer, per-image, and summary artifacts were generated.",
        "- No BCSS test, LUAD, best-epoch selection, or post-hoc tuning was used.",
        "",
        "## 17. Epoch0 initialization observation",
        "",
        summary["epoch0_audit"]["method"],
        "",
        "This reconstructs initialization after training has finished; it is not "
        "a contemporaneous training log. Shared raw features and q are computed "
        "once, then the frozen GS/RCS context scaling is applied. Initial gammas "
        "are zero, so attenuated context does not yet contribute to the output.",
        "",
        "| Variant | Mean r | Mean suppression | Context RMS before | after | ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary["epoch0_audit"]["context_strength"].items():
        lines.append(
            f"| {variant} | {row['mean_reliability']:.6f} | {row['mean_suppression']:.6f} | "
            f"{row['context_before_rms']:.6f} | {row['context_after_rms']:.6f} | {row['context_rms_ratio']:.6f} |"
        )
    lines += [
        "",
        f"DECISION = {summary['decision']}",
        "",
    ]
    return "\n".join(lines)

def semantic_preservation_audit(receiver_model, checkpoint, image):
    shadow = load_model("none", checkpoint)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        receiver = receiver_model.forward_rddr_context_diagnostics(image)["F28_raw"].float()
        reference = shadow.forward_rddr_context_diagnostics(image)["F28_raw"].float()
    difference = float((receiver - reference).abs().max().item())
    cosine = float(
        F.cosine_similarity(receiver.flatten(), reference.flatten(), dim=0).item()
    )
    del shadow, receiver, reference
    torch.cuda.empty_cache()
    return {"max_abs_diff": difference, "cosine": cosine}


def validate_run_artifacts(directory):
    directory = Path(directory)
    if (directory / "exit_code").read_text().strip() != "0":
        raise AssertionError(f"Training failed: {directory}")
    with (directory / "training_curve.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 25 or int(rows[-1]["epoch"]) != 25:
        raise AssertionError(f"Expected a complete 25-epoch curve: {directory}")
    for epoch in MILESTONES:
        path = checkpoint_for_epoch(directory, epoch)
        if not path.is_file():
            raise FileNotFoundError(path)
    return rows


def main():
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-checkpoint", required=True)
    parser.add_argument("--gs-dir", required=True)
    parser.add_argument("--rcs-dir", required=True)
    parser.add_argument("--phase0-dir", required=True)
    parser.add_argument("--phase1-dir", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--smoke-json", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    if (output / "rddr_phase2a_summary.json").exists():
        raise FileExistsError("Use a new output directory; existing results are immutable")
    output.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads(
        (Path(args.phase0_dir) / "rddr_phase0_summary.json").read_text()
    )
    if phase0["decision"] != "RDDR_PHASE0_GO":
        raise AssertionError("Phase-0 GO artifact is required")
    if sha256_file(args.c0_checkpoint) != C0_SHA256:
        raise AssertionError("C0 checkpoint SHA256 mismatch")
    helper_sha = sha256_file(REPOSITORY_ROOT / "network" / "rddr_context.py")
    model_source_sha = sha256_file(REPOSITORY_ROOT / "network" / "resnet38_cls.py")
    if helper_sha != CONTEXT_HELPER_SHA256:
        raise AssertionError("Locked RDDR Phase-2A JSD helper SHA256 mismatch")
    if model_source_sha != MODEL_SOURCE_SHA256:
        raise AssertionError("Locked RDDR Phase-2A model source SHA256 mismatch")
    validate_run_artifacts(args.gs_dir)
    validate_run_artifacts(args.rcs_dir)
    set_seed(42)
    phase0_threshold = float(phase0["thresholds"]["S_JS"]["0.2"]) / math.log(2.0)
    checkpoints = {
        "C0": Path(args.c0_checkpoint),
        "GS": checkpoint_for_epoch(args.gs_dir, 25),
        "RCS": checkpoint_for_epoch(args.rcs_dir, 25),
    }
    checkpoint_sha = {name: sha256_file(path) for name, path in checkpoints.items()}
    dataset = SortedValidationDataset(args.val_root, max_images=args.max_images)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    q_loader = DataLoader(
        dataset, batch_size=20, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    thresholds = component_thresholds(args.val_root)
    models = {
        "C0": load_model("none", checkpoints["C0"]),
        "GS": load_model("global", checkpoints["GS"]),
        "RCS": load_model("receiver", checkpoints["RCS"]),
    }
    inference_parity = official_inference_parity(models, dataset, args.val_root)
    sample_image = dataset[0][1].unsqueeze(0).cuda()
    semantic_preservation = semantic_preservation_audit(
        models["RCS"], checkpoints["RCS"], sample_image
    )
    del sample_image
    if semantic_preservation["max_abs_diff"] != 0.0:
        raise AssertionError("Receiver mode modified F before HFRM28_1")

    stages = {name: StageAccumulator() for name in models}
    zones = {name: ZoneMetricAccumulator() for name in models}
    components = {
        name: ComponentMetricAccumulator(thresholds) for name in models
    }
    context = {name: ContextAccumulator(name) for name in ("GS", "RCS")}
    quintiles = QuintileAccumulator()
    fixed = {
        name: TransitionAccumulator(("Top20", "Bottom80"))
        for name in ("GS", "RCS")
    }
    ch_groups = (
        "Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct"
    )
    ch = {
        name: TransitionAccumulator(ch_groups) for name in ("GS", "RCS")
    }
    per_image_rows = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.time()
    with torch.inference_mode():
        for index, (image_ids, image, originals, truths) in enumerate(loader, start=1):
            image_id = image_ids[0]
            image = image.cuda(non_blocking=True)
            truth = truths[0].numpy().astype(np.uint8)
            output_size = truth.shape
            predictions = {
                name: official_tta_predictions(model, image, output_size)
                for name, model in models.items()
            }
            canonical = {
                name: canonical_diagnostics(model, image, output_size)
                for name, model in models.items()
            }
            for name in models:
                stages[name].update(truth, predictions[name])
                zones[name].update(truth, predictions[name]["Final"])
                components[name].update(truth, predictions[name]["Final"])
            for name in ("GS", "RCS"):
                context[name].update(canonical[name][0])
            quintiles.update(truth, predictions, canonical)

            foreground = truth < BACKGROUND
            c0_q = canonical["C0"][1]
            strata = {
                "Top20": foreground & (c0_q >= phase0_threshold),
                "Bottom80": foreground & (c0_q < phase0_threshold),
            }
            c0_final = predictions["C0"]["Final"]
            for name in ("GS", "RCS"):
                candidate = predictions[name]["Final"]
                for label, mask in strata.items():
                    fixed[name].update(label, mask, truth, c0_final, candidate)

            raw_prediction = canonical["C0"][2]
            rect_prediction = canonical["C0"][3]
            group_masks = {
                "Corrected_by_CH": foreground & (raw_prediction != truth) & (rect_prediction == truth),
                "Still_Wrong": foreground & (raw_prediction != truth) & (rect_prediction != truth),
                "Harmed_by_CH": foreground & (raw_prediction == truth) & (rect_prediction != truth),
                "Stable_Correct": foreground & (raw_prediction == truth) & (rect_prediction == truth),
            }
            for name in ("GS", "RCS"):
                for label, mask in group_masks.items():
                    ch[name].update(
                        label, mask, truth, c0_final, predictions[name]["Final"]
                    )
            row = {"image_id": image_id}
            for name in models:
                histogram = stages[name].per_image_final[-1]
                for y in range(5):
                    for x in range(5):
                        row[f"{name}_hist_{y}_{x}"] = int(histogram[y, x])
            per_image_rows.append(row)
            if index % 100 == 0 or index == len(dataset):
                print(f"RDDR_PHASE2A_EVAL {index}/{len(dataset)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started

    metrics = {name: accumulator.result() for name, accumulator in stages.items()}
    zone_results = {name: accumulator.result() for name, accumulator in zones.items()}
    object_results = {
        name: accumulator.result() for name, accumulator in components.items()
    }
    if len(dataset) == EXPECTED_VAL and abs(metrics["C0"]["Final"]["mIoU"] - 0.673104) > 1.0e-4:
        raise AssertionError("C0 validation parity check failed")
    context_results = {
        name: accumulator.result() for name, accumulator in context.items()
    }
    context_rows = list(context_results.values())
    quintile_result = quintiles.result()
    quintile_rows = quintile_result["rows"]
    fixed_rows = [
        row for name, accumulator in fixed.items()
        for row in accumulator.rows(name, "fixed_phase0_strata")
    ]
    ch_rows = [
        row for name, accumulator in ch.items()
        for row in accumulator.rows(name, "c0_ch_groups")
    ]
    fixed_summary = {
        name: {
            row["group"]: row
            for row in accumulator.rows(name, "fixed_phase0_strata")
        }
        for name, accumulator in fixed.items()
    }
    ch_summary = {
        name: {
            row["group"]: row
            for row in accumulator.rows(name, "c0_ch_groups")
        }
        for name, accumulator in ch.items()
    }
    histograms = {
        name: np.asarray(accumulator.per_image_final)
        for name, accumulator in stages.items()
    }
    bootstrap = {}
    bootstrap_values = {}
    for label, base, candidate in (
        ("RCS-C0", "C0", "RCS"),
        ("RCS-GS", "GS", "RCS"),
        ("GS-C0", "C0", "GS"),
    ):
        bootstrap[label], bootstrap_values[label] = paired_bootstrap_miou(
            histograms[base], histograms[candidate],
            args.bootstrap_resamples, seed=42,
        )
    bootstrap_rows = [
        {
            "replicate": index,
            "RCS_minus_C0": bootstrap_values["RCS-C0"][index],
            "RCS_minus_GS": bootstrap_values["RCS-GS"][index],
            "GS_minus_C0": bootstrap_values["GS-C0"][index],
        }
        for index in range(args.bootstrap_resamples)
    ]
    training_rows = load_training_curves(args.gs_dir, args.rcs_dir)
    optimizer_audits = load_optimizer_audits(args.gs_dir, args.rcs_dir)
    q_rows, gamma_rows = compute_dynamics(
        args.gs_dir, args.rcs_dir, args.c0_checkpoint,
        args.phase1_dir, q_loader,
    )
    epoch0 = audit_epoch0(args.pretrained, q_loader)
    q_rows.insert(0, epoch0["q"])

    per_class_rows = []
    for variant in ("C0", "GS", "RCS"):
        for class_id in range(4):
            per_class_rows.append(
                {
                    "variant": variant,
                    "class": class_id,
                    **{
                        f"{stage}_IoU": metrics[variant][stage]["class_iou"][str(class_id)]
                        for stage in ("CAM56", "CAM28_1", "CAM28_2", "CAMdeep", "Final")
                    },
                }
            )

    rcs_c0 = bootstrap["RCS-C0"]
    rcs_gs = bootstrap["RCS-GS"]
    gate_a = (
        metrics["RCS"]["Final"]["mIoU"] > metrics["C0"]["Final"]["mIoU"]
        and rcs_c0["ci95_low"] >= 0.0
    )
    specificity_fallback = (
        metrics["RCS"]["CAM28_1"]["mIoU"] > metrics["GS"]["CAM28_1"]["mIoU"]
        and fixed_summary["RCS"]["Top20"]["net_repair"]
        > fixed_summary["GS"]["Top20"]["net_repair"]
    )
    gate_b = (
        metrics["RCS"]["Final"]["mIoU"] > metrics["GS"]["Final"]["mIoU"]
        and (rcs_gs["ci95_low"] >= 0.0 or specificity_fallback)
    )
    interior_delta = (
        zone_results["RCS"]["interior_gt_7"]["accuracy"]
        - zone_results["C0"]["interior_gt_7"]["accuracy"]
    )
    large_delta = (
        object_results["RCS"]["large"]["diagnostic_size_restricted_mIoU"]
        - object_results["C0"]["large"]["diagnostic_size_restricted_mIoU"]
    )
    cam_delta = (
        metrics["RCS"]["CAM28_1"]["mIoU"]
        - metrics["C0"]["CAM28_1"]["mIoU"]
    )
    gate_c = cam_delta >= 0.0 and interior_delta >= -0.001 and large_delta >= -0.002
    harmed_rcs = ch_summary["RCS"]["Harmed_by_CH"]["net_repair"]
    harmed_gs = ch_summary["GS"]["Harmed_by_CH"]["net_repair"]
    stable_rcs = ch_summary["RCS"]["Stable_Correct"]["net_repair"]
    gate_d = harmed_rcs > 0.0 and harmed_rcs > harmed_gs and stable_rcs >= -0.001
    gates = {
        "A": {
            "requirement": "RCS mIoU > C0 and RCS-C0 CI low >= 0",
            "result": f"delta={rcs_c0['observed_delta_mIoU']:+.6f}, low={rcs_c0['ci95_low']:+.6f}",
            "pass": gate_a,
        },
        "B": {
            "requirement": "RCS > GS with nonnegative CI low or CAM28_1+Top20 fallback",
            "result": f"delta={rcs_gs['observed_delta_mIoU']:+.6f}, low={rcs_gs['ci95_low']:+.6f}, fallback={specificity_fallback}",
            "pass": gate_b,
        },
        "C": {
            "requirement": "RCS CAM28_1 >= C0, interior >= -0.10 pp, large mIoU >= -0.20 pp",
            "result": f"CAM={cam_delta:+.6f}, interior={interior_delta:+.6f}, large={large_delta:+.6f}",
            "pass": gate_c,
        },
        "D": {
            "requirement": "RCS Harmed-by-CH > 0 and > GS; Stable-Correct >= -0.10 pp",
            "result": f"RCS_harmed={harmed_rcs:+.6f}, GS_harmed={harmed_gs:+.6f}, stable={stable_rcs:+.6f}",
            "pass": gate_d,
        },
    }
    full = len(dataset) == EXPECTED_VAL and args.bootstrap_resamples == 10000
    decision = phase2a_decision(gates, full=full)
    failed = [name for name, row in gates.items() if not row["pass"]]
    interpretations = {
        "RDDR_PHASE2A_GO": "All preregistered gates pass. Detached hierarchical disagreement is useful as receiver-side reliability for spatial context suppression without altering F.",
        "CONTEXT_REDUCTION_WORKS_SPATIAL_SPECIFICITY_FAIL": "Context reduction improves the model, but the preregistered evidence does not establish extra utility from spatial q targeting over mean-matched global scaling.",
        "LOCAL_CH_HARM_REDUCED_NO_GLOBAL_GAIN": "Receiver gating reduces the frozen Harmed-by-CH population locally, but the effect does not translate into a reliable global mIoU gain.",
        "CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE": "The semantic-safety gate fails: even receiver-only context suppression damages CAM28_1, interior, or large-region behavior under the frozen thresholds.",
        "RDDR_PHASE2A_NOGO": "The frozen receiver-side context-suppression hypothesis does not establish global or Harmed-by-CH utility.",
        "RDDR_PHASE2A_SMOKE_ONLY": "This is an engineering smoke only and cannot produce a scientific decision.",
    }
    scientific = interpretations[decision]
    if failed:
        scientific += f" Failed gates: {', '.join(failed)}. No post-hoc transformation or tuning is permitted."

    smoke = json.loads(Path(args.smoke_json).read_text())
    summary = {
        "decision": decision,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "evaluation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "a0_commit": A0_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "commands": {
            "training": " ".join(
                shlex.quote(item)
                for item in (
                    "bash", "tools/run_rddr_phase2a_server.sh",
                    str(Path(args.gs_dir).parent), args.pretrained,
                    args.train_root, args.python_executable,
                )
            ),
            "analysis": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
        },
        "images": len(dataset),
        "metrics": metrics,
        "zones": zone_results,
        "object_size": object_results,
        "context_strength": context_results,
        "fixed_strata": fixed_summary,
        "ch_transition": {
            name: accumulator.rows(name, "c0_ch_groups")
            for name, accumulator in ch.items()
        },
        "quintile_thresholds": {
            "full": quintile_result["thresholds_full"],
            "feature": quintile_result["thresholds_feature"],
        },
        "quintile_analysis": quintile_rows,
        "q_dynamics": q_rows,
        "epoch0_audit": epoch0,
        "gamma_dynamics": gamma_rows,
        "bootstrap": bootstrap,
        "gates": gates,
        "scientific_interpretation": scientific,
        "engineering": {
            "identity_max_abs_diff": max(
                smoke["initial_equivalence"]["max_abs_diff_at_zero_gamma"].values()
            ),
            "context_helper_sha256": helper_sha,
            "model_source_sha256": model_source_sha,
            "additional_trainable_parameters": 0,
            "parameters": {
                "C0": smoke["variants"]["receiver"]["parameters"]["total"],
                "GS": smoke["variants"]["global"]["parameters"]["total"],
                "RCS": smoke["variants"]["receiver"]["parameters"]["total"],
            },
            "semantic_preservation": semantic_preservation,
            "official_inference_parity": inference_parity,
            "optimizer_audits": optimizer_audits,
            "test_used": False,
            "luad_used": False,
            "checkpoint_selection": "FINAL epoch25 only",
        },
        "runtime": {
            "seconds": elapsed,
            "total_seconds": time.perf_counter() - total_started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "training_seconds": {
                "GS": training_runtime(args.gs_dir),
                "RCS": training_runtime(args.rcs_dir),
            },
        },
    }
    write_csv(output / "rddr_phase2a_training_curves.csv", training_rows)
    write_json(output / "rddr_phase2a_optimizer_audit.json", optimizer_audits)
    write_csv(output / "rddr_phase2a_q_dynamics.csv", q_rows)
    write_csv(output / "rddr_phase2a_context_strength.csv", context_rows)
    write_csv(output / "rddr_phase2a_gamma_dynamics.csv", gamma_rows)
    write_csv(output / "rddr_phase2a_fixed_strata.csv", fixed_rows)
    write_csv(output / "rddr_phase2a_ch_transition.csv", ch_rows)
    write_csv(output / "rddr_phase2a_quintile_analysis.csv", quintile_rows)
    write_csv(output / "rddr_phase2a_per_class.csv", per_class_rows)
    write_csv(output / "rddr_phase2a_bootstrap.csv", bootstrap_rows)
    write_csv(output / "rddr_phase2a_per_image.csv", per_image_rows)
    write_json(output / "rddr_phase2a_summary.json", summary)
    (output / "rddr_phase2a_dross_aware_context_suppression_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


if __name__ == "__main__":
    main()
