"""Evaluate the frozen RDDR Phase-1 C0/UC/DD utility gate on BCSS validation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.dross_disposal import compute_rddr_dross_score  # noqa: E402
from network.resnet38_cls import Net_CAM  # noqa: E402
from tools.rddr_phase1_analysis_common import (  # noqa: E402
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
IMPLEMENTATION_COMMIT = "4e08c9d228ee269f7754c5f6b78ca734cd165c61"
C0_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
MILESTONES = (1, 5, 10, 15, 20, 25)


def load_model(mode, checkpoint, device="cuda"):
    model = Net_CAM(4, rddr_phase1_mode=mode)
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
        for name, cam in zip(cam_lists, (cam56, cam28_1, cam28_2, camdeep)):
            cam = F.interpolate(
                cam, output_size, mode="bilinear", align_corners=False
            )[0]
            if cam_flip_dims:
                cam = torch.flip(cam, dims=cam_flip_dims)
            cam_lists[name].append(cam.float().cpu().numpy())
        probabilities.append(probability.float().cpu().numpy()[0])
    averaged = {
        name: minmax_normalize(np.mean(values, axis=0))
        for name, values in cam_lists.items()
    }
    presence = presence_from_probability(np.mean(probabilities, axis=0))
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


def canonical_diagnostics(model, image, output_size):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        values = model.forward_rddr_diagnostics(image)
        raw_logits = model.ic1(values["F28_raw"])
        rect_logits = model.ic1(values["F28_rect"])
        deep_logits = model.fc8(values["Ddeep"])
    raw_up = F.interpolate(raw_logits.float(), output_size, mode="bilinear", align_corners=False)
    rect_up = F.interpolate(rect_logits.float(), output_size, mode="bilinear", align_corners=False)
    deep_up = F.interpolate(deep_logits.float(), output_size, mode="bilinear", align_corners=False)
    phase0_q = compute_rddr_dross_score(raw_up, deep_up)[0, 0].cpu().numpy()
    return values, phase0_q, raw_up.argmax(1)[0].cpu().numpy(), rect_up.argmax(1)[0].cpu().numpy()


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


class DisposalAccumulator:
    def __init__(self, variant):
        self.variant = variant
        self.feature_sq = 0.0
        self.component_sq = 0.0
        self.delta_sq = 0.0
        self.feature_count = 0
        self.q_values = []
        self.delta_pixel_rms = []
        self.cosine = []
        self.norm_ratio = []

    def update(self, values):
        feature = values["F28_raw"].float()
        clean = values["F28_clean"].float()
        component = values["dross_component"].float()
        delta = values["delta_feature"].float()
        q = values["q"].float()
        self.feature_sq += float(feature.square().sum().item())
        self.component_sq += float(component.square().sum().item())
        self.delta_sq += float(delta.square().sum().item())
        self.feature_count += feature.numel()
        self.q_values.append(q.flatten().cpu().numpy())
        self.delta_pixel_rms.append(delta.square().mean(1).sqrt().flatten().cpu().numpy())
        self.cosine.append(
            F.cosine_similarity(clean, feature, dim=1, eps=1.0e-8).flatten().cpu().numpy()
        )
        self.norm_ratio.append(
            (clean.norm(dim=1) / feature.norm(dim=1).clamp_min(1.0e-8))
            .flatten()
            .cpu()
            .numpy()
        )

    def result(self):
        q = np.concatenate(self.q_values)
        delta = np.concatenate(self.delta_pixel_rms)
        cosine = np.concatenate(self.cosine)
        norm_ratio = np.concatenate(self.norm_ratio)
        rows = []
        if np.all(q == q[0]):
            named_masks = (("AllPixels", np.ones(q.shape, dtype=bool)),)
        else:
            quantiles = np.quantile(q, (0.2, 0.4, 0.6, 0.8))
            bins = np.digitize(q, quantiles, right=True)
            named_masks = tuple(
                (name, bins == index)
                for index, name in enumerate(
                    ("Bottom20", "20-40", "40-60", "60-80", "Top20")
                )
            )
        for name, mask in named_masks:
            rows.append(
                {
                    "variant": self.variant,
                    "bin": name,
                    "pixels": int(mask.sum()),
                    "q_mean": float(q[mask].mean()),
                    "delta_pixel_rms": float(np.sqrt(np.mean(delta[mask] ** 2))),
                    "feature_cosine": float(cosine[mask].mean()),
                    "feature_norm_ratio": float(norm_ratio[mask].mean()),
                }
            )
        overall = {
            "variant": self.variant,
            "RMS_F": math.sqrt(self.feature_sq / self.feature_count),
            "RMS_D": math.sqrt(self.component_sq / self.feature_count),
            "RMS_DeltaF": math.sqrt(self.delta_sq / self.feature_count),
            "RMS_DeltaF_over_RMS_F": math.sqrt(self.delta_sq / self.feature_sq),
            "q_mean": float(q.mean()),
            "q_std": float(q.std()),
            "q_min": float(q.min()),
            "q_max": float(q.max()),
        }
        return overall, rows


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


def compute_q_dynamics(dd_dir, c0_checkpoint, loader):
    rows = []
    checkpoints = [("Phase0-C0", 0, "none", Path(c0_checkpoint))]
    checkpoints += [
        ("DD", epoch, "dd", checkpoint_for_epoch(dd_dir, epoch))
        for epoch in MILESTONES
    ]
    for source, epoch, mode, checkpoint in checkpoints:
        model = load_model(mode, checkpoint)
        q_values = []
        with torch.inference_mode():
            for _, image, _, _ in loader:
                image = image.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values = model.forward_rddr_diagnostics(image)
                    if mode == "none":
                        q = compute_rddr_dross_score(
                            model.ic1(values["F28_raw"]),
                            model.fc8(values["Ddeep"]),
                        )
                    else:
                        q = values["q"]
                q_values.append(q.float().flatten().cpu().numpy())
        rows.append(distribution(q_values, epoch, source))
        del model
        torch.cuda.empty_cache()
        print(f"RDDR_Q_DYNAMICS source={source} epoch={epoch}", flush=True)
    return rows


def load_training_curves(uc_dir, dd_dir):
    rows = []
    for variant, directory in (("UC", uc_dir), ("DD", dd_dir)):
        with (Path(directory) / "training_curve.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append({"variant": variant, **row})
    return rows


def load_optimizer_audits(uc_dir, dd_dir):
    return {
        variant: json.loads((Path(directory) / "optimizer_audit.json").read_text())
        for variant, directory in (("UC", uc_dir), ("DD", dd_dir))
    }


def training_runtime(directory):
    text = (Path(directory) / "train.log").read_text(errors="replace")
    matches = re.findall(r"Total Training Time:\s*([0-9.]+)s", text)
    return float(matches[-1]) if matches else float("nan")


def render_report(summary):
    metric = summary["metrics"]
    zones = summary["zones"]
    objects = summary["object_size"]
    fixed = summary["fixed_strata"]
    gates = summary["gates"]
    bootstrap = summary["bootstrap"]
    lines = [
        "# RDDR Phase-1 Spatial-Semantic Dross Disposal Report",
        "",
        "## 1. Frozen provenance and commands",
        "",
        f"- Implementation commit: `{summary['implementation_commit']}`",
        f"- Pure A0 commit: `{summary['a0_commit']}`",
        f"- C0 checkpoint SHA256: `{summary['checkpoint_sha256']['C0']}`",
        f"- UC checkpoint SHA256: `{summary['checkpoint_sha256']['UC']}`",
        f"- DD checkpoint SHA256: `{summary['checkpoint_sha256']['DD']}`",
        "- Dataset/split: BCSS validation only; no test or LUAD access.",
        "",
        "```bash",
        summary["commands"]["training"],
        summary["commands"]["analysis"],
        "```",
        "",
        "## 2. Architecture and identity contract",
        "",
        "Only the HFRM28_1 input changes. UC uses `F_clean=F-DDA(F)`; DD uses "
        "`F_clean=F-q*DDA(F)` with detached normalized JSD q. HFRM56, "
        "HFRM28_2, heads, loss, optimizer, inference, and metric are unchanged.",
        f"Initial C0/DD FP32 maximum absolute difference: `{summary['engineering']['identity_max_abs_diff']:.8g}`.",
        "",
        "| Variant | Total parameters | Added parameters | Added MACs@28×28 | Added conv FLOPs@28×28 |",
        "|---|---:|---:|---:|---:|",
        f"| C0 | {summary['engineering']['parameters']['C0']} | 0 | 0 | 0 |",
        f"| UC | {summary['engineering']['parameters']['UC']} | {summary['engineering']['dda_parameters']} | {summary['engineering']['dda_macs_28x28']} | {2*summary['engineering']['dda_macs_28x28']} |",
        f"| DD | {summary['engineering']['parameters']['DD']} | {summary['engineering']['dda_parameters']} | {summary['engineering']['dda_macs_28x28']} + analytical JSD | {2*summary['engineering']['dda_macs_28x28']} + analytical JSD |",
        "",
        "## 3. Training protocol equivalence",
        "",
        "UC and DD use seed42, batch20, BF16, epoch0→25, released augmentation, "
        "loss 0.10/0.15/0.25/0.50, released PolyOptimizer and FINAL checkpoint selection. "
        "Both use the same DDA initialization and scratch-LR groups. No validation or test "
        "metric influenced training or checkpoint selection.",
        "",
        "## 4. Overall metrics and CAM hierarchy",
        "",
        "| Variant | CAM56 mIoU | CAM28_1 mIoU | CAM28_2 mIoU | CAMdeep mIoU | Final mIoU | Final mDice |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in ("C0", "UC", "DD"):
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
    for variant in ("C0", "UC", "DD"):
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
    for variant in ("C0", "UC", "DD"):
        values = metric[variant]["Final"]["class_iou"]
        lines.append(
            f"| {variant} | {100*values['0']:.4f} | {100*values['1']:.4f} | "
            f"{100*values['2']:.4f} | {100*values['3']:.4f} |"
        )
    lines += [
        "",
        "## 7. q dynamics, disposal, and feature preservation",
        "",
        "| Source | Epoch | Mean | Std | Min | p05 | p25 | p50 | p75 | p95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["q_dynamics"]:
        lines.append(
            f"| {row['source']} | {row['epoch']} | {row['mean']:.6f} | {row['std']:.6f} | {row['min']:.6f} | "
            f"{row['p05']:.6f} | {row['p25']:.6f} | {row['p50']:.6f} | "
            f"{row['p75']:.6f} | {row['p95']:.6f} | {row['max']:.6f} |"
        )
    lines += [
        "",
        f"At Epoch25 DD: q={summary['disposal']['DD']['q_mean']:.6f}±{summary['disposal']['DD']['q_std']:.6f}; "
        f"RMS(ΔF)/RMS(F)={summary['disposal']['DD']['RMS_DeltaF_over_RMS_F']:.6f}.",
        "",
        "| Variant | RMS(F) | RMS(D(F)) | RMS(ΔF) | RMS(ΔF)/RMS(F) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("UC", "DD"):
        row = summary["disposal"][variant]
        lines.append(
            f"| {variant} | {row['RMS_F']:.6f} | {row['RMS_D']:.6f} | "
            f"{row['RMS_DeltaF']:.6f} | {row['RMS_DeltaF_over_RMS_F']:.6f} |"
        )
    lines += [
        "",
        "| Variant/bin | q mean | ΔF pixel RMS | cos(Fclean,F) | norm ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in ("UC", "DD"):
        for row in summary["disposal_bins"][variant]:
            lines.append(
                f"| {variant}/{row['bin']} | {row['q_mean']:.6f} | "
                f"{row['delta_pixel_rms']:.6f} | {row['feature_cosine']:.6f} | "
                f"{row['feature_norm_ratio']:.6f} |"
            )
    lines += [
        "",
        "## 8. Frozen Phase-0 strata and CH transition re-audit",
        "",
        "| Variant | Top20 net repair | Bottom80 net repair |",
        "|---|---:|---:|",
    ]
    for variant in ("UC", "DD"):
        lines.append(
            f"| {variant} | {100*fixed[variant]['Top20']['net_repair']:+.4f} pp | "
            f"{100*fixed[variant]['Bottom80']['net_repair']:+.4f} pp |"
        )
    lines += [
        "",
        "C0-defined Corrected-by-CH / Still-Wrong / Harmed-by-CH / Stable-Correct "
        "groups are never redefined using UC or DD.",
        "",
        "| Variant/group | Repair | Harm | Net accuracy change |",
        "|---|---:|---:|---:|",
    ]
    for variant in ("UC", "DD"):
        for row in summary["ch_transition"][variant]:
            lines.append(
                f"| {variant}/{row['group']} | {100*row['repair_rate']:.4f} pp | "
                f"{100*row['harm_rate']:.4f} pp | {100*row['net_repair']:+.4f} pp |"
            )
    lines += [
        "",
        "## 9. UC versus DD and paired bootstrap",
        "",
        "| Variant | Correction RMS ratio | Top20 repair | Bottom80 harm | CAM28_1 mIoU | Final mIoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in ("UC", "DD"):
        lines.append(
            f"| {variant} | {summary['disposal'][variant]['RMS_DeltaF_over_RMS_F']:.6f} | "
            f"{100*fixed[variant]['Top20']['repair_rate']:.4f} pp | "
            f"{100*fixed[variant]['Bottom80']['harm_rate']:.4f} pp | "
            f"{100*metric[variant]['CAM28_1']['mIoU']:.4f} | "
            f"{100*metric[variant]['Final']['mIoU']:.4f} |"
        )
    lines += [
        "",
        "| Comparison | Observed ΔmIoU | Bootstrap mean | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for name in ("DD-C0", "DD-UC", "UC-C0"):
        row = bootstrap[name]
        lines.append(
            f"| {name} | {100*row['observed_delta_mIoU']:+.4f} pp | "
            f"{100*row['bootstrap_mean']:+.4f} pp | "
            f"[{100*row['ci95_low']:+.4f}, {100*row['ci95_high']:+.4f}] pp |"
        )
    lines += [
        "",
        "## 10. Preregistered gates",
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
        "## 11. Scientific interpretation",
        "",
        summary["scientific_interpretation"],
        "",
        "## 12. Engineering and artifact record",
        "",
        f"- Analysis runtime: {summary['runtime']['seconds']/60:.2f} min; peak CUDA memory "
        f"{summary['runtime']['peak_cuda_memory_bytes']/2**30:.3f} GiB.",
        f"- UC/DD training runtime: {summary['runtime']['training_seconds']['UC']/60:.2f} / "
        f"{summary['runtime']['training_seconds']['DD']/60:.2f} min.",
        "- Required optimizer, training-curve, q, disposal, fixed-strata, CH, per-class, "
        "bootstrap, and summary artifacts were generated.",
        "- No BCSS test, LUAD, best-epoch selection, or post-hoc parameter tuning was used.",
        "",
        f"DECISION = {summary['decision']}",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-checkpoint", required=True)
    parser.add_argument("--uc-dir", required=True)
    parser.add_argument("--dd-dir", required=True)
    parser.add_argument("--phase0-dir", required=True)
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
    output.mkdir(parents=True, exist_ok=True)
    phase0 = json.loads((Path(args.phase0_dir) / "rddr_phase0_summary.json").read_text())
    if phase0["decision"] != "RDDR_PHASE0_GO":
        raise AssertionError("Phase-0 GO artifact is required")
    if sha256_file(args.c0_checkpoint) != C0_SHA256:
        raise AssertionError("C0 checkpoint SHA256 mismatch")
    set_seed(42)
    phase0_threshold = float(phase0["thresholds"]["S_JS"]["0.2"]) / math.log(2.0)
    checkpoints = {
        "C0": Path(args.c0_checkpoint),
        "UC": checkpoint_for_epoch(args.uc_dir, 25),
        "DD": checkpoint_for_epoch(args.dd_dir, 25),
    }
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha = {name: sha256_file(path) for name, path in checkpoints.items()}
    dataset = SortedValidationDataset(args.val_root, max_images=args.max_images)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    q_loader = DataLoader(
        dataset,
        batch_size=20,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    thresholds = component_thresholds(args.val_root)
    models = {
        "C0": load_model("none", checkpoints["C0"]),
        "UC": load_model("uc", checkpoints["UC"]),
        "DD": load_model("dd", checkpoints["DD"]),
    }
    stages = {name: StageAccumulator() for name in models}
    zones = {name: ZoneMetricAccumulator() for name in models}
    components = {
        name: ComponentMetricAccumulator(thresholds) for name in models
    }
    disposal = {
        name: DisposalAccumulator(name) for name in ("UC", "DD")
    }
    fixed = {
        name: TransitionAccumulator(("Top20", "Bottom80"))
        for name in ("UC", "DD")
    }
    ch_groups = ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct")
    ch = {
        name: TransitionAccumulator(ch_groups) for name in ("UC", "DD")
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
            for name in ("UC", "DD"):
                disposal[name].update(canonical[name][0])

            foreground = truth < BACKGROUND
            c0_q = canonical["C0"][1]
            strata = {
                "Top20": foreground & (c0_q >= phase0_threshold),
                "Bottom80": foreground & (c0_q < phase0_threshold),
            }
            c0_final = predictions["C0"]["Final"]
            for name in ("UC", "DD"):
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
            for name in ("UC", "DD"):
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
                print(f"RDDR_PHASE1_EVAL {index}/{len(dataset)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started

    metrics = {name: accumulator.result() for name, accumulator in stages.items()}
    zone_results = {name: accumulator.result() for name, accumulator in zones.items()}
    object_results = {
        name: accumulator.result() for name, accumulator in components.items()
    }
    disposal_results = {}
    disposal_bin_results = {}
    disposal_rows = []
    for name, accumulator in disposal.items():
        overall, rows = accumulator.result()
        disposal_results[name] = overall
        disposal_bin_results[name] = rows
        disposal_rows.append({"variant": name, "bin": "Overall", **overall})
        disposal_rows.extend(rows)
    fixed_rows = [
        row
        for name, accumulator in fixed.items()
        for row in accumulator.rows(name, "fixed_phase0_strata")
    ]
    ch_rows = [
        row
        for name, accumulator in ch.items()
        for row in accumulator.rows(name, "c0_ch_groups")
    ]
    fixed_summary = {
        name: {row["group"]: row for row in accumulator.rows(name, "fixed_phase0_strata")}
        for name, accumulator in fixed.items()
    }
    histograms = {
        name: np.asarray(accumulator.per_image_final)
        for name, accumulator in stages.items()
    }
    bootstrap = {}
    bootstrap_values = {}
    for label, base, candidate in (
        ("DD-C0", "C0", "DD"),
        ("DD-UC", "UC", "DD"),
        ("UC-C0", "C0", "UC"),
    ):
        bootstrap[label], bootstrap_values[label] = paired_bootstrap_miou(
            histograms[base], histograms[candidate], args.bootstrap_resamples, seed=42
        )
    bootstrap_rows = [
        {
            "replicate": index,
            "DD_minus_C0": bootstrap_values["DD-C0"][index],
            "DD_minus_UC": bootstrap_values["DD-UC"][index],
            "UC_minus_C0": bootstrap_values["UC-C0"][index],
        }
        for index in range(args.bootstrap_resamples)
    ]
    training_rows = load_training_curves(args.uc_dir, args.dd_dir)
    optimizer_audits = load_optimizer_audits(args.uc_dir, args.dd_dir)
    q_rows = compute_q_dynamics(args.dd_dir, args.c0_checkpoint, q_loader)

    per_class_rows = []
    for variant in ("C0", "UC", "DD"):
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

    dd_c0 = bootstrap["DD-C0"]
    dd_uc = bootstrap["DD-UC"]
    gate_a = metrics["DD"]["Final"]["mIoU"] > metrics["C0"]["Final"]["mIoU"] and dd_c0["ci95_low"] >= 0.0
    mechanism_fallback = (
        metrics["DD"]["CAM28_1"]["mIoU"] > metrics["UC"]["CAM28_1"]["mIoU"]
        and fixed_summary["DD"]["Top20"]["net_repair"]
        > fixed_summary["UC"]["Top20"]["net_repair"]
    )
    gate_b = metrics["DD"]["Final"]["mIoU"] > metrics["UC"]["Final"]["mIoU"] and (
        dd_uc["ci95_low"] >= 0.0 or mechanism_fallback
    )
    interior_delta = (
        zone_results["DD"]["interior_gt_7"]["accuracy"]
        - zone_results["C0"]["interior_gt_7"]["accuracy"]
    )
    gate_c = (
        metrics["DD"]["CAM28_1"]["mIoU"] >= metrics["C0"]["CAM28_1"]["mIoU"]
        and interior_delta >= -0.001
    )
    gate_d = (
        fixed_summary["DD"]["Top20"]["net_repair"] > 0.0
        and fixed_summary["DD"]["Top20"]["net_repair"]
        > fixed_summary["DD"]["Bottom80"]["net_repair"]
    )
    gates = {
        "A": {
            "requirement": "DD mIoU > C0 and DD-C0 CI low >= 0",
            "result": f"delta={dd_c0['observed_delta_mIoU']:+.6f}, low={dd_c0['ci95_low']:+.6f}",
            "pass": gate_a,
        },
        "B": {
            "requirement": "DD > UC with nonnegative CI low or CAM28_1+Top20 fallback",
            "result": f"delta={dd_uc['observed_delta_mIoU']:+.6f}, low={dd_uc['ci95_low']:+.6f}, fallback={mechanism_fallback}",
            "pass": gate_b,
        },
        "C": {
            "requirement": "DD CAM28_1 >= C0 and interior accuracy delta >= -0.10 pp",
            "result": f"CAMdelta={metrics['DD']['CAM28_1']['mIoU']-metrics['C0']['CAM28_1']['mIoU']:+.6f}, interior={interior_delta:+.6f}",
            "pass": gate_c,
        },
        "D": {
            "requirement": "DD Top20 net > 0 and > DD Bottom80 net",
            "result": f"top={fixed_summary['DD']['Top20']['net_repair']:+.6f}, bottom={fixed_summary['DD']['Bottom80']['net_repair']:+.6f}",
            "pass": gate_d,
        },
    }
    full = len(dataset) == EXPECTED_VAL and args.bootstrap_resamples == 10000
    if not full:
        decision = "RDDR_PHASE1_SMOKE_ONLY"
    elif all(row["pass"] for row in gates.values()):
        decision = "RDDR_PHASE1_GO"
    elif gate_a and gate_c and gate_d and not gate_b:
        decision = "ADAPTER_GAIN_DROSS_SPECIFICITY_FAIL"
    elif gate_d and not gate_a:
        decision = "LOCAL_DROSS_REPAIR_NO_GLOBAL_GAIN"
    elif not gate_c:
        decision = "DROSS_DISPOSAL_SEMANTIC_DAMAGE"
    else:
        decision = "RDDR_PHASE1_NOGO"
    failed = [name for name, row in gates.items() if not row["pass"]]
    scientific = (
        "All preregistered gates pass; conditioned subtractive disposal improves the "
        "official validation result over both C0 and its parameter-matched UC control, "
        "while concentrating benefit in the frozen Phase-0 high-dross population."
        if decision == "RDDR_PHASE1_GO"
        else "The frozen subtractive-disposal hypothesis does not establish the full "
        f"causal utility chain. Failed gates: {', '.join(failed)}. No post-hoc change "
        "to q, temperature, adapter depth, kernel, reduction, or checkpoint selection is permitted."
    )
    smoke = json.loads(Path(args.smoke_json).read_text())
    summary = {
        "decision": decision,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "a0_commit": A0_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "commands": {
            "training": " ".join(
                shlex.quote(item)
                for item in (
                    "bash",
                    "tools/run_rddr_phase1_server.sh",
                    str(Path(args.uc_dir).parent),
                    args.pretrained,
                    args.train_root,
                    args.python_executable,
                )
            ),
            "analysis": " ".join(shlex.quote(item) for item in sys.argv),
        },
        "images": len(dataset),
        "metrics": metrics,
        "zones": zone_results,
        "object_size": object_results,
        "disposal": disposal_results,
        "disposal_bins": disposal_bin_results,
        "fixed_strata": fixed_summary,
        "ch_transition": {name: accumulator.rows(name, "c0_ch_groups") for name, accumulator in ch.items()},
        "q_dynamics": q_rows,
        "bootstrap": bootstrap,
        "gates": gates,
        "scientific_interpretation": scientific,
        "engineering": {
            "identity_max_abs_diff": smoke["identity"]["max_abs_diff"],
            "dda_parameters": smoke["variants"]["dd"]["parameters"]["dda"],
            "dda_macs_28x28": smoke["variants"]["dd"]["dda_macs_28x28"],
            "parameters": {
                "C0": smoke["variants"]["dd"]["parameters"]["total"]
                - smoke["variants"]["dd"]["parameters"]["dda"],
                "UC": smoke["variants"]["uc"]["parameters"]["total"],
                "DD": smoke["variants"]["dd"]["parameters"]["total"],
            },
            "optimizer_audits": optimizer_audits,
            "test_used": False,
            "luad_used": False,
            "checkpoint_selection": "FINAL epoch25 only",
        },
        "runtime": {
            "seconds": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "training_seconds": {
                "UC": training_runtime(args.uc_dir),
                "DD": training_runtime(args.dd_dir),
            },
        },
    }
    write_csv(output / "rddr_phase1_training_curves.csv", training_rows)
    write_json(output / "rddr_phase1_optimizer_audit.json", optimizer_audits)
    write_csv(output / "rddr_phase1_q_dynamics.csv", q_rows)
    write_csv(output / "rddr_phase1_disposal_diagnostics.csv", disposal_rows)
    write_csv(output / "rddr_phase1_fixed_strata_transition.csv", fixed_rows)
    write_csv(output / "rddr_phase1_ch_transition.csv", ch_rows)
    write_csv(output / "rddr_phase1_per_class.csv", per_class_rows)
    write_csv(output / "rddr_phase1_bootstrap.csv", bootstrap_rows)
    write_csv(output / "rddr_phase1_per_image.csv", per_image_rows)
    write_json(output / "rddr_phase1_summary.json", summary)
    (output / "rddr_phase1_spatial_semantic_dross_disposal_report.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"DECISION = {decision}", flush=True)


if __name__ == "__main__":
    main()
