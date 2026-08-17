"""Frozen Phase-0 feasibility audit for the analytical CDSR need signal.

This tool is intentionally model-read-only. It reuses the trained A0 CAM
heads before and after HFRM, computes the preregistered analytical need map,
and uses BCSS validation masks only for offline signal-ranking analysis.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net_CAM


STAGES = ("stage1", "stage2", "stage3")
STAGE_LABELS = {
    "stage1": "F56",
    "stage2": "F28_1",
    "stage3": "F28_2",
}
CLASS_NAMES = {
    0: "Tumor",
    1: "Stroma",
    2: "Inflammatory",
    3: "Necrosis",
}
EPSILON = 1e-8
BOOTSTRAP_REPEATS = 2000
SEED = 42


class BCSSValidationDataset(Dataset):
    def __init__(self, root, image_size=224):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.image_dir = self.root / "img"
        self.mask_dir = self.root / "mask"
        images = sorted(
            path
            for path in self.image_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        mask_paths = sorted(
            path
            for path in self.mask_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        masks = {path.stem: path for path in mask_paths}
        if len(masks) != len(mask_paths):
            raise RuntimeError("Duplicate validation mask stems detected")
        missing = [path.name for path in images if path.stem not in masks]
        if missing:
            raise RuntimeError(f"Missing validation masks: {missing[:5]}")
        image_stems = {path.stem for path in images}
        extra = [path.name for path in mask_paths if path.stem not in image_stems]
        if extra:
            raise RuntimeError(f"Masks without validation images: {extra[:5]}")
        self.samples = [(path, masks[path.stem]) for path in images]
        if len(self.samples) != 3418 or len(mask_paths) != 3418:
            raise RuntimeError(
                "Expected 3418 matched BCSS validation samples, got "
                f"{len(self.samples)} images and {len(mask_paths)} masks"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.copy()
        if image.size != (self.image_size, self.image_size):
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
            )
        if mask.size != (self.image_size, self.image_size):
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
        image = TF.normalize(
            TF.to_tensor(image),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        mask = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image, mask, index, image_path.name


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load failed: {incompatible}")


def normalized_entropy(probabilities):
    classes = probabilities.shape[1]
    entropy = -(
        probabilities
        * probabilities.clamp_min(EPSILON).log()
    ).sum(dim=1)
    return (entropy / math.log(classes)).clamp(0.0, 1.0)


def normalized_jsd(first_probabilities, second_probabilities):
    """Return pixel-wise Jensen-Shannon divergence normalized to [0, 1]."""
    mixture = 0.5 * (first_probabilities + second_probabilities)
    first_kl = (
        first_probabilities
        * (
            first_probabilities.clamp_min(EPSILON).log()
            - mixture.clamp_min(EPSILON).log()
        )
    ).sum(dim=1)
    second_kl = (
        second_probabilities
        * (
            second_probabilities.clamp_min(EPSILON).log()
            - mixture.clamp_min(EPSILON).log()
        )
    ).sum(dim=1)
    return (0.5 * (first_kl + second_kl) / math.log(2.0)).clamp(
        0.0, 1.0
    )


def analytical_need(stage_logits, deep_logits):
    stage_probabilities = torch.softmax(
        stage_logits.detach().float(), dim=1
    )
    deep_probabilities = torch.softmax(
        deep_logits.detach().float(), dim=1
    )
    deep_probabilities = F.interpolate(
        deep_probabilities,
        size=stage_probabilities.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    disagreement = normalized_jsd(
        stage_probabilities, deep_probabilities
    )
    uncertainty = normalized_entropy(stage_probabilities)
    deep_reliability = (1.0 - normalized_entropy(deep_probabilities)).clamp(
        0.0, 1.0
    )
    ambiguity = 1.0 - (1.0 - disagreement) * (1.0 - uncertainty)
    need = (deep_reliability * ambiguity).clamp(0.0, 1.0)
    return {
        "probabilities": stage_probabilities,
        "disagreement": disagreement,
        "uncertainty": uncertainty,
        "deep_reliability": deep_reliability,
        "need": need,
    }


def safe_binary_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.uint8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or np.unique(labels).size < 2:
        return {"auroc": None, "aupr": None}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
    }


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def cohens_d(positive, negative):
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size < 2 or negative.size < 2:
        return None
    denominator = positive.size + negative.size - 2
    pooled_variance = (
        (positive.size - 1) * positive.var(ddof=1)
        + (negative.size - 1) * negative.var(ddof=1)
    ) / denominator
    if pooled_variance <= 0.0:
        return 0.0
    return float((positive.mean() - negative.mean()) / math.sqrt(pooled_variance))


def bootstrap_image_net_gap(
    image_ids,
    corrected,
    harmed,
    bottom_mask,
    top_mask,
    image_count,
):
    def per_image_net(mask):
        counts = np.bincount(image_ids[mask], minlength=image_count)
        corrected_counts = np.bincount(
            image_ids[mask & corrected], minlength=image_count
        )
        harmed_counts = np.bincount(
            image_ids[mask & harmed], minlength=image_count
        )
        rates = np.full(image_count, np.nan, dtype=np.float64)
        valid = counts > 0
        rates[valid] = (
            corrected_counts[valid] - harmed_counts[valid]
        ) / counts[valid]
        return rates

    bottom_rates = per_image_net(bottom_mask)
    top_rates = per_image_net(top_mask)
    valid = np.isfinite(bottom_rates) & np.isfinite(top_rates)
    differences = top_rates[valid] - bottom_rates[valid]
    if differences.size == 0:
        return {
            "paired_images": 0,
            "mean_difference": None,
            "bootstrap_ci95": [None, None],
        }
    rng = np.random.default_rng(SEED)
    bootstrapped = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        indices = rng.integers(0, differences.size, size=differences.size)
        bootstrapped[repeat] = differences[indices].mean()
    return {
        "paired_images": int(differences.size),
        "mean_difference": float(differences.mean()),
        "bootstrap_ci95": [
            float(np.quantile(bootstrapped, 0.025)),
            float(np.quantile(bootstrapped, 0.975)),
        ],
    }


def quartile_analysis(
    scores, raw_wrong, corrected, harmed, image_ids, image_count
):
    q25, q75 = np.quantile(scores, [0.25, 0.75])
    bottom = scores <= q25
    top = scores >= q75

    def summarize(mask):
        count = int(mask.sum())
        corrected_rate = float(corrected[mask].mean())
        harmed_rate = float(harmed[mask].mean())
        return {
            "count": count,
            "raw_error_rate": float(raw_wrong[mask].mean()),
            "corrected_rate": corrected_rate,
            "harmed_rate": harmed_rate,
            "net_correction_rate": corrected_rate - harmed_rate,
        }

    bottom_summary = summarize(bottom)
    top_summary = summarize(top)
    paired = bootstrap_image_net_gap(
        image_ids,
        corrected,
        harmed,
        bottom,
        top,
        image_count,
    )
    return {
        "q25": float(q25),
        "q75": float(q75),
        "bottom": bottom_summary,
        "top": top_summary,
        "raw_error_rate_gap": (
            top_summary["raw_error_rate"]
            - bottom_summary["raw_error_rate"]
        ),
        "net_correction_rate_gap": (
            top_summary["net_correction_rate"]
            - bottom_summary["net_correction_rate"]
        ),
        "paired_image_net_gap": paired,
    }


def analyze_subset(scores, gt, raw_wrong, corrected, harmed, selection):
    subset_scores = scores[selection]
    subset_wrong = raw_wrong[selection]
    raw_metrics = safe_binary_metrics(subset_wrong, subset_scores)
    transition = selection & (corrected | harmed)
    transition_labels = corrected[transition].astype(np.uint8)
    transition_scores = scores[transition]
    transition_metrics = safe_binary_metrics(
        transition_labels, transition_scores
    )
    return {
        "pixels": int(selection.sum()),
        "raw_error": {
            **raw_metrics,
            "wrong": distribution_summary(subset_scores[subset_wrong]),
            "correct": distribution_summary(subset_scores[~subset_wrong]),
            "cohens_d_wrong_minus_correct": cohens_d(
                subset_scores[subset_wrong], subset_scores[~subset_wrong]
            ),
        },
        "corrected_vs_harmed": {
            **transition_metrics,
            "corrected": distribution_summary(scores[selection & corrected]),
            "harmed": distribution_summary(scores[selection & harmed]),
            "cohens_d_corrected_minus_harmed": cohens_d(
                scores[selection & corrected], scores[selection & harmed]
            ),
        },
    }


def analyze_stage(chunks, image_count):
    arrays = {
        name: np.concatenate([chunk[name] for chunk in chunks])
        for name in chunks[0]
    }
    scores = arrays["need"].astype(np.float64)
    gt = arrays["gt"]
    raw_wrong = arrays["raw_wrong"].astype(bool)
    corrected = arrays["corrected"].astype(bool)
    harmed = arrays["harmed"].astype(bool)
    image_ids = arrays["image_id"].astype(np.int64)
    overall = analyze_subset(
        scores,
        gt,
        raw_wrong,
        corrected,
        harmed,
        np.ones(scores.shape, dtype=bool),
    )
    overall["need"] = distribution_summary(scores)
    overall["quartiles"] = quartile_analysis(
        scores,
        raw_wrong,
        corrected,
        harmed,
        image_ids,
        image_count,
    )
    overall["per_class"] = {
        str(class_id): {
            "name": CLASS_NAMES[class_id],
            **analyze_subset(
                scores,
                gt,
                raw_wrong,
                corrected,
                harmed,
                gt == class_id,
            ),
        }
        for class_id in range(4)
    }
    return overall


def evaluate_decision(stage_results):
    stage_decisions = {}
    for stage, result in stage_results.items():
        raw_auroc = result["raw_error"]["auroc"]
        transition_auroc = result["corrected_vs_harmed"]["auroc"]
        transition_d = result["corrected_vs_harmed"][
            "cohens_d_corrected_minus_harmed"
        ]
        paired_ci = result["quartiles"]["paired_image_net_gap"][
            "bootstrap_ci95"
        ]
        conditions = {
            "raw_error_auroc_ge_0_60": raw_auroc is not None and raw_auroc >= 0.60,
            "corrected_harmed_auroc_ge_0_55": (
                transition_auroc is not None and transition_auroc >= 0.55
            ),
            "corrected_harmed_d_ge_0_20": (
                transition_d is not None and transition_d >= 0.20
            ),
            "top_bottom_net_gap_ci_above_zero": (
                paired_ci[0] is not None and paired_ci[0] > 0.0
            ),
        }
        major_reverse = (
            (transition_auroc is not None and transition_auroc < 0.45)
            or (transition_d is not None and transition_d < -0.20)
            or (paired_ci[1] is not None and paired_ci[1] < 0.0)
        )
        stage_decisions[stage] = {
            "conditions": conditions,
            "condition_count": int(sum(conditions.values())),
            "stage_go": sum(conditions.values()) >= 2,
            "strong_go": (
                raw_auroc is not None
                and transition_auroc is not None
                and raw_auroc >= 0.65
                and transition_auroc >= 0.60
            ),
            "major_reverse": bool(major_reverse),
        }
    go_stages = sum(item["stage_go"] for item in stage_decisions.values())
    strong_stages = sum(
        item["strong_go"] for item in stage_decisions.values()
    )
    major_reverses = sum(
        item["major_reverse"] for item in stage_decisions.values()
    )
    signal_go = go_stages >= 2 and major_reverses == 0
    return {
        "stage_decisions": stage_decisions,
        "go_stage_count": go_stages,
        "strong_go_stage_count": strong_stages,
        "major_reverse_stage_count": major_reverses,
        "strong_go": strong_stages >= 2 and major_reverses == 0,
        "signal_go": signal_go,
        "token": "CDSR_SIGNAL_GO" if signal_go else "CDSR_SIGNAL_NOGO",
    }


def format_float(value, digits=4):
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def render_report(result):
    protocol = result["protocol"]
    decision = result["decision"]
    lines = [
        "# CDSR Need-Signal Feasibility Audit",
        "",
        "## 1. Frozen protocol",
        "",
        "This is a zero-training Phase-0 audit. It uses BCSS validation only,",
        "the A0 seed-42 final checkpoint, no test data, no retraining, and no",
        "formula or weight search. Raw and post-HFRM logits reuse the same",
        "trained stage CAM heads. Risk math is detached FP32.",
        "",
        f"- validation samples: {protocol['validation_samples']}",
        f"- checkpoint: `{protocol['checkpoint']}`",
        f"- checkpoint SHA256: `{protocol['checkpoint_sha256']}`",
        "- checkpoint loading: strict state-dict match (no missing or unexpected keys)",
        "- model path: official A0 HFRM with CH context (`context_mode=ch`)",
        f"- PyTorch: `{protocol['pytorch']}`",
        f"- device: `{protocol['device']}`",
        f"- network forward precision: `{protocol['amp_dtype']}`",
        "- analysis pixels: native-stage pixels whose GT is foreground 0-3;",
        "  background 4 is excluded because the CAMs have four foreground",
        "  channels and the official SSHR metric excludes/overwrites background",
        "- GT is resized to each native stage by nearest-neighbor interpolation",
        "",
        "## 2. Preregistered need signal",
        "",
        "```text",
        "D = JSD(P_stage, P_deep) / ln(2)",
        "U = entropy(P_stage) / ln(C)",
        "R = 1 - entropy(P_deep) / ln(C)",
        "N = R * (1 - (1-D) * (1-U))",
        "```",
        "",
        "No class-presence labels, GT class IDs, thresholds, temperatures, or",
        "manual D/U weights are used in the signal.",
        "",
        "## 3. Overall signal ranking",
        "",
        "| Stage | pixels | raw-error AUROC | raw-error AUPR | corrected/harmed AUROC | corrected/harmed AUPR | corrected-harmed Cohen d |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        metrics = result["stages"][stage]
        lines.append(
            "| {label} | {pixels:,} | {raw_auc} | {raw_ap} | {tr_auc} | {tr_ap} | {effect} |".format(
                label=STAGE_LABELS[stage],
                pixels=metrics["pixels"],
                raw_auc=format_float(metrics["raw_error"]["auroc"]),
                raw_ap=format_float(metrics["raw_error"]["aupr"]),
                tr_auc=format_float(metrics["corrected_vs_harmed"]["auroc"]),
                tr_ap=format_float(metrics["corrected_vs_harmed"]["aupr"]),
                effect=format_float(
                    metrics["corrected_vs_harmed"][
                        "cohens_d_corrected_minus_harmed"
                    ]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 4. Need distributions",
            "",
            "| Stage | group | count | mean | median | std |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        metrics = result["stages"][stage]
        groups = {
            "raw wrong": metrics["raw_error"]["wrong"],
            "raw correct": metrics["raw_error"]["correct"],
            "corrected by HFRM": metrics["corrected_vs_harmed"]["corrected"],
            "harmed by HFRM": metrics["corrected_vs_harmed"]["harmed"],
        }
        for name, summary in groups.items():
            lines.append(
                f"| {STAGE_LABELS[stage]} | {name} | {summary['count']:,} | "
                f"{format_float(summary['mean'])} | "
                f"{format_float(summary['median'])} | "
                f"{format_float(summary['std'])} |"
            )
    lines.extend(
        [
            "",
            "## 5. Quartile analysis",
            "",
            "The preregistered interpretation of 'clearly higher' is an",
            "image-paired bootstrap 95% CI whose lower bound is above zero.",
            "",
            "| Stage | bottom raw error | top raw error | bottom net correction | top net correction | top-bottom net gap | paired-image 95% CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        quartiles = result["stages"][stage]["quartiles"]
        ci = quartiles["paired_image_net_gap"]["bootstrap_ci95"]
        lines.append(
            f"| {STAGE_LABELS[stage]} | "
            f"{format_float(quartiles['bottom']['raw_error_rate'])} | "
            f"{format_float(quartiles['top']['raw_error_rate'])} | "
            f"{format_float(quartiles['bottom']['net_correction_rate'])} | "
            f"{format_float(quartiles['top']['net_correction_rate'])} | "
            f"{format_float(quartiles['net_correction_rate_gap'])} | "
            f"[{format_float(ci[0])}, {format_float(ci[1])}] |"
        )
    lines.extend(
        [
            "",
            "## 6. Per-class analysis",
            "",
            "Per-class results are analysis-only and do not alter the signal.",
            "",
            "| Stage | GT class | pixels | raw AUROC | corrected/harmed AUROC | corrected-harmed Cohen d |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for stage in STAGES:
        for class_id in range(4):
            metrics = result["stages"][stage]["per_class"][str(class_id)]
            lines.append(
                f"| {STAGE_LABELS[stage]} | {class_id} {metrics['name']} | "
                f"{metrics['pixels']:,} | "
                f"{format_float(metrics['raw_error']['auroc'])} | "
                f"{format_float(metrics['corrected_vs_harmed']['auroc'])} | "
                f"{format_float(metrics['corrected_vs_harmed']['cohens_d_corrected_minus_harmed'])} |"
            )
    lines.extend(
        [
            "",
            "## 7. Go / No-Go evaluation",
            "",
            "| Stage | passed conditions | stage go | strong go | major reverse |",
            "|---|---:|---|---|---|",
        ]
    )
    for stage in STAGES:
        item = decision["stage_decisions"][stage]
        lines.append(
            f"| {STAGE_LABELS[stage]} | {item['condition_count']}/4 | "
            f"{item['stage_go']} | {item['strong_go']} | "
            f"{item['major_reverse']} |"
        )
    lines.extend(
        [
            "",
            f"- Go stages: {decision['go_stage_count']}/3",
            f"- Strong-Go stages: {decision['strong_go_stage_count']}/3",
            f"- Major reverse stages: {decision['major_reverse_stage_count']}/3",
            f"- Final decision: **{decision['token']}**",
            "",
            "## 8. Interpretation",
            "",
        ]
    )
    if decision["strong_go"]:
        lines.extend(
            [
                "This is a Strong Go under the frozen criteria. At least two",
                "stages strongly rank both raw errors and beneficial versus",
                "harmful HFRM transitions.",
            ]
        )
    elif decision["signal_go"]:
        lines.extend(
            [
                "This is a Go, not a Strong Go. F56 provides the clearest",
                "corrected-versus-harmed signal; F28_1 is weaker, and F28_2",
                "passes through raw-error ranking plus quartile net correction",
                "rather than direct corrected-versus-harmed discrimination.",
                "The per-class reversals shown above are a material limitation",
                "and must not be hidden or used to retune the frozen formula.",
            ]
        )
    else:
        lines.extend(
            [
                "The frozen signal does not provide sufficiently consistent",
                "evidence to justify CDSR model engineering.",
            ]
        )
    lines.extend(["", "## 9. Development consequence", ""])
    if decision["signal_go"]:
        lines.extend(
            [
                "The frozen analytical signal passes the preregistered Phase-0",
                "gate. CDSR model engineering is permitted, but no model code or",
                "training is included in this Phase-0 branch.",
            ]
        )
    else:
        lines.extend(
            [
                "The frozen analytical signal does not pass the preregistered",
                "Phase-0 gate. Stop CDSR: do not implement the model, train, use",
                "test data, or tune the need formula.",
            ]
        )
    lines.extend(["", decision["token"]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--amp-dtype", choices=("none", "bf16"), default="bf16")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("The full BCSS Phase-0 audit requires CUDA")

    device = torch.device("cuda")
    dataset = BCSSValidationDataset(args.val_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    model = Net_CAM(
        n_class=4,
        rectifier_type="hfrm",
        context_mode="ch",
    )
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    stage_heads = {
        "stage1": model.ic_56,
        "stage2": model.ic1,
        "stage3": model.ic2,
    }
    chunks = {stage: [] for stage in STAGES}
    amp_enabled = args.amp_dtype == "bf16"
    processed = 0
    with torch.inference_mode():
        for images, masks, image_ids, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=amp_enabled
            ):
                feat_56, feat_28_1, feat_28_2, feat_deep = \
                    model._extract_backbone_features(images)
                raw_features = {
                    "stage1": feat_56,
                    "stage2": feat_28_1,
                    "stage3": feat_28_2,
                }
                (
                    rectified_56,
                    rectified_28_1,
                    rectified_28_2,
                    _,
                ) = model._rectify_features(
                    feat_56,
                    feat_28_1,
                    feat_28_2,
                    feat_deep,
                )
                rectified_features = {
                    "stage1": rectified_56,
                    "stage2": rectified_28_1,
                    "stage3": rectified_28_2,
                }
                raw_logits = {
                    stage: stage_heads[stage](raw_features[stage])
                    for stage in STAGES
                }
                post_logits = {
                    stage: stage_heads[stage](rectified_features[stage])
                    for stage in STAGES
                }
                deep_logits = model.fc8(feat_deep)

            for stage in STAGES:
                risk = analytical_need(raw_logits[stage], deep_logits)
                stage_mask = F.interpolate(
                    masks[:, None].float(),
                    size=raw_logits[stage].shape[-2:],
                    mode="nearest",
                )[:, 0].long()
                foreground = (stage_mask >= 0) & (stage_mask < 4)
                raw_prediction = raw_logits[stage].float().argmax(dim=1)
                post_prediction = post_logits[stage].float().argmax(dim=1)
                raw_correct = raw_prediction == stage_mask
                post_correct = post_prediction == stage_mask
                raw_wrong = ~raw_correct
                corrected = raw_wrong & post_correct
                harmed = raw_correct & (~post_correct)
                spatial = stage_mask.shape[-2] * stage_mask.shape[-1]
                expanded_image_ids = image_ids[:, None].repeat(1, spatial)
                chunks[stage].append(
                    {
                        "need": risk["need"][foreground]
                        .detach().cpu().numpy().astype(np.float32),
                        "gt": stage_mask[foreground]
                        .detach().cpu().numpy().astype(np.uint8),
                        "raw_wrong": raw_wrong[foreground]
                        .detach().cpu().numpy().astype(np.uint8),
                        "corrected": corrected[foreground]
                        .detach().cpu().numpy().astype(np.uint8),
                        "harmed": harmed[foreground]
                        .detach().cpu().numpy().astype(np.uint8),
                        "image_id": expanded_image_ids.reshape(
                            tuple(stage_mask.shape)
                        )[foreground.cpu()].numpy().astype(np.int32),
                    }
                )
            processed += images.shape[0]
            if processed % 200 == 0 or processed == len(dataset):
                print(
                    f"processed={processed}/{len(dataset)}",
                    flush=True,
                )

    stage_results = {
        stage: analyze_stage(chunks[stage], len(dataset))
        for stage in STAGES
    }
    decision = evaluate_decision(stage_results)
    result = {
        "protocol": {
            "dataset": "BCSS validation",
            "validation_samples": len(dataset),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "seed": SEED,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "amp_dtype": args.amp_dtype,
            "risk_dtype": "float32",
            "device": torch.cuda.get_device_name(device),
            "pytorch": torch.__version__,
            "background_handling": "exclude GT label 4; analyze foreground 0-3",
            "gt_alignment": "nearest resize to native stage resolution",
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "stages": stage_results,
        "decision": decision,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(render_report(result), encoding="utf-8")
    print(decision["token"], flush=True)


if __name__ == "__main__":
    main()
