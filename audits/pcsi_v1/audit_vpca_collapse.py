"""Frozen P3 VPCA root-cause audit using only image-level training labels."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from network.plip_adapter import FrozenPLIPAdapter
from tools.pdsr_vpca_phase0.common import CommonEvalDataset, label_from_name, load_concepts, sha256, write_json


TAUS = (.03, .05, .10, .20, .50, 1.00)
POOLS = ("mean", "top50", "top20", "top10", "max")


def softmax(x, axis=-1):
    x = np.asarray(x, np.float64); value = np.exp(x - x.max(axis=axis, keepdims=True))
    return value / value.sum(axis=axis, keepdims=True)


def entropy(values, axis=-1):
    p = np.clip(values, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=axis)


def pairwise_spearman(scores, seed=42, pairs=2000):
    if len(scores) < 2: return float("nan")
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(scores), pairs); b = rng.integers(0, len(scores), pairs)
    ranks = np.argsort(np.argsort(scores, axis=1), axis=1)
    delta = ranks[a] - ranks[b]
    return float((1 - 6 * np.square(delta).sum(1) / (8 * (8 * 8 - 1))).mean())


def pairwise_js(probabilities, seed=42, pairs=2000):
    if len(probabilities) < 2: return float("nan")
    rng = np.random.default_rng(seed)
    a = probabilities[rng.integers(0, len(probabilities), pairs)]
    b = probabilities[rng.integers(0, len(probabilities), pairs)]
    m = .5 * (a + b)
    return float((.5 * ((a * np.log(np.clip(a / m, 1e-12, None))).sum(1)
                          + (b * np.log(np.clip(b / m, 1e-12, None))).sum(1))).mean())


def top_frequencies(scores):
    order = np.argsort(-scores, axis=1)
    first = np.bincount(order[:, 0], minlength=8) / len(scores)
    second = np.bincount(order[:, 1], minlength=8) / len(scores)
    return first, second


def plot_text_space(output: Path, concepts: list[str], cosine: np.ndarray, class_names: list[str]):
    root = output / "text_cosine_matrices"; root.mkdir(parents=True, exist_ok=True)
    for cls, name in enumerate(class_names):
        matrix = cosine[cls * 8:(cls + 1) * 8, cls * 8:(cls + 1) * 8]
        fig, ax = plt.subplots(figsize=(11, 7))
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.set_xticklabels(range(1, 9)); ax.set_yticklabels(range(1, 9))
        ax.set_title(f"{name}: PLIP concept cosine (1–8)")
        fig.colorbar(im, ax=ax, fraction=.04)
        caption = "\n".join(f"{j + 1}. {concepts[cls * 8 + j]}" for j in range(8))
        fig.text(.05, .02, caption, fontsize=8, va="bottom")
        fig.tight_layout(rect=(0, .30, 1, 1)); fig.savefig(root / f"class_{cls}_cosine.png", dpi=140); plt.close(fig)
    np.save(root / "all_concept_cosine.npy", cosine)


def plot_training_examples(output: Path, image_root: Path, names: np.ndarray, labels: np.ndarray,
                           g: np.ndarray, concepts: list[str], class_names: list[str]):
    root = output.parent / "visualizations" / "vpca_training_no_gt"; root.mkdir(parents=True, exist_ok=True)
    file_by_name = {path.stem: path for path in [*image_root.glob("*.png"), *image_root.glob("*.jpg")]}
    for cls, class_name in enumerate(class_names):
        selected = np.flatnonzero(labels[:, cls])[:10]
        for rank, index in enumerate(selected, 1):
            image = np.asarray(Image.open(file_by_name[str(names[index])]).convert("RGB"))
            values = g[index, cls]; q01 = softmax(values / .1); q05 = softmax(values / .5)
            order = np.argsort(-values)
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            axes[0].imshow(image); axes[0].axis("off"); axes[0].set_title(f"{class_name} image (image-level label only)")
            x = np.arange(8); axes[1].bar(x - .25, values, width=.25, label="raw g20")
            axes[1].bar(x, q01, width=.25, label="q tau=.1")
            axes[1].bar(x + .25, q05, width=.25, label="q tau=.5")
            axes[1].set_xticks(x); axes[1].set_xticklabels([str(j + 1) for j in range(8)])
            axes[1].set_title("Top concept %d; raw rank %s" % (int(order[0] + 1), list((order + 1).astype(int))))
            axes[1].legend(fontsize=8)
            caption = "\n".join(f"{j + 1}. {concepts[cls * 8 + j]}" for j in range(8))
            fig.text(.52, .02, caption, fontsize=7, va="bottom")
            fig.tight_layout(rect=(0, .28, 1, 1)); fig.savefig(root / f"class_{cls}_{rank:02d}.png", dpi=120); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plip", type=Path, required=True)
    parser.add_argument("--p3-adapter", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--concept-cache", type=Path, required=True)
    parser.add_argument("--training-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "vpca_root_cause.json").exists(): raise FileExistsError("VPCA audit is write-once")
    concepts, bank = load_concepts(args.concepts); class_names = list(bank["class_index"].values())
    embeddings = torch.load(args.concept_cache, map_location="cpu", weights_only=False)["embeddings"].float()
    embeddings = F.normalize(embeddings, dim=-1)
    cosine = (embeddings @ embeddings.T).numpy()
    plot_text_space(args.output, concepts, cosine, class_names)
    text_rows = []
    for cls, name in enumerate(class_names):
        block = cosine[cls * 8:(cls + 1) * 8, cls * 8:(cls + 1) * 8]
        offdiag = block[np.triu_indices(8, 1)]
        centered = embeddings[cls * 8:(cls + 1) * 8].numpy()
        centered = centered - centered.mean(0, keepdims=True)
        singular = np.linalg.svd(centered, compute_uv=False)
        weights = singular / singular.sum(); effective_rank = float(np.exp(entropy(weights)))
        text_rows.append({"class": cls, "class_name": name, "mean_cosine": float(offdiag.mean()),
                          "median_cosine": float(np.median(offdiag)), "min_cosine": float(offdiag.min()),
                          "max_cosine": float(offdiag.max()), "std_cosine": float(offdiag.std()),
                          "effective_rank": effective_rank,
                          "text_space_collapse": bool(offdiag.mean() > .9 or effective_rank < 2.5)})
    pd.DataFrame(text_rows).to_csv(args.output / "text_effective_rank.csv", index=False)
    cross = np.zeros((4, 4))
    for c in range(4):
        for other in range(4):
            block = cosine[c * 8:(c + 1) * 8, other * 8:(other + 1) * 8]
            cross[c, other] = float(block[np.triu_indices(8, 1)].mean()) if c == other else float(block.mean())
    pd.DataFrame(cross, index=class_names, columns=class_names).to_csv(args.output / "text_cross_class_confusion.csv")
    plip = FrozenPLIPAdapter(args.plip).cuda().eval()
    state = torch.load(args.p3_adapter, map_location="cpu", weights_only=False)
    projection = nn.Linear(plip.hidden_size, 512).cuda().eval()
    projection.load_state_dict({"weight": state["pdsr.projections.2.weight"],
                                "bias": state["pdsr.projections.2.bias"]}, strict=True)
    for module in (plip, projection):
        for parameter in module.parameters(): parameter.requires_grad_(False)
    dataset = CommonEvalDataset(args.training_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    names_all, labels_all = [], []; pooled = {key: [] for key in POOLS}
    text_cuda = embeddings.cuda()
    with torch.inference_mode():
        for batch_index, (names, raw) in enumerate(loader, 1):
            raw = raw.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                token = plip.dense_tokens(raw)[-1]
                projected = projection(token.float()).transpose(1, 2).reshape(len(names), 512, 7, 7)
            response = torch.einsum("bdhw,kd->bkhw", F.normalize(projected.float(), dim=1), text_cuda)
            flat = response.flatten(2)
            scores = {
                "mean": flat.mean(-1), "top50": flat.topk(25, dim=-1).values.mean(-1),
                "top20": flat.topk(9, dim=-1).values.mean(-1),
                "top10": flat.topk(4, dim=-1).values.mean(-1), "max": flat.max(-1).values,
            }
            for key in POOLS: pooled[key].append(scores[key].cpu().numpy().astype(np.float32))
            names_all.extend(str(name) for name in names)
            labels_all.extend(label_from_name(str(name)).numpy().astype(np.uint8) for name in names)
            if batch_index % 50 == 0:
                print(json.dumps({"event": "PCSI_VPCA_PROGRESS", "images": len(names_all)}), flush=True)
    names_all = np.asarray(names_all); labels_all = np.stack(labels_all)
    pooled = {key: np.concatenate(value).reshape(-1, 4, 8) for key, value in pooled.items()}
    np.savez_compressed(args.output / "vpca_raw_scores.npz", image_ids=names_all, image_level_labels=labels_all,
                        **{f"g_{key}": value for key, value in pooled.items()})
    visual_rows, pooling_rows, temp_rows, bias_rows, sensitivity_rows = [], [], [], [], []
    rho = softmax(np.log(np.exp(pooled["top20"]).sum(-1)), axis=1)
    q01 = softmax(pooled["top20"] / .1, axis=-1)
    rho_rows = []
    for cls, name in enumerate(class_names):
        selected = labels_all[:, cls].astype(bool); scores = pooled["top20"][selected, cls]
        first, second = top_frequencies(scores)
        mean_g = scores.mean(0); var_image = scores.var(0); var_concept = mean_g.var()
        isr = float(var_image.mean() / (var_concept + 1e-12))
        visual_rows.append({"class": cls, "class_name": name, "positive_images": int(selected.sum()),
                            "raw_top1_max_frequency": float(first.max()),
                            "raw_top1_concept": int(first.argmax()), "raw_top1_frequency_by_concept": json.dumps(first.tolist()),
                            "raw_top2_frequency_by_concept": json.dumps(second.tolist()),
                            "ranking_entropy": float(entropy(first)),
                            "pairwise_spearman": pairwise_spearman(scores),
                            "score_std_by_concept": json.dumps(scores.std(0).tolist())})
        sensitivity_rows.append({"class": cls, "class_name": name, "mean_image_variance": float(var_image.mean()),
                                 "variance_of_concept_means": float(var_concept), "ISR": isr})
        for concept in range(8):
            bias_rows.append({"class": cls, "concept": concept, "concept_name": concepts[cls * 8 + concept],
                              "mean_g20": float(mean_g[concept]), "score_std": float(scores[:, concept].std()),
                              "global_bias": float(mean_g[concept] - mean_g.mean()),
                              "top1_frequency": float(first[concept])})
        for pool in POOLS:
            values = pooled[pool][selected, cls]; freq, _ = top_frequencies(values)
            pooling_rows.append({"class": cls, "pooling": pool, "top1_max_frequency": float(freq.max()),
                                 "top1_concept": int(freq.argmax()), "top1_entropy": float(entropy(freq))})
        for tau in TAUS:
            probabilities = softmax(scores / tau)
            freq, _ = top_frequencies(probabilities)
            temp_rows.append({"class": cls, "tau": tau, "raw_top1_max_frequency": float(first.max()),
                              "q_top1_max_frequency": float(freq.max()), "mean_entropy": float(entropy(probabilities).mean()),
                              "top2_mass_mean": float(np.sort(probabilities, axis=1)[:, -2:].sum(1).mean()),
                              "inter_image_js": pairwise_js(probabilities)})
        rho_rows.append({"class": cls, "class_name": name, "mean_rho_all": float(rho[:, cls].mean()),
                         "mean_rho_positive": float(rho[selected, cls].mean()),
                         "mean_rho_negative": float(rho[~selected, cls].mean()),
                         "rho_top1_frequency": float((rho.argmax(1) == cls).mean()),
                         "q_entropy_positive": float(entropy(q01[selected, cls]).mean())})
    pd.DataFrame(visual_rows).to_csv(args.output / "visual_rank_statistics.csv", index=False)
    pd.DataFrame(pooling_rows).to_csv(args.output / "pooling_audit.csv", index=False)
    pd.DataFrame(temp_rows).to_csv(args.output / "temperature_audit.csv", index=False)
    pd.DataFrame(rho_rows).to_csv(args.output / "rho_audit.csv", index=False)
    pd.DataFrame(bias_rows).to_csv(args.output / "concept_bias.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(args.output / "image_sensitivity.csv", index=False)
    plot_training_examples(args.output, args.training_images, names_all, labels_all,
                           pooled["top20"], concepts, class_names)
    text_collapse = any(row["text_space_collapse"] for row in text_rows)
    visual_lock = any(row["raw_top1_max_frequency"] > .8 and row["pairwise_spearman"] > .75 for row in visual_rows)
    pooling_frame = pd.DataFrame(pooling_rows)
    topk_lock = any(
        max(float(pooling_frame[(pooling_frame["class"] == cls) & (pooling_frame.pooling == key)].top1_max_frequency.iloc[0])
            for key in ("mean", "top50")) < .8
        and max(float(pooling_frame[(pooling_frame["class"] == cls) & (pooling_frame.pooling == key)].top1_max_frequency.iloc[0])
                for key in ("top20", "top10", "max")) > .8
        for cls in range(4))
    # Softmax preserves argmax exactly for every positive temperature. The plan's
    # top1-based temperature-lock criterion is mathematically impossible.
    temperature_sharpening = False
    global_bias = any(row["raw_top1_max_frequency"] > .8 and sensitivity_rows[i]["ISR"] < 1
                      for i, row in enumerate(visual_rows))
    rho_dominance = bool(np.bincount(rho.argmax(1), minlength=4).max() / len(rho) > .8
                         and np.median(rho.max(1)) > .8)
    flags = {"TEXT_SPACE_COLLAPSE": text_collapse,
             "VISUAL_RESPONSE_RANK_LOCK": visual_lock,
             "TOPK_AGGREGATION_LOCK": topk_lock,
             "TEMPERATURE_SHARPENING": temperature_sharpening,
             "GLOBAL_CONCEPT_BIAS": global_bias,
             "RHO_CLASS_DOMINANCE": rho_dominance}
    labels = [name for name, flag in flags.items() if flag]
    if len(labels) > 1: labels.append("COMBINED_FAILURE")
    if not labels: labels = ["NO_CLEAR_ROOT_CAUSE"]
    decision = {"VPCA_V1_STATUS": "CLOSED", "VPCA_ROOT_CAUSE": labels, **flags,
                "concept_top1_lock_occurs_at": "raw g20 before q/rho" if all(row["raw_top1_max_frequency"] > .8 for row in visual_rows) else "not all classes at raw g20",
                "temperature_top1_note": "Positive-temperature softmax is rank-preserving; tau can sharpen entropy but cannot create a new top1 lock",
                "images": len(names_all), "training_only": True, "segmentation_gt_opened": False,
                "p3_adapter_sha256": sha256(args.p3_adapter), "concept_cache_sha256": sha256(args.concept_cache),
                "mean_within_class_cosine": {str(row["class"]): row["mean_cosine"] for row in text_rows},
                "effective_rank": {str(row["class"]): row["effective_rank"] for row in text_rows},
                "raw_top1_frequency": {str(row["class"]): row["raw_top1_max_frequency"] for row in visual_rows},
                "pairwise_spearman": {str(row["class"]): row["pairwise_spearman"] for row in visual_rows},
                "ISR": {str(row["class"]): row["ISR"] for row in sensitivity_rows},
                "rho_max_median": float(np.median(rho.max(1))),
                "rho_top1_max_frequency": float(np.bincount(rho.argmax(1), minlength=4).max() / len(rho))}
    write_json(args.output / "vpca_root_cause.json", decision)
    print(json.dumps({"event": "PCSI_VPCA_AUDIT_COMPLETE", "root_causes": labels,
                      "images": len(names_all)}), flush=True)


if __name__ == "__main__":
    main()
