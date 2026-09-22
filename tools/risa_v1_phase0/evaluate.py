"""Fixed-E5 RISA evaluation, mechanism audits, counterfactual, and figures."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.umrf_v1.core import softmax_class
from network.cirv import extract_regions
from network.risa_v1 import RISAAdapter
from tools.eval_gcqm_full25_bcss_seed42 import (
    TTA, foreground_confusion, normalize_cam, prediction_from_cam, presence, scores_from_confusion,
)
from tools.pdsr_vpca_phase0.common import CommonEvalDataset
from tools.risa_v1_phase0.common import (
    CHECKPOINT_SHA256, CLASSES, REFERENCE_MIOU, sha256, write_csv, write_json,
)

VARIANTS = ("B0", "R1", "R2", "R3")


def resize_unflip(value: torch.Tensor, flip: tuple[int, ...]) -> torch.Tensor:
    value = F.interpolate(value.float(), size=(224, 224), mode="bilinear", align_corners=False)
    return torch.flip(value, dims=tuple(dimension + 1 for dimension in flip)) if flip else value


def normalize_batch(value: np.ndarray) -> np.ndarray:
    lower = value.min(axis=(2, 3), keepdims=True)
    upper = value.max(axis=(2, 3), keepdims=True)
    return (value - lower) / (upper - lower + 1.e-8)


def component_masks(prediction: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    return {(int(row["class_id"]), int(row["component_id"])): row["mask"]
            for row in extract_regions(prediction)}


def component_probability(score: np.ndarray, mask: np.ndarray) -> np.ndarray:
    probability = softmax_class(score)
    return probability[:, mask].mean(1)


def margin_summary(value: np.ndarray) -> dict:
    value = np.asarray(value, dtype=np.float64)
    return {"mean": float(np.mean(value)), "median": float(np.median(value)),
            "q25": float(np.quantile(value, .25)), "q75": float(np.quantile(value, .75)),
            "positive_fraction": float(np.mean(value > 0)), "count": int(len(value))}


def rate(rows: pd.DataFrame, condition: np.ndarray) -> dict:
    if len(rows) == 0:
        return {"components": 0, "component_rate": None, "pixel_area_weighted_rate": None}
    condition = np.asarray(condition, dtype=np.float64)
    return {"components": int(len(rows)), "component_rate": float(condition.mean()),
            "pixel_area_weighted_rate": float(np.average(condition, weights=rows.area.to_numpy()))}


@torch.inference_mode()
def infer(model: RISAAdapter, loader: DataLoader, ids: np.ndarray, truth_root: Path,
          bank_by_id: dict[str, np.ndarray], frame: pd.DataFrame, artifact: Path) -> dict:
    predictions = {variant: [] for variant in VARIANTS}
    confusions = {variant: [] for variant in VARIANTS}
    identity_rows: list[dict] = []
    query_pi, query_margin, query_entropy, query_gate = [], [], [], []
    deep_labels, soft_presence, image_labels = [], [], []
    by_image = {str(key): group for key, group in frame.groupby(frame.image_id.astype(str))}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    processed = 0
    for names, raw in loader:
        raw = raw.cuda(non_blocking=True)
        dummy = torch.ones((raw.shape[0], 4), device=raw.device)
        maps = {"R1": [], "R2": []}
        pi_views, margin_views, entropy_views, gate_views, presence_views, deep_views = [], [], [], [], [], []
        for input_flip, output_flip in TTA:
            value = torch.flip(raw, dims=input_flip) if input_flip else raw
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base = model.frozen_forward(value, dummy)
                r1 = model.forward_from_base(base, hard_refinement=False)
                r2 = model.forward_from_base(base, hard_refinement=True)
            maps["R1"].append(resize_unflip(r1["class_map"], output_flip).cpu())
            maps["R2"].append(resize_unflip(r2["class_map"], output_flip).cpu())
            pi_views.append(r2["identity_prob"].float().cpu())
            top = r2["identity_prob"].float().topk(2, dim=-1).values
            margin_views.append((top[..., 0] - top[..., 1]).cpu())
            entropy_views.append(r2["identity_entropy"].float().cpu())
            gate_views.append(r2["hard_gate"].float().cpu())
            presence_views.append(r2["presence_prob"].float().cpu())
            deep_views.append(base["deep_gate"].float().cpu())
        averaged = {key: normalize_batch(torch.stack(value).mean(0).numpy()) for key, value in maps.items()}
        pi = torch.stack(pi_views).mean(0).numpy()
        qmargin = torch.stack(margin_views).mean(0).numpy()
        qentropy = torch.stack(entropy_views).mean(0).numpy()
        qgate = torch.stack(gate_views).mean(0).numpy()
        psoft = torch.stack(presence_views).mean(0).numpy()
        dprob = torch.stack(deep_views).mean(0).numpy()
        for batch_index, raw_name in enumerate(names):
            image_id = str(raw_name)
            truth = np.asarray(Image.open(truth_root / f"{image_id}.png"))
            baseline = np.asarray(bank_by_id[image_id], dtype=np.uint8)
            hard_label = presence(dprob[batch_index])
            pred = {
                "B0": baseline,
                "R1": prediction_from_cam(averaged["R1"][batch_index], hard_label, np.empty((224, 224))),
                "R2": prediction_from_cam(averaged["R2"][batch_index], hard_label, np.empty((224, 224))),
                "R3": np.argmax((averaged["R2"][batch_index] * psoft[batch_index, :, None, None]).transpose(1, 2, 0), axis=2).astype(np.uint8),
            }
            scores = {"R1": averaged["R1"][batch_index], "R2": averaged["R2"][batch_index],
                      "R3": averaged["R2"][batch_index] * psoft[batch_index, :, None, None]}
            probability_maps = {variant: softmax_class(score) for variant, score in scores.items()}
            for variant in VARIANTS:
                predictions[variant].append(pred[variant].astype(np.uint8))
                confusions[variant].append(foreground_confusion(truth, pred[variant]))
            deep_labels.append(hard_label.astype(np.uint8))
            soft_presence.append(psoft[batch_index])
            # Validation segmentation is permitted only after fixed-E5 training;
            # here it supplies the post-hoc image-level presence reference.
            image_labels.append(np.asarray([(truth == class_id).any() for class_id in range(4)], dtype=np.float32))
            query_pi.append(pi[batch_index].astype(np.float16))
            query_margin.append(qmargin[batch_index].astype(np.float16))
            query_entropy.append(qentropy[batch_index].astype(np.float16))
            query_gate.append(qgate[batch_index].astype(np.float16))
            masks = component_masks(baseline)
            rows = by_image.get(image_id)
            if rows is not None:
                for _, source in rows.iterrows():
                    key = (int(source.baseline_class), int(source.component_id))
                    mask = masks[key]
                    row = {"image_id": image_id, "component_id": key[1], "baseline_class": key[0],
                           "area": int(source.area), "true_class": int(source.true_class),
                           "evaluable": bool(source.evaluable), "m1": bool(source.m1),
                           "baseline_correct": bool(source.baseline_correct),
                           "sequential5_pred": int(source.sequential5_pred),
                           "sequential4_pred": int(source.sequential4_pred),
                           "sequential3_pred": int(source.sequential3_pred),
                           "true_class_gate_off": bool(source.evaluable and hard_label[int(source.true_class)] == 0)}
                    true_class, rival = int(source.true_class), int(source.sequential5_pred)
                    row["baseline_margin_true_rival"] = float(source[f"sequential5_p{true_class}"] - source[f"sequential5_p{rival}"]) if source.evaluable else np.nan
                    for variant in ("R1", "R2", "R3"):
                        probability = probability_maps[variant][:, mask].mean(1)
                        row[f"{variant}_class"] = int(probability.argmax())
                        row[f"{variant}_margin_true_rival"] = float(probability[true_class] - probability[rival]) if source.evaluable else np.nan
                        for class_id in range(4):
                            row[f"{variant}_p{class_id}"] = float(probability[class_id])
                    identity_rows.append(row)
        processed += len(names)
        if processed % 400 < len(names) or processed == len(loader.dataset):
            print(json.dumps({"event": "RISA_EVAL_PROGRESS", "images": processed, "total": len(loader.dataset)}), flush=True)
    arrays = {variant: np.stack(value) for variant, value in predictions.items()}
    hist = {variant: np.stack(value) for variant, value in confusions.items()}
    np.savez_compressed(artifact / "all_variant_predictions.npz", image_ids=ids, **{f"prediction_{key}": value for key, value in arrays.items()},
                        **{f"confusion_{key}": value for key, value in hist.items()})
    np.savez_compressed(artifact / "risa_query_identity.npz", image_ids=ids, pi_qc=np.stack(query_pi),
                        margin=np.stack(query_margin), entropy=np.stack(query_entropy), hard_gate=np.stack(query_gate))
    identity = pd.DataFrame(identity_rows)
    identity.to_parquet(artifact / "component_identity_evidence.parquet", index=False)
    return {"predictions": arrays, "hist": hist, "identity": identity,
            "deep_labels": np.stack(deep_labels), "soft_presence": np.stack(soft_presence),
            "image_labels": np.stack(image_labels), "query_pi": np.stack(query_pi).astype(np.float32),
            "query_entropy": np.stack(query_entropy).astype(np.float32),
            "runtime": {"images": processed, "seconds": time.perf_counter() - started,
                        "seconds_per_image_all_variants": (time.perf_counter() - started) / processed,
                        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}}


@torch.inference_mode()
def region_dependency(model: RISAAdapter, dataset: CommonEvalDataset, ids: np.ndarray,
                      frame: pd.DataFrame, bank_by_id: dict[str, np.ndarray], count: int = 128) -> dict:
    hard = frame[
        frame.evaluable & (frame.sequential5_pred != frame.true_class)
        & (frame.sequential4_pred != frame.true_class) & (frame.sequential3_pred != frame.true_class)
    ]
    candidates = np.asarray(sorted(hard.image_id.astype(str).unique()))
    rng = np.random.default_rng(20260922)
    selected = rng.choice(candidates, size=min(count, len(candidates)), replace=False)
    index_by_id = {str(name): index for index, name in enumerate(ids)}
    by_image = {str(key): group for key, group in hard.groupby(hard.image_id.astype(str))}
    margins = {name: [] for name in ("real", "circular_shift", "spatial_permutation")}
    changed = {name: [] for name in ("circular_shift", "spatial_permutation")}
    for image_number, image_id in enumerate(selected):
        _name, raw = dataset[index_by_id[str(image_id)]]
        raw = raw[None].cuda(); dummy = torch.ones((1, 4), device=raw.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            base = model.frozen_forward(raw, dummy)
            real = model.forward_from_base(base, True)
        responsibility = real["responsibility"]
        height, width = responsibility.shape[-2:]
        dy = int(rng.integers(1, height)); dx = int(rng.integers(1, width))
        shifted = torch.roll(responsibility, shifts=(dy, dx), dims=(-2, -1))
        permutation = torch.as_tensor(rng.permutation(height * width), device=responsibility.device)
        permuted = responsibility.flatten(2)[..., permutation].reshape_as(responsibility)
        features = base["features"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            shift_out = model.risa(features["F3"], features["F4"], features["F5"], shifted, True)
            perm_out = model.risa(features["F3"], features["F4"], features["F5"], permuted, True)
        outputs = {"real": real, "circular_shift": shift_out, "spatial_permutation": perm_out}
        real_top = real["identity_prob"].argmax(-1)
        for name, output in outputs.items():
            class_map = torch.einsum("bqc,bqhw->bchw", output["identity_prob"].float(), responsibility.float()).clamp(0, 1)
            score = normalize_cam(F.interpolate(class_map, (224, 224), mode="bilinear", align_corners=False)[0].cpu().numpy())
            probability = softmax_class(score)
            masks = component_masks(bank_by_id[str(image_id)])
            for _, row in by_image[str(image_id)].iterrows():
                mask = masks[(int(row.baseline_class), int(row.component_id))]
                pooled = probability[:, mask].mean(1)
                margins[name].append(float(pooled[int(row.true_class)] - pooled[int(row.sequential5_pred)]))
            if name != "real":
                changed[name].append(float((output["identity_prob"].argmax(-1) != real_top).float().mean()))
        if (image_number + 1) % 32 == 0:
            print(json.dumps({"event": "RISA_REGION_DEPENDENCY_PROGRESS", "images": image_number + 1}), flush=True)
    summary = {name: margin_summary(np.asarray(values)) for name, values in margins.items()}
    change = {name: float(np.mean(values)) for name, values in changed.items()}
    passed = (summary["real"]["mean"] > summary["circular_shift"]["mean"]
              and summary["real"]["mean"] > summary["spatial_permutation"]["mean"]
              and min(change.values()) >= .05)
    return {"REGION_DEPENDENCY_PASS": bool(passed), "seed": 20260922,
            "images": int(len(selected)), "image_ids": selected.tolist(),
            "true_rival_margin": summary, "query_top1_change_fraction": change,
            "pass_rule": "real margin > both counterfactual margins and each query top1 change >= 0.05"}


@torch.inference_mode()
def runtime_audit(model: RISAAdapter, dataset: CommonEvalDataset, base_parameters: int,
                  risa_parameters: int) -> dict:
    _name, raw = dataset[0]
    raw = raw[None].cuda(); dummy = torch.ones((1, 4), device=raw.device)
    def measure(full: bool) -> tuple[float, float]:
        for _ in range(5):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base = model.frozen_forward(raw, dummy)
                if full: model.forward_from_base(base, True)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        for _ in range(30):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                base = model.frozen_forward(raw, dummy)
                if full: model.forward_from_base(base, True)
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / 30, torch.cuda.max_memory_allocated() / 1024 ** 3
    baseline_time, baseline_memory = measure(False)
    risa_time, risa_memory = measure(True)
    q, n, d, k, c = 196, 56 * 56, 128, int(56 * 56 * .20), 4
    convolution = 2 * (256 * d * n + 512 * d * 28 * 28 + 1024 * d * 28 * 28 + 3 * d * d * n)
    dynamic = 2 * q * n * d * 2 + 2 * q * k * d + 3 * 2 * q * d * c + 2 * q * n * c
    extra_flops = int(convolution + dynamic)
    baseline_flops = 214_910_692_676
    return {"baseline_parameters": base_parameters, "risa_parameters": risa_parameters,
            "total_parameters": base_parameters + risa_parameters,
            "additional_percent": 100 * risa_parameters / base_parameters,
            "lightweight_target_pass": risa_parameters < 1_000_000,
            "baseline_flops_per_view": baseline_flops, "risa_extra_flops_per_view": extra_flops,
            "risa_total_flops_per_view": baseline_flops + extra_flops,
            "extra_flops_percent": 100 * extra_flops / baseline_flops,
            "flops_method": "HQMR torch.profiler reference plus analytical RISA multiply/add count at 224x224",
            "baseline_seconds_per_image_per_view": baseline_time,
            "risa_seconds_per_image_per_view": risa_time,
            "runtime_overhead_percent": 100 * (risa_time / baseline_time - 1),
            "baseline_peak_vram_gib": baseline_memory, "risa_peak_vram_gib": risa_memory}


def analyze(bundle: dict, frame: pd.DataFrame, isolation: dict, dependency: dict,
            runtime: dict) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = {variant: scores_from_confusion(bundle["hist"][variant].sum(0)) for variant in VARIANTS}
    identity = bundle["identity"]
    hard = identity[
        identity.evaluable & (identity.sequential5_pred != identity.true_class)
        & (identity.sequential4_pred != identity.true_class) & (identity.sequential3_pred != identity.true_class)
    ].copy()
    correct = identity[identity.evaluable & identity.baseline_correct].copy()
    hard_rows, preservation_rows, per_class_rows = [], [], []
    identity_audit, preservation = {}, {}
    for variant in ("R1", "R2", "R3"):
        recovered = hard[f"{variant}_class"].to_numpy() == hard.true_class.to_numpy()
        identity_audit[variant] = {
            "HIRR": rate(hard, recovered),
            "margin": margin_summary(hard[f"{variant}_margin_true_rival"].to_numpy()),
            "margin_delta": margin_summary((hard[f"{variant}_margin_true_rival"] - hard.baseline_margin_true_rival).to_numpy()),
            "per_class": {},
        }
        kept = correct[f"{variant}_class"].to_numpy() == correct.true_class.to_numpy()
        preservation[variant] = rate(correct, kept)
        for class_id in range(4):
            h = hard[hard.true_class == class_id]
            c = correct[correct.true_class == class_id]
            h_recovered = h[f"{variant}_class"].to_numpy() == h.true_class.to_numpy()
            c_kept = c[f"{variant}_class"].to_numpy() == c.true_class.to_numpy()
            delta_margin = (h[f"{variant}_margin_true_rival"] - h.baseline_margin_true_rival).to_numpy()
            identity_audit[variant]["per_class"][str(class_id)] = {
                "HIRR": rate(h, h_recovered), "margin_delta": margin_summary(delta_margin),
                "CRP": rate(c, c_kept),
            }
            per_class_rows.append({"variant": variant, "class_id": class_id, "class_name": CLASSES[class_id],
                                   "HIRR": float(h_recovered.mean()) if len(h) else np.nan,
                                   "margin_delta_mean": float(np.mean(delta_margin)) if len(h) else np.nan,
                                   "CRP": float(c_kept.mean()) if len(c) else np.nan,
                                   "IoU": metrics[variant]["class_iou"][str(class_id)],
                                   "IoU_delta_pp": 100 * (metrics[variant]["class_iou"][str(class_id)] - metrics["B0"]["class_iou"][str(class_id)])})
        hard_rows.append({"variant": variant, "components": len(hard),
                          "HIRR": identity_audit[variant]["HIRR"]["component_rate"],
                          "HIRR_area_weighted": identity_audit[variant]["HIRR"]["pixel_area_weighted_rate"],
                          **{f"margin_{key}": value for key, value in identity_audit[variant]["margin"].items() if key != "count"},
                          **{f"margin_delta_{key}": value for key, value in identity_audit[variant]["margin_delta"].items() if key != "count"}})
        preservation_rows.append({"variant": variant, **preservation[variant]})
    top = bundle["query_pi"].argmax(-1)
    histogram = np.bincount(top.flatten(), minlength=4) / top.size
    coverage = np.stack([(top == class_id).any(1) for class_id in range(4)], axis=1).mean(0)
    collapse = bool(histogram.max() > .70)
    gate_off = identity[identity.evaluable & identity.true_class_gate_off]
    gate_rescue = {variant: rate(gate_off, gate_off[f"{variant}_class"].to_numpy() == gate_off.true_class.to_numpy())
                   for variant in ("R2", "R3")}
    absent = bundle["image_labels"] == 0
    false_activation = {}
    for variant in ("R2", "R3"):
        activated = np.stack([(bundle["predictions"][variant] == class_id).any((1, 2)) for class_id in range(4)], axis=1)
        false_activation[variant] = float(activated[absent].mean())
    class_margin_positive = {variant: sum(identity_audit[variant]["per_class"][str(c)]["margin_delta"]["mean"] > 0 for c in range(4))
                             for variant in ("R2", "R3")}
    candidate = max(("R2", "R3"), key=lambda name: metrics[name]["mIoU"])
    h = identity_audit[candidate]["HIRR"]["component_rate"] or 0.
    crp = preservation[candidate]["component_rate"] or 0.
    gain = metrics[candidate]["mIoU"] - metrics["B0"]["mIoU"]
    common = not collapse and isolation["ISOLATION_PASS"] and dependency["REGION_DEPENDENCY_PASS"]
    strong = common and h >= .15 and crp >= .98 and gain >= .006 and class_margin_positive[candidate] == 4
    go = common and h >= .10 and crp >= .97 and gain >= .003 and class_margin_positive[candidate] >= 3
    overall_margin = identity_audit[candidate]["margin_delta"]["mean"]
    mechanism_improved = h >= .03 and overall_margin > 0
    nogo = h < .03 or overall_margin <= 0 or crp < .95 or collapse or (gain < .001 and not mechanism_improved)
    decision = "STRONG_GO" if strong else "GO" if go else "NOGO"
    failures = []
    if collapse: failures.append("CLASS_COLLAPSE")
    if h < .03: failures.append("MIL_SUPERVISION_TOO_WEAK")
    if crp < .95: failures.append("DYNAMIC_ASSIGNMENT_HARMS_CORRECT_REGIONS")
    hard_gain = identity_audit["R2"]["HIRR"]["component_rate"] - identity_audit["R1"]["HIRR"]["component_rate"]
    if hard_gain <= .01: failures.append("HARD_RIVAL_REFINER_INEFFECTIVE")
    if false_activation["R3"] > false_activation["R2"] and gate_rescue["R3"]["component_rate"] <= gate_rescue["R2"]["component_rate"]:
        failures.append("SOFT_PRESENCE_OVERACTIVATION")
    if not dependency["REGION_DEPENDENCY_PASS"]: failures.append("REGION_READING_FAILURE")
    if decision == "NOGO" and not failures: failures.append("IDENTITY_FEATURE_INSUFFICIENT")
    segmentation_rows = [{"variant": variant, "mIoU": metrics[variant]["mIoU"], "mDice": metrics[variant]["mDice"],
                          **{f"IoU_C{c}": metrics[variant]["class_iou"][str(c)] for c in range(4)},
                          **{f"Dice_C{c}": metrics[variant]["class_dice"][str(c)] for c in range(4)}} for variant in VARIANTS]
    result = {"RISA_V1_PHASE0_DECISION": decision, "PRIMARY_VARIANT": candidate,
              "PRIMARY_MECHANISM_CONCLUSION": ("RISA recovered coherent-wrong region identity while preserving spatial responsibility."
                                               if decision != "NOGO" else "RISA-v1 did not jointly satisfy identity recovery, preservation, segmentation, and causal-dependency gates."),
              "metrics": metrics, "identity_audit": identity_audit, "correct_region_preservation": preservation,
              "class_breadth_positive_margin": class_margin_positive,
              "class_collapse": {"CLASS_COLLAPSE": collapse, "class_histogram": histogram.tolist(),
                                 "mean_identity_entropy": float(bundle["query_entropy"].mean()),
                                 "image_level_class_coverage": coverage.tolist(), "threshold": .70},
              "hard_rival_ablation": {"R2_minus_R1_HIRR": hard_gain,
                                       "R2_minus_R1_margin_delta": identity_audit["R2"]["margin_delta"]["mean"] - identity_audit["R1"]["margin_delta"]["mean"],
                                       "R2_minus_R1_CRP": preservation["R2"]["component_rate"] - preservation["R1"]["component_rate"]},
              "soft_presence_ablation": {"R3_minus_R2_mIoU_pp": 100 * (metrics["R3"]["mIoU"] - metrics["R2"]["mIoU"]),
                                           "gate_off_rescue": gate_rescue, "false_class_activation_rate": false_activation},
              "region_dependency": dependency, "isolation": isolation, "parameter_runtime": runtime,
              "evaluation_runtime": bundle["runtime"], "hard_m1_components": int(len(hard)),
              "m1_components": int(frame.m1.sum()), "failure_attribution": sorted(set(failures)),
              "decision_evidence": {"candidate": candidate, "HIRR": h, "CRP": crp,
                                    "mIoU_gain_pp": 100 * gain, "margin_delta": overall_margin,
                                    "positive_margin_classes": class_margin_positive[candidate],
                                    "common_causal_gates": common, "nogo_rule_triggered": bool(nogo)}}
    return result, pd.DataFrame(segmentation_rows), pd.DataFrame(hard_rows), pd.DataFrame(preservation_rows), pd.DataFrame(per_class_rows)


def visualize_group(rows: pd.DataFrame, name: str, model: RISAAdapter, dataset: CommonEvalDataset,
                    ids: np.ndarray, bank_by_id: dict[str, np.ndarray], truth_root: Path,
                    r2_predictions: np.ndarray, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    index_by_id = {str(value): index for index, value in enumerate(ids)}
    count = 0
    for _, row in rows.head(20).iterrows():
        image_id = str(row.image_id); image_index = index_by_id[image_id]
        _name, raw = dataset[image_index]
        image = np.asarray(Image.open(dataset.root / f"{image_id}.png").convert("RGB"))
        truth = np.asarray(Image.open(truth_root / f"{image_id}.png"))
        baseline = bank_by_id[image_id]
        masks = component_masks(baseline); component = masks[(int(row.baseline_class), int(row.component_id))]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            base = model.frozen_forward(raw[None].cuda(), torch.ones((1, 4), device="cuda"))
            risa = model.forward_from_base(base, True)
        responsibility = risa["responsibility"][0].float()
        component_grid = F.interpolate(torch.as_tensor(component, device="cuda", dtype=torch.float32)[None, None],
                                       responsibility.shape[-2:], mode="area")[0, 0]
        overlap = (responsibility * component_grid).sum((1, 2)) / component_grid.sum().clamp_min(1)
        query = int(overlap.argmax())
        resp = F.interpolate(responsibility[query][None, None], (224, 224), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        rival = F.interpolate(risa["rival_evidence"][0, query].reshape(1, 1, *responsibility.shape[-2:]), (224, 224), mode="bilinear", align_corners=False)[0, 0].float().cpu().numpy()
        selection = F.interpolate(risa["selection_score"][0, query].reshape(1, 1, *responsibility.shape[-2:]), (224, 224), mode="nearest")[0, 0].float().cpu().numpy()
        selected = np.zeros(responsibility.shape[-2] * responsibility.shape[-1], dtype=np.float32)
        selected[risa["selected_index"][0, query].cpu().numpy()] = 1
        selected = F.interpolate(torch.from_numpy(selected.reshape(1, 1, *responsibility.shape[-2:])).float(), (224, 224), mode="nearest")[0, 0].numpy()
        if name in ("recovered_hard_m1", "harmed_correct"):
            values = (image, truth, baseline, r2_predictions[image_index], resp)
            titles = ("Image", "GT", "B0", "R2", f"Responsibility q={query}")
            fig, axes = plt.subplots(1, 5, figsize=(15, 3))
            for axis, value, title in zip(axes, values, titles):
                axis.imshow(value, cmap=None if title == "Image" else "tab10", vmin=None if title == "Image" else 0,
                            vmax=None if title == "Image" else 4)
                axis.set_title(title); axis.axis("off")
            fig.suptitle(f"B0={int(row.sequential5_pred)} R2={int(row.R2_class)} true={int(row.true_class)} margin={float(row.R2_margin_true_rival):+.3f}")
        elif name == "hard_rival_token_maps":
            fig, axes = plt.subplots(1, 4, figsize=(12, 3))
            for axis, value, title in zip(axes, (resp, rival, selection, selected), ("R_q", "|d_q|", "R_q|d_q|", "Top-K 20%")):
                axis.imshow(value, cmap="magma"); axis.set_title(title); axis.axis("off")
            fig.suptitle(f"{image_id} query={query}")
        else:
            probability = risa["identity_prob"][0, query].float().cpu().numpy()
            order = np.argsort(-probability)
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].imshow(resp, cmap="magma"); axes[0].axis("off"); axes[0].set_title(f"query={query}, area={float(responsibility[query].sum()):.1f}")
            axes[1].bar(np.arange(4), probability); axes[1].set_xticks(np.arange(4), CLASSES, rotation=20); axes[1].set_ylim(0, 1)
            axes[1].set_title(f"top1={order[0]} top2={order[1]} margin={probability[order[0]]-probability[order[1]]:.3f} g={float(risa['hard_gate'][0,query]):.3f}")
        fig.tight_layout(); fig.savefig(output / f"{count:02d}_{image_id}.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        count += 1
    return count


def generate_visualizations(result: dict, bundle: dict, model: RISAAdapter, dataset: CommonEvalDataset,
                            ids: np.ndarray, bank_by_id: dict[str, np.ndarray], truth_root: Path,
                            artifact: Path) -> dict:
    frame = bundle["identity"]
    hard = frame[frame.evaluable & (frame.sequential5_pred != frame.true_class)
                 & (frame.sequential4_pred != frame.true_class) & (frame.sequential3_pred != frame.true_class)]
    recovered = hard[hard.R2_class == hard.true_class].sort_values("R2_margin_true_rival", ascending=False).drop_duplicates("image_id")
    correct = frame[frame.evaluable & frame.baseline_correct]
    harmed = correct[correct.R2_class != correct.true_class].sort_values("area", ascending=False).drop_duplicates("image_id")
    token = hard.sort_values("R2_margin_true_rival").drop_duplicates("image_id")
    identity = hard.sort_values("area", ascending=False).drop_duplicates("image_id")
    root = artifact / "visualizations"
    counts = {
        "recovered_hard_m1": visualize_group(recovered, "recovered_hard_m1", model, dataset, ids, bank_by_id, truth_root, bundle["predictions"]["R2"], root / "A_recovered_hard_m1"),
        "harmed_correct": visualize_group(harmed, "harmed_correct", model, dataset, ids, bank_by_id, truth_root, bundle["predictions"]["R2"], root / "B_harmed_correct_regions"),
        "hard_rival_token_maps": visualize_group(token, "hard_rival_token_maps", model, dataset, ids, bank_by_id, truth_root, bundle["predictions"]["R2"], root / "C_hard_rival_token_maps"),
        "identity_probability": visualize_group(identity, "identity_probability", model, dataset, ids, bank_by_id, truth_root, bundle["predictions"]["R2"], root / "D_identity_probability"),
    }
    write_json(root / "visualization_manifest.json", counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--umrf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    artifact = args.output / "artifacts/risa_v1_phase0"
    runtime = json.loads((artifact / "training_runtime.json").read_text())
    if runtime["status"] != "TRAINING_COMPLETE" or runtime["epochs"] != 5 or runtime["optimizer_steps"] != 5855:
        raise AssertionError("Fixed E5 training is incomplete")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("HQMR checkpoint changed")
    checkpoint = Path(runtime["checkpoint"])
    model = RISAAdapter(args.checkpoint).cuda().eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_trainable_state_dict(state["state_dict"])
    dataset = CommonEvalDataset(args.val_root / "img")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    ids = np.asarray([path.stem for path in dataset.files])
    bank_ids = np.load(args.umrf / "image_ids.npy", allow_pickle=False).astype(str)
    bank = np.load(args.umrf / "gt_free_prediction_maps.uint8.npy", mmap_mode="r")
    bank_by_id = {name: bank[0, index] for index, name in enumerate(bank_ids)}
    if len(ids) != 3418 or set(ids) != set(bank_ids):
        raise AssertionError("Frozen validation/manifest universe changed")
    frame = pd.read_parquet(args.umrf / "component_evidence_with_gt.parquet")
    hard_count = int((frame.evaluable & (frame.sequential5_pred != frame.true_class)
                      & (frame.sequential4_pred != frame.true_class) & (frame.sequential3_pred != frame.true_class)).sum())
    if hard_count != 5037 or int(frame.m1.sum()) != 4402:
        raise AssertionError("Frozen Hard-M1/M1 manifests changed")
    bundle = infer(model, loader, ids, args.val_root / "mask", bank_by_id, frame, artifact)
    # The uint8 frozen UMRF bank is a quantized cache of the fresh baseline.
    # Apply the preregistered 0.01 percentage-point replay tolerance.
    if 100 * abs(scores_from_confusion(bundle["hist"]["B0"].sum(0))["mIoU"] - REFERENCE_MIOU) > .01:
        raise AssertionError("B0 frozen bank exceeds the 0.01 pp replay tolerance")
    dependency = region_dependency(model, dataset, ids, frame, bank_by_id)
    write_json(artifact / "region_dependency_audit.json", dependency)
    isolation = json.loads((artifact / "gradient_causal_isolation_audit.json").read_text())
    compute = runtime_audit(model, dataset, runtime["baseline_parameters"], runtime["risa_parameters"])
    write_json(artifact / "parameter_runtime_audit.json", compute)
    result, segmentation, hard_audit, preservation, per_class = analyze(bundle, frame, isolation, dependency, compute)
    write_csv(artifact / "all_variant_metrics.csv", segmentation.to_dict("records"))
    write_csv(artifact / "hardm1_identity_audit.csv", hard_audit.to_dict("records"))
    write_csv(artifact / "correct_region_preservation.csv", preservation.to_dict("records"))
    write_csv(artifact / "per_class_identity_metrics.csv", per_class.to_dict("records"))
    result["visualizations"] = generate_visualizations(result, bundle, model, dataset, ids, bank_by_id, args.val_root / "mask", artifact)
    write_json(artifact / "final_result.json", result)
    primary = result["PRIMARY_VARIANT"]
    research = {
        "baseline_replay_pass": True, "isolation_pass": isolation["ISOLATION_PASS"],
        "baseline_miou": result["metrics"]["B0"]["mIoU"], "r1_miou": result["metrics"]["R1"]["mIoU"],
        "r2_miou": result["metrics"]["R2"]["mIoU"], "r3_miou": result["metrics"]["R3"]["mIoU"],
        "hardm1_identity_recovery": result["identity_audit"][primary]["HIRR"]["component_rate"],
        "correct_region_preservation": result["correct_region_preservation"][primary]["component_rate"],
        "true_rival_margin_delta": result["identity_audit"][primary]["margin_delta"]["mean"],
        "per_class_margin_delta": {key: value["margin_delta"]["mean"] for key, value in result["identity_audit"][primary]["per_class"].items()},
        "class_collapse": result["class_collapse"]["CLASS_COLLAPSE"],
        "region_dependency_pass": dependency["REGION_DEPENDENCY_PASS"],
        "hard_rival_gain": result["hard_rival_ablation"]["R2_minus_R1_HIRR"],
        "soft_presence_gain": result["soft_presence_ablation"]["R3_minus_R2_mIoU_pp"],
        "risa_params": compute["risa_parameters"], "decision": result["RISA_V1_PHASE0_DECISION"],
    }
    write_json(artifact / "risa_v1_research_state.json", research)
    write_json(artifact / "jev_risa_v1_sidecar.json", {"JEV_STATUS": "SKIPPED_NO_API" if not os.environ.get("TYPESAFE_API_KEY") else "NOT_RUN_TO_PRESERVE_PYTHON_DECISION",
                                                        "reason": "TYPESAFE_API_KEY unavailable" if not os.environ.get("TYPESAFE_API_KEY") else "Python preregistered decision is authoritative"})
    print(json.dumps({"event": "RISA_EVALUATION_COMPLETE", "decision": result["RISA_V1_PHASE0_DECISION"],
                      "mIoU": {key: value["mIoU"] for key, value in result["metrics"].items()},
                      "HIRR": {key: value["HIRR"]["component_rate"] for key, value in result["identity_audit"].items()}}), flush=True)


if __name__ == "__main__":
    main()
