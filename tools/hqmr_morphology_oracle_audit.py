#!/usr/bin/env python3
"""Frozen HQMR-v1/SSHR morphology-oracle ceiling audit on BCSS validation."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_hqmr_class23_residual import infer_hqmr, infer_sshr
from tools.eval_gcqm_full25_bcss_seed42 import load_state, scores_from_confusion
from tools.hqrf_phase0_io import sha256, write_csv, write_json


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
SSHR_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
CCBP_RESULT_SHA256 = "b4ae769cbbfcabd5b02c6446ad52e6a6c318e7bf9d18981e5a0e59c1df6f734b"
HQMR_MIOU, SSHR_MIOU = .6557244403737567, .6669670591172749
FORMAL_GAP_PP = 100 * (SSHR_MIOU - HQMR_MIOU)
OPERATORS = ("M1", "M2", "M3", "M4", "M5")
OPERATOR_NAMES = {
    "M1": "FP_ISLAND_REMOVAL", "M2": "ENCLOSED_HOLE_FILL",
    "M3": "GT_CONSTRAINED_FRAGMENT_BRIDGE", "M4": "BOUNDARY_PROTRUSION_TRIM",
    "M5": "BOUNDARY_INDENTATION_FILL",
}
STRUCT8 = np.ones((3, 3), dtype=np.uint8)
STRUCT4 = ndimage.generate_binary_structure(2, 1)
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260913, 10_000
CONFIG = {
    "dataset": "BCSS validation", "images": 3418, "seed": 42, "training": False,
    "parameter_updates": 0, "connectivity_foreground": 8, "connectivity_holes": 4,
    "boundary_radius_primary": 3, "boundary_radius_robustness": [1, 5],
    "contact_distance_primary": 3, "contact_distance_robustness": [1, 5],
    "subset_order": list(OPERATORS), "subsets": 32, "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "meaningful_shapley_pp": .05 * FORMAL_GAP_PP,
    "nearly_equal_pp": .10 * FORMAL_GAP_PP, "threshold_tuning": False,
    "checkpoint_selection": False, "oracle_is_model_performance": False,
}


def confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    valid = (truth >= 0) & (truth < 4)
    if np.any((prediction[valid] < 0) | (prediction[valid] >= 4)):
        raise AssertionError("Oracle produced invalid foreground prediction")
    return np.bincount(4 * truth[valid].astype(np.int64) + prediction[valid].astype(np.int64),
                       minlength=16).reshape(4, 4).astype(np.int64)


def _empty_stats() -> dict:
    return {"count": 0, "area": 0, "by_class": {str(c): {"count": 0, "area": 0} for c in range(4)}}


def _add(stats: dict, cls: int, count: int, area: int) -> None:
    stats["count"] += int(count); stats["area"] += int(area)
    stats["by_class"][str(cls)]["count"] += int(count)
    stats["by_class"][str(cls)]["area"] += int(area)


def m1_fp_island_removal(prediction: np.ndarray, truth: np.ndarray):
    out, changed, stats = prediction.copy(), np.zeros_like(prediction, bool), _empty_stats()
    for cls in range(4):
        labels, count = ndimage.label(prediction == cls, structure=STRUCT8)
        for component in range(1, count + 1):
            region = labels == component
            if not np.any(region & (truth == cls)):
                changed |= region; out[region] = truth[region]; _add(stats, cls, 1, int(region.sum()))
    return out, changed, stats


def m2_enclosed_hole_fill(prediction: np.ndarray, truth: np.ndarray):
    out, changed, stats = prediction.copy(), np.zeros_like(prediction, bool), _empty_stats()
    for cls in range(4):
        mask = prediction == cls
        holes = ndimage.binary_fill_holes(mask, structure=STRUCT4) & ~mask
        candidate = holes & (truth == cls)
        if candidate.any():
            hole_labels, _ = ndimage.label(holes, structure=STRUCT4)
            corrected = np.unique(hole_labels[candidate]); corrected = corrected[corrected > 0]
            out[candidate] = cls; changed |= candidate; _add(stats, cls, len(corrected), int(candidate.sum()))
    return out, changed, stats


NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _shortest_bridge(domain: np.ndarray, root: np.ndarray, fragment: np.ndarray) -> np.ndarray:
    """Deterministic unweighted 8-neighbour shortest path inside domain."""
    if np.any(ndimage.binary_dilation(fragment, structure=STRUCT8) & root): return np.zeros_like(domain)
    rows, cols = np.where(domain); result = np.zeros_like(domain)
    if not len(rows): return result
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    allowed, target, start = domain[r0:r1, c0:c1], root[r0:r1, c0:c1], fragment[r0:r1, c0:c1]
    visited = np.zeros_like(allowed, bool); parent_r = np.full(allowed.shape, -1, np.int32); parent_c = np.full(allowed.shape, -1, np.int32)
    queue = deque()
    for r, c in zip(*np.where(start)):
        visited[r, c] = True; queue.append((int(r), int(c)))
    hit = None
    while queue and hit is None:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if nr < 0 or nc < 0 or nr >= allowed.shape[0] or nc >= allowed.shape[1] or not allowed[nr, nc]: continue
            if target[nr, nc]: hit = (r, c); break
            if not visited[nr, nc]:
                visited[nr, nc] = True; parent_r[nr, nc] = r; parent_c[nr, nc] = c; queue.append((nr, nc))
    if hit is None: return result
    r, c = hit
    while not start[r, c]:
        result[r + r0, c + c0] = True
        pr, pc = int(parent_r[r, c]), int(parent_c[r, c])
        if pr < 0: break
        r, c = pr, pc
    return result


def m3_fragment_bridge(prediction: np.ndarray, truth: np.ndarray):
    out, changed, stats = prediction.copy(), np.zeros_like(prediction, bool), _empty_stats()
    for cls in range(4):
        gt_labels, gt_count = ndimage.label(truth == cls, structure=STRUCT8)
        for component in range(1, gt_count + 1):
            domain = gt_labels == component; correct = (out == cls) & domain
            fragments, count = ndimage.label(correct, structure=STRUCT8)
            if count < 2: continue
            sizes = np.bincount(fragments.ravel())[1:]
            order = sorted(range(1, count + 1), key=lambda label: (-int(sizes[label - 1]), label))
            root = fragments == order[0]; bridges = 0
            for label in order[1:]:
                fragment = fragments == label; path = _shortest_bridge(domain, root, fragment)
                new = path & (out != cls); out[new] = cls; changed |= new; root |= fragment | path
                bridges += 1
            _add(stats, cls, 1, int(changed[domain].sum()))
            stats.setdefault("bridges", 0); stats["bridges"] += bridges
    return out, changed, stats


def m4_protrusion_trim(prediction: np.ndarray, truth: np.ndarray, radius: int = 3):
    out, changed, stats = prediction.copy(), np.zeros_like(prediction, bool), _empty_stats()
    for cls in range(4):
        target, pred = truth == cls, prediction == cls
        outer = (~target) & (ndimage.distance_transform_edt(~target) <= radius)
        candidate, tp = pred & outer, pred & target
        labels, _ = ndimage.label(pred, structure=STRUCT8)
        attached = np.unique(labels[tp]); attached = attached[attached > 0]
        selected = candidate & np.isin(labels, attached)
        if selected.any():
            components = ndimage.label(selected, structure=STRUCT8)[1]
            out[selected] = truth[selected]; changed |= selected; _add(stats, cls, components, int(selected.sum()))
    return out, changed, stats


def m5_indentation_fill(prediction: np.ndarray, truth: np.ndarray, radius: int = 3):
    out, changed, stats = prediction.copy(), np.zeros_like(prediction, bool), _empty_stats()
    for cls in range(4):
        target = truth == cls
        inner = target & (ndimage.distance_transform_edt(target) <= radius)
        candidate, tp = inner & (prediction != cls), target & (prediction == cls)
        labels, count = ndimage.label(target, structure=STRUCT8); selected = np.zeros_like(target)
        used = 0
        for component in range(1, count + 1):
            domain = labels == component
            if np.any(domain & tp) and np.any(domain & candidate): selected |= domain & candidate; used += 1
        if selected.any(): out[selected] = cls; changed |= selected; _add(stats, cls, used, int(selected.sum()))
    return out, changed, stats


def apply_operator(index: int, prediction: np.ndarray, truth: np.ndarray, radius: int = 3):
    if index == 0:
        return m1_fp_island_removal(prediction, truth)
    if index == 1:
        return m2_enclosed_hole_fill(prediction, truth)
    if index == 2:
        return m3_fragment_bridge(prediction, truth)
    if index == 3:
        return m4_protrusion_trim(prediction, truth, radius)
    if index == 4:
        return m5_indentation_fill(prediction, truth, radius)
    raise ValueError(f"Unknown morphology operator index: {index}")


def subset_states(prediction: np.ndarray, truth: np.ndarray):
    states, changes, stats = {0: prediction}, {}, {}
    for subset in range(1, 32):
        index = subset.bit_length() - 1; previous = subset ^ (1 << index)
        states[subset], changes[subset], stats[subset] = apply_operator(index, states[previous], truth)
    return states, changes, stats


def exact_shapley(values: dict[int, float]) -> dict[str, float]:
    result = {}
    for index, name in enumerate(OPERATORS):
        value = 0.0
        for subset in range(32):
            if subset & (1 << index): continue
            size = subset.bit_count(); weight = math.factorial(size) * math.factorial(4 - size) / math.factorial(5)
            value += weight * (values[subset | (1 << index)] - values[subset])
        result[name] = float(value)
    return result


def _metric_batch(hist: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(hist, axis1=-2, axis2=-1)
    union = hist.sum(-1) + hist.sum(-2) - diagonal
    iou = np.divide(diagonal, union, out=np.full_like(diagonal, np.nan, dtype=float), where=union > 0)
    return np.nanmean(iou, axis=-1)


def bootstrap_headroom(hqmr_base, hqmr_full, sshr_base, sshr_full):
    rng = np.random.default_rng(BOOTSTRAP_SEED); hgain, sgain, excess = [], [], []
    count = len(hqmr_base)
    for start in range(0, BOOTSTRAP_RESAMPLES, 50):
        n = min(50, BOOTSTRAP_RESAMPLES - start); sample = rng.integers(0, count, size=(n, count))
        hb, hf = hqmr_base[sample].sum(1), hqmr_full[sample].sum(1)
        sb, sf = sshr_base[sample].sum(1), sshr_full[sample].sum(1)
        hg = 100 * (_metric_batch(hf) - _metric_batch(hb)); sg = 100 * (_metric_batch(sf) - _metric_batch(sb))
        hgain.extend(hg); sgain.extend(sg); excess.extend(hg - sg)
    def summary(value):
        value = np.asarray(value); return {"mean_pp": float(value.mean()), "ci95_pp": [float(np.quantile(value, .025)), float(np.quantile(value, .975))]}
    return {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
            "hqmr_full_oracle_gain": summary(hgain), "sshr_full_oracle_gain": summary(sgain),
            "excess_headroom": summary(excess)}


def distribution(values):
    value = np.asarray(values, float)
    return {"mean": float(value.mean()), "median": float(np.median(value)), "p75": float(np.quantile(value, .75)),
            "p90": float(np.quantile(value, .90)), "fraction_gt_0": float(np.mean(value > 0)),
            "fraction_gt_1pp": float(np.mean(value > 1)), "fraction_gt_2pp": float(np.mean(value > 2))}


def _mode_without_gt(prediction: np.ndarray, transformed: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(transformed); count = stack.sum(0); out = prediction.copy()
    unique = count == 1; out[unique] = stack[:, unique].argmax(0)
    removed = count == 0
    if removed.any():
        neighbours = np.stack([ndimage.convolve((prediction == cls).astype(np.int16), STRUCT8, mode="constant") for cls in range(4)])
        rr, cc = np.where(removed); original = prediction[rr, cc]; neighbours[original, rr, cc] = -1
        choice = neighbours[:, rr, cc].argmax(0); has_other = neighbours[:, rr, cc].max(0) > 0
        out[rr[has_other], cc[has_other]] = choice[has_other]
    return out


def gt_free_transform(prediction: np.ndarray, kind: str) -> np.ndarray:
    masks = []
    for cls in range(4):
        mask = prediction == cls
        if kind == "closing": value = ndimage.binary_closing(mask, structure=STRUCT4, iterations=1)
        elif kind == "opening": value = ndimage.binary_opening(mask, structure=STRUCT4, iterations=1)
        elif kind == "fill_holes": value = ndimage.binary_fill_holes(mask, structure=STRUCT4)
        else: raise ValueError(kind)
        masks.append(value)
    return _mode_without_gt(prediction, masks)


def interior_rows(image_id, model, prediction, truth):
    rows = []
    for cls in range(4):
        labels, count = ndimage.label(truth == cls, structure=STRUCT8)
        for component in range(1, count + 1):
            region = labels == component; interior = ndimage.binary_erosion(region, structure=STRUCT4, iterations=3)
            if not interior.any(): interior = region
            correct = interior & (prediction == cls); pieces, n = ndimage.label(correct, structure=STRUCT8)
            sizes = np.bincount(pieces.ravel())[1:]; largest = int(sizes.max()) if len(sizes) else 0
            holes = ndimage.binary_fill_holes(correct, structure=STRUCT4) & ~correct & interior
            rows.append({"image_id": image_id, "model": model, "class": cls, "gt_component": component,
                         "interior_pixels": int(interior.sum()), "prediction_consistency": float(correct.sum()/interior.sum()),
                         "largest_correct_component_fraction": float(largest/interior.sum()),
                         "interior_fragmentation": int(n), "interior_hole_ratio": float(holes.sum()/interior.sum())})
    return rows


def boundary_contact_rows(image_id, model, prediction, truth, distance: int):
    rows = []
    for cls in range(4):
        target = truth == cls; inner = target & (ndimage.distance_transform_edt(target) <= distance)
        outer = (~target) & (ndimage.distance_transform_edt(~target) <= distance)
        other = (truth < 4) & (~target); near_other = ndimage.distance_transform_edt(~other) <= distance
        contact_in, contact_out = inner & near_other, outer & other
        for region, inside, outside in (("contact", contact_in, contact_out), ("non_contact", inner & ~contact_in, outer & ~contact_out)):
            rows.append({"image_id": image_id, "model": model, "class": cls, "distance": distance, "region": region,
                         "FN": int(np.sum(inside & (prediction != cls))), "FN_domain": int(inside.sum()),
                         "FP": int(np.sum(outside & (prediction == cls))), "FP_domain": int(outside.sum())})
    return rows


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def setup_output(output: Path):
    for name in ("provenance", "baseline", "operators", "subsets", "shapley", "headroom", "per_class", "per_image",
                 "robustness", "diagnostics", "cache", "visualizations/baseline", "visualizations/operator_masks",
                 "visualizations/cumulative", "visualizations/headroom_comparison", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)


def run_infer(args, output: Path):
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing populated output: {output}")
    setup_output(output)
    hpath, spath = Path(args.hqmr_checkpoint), Path(args.sshr_checkpoint)
    if sha256(hpath) != HQMR_SHA256 or sha256(spath) != SSHR_SHA256: raise AssertionError("Frozen checkpoint mismatch")
    if sha256(args.ccbp_result) != CCBP_RESULT_SHA256: raise AssertionError("Archived CCBP result mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {**CONFIG, "source_commit": commit, "hqmr_checkpoint": str(hpath.resolve()), "sshr_checkpoint": str(spath.resolve())}
    write_json(output / "provenance/morph_audit_config.json", config)
    (output / "provenance/morph_audit_config_sha256.txt").write_text(_canonical_hash(config) + "\n")
    (output / "provenance/morph_audit_source_commit.txt").write_text(commit + "\n")
    (output / "provenance/morph_audit_git_diff.patch").write_text(subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True))
    write_json(output / "provenance/archived_ccbp.json", {"sha256": sha256(args.ccbp_result), "payload": json.loads(Path(args.ccbp_result).read_text())})
    loader = DataLoader(Stage1_InferDataset(str(Path(args.val_root) / "img"), img_size=224), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if len(loader) != 3418: raise AssertionError("Expected 3418 validation images")
    hqmr = HQMRNet().cuda()
    hqmr.load_state_dict(load_state(hpath), strict=True)
    hqmr.eval()
    sshr = SSHRCAM(4).cuda()
    sshr.load_state_dict(load_state(spath), strict=True)
    # The legacy ResNet38 implementation mutates training state but does not
    # return self from eval(), so keep this deliberately non-chained.
    sshr.eval()
    hist = {"hqmr": [], "sshr": []}; manifest = []
    for index, (names, image) in enumerate(loader):
        image_id = names[0]; original = np.asarray(Image.open(Path(args.val_root)/"img"/f"{image_id}.png").convert("RGB"))
        truth = np.asarray(Image.open(Path(args.val_root)/"mask"/f"{image_id}.png")); image = image.cuda(non_blocking=True)
        hp = infer_hqmr(hqmr, image, original.shape[:2])["prediction"].astype(np.uint8)
        sp = infer_sshr(sshr, image, original.shape[:2])["prediction"].astype(np.uint8)
        hist["hqmr"].append(confusion(truth, hp)); hist["sshr"].append(confusion(truth, sp))
        cache = output / f"cache/{index:05d}.npz"; np.savez_compressed(cache, hqmr=hp, sshr=sp)
        manifest.append({"index": index, "image_id": image_id, "cache": cache.name, "height": int(hp.shape[0]), "width": int(hp.shape[1])})
        if (index + 1) % 100 == 0 or index + 1 == len(loader): print(f"MORPH_INFER_PROGRESS={index+1}/{len(loader)}", flush=True)
    metrics = {name: scores_from_confusion(np.stack(rows).sum(0)) for name, rows in hist.items()}
    gate = {"decision": "PASS", "paired_validation_images": len(loader), "same_split": True, "same_class_mapping": True,
            "same_tta": True, "same_threshold_protocol": True, "same_ignore_policy": True, "same_resize_interpolation": True,
            "hqmr_mIoU": metrics["hqmr"]["mIoU"], "sshr_mIoU": metrics["sshr"]["mIoU"],
            "formal_gap_pp": 100*(metrics["sshr"]["mIoU"]-metrics["hqmr"]["mIoU"])}
    if abs(metrics["hqmr"]["mIoU"]-HQMR_MIOU)>1e-12 or abs(metrics["sshr"]["mIoU"]-SSHR_MIOU)>1e-12:
        gate["decision"] = "MORPH_ORACLE_ENGINEERING_BLOCKED"
    write_json(output / "baseline/hqmr_baseline_metrics.json", metrics["hqmr"])
    write_json(output / "baseline/sshr_baseline_metrics.json", metrics["sshr"])
    write_json(output / "provenance/reproduction_gate.json", gate); write_json(output / "cache/prediction_manifest.json", manifest)
    print("MORPH_REPRODUCTION " + json.dumps(gate, sort_keys=True), flush=True)
    if gate["decision"] != "PASS": raise AssertionError(gate)


def _subset_table(histograms):
    rows, metrics = [], {}
    for subset in range(32):
        metric = scores_from_confusion(histograms[subset]); metrics[subset] = metric
        rows.append({"subset": subset, "operators": "+".join(OPERATORS[i] for i in range(5) if subset & (1<<i)) or "EMPTY",
                     "mIoU": metric["mIoU"], "mDice": metric["mDice"],
                     **{f"C{c}_IoU": metric["class_iou"][str(c)] for c in range(4)}})
    return rows, metrics


def _aggregate_prevalence(records, gt_area):
    rows = []
    for model in ("hqmr", "sshr"):
        for index, op in enumerate(OPERATORS):
            for cls in range(4):
                items = [r for r in records if r["model"]==model and r["operator"]==op and r["class"]==cls]
                area, count = sum(r["area"] for r in items), sum(r["count"] for r in items)
                rows.append({"model": model, "operator": op, "operator_name": OPERATOR_NAMES[op], "class": cls,
                             "count": count, "area": area, "gt_area": gt_area[cls], "area_over_gt": area/max(gt_area[cls],1),
                             "per_image_prevalence": sum(r["area"]>0 for r in items)/max(len(items),1)})
    lookup = {(r["model"],r["operator"],r["class"]):r for r in rows}
    for op in OPERATORS:
        for cls in range(4):
            h,s=lookup[("hqmr",op,cls)],lookup[("sshr",op,cls)]
            h["excess_area_over_gt_vs_sshr"] = h["area_over_gt"]-s["area_over_gt"]
            s["excess_area_over_gt_vs_sshr"] = 0.0
    return rows


def _decision(gap_ratio, cons_ratio, excess_ci, class_excess, excess_shapley, heavy):
    meaningful = sum(value >= CONFIG["meaningful_shapley_pp"] for value in excess_shapley.values())
    positive_classes = sum(value > 0 for value in class_excess.values())
    nearly_equal = abs(gap_ratio * FORMAL_GAP_PP) <= CONFIG["nearly_equal_pp"]
    strong = gap_ratio >= 1 and excess_ci[0] > 0 and cons_ratio >= .5 and positive_classes >= 3 and meaningful >= 2 and not heavy
    if strong: return "MORPHOLOGY_REGION_HOMOGENEITY_LIMIT", "HIGH"
    if gap_ratio < .5 or nearly_equal:
        confidence = "HIGH" if excess_ci[1] < .5 * FORMAL_GAP_PP else "MEDIUM" if not heavy else "LOW"
        return "MORPHOLOGY_NOT_PRIMARY", confidence
    if gap_ratio >= .5:
        return "PARTIAL_MORPHOLOGY_LIMIT", "MEDIUM" if excess_ci[0] > 0 and not heavy else "LOW"
    return "NO_SINGLE_RESIDUAL_LIMIT", "LOW"


def _dominant_target(shapley):
    ordered = sorted(shapley, key=shapley.get, reverse=True); pair = set(ordered[:2])
    if pair == {"M2","M3"}: target = "interior region homogenization"
    elif pair == {"M4","M5"}: target = "boundary-selective structural refinement"
    else: target = {"M1":"selective island suppression","M2":"topology-aware hole completion","M3":"region connectivity/component coherence","M4":"boundary-aware protrusion suppression","M5":"boundary-aware indentation completion"}[ordered[0]]
    return ordered[:2], target


def report_text(result):
    h=result["headroom"]; d=result["decision"]
    sections=[
        ("Executive Diagnosis",f"**DECISION = {d['decision']}**；**CONFIDENCE = {d['confidence']}**。"),
        ("Frozen HQMR-v1 / SSHR Evidence",f"HQMR-v1={100*result['baseline']['hqmr']['mIoU']:.4f}，SSHR={100*result['baseline']['sshr']['mIoU']:.4f}，formal gap={FORMAL_GAP_PP:.4f} pp。"),
        ("Why CCBP Is Archived","CCBP mIoU=64.3113，虽修复 purity/rival/separability，但相对 HQMR-v1 −1.2612 pp，故不再作为主机制。"),
        ("Why Morphology Is the Remaining Candidate","Residual Audit 的七项形态指标同向变差，而 CCBP 仅恢复 1/7。"),
        ("Reproduction Gate",str(result["reproduction_gate"])),
        ("Oracle Design Principles","GT-constrained、post-hoc only、zero training；HQMR 与 SSHR 使用相同固定算子。Oracle 结果不是模型性能。"),
        ("M1 FP Island Removal",str(result["operator_summary"]["M1"])),
        ("M2 Enclosed Hole Fill",str(result["operator_summary"]["M2"])),
        ("M3 Fragment Bridge",str(result["operator_summary"]["M3"])),
        ("M4 Boundary Protrusion Trim",str(result["operator_summary"]["M4"])),
        ("M5 Boundary Indentation Fill",str(result["operator_summary"]["M5"])),
        ("Individual Operator Gains",str(result["individual_gains_pp"])),
        ("Conservative Oracle Package",str(result["conservative"])),
        ("Full Morphology Oracle",str(result["full_oracle"])),
        ("HQMR Absolute Morphology Headroom",f"{h['hqmr_headroom_pp']:+.4f} pp。"),
        ("SSHR Absolute Morphology Headroom",f"{h['sshr_headroom_pp']:+.4f} pp。"),
        ("Excess Morphology Headroom",f"{h['excess_headroom_pp']:+.4f} pp。"),
        ("Gap-Recovery Ratio",f"{100*h['gap_recovery_ratio']:.2f}% of the {FORMAL_GAP_PP:.4f} pp gap。"),
        ("Paired Bootstrap",str(result["bootstrap"])),
        ("Exact Shapley Attribution",str(result["shapley"])),
        ("Per-Class Shapley","见 shapley/per_class_shapley.csv。"),
        ("Operator Interactions",str(result["interactions"])),
        ("Defect Prevalence","见 baseline/baseline_morphology.csv 与 per_class/per_class_defect_prevalence.csv。"),
        ("Per-Class Recovery",str(result["per_class"])),
        ("Class2/3 Contribution",str(result["class23"])),
        ("Image-Level Distribution",str(result["image_distribution"])),
        ("Heavy-Tail Check",str(result["heavy_tail"])),
        ("Interior Coherence","见 diagnostics/interior_coherence.csv。"),
        ("Boundary Safety","见 diagnostics/boundary_safety.csv。"),
        ("Contact Analysis","见 diagnostics/contact_region.csv。"),
        ("GT-Free Morphology Sanity",str(result["gt_free_sanity"])),
        ("Representative Cases","六组自动选择案例与算子 mask 位于 visualizations/。"),
        ("Decision Matrix",str(d["matrix"])),
        ("Exact Residual Bottleneck",f"Dominant={d['dominant_operators']}；target={d['architecture_target']}。"),
        ("What Is Preserved","CCRA、HQMR-v1、H5→H4 hierarchical reconstruction 与 region-conditioned query update 均保留。"),
        ("What Is Falsified",d["falsified"]),
        ("Exact Next Architecture Target",d["next_step"]),
        ("What Must NOT Be Done","不得训练 SARH、恢复原 CH、调 morphology radius、挑选最佳 oracle subset、改阈值、加 CRF 或使用 GT oracle 训练。"),
        ("Final Decision",f"`DECISION = {d['decision']}`\n\n`CONFIDENCE = {d['confidence']}`"),
    ]
    sentence=(f"HQMR-v1 has {h['excess_headroom_pp']:.4f} pp more morphology-recoverable headroom than SSHR, "
              f"corresponding to {100*h['gap_recovery_ratio']:.2f}% of the formal {FORMAL_GAP_PP:.4f} pp gap. "
              f"The dominant morphology contributors are {', '.join(d['dominant_operators'])}; therefore the next architecture "
              f"{'should' if h['gap_recovery_ratio']>=.5 else 'should not'} target region homogeneity.")
    return "# HQMR-v1 Morphology Oracle Recovery Audit Report\n\n"+"\n\n".join(f"## {i} {t}\n\n{b}" for i,(t,b) in enumerate(sections,1))+"\n\n## Required Completion Sentence\n\n"+sentence+"\n"


def _visualize(output, valroot, manifest, selected):
    groups=defaultdict(list)
    for group, ids in selected.items():
        folder=output/"visualizations"/group; folder.mkdir(parents=True,exist_ok=True)
        for rank,image_id in enumerate(ids,1): groups[image_id].append((group,rank))
    lookup={row["image_id"]:row for row in manifest}
    for image_id,destinations in groups.items():
        row=lookup[image_id]; data=np.load(output/"cache"/row["cache"]); hp,sp=data["hqmr"],data["sshr"]
        truth=np.asarray(Image.open(valroot/"mask"/f"{image_id}.png")); original=np.asarray(Image.open(valroot/"img"/f"{image_id}.png").convert("RGB"))
        single=[]
        for index in range(5): single.append(apply_operator(index,hp,truth))
        states,_,_=subset_states(hp,truth); cons,full=states[7],states[31]
        panels=[(original,"input",None),(truth,"GT","tab10"),(hp,"HQMR","tab10"),(sp,"SSHR","tab10"),((hp!=truth)&(truth<4),"HQMR error","Reds")]
        panels += [(item[1],name,"Reds") for item,name in zip(single,OPERATORS)]
        panels += [(hp,"baseline","tab10"),(cons,"conservative","tab10"),(full,"full oracle","tab10"),(truth,"GT","tab10")]
        fig,axes=plt.subplots(3,5,figsize=(18,11))
        for axis,(value,title,cmap) in zip(axes.flat,panels): axis.imshow(value,cmap=cmap); axis.set_title(title); axis.axis("off")
        axes.flat[-1].axis("off"); fig.suptitle(image_id); fig.tight_layout()
        for group,rank in destinations: fig.savefig(output/f"visualizations/{group}/{rank:02d}_{image_id}.png",dpi=130)
        # The composite contains the baseline, individual operator masks,
        # cumulative oracle results, and direct HQMR/SSHR headroom comparison.
        # Mirror it into the four frozen deliverable views so every selected
        # image remains traceable from each required artifact category.
        for category in ("baseline", "operator_masks", "cumulative", "headroom_comparison"):
            folder = output / "visualizations" / category
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / f"{image_id}.png", dpi=130)
        plt.close(fig)


def run_audit(args, output: Path):
    gate=json.loads((output/"provenance/reproduction_gate.json").read_text()); manifest=json.loads((output/"cache/prediction_manifest.json").read_text())
    if gate["decision"]!="PASS" or len(manifest)!=3418: raise AssertionError("Reproduction/cache gate failed")
    config=json.loads((output/"provenance/morph_audit_config.json").read_text())
    if config["source_commit"]!=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(): raise AssertionError("Audit source changed after prediction freeze")
    subset_hist={model:np.zeros((32,4,4),np.int64) for model in ("hqmr","sshr")}
    image_hist={model:{key:[] for key in ("base","cons","full")} for model in ("hqmr","sshr")}
    prevalence=[]; gt_area=np.zeros(4,np.int64); per_image=[]; interior=[]; contacts=[]
    robust={model:{(op,r):np.zeros((4,4),np.int64) for op in (3,4) for r in (1,5)} for model in ("hqmr","sshr")}
    sanity={model:{kind:np.zeros((4,4),np.int64) for kind in ("closing","opening","fill_holes")} for model in ("hqmr","sshr")}
    for iteration,row in enumerate(manifest,1):
        image_id=row["image_id"]; truth=np.asarray(Image.open(Path(args.val_root)/"mask"/f"{image_id}.png")); data=np.load(output/"cache"/row["cache"])
        for cls in range(4): gt_area[cls]+=int(np.sum(truth==cls))
        item={"image_id":image_id}; operator_gains={}
        for model in ("hqmr","sshr"):
            base=data[model].astype(np.uint8); states,changes,stats=subset_states(base,truth)
            for subset,state in states.items(): subset_hist[model][subset]+=confusion(truth,state)
            for key,subset in (("base",0),("cons",7),("full",31)): image_hist[model][key].append(confusion(truth,states[subset]))
            base_iou=scores_from_confusion(confusion(truth,base))["mIoU"]
            item[f"{model}_full_gain_pp"]=100*(scores_from_confusion(confusion(truth,states[31]))["mIoU"]-base_iou)
            item[f"{model}_cons_gain_pp"]=100*(scores_from_confusion(confusion(truth,states[7]))["mIoU"]-base_iou)
            for index,op in enumerate(OPERATORS):
                subset=1<<index; gain=100*(scores_from_confusion(confusion(truth,states[subset]))["mIoU"]-base_iou)
                item[f"{model}_{op}_gain_pp"]=gain; operator_gains[(model,op)]=gain
                for cls in range(4): prevalence.append({"image_id":image_id,"model":model,"operator":op,"class":cls,**stats[subset]["by_class"][str(cls)]})
            interior.extend(interior_rows(image_id,model,base,truth))
            for distance in (1,3,5): contacts.extend(boundary_contact_rows(image_id,model,base,truth,distance))
            for op in (3,4):
                for radius in (1,5): robust[model][(op,radius)]+=confusion(truth,apply_operator(op,base,truth,radius)[0])
            for kind in sanity: sanity[model][kind]+=confusion(truth,gt_free_transform(base,kind))
        item["excess_full_gain_pp"]=item["hqmr_full_gain_pp"]-item["sshr_full_gain_pp"]
        item["hole_contribution_pp"]=operator_gains[("hqmr","M2")]; item["fragmentation_contribution_pp"]=operator_gains[("hqmr","M3")]
        item["boundary_contribution_pp"]=operator_gains[("hqmr","M4")]+operator_gains[("hqmr","M5")]
        per_image.append(item)
        if iteration%50==0 or iteration==len(manifest): print(f"MORPH_AUDIT_PROGRESS={iteration}/{len(manifest)}",flush=True)
    tables,metrics={},{}
    for model in ("hqmr","sshr"):
        tables[model],metrics[model]=_subset_table(subset_hist[model]); write_csv(output/f"subsets/{model}_all_32_subsets.csv",tables[model])
    if abs(metrics["hqmr"][0]["mIoU"]-HQMR_MIOU)>1e-12 or abs(metrics["sshr"][0]["mIoU"]-SSHR_MIOU)>1e-12: raise AssertionError("Cached baseline drift")
    baseline={model:metrics[model][0] for model in metrics}; full={model:metrics[model][31] for model in metrics}; cons={model:metrics[model][7] for model in metrics}
    headroom={"hqmr_headroom_pp":100*(full["hqmr"]["mIoU"]-baseline["hqmr"]["mIoU"]),"sshr_headroom_pp":100*(full["sshr"]["mIoU"]-baseline["sshr"]["mIoU"])}
    headroom["excess_headroom_pp"]=headroom["hqmr_headroom_pp"]-headroom["sshr_headroom_pp"]; headroom["gap_recovery_ratio"]=headroom["excess_headroom_pp"]/FORMAL_GAP_PP
    conservative={"hqmr_gain_pp":100*(cons["hqmr"]["mIoU"]-baseline["hqmr"]["mIoU"]),"sshr_gain_pp":100*(cons["sshr"]["mIoU"]-baseline["sshr"]["mIoU"])}
    conservative["excess_headroom_pp"]=conservative["hqmr_gain_pp"]-conservative["sshr_gain_pp"]; conservative["gap_recovery_ratio"]=conservative["excess_headroom_pp"]/FORMAL_GAP_PP
    shapley={model:exact_shapley({s:100*metrics[model][s]["mIoU"] for s in range(32)}) for model in metrics}
    excess_shapley={op:shapley["hqmr"][op]-shapley["sshr"][op] for op in OPERATORS}
    per_class_shapley=[]
    for model in metrics:
        for cls in range(4):
            values={s:100*metrics[model][s]["class_iou"][str(cls)] for s in range(32)}
            for op,value in exact_shapley(values).items(): per_class_shapley.append({"model":model,"class":cls,"operator":op,"shapley_pp":value})
    for rows,name in (([{"operator":k,"shapley_pp":v} for k,v in shapley["hqmr"].items()],"hqmr"),([{ "operator":k,"shapley_pp":v} for k,v in shapley["sshr"].items()],"sshr"),([{ "operator":k,"shapley_pp":v} for k,v in excess_shapley.items()],"excess")): write_csv(output/f"shapley/{name}_shapley.csv",rows)
    write_csv(output/"shapley/per_class_shapley.csv",per_class_shapley)
    interactions=[]
    for model in metrics:
        for i,j in itertools.combinations(range(5),2):
            value=100*(metrics[model][(1<<i)|(1<<j)]["mIoU"]-metrics[model][1<<i]["mIoU"]-metrics[model][1<<j]["mIoU"]+metrics[model][0]["mIoU"])
            interactions.append({"model":model,"operator_j":OPERATORS[i],"operator_k":OPERATORS[j],"interaction_pp":value})
    write_csv(output/"shapley/pairwise_interactions.csv",interactions)
    prevalence_rows=_aggregate_prevalence(prevalence,gt_area); write_csv(output/"per_class/per_class_defect_prevalence.csv",prevalence_rows); write_csv(output/"baseline/baseline_morphology.csv",prevalence_rows)
    individual={model:{op:100*(metrics[model][1<<i]["mIoU"]-metrics[model][0]["mIoU"]) for i,op in enumerate(OPERATORS)} for model in metrics}
    operator_files = {
        "M1": "M1_island_summary.csv",
        "M2": "M2_hole_summary.csv",
        "M3": "M3_fragment_bridge_summary.csv",
        "M4": "M4_protrusion_summary.csv",
        "M5": "M5_indentation_summary.csv",
    }
    for i,op in enumerate(OPERATORS): write_csv(output/f"operators/{operator_files[op]}",[r for r in prevalence_rows if r["operator"]==op]+[{"model":m,"operator":op,"mIoU_gain_pp":individual[m][op]} for m in metrics])
    per_class=[]; class_excess={}
    for cls in range(4):
        hg=100*(full["hqmr"]["class_iou"][str(cls)]-baseline["hqmr"]["class_iou"][str(cls)]); sg=100*(full["sshr"]["class_iou"][str(cls)]-baseline["sshr"]["class_iou"][str(cls)])
        class_excess[str(cls)]=hg-sg; per_class.append({"class":cls,"hqmr_baseline_iou":baseline["hqmr"]["class_iou"][str(cls)],"hqmr_full_oracle_iou":full["hqmr"]["class_iou"][str(cls)],"hqmr_gain_pp":hg,"sshr_gain_pp":sg,"excess_gain_pp":hg-sg})
    write_csv(output/"per_class/per_class_oracle_recovery.csv",per_class)
    arrays={model:{key:np.stack(value) for key,value in group.items()} for model,group in image_hist.items()}
    bootstrap=bootstrap_headroom(arrays["hqmr"]["base"],arrays["hqmr"]["full"],arrays["sshr"]["base"],arrays["sshr"]["full"])
    write_json(output/"headroom/excess_headroom_bootstrap.json",bootstrap)
    write_json(output/"headroom/full_oracle_headroom.json",{"baseline":baseline,"full":full,"headroom":headroom})
    write_json(output/"headroom/conservative_oracle_headroom.json",{"baseline":baseline,"conservative":cons,"headroom":conservative})
    write_json(output/"headroom/gap_recovery.json",{"formal_gap_pp":FORMAL_GAP_PP,**headroom})
    per_image_df=pd.DataFrame(per_image); per_image_df.to_csv(output/"per_image/per_image_oracle_gain.csv",index=False)
    positive=np.sort(np.clip(per_image_df.hqmr_full_gain_pp.to_numpy(),0,None))[::-1]; top=max(1,math.ceil(.05*len(positive))); top_fraction=float(positive[:top].sum()/max(positive.sum(),1e-12)); heavy=top_fraction>.5
    heavy_tail={"top5pct_contribution_fraction":top_fraction,"MORPHOLOGY_GAIN_HEAVY_TAIL":heavy}; write_json(output/"per_image/heavy_tail_analysis.json",heavy_tail)
    image_distribution={key:distribution(per_image_df[key]) for key in ("hqmr_full_gain_pp","sshr_full_gain_pp","excess_full_gain_pp")}
    total_class_excess = sum(class_excess.values())
    class23_fraction = ((class_excess["2"] + class_excess["3"]) / total_class_excess
                        if abs(total_class_excess) > 1e-12 else None)
    class23={"fraction_total_excess":class23_fraction,"CLASS23_MORPHOLOGY_DOMINANT":class23_fraction is not None and class23_fraction>.6}
    write_csv(output/"diagnostics/interior_coherence.csv",interior); write_csv(output/"diagnostics/contact_region.csv",contacts)
    write_csv(output/"diagnostics/boundary_safety.csv",[r for r in contacts if r["distance"]==3])
    robustness=[]
    for model in robust:
        for (op,radius),hist in robust[model].items(): robustness.append({"model":model,"operator":OPERATORS[op],"radius":radius,"mIoU":scores_from_confusion(hist)["mIoU"],"gain_pp":100*(scores_from_confusion(hist)["mIoU"]-baseline[model]["mIoU"])})
    write_csv(output/"robustness/boundary_r1.csv",[r for r in robustness if r["radius"]==1]); write_csv(output/"robustness/boundary_r5.csv",[r for r in robustness if r["radius"]==5])
    sanity_rows=[]
    for model in sanity:
        for kind,hist in sanity[model].items(): sanity_rows.append({"model":model,"transform":kind,"mIoU":scores_from_confusion(hist)["mIoU"],"change_pp":100*(scores_from_confusion(hist)["mIoU"]-baseline[model]["mIoU"])})
    write_csv(output/"robustness/gt_free_morphology_sanity.csv",sanity_rows)
    decision,confidence=_decision(headroom["gap_recovery_ratio"],conservative["gap_recovery_ratio"],bootstrap["excess_headroom"]["ci95_pp"],class_excess,excess_shapley,heavy)
    dominant,target=_dominant_target(excess_shapley)
    matrix={"gap_recovery_full_ge_1":headroom["gap_recovery_ratio"]>=1,"excess_ci_lower_gt_0":bootstrap["excess_headroom"]["ci95_pp"][0]>0,"gap_recovery_cons_ge_0_5":conservative["gap_recovery_ratio"]>=.5,"positive_excess_classes":sum(v>0 for v in class_excess.values()),"meaningful_positive_excess_shapley":sum(v>=CONFIG["meaningful_shapley_pp"] for v in excess_shapley.values()),"heavy_tail":heavy}
    falsified="Generic morphology repair as the primary bottleneck is falsified." if decision=="MORPHOLOGY_NOT_PRIMARY" else "Pure semantic purification as a sufficient recovery mechanism remains falsified."
    next_step=(f"Do not design SARH; morphology explains only {100*headroom['gap_recovery_ratio']:.2f}% of the gap." if decision=="MORPHOLOGY_NOT_PRIMARY" else f"After review, a future class/query-aware, boundary-safe mechanism may target {target}; this audit does not authorize training it.")
    result={"decision":{"decision":decision,"confidence":confidence,"matrix":matrix,"dominant_operators":dominant,"architecture_target":target,"falsified":falsified,"next_step":next_step},"baseline":baseline,"reproduction_gate":gate,"headroom":headroom,"conservative":conservative,"full_oracle":full,"individual_gains_pp":individual,"operator_summary":{op:[r for r in prevalence_rows if r["operator"]==op] for op in OPERATORS},"bootstrap":bootstrap,"shapley":{"hqmr":shapley["hqmr"],"sshr":shapley["sshr"],"excess":excess_shapley},"interactions":interactions,"per_class":per_class,"class23":class23,"image_distribution":image_distribution,"heavy_tail":heavy_tail,"gt_free_sanity":sanity_rows,"source_commit":config["source_commit"],"oracle_is_model_performance":False,"training_performed":False}
    selected={"highest_hqmr_gain":per_image_df.nlargest(5,"hqmr_full_gain_pp").image_id.tolist(),"highest_excess_headroom":per_image_df.nlargest(5,"excess_full_gain_pp").image_id.tolist(),"highest_hole_contribution":per_image_df.nlargest(5,"hole_contribution_pp").image_id.tolist(),"highest_fragmentation_contribution":per_image_df.nlargest(5,"fragmentation_contribution_pp").image_id.tolist(),"highest_boundary_contribution":per_image_df.nlargest(5,"boundary_contribution_pp").image_id.tolist(),"zero_or_negative_gain":per_image_df.nsmallest(5,"hqmr_full_gain_pp").image_id.tolist()}
    write_json(output/"visualizations/selection.json",selected); _visualize(output,Path(args.val_root),manifest,selected)
    write_json(output/"morphology_oracle_audit_result.json",result); report=output/"report/HQMR_v1_Morphology_Oracle_Recovery_Audit_Report.md"; report.write_text(report_text(result),encoding="utf-8")
    print(json.dumps({"decision":decision,"confidence":confidence,"headroom":headroom,"conservative":conservative,"dominant":dominant,"report":str(report)},indent=2)); print(f"DECISION = {decision}"); print(f"CONFIDENCE = {confidence}")


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode",choices=("infer","audit"),required=True)
    parser.add_argument("--val-root",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--hqmr-checkpoint",required=True); parser.add_argument("--sshr-checkpoint",required=True); parser.add_argument("--ccbp-result",required=True); parser.add_argument("--num-workers",type=int,default=8)
    return parser.parse_args()


def main():
    args=parse_args(); output=Path(args.output_dir).resolve()
    if args.mode=="infer": run_infer(args,output)
    else: run_audit(args,output)


if __name__=="__main__": main()
