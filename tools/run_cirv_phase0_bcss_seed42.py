#!/usr/bin/env python3
"""Build the frozen CIRV prototype bank, then run the registered Phase-0 audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.cirv import calibrate_evidence, select_source_regions, spherical_kmeans
from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    TTA, foreground_confusion, load_state, normalize_cam, paired_bootstrap,
    prediction_from_cam, presence, resize_unflip, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
MORPH_RESULT_SHA256 = "8f4e7b9a3b17f5eabdf229495f239217fc130a59814e06d09f797f83dc165579"
M1_RESULT_SHA256 = "41d5c0fc7f6e045846f872c4bc4d9419d5b9357baba24e70b146aa32711834ca"
EXPECTED_HQMR = 0.6557244403737567
EXPECTED_IMAGES = 3418
BOOTSTRAP_SEED = 20260914
BOOTSTRAP_RESAMPLES = 10_000
CONFIG = {
    "experiment": "CCRA + HQMR-v1 + CIRV Phase0", "dataset": "BCSS", "seed": 42,
    "classes": 4, "prototypes_per_class": 4, "embedding": "Stage3 HQMR K4 stopgrad",
    "region_connectivity": 8, "train_source_per_class_image": 1,
    "source_selection": "present class; predicted class; >=1 reliable-positive; max anchor_count",
    "train_source_augmentation": "none (deterministic frozen inference pass)",
    "prototype_initialization": "deterministic spherical kmeans seed42",
    "cosine_scale": 5.0, "ema": 0.99, "ratio_clip": [0.25, 4.0],
    "fusion": "equal product of p_base_R and p_proto_R", "trainable_parameters": 0,
    "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "validation_firewall": "validation accessed only after static bank frozen and hashed",
}


def setup(output: Path) -> None:
    for name in ("provenance", "phase0", "tests", "prototype_bank", "logs", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def parse_label(image_id: str) -> np.ndarray:
    try:
        label = image_id.split("]", 1)[0].split("[")[-1]
        value = np.asarray([int(label[index]) for index in range(4)], np.float32)
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse BCSS image label: {image_id}") from exc
    if not np.isin(value, [0, 1]).all():
        raise ValueError(f"Invalid BCSS image label: {image_id}")
    return value


def provenance(args, output: Path) -> None:
    m1_result = Path(args.m1_output) / "false_component_audit_result.json"
    if not m1_result.exists():
        candidates = list(Path(args.m1_output).glob("*result*.json"))
        if len(candidates) == 1:
            m1_result = candidates[0]
    if sha256(m1_result) != M1_RESULT_SHA256:
        raise AssertionError("False-Component Audit archive mismatch")
    config = {**CONFIG, "source_commit": git_commit(), "trainroot": str(Path(args.trainroot).resolve()),
              "val_root": str(Path(args.val_root).resolve()),
              "hqmr_checkpoint": str(Path(args.hqmr_checkpoint).resolve()),
              "m1_audit_output": str(Path(args.m1_output).resolve())}
    write_json(output / "provenance/cirv_config.json", config)
    (output / "provenance/cirv_config_sha256.txt").write_text(
        sha256(output / "provenance/cirv_config.json") + "\n")
    (output / "provenance/cirv_source_commit.txt").write_text(git_commit() + "\n")
    (output / "provenance/cirv_git_diff.patch").write_text(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT, text=True), encoding="utf-8")
    write_json(output / "provenance/false_component_audit_archive.json", {
        "path": str(m1_result.resolve()), "sha256": sha256(m1_result),
        "decision": json.loads(m1_result.read_text())["decision"]})


def load_hqmr(path: Path) -> HQMRNet:
    if sha256(path) != HQMR_SHA256:
        raise AssertionError("Frozen HQMR checkpoint mismatch")
    model = HQMRNet().cuda()
    model.load_state_dict(load_state(path), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def build_bank(args, output: Path) -> None:
    if (output / "phase0/static_prototype_bank.npy").exists():
        raise FileExistsError("Static bank already exists; refusing overwrite")
    setup(output); provenance(args, output)
    model = load_hqmr(Path(args.hqmr_checkpoint))
    dataset = Stage1_InferDataset(args.trainroot, img_size=224)
    if len(dataset) != 23422:
        raise AssertionError(f"Expected 23422 BCSS training images, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    sources: list[dict] = []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for batch_index, (names, images) in enumerate(loader, 1):
        labels_np = np.stack([parse_label(name) for name in names])
        labels = torch.from_numpy(labels_np).cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(images, labels, step=29275, hqmr_mode="full")
        item = result["stages"][2]["hqmr"]
        predictions = item["mixture"].detach().argmax(1).cpu().numpy()
        anchors = result["target_detail"]["positive"].detach().float()
        if anchors.shape[-2:] != item["mixture"].shape[-2:]:
            anchors = F.interpolate(anchors, size=item["mixture"].shape[-2:], mode="nearest")
        anchors_np = anchors.bool().cpu().numpy()
        key4 = item["key4"].detach()
        for sample, name in enumerate(names):
            rows = select_source_regions(predictions[sample], labels_np[sample], anchors_np[sample], key4[sample])
            for row in rows:
                embedding = row.pop("embedding")
                sources.append({"image_id": name, **row,
                                **{f"z_{index:03d}": float(value) for index, value in enumerate(embedding)}})
        if batch_index % 100 == 0:
            print(json.dumps({"event": "train_source_progress", "batches": batch_index,
                              "images": min(batch_index * args.batch_size, len(dataset)),
                              "sources": len(sources), "elapsed_s": time.perf_counter() - started}), flush=True)
    frame = pd.DataFrame(sources)
    if frame.empty:
        raise RuntimeError("No CIRV source regions selected")
    source_path = output / "phase0/train_source_regions.parquet"
    frame.to_parquet(source_path, index=False, compression="zstd")
    z_columns = [f"z_{index:03d}" for index in range(256)]
    bank, reports = [], []
    for cls in range(4):
        values = frame.loc[frame.class_id == cls, z_columns].to_numpy(np.float32)
        centers, assignment, report = spherical_kmeans(values, k=4, seed=42)
        bank.append(centers); report.update({"class_id": cls})
        reports.append(report)
        frame.loc[frame.class_id == cls, "prototype_assignment"] = assignment
    bank = np.stack(bank).astype(np.float32)
    bank_path = output / "phase0/static_prototype_bank.npy"
    np.save(bank_path, bank, allow_pickle=False)
    digest = sha256(bank_path)
    (output / "phase0/static_prototype_bank_sha256.txt").write_text(digest + "\n")
    np.save(output / "prototype_bank/epoch05_bank.npy", bank, allow_pickle=False)
    write_json(output / "prototype_bank/init_kmeans_report.json", {
        "classes": reports, "all_normalized": bool(np.allclose(np.linalg.norm(bank, axis=2), 1, atol=1e-5)),
        "source_count": int(len(frame)), "source_count_per_class": {
            str(cls): int((frame.class_id == cls).sum()) for cls in range(4)}})
    marker = {"status": "FROZEN", "bank_sha256": digest, "source_sha256": sha256(source_path),
              "checkpoint_sha256": sha256(Path(args.hqmr_checkpoint)), "validation_accessed": False,
              "source_count": int(len(frame)), "seconds": time.perf_counter() - started,
              "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
    write_json(output / "phase0/bank_frozen_before_validation.json", marker)
    print(json.dumps(marker, indent=2), flush=True)


@torch.no_grad()
def infer_hqmr_cirv_inputs(model: HQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    full, gates, keys = [], [], []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
        item = output["stages"][2]["hqmr"]
        full.append(resize_unflip(item["mixture"], original_hw, cam_flip).float().cpu())
        key = item["key4"][0]
        if cam_flip:
            key = torch.flip(key, dims=cam_flip)
        keys.append(key.float().cpu())
        gates.append(output["deep_gate"].float().cpu())
    scores = normalize_cam(torch.stack(full).mean(0).numpy())
    g = torch.stack(gates).mean(0).numpy()[0].astype(np.float32)
    label = presence(g)
    return {"scores": scores, "g": g, "label": label,
            "prediction": prediction_from_cam(scores, label, np.empty(original_hw)),
            "key4": torch.stack(keys).mean(0)}


def phase0_decision(delta_pp: float, bootstrap: dict, corrected: float,
                    valid_harm: float, switch_precision: float) -> tuple[str, dict]:
    strong = {"delta_ge_0_60": delta_pp >= .60, "ci_lower_gt_0": bootstrap["miou_ci95_pp"][0] > 0,
              "m1_corrected_ge_0_50": corrected >= .50, "valid_harm_le_0_03": valid_harm <= .03,
              "switch_precision_ge_0_70": switch_precision >= .70}
    go = {"delta_ge_0_30": delta_pp >= .30, "paired_mean_gt_0": bootstrap["miou_delta_pp_mean"] > 0,
          "m1_corrected_ge_0_40": corrected >= .40, "valid_harm_le_0_05": valid_harm <= .05,
          "switch_precision_ge_0_60": switch_precision >= .60}
    nogo = {"delta_le_0": delta_pp <= 0, "m1_corrected_lt_0_25": corrected < .25,
            "valid_harm_gt_0_08": valid_harm > .08, "switch_precision_lt_0_50": switch_precision < .50}
    if all(strong.values()): decision = "CIRV_PHASE0_STRONG_GO"
    elif all(go.values()): decision = "CIRV_PHASE0_GO"
    elif any(nogo.values()): decision = "CIRV_PHASE0_NOGO"
    else: decision = "CIRV_PHASE0_UNCERTAIN"
    return decision, {"strong": strong, "go": go, "nogo": nogo}


def report_text(result: dict) -> str:
    base, cirv = result["base"], result["cirv"]
    mechanism = result["mechanism"]
    return f"""# CCRA + HQMR-v1 + CIRV Phase 0 实验报告

## 结论

**{result['decision']}**

静态 CIRV 相对冻结 HQMR-v1 的 mIoU 变化为 **{result['delta_miou_pp']:+.4f} pp**，配对 bootstrap 95% CI 为 **[{result['bootstrap']['miou_ci95_pp'][0]:+.4f}, {result['bootstrap']['miou_ci95_pp'][1]:+.4f}] pp**。本报告严格依据预注册门限；Phase 0 不是正式 Full25 结果。

## 冻结边界与复现

- HQMR checkpoint SHA256：`{result['checkpoint_sha256']}`
- static prototype bank SHA256：`{result['bank_sha256']}`
- HQMR 基线复现：{100*base['mIoU']:.4f}% mIoU（目标 65.5724%）
- 验证集：{result['images']} 张 BCSS 图像，验证仅在 prototype bank 冻结并哈希后访问
- CIRV 新增可训练参数：0

## 主要结果

| 指标 | HQMR-v1 | HQMR-v1 + static CIRV | 变化 |
|---|---:|---:|---:|
| mIoU | {100*base['mIoU']:.4f}% | {100*cirv['mIoU']:.4f}% | {result['delta_miou_pp']:+.4f} pp |
| mDice | {100*base['mDice']:.4f}% | {100*cirv['mDice']:.4f}% | {100*(cirv['mDice']-base['mDice']):+.4f} pp |

## M1 与安全性

- M1 false-area corrected fraction：{100*mechanism['m1_false_area_corrected_fraction']:.3f}%
- valid-area harmed：{100*mechanism['valid_area_harmed']:.3f}%
- switch precision：{100*mechanism['switch_precision']:.3f}%
- switch recall on M1 false area：{100*mechanism['switch_recall_on_m1_false_area']:.3f}%
- prototype-only false-component correct-class rate：{100*mechanism['prototype_only_correct_class_rate']:.3f}%
- fused false-component correct-class rate：{100*mechanism['fused_correct_class_rate']:.3f}%
- ratio clip activation：{100*mechanism['ratio_clip_activation_fraction']:.3f}%{('（CIRV_RATIO_SATURATION）' if mechanism['ratio_saturation'] else '')}

## 门控解释

完整精确门控矩阵见 `phase0_metrics.json`。若结论为 NOGO 或 UNCERTAIN，本轮按方案停止，不启动 Full25；只有 GO/STRONG_GO 才允许 fresh Full25。

CCRA determines class responsibility, HQMR reconstructs spatial tissue regions, and CIRV verifies each reconstructed region against a cross-image multi-prototype tissue memory so that semantically plausible but globally inconsistent false regions can be reclassified rather than blindly suppressed.
"""


@torch.no_grad()
def validate(args, output: Path) -> str:
    marker_path = output / "phase0/bank_frozen_before_validation.json"
    bank_path = output / "phase0/static_prototype_bank.npy"
    if not marker_path.exists() or not bank_path.exists():
        raise RuntimeError("Validation firewall: static bank is not frozen")
    marker = json.loads(marker_path.read_text())
    if marker["status"] != "FROZEN" or marker["validation_accessed"]:
        raise RuntimeError("Validation firewall marker invalid")
    if marker["bank_sha256"] != sha256(bank_path):
        raise AssertionError("Static bank hash mismatch")
    valroot = Path(args.val_root)
    if len(list((valroot / "img").glob("*.png"))) != EXPECTED_IMAGES:
        raise AssertionError("Expected 3418 BCSS validation images")
    morphology = Path(args.morphology_output)
    morph_result = morphology / "morphology_oracle_audit_result.json"
    if sha256(morph_result) != MORPH_RESULT_SHA256:
        raise AssertionError("Morphology Oracle archive mismatch")
    manifest = json.loads((morphology / "cache/prediction_manifest.json").read_text())
    cached = {row["image_id"]: row["cache"] for row in manifest}
    bank = np.load(bank_path, allow_pickle=False)
    model = load_hqmr(Path(args.hqmr_checkpoint))
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    base_hist, cirv_hist, per_image, components = [], [], [], []
    totals = {"m1_area": 0, "m1_corrected_area": 0, "valid_correct_area": 0,
              "valid_harmed_area": 0, "switched_area": 0, "switched_correct_area": 0,
              "false_count": 0, "proto_correct_count": 0, "fused_correct_count": 0,
              "ratios": 0, "clipped": 0}
    started = time.perf_counter(); cache_mismatch = 0
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]
        original = Image.open(valroot / "img" / f"{image_id}.png")
        truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png"))
        item = infer_hqmr_cirv_inputs(model, image.cuda(non_blocking=True), (original.height, original.width))
        frozen = np.load(morphology / "cache" / cached[image_id])
        cache_mismatch += int(not np.array_equal(item["prediction"], frozen["hqmr"]))
        calibrated, records = calibrate_evidence(
            item["scores"], item["prediction"], item["key4"], bank, item["g"], mode="full")
        cirv_prediction = np.argmax(calibrated, axis=0).astype(np.int64)
        bh = foreground_confusion(truth, item["prediction"]); ch = foreground_confusion(truth, cirv_prediction)
        base_hist.append(bh); cirv_hist.append(ch)
        per_image.append({"image_id": image_id, "base_mIoU": scores_from_confusion(bh)["mIoU"],
                          "cirv_mIoU": scores_from_confusion(ch)["mIoU"]})
        for record in records:
            mask = record.pop("mask"); valid_truth = truth[mask]; valid_truth = valid_truth[valid_truth < 4]
            if len(valid_truth) == 0:
                continue
            cls = record["class_id"]; majority = int(np.bincount(valid_truth, minlength=4).argmax())
            is_false = int(np.sum(mask & (truth == cls)) == 0)
            proto_correct = int(record["proto_class"] == majority)
            fused_correct = int(record["fused_class"] == majority)
            switched = int(record["fused_class"] != cls)
            area = int(mask.sum())
            base_correct = mask & (truth == cls)
            harmed = base_correct & (cirv_prediction != truth)
            if not is_false:
                totals["valid_correct_area"] += int(base_correct.sum())
                totals["valid_harmed_area"] += int(harmed.sum())
            if is_false:
                totals["false_count"] += 1; totals["m1_area"] += area
                totals["proto_correct_count"] += proto_correct
                totals["fused_correct_count"] += fused_correct
                totals["m1_corrected_area"] += area * fused_correct
            if switched:
                totals["switched_area"] += area
                totals["switched_correct_area"] += area * fused_correct
            raw = record["ratio_raw"]
            totals["ratios"] += int(raw.size)
            totals["clipped"] += int(np.sum((raw < .25) | (raw > 4.0)))
            components.append({"image_id": image_id, "class_id": cls,
                               "component_id": record["component_id"], "area": area,
                               "gt_majority_class": majority, "is_m1_false": is_false,
                               "proto_class": record["proto_class"], "fused_class": record["fused_class"],
                               "proto_correct": proto_correct, "fused_correct": fused_correct,
                               "switched": switched,
                               "p_base": json.dumps(record["p_base"].tolist()),
                               "p_proto": json.dumps(record["p_proto"].tolist()),
                               "p_fuse": json.dumps(record["p_fuse"].tolist()),
                               "ratio": json.dumps(record["ratio"].tolist())})
        if index % 100 == 0:
            print(json.dumps({"event": "phase0_val_progress", "images": index,
                              "elapsed_s": time.perf_counter() - started,
                              "cache_mismatch": cache_mismatch}), flush=True)
    base_hist = np.stack(base_hist); cirv_hist = np.stack(cirv_hist)
    base = scores_from_confusion(base_hist.sum(0)); cirv = scores_from_confusion(cirv_hist.sum(0))
    reproduction = abs(base["mIoU"] - EXPECTED_HQMR) <= 1e-10 and cache_mismatch == 0
    bootstrap = paired_bootstrap(base_hist, cirv_hist, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    mechanism = {
        "m1_false_area_corrected_fraction": totals["m1_corrected_area"] / max(totals["m1_area"], 1),
        "valid_area_harmed": totals["valid_harmed_area"] / max(totals["valid_correct_area"], 1),
        "switch_precision": totals["switched_correct_area"] / max(totals["switched_area"], 1),
        "switch_recall_on_m1_false_area": totals["m1_corrected_area"] / max(totals["m1_area"], 1),
        "prototype_only_correct_class_rate": totals["proto_correct_count"] / max(totals["false_count"], 1),
        "fused_correct_class_rate": totals["fused_correct_count"] / max(totals["false_count"], 1),
        "ratio_clip_activation_fraction": totals["clipped"] / max(totals["ratios"], 1),
        "ratio_saturation": totals["clipped"] / max(totals["ratios"], 1) > .20,
        **totals,
    }
    delta = 100 * (cirv["mIoU"] - base["mIoU"])
    decision, gates = phase0_decision(delta, bootstrap,
        mechanism["m1_false_area_corrected_fraction"], mechanism["valid_area_harmed"],
        mechanism["switch_precision"])
    if not reproduction:
        decision = "CIRV_PHASE0_ENGINEERING_BLOCKED"
    result = {"decision": decision, "delta_miou_pp": delta, "base": base, "cirv": cirv,
              "bootstrap": bootstrap, "mechanism": mechanism, "gates": gates,
              "base_reproduction": {"pass": reproduction, "expected": EXPECTED_HQMR,
                                    "actual": base["mIoU"], "cache_prediction_mismatch_images": cache_mismatch},
              "images": len(base_hist), "checkpoint_sha256": sha256(Path(args.hqmr_checkpoint)),
              "bank_sha256": sha256(bank_path), "seconds": time.perf_counter() - started,
              "source_commit": git_commit()}
    write_json(output / "phase0/phase0_metrics.json", result)
    write_json(output / "phase0/phase0_bootstrap.json", bootstrap)
    write_csv(output / "phase0/phase0_per_component.csv", components)
    write_csv(output / "phase0/phase0_per_image.csv", per_image)
    report = output / "phase0/CIRV_Phase0_Report.md"
    report.write_text(report_text(result), encoding="utf-8")
    write_json(output / "phase0/validation_firewall_closed.json", {
        "bank_frozen_before_validation": True, "bank_sha256": sha256(bank_path),
        "validation_images": len(base_hist), "decision": decision})
    marker["validation_accessed"] = True; marker["validation_completed"] = True
    write_json(marker_path, marker)
    print(json.dumps({"decision": decision, "delta_miou_pp": delta,
                      "base_miou": base["mIoU"], "cirv_miou": cirv["mIoU"],
                      "bootstrap_ci": bootstrap["miou_ci95_pp"], "mechanism": mechanism}, indent=2), flush=True)
    print(decision, flush=True)
    return decision


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("build-bank", "validate"))
    parser.add_argument("--trainroot", required=True); parser.add_argument("--val-root", required=True)
    parser.add_argument("--hqmr-checkpoint", required=True); parser.add_argument("--morphology-output", required=True)
    parser.add_argument("--m1-output", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); output = Path(args.output_dir).resolve()
    if args.mode == "build-bank": build_bank(args, output)
    else: validate(args, output)


if __name__ == "__main__":
    main()
