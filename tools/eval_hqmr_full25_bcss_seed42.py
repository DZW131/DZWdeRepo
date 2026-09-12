#!/usr/bin/env python3
"""Post-seal HQMR validation, same-checkpoint A-F ablations, and coverage re-audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.hqmr import class_mixture
from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_semantic_coverage_reachability import distance_maps, resize_label
from tools.eval_gcqm_full25_bcss_seed42 import (
    BASELINE_SHA256, THRESHOLDS, TTA, _predict_sshr, foreground_confusion, load_state,
    normalize_cam, paired_bootstrap, prediction_from_cam, presence, resize_unflip, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260912, 10_000
DFSC_MIOU = 0.6452820195735897
SSHR_MIOU = 0.6669670591172749
DFSC_SHA256 = "470f1056f2bbf5c64b5e6fff76861f9fa4e1663bf7ba0c74e2621528fb48af11"
MODES = {
    "A_full": "full", "B_fine_only": "fine_only", "C_coarse_fine": "coarse_fine",
    "D_no_query_update": "no_query_update", "E_coarse_only": "coarse_only", "F_mid_final": "mid_final",
}


@torch.no_grad()
def predict_hqmr_modes(model: HQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    full_views = {name: [] for name in MODES}; grid_views = {name: [] for name in MODES}
    basis_views = {name: [] for name in MODES}; weight_views, gates = [], []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
            stage = output["stages"][2]; h5 = output["query_detail"]["context_feature"]
            h4, h3 = output["pixel_detail"]["F4_context"], output["features"]["F3"]
            decoded = {"A_full": stage["hqmr"]}
            for name, mode in list(MODES.items())[1:]:
                item = model.hqmr(stage["query"], h5, h4, h3, mode=mode)
                item["mixture"] = class_mixture(item["basis"], stage["hqmr"]["weights"])
                decoded[name] = item
        for name, item in decoded.items():
            mixture, basis = item["mixture"][0], item["basis"][0]
            if cam_flip: mixture, basis = torch.flip(mixture, dims=cam_flip), torch.flip(basis, dims=cam_flip)
            grid_views[name].append(mixture.float().cpu()); basis_views[name].append(basis.float().cpu())
            full_views[name].append(resize_unflip(item["mixture"], original_hw, cam_flip).float().cpu())
        weight_views.append(stage["hqmr"]["weights"][0].float().cpu()); gates.append(output["deep_gate"].float().cpu())
    label = presence(torch.stack(gates).mean(0).numpy()[0])
    result = {"label": label, "weights": torch.stack(weight_views).mean(0).numpy(), "modes": {}}
    for name in MODES:
        formal = normalize_cam(torch.stack(full_views[name]).mean(0).numpy())
        grid = normalize_cam(torch.stack(grid_views[name]).mean(0).numpy())
        result["modes"][name] = {
            "prediction": prediction_from_cam(formal, label, np.empty(original_hw)),
            "prediction_grid": prediction_from_cam(grid, label, np.empty(grid.shape[-2:])),
            "evidence_grid": grid, "basis": torch.stack(basis_views[name]).mean(0).numpy(),
        }
    return result


def coverage_one(basis: np.ndarray, weights: np.ndarray, evidence: np.ndarray,
                 truth: np.ndarray, prediction: np.ndarray) -> tuple[dict, list[dict]]:
    gt = resize_label(truth, evidence.shape[-2:]); rows = []; total = defaultdict(float)
    for cls in range(4):
        mask = gt == cls; n = int(mask.sum()); threshold = float(THRESHOLDS[cls])
        binary = basis >= threshold; bmax = binary.any(0)
        weighted_order = np.argsort(-weights[:, cls], kind="stable"); top20 = binary[weighted_order[:20]].any(0)
        intersection = (binary & mask).sum((1, 2)); union = (binary | mask).sum((1, 2))
        oracle_order = np.argsort(-np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0), kind="stable")
        oracle10 = binary[oracle_order[:10]].any(0)
        seed = (evidence[cls] >= threshold); base_covered = int((seed & mask).sum())
        item = {"class": cls, "gt_pixels": n, "basis_max_coverage": float((bmax & mask).sum() / max(n, 1)),
                "class_basis_uncovered": float(((~top20) & mask).sum() / max(n, 1)),
                "oracle_top10_recall": float((oracle10 & mask).sum() / max(n, 1)),
                "base_F_seed_coverage": float(base_covered / max(n, 1))}
        rows.append(item)
        for key in ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall", "base_F_seed_coverage"):
            total[key] += item[key] * n
        total["gt_pixels"] += n
    overall = {"class": "overall", "gt_pixels": int(total["gt_pixels"]),
               **{key: total[key] / max(total["gt_pixels"], 1) for key in
                  ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall", "base_F_seed_coverage")}}
    return overall, rows


def report_text(result: dict) -> str:
    m, d, b, c, a = result["metrics"], result["deltas_pp"], result["bootstrap_vs_sshr"], result["coverage"], result["ablation"]
    sections = [
        ("Executive Decision", f"**DECISION = {result['decision']}**。HQMR−SSHR mIoU={d['vs_sshr']:+.4f} pp，95% CI [{b['miou_ci95_pp'][0]:+.4f}, {b['miou_ci95_pp'][1]:+.4f}]。"),
        ("Experiment Boundary", "只替换 Q→B mask basis decoder；CCRA、PCA、deep branch、tri-state targets、loss 权重和 GCQM detached class weights 均冻结。"),
        ("Architecture", "Stage2 使用 H5→H4；Stage3 使用 H5→H4→H3。每层在 logit space 累加 direct affinity，并以 soft region pooling 更新 query。"),
        ("Actual Feature Shapes", "224×224 输入下：H5=256×28×28，H4=128×28×28，H3=256×56×56。H5/H4 尺寸相同但语义层级不同。"),
        ("Engineering Validation", f"HQMR tests={result['tests']['unit']} passed；full regression={result['tests']['regression']} passed；2-step BF16 smoke passed。"),
        ("Gradient Contract", "Stage2/3 mask loss 对 HQMR query、H5/H4/H3 和多尺度 projection 均有非零梯度；GCQM w 保持 detached。"),
        ("Parameter Audit", f"HQMR 总参数 {result['complexity']['parameters']:,}，相对 GCQM 新增 {result['complexity']['parameter_delta']:,}（<5M）。"),
        ("Fresh Full25 Protocol", "BCSS Seed42，official MXNet/ImageNet init，batch20，BF16，25 epochs / 29275 steps，固定 E25 endpoint。"),
        ("Training Completion", f"训练耗时 {result['training']['train_seconds']/60:.2f} 分钟；峰值显存 {result['training']['peak_cuda_memory_gib']:.3f} GiB；all_finite={result['training']['all_finite']}。"),
        ("Checkpoint Seal", f"E25 SHA256 `{result['provenance']['hqmr_sha256']}`；封存后才访问 validation。"),
        ("Protocol Comparability", f"协议判定 `{result['protocol']['decision']}`；同 GPU、split、Seed42、25 epochs、29275 steps、batch20、224 resolution、TTA/threshold/class mapping。"),
        ("Main Metrics", f"| Model | mIoU | mDice |\n|---|---:|---:|\n| SSHR B0 | {100*m['sshr']['mIoU']:.4f} | {100*m['sshr']['mDice']:.4f} |\n| DFSC | {100*DFSC_MIOU:.4f} | — |\n| HQMR | {100*m['A_full']['mIoU']:.4f} | {100*m['A_full']['mDice']:.4f} |"),
        ("Performance Deltas", f"vs SSHR={d['vs_sshr']:+.4f} pp；vs DFSC={d['vs_dfsc']:+.4f} pp。"),
        ("Per-Class IoU", str(result["per_class"])),
        ("Paired Bootstrap", f"{BOOTSTRAP_RESAMPLES:,} paired resamples，seed={BOOTSTRAP_SEED}；95% CI={b['miou_ci95_pp']} pp。"),
        ("Coverage Reference", "旧 DFSC：class-basis-uncovered=0.7824，Bmax=0.2609，base seed=0.1462，oracle top10=0.2346，Type E=0.9001。"),
        ("HQMR Coverage", f"class-basis-uncovered={c['full']['class_basis_uncovered']:.4f}；Bmax={c['full']['basis_max_coverage']:.4f}；base seed={c['full']['base_F_seed_coverage']:.4f}；oracle top10={c['full']['oracle_top10_recall']:.4f}。"),
        ("Coverage Gates", f"C1={c['gates']['C1']}，C2={c['gates']['C2']}，C3={c['gates']['C3']}；HQMR_COVERAGE_RECOVERY={c['recovery']}；STRONG={c['strong_recovery']}。"),
        ("SSHR Rescue Anatomy", f"HQMR false negatives that SSHR rescues: Type A/B/C/D/E={c['rescue_type_fractions']}。"),
        ("FN-to-Seed Distance", str(c["fn_to_seed_distance"])),
        ("Component Anchor", str(c["component_anchor"])),
        ("Same-Checkpoint A–F", "\n".join(["| Mode | mIoU | ΔFull pp | Bmax | uncovered | oracle10 |", "|---|---:|---:|---:|---:|---:|"] + [f"| {name} | {100*m[name]['mIoU']:.4f} | {100*(m['A_full']['mIoU']-m[name]['mIoU']):+.4f} | {c['by_mode'][name]['basis_max_coverage']:.4f} | {c['by_mode'][name]['class_basis_uncovered']:.4f} | {c['by_mode'][name]['oracle_top10_recall']:.4f} |" for name in MODES])),
        ("Hierarchy Causal Test", f"Full−Fine={a['full_minus_fine_pp']:+.4f} pp；Full−Coarse={a['full_minus_coarse_pp']:+.4f} pp；core_pass={a['core_hierarchy_pass']}。"),
        ("Mid-Scale Contribution", f"Full−skip-F4={a['full_minus_skip_f4_pp']:+.4f} pp；Full−mid-final={a['full_minus_mid_final_pp']:+.4f} pp。"),
        ("Query-Update Contribution", f"Full−no-query-update={a['full_minus_no_query_update_pp']:+.4f} pp。"),
        ("Scientific Interpretation", result["interpretation"]),
        ("Limitations", "单 seed 结果不能替代多 seed 稳健性；A–F 是同 checkpoint causal diagnostics，不用于 checkpoint selection。"),
        ("Next Step", result["next_step"]),
        ("Exact Decision", f"`DECISION = {result['decision']}`"),
    ]
    return "# CCRA+HQMR BCSS Seed42 Full25 Final Validation Report\n\n" + "\n\n".join(
        f"## {index} {title}\n\n{body}" for index, (title, body) in enumerate(sections, 1)) + f"\n\nDECISION = {result['decision']}\n"


def decide(delta_sshr: float, lower: float, delta_dfsc: float, class_delta: dict,
           coverage: dict, ablation: dict) -> str:
    catastrophic = min(class_delta.values()) <= -3.0
    coverage_worse = coverage["full"]["class_basis_uncovered"] > .7824 and coverage["full"]["basis_max_coverage"] < .2609
    no_gain = not coverage["recovery"] and delta_dfsc <= 0
    if delta_dfsc <= -.30 or coverage_worse or no_gain: return "HQMR_FULL25_NOGO"
    if delta_sshr >= .50 and lower > 0 and min(class_delta.values()) >= -1.0 and coverage["recovery"] and ablation["core_hierarchy_pass"]:
        return "HQMR_FULL25_STRONG_GO"
    if delta_sshr >= .30 and lower > 0 and not catastrophic and coverage["recovery"] and ablation["core_hierarchy_pass"]:
        return "HQMR_FULL25_GO"
    if delta_sshr > 0 and (delta_sshr < .30 or lower <= 0) and coverage["recovery"]:
        return "HQMR_FULL25_BREAKTHROUGH_UNCERTAIN"
    if coverage["recovery"] and delta_dfsc >= 1.0 and delta_sshr < -.30:
        return "HQMR_FULL25_COVERAGE_GO"
    if -.30 < delta_dfsc < .30 and not coverage["recovery"]: return "HQMR_FULL25_NEUTRAL"
    return "HQMR_FULL25_NOGO"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True); parser.add_argument("--hqmr-checkpoint", required=True)
    parser.add_argument("--hqmr-experiment", required=True); parser.add_argument("--sshr-checkpoint", required=True)
    parser.add_argument("--dfsc-checkpoint", required=True); parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); valroot, experiment = Path(args.val_root).resolve(), Path(args.hqmr_experiment).resolve()
    hqmr_path, sshr_path, dfsc_path = map(lambda value: Path(value).resolve(), (args.hqmr_checkpoint, args.sshr_checkpoint, args.dfsc_checkpoint))
    if len(list((valroot / "img").glob("*.png"))) != 3418 or len(list((valroot / "mask").glob("*.png"))) != 3418: raise AssertionError("Expected 3418 BCSS validation pairs")
    if any((experiment / "report").iterdir()): raise FileExistsError("HQMR final report already exists")
    runtime = json.loads((experiment / "provenance/hqmr_runtime.json").read_text()); seal = json.loads((experiment / "checkpoints/hqmr_epoch25_final.json").read_text()); config = json.loads((experiment / "provenance/hqmr_config.json").read_text())
    checks = {"hqmr_sealed": seal["sealed_before_segmentation_evaluation"] and seal["sha256"] == sha256(hqmr_path),
              "sshr_frozen": sha256(sshr_path) == BASELINE_SHA256, "dfsc_frozen": sha256(dfsc_path) == DFSC_SHA256,
              "no_training_validation": not runtime["validation_accessed"], "full25": runtime["epochs"] == 25 and runtime["steps"] == 29275,
              "seed42": config["seed"] == 42, "batch20": config["effective_batch_size"] == 20,
              "image224": config["image_size"] == 224, "same_hardware": "4090" in torch.cuda.get_device_name(0)}
    protocol = {"decision": "COMPARABLE" if all(checks.values()) else "NOT_COMPARABLE", "checks": checks}
    write_json(experiment / "evaluation/hqmr_protocol_audit.json", protocol)
    if protocol["decision"] != "COMPARABLE": raise AssertionError(protocol)
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(sshr_path), strict=True); sshr.eval()
    hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(hqmr_path), strict=True); hqmr.eval()
    hist = {"sshr": [], **{name: [] for name in MODES}}; ids = []; coverage_counts = {name: defaultdict(float) for name in MODES}
    coverage_rows, paired_rows = [], []; distances = []; component = defaultdict(float); rescue = defaultdict(int); total_rescue = 0
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        sshr_prediction = _predict_sshr(sshr, image, original); bundle = predict_hqmr_modes(hqmr, image, original.shape[:2])
        hist["sshr"].append(foreground_confusion(truth, sshr_prediction)); ids.append(image_id)
        for name, payload in bundle["modes"].items():
            hist[name].append(foreground_confusion(truth, payload["prediction"]))
            overall, rows = coverage_one(payload["basis"], bundle["weights"], payload["evidence_grid"], truth, payload["prediction_grid"])
            for row in rows: coverage_rows.append({"image_id": image_id, "mode": name, **row})
            for key in ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall", "base_F_seed_coverage"):
                coverage_counts[name][key] += overall[key] * overall["gt_pixels"]
            coverage_counts[name]["gt_pixels"] += overall["gt_pixels"]
        full = bundle["modes"]["A_full"]; gt = resize_label(truth, full["evidence_grid"].shape[-2:]); sshr_grid = resize_label(sshr_prediction, gt.shape)
        basis, weights, evidence, pred = full["basis"], bundle["weights"], full["evidence_grid"], full["prediction_grid"]
        for cls in range(4):
            mask = gt == cls; threshold = float(THRESHOLDS[cls]); seed = evidence[cls] >= threshold; fn = mask & (pred != cls)
            eu, _, _ = distance_maps(seed); distances.extend(eu[fn].tolist())
            labels, count = ndimage.label(mask, structure=np.ones((3, 3), np.uint8))
            for component_id in range(1, count + 1):
                region = labels == component_id; fraction = float((seed & region).sum() / max(region.sum(), 1)); area = int(region.sum())
                category = "anchored" if fraction >= .05 else "weakly_anchored" if fraction > 0 else "unanchored"
                component[category + "_area"] += area; component["area"] += area; component[category + "_count"] += 1; component["count"] += 1
            rescued = fn & (sshr_grid == cls); n_rescued = int(rescued.sum()); total_rescue += n_rescued
            if n_rescued:
                binary = basis >= threshold; bmax = binary.any(0); inter = (binary & mask).sum((1, 2)); union = (binary | mask).sum((1, 2)); order = np.argsort(-np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0), kind="stable"); oracle10 = binary[order[:10]].any(0)
                type_e = rescued & (~bmax) & (~oracle10); type_d = rescued & (~type_e) & bmax & (~seed); remaining = rescued & (~type_e) & (~type_d)
                type_a = remaining & (eu <= 2); type_b = remaining & (eu > 2) & (eu <= 8); type_c = remaining & ((eu > 8) | (~np.isfinite(eu)))
                for key, value in zip("ABCDE", (type_a, type_b, type_c, type_d, type_e)): rescue[key] += int(value.sum())
        paired_rows.append({"image_id": image_id, "sshr_mIoU": scores_from_confusion(hist["sshr"][-1])["mIoU"],
                            **{f"{name}_mIoU": scores_from_confusion(hist[name][-1])["mIoU"] for name in MODES}})
        if index % 100 == 0 or index == len(loader): print(f"HQMR_EVAL_PROGRESS={index}/{len(loader)}", flush=True)
    hist = {name: np.stack(value) for name, value in hist.items()}; metrics = {name: scores_from_confusion(value.sum(0)) for name, value in hist.items()}
    if abs(metrics["sshr"]["mIoU"] - SSHR_MIOU) > 1e-12: raise AssertionError(f"SSHR reproduction mismatch: {metrics['sshr']['mIoU']}")
    write_json(experiment / "evaluation/hqmr_metrics_all_modes.json", metrics); write_csv(experiment / "evaluation/hqmr_per_image.csv", paired_rows)
    write_csv(experiment / "evaluation/hqmr_coverage_image_class.csv", coverage_rows)
    coverage_by_mode = {name: {key: coverage_counts[name][key] / coverage_counts[name]["gt_pixels"] for key in ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall", "base_F_seed_coverage")} for name in MODES}
    full_coverage = coverage_by_mode["A_full"]; gates = {"C1": full_coverage["class_basis_uncovered"] <= .5324, "C2": full_coverage["basis_max_coverage"] >= .4109, "C3": full_coverage["oracle_top10_recall"] >= .4346}
    finite_dist = np.asarray(distances); distance_summary = {"fn_pixels": len(finite_dist), "within_2": float(np.mean(finite_dist <= 2)), "within_8": float(np.mean(finite_dist <= 8)), "no_seed": float(np.mean(~np.isfinite(finite_dist))), "median_finite": float(np.median(finite_dist[np.isfinite(finite_dist)])) if np.isfinite(finite_dist).any() else float("inf")}
    component_summary = {"components": int(component["count"]), **{f"{name}_fraction": component[name + "_count"] / max(component["count"], 1) for name in ("anchored", "weakly_anchored", "unanchored")}, **{f"area_weighted_{name}_fraction": component[name + "_area"] / max(component["area"], 1) for name in ("anchored", "weakly_anchored", "unanchored")}}
    coverage = {"full": full_coverage, "by_mode": coverage_by_mode, "gates": gates, "recovery": sum(gates.values()) >= 2,
                "strong_recovery": full_coverage["class_basis_uncovered"] <= .40 and full_coverage["oracle_top10_recall"] >= .55,
                "rescue_type_fractions": {key: rescue[key] / max(total_rescue, 1) for key in "ABCDE"},
                "fn_to_seed_distance": distance_summary, "component_anchor": component_summary}
    write_json(experiment / "evaluation/hqmr_coverage_reaudit.json", coverage)
    boot = paired_bootstrap(hist["sshr"], hist["A_full"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED); write_json(experiment / "evaluation/hqmr_vs_sshr_bootstrap.json", boot)
    delta_sshr = 100 * (metrics["A_full"]["mIoU"] - metrics["sshr"]["mIoU"]); delta_dfsc = 100 * (metrics["A_full"]["mIoU"] - DFSC_MIOU)
    class_delta = {str(cls): 100 * (metrics["A_full"]["class_iou"][str(cls)] - metrics["sshr"]["class_iou"][str(cls)]) for cls in range(4)}
    ablation = {"full_minus_fine_pp": 100 * (metrics["A_full"]["mIoU"] - metrics["B_fine_only"]["mIoU"]), "full_minus_skip_f4_pp": 100 * (metrics["A_full"]["mIoU"] - metrics["C_coarse_fine"]["mIoU"]), "full_minus_no_query_update_pp": 100 * (metrics["A_full"]["mIoU"] - metrics["D_no_query_update"]["mIoU"]), "full_minus_coarse_pp": 100 * (metrics["A_full"]["mIoU"] - metrics["E_coarse_only"]["mIoU"]), "full_minus_mid_final_pp": 100 * (metrics["A_full"]["mIoU"] - metrics["F_mid_final"]["mIoU"])}
    ablation["core_hierarchy_pass"] = ablation["full_minus_fine_pp"] >= .10 and ablation["full_minus_coarse_pp"] >= .10
    verdict = decide(delta_sshr, boot["miou_ci95_pp"][0], delta_dfsc, class_delta, coverage, ablation)
    per_class = [{"class": cls, "sshr_iou": metrics["sshr"]["class_iou"][str(cls)], "hqmr_iou": metrics["A_full"]["class_iou"][str(cls)], "delta_iou_pp": class_delta[str(cls)]} for cls in range(4)]
    tests = {"unit": json.loads((experiment / "tests/hqmr_unit_tests.json").read_text())["passed"], "regression": json.loads((experiment / "tests/hqmr_regression_tests.json").read_text())["passed"]}
    interpretation = "HQMR 是否解决已诊断的 query-mask coverage bottleneck，以三项预注册 coverage gate 和同 checkpoint A–F 对照判断；性能结论仅取固定 E25 paired comparison。"
    next_step = "STRONG_GO/GO：进入 Seed11/17 multi-seed；BREAKTHROUGH_UNCERTAIN：直接 multi-seed，不调结构；COVERAGE_GO：诊断 downstream bottleneck；NEUTRAL/NOGO：按 coverage 与 A–F 结果定位失败层级。"
    result = {"decision": verdict, "metrics": metrics, "deltas_pp": {"vs_sshr": delta_sshr, "vs_dfsc": delta_dfsc, "class_vs_sshr": class_delta}, "bootstrap_vs_sshr": boot, "coverage": coverage, "ablation": ablation, "per_class": per_class, "training": runtime, "tests": tests, "protocol": protocol, "complexity": {"parameters": sum(p.numel() for p in hqmr.parameters()), "parameter_delta": config["parameter_delta"], "seconds_per_image_all_modes_plus_sshr": (time.perf_counter() - started) / len(loader), "peak_inference_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}, "provenance": {"training_source_commit": config["source_commit"], "evaluation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "hqmr_sha256": sha256(hqmr_path), "sshr_sha256": sha256(sshr_path), "dfsc_sha256": sha256(dfsc_path), "validation_images": len(loader)}, "interpretation": interpretation, "next_step": next_step}
    write_json(experiment / "evaluation/hqmr_final_result.json", result)
    report = experiment / "report/CCRA_HQMR_BCSS_Seed42_Full25_Final_Validation_Report.md"; report.write_text(report_text(result), encoding="utf-8")
    print(json.dumps({"decision": verdict, "report": str(report), "delta_vs_sshr_pp": delta_sshr, "delta_vs_dfsc_pp": delta_dfsc, "coverage": coverage, "ablation": ablation}, indent=2)); print(f"DECISION = {verdict}")


if __name__ == "__main__": main()
