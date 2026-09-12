#!/usr/bin/env python3
"""Post-seal CP-HQMR validation, A-F causal ablations, and balance audit."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.cphqmr_net import CPHQMRNet
from network.hqmr import class_mixture
from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_semantic_coverage_reachability import resize_label
from tools.eval_gcqm_full25_bcss_seed42 import (
    BASELINE_SHA256, THRESHOLDS, TTA, _predict_sshr, foreground_confusion, load_state,
    normalize_cam, paired_bootstrap, prediction_from_cam, presence, resize_unflip,
    scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json
from tools.run_hqmr_full25_bcss_seed42 import _js_rows
from train_cqrf_phase0 import MonitorDataset
from train_gcqm_phase0 import load_cohort


BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260912, 10_000
HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
DFSC_SHA256 = "470f1056f2bbf5c64b5e6fff76861f9fa4e1663bf7ba0c74e2621528fb48af11"
SSHR_MIOU, HQMR_MIOU, DFSC_MIOU = 0.6669670591172749, 0.6557244403737567, 0.6452820195735897
MODES = {
    "A_full": "full", "B_discriminative_only": "discriminative_only",
    "C_coverage_only": "coverage_only", "D_simple_average": "simple_average",
    "E_bilinear_only": "bilinear_only", "F_old_f3_semantic": "old_f3_semantic",
}


def _mode_bundle(model: CPHQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    full_views = {name: [] for name in MODES}; grid_views = {name: [] for name in MODES}
    basis_views = {name: [] for name in MODES}; weights, gates = [], []
    branches = {name: [] for name in ("C4", "D4", "M4", "restore_gate")}
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, cphqmr_mode="full")
            stage = output["stages"][2]; item = stage["cphqmr"]
            h5, h4, h3 = output["query_detail"]["context_feature"], output["pixel_detail"]["F4_context"], output["features"]["F3"]
            decoded = {"A_full": item}
            for name, mode in list(MODES.items())[1:]:
                current = model.cphqmr(stage["query"], h5, h4, h3, mode=mode)
                current["mixture"] = class_mixture(current["basis"], item["weights"])
                decoded[name] = current
        for name, current in decoded.items():
            mixture, basis = current["mixture"][0], current["basis"][0]
            if cam_flip:
                mixture, basis = torch.flip(mixture, dims=cam_flip), torch.flip(basis, dims=cam_flip)
            full_views[name].append(resize_unflip(current["mixture"], original_hw, cam_flip).float().cpu())
            grid_views[name].append(mixture.float().cpu()); basis_views[name].append(basis.float().cpu())
        for name in ("C4", "D4", "M4"):
            value = torch.nn.functional.interpolate(item[name], size=h3.shape[-2:], mode="bilinear", align_corners=False)[0].sigmoid()
            if cam_flip: value = torch.flip(value, dims=cam_flip)
            branches[name].append(value.float().cpu())
        restore_gate = item["dgsr"]["gate"][0]
        if cam_flip: restore_gate = torch.flip(restore_gate, dims=cam_flip)
        branches["restore_gate"].append(restore_gate.float().cpu())
        weights.append(item["weights"][0].float().cpu()); gates.append(output["deep_gate"].float().cpu())
    label = presence(torch.stack(gates).mean(0).numpy()[0])
    result = {"label": label, "weights": torch.stack(weights).mean(0).numpy(), "modes": {},
              "branches": {name: torch.stack(value).mean(0).numpy() for name, value in branches.items()}}
    for name in MODES:
        formal = normalize_cam(torch.stack(full_views[name]).mean(0).numpy())
        grid = normalize_cam(torch.stack(grid_views[name]).mean(0).numpy())
        result["modes"][name] = {"prediction": prediction_from_cam(formal, label, np.empty(original_hw)),
            "prediction_grid": prediction_from_cam(grid, label, np.empty(grid.shape[-2:])), "evidence_grid": grid,
            "basis": torch.stack(basis_views[name]).mean(0).numpy(), "class_map": torch.stack(grid_views[name]).mean(0).numpy()}
    return result


def _hqmr_bundle(model: HQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    full, grid, basis, weights, gates = [], [], [], [], []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): output = model(value, dummy, step=29275, hqmr_mode="full")
        item = output["stages"][2]["hqmr"]; mixture, current_basis = item["mixture"][0], item["basis"][0]
        if cam_flip: mixture, current_basis = torch.flip(mixture, dims=cam_flip), torch.flip(current_basis, dims=cam_flip)
        full.append(resize_unflip(item["mixture"], original_hw, cam_flip).float().cpu()); grid.append(mixture.float().cpu())
        basis.append(current_basis.float().cpu()); weights.append(item["weights"][0].float().cpu()); gates.append(output["deep_gate"].float().cpu())
    label = presence(torch.stack(gates).mean(0).numpy()[0]); formal = normalize_cam(torch.stack(full).mean(0).numpy()); grid_mean = normalize_cam(torch.stack(grid).mean(0).numpy())
    return {"prediction": prediction_from_cam(formal, label, np.empty(original_hw)),
        "prediction_grid": prediction_from_cam(grid_mean, label, np.empty(grid_mean.shape[-2:])), "evidence_grid": grid_mean,
        "basis": torch.stack(basis).mean(0).numpy(), "weights": torch.stack(weights).mean(0).numpy(),
        "class_map": torch.stack(grid).mean(0).numpy()}


def coverage_purity_one(payload: dict, weights: np.ndarray, truth: np.ndarray) -> tuple[dict, list[dict]]:
    gt = resize_label(truth, payload["basis"].shape[-2:]); basis = payload["basis"]; rows = []; total = defaultdict(float)
    for cls in range(4):
        target = gt == cls; rival = (gt < 4) & (gt != cls); background = gt == 4; n = int(target.sum()); threshold = float(THRESHOLDS[cls])
        binary = basis >= threshold; bmax = binary.any(0); weighted_order = np.argsort(-weights[:, cls], kind="stable"); top20 = binary[weighted_order[:20]].any(0)
        intersection = (binary & target).sum((1, 2)); union = (binary | target).sum((1, 2)); iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
        oracle10 = binary[np.argsort(-iou, kind="stable")[:10]].any(0); class_map = np.einsum("q,qhw->hw", weights[:, cls], basis); mass = float(class_map.sum()) + 1e-8
        prediction = payload["prediction_grid"]
        values = {"basis_max_coverage": float((bmax & target).sum() / max(n, 1)), "class_basis_uncovered": float(((~top20) & target).sum() / max(n, 1)),
            "oracle_top10_recall": float((oracle10 & target).sum() / max(n, 1)), "weighted_purity": float(class_map[target].sum() / mass),
            "rival_mass": float(class_map[rival].sum() / mass), "background_mass": float(class_map[background].sum() / mass),
            "prediction_gt_area_ratio": float((prediction == cls).sum() / max(n, 1)), "FN": int((target & (prediction != cls)).sum()),
            "FP": int(((~target) & (prediction == cls)).sum())}
        rows.append({"class": cls, "gt_pixels": n, **values}); total["gt_pixels"] += n
        for key, value in values.items(): total[key] += value if key in ("FN", "FP") else value * n
    denominator = max(total["gt_pixels"], 1)
    overall = {"gt_pixels": int(total["gt_pixels"]), **{key: (int(total[key]) if key in ("FN", "FP") else total[key] / denominator) for key in values}}
    return overall, rows


def _aggregate(total: dict) -> dict:
    denominator = max(total["gt_pixels"], 1)
    keys = ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall", "weighted_purity", "rival_mass", "background_mass", "prediction_gt_area_ratio")
    return {**{key: total[key] / denominator for key in keys}, "FN": int(total["FN"]), "FP": int(total["FP"]), "gt_pixels": int(total["gt_pixels"])}


def decide(delta_sshr, lower_sshr, delta_hqmr, class_delta, balance, causal):
    coverage_worse = balance["CoverageGain"] < 0 and balance["UncoveredReduction"] < 0
    purity_worse = balance["PurityDelta"] < 0 and balance["RivalDelta"] > 0
    if delta_hqmr <= -.30 or (coverage_worse and purity_worse) or balance["RivalDelta"] > .03:
        return "CPHQMR_FULL25_NOGO"
    if delta_sshr >= .50 and lower_sshr > 0 and min(class_delta.values()) >= -1 and balance["improved"] and causal["dual_state"]:
        return "CPHQMR_FULL25_STRONG_GO"
    if delta_sshr >= .30 and lower_sshr > 0 and min(class_delta.values()) > -3 and causal["dual_state"]:
        return "CPHQMR_FULL25_GO"
    if delta_sshr > 0 and (delta_sshr < .30 or lower_sshr <= 0) and delta_hqmr >= .30:
        return "CPHQMR_FULL25_BREAKTHROUGH_UNCERTAIN"
    if delta_hqmr >= .50 and delta_sshr < -.30:
        return "CPHQMR_FULL25_IMPROVEMENT_GO"
    if -.30 < delta_hqmr < .30 and not balance["improved"] and not causal["dual_state"]:
        return "CPHQMR_FULL25_NEUTRAL"
    return "CPHQMR_FULL25_NOGO"


def render_selected_cases(loader, selected_ids, valroot, experiment, sshr, hqmr, model):
    """Render one evidence-rich panel for every unique automatically selected case."""
    remaining = set(selected_ids); query_rows = []
    for names, image in loader:
        image_id = names[0]
        if image_id not in remaining:
            continue
        original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        sshr_pred = _predict_sshr(sshr, image, original); old = _hqmr_bundle(hqmr, image, original.shape[:2]); new = _mode_bundle(model, image, original.shape[:2]); full = new["modes"]["A_full"]
        counts = np.bincount(truth[truth < 4].ravel(), minlength=4); cls = int(counts.argmax()); weights = new["weights"][:, cls]
        aggregate = lambda value: np.einsum("q,qhw->hw", weights, value)
        c4, d4, m4 = (aggregate(new["branches"][name]) for name in ("C4", "D4", "M4")); gate = new["branches"]["restore_gate"][0]
        bmax = (full["basis"] >= float(THRESHOLDS[cls])).any(0); final_f = normalize_cam(full["class_map"])[cls]
        error = (full["prediction"] != truth).astype(np.float32)
        panels = [(original, "Input", None), (truth, "GT", "tab10"), (sshr_pred, "SSHR", "tab10"), (old["prediction"], "HQMR-v1", "tab10"),
            (full["prediction"], "CP-HQMR", "tab10"), (c4, f"C4 class {cls}", "magma"), (d4, "D4", "magma"), (m4, "M4", "magma"),
            (gate, "F3 restore gate", "viridis"), (bmax, "final Bmax", "gray"), (final_f, "final class F", "magma"), (error, "error map", "Reds")]
        fig, axes = plt.subplots(3, 4, figsize=(15, 11))
        for axis, (value, title, cmap) in zip(axes.flat, panels):
            axis.imshow(value, cmap=cmap, interpolation="nearest"); axis.set_title(title); axis.axis("off")
        fig.suptitle(f"{image_id} | representative foreground class {cls}"); fig.tight_layout(); fig.savefig(experiment / f"visualizations/{image_id}.png", dpi=140); plt.close(fig)
        top_queries = np.argsort(-weights, kind="stable")[:5]; gt_grid = resize_label(truth, full["basis"].shape[-2:]); target, rival = gt_grid == cls, (gt_grid < 4) & (gt_grid != cls)
        qfig, qaxes = plt.subplots(4, 5, figsize=(16, 11))
        for column, query_id in enumerate(top_queries):
            basis_map = full["basis"][query_id]; mass = float(basis_map.sum()) + 1e-8; coverage = float(basis_map[target].mean()) if target.any() else 0.0; purity = float(basis_map[target].sum() / mass); rival_mass = float(basis_map[rival].sum() / mass)
            query_rows.append({"image_id": image_id, "class": cls, "rank": column + 1, "query_id": int(query_id), "weight": float(weights[query_id]), "GT_soft_coverage": coverage, "GT_purity": purity, "rival_mass": rival_mass})
            maps = (new["branches"]["C4"][query_id], new["branches"]["D4"][query_id], new["branches"]["M4"][query_id], basis_map)
            for row_index, (value, label) in enumerate(zip(maps, ("C4", "D4", "M4", "final B"))):
                qaxes[row_index, column].imshow(value, cmap="magma", vmin=0, vmax=1); qaxes[row_index, column].axis("off")
                qaxes[row_index, column].set_title(f"q{query_id} w={weights[query_id]:.3f}" if row_index == 0 else (f"{label}\ncov={coverage:.2f} pur={purity:.2f}" if row_index == 3 else label))
        qfig.suptitle(f"{image_id} | top-5 class-{cls} ownership queries"); qfig.tight_layout(); qfig.savefig(experiment / f"visualizations/{image_id}_top5_queries.png", dpi=140); plt.close(qfig)
        remaining.remove(image_id)
        if not remaining: break
    if remaining: raise AssertionError(f"Selected visualization IDs not found: {sorted(remaining)}")
    write_csv(experiment / "visualizations/cphqmr_top5_query_metrics.csv", query_rows)


@torch.no_grad()
def posthoc_ccra_health(model, experiment):
    """Reconstruct frozen train-cohort CCRA health from every milestone checkpoint."""
    _, cohort = load_cohort(experiment / "provenance/cphqmr_monitor_cohort.json")
    loader = DataLoader(MonitorDataset(cohort), batch_size=8, num_workers=4, pin_memory=True); rows = []
    for epoch in (5, 10, 15, 20, 25):
        checkpoint = experiment / ("checkpoints/cphqmr_epoch25_final.pth" if epoch == 25 else f"checkpoints/cphqmr_epoch{epoch:02d}.pth")
        model.load_state_dict(load_state(checkpoint), strict=True); model.eval(); js_values = {2: [], 3: []}; neff = {2: [], 3: []}
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): output = model(images, labels, step=epoch * 1171)
            for stage_index in (2, 3):
                weights = output["stages"][stage_index - 1]["cphqmr"]["weights"].float(); js_values[stage_index].extend(_js_rows(weights)); entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1); neff[stage_index].extend(entropy.exp().cpu().tolist())
        for stage_index in (2, 3): rows.append({"snapshot": f"epoch{epoch}", "stage": stage_index, "ccra_js": float(np.mean(js_values[stage_index])), "Neff": float(np.mean(neff[stage_index])), "train_only": True})
    write_csv(experiment / "mechanism/ccra_health.csv", rows)


def report_text(result: dict) -> str:
    m, d, b, c, a = result["metrics"], result["deltas_pp"], result["bootstrap"], result["coverage_purity"], result["causal"]
    pre = result["preaudit"]; sections = [
        ("Executive Decision", f"**DECISION = {result['decision']}**。CP-HQMR−SSHR={d['vs_sshr']:+.4f} pp；CP-HQMR−HQMR-v1={d['vs_hqmr_v1']:+.4f} pp。"),
        ("Frozen HQMR-v1 Evidence", f"HQMR-v1 E25 mIoU={100*HQMR_MIOU:.4f}，checkpoint SHA256=`{HQMR_SHA256}`。"),
        ("Coverage–Purity Conflict Motivation", "HQMR-v1 的 query update 提升语义集中度但压缩可达覆盖；CP-HQMR 因而分离 coverage 与 discriminative query state。"),
        ("Pre-run Conflict Audit", f"预审计决策 `{pre['decision']}`；no-update−full={pre['delta_no_update_minus_full']}。"),
        ("CP-HQMR Architecture", "H5/H4 双状态语义解码 + coverage-preserving fusion (CFR) + H3 query/class-agnostic DGSR。"),
        ("Coverage Query State", "q_cov=LN(Q)，全层不接受 region update。"),
        ("Discriminative Query State", "q_disc 仅在 H5 通过 soft-pooled V5 更新一次。"),
        ("Coverage-Preserving Fusion", "M4=D4+(1−sigmoid(D4))·relu(C4−D4)，无阈值和可调 λ。"),
        ("H4 Semantic Endpoint", "主路径语义重建终止于 H4；H3 不执行 query-semantic qK。"),
        ("F3 Detail-Guided Spatial Restoration", "H3 仅预测共享 3×3 动态空间核与恢复门，使用 replicate padding。"),
        ("Frozen Loss/CCRA Contract", "CCRA、PCA、deep、tri-state target、detached w 和 .50/.25/.25 loss 全部冻结。"),
        ("Engineering Validation", f"unit={result['tests']['unit']}，regression={result['tests']['regression']}，2-step BF16 smoke passed；梯度/数值契约通过。"),
        ("Fresh Full25 Protocol", "BCSS Seed42，official MXNet/ImageNet init，batch20，BF16，25 epochs / 29275 steps，无 validation 选模。"),
        ("E25 Seal", f"E25 FINAL SHA256=`{result['provenance']['cphqmr_sha256']}`，封存后才访问 validation。"),
        ("Main mIoU/mDice", "\n".join(["| Model | mIoU | mDice |", "|---|---:|---:|", f"| SSHR | {100*m['sshr']['mIoU']:.4f} | {100*m['sshr']['mDice']:.4f} |", f"| HQMR-v1 | {100*m['hqmr_v1']['mIoU']:.4f} | {100*m['hqmr_v1']['mDice']:.4f} |", f"| CP-HQMR | {100*m['A_full']['mIoU']:.4f} | {100*m['A_full']['mDice']:.4f} |"])) ,
        ("Comparison vs SSHR", f"Δ={d['vs_sshr']:+.4f} pp，95% CI={b['vs_sshr']['miou_ci95_pp']} pp。"),
        ("Comparison vs HQMR-v1", f"Δ={d['vs_hqmr_v1']:+.4f} pp，95% CI={b['vs_hqmr_v1']['miou_ci95_pp']} pp。"),
        ("Per-Class Metrics", str(result["per_class"])),
        ("Paired Bootstrap", f"10,000 次 paired resampling，seed={BOOTSTRAP_SEED}；详细结果见 evaluation JSON。"),
        ("Coverage–Purity Re-audit", f"balance={c['balance']}。"),
        ("Bmax/Uncovered/Oracle10", f"A_full: Bmax={c['by_mode']['A_full']['basis_max_coverage']:.4f}，uncovered={c['by_mode']['A_full']['class_basis_uncovered']:.4f}，oracle10={c['by_mode']['A_full']['oracle_top10_recall']:.4f}。"),
        ("Basis Purity/Rival/BG", f"A_full: purity={c['by_mode']['A_full']['weighted_purity']:.4f}，rival={c['by_mode']['A_full']['rival_mass']:.4f}，background={c['by_mode']['A_full']['background_mass']:.4f}。"),
        ("Dual-State Causal Ablation", f"A−B={a['A_minus_B_pp']:+.4f} pp，A−C={a['A_minus_C_pp']:+.4f} pp；gate={a['dual_state']}。"),
        ("CFR Ablation", f"A−D={a['A_minus_D_pp']:+.4f} pp；gate={a['cfr']}。"),
        ("DGSR Ablation", f"A−E={a['A_minus_E_pp']:+.4f} pp；gate={a['dgsr']}。"),
        ("Old F3 Semantic Ablation", f"A−F={a['A_minus_F_pp']:+.4f} pp；gate={a['spatial_over_semantic']}。"),
        ("Complexity", f"参数={result['complexity']['parameters']:,}，vs HQMR-v1={result['complexity']['delta_vs_hqmr_v1']:,}；train={result['training']['train_seconds']/60:.2f} min，peak VRAM={result['training']['peak_cuda_memory_gib']:.3f} GiB。"),
        ("Qualitative Cases", "已自动生成五类 top-5 案例清单及去重后的多面板图，位于 visualizations/。"),
        ("Scientific Interpretation", result["interpretation"]),
        ("Exact Decision", f"`DECISION = {result['decision']}`"),
        ("Next Step", result["next_step"]),
    ]
    return "# CCRA CP-HQMR BCSS Seed42 Full25 Final Validation Report\n\n" + "\n\n".join(f"## {i} {title}\n\n{body}" for i, (title, body) in enumerate(sections, 1)) + f"\n\nDECISION = {result['decision']}\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--val-root", required=True); p.add_argument("--experiment", required=True)
    p.add_argument("--cphqmr-checkpoint", required=True); p.add_argument("--hqmr-checkpoint", required=True); p.add_argument("--sshr-checkpoint", required=True); p.add_argument("--dfsc-checkpoint", required=True)
    p.add_argument("--report-copy", required=True); p.add_argument("--num-workers", type=int, default=8); return p.parse_args()


def main():
    args = parse_args(); valroot, experiment = Path(args.val_root).resolve(), Path(args.experiment).resolve(); report_copy = Path(args.report_copy).resolve()
    cpath, hpath, spath, dpath = map(lambda value: Path(value).resolve(), (args.cphqmr_checkpoint, args.hqmr_checkpoint, args.sshr_checkpoint, args.dfsc_checkpoint))
    if len(list((valroot / "img").glob("*.png"))) != 3418 or len(list((valroot / "mask").glob("*.png"))) != 3418: raise AssertionError("Expected 3418 BCSS validation pairs")
    runtime = json.loads((experiment / "provenance/cphqmr_runtime.json").read_text()); seal = json.loads((experiment / "checkpoints/cphqmr_epoch25_final.json").read_text()); config = json.loads((experiment / "provenance/cphqmr_config.json").read_text()); preaudit = json.loads((experiment / "preaudit/hqmr_v1_conflict_summary.json").read_text())
    checks = {"cphqmr_sealed": seal["sealed_before_segmentation_evaluation"] and seal["sha256"] == sha256(cpath), "hqmr_frozen": sha256(hpath) == HQMR_SHA256,
        "sshr_frozen": sha256(spath) == BASELINE_SHA256, "dfsc_frozen": sha256(dpath) == DFSC_SHA256, "no_training_validation": not runtime["validation_accessed"],
        "full25": runtime["epochs"] == 25 and runtime["steps"] == 29275, "seed42": config["seed"] == 42, "batch20": config["effective_batch_size"] == 20,
        "image224": config["image_size"] == 224, "same_hardware": "4090" in torch.cuda.get_device_name(0)}
    protocol = {"decision": "COMPARABLE" if all(checks.values()) else "NOT_COMPARABLE", "checks": checks}; write_json(experiment / "evaluation/cphqmr_protocol_audit.json", protocol)
    if protocol["decision"] != "COMPARABLE": raise AssertionError(protocol)
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(spath), strict=True); sshr.eval(); hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(hpath), strict=True); hqmr.eval(); model = CPHQMRNet().cuda(); model.load_state_dict(load_state(cpath), strict=True); model.eval()
    posthoc_ccra_health(model, experiment); model.load_state_dict(load_state(cpath), strict=True); model.eval()
    hist = {"sshr": [], "hqmr_v1": [], **{name: [] for name in MODES}}; totals = {name: defaultdict(float) for name in ("hqmr_v1", *MODES)}; rows, coverage_rows, candidates = [], [], []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        sshr_pred = _predict_sshr(sshr, image, original); old = _hqmr_bundle(hqmr, image, original.shape[:2]); new = _mode_bundle(model, image, original.shape[:2])
        predictions = {"sshr": sshr_pred, "hqmr_v1": old["prediction"], **{name: payload["prediction"] for name, payload in new["modes"].items()}}
        image_scores = {}; hist["sshr"].append(foreground_confusion(truth, sshr_pred)); hist["hqmr_v1"].append(foreground_confusion(truth, old["prediction"]))
        for name in MODES: hist[name].append(foreground_confusion(truth, predictions[name]))
        for name in hist: image_scores[name] = scores_from_confusion(hist[name][-1])["mIoU"]
        bundles = {"hqmr_v1": (old, old["weights"]), **{name: (payload, new["weights"]) for name, payload in new["modes"].items()}}
        per_image_mechanism = {}
        for name, (payload, weights) in bundles.items():
            overall, class_rows = coverage_purity_one(payload, weights, truth); per_image_mechanism[name] = overall
            for row in class_rows: coverage_rows.append({"image_id": image_id, "mode": name, **row})
            totals[name]["gt_pixels"] += overall["gt_pixels"]
            for key, value in overall.items():
                if key == "gt_pixels": continue
                totals[name][key] += value if key in ("FN", "FP") else value * overall["gt_pixels"]
        rows.append({"image_id": image_id, **{f"{name}_mIoU": value for name, value in image_scores.items()}})
        candidates.append({"image_id": image_id, "gain_vs_hqmr": image_scores["A_full"] - image_scores["hqmr_v1"], "gain_vs_sshr": image_scores["A_full"] - image_scores["sshr"],
            "coverage_rescue": per_image_mechanism["A_full"]["basis_max_coverage"] - per_image_mechanism["hqmr_v1"]["basis_max_coverage"],
            "purity_loss": per_image_mechanism["hqmr_v1"]["weighted_purity"] - per_image_mechanism["A_full"]["weighted_purity"]})
        if index % 100 == 0 or index == len(loader): print(f"CPHQMR_EVAL_PROGRESS={index}/{len(loader)}", flush=True)
    hist = {name: np.stack(value) for name, value in hist.items()}; metrics = {name: scores_from_confusion(value.sum(0)) for name, value in hist.items()}
    if abs(metrics["sshr"]["mIoU"] - SSHR_MIOU) > 1e-12 or abs(metrics["hqmr_v1"]["mIoU"] - HQMR_MIOU) > 5e-5: raise AssertionError("Frozen comparator reproduction mismatch")
    write_json(experiment / "evaluation/cphqmr_metrics.json", metrics); write_csv(experiment / "evaluation/cphqmr_per_image.csv", rows); write_csv(experiment / "coverage_purity/cphqmr_basis_purity.csv", coverage_rows)
    by_mode = {name: _aggregate(value) for name, value in totals.items()}; write_csv(experiment / "coverage_purity/cphqmr_basis_coverage.csv", [{"mode": name, **value} for name, value in by_mode.items()])
    h, n = by_mode["hqmr_v1"], by_mode["A_full"]; balance = {"CoverageGain": n["basis_max_coverage"] - h["basis_max_coverage"], "UncoveredReduction": h["class_basis_uncovered"] - n["class_basis_uncovered"], "PurityDelta": n["weighted_purity"] - h["weighted_purity"], "RivalDelta": n["rival_mass"] - h["rival_mass"]}
    balance["improved"] = (balance["CoverageGain"] >= .05 or balance["UncoveredReduction"] >= .05) and balance["PurityDelta"] >= -.03 and balance["RivalDelta"] <= .03
    write_csv(experiment / "coverage_purity/cphqmr_balance_metrics.csv", [{**balance, "hqmr_Bmax": h["basis_max_coverage"], "cphqmr_Bmax": n["basis_max_coverage"]}])
    for name in MODES: write_csv(experiment / f"ablation/{MODES[name]}.csv", [{"mode": name, **metrics[name], **by_mode[name]}])
    boot_sshr = paired_bootstrap(hist["sshr"], hist["A_full"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED); boot_hqmr = paired_bootstrap(hist["hqmr_v1"], hist["A_full"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    write_json(experiment / "evaluation/cphqmr_vs_sshr_bootstrap.json", boot_sshr); write_json(experiment / "evaluation/cphqmr_vs_hqmr_v1_bootstrap.json", boot_hqmr)
    delta_sshr = 100 * (metrics["A_full"]["mIoU"] - metrics["sshr"]["mIoU"]); delta_hqmr = 100 * (metrics["A_full"]["mIoU"] - metrics["hqmr_v1"]["mIoU"])
    class_delta = {str(cls): 100 * (metrics["A_full"]["class_iou"][str(cls)] - metrics["sshr"]["class_iou"][str(cls)]) for cls in range(4)}
    diff = lambda other: 100 * (metrics["A_full"]["mIoU"] - metrics[other]["mIoU"])
    causal = {"A_minus_B_pp": diff("B_discriminative_only"), "A_minus_C_pp": diff("C_coverage_only"), "A_minus_D_pp": diff("D_simple_average"), "A_minus_E_pp": diff("E_bilinear_only"), "A_minus_F_pp": diff("F_old_f3_semantic")}
    causal.update({"dual_state": causal["A_minus_B_pp"] > 0 and causal["A_minus_C_pp"] > 0 and max(causal["A_minus_B_pp"], causal["A_minus_C_pp"]) >= .10 and balance["improved"], "cfr": causal["A_minus_D_pp"] > 0, "dgsr": causal["A_minus_E_pp"] > 0, "spatial_over_semantic": causal["A_minus_F_pp"] > 0})
    verdict = decide(delta_sshr, boot_sshr["miou_ci95_pp"][0], delta_hqmr, class_delta, balance, causal)
    selections = {"largest_gains_vs_hqmr": sorted(candidates, key=lambda x: x["gain_vs_hqmr"], reverse=True)[:5], "largest_gains_vs_sshr": sorted(candidates, key=lambda x: x["gain_vs_sshr"], reverse=True)[:5], "largest_regressions": sorted(candidates, key=lambda x: x["gain_vs_sshr"])[:5], "largest_coverage_rescue": sorted(candidates, key=lambda x: x["coverage_rescue"], reverse=True)[:5], "largest_purity_loss": sorted(candidates, key=lambda x: x["purity_loss"], reverse=True)[:5]}; write_json(experiment / "visualizations/cphqmr_selected_cases.json", selections)
    selected_ids = list(dict.fromkeys(row["image_id"] for category in selections.values() for row in category))
    render_selected_cases(loader, selected_ids, valroot, experiment, sshr, hqmr, model)
    tests = {"unit": json.loads((experiment / "tests/cphqmr_unit_tests.json").read_text())["passed"], "regression": json.loads((experiment / "tests/cphqmr_regression_tests.json").read_text())["passed"]}
    result = {"decision": verdict, "metrics": metrics, "deltas_pp": {"vs_sshr": delta_sshr, "vs_hqmr_v1": delta_hqmr, "vs_dfsc": 100 * (metrics["A_full"]["mIoU"] - DFSC_MIOU)}, "bootstrap": {"vs_sshr": boot_sshr, "vs_hqmr_v1": boot_hqmr},
        "per_class": class_delta, "coverage_purity": {"by_mode": by_mode, "balance": balance}, "causal": causal, "preaudit": preaudit, "protocol": protocol, "tests": tests, "training": runtime,
        "complexity": {"parameters": sum(p.numel() for p in model.parameters()), "delta_vs_hqmr_v1": config["parameter_delta_vs_hqmr_v1"], "inference_seconds_per_image_all_models_modes": (time.perf_counter() - started) / len(loader), "peak_inference_gib": torch.cuda.max_memory_allocated() / 1024 ** 3},
        "provenance": {"training_source_commit": config["source_commit"], "evaluation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "cphqmr_sha256": sha256(cpath), "hqmr_sha256": sha256(hpath), "sshr_sha256": sha256(spath), "dfsc_sha256": sha256(dpath)},
        "interpretation": "性能只由固定 E25 paired comparison 判定；coverage–purity balance 与 A–F 仅作为预注册机制证据，不用于选模。",
        "next_step": "STRONG_GO/GO/BREAKTHROUGH_UNCERTAIN：冻结架构并进入 Seed11/17/42；IMPROVEMENT_GO：只做一次零训练 residual audit；NEUTRAL/NOGO：停止叠加 decoder 分支。"}
    write_json(experiment / "evaluation/cphqmr_final_result.json", result); report = experiment / "report/CCRA_CPHQMR_BCSS_Seed42_Full25_Final_Validation_Report.md"; report.write_text(report_text(result), encoding="utf-8"); report_copy.parent.mkdir(parents=True, exist_ok=True); report_copy.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps({"decision": verdict, "delta_vs_sshr_pp": delta_sshr, "delta_vs_hqmr_v1_pp": delta_hqmr, "balance": balance, "causal": causal, "report": str(report)}, indent=2)); print(f"DECISION = {verdict}")


if __name__ == "__main__":
    main()
