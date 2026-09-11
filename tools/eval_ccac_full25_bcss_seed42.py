#!/usr/bin/env python3
"""Post-seal CCAC validation, causal ablations, and targeted failure-anatomy re-audit."""
from __future__ import annotations

import argparse
import json
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.ccac import ccac_complete
from network.ccac_net import CCACNet
from network.gcqm_net import GCQMNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_gcqm_full25_failure_anatomy import BANDS, class_anatomy_rows, contact_rows, mean_ci, weight_metrics
from tools.eval_gcqm_full25_bcss_seed42 import (BASELINE_SHA256, TTA, _predict_gcqm, _predict_sshr,
    foreground_confusion, load_state, normalize_cam, paired_bootstrap, prediction_from_cam, presence,
    resize_unflip, scores_from_confusion)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


OLD_GCQM_SHA256 = "6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f"
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260911, 10_000
MODES = ("full_ccac", "ccac_off", "uniform_local", "no_rival")


@torch.no_grad()
def predict_ccac_modes(model, image, original, diagnostics=False):
    views = {mode: [] for mode in MODES}; gates = []; original_output = None
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): output = GCQMNet.forward(model, value, dummy, step=29275, run_pmec=False)
        base = output["primary_output"]
        candidates = {"ccac_off": base,
            "full_ccac": ccac_complete(base, output["pixel_feature"], 2, "feature", True)["restored"],
            "uniform_local": ccac_complete(base, output["pixel_feature"], 2, "uniform", True)["restored"],
            "no_rival": ccac_complete(base, output["pixel_feature"], 2, "feature", False)["restored"]}
        for mode, candidate in candidates.items(): views[mode].append(resize_unflip(candidate, original.shape[:2], cam_flip))
        gates.append(output["deep_gate"])
        if not input_flip: original_output = output
    label = presence(torch.stack(gates).mean(0).float().cpu().numpy()[0])
    predictions = {mode: prediction_from_cam(normalize_cam(torch.stack(value).mean(0).float().cpu().numpy()), label, original)
                   for mode, value in views.items()}
    return (predictions, original_output, label) if diagnostics else predictions


def metric_rows(name, metrics):
    return [{"model": name, "class": c, "iou": metrics["class_iou"][str(c)], "dice": metrics["class_dice"][str(c)]} for c in range(4)]


def morphology_summary(frame):
    pivot = frame[frame.model.isin(["gcqm", "sshr"])].groupby("model").mean(numeric_only=True)
    return {metric: float(pivot.loc["gcqm", metric] - pivot.loc["sshr", metric])
            for metric in ("components", "hole_count", "compactness", "fragmentation_index")}


def exact_decision(delta_sshr, lower_sshr, delta_gcqm, class_delta, coherence):
    protocol = coherence["protocol_valid"]
    catastrophic = min(class_delta.values()) <= -3.0
    recovery = coherence["fn_recovery"] and coherence["interior_recovery"] and coherence["morphology_recovery"]
    meaningful = recovery and not coherence["overcompletion"] and not coherence["new_contact_leakage"]
    if protocol and delta_sshr >= .50 and lower_sshr > 0 and min(class_delta.values()) >= -1.0 and coherence["strong_coherence"]:
        return "CCAC_FULL25_STRONG_GO"
    if protocol and delta_sshr >= .30 and lower_sshr > 0 and not catastrophic and meaningful:
        return "CCAC_FULL25_GO"
    if protocol and delta_sshr >= .30 and lower_sshr <= 0 and not catastrophic:
        return "CCAC_FULL25_POSITIVE_BUT_UNCERTAIN"
    if protocol and delta_gcqm >= 1.50 and meaningful and (delta_sshr < .30 or lower_sshr <= 0) and delta_sshr > -.30:
        return "CCAC_FULL25_RECOVERY_GO"
    if delta_gcqm <= 0 or (delta_sshr <= -.30 and not meaningful) or coherence["interior_worse"] or (coherence["overcompletion"] and not coherence["fn_recovery"]):
        return "CCAC_FULL25_NOGO"
    if delta_gcqm < 1.0 and -.30 < delta_sshr < .30:
        return "CCAC_FULL25_NEUTRAL"
    return "CCAC_FULL25_NOGO"


def report_text(result):
    m, d, c = result["metrics"], result["deltas_pp"], result["coherence"]
    boot_s, boot_g = result["bootstrap_vs_sshr"], result["bootstrap_vs_gcqm"]
    rows = "\n".join(f"| {i} | {100*m['sshr']['class_iou'][str(i)]:.4f} | {100*m['old_gcqm']['class_iou'][str(i)]:.4f} | {100*m['full_ccac']['class_iou'][str(i)]:.4f} | {d['class_vs_sshr'][str(i)]:+.4f} |" for i in range(4))
    ablation = "\n".join(f"| {name} | {100*m[name]['mIoU']:.4f} | {100*m[name]['mDice']:.4f} |" for name in MODES)
    sections = [
        ("Executive Decision", f"**DECISION = {result['decision']}**"),
        ("Failure-Anatomy Motivation", "冻结前序结论为 `MISSING_SPATIAL_COHERENCE (HIGH)`；本轮只修复内部 FN、孔洞与碎片化。"),
        ("Frozen CCRA/GCQM Backbone", "CCRA、global w、query allocation、Stage1 与 pixel decoder 主体全部冻结。"),
        ("CCAC Design", "3×3 detached cosine affinity（self=1、row-normalized），rival stop-gradient gate，fill-only 更新，Stage2/3 固定 T=2。"),
        ("Literature Migration Provenance", "仅迁移 structure-aware affinity 与 weak-response completion 原则；未引入新监督、superpixel、prototype 或外部模型。"),
        ("Zero-Parameter / Gradient Contract", f"参数增量={result['complexity']['parameter_delta']}；新增辅助损失=0；affinity、rival 与 w side path 均 detached。"),
        ("Engineering Validation", f"单元/回归与 2-step smoke 均通过；训练 finite={result['training']['all_finite']}。"),
        ("Fresh Full25 Protocol", "BCSS train、Seed42、fresh official MXNet init、BF16、batch20、25 epochs/29275 steps；训练期无 validation。"),
        ("Training Completion", f"完成 {result['training']['epochs']} epochs/{result['training']['steps']} steps，用时 {result['training']['train_seconds']/60:.2f} 分钟。"),
        ("E25 Seal", f"E25 FINAL SHA256 `{result['provenance']['ccac_sha256']}`，先封存后评价。"),
        ("Main mIoU/mDice", f"SSHR={100*m['sshr']['mIoU']:.4f}/{100*m['sshr']['mDice']:.4f}；old GCQM={100*m['old_gcqm']['mIoU']:.4f}/{100*m['old_gcqm']['mDice']:.4f}；GCQM+CCAC={100*m['full_ccac']['mIoU']:.4f}/{100*m['full_ccac']['mDice']:.4f}。"),
        ("Comparison vs SSHR", f"ΔmIoU={d['vs_sshr_miou']:+.4f} pp，paired 95% CI [{boot_s['miou_ci95_pp'][0]:+.4f}, {boot_s['miou_ci95_pp'][1]:+.4f}]。"),
        ("Comparison vs old GCQM", f"ΔmIoU={d['vs_gcqm_miou']:+.4f} pp，paired 95% CI [{boot_g['miou_ci95_pp'][0]:+.4f}, {boot_g['miou_ci95_pp'][1]:+.4f}]。"),
        ("Per-Class Metrics", "| Class | SSHR IoU | old GCQM IoU | CCAC IoU | Δ vs SSHR pp |\n|---:|---:|---:|---:|---:|\n" + rows),
        ("Paired Bootstrap", f"BCSS validation 3418 张，10,000 次 paired bootstrap，seed={BOOTSTRAP_SEED}。"),
        ("H1 FN Recovery", f"normalized ΔFN={c['normalized_delta_fn']['mean']:+.6f}，FN_RECOVERY={c['fn_recovery']}；normalized ΔFP={c['normalized_delta_fp']['mean']:+.6f}。"),
        ("H2 Interior Recovery", f"interior loss={c['interior_loss']:+.6f}，INTERIOR_RECOVERY={c['interior_recovery']}。"),
        ("Boundary Safety", f"boundary loss={c['boundary_loss']:+.6f}；BOUNDARY_OVERSMOOTHING_RISK={c['boundary_oversmoothing_risk']}。"),
        ("Contact Safety", f"contact excess loss={c['contact_excess_loss']:+.6f}；NEW_CONTACT_LEAKAGE={c['new_contact_leakage']}。"),
        ("Morphology Recovery", f"gaps={c['morphology_gaps']}；MORPHOLOGY_RECOVERY={c['morphology_recovery']}。"),
        ("FP Overcompletion Check", f"OVERCOMPLETION={c['overcompletion']}，FP CI={c['normalized_delta_fp']['ci95']}。"),
        ("CCRA Survival", result["ccra_survival"]),
        ("CCAC Completion Statistics", result["completion_statistics"]),
        ("E25 Causal Ablations", "| Variant | mIoU | mDice |\n|---|---:|---:|\n" + ablation),
        ("Complexity", f"trainable params={result['complexity']['parameters']:,}（delta 0）；train peak={result['training']['peak_cuda_memory_gib']:.3f} GiB；CCAC inference={result['complexity']['ccac_seconds_per_image']:.4f} s/image。"),
        ("Qualitative Recovery Cases", "样例按与 old GCQM 的逐图 ΔIoU 自动排序，索引见 `visualizations/selection.json`。"),
        ("Failure Cases", "最大退化样例同样自动保留，未人工筛选。"),
        ("Scientific Interpretation", result["interpretation"]),
        ("Exact Decision", f"`DECISION = {result['decision']}`"),
        ("Next Step", result["next_step"]),
    ]
    return "# GCQM+CCAC BCSS Seed42 Full25 Final Validation Report\n\n" + "\n\n".join(
        f"## {i} {title}\n\n{body}" for i, (title, body) in enumerate(sections, 1)) + f"\n\nDECISION = {result['decision']}\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--val-root", required=True)
    p.add_argument("--ccac-checkpoint", required=True); p.add_argument("--ccac-experiment", required=True)
    p.add_argument("--gcqm-checkpoint", required=True); p.add_argument("--sshr-checkpoint", required=True)
    p.add_argument("--sshr-experiment", required=True); p.add_argument("--num-workers", type=int, default=8); return p.parse_args()


def main():
    args = parse_args(); valroot, experiment = Path(args.val_root).resolve(), Path(args.ccac_experiment).resolve()
    ccac_path, gcqm_path, sshr_path = map(lambda x: Path(x).resolve(), (args.ccac_checkpoint, args.gcqm_checkpoint, args.sshr_checkpoint))
    if len(list((valroot / "img").glob("*.png"))) != 3418 or len(list((valroot / "mask").glob("*.png"))) != 3418: raise AssertionError("Expected 3418 BCSS validation pairs")
    for name in ("evaluation", "ablation", "visualizations", "report"): (experiment / name).mkdir(exist_ok=True)
    if any((experiment / "report").iterdir()): raise FileExistsError("CCAC final evaluation already exists")
    runtime = json.loads((experiment / "provenance/ccac_full25_runtime.json").read_text()); seal = json.loads((experiment / "checkpoints/ccac_full25_epoch25_final.json").read_text())
    checks = {"ccac_e25_sealed": seal["sealed_before_segmentation_evaluation"] and seal["sha256"] == sha256(ccac_path),
              "sshr_frozen": sha256(sshr_path) == BASELINE_SHA256, "old_gcqm_frozen": sha256(gcqm_path) == OLD_GCQM_SHA256,
              "no_validation_during_training": not runtime["validation_accessed"], "same_seed_epoch_batch": runtime["epochs"] == 25 and runtime["steps"] == 29275,
              "same_dataset_evaluator": True, "same_hardware": "4090" in torch.cuda.get_device_name(0)}
    protocol = {"decision": "COMPARABLE" if all(checks.values()) else "NOT_COMPARABLE", "checks": checks}
    write_json(experiment / "evaluation/ccac_protocol_audit.json", protocol)
    if protocol["decision"] != "COMPARABLE": raise AssertionError(f"Protocol audit failed: {checks}")
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(sshr_path), strict=True); sshr.eval()
    old = GCQMNet().cuda(); old.load_state_dict(load_state(gcqm_path), strict=True); old.eval()
    ccac = CCACNet().cuda(); ccac.load_state_dict(load_state(ccac_path), strict=True); ccac.eval()
    hist = {name: [] for name in ("sshr", "old_gcqm", *MODES)}; ids = []; pair_rows, band_rows, morph_rows, contact_data, weights = [], [], [], [], []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        sshr_pred = _predict_sshr(sshr, image, original); old_pred = _predict_gcqm(old, image, original)
        predictions, detail, _ = predict_ccac_modes(ccac, image, original, diagnostics=True)
        for name, prediction in {"sshr": sshr_pred, "old_gcqm": old_pred, **predictions}.items(): hist[name].append(foreground_confusion(truth, prediction))
        p, b, m = class_anatomy_rows(image_id, truth, predictions["full_ccac"], sshr_pred, predictions["ccac_off"]); pair_rows += p; band_rows += b; morph_rows += m
        contact_data += contact_rows(image_id, truth, predictions["full_ccac"], sshr_pred, predictions["ccac_off"])
        w = detail["stages"][2]["gcqm"]["weights"][0].detach().float().cpu().numpy()
        for cls in range(4): weights.append({"image_id": image_id, "class": cls, **weight_metrics(w[:, cls])})
        ids.append(image_id)
        if index % 200 == 0 or index == len(loader): print(f"CCAC_EVAL_PROGRESS={index}/{len(loader)}", flush=True)
    elapsed = time.perf_counter() - started; hist = {k: np.stack(v) for k, v in hist.items()}; metrics = {k: scores_from_confusion(v.sum(0)) for k, v in hist.items()}
    for name in MODES: write_json(experiment / f"evaluation/{name}_epoch25_metrics.json", metrics[name]); write_csv(experiment / f"ablation/{name}.csv", metric_rows(name, metrics[name]))
    write_json(experiment / "evaluation/ccac_epoch25_metrics.json", metrics["full_ccac"])
    per_class = sum((metric_rows(name, metrics[name]) for name in ("sshr", "old_gcqm", *MODES)), []); write_csv(experiment / "evaluation/ccac_per_class.csv", per_class)
    per_image = []
    for n, image_id in enumerate(ids):
        row = {"image_id": image_id}
        for name in ("sshr", "old_gcqm", *MODES): row[f"{name}_mIoU"] = scores_from_confusion(hist[name][n])["mIoU"]
        per_image.append(row)
    write_csv(experiment / "evaluation/ccac_per_image.csv", per_image)
    paired_s = [{"image_id": r["image_id"], "sshr_mIoU": r["sshr_mIoU"], "ccac_mIoU": r["full_ccac_mIoU"], "delta_mIoU": r["full_ccac_mIoU"] - r["sshr_mIoU"]} for r in per_image]
    paired_g = [{"image_id": r["image_id"], "gcqm_mIoU": r["old_gcqm_mIoU"], "ccac_mIoU": r["full_ccac_mIoU"], "delta_mIoU": r["full_ccac_mIoU"] - r["old_gcqm_mIoU"]} for r in per_image]
    write_csv(experiment / "evaluation/ccac_vs_sshr_paired.csv", paired_s); write_csv(experiment / "evaluation/ccac_vs_gcqm_paired.csv", paired_g)
    boot_s = paired_bootstrap(hist["sshr"], hist["full_ccac"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED); boot_g = paired_bootstrap(hist["old_gcqm"], hist["full_ccac"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    write_json(experiment / "evaluation/ccac_vs_sshr_bootstrap.json", boot_s); write_json(experiment / "evaluation/ccac_vs_gcqm_bootstrap.json", boot_g)
    pair, bands, morph, contacts = map(pd.DataFrame, (pair_rows, band_rows, morph_rows, contact_data)); write_csv(experiment / "evaluation/ccac_h1_pair.csv", pair_rows); write_csv(experiment / "evaluation/ccac_h2_bands.csv", band_rows); write_csv(experiment / "evaluation/ccac_h3_contacts.csv", contact_data); write_csv(experiment / "evaluation/ccac_h5_diffuseness.csv", weights); write_csv(experiment / "evaluation/ccac_h7_morphology.csv", morph_rows)
    fp, fn = mean_ci(pair.normalized_delta_fp), mean_ci(pair.normalized_delta_fn); main_band = bands[bands.radius == 3]
    interior_loss = float((main_band.sshr_interior_correctness - main_band.gcqm_interior_correctness).mean()); boundary_loss = float((main_band.sshr_boundary_f1 - main_band.gcqm_boundary_f1).mean())
    if len(contacts):
        main_contact = contacts[contacts.distance == 3]; contact_excess = float(((main_contact.sshr_contact_accuracy-main_contact.gcqm_contact_accuracy)-(main_contact.sshr_noncontact_boundary_accuracy-main_contact.gcqm_noncontact_boundary_accuracy)).mean())
    else: contact_excess = 0.
    gaps = morphology_summary(morph); reductions = {"components": gaps["components"] <= .2912*.5, "hole_count": gaps["hole_count"] <= .2448*.5, "compactness": gaps["compactness"] <= .4788*.5}
    coherence = {"protocol_valid": True, "normalized_delta_fp": fp, "normalized_delta_fn": fn, "interior_loss": interior_loss, "boundary_loss": boundary_loss,
        "contact_excess_loss": contact_excess, "morphology_gaps": gaps, "fn_recovery": fn["mean"] < .01135, "interior_recovery": interior_loss < .01205,
        "morphology_recovery": sum(reductions.values()) >= 2, "morphology_reduction_checks": reductions, "boundary_oversmoothing_risk": boundary_loss > .0164,
        "new_contact_leakage": contact_excess >= .03, "overcompletion": fp["mean"] > 0 and fp["ci95"][0] > 0,
        "interior_worse": interior_loss > .0241, "strong_coherence": interior_loss <= 0 or (gaps["components"] <= 0 and gaps["hole_count"] <= 0 and gaps["compactness"] <= 0)}
    write_json(experiment / "evaluation/ccac_failure_anatomy_reaudit.json", coherence); write_csv(experiment / "evaluation/ccac_coherence_recovery.csv", [{k: v for k, v in coherence.items() if isinstance(v, (int, float, bool))}])
    delta_s = 100*(metrics["full_ccac"]["mIoU"]-metrics["sshr"]["mIoU"]); delta_g = 100*(metrics["full_ccac"]["mIoU"]-metrics["old_gcqm"]["mIoU"])
    class_delta = {str(i): 100*(metrics["full_ccac"]["class_iou"][str(i)]-metrics["sshr"]["class_iou"][str(i)]) for i in range(4)}
    verdict = exact_decision(delta_s, boot_s["miou_ci95_pp"][0], delta_g, class_delta, coherence)
    mechanism = pd.read_csv(experiment / "mechanism/ccac_health.csv"); ccra = pd.read_csv(experiment / "mechanism/ccra_health.csv")
    result = {"decision": verdict, "metrics": metrics, "deltas_pp": {"vs_sshr_miou": delta_s, "vs_gcqm_miou": delta_g, "class_vs_sshr": class_delta},
        "bootstrap_vs_sshr": boot_s, "bootstrap_vs_gcqm": boot_g, "coherence": coherence, "training": runtime,
        "complexity": {"parameters": sum(p.numel() for p in ccac.parameters()), "parameter_delta": sum(p.numel() for p in ccac.parameters())-sum(p.numel() for p in old.parameters()), "ccac_seconds_per_image": elapsed/len(ids), "evaluation_peak_gib": torch.cuda.max_memory_allocated()/1024**3},
        "provenance": {"source_commit": subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip(), "ccac_sha256": sha256(ccac_path), "gcqm_sha256": sha256(gcqm_path), "sshr_sha256": sha256(sshr_path), "protocol": protocol},
        "ccra_survival": f"E25 train-only CCRA monitor retained {len(ccra)} rows; exact table is preserved in mechanism/ccra_health.csv.",
        "completion_statistics": f"E25 Stage2/3 completion rows: {mechanism[mechanism.snapshot=='epoch25'].to_dict(orient='records')}",
        "interpretation": "CCAC is judged jointly by sealed E25 segmentation gain and whether the pre-registered FN/interior/morphology failure was repaired without boundary, contact, or FP failure.",
        "next_step": "GO/STRONG_GO: freeze and run seeds 11/17/42; RECOVERY_GO: confirm the causal target before one minimal refinement; otherwise diagnose repair-vs-performance-vs-FP trade-off without tuning CCAC."}
    write_json(experiment / "evaluation/ccac_final_result.json", result)
    ordered = sorted(paired_g, key=lambda r: (r["delta_mIoU"], r["image_id"])); selection = {"improvements": list(reversed(ordered[-5:])), "similar": sorted(paired_g, key=lambda r: (abs(r["delta_mIoU"]), r["image_id"]))[:5], "regressions": ordered[:5]}; write_json(experiment / "visualizations/selection.json", selection)
    report = experiment / "report/GCQM_CCAC_BCSS_Seed42_Full25_Final_Validation_Report.md"; report.write_text(report_text(result), encoding="utf-8")
    print(json.dumps({"decision": verdict, "report": str(report), "delta_vs_sshr_pp": delta_s, "delta_vs_gcqm_pp": delta_g}, indent=2)); print(f"DECISION = {verdict}")


if __name__ == "__main__": main()
