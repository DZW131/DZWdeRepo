#!/usr/bin/env python3
"""Frozen-E25, zero-training HQMR-v1 Class-2/3 residual failure audit."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_gcqm_full25_failure_anatomy import boundary_band, mask_morphology, mean_ci, correlation_ci
from tools.eval_gcqm_full25_bcss_seed42 import (
    FIXED_WEIGHTS, THRESHOLDS, TTA, foreground_confusion, load_state, normalize_cam,
    prediction_from_cam, presence, resize_unflip, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
SSHR_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
CPHQMR_SHA256 = "980cb80cba116a758b1026814168c66611f4c3388be3892afe6af05aea235d7e"
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260913, 10_000
EXPECTED_HQMR = [0.755612, 0.692000, 0.558130, 0.617156]
EXPECTED_SSHR = [0.763582, 0.701575, 0.573211, 0.629500]
EPS = 1e-8


def ratio(num, den): return float(num / den) if den else 0.0


def confusion5(truth, prediction):
    valid = (truth >= 0) & (truth <= 4); return np.bincount(5 * truth[valid].astype(np.int64) + prediction[valid].astype(np.int64), minlength=25).reshape(5, 5)


def normalized_matrix(matrix, axis):
    denominator = matrix.sum(axis=axis, keepdims=True); return np.divide(matrix, denominator, out=np.zeros_like(matrix, dtype=float), where=denominator > 0)


def bootstrap_delta(values): return mean_ci(values, seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES)


def js_divergence(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float); left /= max(left.sum(), EPS); right /= max(right.sum(), EPS); mean = .5 * (left + right)
    return float(.5 * np.sum(left * np.log(np.clip(left / mean, EPS, None))) + .5 * np.sum(right * np.log(np.clip(right / mean, EPS, None))))


@torch.no_grad()
def infer_sshr(model, image, original_hw):
    views, probabilities = [[], [], []], []
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): _, c1, c2, deep, probability = model.forward_cam(value)
        for index, cam in enumerate((c1, c2, deep)): views[index].append(resize_unflip(cam, original_hw, cam_flip).float().cpu())
        probabilities.append(probability.float().cpu())
    normalized = [normalize_cam(torch.stack(value).mean(0).numpy()) for value in views]; scores = sum(weight * cam for weight, cam in zip(FIXED_WEIGHTS, normalized)); label = presence(torch.stack(probabilities).mean(0).numpy()[0])
    return {"scores": scores, "label": label, "prediction": prediction_from_cam(scores, label, np.empty(original_hw))}


@torch.no_grad()
def infer_hqmr(model, image, original_hw):
    full, grid, bases, weights, gates = [], [], [], [], []; dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): output = model(value, dummy, step=29275, hqmr_mode="full")
        item = output["stages"][2]["hqmr"]; mixture, basis = item["mixture"], item["basis"]
        full.append(resize_unflip(mixture, original_hw, cam_flip).float().cpu()); grid.append(resize_unflip(mixture, basis.shape[-2:], cam_flip).float().cpu())
        basis_view = basis[0]
        if cam_flip: basis_view = torch.flip(basis_view, dims=cam_flip)
        bases.append(basis_view.float().cpu()); weights.append(item["weights"][0].float().cpu()); gates.append(output["deep_gate"].float().cpu())
    scores = normalize_cam(torch.stack(full).mean(0).numpy()); grid_scores = normalize_cam(torch.stack(grid).mean(0).numpy()); label = presence(torch.stack(gates).mean(0).numpy()[0])
    return {"scores": scores, "grid_scores": grid_scores, "label": label, "prediction": prediction_from_cam(scores, label, np.empty(original_hw)),
            "basis": torch.stack(bases).mean(0).numpy(), "weights": torch.stack(weights).mean(0).numpy()}


def basis_audit(image_id, truth, bundle):
    basis, weights = bundle["basis"], bundle["weights"]; gt = np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((basis.shape[-1], basis.shape[-2]), Image.Resampling.NEAREST))
    per_query, weighted, tails, topk = [], [], [], []
    for cls in range(4):
        target = gt == cls
        if not target.any(): continue
        rival, background = (gt < 4) & (gt != cls), gt == 4; mass = basis.sum((1, 2)) + EPS
        target_mass, rival_mass, bg_mass = (basis * target).sum((1, 2)), (basis * rival).sum((1, 2)), (basis * background).sum((1, 2))
        purity, rival_fraction, bg_fraction = target_mass / mass, rival_mass / mass, bg_mass / mass; coverage = target_mass / (target.sum() + EPS); oracle_score = purity * coverage
        for query in range(len(basis)):
            per_query.append({"image_id": image_id, "class": cls, "query_id": query, "weight": float(weights[query, cls]), "target_mass": float(target_mass[query]), "rival_mass": float(rival_mass[query]), "background_mass": float(bg_mass[query]), "soft_purity": float(purity[query]), "soft_coverage": float(coverage[query]), "oracle_score": float(oracle_score[query])})
        w = weights[:, cls] / max(weights[:, cls].sum(), EPS); weighted.append({"image_id": image_id, "class": cls, "weighted_purity": float(np.sum(w * purity)), "weighted_rival": float(np.sum(w * rival_fraction)), "weighted_background": float(np.sum(w * bg_fraction)), "weighted_coverage": float(np.sum(w * coverage))})
        actual_order, oracle_order = np.argsort(-weights[:, cls], kind="stable"), np.argsort(-oracle_score, kind="stable")
        groups = (("top10pct", actual_order[:20]), ("middle40pct", actual_order[20:98]), ("bottom50pct", actual_order[98:]))
        for group, chosen in groups:
            tails.append({"image_id": image_id, "class": cls, "group": group, "purity": float(purity[chosen].mean()), "rival": float(rival_fraction[chosen].mean()), "background": float(bg_fraction[chosen].mean()), "coverage": float(coverage[chosen].mean())})
        for k in (1, 3, 5, 10):
            for ranking, order in (("actual", actual_order), ("oracle", oracle_order)):
                chosen = order[:k]; union = basis[chosen].max(0); union_binary = union >= .5
                recall = ratio(int((union_binary & target).sum()), int(target.sum())); union_mass = float(union.sum()) + EPS
                topk.append({"image_id": image_id, "class": cls, "ranking": ranking, "k": k, "recall": recall, "purity": float(union[target].sum() / union_mass), "rival": float(union[rival].sum() / union_mass), "mean_score": float(oracle_score[chosen].mean())})
    return per_query, weighted, tails, topk


def weight_separability(image_id, weights):
    rows = []
    for left, right in itertools.combinations(range(4), 2):
        wl, wr = weights[:, left], weights[:, right]; order_l, order_r = np.argsort(-wl), np.argsort(-wr)
        rho = stats.spearmanr(wl, wr).statistic
        rows.append({"image_id": image_id, "pair": f"{left}-{right}", "js": js_divergence(wl, wr), "top5_overlap": len(set(order_l[:5]) & set(order_r[:5])) / 5, "top10_overlap": len(set(order_l[:10]) & set(order_r[:10])) / 10, "spearman": float(rho), "mean_abs_difference": float(np.abs(wl - wr).mean())})
    return rows


def class_metrics_5(matrix, cls):
    tp = matrix[cls, cls]; fn = matrix[cls].sum() - tp; fp = matrix[:, cls].sum() - tp
    return {"TP": int(tp), "FP": int(fp), "FN": int(fn), "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn), "IoU": ratio(tp, tp + fp + fn), "Dice": ratio(2 * tp, 2 * tp + fp + fn), "normalized_FP": ratio(fp, matrix[cls].sum()), "normalized_FN": ratio(fn, matrix[cls].sum())}


def spatial_rows(image_id, truth, sshr_pred, hqmr_pred, hqmr_scores):
    morphology, bands, contacts, components, depths = [], [], [], [], []
    for cls in (2, 3):
        target = truth == cls; gt_area = int(target.sum())
        if not gt_area: continue
        for model_name, prediction in (("sshr", sshr_pred), ("hqmr", hqmr_pred)):
            morphology.append({"image_id": image_id, "class": cls, "model": model_name, **mask_morphology(prediction == cls, gt_area)})
        for radius in (1, 3, 5):
            boundary = boundary_band(target, radius); interior = ndimage.binary_erosion(target, iterations=radius)
            row = {"image_id": image_id, "class": cls, "radius": radius, "gt_pixels": gt_area}
            for model_name, prediction in (("sshr", sshr_pred), ("hqmr", hqmr_pred)):
                row.update({f"{model_name}_interior_FN": ratio(int((interior & (prediction != cls)).sum()), int(interior.sum())), f"{model_name}_boundary_FN": ratio(int((boundary & target & (prediction != cls)).sum()), int((boundary & target).sum())), f"{model_name}_boundary_FP": ratio(int((boundary & (~target) & (prediction == cls)).sum()), int((boundary & (~target)).sum()))})
            bands.append(row)
        labels, count = ndimage.label(target, structure=np.ones((3, 3), np.uint8)); distance = ndimage.distance_transform_edt(target); margin = np.abs(hqmr_scores[2] - hqmr_scores[3])
        for component_id in range(1, count + 1):
            region = labels == component_id; components.append({"image_id": image_id, "class": cls, "component_id": component_id, "area": int(region.sum()), "mutual_confusion": float(np.mean(hqmr_pred[region] == 5-cls)), "background_fn": 0.0, "semantic_margin": float(margin[region].mean())})
        for name, region in (("0-2", target & (distance <= 2)), ("2-4", target & (distance > 2) & (distance <= 4)), ("4-8", target & (distance > 4) & (distance <= 8)), (">8", target & (distance > 8))):
            if region.any(): depths.append({"image_id": image_id, "class": cls, "depth": name, "pixels": int(region.sum()), "mutual_confusion": float(np.mean(hqmr_pred[region] == 5-cls)), "background_fn": 0.0, "semantic_margin": float(margin[region].mean())})
    for distance in (1, 3, 5):
        m2, m3 = truth == 2, truth == 3; near = (ndimage.binary_dilation(m2, iterations=distance) & m3) | (ndimage.binary_dilation(m3, iterations=distance) & m2)
        if near.any(): contacts.append({"image_id": image_id, "distance": distance, "pixels": int(near.sum()), "sshr_confusion": float(np.mean(((truth == 2) & (sshr_pred == 3) | (truth == 3) & (sshr_pred == 2))[near])), "hqmr_confusion": float(np.mean(((truth == 2) & (hqmr_pred == 3) | (truth == 3) & (hqmr_pred == 2))[near]))})
    return morphology, bands, contacts, components, depths


@torch.no_grad()
def render_selected(loader, selections, valroot, output, sshr, hqmr):
    groups_by_id = defaultdict(list)
    for group, ids in selections.items():
        for rank, image_id in enumerate(ids, 1): groups_by_id[image_id].append((group, rank))
    remaining = set(groups_by_id)
    for names, image in loader:
        image_id = names[0]
        if image_id not in remaining: continue
        original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        s, h = infer_sshr(sshr, image, truth.shape), infer_hqmr(hqmr, image, truth.shape); basis, weights = h["basis"], h["weights"]
        top2, top3 = np.argsort(-weights[:,2])[:5], np.argsort(-weights[:,3])[:5]; contribution2 = np.einsum("q,qhw->hw",weights[:,2],basis); contribution3 = np.einsum("q,qhw->hw",weights[:,3],basis)
        panels=[(original,"input",None),(truth,"GT","tab10"),(s["prediction"],"SSHR prediction","tab10"),(h["prediction"],"HQMR prediction","tab10"),((h["prediction"]!=truth).astype(float),"HQMR error","Reds"),(h["scores"][2],"F2","magma"),(h["scores"][3],"F3","magma"),(h["scores"][2]-h["scores"][3],"F2-F3 margin","coolwarm"),(basis[top2].max(0),"top5 class2 basis","viridis"),(basis[top3].max(0),"top5 class3 basis","viridis"),(contribution2,"class2 contribution","magma"),(contribution3,"class3 contribution","magma")]
        fig,axes=plt.subplots(3,4,figsize=(15,11))
        for ax,(value,title,cmap) in zip(axes.flat,panels): ax.imshow(value,cmap=cmap); ax.set_title(title); ax.axis("off")
        fig.suptitle(image_id); fig.tight_layout()
        for group,rank in groups_by_id[image_id]: fig.savefig(output/f"visualizations/{group}/{rank:02d}_{image_id}.png",dpi=140)
        plt.close(fig)
        gt=np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((basis.shape[-1],basis.shape[-2]),Image.Resampling.NEAREST)); qfig,qaxes=plt.subplots(2,5,figsize=(17,7))
        for row,(cls,chosen) in enumerate(((2,top2),(3,top3))):
            for column,qid in enumerate(chosen):
                value=basis[qid]; mass=float(value.sum())+EPS; m2=float(value[gt==2].sum()/mass); m3=float(value[gt==3].sum()/mass); mb=float(value[gt==4].sum()/mass)
                qaxes[row,column].imshow(value,cmap="viridis",vmin=0,vmax=1); qaxes[row,column].axis("off"); qaxes[row,column].set_title(f"c{cls} q{qid}\nw2={weights[qid,2]:.3f} w3={weights[qid,3]:.3f}\nGT2={m2:.2f} GT3={m3:.2f} BG={mb:.2f}")
        qfig.suptitle(f"{image_id} class2/3 top-weight queries"); qfig.tight_layout(); qfig.savefig(output/f"visualizations/query_views/{image_id}.png",dpi=140); plt.close(qfig)
        remaining.remove(image_id)
        if not remaining: break
    if remaining: raise AssertionError(f"Visualization IDs missing: {sorted(remaining)}")


def report_text(result):
    r, h, matrix = result["reproduction_gate"], result["hypotheses"], result["decision_matrix"]
    sections = [
        ("Executive Diagnosis", f"**DECISION = {result['decision']}**；**CONFIDENCE = {result['confidence']}**。{result['because_sentence']}"),
        ("Frozen HQMR-v1 Evidence", f"HQMR-v1 E25 SHA256 `{HQMR_SHA256}`；mIoU={100*r['hqmr']['mIoU']:.4f}，mDice={100*r['hqmr']['mDice']:.4f}。"),
        ("Reproduction Gate", f"3418-image frozen validation reproduction=`{r['status']}`；SSHR={100*r['sshr']['mIoU']:.4f}，HQMR-v1={100*r['hqmr']['mIoU']:.4f}；TTA/threshold/class mapping/ignore/interpolation unchanged。"),
        ("Why CP-HQMR Is Archived", "CP-HQMR=65.3800 mIoU，低于 HQMR-v1，且双状态 coverage-purity gate 未通过；本轮未加载其权重，仅封存身份。"),
        ("Per-Class Residual Gap", str(result["per_class_gap_pp"])),
        ("Full Confusion Matrix", "raw、row-normalized、column-normalized 和 HQMR−SSHR delta 均保存在 confusion/。正式协议不预测 background，因此 prediction-background 列为 0。"),
        ("Class2/3 Mutual Confusion", f"ΔConf23={h['H1']['delta']:+.4f}，CI={h['H1']['ci95']}，result={h['H1']['result']}。"),
        ("Class2/3 Background FN", f"正式冻结 argmax 无 background 输出；贡献率={h['H2']['contribution']:.4f}，result={h['H2']['result']}。"),
        ("Error Attribution", str(result["error_attribution"])),
        ("Basis Purity", f"Class2/3 vs Class0/1 purity deficit={h['H3']['purity_deficit']:+.4f}；result={h['H3']['result']}。"),
        ("Rival / Background Mass", f"Class2/3 rival excess={h['H3']['rival_excess']:+.4f}；完整分层见 basis/。"),
        ("Top-weight vs Tail", str(result["top_tail_summary"])),
        ("Query-Class Coupling", f"Class2/3 actual-vs-oracle gap={h['H4']['gap_recall']:+.4f}；result={h['H4']['result']}。"),
        ("Actual-vs-Oracle Basis", str(result["actual_oracle_summary"])),
        ("Class Weight Separability", str(result["weight_separability"])),
        ("Pixel-level Class2-vs-Class3 Separability", str(result["pixel_separability"])),
        ("Spatial Morphology", f"worse properties={h['H5']['worse_properties']}；result={h['H5']['result']}。"),
        ("Interior / Boundary", str(result["interior_boundary_summary"])),
        ("Contact", str(result["contact_summary"])),
        ("SSHR Rescue23", f"pixels={result['rescue23']['pixels']}；images={result['rescue23']['images']}。"),
        ("Rescue Type Decomposition", str(result["rescue23"]["types"])),
        ("Semantic Margin", str(result["rescue23"]["margin"])),
        ("Component Scale", "Class-specific tertile results are in spatial/component_size.csv。"),
        ("Interior Depth", "0–2/2–4/4–8/>8 results are in spatial/interior_depth.csv；用于区分边界与深部语义错误。"),
        ("Correlation Analysis", "Spearman estimates and 10,000-image bootstrap CIs (seed=20260913) are in correlation/。"),
        ("Representative Cases", "五类自动 top-5 cases 与 query-level panels 位于 visualizations/；未人工挑选。"),
        ("Decision Matrix", "\n".join(["| Hypothesis | Result | Confidence | Core evidence |", "|---|---|---|---|"] + [f"| {name} | {row['result']} | {row['confidence']} | {row['evidence']} |" for name, row in matrix.items()])),
        ("Exact Residual Bottleneck", result["because_sentence"]),
        ("What Is Preserved", "CCRA、HQMR-v1、H5→H4 hierarchy、region-conditioned query update、detached w 和 frozen weak supervision。"),
        ("What Is Archived", "CP-HQMR dual state/CFR/DGSR primary path、F3 semantic re-decoding、CCAC/DFSC/local propagation、top-k sparsification、pixel-wise FOMD restoration。"),
        ("Exact Next Architecture Target", result["next_target"]),
        ("What Must NOT Be Done", "不得自动恢复 old FOMD，不得再叠加 decoder branch，不得训练或调阈值；下一轮只针对 primary bottleneck。"),
        ("Final Decision", f"`DECISION = {result['decision']}`\n\n`CONFIDENCE = {result['confidence']}`"),
    ]
    return "# HQMR-v1 Class-2/3 Residual Failure Audit Report\n\n" + "\n\n".join(f"## {i} {title}\n\n{body}" for i, (title, body) in enumerate(sections, 1)) + f"\n\nDECISION = {result['decision']}\nCONFIDENCE = {result['confidence']}\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--val-root", required=True); p.add_argument("--hqmr-checkpoint", required=True); p.add_argument("--sshr-checkpoint", required=True); p.add_argument("--cphqmr-result", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--num-workers", type=int, default=8); return p.parse_args()


def main():
    args = parse_args(); valroot, hpath, spath, cresult, output = map(lambda value: Path(value).resolve(), (args.val_root, args.hqmr_checkpoint, args.sshr_checkpoint, args.cphqmr_result, args.output_dir))
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    if sha256(hpath) != HQMR_SHA256 or sha256(spath) != SSHR_SHA256: raise AssertionError("Frozen checkpoint mismatch")
    cp = json.loads(cresult.read_text());
    if cp["provenance"]["cphqmr_sha256"] != CPHQMR_SHA256 or cp["decision"] != "CPHQMR_FULL25_NEUTRAL": raise AssertionError("CP-HQMR archive mismatch")
    for name in ("provenance", "paired", "confusion", "fp_fn", "basis", "weighting", "spatial", "sshr_rescue", "correlation", "visualizations/losses", "visualizations/wins", "visualizations/confusion_2to3", "visualizations/confusion_3to2", "visualizations/background_fn", "visualizations/query_views", "report"): (output / name).mkdir(parents=True, exist_ok=True)
    config = {"audit": "HQMR-v1 Class2/3 residual", "zero_training": True, "parameter_updates": 0, "threshold_tuning": False, "checkpoint_selection": False, "seed": 42, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "thresholds": THRESHOLDS.tolist(), "tta_views": 3, "background_prediction_available": False, "background_note": "Frozen evaluator argmaxes four foreground scores; no post-hoc background threshold is introduced."}
    write_json(output / "provenance/audit_config.json", config); (output / "provenance/audit_source_commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()+"\n"); (output / "provenance/audit_git_diff.patch").write_text(subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True))
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if len(loader) != 3418: raise AssertionError("Expected 3418 validation images")
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(spath), strict=True); sshr.eval(); hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(hpath), strict=True); hqmr.eval()
    hists = {"sshr": [], "hqmr": []}; matrix_by_image = {"sshr": [], "hqmr": []}; targeted, pairs, per_query, weighted, tails, topk, separation = [], [], [], [], [], [], []
    morph, bands, contacts, components, depths, rescue_types, rescue_evidence, rescue_margin = [], [], [], [], [], [], [], []
    pixel_y, pixel_sshr, pixel_hqmr = [], [], []; candidate_rows = []
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        s, h = infer_sshr(sshr, image, truth.shape), infer_hqmr(hqmr, image, truth.shape); sp, hp = s["prediction"], h["prediction"]
        sh, hh = foreground_confusion(truth, sp), foreground_confusion(truth, hp); hists["sshr"].append(sh); hists["hqmr"].append(hh); matrix_by_image["sshr"].append(confusion5(truth, sp)); matrix_by_image["hqmr"].append(confusion5(truth, hp))
        sm, hm = scores_from_confusion(sh)["mIoU"], scores_from_confusion(hh)["mIoU"]
        row = {"image_id": image_id, "sshr_IoU": sm, "hqmr_IoU": hm, "DeltaIoU": hm-sm, "GT_area_c2": int((truth==2).sum()), "GT_area_c3": int((truth==3).sum())}
        for cls, rival_cls in ((2, 3), (3, 2)):
            gt = truth == cls; n = max(int(gt.sum()), 1)
            for key, pred in (("sshr", sp), ("hqmr", hp)):
                row[f"{key}_{cls}to{rival_cls}"] = int((gt & (pred == rival_cls)).sum()); row[f"{key}_{cls}toBG"] = int((gt & (pred == 4)).sum()); row[f"{key}_{cls}to01"] = int((gt & ((pred == 0)|(pred == 1))).sum())
            targeted.append({"image_id": image_id, "class": cls, "mutual_delta": row[f"hqmr_{cls}to{rival_cls}"]/n-row[f"sshr_{cls}to{rival_cls}"]/n, "background_delta": 0.0, "to01_delta": row[f"hqmr_{cls}to01"]/n-row[f"sshr_{cls}to01"]/n})
        pq, wb, tt, tk = basis_audit(image_id, truth, h); per_query += pq; weighted += wb; tails += tt; topk += tk; separation += weight_separability(image_id, h["weights"])
        current_w = {r["class"]: r for r in wb}; current_top = {(r["class"],r["ranking"],r["k"]):r for r in tk}
        for cls in (2,3):
            if cls in current_w:
                row.update({f"c{cls}_purity": current_w[cls]["weighted_purity"], f"c{cls}_rival": current_w[cls]["weighted_rival"], f"c{cls}_actual_top10_recall": current_top[(cls,"actual",10)]["recall"], f"c{cls}_oracle_top10_recall": current_top[(cls,"oracle",10)]["recall"], f"c{cls}_router_gap": current_top[(cls,"oracle",10)]["recall"]-current_top[(cls,"actual",10)]["recall"]})
        sr = spatial_rows(image_id, truth, sp, hp, h["scores"]); morph += sr[0]; bands += sr[1]; contacts += sr[2]; components += sr[3]; depths += sr[4]
        valid23 = (truth == 2) | (truth == 3); pixel_y.extend((truth[valid23] == 2).astype(np.uint8).tolist()); pixel_sshr.extend((s["scores"][2]-s["scores"][3])[valid23].tolist()); pixel_hqmr.extend((h["scores"][2]-h["scores"][3])[valid23].tolist())
        rescue = valid23 & (sp == truth) & (hp != truth); type_masks = {"R1_2to3": rescue&(truth==2)&(hp==3), "R2_3to2": rescue&(truth==3)&(hp==2), "R3_2toBG": rescue&(truth==2)&(hp==4), "R4_3toBG": rescue&(truth==3)&(hp==4), "R5_to01": rescue&((hp==0)|(hp==1)), "R6_other": rescue&~(((truth==2)&(hp==3))|((truth==3)&(hp==2))|(hp==4)|(hp==0)|(hp==1))}
        for kind, mask in type_masks.items():
            if mask.any(): rescue_types.append({"image_id": image_id, "type": kind, "pixels": int(mask.sum()), "components": int(ndimage.label(mask)[1]), "image_present": 1})
        for cls in (2,3):
            class_rescue=rescue&(truth==cls)
            if not class_rescue.any() or cls not in current_w: continue
            predicted=hp[class_rescue]; f_gt=h["scores"][cls][class_rescue]; f_pred=np.asarray([h["scores"][p,y,x] for p,(y,x) in zip(predicted,np.argwhere(class_rescue))]); rivals=np.max(np.delete(h["scores"],cls,axis=0),axis=0)[class_rescue]
            gt_grid=np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((h["basis"].shape[-1],h["basis"].shape[-2]),Image.Resampling.NEAREST)); rescue_grid=np.asarray(Image.fromarray(class_rescue.astype(np.uint8)).resize((h["basis"].shape[-1],h["basis"].shape[-2]),Image.Resampling.NEAREST)).astype(bool); target_grid=gt_grid==cls; rival_grid=(gt_grid<4)&(gt_grid!=cls); mass=h["basis"].sum((1,2))+EPS; purity=(h["basis"]*target_grid).sum((1,2))/mass; coverage=(h["basis"]*target_grid).sum((1,2))/(target_grid.sum()+EPS); oracle_order=np.argsort(-(purity*coverage)); actual_order=np.argsort(-h["weights"][:,cls]); bmax=h["basis"].max(0); actual5=h["basis"][actual_order[:5]].max(0); oracle5=h["basis"][oracle_order[:5]].max(0)
            rescue_evidence.append({"image_id":image_id,"class":cls,"pixels":int(class_rescue.sum()),"F_GT":float(f_gt.mean()),"F_pred":float(f_pred.mean()),"strongest_rival_score":float(rivals.mean()),"semantic_margin":float((f_gt-rivals).mean()),"Bmax_GT":float(bmax[rescue_grid].mean()) if rescue_grid.any() else 0.,"actual_top5_union_GT_support":float(actual5[rescue_grid].mean()) if rescue_grid.any() else 0.,"oracle_top5_union_GT_support":float(oracle5[rescue_grid].mean()) if rescue_grid.any() else 0.,"weight_top5":json.dumps(h["weights"][actual_order[:5],cls].tolist()),"weight_entropy":float(-(h["weights"][:,cls]*np.log(np.clip(h["weights"][:,cls],EPS,None))).sum()),"oracle_correctable_fraction":float(np.mean(oracle5[rescue_grid]>=.5)) if rescue_grid.any() else 0.})
        mutual = type_masks["R1_2to3"] | type_masks["R2_3to2"]
        if mutual.any():
            signed = h["scores"][2]-h["scores"][3]; margins = np.abs(signed[mutual]); rescue_margin.append({"image_id": image_id, "pixels": int(mutual.sum()), "mean_abs_margin": float(margins.mean()), "fraction_lt_005": float(np.mean(margins<.05)), "fraction_lt_010": float(np.mean(margins<.10))})
        candidate_rows.append({"image_id": image_id, "delta_iou": hm-sm, "confusion_2to3": ratio(row["hqmr_2to3"], row["GT_area_c2"]), "confusion_3to2": ratio(row["hqmr_3to2"], row["GT_area_c3"]), "background_fn": 0.0})
        pairs.append(row)
        if index % 100 == 0 or index == len(loader): print(f"HQMR_CLASS23_AUDIT_PROGRESS={index}/{len(loader)}", flush=True)
    hist = {name: np.stack(value) for name,value in hists.items()}; metrics = {name:scores_from_confusion(value.sum(0)) for name,value in hist.items()}; matrices = {name:np.stack(value) for name,value in matrix_by_image.items()}
    reproduction = {"status":"PASS", "images":len(loader), "hqmr":metrics["hqmr"], "sshr":metrics["sshr"], "checks":{}}
    for name, expected in (("hqmr",EXPECTED_HQMR),("sshr",EXPECTED_SSHR)):
        reproduction["checks"][name] = all(abs(metrics[name]["class_iou"][str(c)]-expected[c]) < 5e-5 for c in range(4))
    reproduction["checks"]["mIoU"] = abs(metrics["hqmr"]["mIoU"]-.6557244403737567)<1e-12 and abs(metrics["sshr"]["mIoU"]-.6669670591172749)<1e-12
    if not all(reproduction["checks"].values()): reproduction["status"]="FAIL"; reproduction["decision"]="HQMR_RESIDUAL_AUDIT_ENGINEERING_BLOCKED"
    write_json(output/"provenance/reproduction_gate.json",reproduction)
    if reproduction["status"]!="PASS": raise AssertionError(reproduction)
    labels=["class0","class1","class2","class3","background"]
    for name in ("sshr","hqmr"):
        raw=matrices[name].sum(0); write_csv(output/f"confusion/{name}_confusion_matrix.csv",[{"GT":labels[i],**{labels[j]:int(raw[i,j]) for j in range(5)}} for i in range(5)]); write_csv(output/f"confusion/{name}_confusion_row_normalized.csv",[{"GT":labels[i],**{labels[j]:normalized_matrix(raw,1)[i,j] for j in range(5)}} for i in range(5)]); write_csv(output/f"confusion/{name}_confusion_column_normalized.csv",[{"GT":labels[i],**{labels[j]:normalized_matrix(raw,0)[i,j] for j in range(5)}} for i in range(5)])
    delta=matrices["hqmr"].sum(0)-matrices["sshr"].sum(0); write_csv(output/"confusion/confusion_delta.csv",[{"GT":labels[i],**{labels[j]:int(delta[i,j]) for j in range(5)}} for i in range(5)])
    target_df=pd.DataFrame(targeted); pair_for_boot=pd.DataFrame(pairs); denominator=(pair_for_boot.GT_area_c2+pair_for_boot.GT_area_c3).clip(lower=1)
    combined=pd.DataFrame({"image_id":pair_for_boot.image_id,
        "mutual_delta":((pair_for_boot.hqmr_2to3+pair_for_boot.hqmr_3to2)-(pair_for_boot.sshr_2to3+pair_for_boot.sshr_3to2))/denominator,
        "background_delta":((pair_for_boot.hqmr_2toBG+pair_for_boot.hqmr_3toBG)-(pair_for_boot.sshr_2toBG+pair_for_boot.sshr_3toBG))/denominator,
        "to01_delta":((pair_for_boot.hqmr_2to01+pair_for_boot.hqmr_3to01)-(pair_for_boot.sshr_2to01+pair_for_boot.sshr_3to01))/denominator})
    target_boot={key:bootstrap_delta(combined[key]) for key in ("mutual_delta","background_delta","to01_delta")}; target_boot["per_class"]={str(cls):{key:bootstrap_delta(group[key]) for key in ("mutual_delta","background_delta","to01_delta")} for cls,group in target_df.groupby("class")}; write_json(output/"confusion/class23_confusion_bootstrap.json",target_boot)
    fpfn=[]
    for cls in range(5):
        for name in ("sshr","hqmr"): fpfn.append({"class":cls,"model":name,**class_metrics_5(matrices[name].sum(0),cls)})
    write_csv(output/"fp_fn/per_class_fp_fn.csv",fpfn); write_csv(output/"fp_fn/class23_background_fn.csv",[{"class":cls,"model":name,"pixels":int(matrices[name].sum(0)[cls,4]),"rate":ratio(matrices[name].sum(0)[cls,4],matrices[name].sum(0)[cls].sum())} for cls in (2,3) for name in ("sshr","hqmr")])
    attr=[]
    for label,key in (("2_to_3","2to3"),("3_to_2","3to2"),("2_to_BG","2toBG"),("3_to_BG","3toBG"),("2_to_01","2to01"),("3_to_01","3to01")):
        attr.append({"category":label,"hqmr_pixels":int(pd.DataFrame(pairs)[f"hqmr_{key}"].sum()),"sshr_pixels":int(pd.DataFrame(pairs)[f"sshr_{key}"].sum()),"excess_pixels":int(pd.DataFrame(pairs)[f"hqmr_{key}"].sum()-pd.DataFrame(pairs)[f"sshr_{key}"].sum())})
    for label,selector in (("BG_to_23",lambda value:int(value[4,2]+value[4,3])),("01_to_23",lambda value:int(value[0:2,2:4].sum()))):
        hv,sv=selector(matrices["hqmr"].sum(0)),selector(matrices["sshr"].sum(0)); attr.append({"category":label,"hqmr_pixels":hv,"sshr_pixels":sv,"excess_pixels":hv-sv})
    write_csv(output/"fp_fn/error_attribution.csv",attr)
    for path,data in (("basis/per_query_basis_quality.csv",per_query),("basis/weighted_basis_quality.csv",weighted),("basis/top_tail_basis_quality.csv",tails),("basis/oracle_basis_quality.csv",[r for r in topk if r["ranking"]=="oracle"]),("weighting/actual_vs_oracle_topk.csv",topk),("weighting/class_weight_js.csv",separation),("weighting/class_weight_topk_overlap.csv",separation),("spatial/morphology_per_class.csv",morph),("spatial/interior_boundary.csv",bands),("spatial/contact_region.csv",contacts),("spatial/interior_depth.csv",depths),("sshr_rescue/rescue23_basis_evidence.csv",rescue_evidence),("sshr_rescue/rescue23_margin.csv",rescue_margin),("paired/paired_image_table.csv",pairs)): write_csv(output/path,data)
    component_df=pd.DataFrame(components)
    if len(component_df):
        # Rank-based bins remain well-defined even when many components share an area.
        component_df["size_tertile"]=component_df.groupby("class")["area"].transform(
            lambda x:pd.cut(x.rank(method="first",pct=True),[0,1/3,2/3,1],labels=["small","medium","large"],include_lowest=True)
        )
        component_df.to_csv(output/"spatial/component_size.csv",index=False)
    weighted_df,pairs_df,topk_df,tail_df,sep_df,morph_df,bands_df,contact_df=pd.DataFrame(weighted),pd.DataFrame(pairs),pd.DataFrame(topk),pd.DataFrame(tails),pd.DataFrame(separation),pd.DataFrame(morph),pd.DataFrame(bands),pd.DataFrame(contacts)
    morph_image=morph_df.groupby(["image_id","model"])[["components","fragmentation_index","hole_count","compactness","area_ratio_gt"]].mean().unstack("model"); morph_delta=pd.DataFrame({"image_id":morph_image.index})
    for metric in ("components","fragmentation_index","hole_count","compactness","area_ratio_gt"): morph_delta[f"class23_{metric}_delta"]=(morph_image[(metric,"hqmr")]-morph_image[(metric,"sshr")]).to_numpy()
    pairs_df=pairs_df.merge(morph_delta,on="image_id",how="left"); pairs_df.to_csv(output/"paired/paired_image_table.csv",index=False)
    quality=weighted_df[["image_id","class","weighted_purity","weighted_rival"]].copy(); actual10=topk_df[(topk_df.k==10)&(topk_df.ranking=="actual")][["image_id","class","recall"]].rename(columns={"recall":"actual10"}); oracle10=topk_df[(topk_df.k==10)&(topk_df.ranking=="oracle")][["image_id","class","recall"]].rename(columns={"recall":"oracle10"}); quality=quality.merge(actual10,on=["image_id","class"]).merge(oracle10,on=["image_id","class"]); quality["router_gap"]=quality.oracle10-quality.actual10
    if len(component_df): component_df=component_df.merge(quality,on=["image_id","class"],how="left"); component_df.to_csv(output/"spatial/component_size.csv",index=False)
    depth_df=pd.DataFrame(depths)
    if len(depth_df): depth_df=depth_df.merge(quality,on=["image_id","class"],how="left"); depth_df.to_csv(output/"spatial/interior_depth.csv",index=False)
    wm=weighted_df.groupby("class").mean(numeric_only=True); control=wm.loc[[0,1]].mean(); class23=wm.loc[[2,3]].mean(); purity_deficit=float(control.weighted_purity-class23.weighted_purity); rival_excess=float(class23.weighted_rival-control.weighted_rival)
    actual=topk_df[(topk_df.k==10)&(topk_df.ranking=="actual")].groupby("class").mean(numeric_only=True); oracle=topk_df[(topk_df.k==10)&(topk_df.ranking=="oracle")].groupby("class").mean(numeric_only=True); actual23=float(actual.loc[[2,3]].recall.mean()); oracle23=float(oracle.loc[[2,3]].recall.mean()); gap=oracle23-actual23
    delta_conf=float(combined.mutual_delta.mean()); h1="STRONG" if delta_conf>=.015 and target_boot["mutual_delta"]["ci95"][0]>0 else "MODERATE" if delta_conf>=.0075 else "WEAK"
    excess_fn=sum(max(0,r["excess_pixels"]) for r in attr if r["category"].startswith(("2_","3_"))); bg_excess=sum(max(0,r["excess_pixels"]) for r in attr if "BG" in r["category"]); bg_contribution=ratio(bg_excess,excess_fn); h2="STRONG" if bg_contribution>=.5 and target_boot["background_delta"]["ci95"][0]>0 else "MODERATE" if bg_contribution>=.3 else "WEAK"
    h3="STRONG" if purity_deficit>=.08 or rival_excess>=.08 else "MODERATE" if purity_deficit>=.04 or rival_excess>=.04 else "WEAK"; h4="STRONG" if (oracle23>=.70 and actual23<=.50) or gap>=.20 else "WEAK"
    morph_mean=morph_df.groupby(["class","model"]).mean(numeric_only=True); properties=("components","small_component_fraction","hole_count","hole_area_fraction","perimeter_area_ratio","compactness","fragmentation_index")
    worse=[p for p in properties if all(morph_mean.loc[(c,"hqmr"),p]>morph_mean.loc[(c,"sshr"),p] for c in (2,3))]; h5="STRONG" if len(worse)>=2 and h1!="STRONG" and h2!="STRONG" else "WEAK"
    y=np.asarray(pixel_y); separability={}
    for name,score in (("sshr",np.asarray(pixel_sshr)),("hqmr",np.asarray(pixel_hqmr))):
        pred=score>=0; margin=np.abs(score); separability[name]={"AUROC":float(roc_auc_score(y,score)),"AUPRC":float(average_precision_score(y,score)),"balanced_accuracy":float(.5*(np.mean(pred[y==1])+np.mean(~pred[y==0]))),"margin_mean":float(margin.mean()),"margin_median":float(np.median(margin)),"margin_p10_p25_p50":[float(np.quantile(margin,q)) for q in (.1,.25,.5)],"fraction_abs_margin_lt_005":float(np.mean(margin<.05)),"fraction_abs_margin_lt_010":float(np.mean(margin<.10))}
    rescue_df=pd.DataFrame(rescue_types); type_summary=[]
    if len(rescue_df):
        for kind,group in rescue_df.groupby("type"): type_summary.append({"type":kind,"pixels":int(group.pixels.sum()),"pixel_fraction":float(group.pixels.sum()/rescue_df.pixels.sum()),"area_weighted_fraction":float(group.pixels.sum()/rescue_df.pixels.sum()),"images":int(group.image_id.nunique()),"image_fraction":float(group.image_id.nunique()/rescue_df.image_id.nunique()),"components":int(group.components.sum()),"component_fraction":float(group.components.sum()/rescue_df.components.sum())})
    write_csv(output/"sshr_rescue/rescue23_type_summary.csv",type_summary)
    band3=bands_df[bands_df.radius==3]; ib={name:float((band3[f"hqmr_{name}"]-band3[f"sshr_{name}"]).mean()) for name in ("interior_FN","boundary_FN","boundary_FP")}; contact_summary={d:float((contact_df[contact_df.distance==d].hqmr_confusion-contact_df[contact_df.distance==d].sshr_confusion).mean()) for d in (1,3,5)} if len(contact_df) else {}
    tail_summary=tail_df.groupby(["class","group"])[["purity","rival","background","coverage"]].mean().reset_index().to_dict("records"); actual_oracle={"actual_top10_recall_class23":actual23,"oracle_top10_recall_class23":oracle23,"gap":gap}; weight_summary=sep_df.groupby("pair")[["js","top5_overlap","top10_overlap","spearman","mean_abs_difference"]].mean().to_dict("index")
    correlation_features=pairs_df.copy(); correlation_features["mutual_confusion"]=(correlation_features.hqmr_2to3+correlation_features.hqmr_3to2)/(correlation_features.GT_area_c2+correlation_features.GT_area_c3).clip(lower=1); correlation_features["background_fn"]=0.; correlation_features["weighted_purity"]=(correlation_features.get("c2_purity",0)+correlation_features.get("c3_purity",0))/2; correlation_features["rival_mass"]=(correlation_features.get("c2_rival",0)+correlation_features.get("c3_rival",0))/2; correlation_features["router_gap"]=(correlation_features.get("c2_router_gap",0)+correlation_features.get("c3_router_gap",0))/2
    corr=[]; boot={}
    for key in ("mutual_confusion","background_fn","weighted_purity","rival_mass","router_gap"):
        value=correlation_ci(correlation_features[key],correlation_features.DeltaIoU,seed=BOOTSTRAP_SEED,resamples=BOOTSTRAP_RESAMPLES); corr.append({"metric":key,**value}); boot[key]=value
    write_csv(output/"correlation/residual_correlations.csv",corr); write_json(output/"correlation/residual_bootstrap.json",boot)
    hypotheses={"H1":{"result":h1,"delta":delta_conf,"ci95":target_boot["mutual_delta"]["ci95"]},"H2":{"result":h2,"contribution":bg_contribution},"H3":{"result":h3,"purity_deficit":purity_deficit,"rival_excess":rival_excess},"H4":{"result":h4,"gap_recall":gap,"actual10":actual23,"oracle10":oracle23},"H5":{"result":h5,"worse_properties":worse}}
    strong=[k for k,v in hypotheses.items() if v["result"]=="STRONG"]
    mapping={"H1":"CLASS23_INTERCLASS_CONFUSION","H2":"CLASS23_BACKGROUND_UNDERCALL","H3":"CLASS23_BASIS_PURITY_LIMIT","H4":"CLASS23_QUERY_CLASS_COUPLING_LIMIT","H5":"CLASS23_SPATIAL_MORPHOLOGY_LIMIT"}
    decision=mapping[strong[0]] if len(strong)==1 else "MIXED_CLASS23_BOTTLENECK" if len(strong)>1 else "NO_SINGLE_RESIDUAL_BOTTLENECK"; confidence="HIGH" if (len(strong)>=1 and sum([h1!="WEAK",h2!="WEAK",h3!="WEAK",h4!="WEAK",h5!="WEAK"])>=3) else "MEDIUM" if strong else "LOW"
    ranked=sorted((("interclass confusion",delta_conf),("background under-call",bg_contribution),("basis purity",max(purity_deficit,rival_excess)),("query-class coupling",gap),("spatial morphology",len(worse)/len(properties))),key=lambda x:x[1],reverse=True); primary=ranked[0][0]
    target_map={"interclass confusion":"class-discriminative query-mask separation and cross-class evidence decoupling","background under-call":"class2/3 foreground-confidence recovery without rival leakage","basis purity":"HQMR basis semantic purity and class2/3 rival-mass suppression","query-class coupling":"the mapping from class-conditioned responsibility to the HQMR basis bank","spatial morphology":"a minimal morphology/coherence correction specific to the observed phenotype"}; next_target=target_map[primary]
    because=f"HQMR-v1's remaining gap to SSHR is dominated by {primary}; therefore the next model should modify {next_target} while preserving CCRA and the validated H5→H4 HQMR reconstruction path."
    matrix={"H1 Class2/3 inter-class confusion":{"result":h1,"confidence":"HIGH" if h1=="STRONG" else "MEDIUM" if h1=="MODERATE" else "LOW","evidence":f"DeltaConf23={delta_conf:+.4f}, CI={target_boot['mutual_delta']['ci95']}"},"H2 Background under-call":{"result":h2,"confidence":"HIGH" if h2=="STRONG" else "LOW","evidence":f"contribution={bg_contribution:.4f}; background output absent by frozen protocol"},"H3 Basis purity limit":{"result":h3,"confidence":"HIGH" if h3=="STRONG" else "MEDIUM" if h3=="MODERATE" else "LOW","evidence":f"purity deficit={purity_deficit:.4f}, rival excess={rival_excess:.4f}"},"H4 Query-class coupling limit":{"result":h4,"confidence":"HIGH" if h4=="STRONG" else "LOW","evidence":f"actual10={actual23:.4f}, oracle10={oracle23:.4f}, gap={gap:.4f}"},"H5 Spatial morphology limit":{"result":h5,"confidence":"HIGH" if h5=="STRONG" else "LOW","evidence":f"same-direction worse properties={worse}"}}
    candidates=pd.DataFrame(candidate_rows); selections={"losses":candidates.nsmallest(5,"delta_iou").image_id.tolist(),"wins":candidates.nlargest(5,"delta_iou").image_id.tolist(),"confusion_2to3":candidates.nlargest(5,"confusion_2to3").image_id.tolist(),"confusion_3to2":candidates.nlargest(5,"confusion_3to2").image_id.tolist(),"background_fn":candidates.nlargest(5,"background_fn").image_id.tolist()}; write_json(output/"visualizations/selection.json",selections); render_selected(loader,selections,valroot,output,sshr,hqmr)
    result={"decision":decision,"confidence":confidence,"reproduction_gate":reproduction,"per_class_gap_pp":{str(c):100*(metrics["hqmr"]["class_iou"][str(c)]-metrics["sshr"]["class_iou"][str(c)]) for c in range(4)},"hypotheses":hypotheses,"decision_matrix":matrix,"error_attribution":attr,"top_tail_summary":tail_summary,"actual_oracle_summary":actual_oracle,"weight_separability":weight_summary,"pixel_separability":separability,"interior_boundary_summary":ib,"contact_summary":contact_summary,"rescue23":{"pixels":int(rescue_df.pixels.sum()) if len(rescue_df) else 0,"images":int(rescue_df.image_id.nunique()) if len(rescue_df) else 0,"types":type_summary,"margin":pd.DataFrame(rescue_margin).mean(numeric_only=True).to_dict() if rescue_margin else {}},"ranked_bottlenecks":ranked,"because_sentence":because,"next_target":next_target,"cp_hqmr_archived":True,"zero_training":True}
    write_json(output/"audit_result.json",result); report=output/"report/HQMR_v1_Class23_Residual_Failure_Audit_Report.md"; report.write_text(report_text(result),encoding="utf-8"); print(json.dumps({"decision":decision,"confidence":confidence,"report":str(report),"ranked":ranked},indent=2)); print(f"DECISION = {decision}"); print(f"CONFIDENCE = {confidence}")


if __name__ == "__main__": main()
