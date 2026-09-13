#!/usr/bin/env python3
"""Frozen-E25 evaluation and causal ablations for CCRA + HQMR-v1 + CCBP."""
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
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.ccbp_net import CCBPNet
from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_gcqm_full25_failure_anatomy import boundary_band, mask_morphology
from tools.eval_gcqm_full25_bcss_seed42 import TTA, _predict_sshr, foreground_confusion
from tools.eval_gcqm_full25_bcss_seed42 import load_state, normalize_cam, paired_bootstrap
from tools.eval_gcqm_full25_bcss_seed42 import prediction_from_cam, presence, resize_unflip, scores_from_confusion
from tools.hqrf_phase0_io import sha256, write_csv, write_json


SSHR_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260913, 10_000
MODES = {"A_full": "full", "B_off": "off", "C_raw_space": "raw_space",
         "D_no_g_prior": "no_g_prior", "E_mean_rival": "mean_rival",
         "F_hard_gate": "hard_gate"}
REFERENCE_PURITY_DEFICIT = .11880401016551179
REFERENCE_RIVAL_EXCESS = .12122298336331072


def _mean(values):
    return float(np.mean(values)) if values else 0.0


@torch.no_grad()
def infer_ccbp(model, image, original_hw):
    views = {name: [] for name in MODES}; probabilities = []
    bases, weights, gates, logits = [], [], [], []
    purified_by_mode = {name: [] for name in MODES}
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, ccbp_mode="full")
        decoded = output["stages"][2]["hqmr"]
        for name, mode in MODES.items():
            payload = decoded["ccbp"] if mode == "full" else model.ccbp(
                decoded["basis"], decoded["weights"], decoded["query4"], decoded["key4"],
                output["deep_gate"], dummy.bool(), mode=mode)
            views[name].append(resize_unflip(payload["mixture"], original_hw, cam_flip))
            pur = payload["purified_basis"][0]
            if cam_flip:
                pur = torch.flip(pur, dims=tuple(dimension + 1 for dimension in cam_flip))
            purified_by_mode[name].append(pur.float().cpu())
        base = decoded["basis"][0]
        if cam_flip:
            base = torch.flip(base, dims=cam_flip)
        gate = decoded["ccbp"]["gate"][0]; logit = decoded["ccbp"]["logits"][0]
        if cam_flip:
            gate = torch.flip(gate, dims=cam_flip); logit = torch.flip(logit, dims=cam_flip)
        bases.append(base.float().cpu())
        weights.append(decoded["weights"][0].float().cpu())
        gates.append(gate.float().cpu()); logits.append(logit.float().cpu())
        probabilities.append(output["deep_gate"])
    label = presence(torch.stack(probabilities).mean(0).float().cpu().numpy()[0])
    scores = {name: normalize_cam(torch.stack(items).mean(0).float().cpu().numpy()) for name, items in views.items()}
    predictions = {name: prediction_from_cam(score, label, np.empty(original_hw)) for name, score in scores.items()}
    return {"scores": scores, "predictions": predictions, "label": label,
            "basis": torch.stack(bases).mean(0).numpy(),
            "purified_basis": torch.stack(purified_by_mode["A_full"]).mean(0).numpy(),
            "purified_by_mode": {name: torch.stack(items).mean(0).numpy()
                                   for name, items in purified_by_mode.items()},
            "weights": torch.stack(weights).mean(0).numpy(),
            "gate": torch.stack(gates).mean(0).numpy(),
            "logits": torch.stack(logits).mean(0).numpy()}


@torch.no_grad()
def infer_hqmr(model, image, original_hw):
    views, gates, bases, weights = [], [], [], []; dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
        views.append(resize_unflip(output["primary_output"], original_hw, cam_flip)); gates.append(output["deep_gate"])
        decoded = output["stages"][2]["hqmr"]; basis = decoded["basis"][0]
        if cam_flip: basis = torch.flip(basis, dims=cam_flip)
        bases.append(basis.float().cpu()); weights.append(decoded["weights"][0].float().cpu())
    score = normalize_cam(torch.stack(views).mean(0).float().cpu().numpy())
    label = presence(torch.stack(gates).mean(0).float().cpu().numpy()[0])
    return {"scores": score, "prediction": prediction_from_cam(score, label, np.empty(original_hw)),
            "basis": torch.stack(bases).mean(0).numpy(), "weights": torch.stack(weights).mean(0).numpy()}


def purity_rows(image_id, truth, bundle):
    basis, weights = bundle["purified_basis"], bundle["weights"]
    gt = np.asarray(Image.fromarray(truth.astype(np.uint8)).resize((basis.shape[-1], basis.shape[-2]), Image.Resampling.NEAREST))
    weighted_rows, tail_rows = [], []
    for cls in range(4):
        target, rival, background = gt == cls, (gt < 4) & (gt != cls), gt == 4
        if not target.any(): continue
        value = basis[:, cls]; mass = value.sum((1, 2)) + 1e-8
        purity = (value * target).sum((1, 2)) / mass
        rival_mass = (value * rival).sum((1, 2)) / mass
        background_mass = (value * background).sum((1, 2)) / mass
        coverage = (value * target).sum((1, 2)) / target.sum()
        w = weights[:, cls] / max(weights[:, cls].sum(), 1e-8)
        weighted_rows.append({"image_id": image_id, "class": cls,
            "weighted_purity": float(np.sum(w * purity)), "weighted_rival": float(np.sum(w * rival_mass)),
            "weighted_background": float(np.sum(w * background_mass)), "weighted_coverage": float(np.sum(w * coverage))})
        order = np.argsort(-weights[:, cls], kind="stable")
        for group, chosen in (("top10pct", order[:20]), ("middle40pct", order[20:98]), ("bottom50pct", order[98:])):
            tail_rows.append({"image_id": image_id, "class": cls, "group": group,
                "purity": float(purity[chosen].mean()), "rival": float(rival_mass[chosen].mean()),
                "background": float(background_mass[chosen].mean()), "coverage": float(coverage[chosen].mean())})
    return weighted_rows, tail_rows


def morphology_rows(image_id, truth, predictions):
    rows, bands = [], []
    for cls in (2, 3):
        target = truth == cls
        if not target.any(): continue
        for name, prediction in predictions.items():
            rows.append({"image_id": image_id, "class": cls, "model": name,
                         **mask_morphology(prediction == cls, int(target.sum()))})
        for radius in (1, 3, 5):
            boundary = boundary_band(target, radius); interior = ndimage.binary_erosion(target, iterations=radius)
            for name, prediction in predictions.items():
                bands.append({"image_id": image_id, "class": cls, "model": name, "radius": radius,
                    "interior_FN": float(np.mean(prediction[interior] != cls)) if interior.any() else 0.0,
                    "boundary_FN": float(np.mean(prediction[boundary & target] != cls)) if (boundary & target).any() else 0.0,
                    "boundary_FP": float(np.mean(prediction[boundary & ~target] == cls)) if (boundary & ~target).any() else 0.0})
    return rows, bands


def attribution(matrix):
    return {"2_to_3": int(matrix[2, 3]), "3_to_2": int(matrix[3, 2]),
            "2_3_to_01": int(matrix[2:4, 0:2].sum()), "BG_to_23": int(matrix[4, 2:4].sum()),
            "01_to_23": int(matrix[0:2, 2:4].sum())}


def confusion5(truth, prediction):
    valid = (truth >= 0) & (truth <= 4)
    return np.bincount(5 * truth[valid].astype(np.int64) + prediction[valid].astype(np.int64), minlength=25).reshape(5, 5)


@torch.no_grad()
def render_selected(loader, selections, valroot, experiment, sshr, hqmr, ccbp):
    groups = defaultdict(list)
    for group, ids in selections.items():
        folder = experiment / "visualizations" / group; folder.mkdir(parents=True, exist_ok=True)
        for rank, image_id in enumerate(ids, 1): groups[image_id].append((group, rank))
    remaining = set(groups); query_folder = experiment / "visualizations/query_views"; query_folder.mkdir(exist_ok=True)
    for batch_names, image in loader:
        image_id = batch_names[0]
        if image_id not in remaining: continue
        original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB"))
        truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        sp = _predict_sshr(sshr, image, original); hp = infer_hqmr(hqmr, image, truth.shape)["prediction"]
        bundle = infer_ccbp(ccbp, image, truth.shape); cp = bundle["predictions"]["A_full"]
        winner = bundle["logits"].argmax(0); margin = bundle["scores"]["A_full"][2] - bundle["scores"]["A_full"][3]
        panels = [(original, "input", None), (truth, "GT", "tab10"), (sp, "SSHR", "tab10"),
                  (hp, "HQMR-v1", "tab10"), (cp, "CCBP", "tab10"),
                  (bundle["scores"]["B_off"].max(0), "F_base max", "magma"),
                  (bundle["scores"]["A_full"].max(0), "F_pur max", "magma"),
                  (winner, "winner map", "tab10"), (bundle["gate"][2], "gate C2", "viridis"),
                  (bundle["gate"][3], "gate C3", "viridis"), (margin, "F2-F3 margin", "coolwarm"),
                  ((cp != truth).astype(float), "CCBP error", "Reds")]
        fig, axes = plt.subplots(3, 4, figsize=(15, 11))
        for axis, (value, title, cmap) in zip(axes.flat, panels): axis.imshow(value, cmap=cmap); axis.set_title(title); axis.axis("off")
        fig.suptitle(image_id); fig.tight_layout()
        for group, rank in groups[image_id]: fig.savefig(experiment / f"visualizations/{group}/{rank:02d}_{image_id}.png", dpi=140)
        plt.close(fig)
        weights, basis, pur = bundle["weights"], bundle["basis"], bundle["purified_basis"]
        qfig, qaxes = plt.subplots(6, 5, figsize=(15, 17))
        for block, cls in enumerate((2, 3)):
            chosen = np.argsort(-weights[:, cls], kind="stable")[:5]
            for column, query in enumerate(chosen):
                for offset, (value, title) in enumerate(((basis[query], "B"), (pur[query, 2], "Bpur2"), (pur[query, 3], "Bpur3"))):
                    axis = qaxes[3 * block + offset, column]; axis.imshow(value, cmap="viridis", vmin=0, vmax=1)
                    axis.set_title(f"c{cls} q{query} {title}\nw2={weights[query,2]:.3f} w3={weights[query,3]:.3f}"); axis.axis("off")
        qfig.suptitle(image_id); qfig.tight_layout(); qfig.savefig(query_folder / f"{image_id}.png", dpi=140); plt.close(qfig)
        remaining.remove(image_id)
        if not remaining: break
    if remaining: raise AssertionError(f"Missing visualization IDs: {sorted(remaining)}")


def decide(delta_sshr, lower_sshr, delta_hqmr, class_sshr, class_hqmr, purity, rival,
           over_suppression, full_minus_off):
    if delta_hqmr <= -.30 or over_suppression or purity["worsened"] or rival["worsened"] or full_minus_off <= -.20:
        return "CCBP_FULL25_NOGO"
    if delta_sshr >= .50 and lower_sshr > 0 and purity["passed"] and rival["passed"] and min(class_sshr.values()) >= -1.0 and full_minus_off >= .20:
        return "CCBP_FULL25_STRONG_GO"
    if delta_sshr >= .30 and lower_sshr > 0 and (purity["passed"] or rival["passed"]) and min(class_hqmr["2"], class_hqmr["3"]) > 0 and min(class_sshr.values()) > -3.0 and full_minus_off > 0:
        return "CCBP_FULL25_GO"
    if delta_sshr > 0 and delta_hqmr >= .30 and (purity["passed"] or rival["passed"]):
        return "CCBP_FULL25_BREAKTHROUGH_UNCERTAIN"
    if delta_hqmr >= .50 and purity["passed"] and rival["passed"] and delta_sshr < -.30:
        return "CCBP_FULL25_PURITY_GO"
    if -.30 < delta_hqmr < .30 and not purity["passed"] and not rival["passed"]:
        return "CCBP_FULL25_NEUTRAL"
    return "CCBP_FULL25_NOGO"


def report_text(result):
    metrics, delta, mechanism = result["metrics"], result["deltas_pp"], result["mechanism"]
    sections = [
        ("Executive Decision", f"**DECISION = {result['decision']}**。CCBP−SSHR={delta['vs_sshr']:+.4f} pp；CCBP−HQMR-v1={delta['vs_hqmr']:+.4f} pp。"),
        ("Frozen Residual-Audit Evidence", "Primary=basis purity；Secondary=spatial morphology；Tertiary=query-class coupling。"),
        ("Why Basis Purity Is the Target", "HQMR-v1 Class2/3 purity deficit=0.1188，rival excess=0.1212；router gap与直接2↔3 confusion均弱。"),
        ("CCBP Architecture", "仅在 Stage3 final HQMR basis 后执行 class-competitive suppress-only purification。"),
        ("Class Query Prototype", "qbar_c=Σ_i w_ic·stopgrad(q_i^disc)，w 保持 detached。"),
        ("Purifier Semantic Space", "复用 HQMR H4 query4/key4；P_q/P_z identity init，不新建 decoder tower。"),
        ("Rival Competition", "t_c=gamma·cosine+log(g_c+eps)，r_c 为最强非自身 foreground class。"),
        ("Suppress-only Gate", "u_c=exp(-ReLU(r_c−t_c))，0<u≤1，无阈值、top-k或扩张。"),
        ("Class-conditioned Purified Basis", "B_pur_ic=B_i·stopgrad(u_c)，F_pur_c=Σ_i w_ic B_pur_ic。"),
        ("Balanced Weak Purifier Loss", "仅训练集 tri-state reliable foreground，按 class 分别平均 CE；background/uncertain 忽略。"),
        ("Gradient Isolation", str(result["gradient_contract"])),
        ("Engineering Validation", str(result["tests"])),
        ("Pre-run Viability", str(result["viability"])),
        ("Fresh Full25 Protocol", "BCSS / Seed42 / fresh official MXNet init / 25 epochs / 29275 steps / effective batch20 / BF16。"),
        ("Training Completion", str(result["training"])),
        ("E25 Seal", f"checkpoint SHA256 `{result['provenance']['ccbp_sha256']}`；封存后才访问 validation。"),
        ("Main mIoU/mDice", f"SSHR={100*metrics['sshr']['mIoU']:.4f}/{100*metrics['sshr']['mDice']:.4f}；HQMR-v1={100*metrics['hqmr']['mIoU']:.4f}/{100*metrics['hqmr']['mDice']:.4f}；CCBP={100*metrics['A_full']['mIoU']:.4f}/{100*metrics['A_full']['mDice']:.4f}。"),
        ("Comparison vs SSHR", f"Δ={delta['vs_sshr']:+.4f} pp；paired 95% CI={result['bootstrap_vs_sshr']['miou_ci95_pp']}。"),
        ("Comparison vs HQMR-v1", f"Δ={delta['vs_hqmr']:+.4f} pp；paired 95% CI={result['bootstrap_vs_hqmr']['miou_ci95_pp']}。"),
        ("Per-Class Results", str(result["per_class"])),
        ("Paired Bootstrap", "10,000 resamples，seed=20260913；仅固定 E25 paired validation。"),
        ("Basis Purity Re-audit", str(mechanism["purity"])),
        ("Rival Mass Re-audit", str(mechanism["rival"])),
        ("Coverage/Recall Safety", str(mechanism["recall_safety"])),
        ("Error Attribution", str(mechanism["error_attribution"])),
        ("Class2/3 Separability", str(mechanism["separability"])),
        ("Morphology Secondary Re-audit", str(mechanism["morphology"])),
        ("Same-Checkpoint Causal Ablations", str(result["ablation"])),
        ("Complexity", str(result["complexity"])),
        ("Qualitative Cases", "自动选择的 gain/regression/rival-reduction/over-suppression case 位于 visualizations/。"),
        ("Scientific Interpretation", result["interpretation"]),
        ("Exact Decision", f"`DECISION = {result['decision']}`"),
        ("Next Step", result["next_step"]),
    ]
    return "# CCRA + HQMR-v1 + CCBP BCSS Seed42 Full25 Final Validation Report\n\n" + "\n\n".join(
        f"## {index} {title}\n\n{body}" for index, (title, body) in enumerate(sections, 1)) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True); parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--ccbp-checkpoint", required=True); parser.add_argument("--hqmr-checkpoint", required=True)
    parser.add_argument("--sshr-checkpoint", required=True); parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); valroot, experiment = Path(args.val_root).resolve(), Path(args.experiment_dir).resolve()
    cpath, hpath, spath = map(lambda value: Path(value).resolve(), (args.ccbp_checkpoint, args.hqmr_checkpoint, args.sshr_checkpoint))
    if sha256(hpath) != HQMR_SHA256 or sha256(spath) != SSHR_SHA256: raise AssertionError("Frozen comparator mismatch")
    if (experiment / "report/CCRA_HQMR_CCBP_BCSS_Seed42_Full25_Final_Validation_Report.md").exists(): raise FileExistsError("Final report already exists")
    runtime = json.loads((experiment / "provenance/ccbp_runtime.json").read_text())
    seal = json.loads((experiment / "checkpoints/ccbp_epoch25_final.json").read_text())
    if runtime["steps"] != 29275 or runtime["validation_accessed"] or seal["sha256"] != sha256(cpath) or not seal["sealed_before_segmentation_evaluation"]:
        raise AssertionError("CCBP E25 seal/protocol mismatch")
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if len(loader) != 3418: raise AssertionError("Expected 3418 validation images")
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(spath), strict=True); sshr.eval()
    hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(hpath), strict=True); hqmr.eval()
    ccbp = CCBPNet().cuda(); ccbp.load_state_dict(load_state(cpath), strict=True); ccbp.eval()
    names = ["sshr", "hqmr", *MODES]
    hist, matrices = {name: [] for name in names}, {name: [] for name in names}
    paired, hqmr_weighted, morphology, interior = [], [], [], []
    weighted_by_mode = {name: [] for name in MODES}; tails_by_mode = {name: [] for name in MODES}
    scores23 = {name: [] for name in MODES}; labels23, gates = [], []
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for index, (batch_names, image) in enumerate(loader, 1):
        image_id = batch_names[0]
        original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB"))
        truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        sp = _predict_sshr(sshr, image, original); hbundle = infer_hqmr(hqmr, image, truth.shape); hp = hbundle["prediction"]
        bundle = infer_ccbp(ccbp, image, truth.shape); predictions = bundle["predictions"]
        current = {"sshr": sp, "hqmr": hp, **predictions}
        row = {"image_id": image_id, "gate_mean": float(bundle["gate"].mean())}
        for name, prediction in current.items():
            confusion = foreground_confusion(truth, prediction); hist[name].append(confusion)
            matrices[name].append(confusion5(truth, prediction)); row[f"{name}_mIoU"] = scores_from_confusion(confusion)["mIoU"]
        paired.append(row)
        for name in MODES:
            mode_bundle = {"purified_basis": bundle["purified_by_mode"][name], "weights": bundle["weights"]}
            purity, tail = purity_rows(image_id, truth, mode_bundle)
            weighted_by_mode[name].extend(purity); tails_by_mode[name].extend(tail)
        hpayload = {"purified_basis": np.repeat(hbundle["basis"][:, None], 4, axis=1), "weights": hbundle["weights"]}
        hpure, _ = purity_rows(image_id, truth, hpayload); hqmr_weighted.extend(hpure)
        m, b = morphology_rows(image_id, truth, {"sshr": sp, "hqmr": hp, **predictions})
        morphology.extend(m); interior.extend(b)
        valid = (truth == 2) | (truth == 3)
        labels23.append((truth[valid] == 2).astype(np.uint8))
        for name in MODES: scores23[name].append((bundle["scores"][name][2] - bundle["scores"][name][3])[valid])
        gates.append(bundle["gate"])
        if index % 100 == 0 or index == len(loader): print(f"CCBP_EVAL_PROGRESS={index}/{len(loader)}", flush=True)
    hist = {name: np.stack(value) for name, value in hist.items()}; matrices = {name: np.stack(value) for name, value in matrices.items()}
    metrics = {name: scores_from_confusion(value.sum(0)) for name, value in hist.items()}
    if abs(metrics["sshr"]["mIoU"] - .6669670591172749) > 1e-12 or abs(metrics["hqmr"]["mIoU"] - .6557244403737567) > 1e-12:
        raise AssertionError("Comparator reproduction mismatch")
    write_json(experiment / "evaluation/ccbp_metrics.json", metrics); write_csv(experiment / "evaluation/ccbp_per_image.csv", paired)
    boot_sshr = paired_bootstrap(hist["sshr"], hist["A_full"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    boot_hqmr = paired_bootstrap(hist["hqmr"], hist["A_full"], BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    write_json(experiment / "evaluation/ccbp_vs_sshr_bootstrap.json", boot_sshr)
    write_json(experiment / "evaluation/ccbp_vs_hqmr_bootstrap.json", boot_hqmr)
    per_class = []
    for cls in range(4):
        per_class.append({"class": cls, **{f"{name}_iou": metrics[name]["class_iou"][str(cls)] for name in ("sshr", "hqmr", "A_full")},
                          "delta_vs_sshr_pp": 100 * (metrics["A_full"]["class_iou"][str(cls)] - metrics["sshr"]["class_iou"][str(cls)]),
                          "delta_vs_hqmr_pp": 100 * (metrics["A_full"]["class_iou"][str(cls)] - metrics["hqmr"]["class_iou"][str(cls)])})
    write_csv(experiment / "evaluation/ccbp_per_class.csv", per_class)
    weighted_frames = {name: pd.DataFrame(rows) for name, rows in weighted_by_mode.items()}
    tail_frames = {name: pd.DataFrame(rows) for name, rows in tails_by_mode.items()}
    weighted_df, hqmr_weighted_df, tail_df = weighted_frames["A_full"], pd.DataFrame(hqmr_weighted), tail_frames["A_full"]
    weighted_df.to_csv(experiment / "purity/ccbp_weighted_basis_quality.csv", index=False)
    tail_df.to_csv(experiment / "purity/ccbp_top_tail_basis_quality.csv", index=False)
    by_class = weighted_df.groupby("class")[["weighted_purity", "weighted_rival", "weighted_background", "weighted_coverage"]].mean()
    hqmr_by_class = hqmr_weighted_df.groupby("class")[["weighted_purity", "weighted_rival", "weighted_background", "weighted_coverage"]].mean()
    control, target = by_class.loc[[0, 1]].mean(), by_class.loc[[2, 3]].mean()
    hqmr_target = hqmr_by_class.loc[[2, 3]].mean()
    purity_deficit = float(control.weighted_purity - target.weighted_purity)
    rival_excess = float(target.weighted_rival - control.weighted_rival)
    purity_gain = float(target.weighted_purity - hqmr_target.weighted_purity)
    rival_drop = float(hqmr_target.weighted_rival - target.weighted_rival)
    purity_gate = {"new_deficit": purity_deficit, "reference_deficit": REFERENCE_PURITY_DEFICIT,
                   "class23_absolute_gain": purity_gain, "passed": purity_deficit <= .060 or purity_gain >= .060,
                   "worsened": purity_gain < 0}
    rival_gate = {"new_excess": rival_excess, "reference_excess": REFERENCE_RIVAL_EXCESS,
                  "class23_absolute_drop": rival_drop, "passed": rival_excess <= .060 or rival_drop >= .060,
                  "worsened": rival_drop < 0}
    write_csv(experiment / "purity/ccbp_rival_mass.csv", by_class.reset_index().to_dict("records"))
    recall = {name: {str(cls): float(matrices[name].sum(0)[cls, cls] / matrices[name].sum(0)[cls].sum())
                     for cls in range(4)} for name in ("sshr", "hqmr", "A_full")}
    recall_drop = {str(cls): 100 * (recall["hqmr"][str(cls)] - recall["A_full"][str(cls)]) for cls in (2, 3)}
    recall_safety = {"recall": recall, "drop_vs_hqmr_pp": recall_drop,
                     "over_suppression": max(recall_drop.values()) > 2.0}
    error_rows = []
    total = {name: matrices[name].sum(0) for name in ("sshr", "hqmr", "A_full")}
    selectors = {"2_to_3": lambda x: x[2, 3], "3_to_2": lambda x: x[3, 2],
                 "2_3_to_01": lambda x: x[2:4, 0:2].sum(), "BG_to_23": lambda x: x[4, 2:4].sum(),
                 "01_to_23": lambda x: x[0:2, 2:4].sum()}
    for category, selector in selectors.items():
        error_rows.append({"category": category, **{name: int(selector(matrix)) for name, matrix in total.items()}})
    write_csv(experiment / "purity/ccbp_error_attribution.csv", error_rows)
    old_excess = max(0, selectors["BG_to_23"](total["hqmr"]) - selectors["BG_to_23"](total["sshr"])) + max(0, selectors["01_to_23"](total["hqmr"]) - selectors["01_to_23"](total["sshr"]))
    new_excess = max(0, selectors["BG_to_23"](total["A_full"]) - selectors["BG_to_23"](total["sshr"])) + max(0, selectors["01_to_23"](total["A_full"]) - selectors["01_to_23"](total["sshr"]))
    contamination_reduction = float((old_excess - new_excess) / max(old_excess, 1))
    labels = np.concatenate(labels23); separability_by_mode = {}
    for name in MODES:
        score = np.concatenate(scores23[name]); prediction = score >= 0; margin = np.abs(score)
        values = {"AUROC": float(roc_auc_score(labels, score)), "AUPRC": float(average_precision_score(labels, score)),
                  "balanced_accuracy": float(.5 * (np.mean(prediction[labels == 1]) + np.mean(~prediction[labels == 0]))),
                  "median_margin": float(np.median(margin)), "fraction_abs_margin_lt_005": float(np.mean(margin < .05))}
        values["targets_passed"] = values["AUROC"] >= .75 and values["balanced_accuracy"] >= .68 and values["fraction_abs_margin_lt_005"] <= .50
        separability_by_mode[name] = values
    separability = separability_by_mode["A_full"]
    write_json(experiment / "purity/ccbp_class23_separability.json", separability)
    morphology_df, interior_df = pd.DataFrame(morphology), pd.DataFrame(interior)
    morphology_df.to_csv(experiment / "morphology/ccbp_morphology.csv", index=False)
    interior_df.to_csv(experiment / "morphology/ccbp_interior_boundary.csv", index=False)
    properties = ("components", "small_component_fraction", "hole_count", "hole_area_fraction", "perimeter_area_ratio", "compactness", "fragmentation_index")
    morph_mean = morphology_df.groupby(["model", "class"])[list(properties)].mean()
    improvements = {}
    for prop in properties:
        base_gap = _mean([abs(morph_mean.loc[("hqmr", cls), prop] - morph_mean.loc[("sshr", cls), prop]) for cls in (2, 3)])
        new_gap = _mean([abs(morph_mean.loc[("A_full", cls), prop] - morph_mean.loc[("sshr", cls), prop]) for cls in (2, 3)])
        improvements[prop] = 1.0 - new_gap / max(base_gap, 1e-12)
    morphology_summary = {"gap_improvement_fraction": improvements,
                          "count_improved_at_least_25pct": sum(value >= .25 for value in improvements.values())}
    gate_array = np.stack(gates)
    gate_summary = {"mean": float(gate_array.mean()), "fraction_lt_095": float(np.mean(gate_array < .95)),
                    "fraction_lt_075": float(np.mean(gate_array < .75)), "fraction_lt_050": float(np.mean(gate_array < .50))}
    ablation, ablation_diagnostics = [], {}
    for name, mode in MODES.items():
        matrix = matrices[name].sum(0); quality = weighted_frames[name].groupby("class")[[
            "weighted_purity", "weighted_rival", "weighted_background", "weighted_coverage"]].mean()
        mode_morphology = morphology_df[morphology_df.model == name].groupby("class")[list(properties)].mean()
        fp = matrix.sum(0) - np.diag(matrix); fn = matrix.sum(1) - np.diag(matrix)
        row = {"name": name, "mode": mode, "mIoU": metrics[name]["mIoU"], "mDice": metrics[name]["mDice"],
               "delta_vs_full_pp": 100 * (metrics[name]["mIoU"] - metrics["A_full"]["mIoU"])}
        details = {"per_class_iou": metrics[name]["class_iou"],
                   "purity": quality.reset_index().to_dict("records"),
                   "recall": {str(cls): float(matrix[cls, cls] / matrix[cls].sum()) for cls in range(4)},
                   "false_positive": {str(cls): int(fp[cls]) for cls in range(4)},
                   "false_negative": {str(cls): int(fn[cls]) for cls in range(4)},
                   "separability": separability_by_mode[name],
                   "morphology_class23": mode_morphology.reset_index().to_dict("records")}
        ablation_diagnostics[name] = details
        ablation.append(row); write_csv(experiment / f"ablation/{mode}.csv", [row])
    write_json(experiment / "ablation/ccbp_same_checkpoint_diagnostics.json", ablation_diagnostics)
    full_minus_off = 100 * (metrics["A_full"]["mIoU"] - metrics["B_off"]["mIoU"])
    class_sshr = {str(cls): 100 * (metrics["A_full"]["class_iou"][str(cls)] - metrics["sshr"]["class_iou"][str(cls)]) for cls in range(4)}
    class_hqmr = {str(cls): 100 * (metrics["A_full"]["class_iou"][str(cls)] - metrics["hqmr"]["class_iou"][str(cls)]) for cls in range(4)}
    delta_sshr = 100 * (metrics["A_full"]["mIoU"] - metrics["sshr"]["mIoU"])
    delta_hqmr = 100 * (metrics["A_full"]["mIoU"] - metrics["hqmr"]["mIoU"])
    verdict = decide(delta_sshr, boot_sshr["miou_ci95_pp"][0], delta_hqmr, class_sshr, class_hqmr,
                     purity_gate, rival_gate, recall_safety["over_suppression"], full_minus_off)
    smoke = json.loads((experiment / "tests/ccbp_smoke.json").read_text())
    tests = {"unit": json.loads((experiment / "tests/ccbp_unit_tests.json").read_text())["passed"],
             "regression": json.loads((experiment / "tests/ccbp_regression_tests.json").read_text())["passed"],
             "smoke": smoke.get("steps") == 2 and smoke.get("finite") is True and
                      smoke.get("no_oom") is True and smoke.get("checkpoint_written") is False}
    gradient = json.loads((experiment / "tests/ccbp_gradient_contract.json").read_text())
    viability = json.loads((experiment / "preaudit/raw_space_viability_summary.json").read_text())
    mechanism = {"purity": purity_gate, "rival": rival_gate, "recall_safety": recall_safety,
                 "error_attribution": {"rows": error_rows, "combined_excess_reduction": contamination_reduction},
                 "separability": separability, "morphology": morphology_summary, "gate": gate_summary}
    interpretation = "CCBP is supported only if performance, purity/rival recovery, and same-checkpoint CCBP-off evidence agree. "
    if verdict == "CCBP_FULL25_NOGO": interpretation += "The registered causal package did not hold; do not add depth, thresholds, propagation, or morphology repair."
    elif verdict == "CCBP_FULL25_PURITY_GO": interpretation += "Purity improved without clearing SSHR; one frozen residual audit is allowed."
    else: interpretation += "Use the exact registered decision boundary; do not tune this Seed42 run."
    next_step = {"CCBP_FULL25_STRONG_GO": "Freeze architecture; run BCSS Seed11/17 and LUAD-HistoSeg.",
                 "CCBP_FULL25_GO": "Freeze architecture; run BCSS Seed11/17 and LUAD-HistoSeg.",
                 "CCBP_FULL25_BREAKTHROUGH_UNCERTAIN": "Run multi-seed directly without tuning CCBP.",
                 "CCBP_FULL25_PURITY_GO": "Run one frozen residual audit of the secondary morphology bottleneck.",
                 "CCBP_FULL25_NEUTRAL": "Classify inactivity versus semantic-space failure; do not tune thresholds.",
                 "CCBP_FULL25_NOGO": "Archive CCBP and classify failure mode A-E; no blind gamma or architecture expansion."}[verdict]
    paired_df = pd.DataFrame(paired)
    new_rival = weighted_df[weighted_df["class"].isin([2, 3])].groupby("image_id").weighted_rival.mean()
    old_rival = hqmr_weighted_df[hqmr_weighted_df["class"].isin([2, 3])].groupby("image_id").weighted_rival.mean()
    paired_df = paired_df.join((old_rival - new_rival).rename("rival_reduction"), on="image_id")
    paired_df["gain_vs_hqmr"] = paired_df.A_full_mIoU - paired_df.hqmr_mIoU
    paired_df["gain_vs_sshr"] = paired_df.A_full_mIoU - paired_df.sshr_mIoU
    selections = {"gain_vs_hqmr": paired_df.nlargest(5, "gain_vs_hqmr").image_id.tolist(),
                  "gain_vs_sshr": paired_df.nlargest(5, "gain_vs_sshr").image_id.tolist(),
                  "regressions": paired_df.nsmallest(5, "gain_vs_hqmr").image_id.tolist(),
                  "rival_reduction": paired_df.nlargest(5, "rival_reduction").image_id.tolist(),
                  "over_suppression": paired_df.sort_values(["gate_mean", "gain_vs_hqmr"]).head(5).image_id.tolist()}
    write_json(experiment / "visualizations/selection.json", selections)
    render_selected(loader, selections, valroot, experiment, sshr, hqmr, ccbp)
    result = {"decision": verdict, "metrics": metrics,
              "deltas_pp": {"vs_sshr": delta_sshr, "vs_hqmr": delta_hqmr,
                            "class_vs_sshr": class_sshr, "class_vs_hqmr": class_hqmr},
              "bootstrap_vs_sshr": boot_sshr, "bootstrap_vs_hqmr": boot_hqmr,
              "per_class": per_class, "mechanism": mechanism, "ablation": ablation,
              "gradient_contract": gradient, "tests": tests, "viability": viability,
              "training": runtime, "complexity": {"parameters": sum(p.numel() for p in ccbp.parameters()),
                  "parameter_delta_vs_hqmr": 131073, "peak_inference_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
                  "seconds_per_image": (time.perf_counter() - started) / len(loader)},
              "provenance": {"ccbp_sha256": sha256(cpath), "hqmr_sha256": sha256(hpath),
                             "sshr_sha256": sha256(spath), "source_commit": subprocess.check_output(
                                 ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "validation_images": len(loader)},
              "interpretation": interpretation, "next_step": next_step,
              "final_sentence": "CCRA decides which queries are responsible for each class; HQMR reconstructs their tissue regions; CCBP compares each class against its strongest rival in the existing HQMR semantic space and suppresses only rival-dominated support, directly targeting the basis-purity bottleneck identified by the frozen residual audit."}
    write_json(experiment / "evaluation/ccbp_final_result.json", result)
    report = experiment / "report/CCRA_HQMR_CCBP_BCSS_Seed42_Full25_Final_Validation_Report.md"
    report.write_text(report_text(result), encoding="utf-8")
    print(json.dumps({"decision": verdict, "delta_vs_sshr_pp": delta_sshr, "delta_vs_hqmr_pp": delta_hqmr,
                      "full_minus_off_pp": full_minus_off, "purity": purity_gate, "rival": rival_gate,
                      "over_suppression": recall_safety["over_suppression"], "report": str(report)}, indent=2))
    print(f"DECISION = {verdict}")


if __name__ == "__main__":
    main()
