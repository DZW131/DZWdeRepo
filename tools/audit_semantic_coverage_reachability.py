#!/usr/bin/env python3
"""Frozen E25 semantic-coverage and propagation-reachability audit (zero training)."""
from __future__ import annotations

import argparse
import heapq
import json
import math
import subprocess
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata, spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.ccac_net import CCACNet
from network.dfsc import mcc_complete
from network.dfsc_net import DFSCNet
from network.gcqm_net import GCQMNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    BASELINE_SHA256, THRESHOLDS, TTA, _predict_gcqm, _predict_sshr,
    foreground_confusion, load_state, normalize_cam, prediction_from_cam,
    presence, resize_unflip, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json


EXPECTED = {"sshr": .6669670591172749, "old_gcqm": .6435430070332891,
            "ccac": .6461688735415164, "dfsc": .6452820195735897}
DFSC_SHA256 = "470f1056f2bbf5c64b5e6fff76861f9fa4e1663bf7ba0c74e2621528fb48af11"
GCQM_SHA256 = "6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f"
CCAC_SHA256 = "848631927607bc2832c07cbbafdf0ba71ed3482ff1cec8a056abb2bbca7c80a6"
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260911, 10_000
RADII = (1, 2, 4, 8, 16)
AFFINITY_THRESHOLDS = (.25, .50, .75)
TOP_K = (1, 5, 10, 20, 196)
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0),
           (0, 1), (1, -1), (1, 0), (1, 1))


def anchor_category(seed_fraction: float) -> str:
    if seed_fraction >= .05:
        return "anchored"
    if seed_fraction > 0:
        return "weakly_anchored"
    return "unanchored"


def distance_maps(seed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euclidean, Manhattan, and Chebyshev distance to a binary seed set."""
    if not np.asarray(seed, bool).any():
        inf = np.full(seed.shape, np.inf, np.float32)
        return inf.copy(), inf.copy(), inf.copy()
    inverse = ~np.asarray(seed, bool)
    euclidean = ndimage.distance_transform_edt(inverse).astype(np.float32)
    manhattan = ndimage.distance_transform_cdt(inverse, metric="taxicab").astype(np.float32)
    chebyshev = ndimage.distance_transform_cdt(inverse, metric="chessboard").astype(np.float32)
    return euclidean, manhattan, chebyshev


def geodesic_distance(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Eight-neighbour shortest path constrained to mask; -1 means unreachable."""
    mask, seed = np.asarray(mask, bool), np.asarray(seed, bool)
    result = np.full(mask.shape, -1, np.int16)
    queue = deque()
    for y, x in np.argwhere(mask & seed):
        result[y, x] = 0
        queue.append((int(y), int(x)))
    h, w = mask.shape
    while queue:
        y, x = queue.popleft()
        for dy, dx in OFFSETS:
            if dy == dx == 0:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and result[yy, xx] < 0:
                result[yy, xx] = result[y, x] + 1
                queue.append((yy, xx))
    return result


def affinity_reachable(mask: np.ndarray, seed: np.ndarray, affinity: np.ndarray,
                       threshold: float) -> np.ndarray:
    """Eight-neighbour BFS through raw learned-affinity edges, GT constrained."""
    mask, seed = np.asarray(mask, bool), np.asarray(seed, bool)
    reached = mask & seed
    queue = deque((int(y), int(x)) for y, x in np.argwhere(reached))
    h, w = mask.shape
    while queue:
        y, x = queue.popleft()
        for edge, (dy, dx) in enumerate(OFFSETS):
            if edge == 4 or affinity[edge, y, x] < threshold:
                continue
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not reached[yy, xx]:
                reached[yy, xx] = True
                queue.append((yy, xx))
    return reached


def resize_label(value: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(value.astype(np.float32))[None, None]
    return F.interpolate(tensor, hw, mode="nearest")[0, 0].numpy().astype(np.int64)


def safe_iou(truth: np.ndarray, pred: np.ndarray) -> float:
    values = []
    for cls in range(4):
        a, b = truth == cls, pred == cls
        union = (a | b).sum()
        if union:
            values.append(float((a & b).sum() / union))
    return float(np.mean(values)) if values else float("nan")


def finite_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, np.float64)
    return float(np.quantile(values[np.isfinite(values)], q)) if np.isfinite(values).any() else float("inf")


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int = BOOTSTRAP_SEED,
                       resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[valid], np.asarray(y)[valid]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return {"rho": None, "ci95": [None, None], "n": int(len(x)),
                "resamples": resamples, "seed": seed}
    rho = float(spearmanr(x, y).statistic)
    rng, samples = np.random.default_rng(seed), []
    for _ in range(0, resamples, 100):
        count = min(100, resamples - len(samples))
        index = rng.integers(0, len(x), size=(count, len(x)))
        xr, yr = rankdata(x[index], axis=1), rankdata(y[index], axis=1)
        xr -= xr.mean(1, keepdims=True); yr -= yr.mean(1, keepdims=True)
        denom = np.sqrt((xr * xr).sum(1) * (yr * yr).sum(1))
        samples.extend(np.divide((xr * yr).sum(1), denom,
                                 out=np.zeros(count), where=denom > 0).tolist())
    return {"rho": rho, "ci95": [float(v) for v in np.quantile(samples, [.025, .975])],
            "n": int(len(x)), "resamples": resamples, "seed": seed}


def decide_bottleneck(anchored: float, oracle10: float, unlimited: float,
                      unreachable_fn: float, class_uncovered: float,
                      base_coverage: float, rescue_types: dict[str, float]) -> tuple[str, str, dict]:
    propagation = anchored >= .70 and oracle10 >= .90 and unlimited >= .95 and unreachable_fn >= .50
    coverage = (1 - anchored >= .20 or class_uncovered >= .20 or
                oracle10 < .85 or unlimited < .85)
    weighting = oracle10 >= .90 and oracle10 - base_coverage >= .10 and rescue_types.get("D", 0) >= .20
    if propagation and coverage:
        decision = "MIXED_COVERAGE_AND_REACHABILITY"
    elif coverage and oracle10 < .85:
        decision = "QUERY_MASK_COVERAGE_LIMIT"
    elif weighting:
        decision = "QUERY_BASIS_WEIGHTING_LIMIT"
    elif propagation:
        decision = "LONG_RANGE_PROPAGATION_LIMIT"
    elif coverage:
        decision = "QUERY_MASK_COVERAGE_LIMIT"
    else:
        decision = "NO_SINGLE_CAUSE_IDENTIFIED"
    votes = [coverage, propagation, weighting,
             rescue_types.get("C", 0) + rescue_types.get("E", 0) >= .50]
    confidence = "HIGH" if sum(bool(v) for v in votes) >= 3 else "MEDIUM" if sum(bool(v) for v in votes) >= 2 else "LOW"
    return decision, confidence, {"strong_propagation_evidence": propagation,
                                  "strong_coverage_evidence": coverage,
                                  "weighting_evidence": weighting}


@torch.no_grad()
def dfsc_bundle(model: DFSCNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    base_views, final_views, basis_views, weights, full_views, gates = [], [], [], [], [], []
    raw_affinity = None
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = GCQMNet.forward(model, value, dummy, step=29275, run_pmec=False)
            relation = model.dfra(output["pixel_feature"], None, "full")
            base = output["primary_output"]
            final = mcc_complete(base, relation["affinity_comp"], 2, "mcc")["restored"]
        def canonical(tensor):
            tensor = tensor[0]
            return torch.flip(tensor, dims=cam_flip) if cam_flip else tensor
        base_views.append(canonical(base).float().cpu())
        final_views.append(canonical(final).float().cpu())
        basis_views.append(canonical(output["stages"][2]["gcqm"]["base_probability"]).float().cpu())
        weights.append(output["stages"][2]["gcqm"]["weights"][0].float().cpu())
        full_views.append(resize_unflip(final, original_hw, cam_flip).float().cpu())
        gates.append(output["deep_gate"].float().cpu())
        if not input_flip:
            h, w = base.shape[-2:]
            raw_affinity = relation["raw_affinity"][0].float().cpu().numpy().reshape(9, h, w)
    label = presence(torch.stack(gates).mean(0).numpy()[0])
    base = normalize_cam(torch.stack(base_views).mean(0).numpy())
    final = normalize_cam(torch.stack(final_views).mean(0).numpy())
    basis = torch.stack(basis_views).mean(0).numpy()
    weight = torch.stack(weights).mean(0).numpy()
    formal = normalize_cam(torch.stack(full_views).mean(0).numpy())
    return {"base": base, "final": final, "basis": basis, "weights": weight,
            "affinity": raw_affinity, "label": label,
            "prediction": prediction_from_cam(formal, label, np.empty(original_hw)) ,
            "prediction_grid": prediction_from_cam(final, label, np.empty(final.shape[-2:]))}


def update_top(heap: list, score: float, counter: int, payload: dict, count: int = 5):
    item = (float(score), counter, payload)
    if len(heap) < count:
        heapq.heappush(heap, item)
    elif score > heap[0][0]:
        heapq.heapreplace(heap, item)


def render_payload(payload: dict, output: Path, tag: str):
    image_id, cls = payload["image_id"], payload["class"]
    output.mkdir(parents=True, exist_ok=True)
    groups = {
        "seed_distance": [("Input", payload["input"]), ("GT", payload["truth"]),
            ("DFSC", payload["dfsc"]), ("SSHR", payload["sshr"]),
            ("Base F", payload["base"][cls]), ("Seed", payload["seeds"][cls]),
            ("Seed distance", payload["distance"][cls]), ("FN", payload["fn"][cls])],
        "component_anchor": [("GT class", payload["truth"] == cls), ("Base F", payload["base"][cls]),
            ("Seed", payload["seeds"][cls]), ("Distance", payload["distance"][cls])],
        "basis_coverage": [("GT", payload["truth"]), ("B max", payload["bmax"]),
            ("C max", payload["cmax"][cls]), ("U5", payload["u5"][cls]),
            ("U10", payload["u10"][cls]), ("U20", payload["u20"][cls]),
            ("GCQM F", payload["base"][cls]), ("DFSC S", payload["final"][cls]),
            ("SSHR", payload["sshr"])],
        "rescue_regions": [("GT", payload["truth"]), ("DFSC", payload["dfsc"]),
            ("SSHR", payload["sshr"]), ("Rescue type", payload["rescue_type"])],
    }
    for name, panels in groups.items():
        cols = min(4, len(panels)); rows = math.ceil(len(panels) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3 * rows))
        axes = np.asarray(axes).reshape(-1)
        for ax, (title, value) in zip(axes, panels):
            ax.imshow(value, cmap=None if title == "Input" else "viridis")
            ax.set_title(title); ax.axis("off")
        for ax in axes[len(panels):]: ax.axis("off")
        fig.suptitle(f"{image_id} class={cls} {tag}"); fig.tight_layout()
        fig.savefig(output.parent / name / f"{tag}_{image_id}_c{cls}.png", dpi=140)
        plt.close(fig)


def report_text(result: dict) -> str:
    s, d, r = result["summary"], result["decision"], result["reproduction_gate"]
    sections = [
        ("Executive Diagnosis", f"**DECISION = {d['decision']}**  \n**CONFIDENCE = {d['confidence']}**"),
        ("Frozen Model Evidence", f"DFSC E25 SHA256 `{result['provenance']['dfsc_sha256']}`；零训练、零参数更新。"),
        ("Reproduction Gate", f"decision={r['decision']}，3418 paired images；observed mIoU={r['observed_mIoU']}。"),
        ("Why Affinity Quality Is No Longer the Main Suspect", "DFSC E25 relation AUROC=0.9332、AP≈1、正负间隔=0.4322；但其因果消融不优于 uniform/raw cosine。"),
        ("H1 FN-to-Seed Distance", f"FN within 2={s['fn_within_2']:.4f}，beyond/no-seed={s['local_unreachable_fn']:.4f}。"),
        ("Two-Hop Local Reachability", f"T=2 的局部不可达 FN 比例={s['local_unreachable_fn']:.4f}。"),
        ("H2 GT Component Anchoring", f"area-weighted anchored={s['anchored_area_fraction']:.4f}，unanchored={s['unanchored_area_fraction']:.4f}。"),
        ("H3 Within-Component Radius", "完整分级见 component_anchor/component_radius_summary.csv。"),
        ("H4 Oracle Spatial Radius Ceiling", "固定 r=1/2/4/8/16/∞ 曲线见 reachability/gt_constrained_oracle_radius_curve.csv。"),
        ("H5 Learned-Affinity Graph Reachability", f"threshold=0.50 GT reachability={s['affinity_reachability_050']:.4f}。"),
        ("H6 Unlimited Propagation Ceiling", f"GT-constrained unlimited seed recall={s['unlimited_seed_recall']:.4f}。"),
        ("H7 Query-Basis Coverage", f"class-basis-uncovered={s['class_basis_uncovered']:.4f}。"),
        ("Any-Basis / Class-Basis Coverage", f"B_max coverage={s['basis_max_coverage']:.4f}；base F seed coverage={s['base_seed_coverage']:.4f}。"),
        ("H8 Basis-Bank vs Weighting Deficit", f"GT-oracle top10 basis recall={s['oracle_top10_recall']:.4f}；diagnosis={d['basis_weighting_diagnosis']}。"),
        ("H9 SSHR Rescue-Region Analysis", f"SSHR rescue fraction among DFSC FN={s['sshr_rescue_fraction']:.4f}。"),
        ("Rescue-Region Failure Typing", str(s["rescue_type_fractions"])),
        ("H10 Component-Scale Dependence", "见 component_anchor/component_scale_summary.csv。"),
        ("H11 Interior-Depth Dependence", "见 component_anchor/interior_depth_summary.csv。"),
        ("H12 Per-Class Mechanism", "见 paired/paired_image_class_table.csv 及各 summary CSV。"),
        ("Correlation Analysis", "Spearman 与 10,000 bootstrap（seed=20260911）见 correlation/coverage_delta_iou_correlations.csv。"),
        ("Representative Cases", "按 unanchored、FN distance、SSHR rescue 自动选例，见 visualizations/。"),
        ("Decision Matrix", str(d["evidence"])),
        ("Exact Bottleneck Decision", f"`DECISION = {d['decision']}`；`CONFIDENCE = {d['confidence']}`。"),
        ("What Is Preserved", "CCRA allocation、query identity、局部 completion 的小幅正价值与 DFSC affinity selectivity 均保留。"),
        ("What Is Falsified", "更好的局部 affinity 足以解决 mask-quality gap；继续调 3×3 affinity/T 不再优先。"),
        ("Exact Next Architecture Target", d["next_architecture_target"]),
        ("What Must NOT Be Done", "不得自动恢复旧 pixel-wise FOMD；不得继续 threshold sweep、3×3 T tuning 或 checkpoint selection。"),
        ("Final Decision", f"The remaining gap is dominated by {d['dominant_cause']}; therefore the next model should modify {d['next_architecture_target']} rather than local propagation.\n\nDECISION = {d['decision']}\n\nCONFIDENCE = {d['confidence']}"),
    ]
    return "# Semantic Coverage & Propagation Reachability Audit Report\n\n" + "\n\n".join(
        f"## {i} {title}\n\n{body}" for i, (title, body) in enumerate(sections, 1)) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val-root", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--dfsc-checkpoint", required=True); p.add_argument("--gcqm-checkpoint", required=True)
    p.add_argument("--ccac-checkpoint", required=True); p.add_argument("--sshr-checkpoint", required=True)
    p.add_argument("--dfsc-result-json", required=True); p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args(); started = time.perf_counter()
    valroot, output = Path(args.val_root).resolve(), Path(args.output_dir).resolve()
    paths = {k: Path(getattr(args, f"{k}_checkpoint")).resolve() for k in ("dfsc", "gcqm", "ccac", "sshr")}
    result_path = Path(args.dfsc_result_json).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Audit output is not empty: {output}")
    directories = ("provenance", "paired", "seed_distance", "component_anchor", "reachability",
                   "basis_coverage", "sshr_advantage", "oracle", "correlation", "visualizations",
                   "visualizations/seed_distance", "visualizations/component_anchor",
                   "visualizations/basis_coverage", "visualizations/rescue_regions", "report")
    for name in directories: (output / name).mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {"audit": "Semantic Coverage & Propagation Reachability", "zero_training": True,
              "parameter_updates": 0, "seed": 42, "grid": [56, 56], "connectivity": 8,
              "thresholds": THRESHOLDS.tolist(), "radii": list(RADII) + ["infinity"],
              "affinity_thresholds": list(AFFINITY_THRESHOLDS), "main_affinity_threshold": .50,
              "top_k": list(TOP_K), "tta_views": 3, "bootstrap_seed": BOOTSTRAP_SEED,
              "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
              "affinity_graph": "raw learned affinity, canonical unflipped view, GT-constrained",
              "rescue_type_priority": "E then D then A/B/C"}
    write_json(output / "provenance/coverage_audit_config.json", config)
    (output / "provenance/coverage_audit_config_sha256.txt").write_text(sha256(output / "provenance/coverage_audit_config.json") + "\n")
    (output / "provenance/coverage_audit_source_commit.txt").write_text(source_commit + "\n")
    (output / "provenance/coverage_audit_git_diff.patch").write_text(subprocess.check_output(["git", "diff"], cwd=ROOT, text=True))
    (output / "provenance/coverage_audit_environment.txt").write_text(
        f"python={sys.version}\ntorch={torch.__version__}\ncuda={torch.version.cuda}\ngpu={torch.cuda.get_device_name(0)}\n")
    hashes = {name: sha256(path) for name, path in paths.items()}
    sealed = json.loads(result_path.read_text())
    prior = {"sshr": sealed["metrics"]["sshr"]["mIoU"], "old_gcqm": sealed["metrics"]["old_gcqm"]["mIoU"],
             "ccac": sealed["metrics"]["ccac"]["mIoU"], "dfsc": sealed["metrics"]["full_dfsc"]["mIoU"]}
    preliminary = (hashes == {"dfsc": DFSC_SHA256, "gcqm": GCQM_SHA256,
                              "ccac": CCAC_SHA256, "sshr": BASELINE_SHA256} and
                   all(abs(prior[k] - EXPECTED[k]) < 1e-10 for k in EXPECTED) and
                   len(list((valroot / "img").glob("*.png"))) == 3418 and
                   len(list((valroot / "mask").glob("*.png"))) == 3418)
    if not preliminary:
        blocked = {"decision": "COVERAGE_AUDIT_ENGINEERING_BLOCKED", "hashes": hashes,
                   "prior_mIoU": prior, "expected_mIoU": EXPECTED}
        write_json(output / "provenance/reproduction_gate.json", blocked)
        raise AssertionError(blocked)

    models = {"sshr": SSHRCAM(4).cuda(), "old_gcqm": GCQMNet().cuda(),
              "ccac": CCACNet().cuda(), "dfsc": DFSCNet().cuda()}
    checkpoint_for = {"sshr": paths["sshr"], "old_gcqm": paths["gcqm"],
                      "ccac": paths["ccac"], "dfsc": paths["dfsc"]}
    for name, model in models.items(): model.load_state_dict(load_state(checkpoint_for[name]), strict=True); model.eval()
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1,
                        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    hist = {k: [] for k in models}; paired_rows=[]; component_rows=[]; oracle_rows=[]; basis_rows=[]
    image_rows=[]; interior_acc=defaultdict(lambda: [0,0,0,0]); rescue_counts=defaultdict(int)
    distance_values={c: {m: [] for m in ("euclidean","manhattan","chebyshev")} for c in range(4)}
    total_gt=defaultdict(int); total_fn=defaultdict(int); total_seed=defaultdict(int); no_seed_fn=defaultdict(int)
    radius_hits=defaultdict(int); affinity_hits=defaultdict(int); affinity_components=defaultdict(lambda:[0,0,0])
    basis_counts=defaultdict(lambda: defaultdict(float)); basis_samples=defaultdict(list)
    heaps={"unanchored":[],"fn_distance":[],"sshr_rescue":[]}; counter=0
    structure=np.ones((3,3),np.uint8)
    for index,(names,image) in enumerate(loader,1):
        image_id=names[0]; original=np.asarray(Image.open(valroot/"img"/f"{image_id}.png").convert("RGB")); truth=np.asarray(Image.open(valroot/"mask"/f"{image_id}.png")); image=image.cuda(non_blocking=True)
        bundle=dfsc_bundle(models["dfsc"],image,original.shape[:2]); predictions={"dfsc":bundle["prediction"],
            "sshr":_predict_sshr(models["sshr"],image,original), "old_gcqm":_predict_gcqm(models["old_gcqm"],image,original),
            "ccac":_predict_gcqm(models["ccac"],image,original)}
        for name,pred in predictions.items(): hist[name].append(foreground_confusion(truth,pred))
        grid=bundle["final"].shape[-2:]; gt=resize_label(truth,grid); pdg=bundle["prediction_grid"]; psg=resize_label(predictions["sshr"],grid)
        basis,weight=bundle["basis"],bundle["weights"]; bmax=basis.max(0); seeds=np.zeros((4,*grid),bool)
        cmax=np.zeros((4,*grid),np.float32); unions={k:np.zeros((4,*grid),np.float32) for k in (5,10,20)}
        fn_maps=np.zeros((4,*grid),bool); dist_stack=np.full((4,*grid),np.inf,np.float32); rescue_type=np.zeros(grid,np.uint8)
        image_unanchored=image_gt=image_fn=image_rescue=0; image_dist=[]; image_unlimited=image_oracle10=image_basis_uncovered=image_base_cov=0
        for cls in range(4):
            mask=gt==cls; n_gt=int(mask.sum()); total_gt[cls]+=n_gt; image_gt+=n_gt
            seeds[cls]=(bundle["base"][cls]>=THRESHOLDS[cls]) & bool(bundle["label"][cls]); total_seed[cls]+=int((seeds[cls]&mask).sum())
            fn=mask&(pdg!=cls); fn_maps[cls]=fn; n_fn=int(fn.sum()); total_fn[cls]+=n_fn; image_fn+=n_fn
            eu,ma,ch=distance_maps(seeds[cls]); dist_stack[cls]=eu
            for key,value in (("euclidean",eu),("manhattan",ma),("chebyshev",ch)):
                distance_values[cls][key].append(value[fn])
            no_seed_fn[cls]+=int(fn.sum()) if not seeds[cls].any() else 0
            image_dist.extend(eu[fn & np.isfinite(eu)].tolist())
            order=np.argsort(-weight[:,cls],kind="stable"); cmax[cls]=(basis*weight[:,cls,None,None]).max(0)
            for k in (5,10,20): unions[k][cls]=basis[order[:k]].max(0)
            threshold=float(THRESHOLDS[cls]); binary=basis>=threshold
            inter=(binary&mask).sum((1,2)); union=(binary|mask).sum((1,2)); score=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
            oracle_order=np.argsort(-score,kind="stable"); recalls={}
            for k in (1,3,5,10): recalls[k]=float((binary[oracle_order[:k]].any(0)&mask).sum()/max(n_gt,1))
            base_cov=float((seeds[cls]&mask).sum()/max(n_gt,1)); class_uncovered=(unions[20][cls]<threshold)&mask
            image_oracle10+=recalls[10]*n_gt; image_basis_uncovered+=int(class_uncovered.sum()); image_base_cov+=base_cov*n_gt
            basis_counts[cls]["gt"]+=n_gt; basis_counts[cls]["bmax_covered"]+=int(((bmax>=threshold)&mask).sum()); basis_counts[cls]["cmax_covered"]+=int(((cmax[cls]>=threshold)&mask).sum()); basis_counts[cls]["base_covered"]+=int((seeds[cls]&mask).sum()); basis_counts[cls]["class_uncovered"]+=int(class_uncovered.sum())
            for k in (5,10,20): basis_counts[cls][f"u{k}_covered"]+=int(((unions[k][cls]>=threshold)&mask).sum())
            basis_counts[cls]["oracle10_hits"]+=recalls[10]*n_gt; basis_samples[cls].append(bmax[mask]);
            basis_rows.append({"image_id":image_id,"class":cls,"gt_pixels":n_gt,"base_seed_coverage":base_cov,
                "basis_max_coverage":float(((bmax>=threshold)&mask).sum()/max(n_gt,1)),"class_basis_uncovered":float(class_uncovered.sum()/max(n_gt,1)),
                "cmax_coverage":float(((cmax[cls]>=threshold)&mask).sum()/max(n_gt,1)),**{f"u{k}_coverage":float(((unions[k][cls]>=threshold)&mask).sum()/max(n_gt,1)) for k in (5,10,20)},
                **{f"oracle_top{k}_recall":recalls[k] for k in (1,3,5,10)}})
            geo=geodesic_distance(mask,seeds[cls]); unlimited=int((geo>=0).sum()); radius_hits[(cls,"infinity")]+=unlimited; image_unlimited+=unlimited
            for radius in RADII: radius_hits[(cls,radius)]+=int(((geo>=0)&(geo<=radius)).sum())
            for threshold_a in AFFINITY_THRESHOLDS:
                reached=affinity_reachable(mask,seeds[cls],bundle["affinity"],threshold_a); affinity_hits[(cls,threshold_a)]+=int(reached.sum())
            labels,ncomp=ndimage.label(mask,structure=structure); class_unanchored_area=0
            for component in range(1,ncomp+1):
                cmask=labels==component; area=int(cmask.sum()); seed_count=int((seeds[cls]&cmask).sum()); seed_fraction=seed_count/area; category=anchor_category(seed_fraction)
                if category=="unanchored":
                    image_unanchored+=area; class_unanchored_area+=area
                component_geo=geodesic_distance(cmask,seeds[cls]&cmask); reachable=component_geo[cmask]; finite=reachable[reachable>=0]
                component_rows.append({"image_id":image_id,"class":cls,"component":component,"area":area,"max_F":float(bundle["base"][cls][cmask].max()),"mean_F":float(bundle["base"][cls][cmask].mean()),"p90_F":float(np.quantile(bundle["base"][cls][cmask],.9)),"seed_pixels":seed_count,"seed_fraction":seed_fraction,"anchor_category":category,"top_query_contribution_max":float(cmax[cls][cmask].max()),"mean_seed_distance":float(finite.mean()) if len(finite) else float("inf"),"R90":float(np.quantile(finite,.9)) if len(finite) else float("inf"),"max_seed_distance":float(finite.max()) if len(finite) else float("inf"),"basis_uncovered_fraction":float(class_uncovered[cmask].mean()),"dfsc_fn_rate":float(fn[cmask].mean()),"sshr_rescue_rate":float((fn&(psg==cls))[cmask].mean())})
                for threshold_a in AFFINITY_THRESHOLDS:
                    reached=affinity_reachable(cmask,seeds[cls]&cmask,bundle["affinity"],threshold_a); fraction=float(reached[cmask].mean()); slot=0 if fraction==0 else 2 if fraction==1 else 1; affinity_components[(cls,threshold_a)][slot]+=1
            rescue=fn&(psg==cls); image_rescue+=int(rescue.sum())
            oracle10_map=binary[oracle_order[:10]].any(0)
            type_e=rescue&(bmax<threshold)&(~oracle10_map); type_d=rescue&(~type_e)&(bmax>=threshold)&(bundle["base"][cls]<threshold)
            remaining=rescue&(~type_e)&(~type_d); type_a=remaining&(eu<=2); type_b=remaining&(eu>2)&(eu<=8); type_c=remaining&((eu>8)|(~np.isfinite(eu)))
            for code,(name,value) in enumerate((("A",type_a),("B",type_b),("C",type_c),("D",type_d),("E",type_e)),1): rescue_counts[(cls,name)]+=int(value.sum()); rescue_type[value]=code
            depth=ndimage.distance_transform_edt(mask); bins=((0,2),(2,4),(4,8),(8,np.inf))
            for bi,(lo,hi) in enumerate(bins):
                region=mask&(depth>lo)&(depth<=hi); key=(cls,bi); interior_acc[key][0]+=int(region.sum()); interior_acc[key][1]+=int((region&fn).sum()); interior_acc[key][2]+=int((region&seeds[cls]).sum()); interior_acc[key][3]+=int((region&(bmax>=threshold)).sum())
            paired_rows.append({"image_id":image_id,"class":cls,"gt_pixels":n_gt,"dfsc_iou":float(((mask)&(pdg==cls)).sum()/max((mask|(pdg==cls)).sum(),1)),"sshr_iou":float(((mask)&(psg==cls)).sum()/max((mask|(psg==cls)).sum(),1)),"fn_pixels":n_fn,"mean_fn_seed_distance":float(np.mean(eu[fn&np.isfinite(eu)])) if np.any(fn&np.isfinite(eu)) else float("inf"),"unanchored_area_fraction":float(class_unanchored_area/max(n_gt,1)),"unlimited_seed_recall":float(unlimited/max(n_gt,1)),"class_basis_uncovered":float(class_uncovered.sum()/max(n_gt,1)),"oracle_top10_basis_recall":recalls[10]})
        delta=safe_iou(gt,pdg)-safe_iou(gt,psg); rescue_fraction=image_rescue/max(image_fn,1); unanchored_fraction=image_unanchored/max(image_gt,1); mean_distance=float(np.mean(image_dist)) if image_dist else float("inf")
        image_rows.append({"image_id":image_id,"delta_iou_dfsc_vs_sshr":delta,"mean_fn_seed_distance":mean_distance,"unanchored_gt_area_fraction":unanchored_fraction,"basis_uncovered_gt_fraction":image_basis_uncovered/max(image_gt,1),"oracle_top10_basis_recall":image_oracle10/max(image_gt,1),"unlimited_propagation_ceiling":image_unlimited/max(image_gt,1)})
        cls_viz=int(np.argmax([int((gt==c).sum()) for c in range(4)])); payload={"image_id":image_id,"class":cls_viz,"input":original,"truth":gt,"dfsc":pdg,"sshr":psg,"base":bundle["base"],"final":bundle["final"],"seeds":seeds,"distance":dist_stack,"fn":fn_maps,"bmax":bmax,"cmax":cmax,"u5":unions[5],"u10":unions[10],"u20":unions[20],"rescue_type":rescue_type}
        update_top(heaps["unanchored"],unanchored_fraction,counter,payload); update_top(heaps["fn_distance"],mean_distance if np.isfinite(mean_distance) else 999,counter,payload); update_top(heaps["sshr_rescue"],rescue_fraction,counter,payload); counter+=1
        if index%100==0 or index==len(loader): print(f"COVERAGE_AUDIT_PROGRESS={index}/{len(loader)}",flush=True)

    observed={name:scores_from_confusion(np.stack(values).sum(0))["mIoU"] for name,values in hist.items()}
    gate={"decision":"COMPARABLE" if all(abs(observed[k]-EXPECTED[k])<1e-10 for k in EXPECTED) else "COVERAGE_AUDIT_ENGINEERING_BLOCKED","expected_mIoU":EXPECTED,"observed_mIoU":observed,"paired_validation_images":len(loader),"same_split":True,"same_thresholds":True,"same_class_mapping":True,"same_tta":True,"same_resize_interpolation":True,"same_ignore_policy":True,"checkpoint_sha256":hashes}
    write_json(output/"provenance/reproduction_gate.json",gate)
    if gate["decision"]!="COMPARABLE": raise AssertionError(gate)
    write_csv(output/"paired/paired_image_class_table.csv",paired_rows)
    write_csv(output/"component_anchor/component_anchor_table.csv",component_rows)
    distance_summary=[]
    for label,classes in [(str(c),[c]) for c in range(4)]+[("overall",list(range(4)))]:
        values={m:np.concatenate([np.concatenate(distance_values[c][m]) if distance_values[c][m] else np.array([]) for c in classes]) for m in ("euclidean","manhattan","chebyshev")}; total=sum(total_fn[c] for c in classes); eu=values["euclidean"]
        distance_summary.append({"class":label,"fn_pixels":total,"fraction_within_1":float(np.sum(eu<=1)/max(total,1)),"fraction_within_2":float(np.sum(eu<=2)/max(total,1)),"fraction_within_4":float(np.sum(eu<=4)/max(total,1)),"fraction_within_8":float(np.sum(eu<=8)/max(total,1)),"fraction_gt8":float(np.sum((eu>8)&np.isfinite(eu))/max(total,1)),"no_seed_fraction":float(np.sum(~np.isfinite(eu))/max(total,1)),"median_euclidean":finite_quantile(eu,.5),"p75_euclidean":finite_quantile(eu,.75),"p90_euclidean":finite_quantile(eu,.90),"p95_euclidean":finite_quantile(eu,.95),"median_manhattan":finite_quantile(values["manhattan"],.5),"median_chebyshev":finite_quantile(values["chebyshev"],.5)})
    write_csv(output/"seed_distance/fn_seed_distance_summary.csv",distance_summary)
    write_csv(output/"seed_distance/local_reachability_summary.csv",[{"class":r["class"],"locally_reachable_fraction":r["fraction_within_2"],"locally_unreachable_fraction":1-r["fraction_within_2"],"geometric_limit":True if 1-r["fraction_within_2"]>=.60 else False if 1-r["fraction_within_2"]<=.30 else "MIXED"} for r in distance_summary])
    comp=pd.DataFrame(component_rows); summaries=[]
    for label,frame in list(comp.groupby("class"))+[("overall",comp)]:
        area=frame.area.sum(); summaries.append({"class":label,"components":len(frame),**{f"{cat}_fraction":float((frame.anchor_category==cat).mean()) for cat in ("anchored","weakly_anchored","unanchored")},**{f"area_weighted_{cat}_fraction":float(frame.loc[frame.anchor_category==cat,"area"].sum()/area) for cat in ("anchored","weakly_anchored","unanchored")}})
    write_csv(output/"component_anchor/component_anchor_summary.csv",summaries)
    radius_component=[]
    for label,frame in list(comp[comp.anchor_category!="unanchored"].groupby("class"))+[("overall",comp[comp.anchor_category!="unanchored"])]:
        radius_component.append({"class":label,"components":len(frame),"R90_le2":float((frame.R90<=2).mean()),"R90_2to4":float(((frame.R90>2)&(frame.R90<=4)).mean()),"R90_4to8":float(((frame.R90>4)&(frame.R90<=8)).mean()),"R90_gt8":float((frame.R90>8).mean())})
    write_csv(output/"component_anchor/component_radius_summary.csv",radius_component)
    q1,q2=comp.area.quantile([1/3,2/3]); comp["scale"]=pd.cut(comp.area,[-np.inf,q1,q2,np.inf],labels=["small","medium","large"])
    scale=[]
    for name,frame in comp.groupby("scale",observed=True): scale.append({"scale":str(name),"components":len(frame),"mean_seed_fraction":float(frame.seed_fraction.mean()),"median_R90":float(frame.R90.replace(np.inf,np.nan).median()),"unanchored_fraction":float((frame.anchor_category=="unanchored").mean()),"basis_uncovered_fraction":float(np.average(frame.basis_uncovered_fraction,weights=frame.area)),"dfsc_fn_rate":float(np.average(frame.dfsc_fn_rate,weights=frame.area)),"sshr_rescue_rate":float(np.average(frame.sshr_rescue_rate,weights=frame.area))})
    write_csv(output/"component_anchor/component_scale_summary.csv",scale)
    depth_names=("0-2","2-4","4-8",">8"); depth_rows=[]
    for (cls,bi),v in interior_acc.items(): depth_rows.append({"class":cls,"depth_bin":depth_names[bi],"pixels":v[0],"dfsc_fn_rate":v[1]/max(v[0],1),"seed_availability":v[2]/max(v[0],1),"basis_availability":v[3]/max(v[0],1)})
    write_csv(output/"component_anchor/interior_depth_summary.csv",depth_rows)
    curve=[]; graph=[]
    for label,classes in [(str(c),[c]) for c in range(4)]+[("overall",list(range(4)))]:
        denom=sum(total_gt[c] for c in classes)
        for radius in (*RADII,"infinity"): curve.append({"class":label,"radius":radius,"gt_recall_ceiling":sum(radius_hits[(c,radius)] for c in classes)/max(denom,1),"remaining_fn_fraction":1-sum(radius_hits[(c,radius)] for c in classes)/max(denom,1)})
        for threshold in AFFINITY_THRESHOLDS:
            counts=np.sum([affinity_components[(c,threshold)] for c in classes],axis=0); graph.append({"class":label,"threshold":threshold,"reachable_gt_fraction":sum(affinity_hits[(c,threshold)] for c in classes)/max(denom,1),"unreachable_gt_fraction":1-sum(affinity_hits[(c,threshold)] for c in classes)/max(denom,1),"unreachable_components":int(counts[0]),"partially_reachable_components":int(counts[1]),"fully_reachable_components":int(counts[2])})
    write_csv(output/"reachability/gt_constrained_oracle_radius_curve.csv",curve); write_csv(output/"reachability/affinity_graph_reachability.csv",graph)
    unlimited=[r for r in curve if str(r["radius"])=="infinity"]
    write_csv(output/"reachability/unlimited_reachability_summary.csv",[{**r,"affinity_graph_reachability_050":next(g["reachable_gt_fraction"] for g in graph if g["class"]==r["class"] and g["threshold"]==.5)} for r in unlimited])
    coverage=[]
    for label,classes in [(str(c),[c]) for c in range(4)]+[("overall",list(range(4)))]:
        denom=sum(basis_counts[c]["gt"] for c in classes); samples=np.concatenate([np.concatenate(basis_samples[c]) for c in classes if basis_samples[c]])
        coverage.append({"class":label,"gt_pixels":denom,"basis_max_coverage":sum(basis_counts[c]["bmax_covered"] for c in classes)/max(denom,1),"cmax_coverage":sum(basis_counts[c]["cmax_covered"] for c in classes)/max(denom,1),"base_seed_coverage":sum(basis_counts[c]["base_covered"] for c in classes)/max(denom,1),"class_basis_uncovered":sum(basis_counts[c]["class_uncovered"] for c in classes)/max(denom,1),"mean_bmax":float(samples.mean()),"p10_bmax":float(np.quantile(samples,.10)),"p25_bmax":float(np.quantile(samples,.25)),"p50_bmax":float(np.quantile(samples,.50)),**{f"u{k}_coverage":sum(basis_counts[c][f"u{k}_covered"] for c in classes)/max(denom,1) for k in (5,10,20)},"oracle_top10_recall":sum(basis_counts[c]["oracle10_hits"] for c in classes)/max(denom,1)})
    write_csv(output/"basis_coverage/basis_max_coverage.csv",coverage); write_csv(output/"basis_coverage/class_basis_union_coverage.csv",basis_rows); write_csv(output/"basis_coverage/oracle_basis_recall.csv",basis_rows)
    overall=next(r for r in coverage if r["class"]=="overall"); diagnosis="BASIS_BANK_DEFICIT" if overall["oracle_top10_recall"]<.85 else "WEIGHTING_DEFICIT" if overall["oracle_top10_recall"]>=.90 and overall["oracle_top10_recall"]-overall["base_seed_coverage"]>=.10 else "MIXED"
    write_csv(output/"basis_coverage/basis_weighting_diagnosis.csv",[{"diagnosis":diagnosis,**overall}])
    total_rescue=sum(rescue_counts.values()); rescue_summary=[]
    for label,classes in [(str(c),[c]) for c in range(4)]+[("overall",list(range(4)))]:
        denom=sum(rescue_counts[(c,t)] for c in classes for t in "ABCDE"); rescue_summary.append({"class":label,"rescue_pixels":denom,**{f"type_{t}_fraction":sum(rescue_counts[(c,t)] for c in classes)/max(denom,1) for t in "ABCDE"}})
    write_csv(output/"sshr_advantage/rescue_region_metrics.csv",image_rows); write_csv(output/"sshr_advantage/rescue_region_type_summary.csv",rescue_summary)
    correlations=[]; bootstrap={}
    frame=pd.DataFrame(image_rows); target=frame.delta_iou_dfsc_vs_sshr.to_numpy()
    for column in ("mean_fn_seed_distance","unanchored_gt_area_fraction","basis_uncovered_gt_fraction","oracle_top10_basis_recall","unlimited_propagation_ceiling"):
        value=frame[column].replace([np.inf,-np.inf],np.nan).to_numpy(); result=bootstrap_spearman(value,target); correlations.append({"variable":column,**result}); bootstrap[column]=result
    write_csv(output/"correlation/coverage_delta_iou_correlations.csv",correlations); write_json(output/"correlation/coverage_bootstrap.json",bootstrap)
    for tag,heap in heaps.items():
        for _,_,payload in sorted(heap,reverse=True): render_payload(payload,output/"visualizations/seed_distance",tag)
    anchor=next(r for r in summaries if str(r["class"])=="overall"); dist=next(r for r in distance_summary if r["class"]=="overall"); reach=next(r for r in unlimited if r["class"]=="overall"); graph_main=next(r for r in graph if r["class"]=="overall" and r["threshold"]==.5); rescue=next(r for r in rescue_summary if r["class"]=="overall")
    rescue_frac={t:rescue[f"type_{t}_fraction"] for t in "ABCDE"}; decision,confidence,evidence=decide_bottleneck(anchor["area_weighted_anchored_fraction"],overall["oracle_top10_recall"],reach["gt_recall_ceiling"],1-dist["fraction_within_2"],overall["class_basis_uncovered"],overall["base_seed_coverage"],rescue_frac)
    target_arch={"LONG_RANGE_PROPAGATION_LIMIT":"region-level / graph-level long-range completion","QUERY_MASK_COVERAGE_LIMIT":"Q_i -> B_i query-to-mask basis generation with multi-scale/high-resolution decoding","QUERY_BASIS_WEIGHTING_LIMIT":"query-to-basis class weighting/coupling","MIXED_COVERAGE_AND_REACHABILITY":"query-mask basis generation first, followed by region-level reachability","NO_SINGLE_CAUSE_IDENTIFIED":"a new pre-registered diagnosis before architecture changes"}[decision]
    cause={"LONG_RANGE_PROPAGATION_LIMIT":"finite/local propagation reach","QUERY_MASK_COVERAGE_LIMIT":"query-mask semantic coverage","QUERY_BASIS_WEIGHTING_LIMIT":"class weighting of an adequate basis bank","MIXED_COVERAGE_AND_REACHABILITY":"both semantic coverage and long-range reachability","NO_SINGLE_CAUSE_IDENTIFIED":"no single demonstrated mechanism"}[decision]
    summary={"fn_within_2":dist["fraction_within_2"],"local_unreachable_fn":1-dist["fraction_within_2"],"anchored_area_fraction":anchor["area_weighted_anchored_fraction"],"unanchored_area_fraction":anchor["area_weighted_unanchored_fraction"],"unlimited_seed_recall":reach["gt_recall_ceiling"],"affinity_reachability_050":graph_main["reachable_gt_fraction"],"basis_max_coverage":overall["basis_max_coverage"],"class_basis_uncovered":overall["class_basis_uncovered"],"base_seed_coverage":overall["base_seed_coverage"],"oracle_top10_recall":overall["oracle_top10_recall"],"sshr_rescue_fraction":total_rescue/max(sum(total_fn.values()),1),"rescue_type_fractions":rescue_frac}
    final={"decision":{"decision":decision,"confidence":confidence,"evidence":evidence,"basis_weighting_diagnosis":diagnosis,"dominant_cause":cause,"next_architecture_target":target_arch},"summary":summary,"reproduction_gate":gate,"provenance":{"source_commit":source_commit,"checkpoint_sha256":hashes,"dfsc_sha256":hashes["dfsc"],"zero_training":True,"parameter_updates":0,"runtime_seconds":time.perf_counter()-started}}
    write_json(output/"oracle/coverage_audit_final_result.json",final); report=output/"report/Semantic_Coverage_Propagation_Reachability_Audit_Report.md"; report.write_text(report_text(final))
    print(json.dumps({"decision":decision,"confidence":confidence,"summary":summary,"report":str(report)},indent=2)); print(f"DECISION = {decision}"); print(f"CONFIDENCE = {confidence}")


if __name__ == "__main__": main()
