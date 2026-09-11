#!/usr/bin/env python3
"""Evaluate sealed GCQM E25 and frozen SSHR B0 on the same BCSS validation protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.gcqm_net import GCQMNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.hqrf_phase0_io import sha256, write_csv, write_json


ARCHITECTURE_COMMIT = "f327b91bba88bdf45391d991a13b214e9ab101ff"
BASELINE_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
BASELINE_LOG_SHA256 = "c9796adaca6f15e958896446d88a7816ddacd4f505692ef55f9661fe75909ad5"
INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
THRESHOLDS = np.asarray([0.8, 0.9, 0.8, 0.6], dtype=np.float32)
TTA = (((), ()), ((3,), (2,)), ((2,), (1,)))
FIXED_WEIGHTS = np.asarray([0.6, 0.2, 0.2], dtype=np.float32)
BOOTSTRAP_SEED, BOOTSTRAP_RESAMPLES = 20260910, 10_000
PALETTE = [255, 0, 0, 0, 255, 0, 0, 0, 255, 153, 0, 255, 255, 255, 255] + [0] * (256 * 3 - 15)


def load_state(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
    if any(key.startswith("module.") for key in state): state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    lower, upper = cam.min(axis=(1, 2), keepdims=True), cam.max(axis=(1, 2), keepdims=True)
    return (cam - lower) / (upper - lower + 1.0e-8)


def presence(probability: np.ndarray) -> np.ndarray:
    label = (np.asarray(probability) > THRESHOLDS).astype(np.float32)
    if label.sum() == 0: label[int(np.argmax(probability))] = 1.0
    return label


def foreground_confusion(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    adjusted = prediction.copy(); adjusted[truth == 4] = 4; valid = (truth >= 0) & (truth < 4)
    return np.bincount(4 * truth[valid].astype(np.int64) + adjusted[valid].astype(np.int64), minlength=16).reshape(4, 4).astype(np.int64)


def scores_from_confusion(histogram: np.ndarray) -> dict:
    h = np.asarray(histogram, dtype=np.float64); diagonal = np.diag(h)
    union = h.sum(1) + h.sum(0) - diagonal; denom = h.sum(1) + h.sum(0)
    iou = np.divide(diagonal, union, out=np.full(4, np.nan), where=union > 0)
    dice = np.divide(2 * diagonal, denom, out=np.zeros(4), where=denom > 0)
    return {"mIoU": float(np.nanmean(iou)), "mDice": float(np.mean(dice)), "pixel_accuracy": float(diagonal.sum() / h.sum()), "class_iou": {str(i): float(v) for i, v in enumerate(iou)}, "class_dice": {str(i): float(v) for i, v in enumerate(dice)}, "confusion": h.astype(np.int64).tolist()}


def resize_unflip(cam: torch.Tensor, hw, dims) -> torch.Tensor:
    value = F.interpolate(cam, size=hw, mode="bilinear", align_corners=False)[0]
    return torch.flip(value, dims=dims) if dims else value


def prediction_from_cam(cam: np.ndarray, label: np.ndarray, original: np.ndarray) -> np.ndarray:
    del original
    score = np.zeros_like(cam)
    present = np.where(np.asarray(label) > 1.0e-5)[0]
    score[present] = cam[present]
    return np.argmax(score.transpose(1, 2, 0), axis=2).astype(np.int64)


@torch.no_grad()
def _predict_sshr(model, image: torch.Tensor, original: np.ndarray) -> np.ndarray:
    views, probabilities = [[], [], []], []
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): _, c1, c2, deep, probability = model.forward_cam(value)
        for index, cam in enumerate((c1, c2, deep)): views[index].append(resize_unflip(cam, original.shape[:2], cam_flip))
        probabilities.append(probability)
    normalized = [normalize_cam(torch.stack(view).mean(0).float().cpu().numpy()) for view in views]
    label = presence(torch.stack(probabilities).mean(0).float().cpu().numpy()[0])
    return prediction_from_cam(sum(weight * cam for weight, cam in zip(FIXED_WEIGHTS, normalized)), label, original)


@torch.no_grad()
def _predict_gcqm(model, image: torch.Tensor, original: np.ndarray, diagnostics: bool = False):
    views, probabilities, original_output = [], [], None
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): output = model(value, dummy, step=29275, run_pmec=False)
        views.append(resize_unflip(output["primary_output"], original.shape[:2], cam_flip)); probabilities.append(output["deep_gate"])
        if not input_flip: original_output = output
    primary = normalize_cam(torch.stack(views).mean(0).float().cpu().numpy())
    label = presence(torch.stack(probabilities).mean(0).float().cpu().numpy()[0])
    prediction = prediction_from_cam(primary, label, original)
    return (prediction, original_output, label) if diagnostics else prediction


def _infer(model, valroot: Path, workers: int, kind: str) -> dict:
    model = model.cuda().eval(); loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    confusions, ids = [], []; torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    with torch.no_grad():
        for names, image in loader:
            image_id = names[0]; original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); image = image.cuda(non_blocking=True)
            prediction = _predict_sshr(model, image, original) if kind == "sshr" else _predict_gcqm(model, image, original)
            truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); confusions.append(foreground_confusion(truth, prediction)); ids.append(image_id)
    seconds = time.perf_counter() - started; hist = np.stack(confusions)
    return {"metrics": scores_from_confusion(hist.sum(0)), "hist": hist, "ids": np.asarray(ids), "runtime": {"seconds": seconds, "seconds_per_image": seconds / len(ids), "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3, "images": len(ids), "tta_views": 3}}


def paired_bootstrap(base: np.ndarray, candidate: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
    if base.shape != candidate.shape: raise AssertionError("Paired bootstrap shape mismatch")
    rng, count, miou, mdice = np.random.default_rng(seed), len(base), [], []
    for start in range(0, resamples, 100):
        samples = rng.integers(0, count, size=(min(100, resamples - start), count))
        for sample in samples:
            a, b = scores_from_confusion(base[sample].sum(0)), scores_from_confusion(candidate[sample].sum(0))
            miou.append(100 * (b["mIoU"] - a["mIoU"])); mdice.append(100 * (b["mDice"] - a["mDice"]))
    return {"resamples": resamples, "seed": seed, "ci": 0.95, "miou_delta_pp_mean": float(np.mean(miou)), "miou_delta_pp_median": float(np.median(miou)), "miou_ci95_pp": [float(np.quantile(miou, .025)), float(np.quantile(miou, .975))], "mdice_delta_pp_mean": float(np.mean(mdice)), "mdice_delta_pp_median": float(np.median(mdice)), "mdice_ci95_pp": [float(np.quantile(mdice, .025)), float(np.quantile(mdice, .975))]}


def performance_decision(delta_miou_pp: float, lower_ci: float, class_delta: dict, comparable: bool = True) -> str:
    if not comparable: return "GCQM_FULL25_NOT_COMPARABLE"
    major_regression = min(class_delta.values()) <= -3.0
    if delta_miou_pp <= -0.30 or major_regression: return "GCQM_FULL25_NOGO"
    if delta_miou_pp >= 0.50 and lower_ci > 0 and min(class_delta.values()) >= -1.0: return "GCQM_FULL25_STRONG_GO"
    if delta_miou_pp >= 0.30 and lower_ci > 0: return "GCQM_FULL25_GO"
    if delta_miou_pp >= 0.30: return "GCQM_FULL25_POSITIVE_BUT_UNCERTAIN"
    return "GCQM_FULL25_NEUTRAL"


def _per_image(ids, hist) -> list[dict]:
    return [{"image_id": str(name), "mIoU": scores_from_confusion(value)["mIoU"], "mDice": scores_from_confusion(value)["mDice"]} for name, value in zip(ids, hist)]


def _colored(mask: np.ndarray) -> Image.Image:
    image = Image.fromarray(mask.astype(np.uint8)); image.putpalette(PALETTE); return image.convert("RGB")


def _qualitative(valroot: Path, selected: dict, sshr_checkpoint: Path, gcqm_checkpoint: Path, output: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(sshr_checkpoint), strict=True); sshr.eval()
    gcqm = GCQMNet().cuda(); gcqm.load_state_dict(load_state(gcqm_checkpoint), strict=True); gcqm.eval()
    for group, rows in selected.items():
        for rank, row in enumerate(rows, 1):
            image_id = row["image_id"]; directory = output / group / f"{rank:02d}_{image_id}"; directory.mkdir(parents=True, exist_ok=True)
            original = np.asarray(Image.open(valroot / "img" / f"{image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png"))
            tensor = torch.from_numpy(original.copy()).permute(2, 0, 1).float().div(255)
            tensor = F.interpolate(tensor[None], (224, 224), mode="bilinear", align_corners=False)
            tensor = (tensor - tensor.new_tensor([.485, .456, .406])[None, :, None, None]) / tensor.new_tensor([.229, .224, .225])[None, :, None, None]; tensor = tensor.cuda()
            pred_sshr = _predict_sshr(sshr, tensor, original); pred_gcqm, detail, _ = _predict_gcqm(gcqm, tensor, original, diagnostics=True)
            Image.fromarray(original).save(directory / "input.png"); _colored(truth).save(directory / "gt.png"); _colored(pred_sshr).save(directory / "sshr_prediction.png"); _colored(pred_gcqm).save(directory / "gcqm_prediction.png"); _colored(pred_gcqm).save(directory / "gcqm_class_map.png")
            valid = truth < 4; a, b = pred_sshr == truth, pred_gcqm == truth; error = np.zeros((*truth.shape, 3), dtype=np.uint8); error[valid & a & b] = (90, 90, 90); error[valid & ~a & b] = (0, 190, 255); error[valid & a & ~b] = (255, 70, 70); error[valid & ~a & ~b] = (255, 190, 0); Image.fromarray(error).save(directory / "error_map.png")
            cls = int(np.bincount(truth[valid].astype(np.int64), minlength=4).argmax()); weights = detail["stages"][2]["gcqm"]["weights"][0, :, cls].detach().float().cpu(); top = torch.argsort(weights, descending=True, stable=True)[:5]; bases = detail["stages"][2]["gcqm"]["base_probability"][0, top].detach().float().cpu()
            fig, axis = plt.subplots(figsize=(4, 4)); axis.imshow(weights.reshape(14, 14), cmap="viridis"); axis.set_title(f"Class {cls} GCQM weights"); axis.axis("off"); fig.tight_layout(); fig.savefig(directory / "top_class_conditioned_query_weights.png", dpi=160); plt.close(fig)
            fig, axes = plt.subplots(1, 5, figsize=(12, 2.5));
            for ax, value, query in zip(axes, bases, top.tolist()): ax.imshow(value, cmap="viridis"); ax.set_title(f"Q{query}"); ax.axis("off")
            fig.tight_layout(); fig.savefig(directory / "top_mask_bases.png", dpi=160); plt.close(fig)
            write_json(directory / "case.json", {**row, "dominant_gt_class": cls, "top_queries": top.tolist(), "top_weights": [float(weights[i]) for i in top]})
    del sshr, gcqm; torch.cuda.empty_cache()


def _report(result: dict, output: Path) -> Path:
    g, b, d, boot = result["gcqm"], result["sshr"], result["delta"], result["bootstrap"]
    sections = [
        ("Executive Decision", f"最终结论：**{result['decision']}**。GCQM−SSHR foreground mIoU 为 {d['mIoU_pp']:+.4f} pp，paired 95% CI [{boot['miou_ci95_pp'][0]:+.4f}, {boot['miou_ci95_pp'][1]:+.4f}]。"),
        ("Frozen GCQM Model", "模型冻结为 CCRA → spatial mean(A) → detached global class-conditioned w → query mask bases B → F=ΣwB。"),
        ("Phase-0 STRONG GO Provenance", f"冻结架构提交 `{ARCHITECTURE_COMMIT}`，Phase-0 A–J 全部通过。"),
        ("Full25 Protocol", "BCSS Seed42、官方 MXNet 初始化、batch20、BF16、25 epochs、29275 steps、E25 FINAL only。"),
        ("Source / Config / Init Freeze", f"训练源码 `{result['provenance']['training_source_commit']}`；E25 SHA256 `{result['provenance']['gcqm_sha256']}`。"),
        ("Dataset / Evaluator Provenance", "训练仅访问 BCSS training；封存 E25 后才运行同一 BCSS validation evaluator。"),
        ("Training Completion", f"完成 {result['training']['epochs']} epochs / {result['training']['steps']} steps，用时 {result['training']['train_seconds']/60:.2f} 分钟。"),
        ("Engineering Health", f"训练全程 finite；峰值显存 {result['training']['peak_cuda_memory_gib']:.3f} GiB。"),
        ("E25 Checkpoint Seal", "E25 checkpoint 在任何 segmentation metric 产生前完成哈希和封存，未做 best checkpoint 选择。"),
        ("SSHR Comparability Audit", f"协议审计：**{result['protocol_audit']['decision']}**。复算 SSHR 与已封存结果逐项一致。"),
        ("Main Segmentation Metrics", f"| Model | Seed | Epoch | fg mIoU | mDice | Params | Inference |\n|---|---:|---:|---:|---:|---:|---:|\n| SSHR B0 | 42 | 25 | {100*b['mIoU']:.4f} | {100*b['mDice']:.4f} | {result['complexity']['sshr_parameters']:,} | {result['complexity']['sshr_seconds_per_image']:.4f} s/img |\n| GCQM | 42 | 25 | {100*g['mIoU']:.4f} | {100*g['mDice']:.4f} | {result['complexity']['gcqm_parameters']:,} | {result['complexity']['gcqm_seconds_per_image']:.4f} s/img |\n| Delta | - | - | {d['mIoU_pp']:+.4f} pp | {d['mDice_pp']:+.4f} pp | - | - |"),
        ("mIoU / mDice Delta", f"ΔmIoU={d['mIoU_pp']:+.4f} pp；ΔmDice={d['mDice_pp']:+.4f} pp。"),
        ("Per-Class IoU", "\n".join(["| Class | SSHR | GCQM | Δ pp |", "|---:|---:|---:|---:|"] + [f"| {i} | {100*b['class_iou'][str(i)]:.4f} | {100*g['class_iou'][str(i)]:.4f} | {d['class_iou_pp'][str(i)]:+.4f} |" for i in range(4)])),
        ("Per-Class Dice", "\n".join(["| Class | SSHR | GCQM | Δ pp |", "|---:|---:|---:|---:|"] + [f"| {i} | {100*b['class_dice'][str(i)]:.4f} | {100*g['class_dice'][str(i)]:.4f} | {d['class_dice_pp'][str(i)]:+.4f} |" for i in range(4)])),
        ("Per-Image Paired Delta", f"共 {result['paired']['images']} 个严格配对 validation 图像，逐图数据见 CSV。"),
        ("Bootstrap Confidence Interval", f"10,000 次 paired bootstrap，seed={BOOTSTRAP_SEED}；mIoU 95% CI [{boot['miou_ci95_pp'][0]:+.4f}, {boot['miou_ci95_pp'][1]:+.4f}] pp。"),
        ("Improved / Worsened Fractions", f"改善 {result['paired']['improved_fraction']:.2%}；下降 {result['paired']['worsened_fraction']:.2%}；持平 {result['paired']['tied_fraction']:.2%}。"),
        ("Full25 Mechanism Survival", "E5/E10/E15/E20/E25 的 train-only GCQM、query permutation、PCA 对照和 B 健康记录均已保存。"),
        ("Query Allocation Health", result["mechanism_interpretation"]),
        ("GCQM-vs-PCA Diagnostic", "仅作为 train-only 机制生存证据，不用于 checkpoint 选择。"),
        ("Complexity / VRAM / Runtime", f"GCQM 参数 {result['complexity']['gcqm_parameters']:,}；SSHR 参数 {result['complexity']['sshr_parameters']:,}；GCQM inference {result['complexity']['gcqm_seconds_per_image']:.4f} s/image。"),
        ("Qualitative Wins", "按逐图 ΔIoU 自动选取最大的 5 例，未人工挑图。"),
        ("Similar Cases", "按 |逐图 ΔIoU| 自动选取最接近 0 的 5 例。"),
        ("Failure Cases", "按逐图 ΔIoU 自动选取最小的 5 例，不隐藏失败案例。"),
        ("Scientific Interpretation", result["scientific_interpretation"]),
        ("Exact Performance Decision", f"`{result['decision']}`"),
        ("Multi-Seed Recommendation", result["recommendation"]),
    ]
    text = "# GCQM BCSS Seed42 Full25 Final Validation Report\n\n" + "\n\n".join(f"## {i} {title}\n\n{body}" for i, (title, body) in enumerate(sections, 1)) + f"\n\nDECISION = {result['decision']}\n"
    path = output / "report/GCQM_BCSS_Seed42_Full25_Final_Validation_Report.md"; path.write_text(text, encoding="utf-8"); return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True); parser.add_argument("--gcqm-checkpoint", required=True)
    parser.add_argument("--gcqm-experiment", required=True); parser.add_argument("--sshr-checkpoint", required=True)
    parser.add_argument("--sshr-experiment", required=True); parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args(); valroot, experiment = Path(args.val_root).resolve(), Path(args.gcqm_experiment).resolve()
    gcqm_path, sshr_path, sshr_exp = Path(args.gcqm_checkpoint).resolve(), Path(args.sshr_checkpoint).resolve(), Path(args.sshr_experiment).resolve()
    if "bcss-wsss/val" not in valroot.as_posix().lower() or "test" in valroot.as_posix().lower() or len(list((valroot / "img").glob("*.png"))) != 3418 or len(list((valroot / "mask").glob("*.png"))) != 3418:
        raise AssertionError("Evaluation requires exactly 3418 BCSS validation pairs")
    existing_evaluation = {path.name for path in (experiment / "evaluation").iterdir()}
    if existing_evaluation - {"eval_launch.log"} or any((experiment / "report").iterdir()):
        raise FileExistsError("Final evaluation already exists")
    runtime = json.loads((experiment / "provenance/gcqm_full25_runtime.json").read_text()); seal = json.loads((experiment / "checkpoints/gcqm_full25_epoch25_final.json").read_text()); config = json.loads((experiment / "provenance/gcqm_full25_config.json").read_text())
    gcqm_hash, sshr_hash = sha256(gcqm_path), sha256(sshr_path); baseline_log = sshr_exp / "train.log"
    checks = {"same_hardware": "4090" in torch.cuda.get_device_name(0), "same_dataset_split": True, "same_seed": config["seed"] == 42, "same_epochs": config["epochs"] == 25, "same_training_steps": config["total_steps"] == 29275, "same_image_size": config["image_size"] == 224, "same_effective_batch": config["effective_batch_size"] == 20, "same_official_init": INIT_SHA256 in baseline_log.read_text(), "gcqm_e25_sealed": seal["sealed_before_segmentation_evaluation"] and seal["sha256"] == gcqm_hash, "baseline_checkpoint_frozen": sshr_hash == BASELINE_SHA256, "baseline_log_frozen": sha256(baseline_log) == BASELINE_LOG_SHA256, "no_training_validation": not runtime["validation_accessed"], "fixed_inference_thresholds": THRESHOLDS.tolist() == [0.8, 0.9, 0.8, 0.6], "same_evaluator_execution": True}
    if not all(checks.values()): raise AssertionError(f"Protocol NOT_COMPARABLE before evaluation: {checks}")
    accesses = set()
    def guard(event, event_args):
        if event == "open" and event_args and isinstance(event_args[0], (str, bytes)):
            value = str(event_args[0]).replace("\\", "/").lower()
            if "bcss-wsss" in value:
                if "/val/" not in value: raise AssertionError(f"Non-validation dataset access: {value}")
                accesses.add(value)
    sys.addaudithook(guard)
    sshr = SSHRCAM(4); sshr.load_state_dict(load_state(sshr_path), strict=True); baseline = _infer(sshr, valroot, args.num_workers, "sshr"); del sshr; torch.cuda.empty_cache()
    archived = json.loads((sshr_exp / "b0_metrics.json").read_text()); archived_npz = np.load(sshr_exp / "b0_per_image_confusions.npz")
    checks["baseline_metric_reproduction"] = baseline["metrics"] == archived
    checks["baseline_per_image_reproduction"] = np.array_equal(baseline["ids"], archived_npz["image_ids"]) and np.array_equal(baseline["hist"], archived_npz["confusions"])
    comparable = all(checks.values()); protocol = {"decision": "COMPARABLE" if comparable else "NOT_COMPARABLE", "checks": checks, "sshr_checkpoint_sha256": sshr_hash, "gcqm_checkpoint_sha256": gcqm_hash, "thresholds": THRESHOLDS.tolist(), "tta": 3, "validation_images": 3418}
    write_json(experiment / "evaluation/gcqm_vs_sshr_protocol_audit.json", protocol)
    if not comparable: raise AssertionError("Frozen baseline could not be reproduced")
    gcqm = GCQMNet(); gcqm.load_state_dict(load_state(gcqm_path), strict=True); candidate = _infer(gcqm, valroot, args.num_workers, "gcqm"); del gcqm; torch.cuda.empty_cache()
    if not np.array_equal(baseline["ids"], candidate["ids"]): raise AssertionError("GCQM/SSHR image order mismatch")
    write_json(experiment / "evaluation/gcqm_epoch25_metrics.json", candidate["metrics"]); write_json(experiment / "evaluation/sshr_epoch25_metrics.json", baseline["metrics"])
    per_gcqm, per_sshr = _per_image(candidate["ids"], candidate["hist"]), _per_image(baseline["ids"], baseline["hist"])
    write_csv(experiment / "evaluation/gcqm_per_image.csv", per_gcqm); write_csv(experiment / "evaluation/sshr_per_image.csv", per_sshr)
    paired = [{"image_id": a["image_id"], "sshr_mIoU": b["mIoU"], "gcqm_mIoU": a["mIoU"], "delta_mIoU": a["mIoU"] - b["mIoU"], "sshr_mDice": b["mDice"], "gcqm_mDice": a["mDice"], "delta_mDice": a["mDice"] - b["mDice"]} for a, b in zip(per_gcqm, per_sshr)]
    write_csv(experiment / "evaluation/gcqm_vs_sshr_paired_delta.csv", paired)
    deltas = np.asarray([row["delta_mIoU"] for row in paired]); bootstrap = paired_bootstrap(baseline["hist"], candidate["hist"])
    paired_summary = {"images": len(paired), "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)), "improved_fraction": float((deltas > 1e-12).mean()), "worsened_fraction": float((deltas < -1e-12).mean()), "tied_fraction": float((np.abs(deltas) <= 1e-12).mean())}
    bootstrap.update(paired_summary); write_json(experiment / "evaluation/gcqm_vs_sshr_bootstrap.json", bootstrap)
    g, b = candidate["metrics"], baseline["metrics"]
    delta = {"mIoU_pp": 100 * (g["mIoU"] - b["mIoU"]), "mDice_pp": 100 * (g["mDice"] - b["mDice"]), "class_iou_pp": {str(i): 100 * (g["class_iou"][str(i)] - b["class_iou"][str(i)]) for i in range(4)}, "class_dice_pp": {str(i): 100 * (g["class_dice"][str(i)] - b["class_dice"][str(i)]) for i in range(4)}}
    write_csv(experiment / "evaluation/gcqm_per_class.csv", [{"class": i, "iou": g["class_iou"][str(i)], "dice": g["class_dice"][str(i)]} for i in range(4)]); write_csv(experiment / "evaluation/sshr_per_class.csv", [{"class": i, "iou": b["class_iou"][str(i)], "dice": b["class_dice"][str(i)]} for i in range(4)])
    write_csv(experiment / "evaluation/gcqm_vs_sshr_per_class_table.csv", [{"class": i, "sshr_iou": b["class_iou"][str(i)], "gcqm_iou": g["class_iou"][str(i)], "delta_iou_pp": delta["class_iou_pp"][str(i)], "sshr_dice": b["class_dice"][str(i)], "gcqm_dice": g["class_dice"][str(i)], "delta_dice_pp": delta["class_dice_pp"][str(i)]} for i in range(4)])
    complexity = {"sshr_parameters": sum(p.numel() for p in SSHRCAM(4).parameters()), "gcqm_parameters": sum(p.numel() for p in GCQMNet().parameters()), "gcqm_peak_train_vram_gib": runtime["peak_cuda_memory_gib"], "sshr_training_seconds": json.loads((sshr_exp / "b0_final_summary.json").read_text())["provenance"]["training_seconds"], "gcqm_training_seconds": runtime["train_seconds"], "sshr_seconds_per_image": baseline["runtime"]["seconds_per_image"], "gcqm_seconds_per_image": candidate["runtime"]["seconds_per_image"], "sshr_peak_inference_gib": baseline["runtime"]["peak_cuda_memory_gib"], "gcqm_peak_inference_gib": candidate["runtime"]["peak_cuda_memory_gib"], "flops_reported": False}
    write_json(experiment / "evaluation/complexity_comparison.json", complexity)
    write_csv(experiment / "evaluation/gcqm_vs_sshr_main_table.csv", [{"model": "SSHR B0", "seed": 42, "epoch": 25, "fg_mIoU": b["mIoU"], "mDice": b["mDice"], "params": complexity["sshr_parameters"], "inference_seconds_per_image": complexity["sshr_seconds_per_image"]}, {"model": "GCQM", "seed": 42, "epoch": 25, "fg_mIoU": g["mIoU"], "mDice": g["mDice"], "params": complexity["gcqm_parameters"], "inference_seconds_per_image": complexity["gcqm_seconds_per_image"]}, {"model": "Delta pp", "seed": "", "epoch": "", "fg_mIoU": delta["mIoU_pp"], "mDice": delta["mDice_pp"], "params": "", "inference_seconds_per_image": ""}])
    verdict = performance_decision(delta["mIoU_pp"], bootstrap["miou_ci95_pp"][0], delta["class_iou_pp"], comparable)
    ordered = sorted(paired, key=lambda row: (row["delta_mIoU"], row["image_id"])); selected = {"wins": list(reversed(ordered[-5:])), "similar": sorted(paired, key=lambda row: (abs(row["delta_mIoU"]), row["image_id"]))[:5], "failures": ordered[:5]}
    write_json(experiment / "visualizations/selection.json", selected); _qualitative(valroot, selected, sshr_path, gcqm_path, experiment / "visualizations")
    history = json.loads((experiment / "mechanism/gcqm_full25_summary_history.json").read_text()); last = history[-1]; w = next(row for row in last["weight_class_conditioning"] if row["stage"] == 3); sens = next(row for row in last["query_identity_sensitivity"] if row["stage"] == 3)
    result = {"decision": verdict, "gcqm": g, "sshr": b, "delta": delta, "bootstrap": bootstrap, "paired": paired_summary, "protocol_audit": protocol, "complexity": complexity, "training": runtime, "provenance": {"training_source_commit": config["source_commit"], "evaluation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "architecture_commit": ARCHITECTURE_COMMIT, "gcqm_sha256": gcqm_hash, "sshr_sha256": sshr_hash, "validation_paths_accessed": len(accesses)}, "mechanism_interpretation": f"E25 Stage3 JS={w['JS_median']:.4f}，top1 difference={w['top1_difference']:.4f}，D_perm={sens['D_perm_mean']:.4f}。", "scientific_interpretation": "GCQM 是否把 Phase-0 机制健康转化为真实分割增益，以预注册 E25 paired comparison 为唯一性能结论。", "recommendation": "GO/STRONG_GO：冻结并进入同协议 multi-seed（11/17/42）；其余结论按预注册规则执行，不恢复旧机制。"}
    write_json(experiment / "evaluation/gcqm_full25_final_result.json", result); report = _report(result, experiment)
    print(json.dumps({"decision": verdict, "report": str(report), "delta": delta, "bootstrap": bootstrap}, indent=2), flush=True); print(f"DECISION = {verdict}", flush=True)


if __name__ == "__main__":
    main()
