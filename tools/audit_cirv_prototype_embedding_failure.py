#!/usr/bin/env python3
"""Zero-training causal anatomy of CIRV prototype and region embeddings."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.cirv import STRUCTURE8, extract_regions, l2_normalize, spherical_kmeans
from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    TTA, foreground_confusion, load_state, normalize_cam, prediction_from_cam,
    presence, resize_unflip, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json
from tools.run_cirv_phase0_bcss_seed42 import infer_hqmr_cirv_inputs, parse_label


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
BANK_A_SHA256 = "973bc5cfb7c58d64a0fcd1f88a51c677cfddf1b468ba36d42fb7ded4e75a6544"
PHASE0_METRICS_SHA256 = "68c73190de58cd2437b4c86b66f70749c1fc2260d88cd0d9454720430c88b7a9"
TRAIN_GT_AUDIT_SHA256 = "5a472941aa9d663a56eb4081e09ad339d1662be29dd0486f13fcc669d96d5d89"
EXPECTED_BASE = 0.6557244403737567
EXPECTED_PROTO_COMPONENT = 0.3768018018018018
EXPECTED_M1_COUNT = 4440
EXPECTED_M1_AREA = 8750254
EPS = 1.0e-8
REPRESENTATIONS = ("R1_K4", "R2_H5", "R3_query_conditioned_H4", "R4_deep_backbone")
BANK_NAMES = ("A", "B", "C", "D")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def setup(output: Path) -> None:
    for name in ("provenance", "sources", "banks", "source_anatomy", "targets", "counterfactual",
                 "representation", "prototype_anatomy", "headroom", "visualizations", "report", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)


def load_model(path: Path) -> HQMRNet:
    if sha256(path) != HQMR_SHA256: raise AssertionError("HQMR checkpoint mismatch")
    model = HQMRNet().cuda(); model.load_state_dict(load_state(path), strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return model


@torch.no_grad()
def pool_masks(masks: list[np.ndarray], feature: torch.Tensor) -> np.ndarray:
    if not masks: return np.empty((0, feature.shape[0]), np.float32)
    value = torch.as_tensor(np.stack(masks), device=feature.device, dtype=torch.float32)[:, None]
    soft = F.interpolate(value, size=feature.shape[-2:], mode="area")[:, 0]
    pooled = torch.einsum("nhw,dhw->nd", soft, feature.detach().float())
    pooled /= soft.sum((1, 2))[:, None].clamp_min(EPS)
    return F.normalize(pooled, dim=1, eps=EPS).cpu().numpy()


@torch.no_grad()
def pool_query_conditioned(masks: list[np.ndarray], classes: list[int], key4: torch.Tensor,
                           qbar: torch.Tensor) -> np.ndarray:
    if not masks: return np.empty((0, key4.shape[0]), np.float32)
    value = torch.as_tensor(np.stack(masks), device=key4.device, dtype=torch.float32)[:, None]
    soft = F.interpolate(value, size=key4.shape[-2:], mode="area")[:, 0]
    alpha = torch.sigmoid(torch.einsum("cd,dhw->chw", qbar.detach().float(), key4.detach().float()) /
                          math.sqrt(key4.shape[0]))
    weighted = soft * alpha[torch.as_tensor(classes, device=key4.device)]
    pooled = torch.einsum("nhw,dhw->nd", weighted, key4.detach().float())
    pooled /= weighted.sum((1, 2))[:, None].clamp_min(EPS)
    return F.normalize(pooled, dim=1, eps=EPS).cpu().numpy()


def largest_gt_regions(truth: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    masks, classes = [], []
    for cls in range(4):
        labels, count = ndimage.label(truth == cls, structure=STRUCTURE8)
        if count:
            sizes = np.bincount(labels.ravel())[1:]
            masks.append(labels == int(np.argmax(sizes) + 1)); classes.append(cls)
    return masks, classes


def all_gt_regions(truth: np.ndarray) -> tuple[list[np.ndarray], list[int], list[int]]:
    masks, classes, component_ids = [], [], []
    for cls in range(4):
        labels, count = ndimage.label(truth == cls, structure=STRUCTURE8)
        for component_id in range(1, count + 1):
            masks.append(labels == component_id); classes.append(cls); component_ids.append(component_id)
    return masks, classes, component_ids


def source_masks(prediction: np.ndarray, labels: np.ndarray, anchors: np.ndarray) -> list[dict]:
    regions = extract_regions(prediction); selected = []
    for cls in range(4):
        if not labels[cls]: continue
        candidates = []
        for region in regions:
            if region["class_id"] != cls: continue
            count = int(np.sum(region["mask"] & anchors[cls]))
            if count: candidates.append((count, region))
        if candidates:
            count, region = sorted(candidates, key=lambda value: (-value[0], value[1]["component_id"]))[0]
            selected.append({"class_id": cls, "component_id": region["component_id"],
                             "area": region["area"], "anchor_count": count, "mask": region["mask"]})
    return selected


def gt_anatomy(mask: np.ndarray, truth: np.ndarray) -> dict:
    counts = np.bincount(truth[mask].ravel(), minlength=5)[:5]
    majority = int(np.argmax(counts[:4]))
    area = int(mask.sum()); majority_count = int(counts[majority])
    foreground = counts[:4]; total_fg = int(foreground.sum())
    distribution = foreground / max(total_fg, 1)
    entropy = float(-np.sum(distribution[distribution > 0] * np.log(distribution[distribution > 0])))
    return {"gt_majority_class": majority, "gt_majority_pixels": majority_count,
            "gt_majority_purity": majority_count / max(area, 1),
            "gt_entropy": entropy, "gt_class_count": int(np.sum(foreground > 0)),
            "cross_class_area_fraction": (area - majority_count) / max(area, 1),
            "ignore_fraction": int(counts[4]) / max(area, 1)}


def with_embeddings(metadata: list[dict], embeddings: list[np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(metadata)
    values = np.stack(embeddings).astype(np.float32)
    return pd.concat((frame.reset_index(drop=True),
                      pd.DataFrame(values, columns=[f"z_{i:03d}" for i in range(values.shape[1])])), axis=1)


def build_bank(frame: pd.DataFrame, label_column: str, z_columns: list[str]) -> tuple[np.ndarray, list[dict]]:
    banks, reports = [], []
    for cls in range(4):
        values = frame.loc[frame[label_column] == cls, z_columns].to_numpy(np.float32)
        centers, assignment, report = spherical_kmeans(values, 4, seed=42)
        report["class_id"] = cls; reports.append(report); banks.append(centers)
    return np.stack(banks).astype(np.float32), reports


def save_bank(path: Path, bank: np.ndarray) -> str:
    np.save(path, bank.astype(np.float32), allow_pickle=False); return sha256(path)


def provenance(args, output: Path) -> None:
    phase0 = Path(args.phase0_output)
    if sha256(phase0 / "phase0/phase0_metrics.json") != PHASE0_METRICS_SHA256: raise AssertionError("Phase0 archive mismatch")
    if sha256(phase0 / "phase0/static_prototype_bank.npy") != BANK_A_SHA256: raise AssertionError("Bank A mismatch")
    if sha256(Path(args.train_gt_audit)) != TRAIN_GT_AUDIT_SHA256: raise AssertionError("Reconstructed TRAIN GT audit mismatch")
    config = {"audit": "CIRV Prototype-Embedding Failure Anatomy", "dataset": "BCSS", "seed": 42,
              "zero_training": True, "parameter_updates": 0, "full25": False,
              "checkpoint": str(Path(args.hqmr_checkpoint).resolve()), "checkpoint_sha256": HQMR_SHA256,
              "phase0_output": str(phase0.resolve()), "bankA_sha256": BANK_A_SHA256,
              "trainroot": str(Path(args.trainroot).resolve()), "train_gt_root": str(Path(args.train_gt_root).resolve()),
              "train_gt_audit": str(Path(args.train_gt_audit).resolve()),
              "val_root": str(Path(args.val_root).resolve()), "k": 4, "kmeans_seed": 42,
              "batch_size": args.batch_size, "num_workers": args.num_workers,
              "source_commit": git_commit(), "validation_firewall": "banks frozen before validation"}
    write_json(output / "provenance/proto_embed_audit_config.json", config)
    (output / "provenance/proto_embed_audit_source_commit.txt").write_text(git_commit() + "\n")
    (output / "provenance/proto_embed_audit_git_diff.patch").write_text(
        subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True))
    write_json(output / "provenance/cirv_phase0_archive.json", {
        "path": str(phase0.resolve()), "metrics_sha256": PHASE0_METRICS_SHA256,
        "bank_sha256": BANK_A_SHA256,
        "decision": json.loads((phase0 / "phase0/phase0_metrics.json").read_text())["decision"]})


@torch.no_grad()
def run_build(args, output: Path) -> None:
    if output.exists() and (output / "banks/banks_frozen_before_validation.json").exists():
        raise FileExistsError("Audit banks already frozen")
    setup(output); provenance(args, output)
    model = load_model(Path(args.hqmr_checkpoint))
    dataset = Stage1_InferDataset(args.trainroot, img_size=224)
    if len(dataset) != 23422: raise AssertionError("Expected 23422 training images")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    bank_meta = {name: [] for name in BANK_NAMES}; bank_z = {name: [] for name in BANK_NAMES}
    d_rep_z = {name: [] for name in REPRESENTATIONS}
    source_rows = []; tensor_shapes = {}; started = time.perf_counter()
    for batch_index, (names, images) in enumerate(loader, 1):
        labels_np = np.stack([parse_label(name) for name in names]); labels = torch.from_numpy(labels_np).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(images.cuda(non_blocking=True), labels, step=29275, hqmr_mode="full")
            key5_batch = model.hqmr.scale5.key(result["query_detail"]["context_feature"])
        item = result["stages"][2]["hqmr"]
        base = item["mixture"].detach(); predictions = base.argmax(1).cpu().numpy()
        anchors = result["target_detail"]["positive"].detach().float()
        if anchors.shape[-2:] != base.shape[-2:]: anchors = F.interpolate(anchors, base.shape[-2:], mode="nearest")
        anchors_np = anchors.bool().cpu().numpy()
        qbar_batch = torch.einsum("bqc,bqd->bcd", item["weights"].detach().float(), item["query4"].detach().float())
        if not tensor_shapes:
            tensor_shapes = {"R1_K4": list(item["key4"].shape[1:]), "R2_H5": list(key5_batch.shape[1:]),
                             "R3_key4": list(item["key4"].shape[1:]), "R3_qbar": list(qbar_batch.shape[1:]),
                             "R4_deep_backbone_F3": list(result["features"]["F3"].shape[1:])}
        for sample, name in enumerate(names):
            truth = np.asarray(Image.open(Path(args.train_gt_root) / f"{name}.png"))
            gt56 = F.interpolate(torch.from_numpy(truth)[None, None].float(), size=base.shape[-2:], mode="nearest")[0, 0].numpy().astype(np.uint8)
            selected = source_masks(predictions[sample], labels_np[sample], anchors_np[sample])
            a_embeddings = pool_masks([row["mask"] for row in selected], item["key4"][sample])
            clean_masks, clean_meta = [], []
            for row, embedding in zip(selected, a_embeddings):
                anatomy = gt_anatomy(row["mask"], gt56); mask = row.pop("mask")
                base_meta = {"image_id": name, **row, **anatomy, "source_label": row["class_id"],
                             "source_label_correct": int(row["class_id"] == anatomy["gt_majority_class"])}
                source_rows.append(base_meta); bank_meta["A"].append(base_meta); bank_z["A"].append(embedding)
                if anatomy["gt_majority_purity"] >= .5:
                    relabel = {**base_meta, "bank_class": anatomy["gt_majority_class"]}
                    bank_meta["B"].append(relabel); bank_z["B"].append(embedding)
                    clean_masks.append(mask & (gt56 == anatomy["gt_majority_class"])); clean_meta.append(relabel)
            clean_embeddings = pool_masks(clean_masks, item["key4"][sample])
            for meta, embedding in zip(clean_meta, clean_embeddings): bank_meta["C"].append(meta); bank_z["C"].append(embedding)
            d_masks, d_classes = largest_gt_regions(truth)
            if d_masks:
                representations = {
                    "R1_K4": pool_masks(d_masks, item["key4"][sample]),
                    "R2_H5": pool_masks(d_masks, key5_batch[sample]),
                    "R4_deep_backbone": pool_masks(d_masks, result["features"]["F3"][sample]),
                    "R3_query_conditioned_H4": pool_query_conditioned(
                        d_masks, d_classes, item["key4"][sample], qbar_batch[sample]),
                }
                for index, (mask, cls) in enumerate(zip(d_masks, d_classes)):
                    meta = {"image_id": name, "bank_class": cls, "class_id": cls,
                            "area": int(mask.sum()), "component_id": 1}
                    bank_meta["D"].append(meta); bank_z["D"].append(representations["R1_K4"][index])
                    for rep in REPRESENTATIONS: d_rep_z[rep].append(representations[rep][index])
        if batch_index % 100 == 0:
            print(json.dumps({"event": "train_build", "images": min(batch_index * args.batch_size, len(dataset)),
                              "A": len(bank_meta["A"]), "D": len(bank_meta["D"]),
                              "elapsed_s": time.perf_counter() - started}), flush=True)
    z_columns = [f"z_{i:03d}" for i in range(256)]
    frames = {name: with_embeddings(bank_meta[name], bank_z[name]) for name in BANK_NAMES}
    frames["A"].to_parquet(output / "sources/bankA_source_regions.parquet", index=False, compression="zstd")
    frames["B"].to_parquet(output / "sources/bankB_gt_relabel_sources.parquet", index=False, compression="zstd")
    frames["C"].to_parquet(output / "sources/bankC_gt_clean_pred_sources.parquet", index=False, compression="zstd")
    frames["D"].to_parquet(output / "sources/bankD_gt_tissue_sources.parquet", index=False, compression="zstd")
    previous = pd.read_parquet(Path(args.phase0_output) / "phase0/train_source_regions.parquet").sort_values(["image_id", "class_id"]).reset_index(drop=True)
    recreated = frames["A"].sort_values(["image_id", "class_id"]).reset_index(drop=True)
    source_identity = len(previous) == len(recreated) and previous[["image_id", "class_id", "component_id", "area", "anchor_count"]].equals(
        recreated[["image_id", "class_id", "component_id", "area", "anchor_count"]])
    embedding_diff = float(np.max(np.abs(previous[z_columns].to_numpy(np.float32) - recreated[z_columns].to_numpy(np.float32)))) if source_identity else None
    banks, reports, hashes = {}, {}, {}
    prior_bank = Path(args.phase0_output) / "phase0/static_prototype_bank.npy"
    shutil.copy2(prior_bank, output / "banks/bankA_K4.npy"); banks["A"] = np.load(prior_bank); hashes["A"] = sha256(output / "banks/bankA_K4.npy")
    recomputed_a, reports["A"] = build_bank(frames["A"].assign(bank_class=frames["A"].class_id), "bank_class", z_columns)
    if not source_identity or embedding_diff is None or embedding_diff > 1.0e-6:
        raise AssertionError(f"Bank A source recreation mismatch: identity={source_identity}, embedding_diff={embedding_diff}")
    bank_a_diff = float(np.max(np.abs(recomputed_a - banks["A"])))
    if bank_a_diff > 1.0e-6:
        raise AssertionError(f"Bank A prototype recreation mismatch: max_abs_diff={bank_a_diff}")
    for name in ("B", "C", "D"):
        banks[name], reports[name] = build_bank(frames[name], "bank_class", z_columns)
        hashes[name] = save_bank(output / f"banks/bank{name}_K4.npy", banks[name])
    d_labels = frames["D"].bank_class.to_numpy(int)
    for rep in REPRESENTATIONS:
        rep_frame = pd.DataFrame(np.stack(d_rep_z[rep]), columns=z_columns); rep_frame["bank_class"] = d_labels
        bank, report = build_bank(rep_frame, "bank_class", z_columns)
        if rep != "R1_K4": hashes[rep] = save_bank(output / f"banks/bankD_{rep}.npy", bank)
        reports[rep] = report
    class_mean = np.stack([l2_normalize(np.stack(d_rep_z["R1_K4"])[d_labels == cls].mean(0)) for cls in range(4)])
    hashes["E"] = save_bank(output / "banks/bankE_class_mean.npy", class_mean)
    write_json(output / "banks/all_bank_hashes.json", hashes)
    pd.DataFrame(source_rows).to_csv(output / "source_anatomy/source_label_contamination.csv", index=False)
    pd.DataFrame(source_rows)[["image_id", "class_id", "area", "gt_majority_class", "gt_majority_purity",
                               "gt_entropy", "gt_class_count", "cross_class_area_fraction", "ignore_fraction"]].to_csv(
                                   output / "source_anatomy/source_region_purity.csv", index=False)
    confusion = np.zeros((4, 4), np.int64)
    for row in source_rows: confusion[row["class_id"], row["gt_majority_class"]] += row["area"]
    write_csv(output / "source_anatomy/source_class_confusion.csv", [
        {"source_class": i, **{f"gt_{j}": int(confusion[i, j]) for j in range(4)}} for i in range(4)])
    source_frame = pd.DataFrame(source_rows)
    source_summary = {}
    for key, subset in [("overall", source_frame)] + [(str(cls), source_frame[source_frame.class_id == cls]) for cls in range(4)]:
        correct = subset.source_label_correct.astype(bool); mixed = subset.gt_majority_purity < .5
        source_summary[key] = {
            "components": int(len(subset)), "area": int(subset.area.sum()),
            "label_correct_component_rate": float(correct.mean()) if len(subset) else 0.0,
            "label_correct_area_rate": float(np.sum(subset.area * correct) / max(subset.area.sum(), 1)),
            "wrong_class_component_rate": float((~correct).mean()) if len(subset) else 0.0,
            "wrong_class_area_rate": float(np.sum(subset.area * ~correct) / max(subset.area.sum(), 1)),
            "mixed_component_rate": float(mixed.mean()) if len(subset) else 0.0,
            "mixed_area_rate": float(np.sum(subset.area * mixed) / max(subset.area.sum(), 1)),
            "mean_majority_purity": float(subset.gt_majority_purity.mean()) if len(subset) else 0.0,
            "area_weighted_majority_purity": float(np.sum(subset.area * subset.gt_majority_purity) / max(subset.area.sum(), 1)),
        }
    write_json(output / "source_anatomy/source_anatomy_summary.json", source_summary)
    identity_text = "# Existing Representation Tensor Identity\n\n" + "\n".join([
        f"- R1 K4: `stage3.hqmr.key4`, shape `{tensor_shapes['R1_K4']}`, HQMR H4 key projection output.",
        f"- R2 H5: `hqmr.scale5.key(query_detail.context_feature)`, shape `{tensor_shapes['R2_H5']}`, exact coarse HQMR key tensor.",
        f"- R3 query-conditioned H4: Stage3 `query4` + detached GCQM weights -> qbar `{tensor_shapes['R3_qbar']}`, applied to K4 `{tensor_shapes['R3_key4']}`.",
        f"- R4 deep backbone: `features.F3` before HQMR scale3 key projection, shape `{tensor_shapes['R4_deep_backbone_F3']}`.",
        "- All four tensors are frozen; masks use soft area downsampling, weighted mean pooling, and L2 normalization."])
    (output / "representation/tensor_identity_report.md").write_text(identity_text + "\n")
    marker = {"status": "FROZEN", "validation_accessed": False, "hashes": hashes,
              "bankA_source_identity": source_identity, "bankA_embedding_max_abs_diff": embedding_diff,
              "bankA_recomputed_max_abs_diff": bank_a_diff,
              "source_counts": {name: len(frames[name]) for name in BANK_NAMES},
              "source_anatomy_summary": source_summary,
              "tensor_shapes": tensor_shapes, "reports": reports,
              "seconds": time.perf_counter() - started}
    write_json(output / "banks/banks_frozen_before_validation.json", marker)
    print(json.dumps(marker, indent=2), flush=True)


def classify(embedding: np.ndarray, bank: np.ndarray) -> tuple[int, float, float, float, int]:
    score = np.einsum("d,ckd->ck", l2_normalize(embedding), l2_normalize(bank)).max(1)
    prototype_ids = np.einsum("d,ckd->ck", l2_normalize(embedding), l2_normalize(bank)).argmax(1)
    pred = int(np.argmax(score)); order = np.argsort(-score)
    return pred, float(score[pred]), float(score[order[1]]), float(score[pred] - score[order[1]]), int(prototype_ids[pred])


def similarity_anatomy(embedding: np.ndarray, bank: np.ndarray, true_class: int) -> dict:
    """Return similarities and margin with respect to the GT class, not the predicted class."""
    scores = np.einsum("d,ckd->ck", l2_normalize(embedding), l2_normalize(bank))
    class_scores = scores.max(1); prototype_ids = scores.argmax(1)
    prediction = int(class_scores.argmax())
    rivals = [cls for cls in range(4) if cls != true_class]
    rival_class = int(max(rivals, key=lambda cls: class_scores[cls]))
    own = float(class_scores[true_class]); rival = float(class_scores[rival_class])
    return {"nearest_class": prediction, "nearest_prototype_id": int(prototype_ids[prediction]),
            "own_similarity": own, "strongest_rival_class": rival_class,
            "strongest_rival_similarity": rival, "prototype_margin": own - rival}


def classify_r3(mask: np.ndarray, key4: torch.Tensor, qbar: torch.Tensor, bank: np.ndarray) -> tuple[int, np.ndarray]:
    embeddings = pool_query_conditioned([mask] * 4, list(range(4)), key4, qbar)
    scores = np.asarray([(embeddings[cls] @ bank[cls].T).max() for cls in range(4)])
    return int(np.argmax(scores)), embeddings


@torch.no_grad()
def infer_representations(model: HQMRNet, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    full, gates = [], []; reps = {name: [] for name in ("R1_K4", "R2_H5", "R4_deep_backbone")}; qbars = []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
            key5 = model.hqmr.scale5.key(output["query_detail"]["context_feature"])
        item = output["stages"][2]["hqmr"]
        full.append(resize_unflip(item["mixture"], original_hw, cam_flip).float().cpu())
        values = {"R1_K4": item["key4"][0], "R2_H5": key5[0], "R4_deep_backbone": output["features"]["F3"][0]}
        for name, feature in values.items():
            if cam_flip: feature = torch.flip(feature, dims=cam_flip)
            reps[name].append(feature.float().cpu())
        qbar = torch.einsum("qc,qd->cd", item["weights"][0].float(), item["query4"][0].float())
        qbars.append(qbar.cpu()); gates.append(output["deep_gate"].float().cpu())
    scores = normalize_cam(torch.stack(full).mean(0).numpy()); g = torch.stack(gates).mean(0).numpy()[0]
    label = presence(g); prediction = prediction_from_cam(scores, label, np.empty(original_hw))
    result = {name: torch.stack(value).mean(0) for name, value in reps.items()}
    result.update({"qbar": torch.stack(qbars).mean(0), "scores": scores, "prediction": prediction, "label": label})
    return result


def purity_stratum(purity: float) -> str:
    if purity < .5: return "P0"
    if purity < .7: return "P1"
    if purity < .9: return "P2"
    return "P3"


def classification_metrics(frame: pd.DataFrame, prediction: str, purity_min: float = .5) -> dict:
    value = frame[frame.purity >= purity_min].copy(); correct = value[prediction] == value.gt_majority_class
    component = float(correct.mean()) if len(value) else 0.0
    area = float(np.sum(value.area * correct) / max(value.area.sum(), 1))
    per_class = {}; confusion_component = np.zeros((4, 4), np.int64); confusion_area = np.zeros((4, 4), np.int64)
    for row in value.itertuples():
        pred = int(getattr(row, prediction)); confusion_component[row.gt_majority_class, pred] += 1; confusion_area[row.gt_majority_class, pred] += row.area
    for cls in range(4):
        subset = value[value.gt_majority_class == cls]; ok = subset[prediction] == cls
        per_class[str(cls)] = {"component_accuracy": float(ok.mean()) if len(subset) else 0.0,
                               "area_accuracy": float(np.sum(subset.area * ok) / max(subset.area.sum(), 1))}
    return {"components": len(value), "area": int(value.area.sum()), "component_accuracy": component,
            "area_weighted_accuracy": area, "macro_component_accuracy": float(np.mean([x["component_accuracy"] for x in per_class.values()])),
            "macro_area_accuracy": float(np.mean([x["area_accuracy"] for x in per_class.values()])),
            "per_class": per_class, "component_confusion": confusion_component.tolist(), "area_confusion": confusion_area.tolist()}


def gate_strength(delta: float) -> str:
    if delta >= .15: return "STRONG"
    if delta >= .07: return "MODERATE"
    return "WEAK"


def decide(label_delta: float, region_delta: float, k4: float, representation: dict,
           impurity_lt70: float, p3_best: float) -> tuple[str, str, list[str]]:
    target_strong = impurity_lt70 >= .50
    rep_strong = k4 < .70
    if target_strong and p3_best >= .70 and not rep_strong:
        return "TARGET_REGION_IMPURITY", "HIGH", ["target impurity", "whole-region assumption", "prototype geometry"]
    if label_delta >= .15 and region_delta < .07 and k4 >= .85:
        return "SOURCE_LABEL_CONTAMINATION", "HIGH", ["source labels", "source purity", "representation"]
    if region_delta >= .15 and k4 >= .85:
        return "SOURCE_REGION_CONTAMINATION", "HIGH", ["source regions", "source labels", "representation"]
    others = [value for key, value in representation.items() if key != "R1_K4"]
    if k4 < .70 and max(others) >= .85:
        return "K4_SPECIFIC_REPRESENTATION_LIMIT", "HIGH", ["K4", "alternative existing representation", "source contamination"]
    if all(value < .70 for value in representation.values()):
        return "EXISTING_REPRESENTATIONS_INSUFFICIENT", "HIGH", ["existing representations", "target impurity", "source contamination"]
    effects = [label_delta, region_delta]
    if (sum(delta >= .07 for delta in effects) >= 2 and max(effects) < .15) or (target_strong and rep_strong):
        return "MIXED_PROTO_EMBED_FAILURE", "HIGH", ["target impurity", "representation", "source contamination"]
    return "MIXED_PROTO_EMBED_FAILURE", "MEDIUM", ["representation", "target impurity", "source contamination"]


def report_text(result: dict) -> str:
    c = result["counterfactual"]; r = result["representation"]; p = result["target_purity"]
    sections = [
        ("1 Executive Diagnosis", f"**DECISION = {result['decision']}**  \n**CONFIDENCE = {result['confidence']}**"),
        ("2 Frozen CIRV-v1 Failure", "HQMR-v1 65.5724% mIoU; static CIRV 62.0893%; product fusion remains archived."),
        ("3 Exact Question", "Locate failure at source labels, source spatial purity, representation, or whole-region target validity."),
        ("4 Reproduction Gate", f"PASS={result['reproduction']['pass']}; base={100*result['reproduction']['base_mIoU']:.4f}%; M1={result['reproduction']['m1_count']}; prototype-only={100*result['reproduction']['prototype_component_accuracy']:.3f}%."),
        ("5 Causal Hypotheses", "H1 source label; H2 source region; H3 K4; H4 target impurity; H5 mixed."),
        ("6 Counterfactual Bank Design", "A weak-pred; B same masks GT relabel; C GT-clean intersection; D largest clean GT tissue; E class mean."),
        ("7 Bank A Weak-Pred", f"Area-weighted M1 accuracy={100*c['A']['area_weighted_accuracy']:.2f}%."),
        ("8 Bank B GT Relabel", f"Area-weighted M1 accuracy={100*c['B']['area_weighted_accuracy']:.2f}%."),
        ("9 Bank C GT-Clean Within Pred", f"Area-weighted M1 accuracy={100*c['C']['area_weighted_accuracy']:.2f}%."),
        ("10 Bank D GT Tissue", f"Area-weighted M1 accuracy={100*c['D']['area_weighted_accuracy']:.2f}%."),
        ("11 Source Label Contamination", f"A→B={100*result['causal_deltas']['A_to_B']:+.2f} pp ({result['causal_deltas']['source_label_strength']})."),
        ("12 Source Region Contamination", f"B→C={100*result['causal_deltas']['B_to_C']:+.2f} pp ({result['causal_deltas']['source_region_strength']})."),
        ("13 M1 Target Purity", f"M1 false area with purity<0.70={100*p['area_fraction_lt_070']:.2f}% ({p['strength']})."),
        ("14 Purity-Stratified Difficulty", json.dumps(result["purity_strata"], ensure_ascii=False)),
        ("15 A/B/C/D Accuracy", "See `counterfactual/*_metrics.json` for component, area, macro and confusion metrics."),
        ("16 Area-Weighted Accuracy", f"A/B/C/D = {100*c['A']['area_weighted_accuracy']:.2f}/{100*c['B']['area_weighted_accuracy']:.2f}/{100*c['C']['area_weighted_accuracy']:.2f}/{100*c['D']['area_weighted_accuracy']:.2f}%."),
        ("17 Per-Class Causal Deltas", "See `counterfactual/per_class_causal_table.csv`."),
        ("18 Clean K4 Tissue-Identity Ceiling", f"K4 clean GT-region area accuracy={100*r['R1_K4']:.2f}% ({result['k4_gate']})."),
        ("19 K4 Prototype Confusion", "See `prototype_anatomy/nearest_proto_confusion.csv`."),
        ("20 Prototype Margin", "Train clean, validation clean and M1 purity-stratified margins are archived."),
        ("21 Source-Target Shift", "See `prototype_anatomy/source_target_shift.csv`."),
        ("22 Existing Representation Matrix", json.dumps(r, ensure_ascii=False)),
        ("23 R1 K4", f"{100*r['R1_K4']:.2f}% area accuracy."),
        ("24 R2 H5", f"{100*r['R2_H5']:.2f}% area accuracy."),
        ("25 R3 Query-Conditioned H4", f"{100*r['R3_query_conditioned_H4']:.2f}% area accuracy."),
        ("26 R4 Deep Backbone", f"{100*r['R4_deep_backbone']:.2f}% area accuracy."),
        ("27 K4-Specific vs Global Representation Limit", result["representation_interpretation"]),
        ("28 Multi-Prototype Sanity", f"K4 minus class mean={100*result['multi_prototype_gain']:+.2f} pp; flag={result['multi_prototype_flag']}."),
        ("29 Whole-Region Oracle Reclassification Ceiling", json.dumps(result["oracle_headroom"], ensure_ascii=False)),
        ("30 Segmentation Headroom Attribution", json.dumps(result["headroom_attribution"], ensure_ascii=False)),
        ("31 Representative Cases", "Automatically selected cases are under `visualizations/`."),
        ("32 Decision Matrix", json.dumps(result["decision_matrix"], ensure_ascii=False)),
        ("33 Exact Root Cause", f"Primary={result['ranking'][0]}; Secondary={result['ranking'][1]}; Tertiary={result['ranking'][2]}."),
        ("34 What Is Preserved", "CCRA, HQMR-v1, hierarchical reconstruction, region-conditioned query update and M1 morphology-oracle conclusion."),
        ("35 What Is Archived", "CIRV-v1 product fusion, ratio calibration, CIRV Full25, CCBP, CP-HQMR, generic morphology and simple region suppression."),
        ("36 Exact Next Architecture Target", result["next_target"]),
        ("37 What Must NOT Be Done", "No Full25, no K/scale/EMA/fusion sweep, no learned verifier, and no HQMR decoder change."),
        ("38 Final Decision", f"DECISION = {result['decision']}  \nCONFIDENCE = {result['confidence']}"),
    ]
    text = "# CIRV Prototype–Embedding Failure Anatomy Audit Report\n\n"
    for title, body in sections: text += f"## {title}\n\n{body}\n\n"
    text += (f"**CIRV-v1 prototype-only accuracy is limited primarily by {result['ranking'][0]}. "
             f"Correcting source labels changes area-weighted M1 target accuracy from {100*c['A']['area_weighted_accuracy']:.2f}% to {100*c['B']['area_weighted_accuracy']:.2f}%; "
             f"spatially cleaning source regions changes it to {100*c['C']['area_weighted_accuracy']:.2f}%; fully GT-clean K4 prototypes achieve {100*c['D']['area_weighted_accuracy']:.2f}%. "
             f"The clean K4 GT-region classification ceiling is {100*r['R1_K4']:.2f}%. Therefore the next model should target {result['next_target']} rather than tune CIRV fusion.**\n")
    return text


def visualize_cases(output: Path, valroot: Path, morphology: Path, manifest: dict, targets: pd.DataFrame) -> None:
    candidates = targets[(targets.purity >= .9) & (targets.pred_A != targets.gt_majority_class)].sort_values("area", ascending=False).head(6)
    for row in candidates.itertuples():
        data = np.load(morphology / "cache" / manifest[row.image_id]); base = data["hqmr"]
        original = np.asarray(Image.open(valroot / "img" / f"{row.image_id}.png").convert("RGB")); truth = np.asarray(Image.open(valroot / "mask" / f"{row.image_id}.png"))
        labels, _ = ndimage.label(base == row.base_class, structure=STRUCTURE8); mask = labels == row.component_id
        overlay = original.copy(); overlay[mask] = (.4 * overlay[mask] + .6 * np.asarray([255, 255, 0])).astype(np.uint8)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3)); axes[0].imshow(original); axes[0].set_title("Image")
        axes[1].imshow(truth, vmin=0, vmax=4); axes[1].set_title("GT")
        axes[2].imshow(base, vmin=0, vmax=3); axes[2].set_title("HQMR")
        axes[3].imshow(overlay); axes[3].set_title(f"A={row.pred_A} D={row.pred_D} GT={row.gt_majority_class}")
        for axis in axes: axis.axis("off")
        fig.tight_layout(); fig.savefig(output / "visualizations" / f"{row.image_id}_c{row.base_class}_r{row.component_id}.png", dpi=140); plt.close(fig)


@torch.no_grad()
def run_evaluate(args, output: Path) -> None:
    marker_path = output / "banks/banks_frozen_before_validation.json"
    if not marker_path.exists(): raise RuntimeError("Validation firewall: banks not frozen")
    marker = json.loads(marker_path.read_text())
    if marker["validation_accessed"]: raise RuntimeError("Validation already accessed")
    if not marker["bankA_source_identity"] or marker["bankA_embedding_max_abs_diff"] > 1.0e-6:
        raise RuntimeError("Validation firewall: Bank A source recreation did not pass")
    banks = {name: np.load(output / f"banks/bank{name}_K4.npy") for name in BANK_NAMES}
    rep_banks = {"R1_K4": banks["D"]}
    for rep in REPRESENTATIONS[1:]: rep_banks[rep] = np.load(output / f"banks/bankD_{rep}.npy")
    class_mean = np.load(output / "banks/bankE_class_mean.npy")
    model = load_model(Path(args.hqmr_checkpoint)); valroot = Path(args.val_root); morphology = Path(args.morphology_output)
    cache_manifest = json.loads((morphology / "cache/prediction_manifest.json").read_text()); cache = {row["image_id"]: row["cache"] for row in cache_manifest}
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    targets, clean_regions, proto_rows, margin_rows = [], [], [], []
    train_d = pd.read_parquet(output / "sources/bankD_gt_tissue_sources.parquet")
    train_z_columns = [column for column in train_d.columns if column.startswith("z_")]
    for row in train_d.itertuples():
        embedding = np.asarray([getattr(row, column) for column in train_z_columns], np.float32)
        anatomy = similarity_anatomy(embedding, banks["D"], int(row.bank_class))
        margin_rows.append({"population": "train_clean_gt", "purity_stratum": "CLEAN",
                            "representation": "R1_K4", "class_id": int(row.bank_class),
                            "area": int(row.area), **anatomy})
    base_hist = []; oracle_hist = {name: [] for name in BANK_NAMES}; cache_mismatch = 0; started = time.perf_counter()
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; original = Image.open(valroot / "img" / f"{image_id}.png"); truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png"))
        result = infer_representations(model, image.cuda(non_blocking=True), (original.height, original.width)); base = result["prediction"]
        frozen = np.load(morphology / "cache" / cache[image_id]); cache_mismatch += int(not np.array_equal(base, frozen["hqmr"]))
        base_hist.append(foreground_confusion(truth, base)); oracle_prediction = {name: base.copy() for name in BANK_NAMES}
        for region in extract_regions(base):
            mask = region["mask"]; valid_truth = truth[mask]; valid_truth = valid_truth[valid_truth < 4]
            if not len(valid_truth) or np.any(mask & (truth == region["class_id"])): continue
            counts = np.bincount(valid_truth, minlength=4); majority = int(np.argmax(counts)); purity = int(counts[majority]) / region["area"]
            embeddings = {"R1_K4": pool_masks([mask], result["R1_K4"])[0],
                          "R2_H5": pool_masks([mask], result["R2_H5"])[0],
                          "R4_deep_backbone": pool_masks([mask], result["R4_deep_backbone"])[0]}
            row = {"image_id": image_id, "base_class": region["class_id"], "component_id": region["component_id"],
                   "area": region["area"], "gt_majority_class": majority, "purity": purity, "stratum": purity_stratum(purity)}
            for name in BANK_NAMES:
                pred, own, rival, margin, proto_id = classify(embeddings["R1_K4"], banks[name]); row[f"pred_{name}"] = pred
                if pred == majority: oracle_prediction[name][mask] = pred
                if name == "D":
                    anatomy = similarity_anatomy(embeddings["R1_K4"], banks[name], majority)
                    proto_rows.append({**row, "representation": "R1_K4", **anatomy})
                    margin_rows.append({"population": "m1_false_target", "purity_stratum": row["stratum"],
                                        "representation": "R1_K4", "class_id": majority,
                                        "area": row["area"], **anatomy})
            for rep in REPRESENTATIONS:
                if rep == "R3_query_conditioned_H4": pred, candidate_z = classify_r3(mask, result["R1_K4"], result["qbar"], rep_banks[rep])
                else: pred, *_ = classify(embeddings[rep], rep_banks[rep])
                row[f"pred_{rep}"] = pred
            mean_score = class_mean @ embeddings["R1_K4"]; row["pred_E"] = int(np.argmax(mean_score))
            targets.append(row)
        for name in BANK_NAMES: oracle_hist[name].append(foreground_confusion(truth, oracle_prediction[name]))
        gt_masks, gt_classes, component_ids = all_gt_regions(truth)
        if gt_masks:
            rep_z = {"R1_K4": pool_masks(gt_masks, result["R1_K4"]), "R2_H5": pool_masks(gt_masks, result["R2_H5"]),
                     "R4_deep_backbone": pool_masks(gt_masks, result["R4_deep_backbone"]),
                     "R3_query_conditioned_H4": pool_query_conditioned(gt_masks, gt_classes, result["R1_K4"], result["qbar"])}
            for item_index, (mask, cls, component_id) in enumerate(zip(gt_masks, gt_classes, component_ids)):
                row = {"image_id": image_id, "gt_class": cls, "component_id": component_id, "area": int(mask.sum())}
                for rep in REPRESENTATIONS:
                    pred, own, rival, margin, proto_id = classify(rep_z[rep][item_index], rep_banks[rep]); row[f"pred_{rep}"] = pred
                    margin_rows.append({"population": "validation_clean_gt", "representation": rep, "class_id": cls,
                                        "purity_stratum": "CLEAN", "area": row["area"],
                                        **similarity_anatomy(rep_z[rep][item_index], rep_banks[rep], cls)})
                row["pred_E"] = int(np.argmax(class_mean @ rep_z["R1_K4"][item_index])); clean_regions.append(row)
        if index % 100 == 0: print(json.dumps({"event": "validation", "images": index, "m1": len(targets), "cache_mismatch": cache_mismatch, "elapsed_s": time.perf_counter() - started}), flush=True)
    target = pd.DataFrame(targets); clean = pd.DataFrame(clean_regions); base_hist = np.stack(base_hist)
    reproduction_proto = float(np.mean(target.pred_A == target.gt_majority_class))
    reproduction = {"images": len(base_hist), "base_mIoU": scores_from_confusion(base_hist.sum(0))["mIoU"],
                    "cache_mismatch": cache_mismatch, "m1_count": len(target), "m1_area": int(target.area.sum()),
                    "prototype_component_accuracy": reproduction_proto}
    reproduction["pass"] = (len(base_hist) == 3418 and reproduction["base_mIoU"] == EXPECTED_BASE and cache_mismatch == 0 and
                            len(target) == EXPECTED_M1_COUNT and int(target.area.sum()) == EXPECTED_M1_AREA and
                            abs(reproduction_proto - EXPECTED_PROTO_COMPONENT) <= 1e-12)
    write_json(output / "provenance/reproduction_gate.json", reproduction)
    target.to_csv(output / "targets/m1_target_purity.csv", index=False)
    purity_rows = []
    for stratum in ("P0", "P1", "P2", "P3"):
        subset = target[target.stratum == stratum]; purity_rows.append({"stratum": stratum, "components": len(subset), "area": int(subset.area.sum()), "area_fraction": float(subset.area.sum() / target.area.sum())})
    write_csv(output / "targets/m1_purity_strata.csv", purity_rows); write_csv(output / "targets/target_area_distribution.csv", purity_rows)
    metrics = {}
    for name in BANK_NAMES:
        metrics[name] = classification_metrics(target, f"pred_{name}"); write_json(output / f"counterfactual/{name}_metrics.json", metrics[name])
    label_delta = metrics["B"]["area_weighted_accuracy"] - metrics["A"]["area_weighted_accuracy"]
    region_delta = metrics["C"]["area_weighted_accuracy"] - metrics["B"]["area_weighted_accuracy"]
    clean_delta = metrics["D"]["area_weighted_accuracy"] - metrics["C"]["area_weighted_accuracy"]
    causal = {"A_to_B": label_delta, "B_to_C": region_delta, "C_to_D": clean_delta,
              "source_label_strength": gate_strength(label_delta), "source_region_strength": gate_strength(region_delta)}
    write_csv(output / "counterfactual/causal_deltas.csv", [{"effect": key, "value": value} for key, value in causal.items()])
    per_class = []
    for cls in range(4):
        values = [metrics[name]["per_class"][str(cls)]["area_accuracy"] for name in BANK_NAMES]
        per_class.append({"class": cls, **{name: values[i] for i, name in enumerate(BANK_NAMES)},
                          "A_to_B": values[1]-values[0], "B_to_C": values[2]-values[1], "C_to_D": values[3]-values[2]})
    write_csv(output / "counterfactual/per_class_causal_table.csv", per_class)
    representation = {}
    rep_rows = []
    for rep in REPRESENTATIONS:
        frame = clean.rename(columns={"gt_class": "gt_majority_class", "pred_" + rep: "prediction", "area": "area"}); frame["purity"] = 1.0
        value = classification_metrics(frame, "prediction"); representation[rep] = value["area_weighted_accuracy"]
        rep_rows.append({"representation": rep, **value}); write_csv(output / f"representation/{rep}.csv", [value])
    write_csv(output / "representation/gt_region_representation_ceiling.csv", rep_rows)
    mean_frame = clean.rename(columns={"gt_class": "gt_majority_class", "pred_E": "prediction"}); mean_frame["purity"] = 1.0
    mean_metric = classification_metrics(mean_frame, "prediction")["area_weighted_accuracy"]
    multi_gain = representation["R1_K4"] - mean_metric
    purity_strata = {name: {bank: classification_metrics(target[target.stratum == name], f"pred_{bank}", 0.0)["area_weighted_accuracy"] for bank in BANK_NAMES} for name in ("P0", "P1", "P2", "P3")}
    p3_representation = {
        rep: classification_metrics(target[target.stratum == "P3"], f"pred_{rep}", 0.0)
        for rep in REPRESENTATIONS
    }
    write_json(output / "representation/p3_representation_metrics.json", p3_representation)
    p_lt70 = float(target.loc[target.purity < .7, "area"].sum() / target.area.sum()); impurity_strength = "STRONG" if p_lt70 >= .5 else "MODERATE" if p_lt70 >= .3 else "WEAK"
    oracle = {}; oracle_hist_sums = {}
    for name in BANK_NAMES:
        hist = np.stack(oracle_hist[name]); score = scores_from_confusion(hist.sum(0)); oracle[name] = score["mIoU"]; oracle_hist_sums[name] = score
        write_csv(output / f"headroom/oracle_headroom_{name}.csv", [{"bank": name, "mIoU": score["mIoU"], "delta_pp": 100*(score["mIoU"]-EXPECTED_BASE), "oracle_not_model_performance": True}])
    headroom = {"label_contribution_pp": 100*(oracle["B"]-oracle["A"]), "region_contribution_pp": 100*(oracle["C"]-oracle["B"]), "clean_source_contribution_pp": 100*(oracle["D"]-oracle["C"])}
    write_json(output / "headroom/headroom_attribution.json", headroom)
    pd.DataFrame(proto_rows).to_csv(output / "prototype_anatomy/prototype_margin.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(output / "prototype_anatomy/source_target_shift.csv", index=False)
    confusion = np.zeros((4, 4), np.int64)
    for row in proto_rows: confusion[row["gt_majority_class"], row["nearest_class"]] += row["area"]
    write_csv(output / "prototype_anatomy/nearest_proto_confusion.csv", [{"gt_class": i, **{f"pred_{j}": int(confusion[i,j]) for j in range(4)}} for i in range(4)])
    write_csv(output / "prototype_anatomy/occupancy_diversity.csv", [{"bank": key, "sha256": value} for key, value in marker["hashes"].items()])
    p3_best = max(value["area_weighted_accuracy"] for value in p3_representation.values())
    decision, confidence, ranking = decide(label_delta, region_delta, representation["R1_K4"], representation, p_lt70, p3_best)
    if not reproduction["pass"]: decision, confidence = "PROTO_EMBED_AUDIT_ENGINEERING_BLOCKED", "HIGH"
    next_targets = {"SOURCE_LABEL_CONTAMINATION": "robust prototype-source class verification",
                    "SOURCE_REGION_CONTAMINATION": "clean prototype-source region construction",
                    "K4_SPECIFIC_REPRESENTATION_LIMIT": "replace CIRV region embedding source",
                    "EXISTING_REPRESENTATIONS_INSUFFICIENT": "learn a dedicated region-semantic representation",
                    "TARGET_REGION_IMPURITY": "subregion splitting / region decomposition",
                    "MIXED_PROTO_EMBED_FAILURE": "jointly repair the ranked primary and secondary failure axes",
                    "PROTO_EMBED_AUDIT_ENGINEERING_BLOCKED": "repair audit engineering before any model work"}
    k4 = representation["R1_K4"]; k4_gate = "K4_REGION_IDENTITY_CAPABLE" if k4 >= .85 else "K4_REGION_IDENTITY_MODERATE" if k4 >= .70 else "K4_REGION_REPRESENTATION_LIMIT_STRONG"
    result = {"decision": decision, "confidence": confidence, "ranking": ranking, "next_target": next_targets[decision],
              "reproduction": reproduction, "counterfactual": metrics, "causal_deltas": causal,
              "target_purity": {"area_fraction_lt_070": p_lt70, "strength": impurity_strength},
              "purity_strata": purity_strata, "p3_representation": p3_representation,
              "source_anatomy": marker["source_anatomy_summary"],
              "representation": representation, "k4_gate": k4_gate,
              "representation_interpretation": "K4-specific" if k4 < .70 and max(v for k,v in representation.items() if k != "R1_K4") >= .85 else "global/mixed",
              "multi_prototype_gain": multi_gain, "multi_prototype_flag": "MULTIPROTOTYPE_GAIN_WEAK" if multi_gain < .03 else "MULTIPROTOTYPE_GAIN_PRESENT",
              "oracle_headroom": oracle, "headroom_attribution": headroom,
              "decision_matrix": {"label_delta_ge_015": label_delta >= .15, "region_delta_ge_015": region_delta >= .15,
                                  "k4_lt_070": k4 < .70, "all_rep_lt_070": all(v < .70 for v in representation.values()),
                                  "target_impurity_strong": p_lt70 >= .50, "p3_best_ge_070": p3_best >= .70,
                                  "p3_best_representation_accuracy": p3_best},
              "seconds": time.perf_counter() - started, "source_commit": git_commit(), "training_performed": False}
    write_json(output / "proto_embed_failure_audit_result.json", result)
    report = output / "report/CIRV_Prototype_Embedding_Failure_Anatomy_Audit_Report.md"; report.write_text(report_text(result), encoding="utf-8")
    visualize_cases(output, valroot, morphology, cache, target)
    marker["validation_accessed"] = True; marker["validation_completed"] = True; marker["decision"] = decision; write_json(marker_path, marker)
    print(json.dumps({"decision": decision, "confidence": confidence, "reproduction": reproduction,
                      "counterfactual_area": {k:v["area_weighted_accuracy"] for k,v in metrics.items()},
                      "representation": representation, "target_impurity_lt70": p_lt70}, indent=2), flush=True)
    print(f"DECISION = {decision}\nCONFIDENCE = {confidence}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--mode", required=True, choices=("build", "evaluate"))
    parser.add_argument("--trainroot", required=True); parser.add_argument("--train-gt-root", required=True); parser.add_argument("--train-gt-audit", required=True)
    parser.add_argument("--val-root", required=True); parser.add_argument("--hqmr-checkpoint", required=True)
    parser.add_argument("--phase0-output", required=True); parser.add_argument("--morphology-output", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--batch-size", type=int, default=12); parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); output = Path(args.output_dir).resolve()
    if args.mode == "build": run_build(args, output)
    else: run_evaluate(args, output)


if __name__ == "__main__": main()
