#!/usr/bin/env python3
"""PSCR-v1: frozen, GT-free prototype-source class reliability audit."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.cirv import STRUCTURE8, l2_normalize, spherical_kmeans
from network.hqmr_net import HQMRNet
from network.resnet38_cls import Net_CAM as SSHRCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_cirv_prototype_embedding_failure import (
    all_gt_regions, classification_metrics, gt_anatomy, largest_gt_regions,
    pool_masks, purity_stratum, source_masks,
)
from tools.eval_gcqm_full25_bcss_seed42 import (
    foreground_confusion, load_state, scores_from_confusion,
)
from tools.hqrf_phase0_io import sha256, write_csv, write_json
from tools.run_cirv_phase0_bcss_seed42 import infer_hqmr_cirv_inputs, parse_label


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
SSHR_SHA256 = "b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70"
BANK_A_SHA256 = "973bc5cfb7c58d64a0fcd1f88a51c677cfddf1b468ba36d42fb7ded4e75a6544"
ANATOMY_RESULT_SHA256 = "6e182214c0446451aa550c8f2091b3f2ebecba7278a249e2d927734b6ef6dfdd"
TRAIN_GT_AUDIT_SHA256 = "5a472941aa9d663a56eb4081e09ad339d1662be29dd0486f13fcc669d96d5d89"
EXPECTED_HQMR = 0.6557244403737567
EXPECTED_M1_COUNT = 4440
EXPECTED_M1_AREA = 8750254
EXPECTED_BANK_A = 0.07583647262076917
ORACLE_BANK_B = 0.2607467768395734
RING_ITERATIONS = 3
EPS = 1.0e-8
BRANCHES = ("CAM56", "CAM28_1", "CAM28_2", "CAMdeep")
POLICIES = ("P1", "P2", "P3")
ABLATIONS = ("A0", "A1", "A2", "A3", "A4")


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def setup(output: Path) -> None:
    for name in ("source_evidence", "policies", "metrics", "prototype_banks", "target_eval",
                 "compatibility", "visualizations", "provenance", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)


def normalize_cam_tensor(cam: torch.Tensor, size: tuple[int, int], unflip: bool = False) -> torch.Tensor:
    value = F.interpolate(cam.float(), size=size, mode="bilinear", align_corners=False)
    if unflip:
        value = torch.flip(value, dims=(3,))
    flat = value.flatten(2)
    low = flat.min(-1, keepdim=True).values[..., None]
    high = flat.max(-1, keepdim=True).values[..., None]
    return (value - low) / (high - low).clamp_min(1.0e-8)


def candidate_softmax(scores: np.ndarray, present: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, np.float32)
    present = np.asarray(present, bool)
    result = np.zeros_like(scores)
    index = np.flatnonzero(present)
    if not len(index):
        raise ValueError("Every training source must have an image-level candidate class")
    value = scores[..., index]
    value = np.exp(value - value.max(axis=-1, keepdims=True))
    result[..., index] = value / value.sum(axis=-1, keepdims=True)
    return result


def evidence_summary(scores: np.ndarray, present: np.ndarray, kind: int) -> dict:
    # scores: view x branch x kind(region/ring/contrast) x class
    probabilities = np.stack([candidate_softmax(scores[view, :, kind, :], present) for view in range(2)])
    branch_top = probabilities.argmax(-1)
    mean_probability = probabilities.mean(1)
    view_top = mean_probability.argmax(-1)
    votes = np.bincount(branch_top[0], minlength=4)
    consensus = int(np.argmax(votes))
    agreement = float(votes[consensus] / len(BRANCHES))
    order = np.argsort(-mean_probability[0])
    margin = float(mean_probability[0, order[0]] - mean_probability[0, order[1]])
    return {
        "probabilities": probabilities, "branch_top": branch_top,
        "consensus": consensus, "agreement": agreement, "margin": margin,
        "view_top_original": int(view_top[0]), "view_top_flip": int(view_top[1]),
        "view_consistent": bool(view_top[0] == view_top[1]),
        "mean_probability_original": mean_probability[0],
    }


def policy_decisions(scores: np.ndarray, base_classes: np.ndarray,
                     present: np.ndarray, policy: str) -> pd.DataFrame:
    rows = []
    for index in range(len(base_classes)):
        kind = 2
        summary = evidence_summary(scores[index], present[index], kind)
        base = int(base_classes[index]); candidate = summary["consensus"]
        action = "KEEP"
        if policy == "P1":
            if candidate != base and summary["agreement"] == 1.0 and summary["view_consistent"]:
                action = "RELABEL"
        elif policy == "P2":
            if (candidate != base and summary["agreement"] >= .75 and summary["margin"] >= .20
                    and summary["view_consistent"]):
                action = "RELABEL"
        elif policy == "P3":
            if candidate == base and summary["agreement"] >= .75:
                action = "KEEP"
            elif (candidate != base and summary["agreement"] >= .75 and summary["margin"] >= .20
                  and summary["view_consistent"]):
                action = "RELABEL"
            else:
                action = "REJECT"
        else:
            raise ValueError(policy)
        final_class = candidate if action == "RELABEL" else base if action == "KEEP" else -1
        base_probability = float(summary["mean_probability_original"][base])
        rival = float(np.max(np.delete(summary["mean_probability_original"], base)))
        rows.append({"source_index": index, "policy": policy, "action": action,
                     "base_class": base, "candidate_class": candidate, "final_class": final_class,
                     "agreement": summary["agreement"], "margin": summary["margin"],
                     "view_consistent": summary["view_consistent"],
                     "agreement_base": float(np.mean(summary["branch_top"][0] == base)),
                     "p_cam_base": base_probability, "base_margin": base_probability - rival,
                     "branch_votes_original": json.dumps(summary["branch_top"][0].tolist()),
                     "branch_votes_flip": json.dumps(summary["branch_top"][1].tolist())})
    return pd.DataFrame(rows)


def ablation_decisions(scores: np.ndarray, base_classes: np.ndarray,
                       present: np.ndarray, name: str) -> pd.DataFrame:
    rows = []
    for index, base_value in enumerate(base_classes):
        base = int(base_value); action = "KEEP"; candidate = base
        if name not in ("A0", "A1"):
            kind = 0 if name == "A2" else 2
            summary = evidence_summary(scores[index], present[index], kind)
            candidate = summary["consensus"]
            use_flip = name == "A4"
            if (candidate != base and summary["agreement"] >= .75 and summary["margin"] >= .20
                    and (summary["view_consistent"] or not use_flip)):
                action = "RELABEL"
        rows.append({"source_index": index, "policy": name, "action": action,
                     "base_class": base, "candidate_class": candidate,
                     "final_class": candidate if action == "RELABEL" else base})
    return pd.DataFrame(rows)


def bbox_geometry(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if not len(xs):
        return {"bbox_y0": -1, "bbox_x0": -1, "bbox_y1": -1, "bbox_x1": -1,
                "aspect_ratio": 0.0, "bbox_fill": 0.0, "context_ratio": 0.0}
    y0, y1, x0, x1 = int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)
    height, width = y1 - y0, x1 - x0
    ring = ndimage.binary_dilation(mask, structure=STRUCTURE8, iterations=RING_ITERATIONS) & ~mask
    return {"bbox_y0": y0, "bbox_x0": x0, "bbox_y1": y1, "bbox_x1": x1,
            "aspect_ratio": width / max(height, 1), "bbox_fill": float(mask.sum() / max(height * width, 1)),
            "context_ratio": float(mask.sum() / max(mask.sum() + ring.sum(), 1))}


def build_bank(embeddings: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    banks, reports = [], []
    for cls in range(4):
        values = embeddings[labels == cls]
        if len(values) < 4:
            raise RuntimeError(f"Class {cls} has insufficient prototype sources: {len(values)}")
        centers, _, report = spherical_kmeans(values, 4, seed=42)
        report["class_id"] = cls; banks.append(centers); reports.append(report)
    return np.stack(banks).astype(np.float32), reports


def save_bank(path: Path, bank: np.ndarray) -> str:
    np.savez_compressed(path, prototypes=bank.astype(np.float32)); return sha256(path)


def classify(embedding: np.ndarray, bank: np.ndarray) -> int:
    scores = np.einsum("d,ckd->ck", l2_normalize(embedding), l2_normalize(bank)).max(1)
    return int(np.argmax(scores))


def verify_paths(args) -> None:
    expected = [(args.hqmr_checkpoint, HQMR_SHA256), (args.sshr_checkpoint, SSHR_SHA256),
                (Path(args.anatomy_output) / "proto_embed_failure_audit_result.json", ANATOMY_RESULT_SHA256),
                (Path(args.phase0_output) / "phase0/static_prototype_bank.npy", BANK_A_SHA256),
                (args.train_gt_audit, TRAIN_GT_AUDIT_SHA256)]
    for path, digest in expected:
        if sha256(Path(path)) != digest:
            raise AssertionError(f"Frozen artifact mismatch: {path}")


@torch.no_grad()
def run_extract(args, output: Path) -> None:
    setup(output); verify_paths(args)
    config_path = ROOT / "configs/audits/pscr_v1.yaml"
    manifest = {
        "audit": "PSCR-v1", "zero_training": True, "parameter_updates": 0,
        "rules_preregistered_before_gt_evaluation": True, "gt_loaded": False,
        "source_commit": commit(), "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path), "hqmr_sha256": HQMR_SHA256,
        "sshr_sha256": SSHR_SHA256, "ring_iterations": RING_ITERATIONS,
        "cam_branches": list(BRANCHES), "views": ["original", "horizontal_flip"],
        "decision_evaluation_physical_separation": True,
    }
    write_json(output / "manifest.json", manifest)
    (output / "provenance/source_commit.txt").write_text(commit() + "\n", encoding="utf-8")
    (output / "provenance/source_diff.patch").write_text(
        subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True), encoding="utf-8")

    hqmr = HQMRNet().cuda(); hqmr.load_state_dict(load_state(Path(args.hqmr_checkpoint)), strict=True); hqmr.eval()
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(Path(args.sshr_checkpoint)), strict=True); sshr.eval()
    for model in (hqmr, sshr):
        for parameter in model.parameters(): parameter.requires_grad_(False)
    dataset = Stage1_InferDataset(args.trainroot, img_size=224)
    if len(dataset) != 23422: raise AssertionError("Expected 23422 BCSS training patches")
    loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    metadata, embeddings, packed_masks, score_blocks = [], [], [], []
    started = time.perf_counter()
    for batch_index, (names, images_cpu) in enumerate(loader, 1):
        images = images_cpu.cuda(non_blocking=True); labels_np = np.stack([parse_label(name) for name in names])
        labels = torch.from_numpy(labels_np).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = hqmr(images, labels, step=29275, hqmr_mode="full")
            cams_original = sshr.forward_cam(images)[:4]
            cams_flip = sshr.forward_cam(torch.flip(images, dims=(3,)))[:4]
        item = result["stages"][2]["hqmr"]; base = item["mixture"].detach()
        predictions = base.argmax(1).cpu().numpy(); anchors = result["target_detail"]["positive"].detach().float()
        if anchors.shape[-2:] != base.shape[-2:]: anchors = F.interpolate(anchors, base.shape[-2:], mode="nearest")
        anchors_np = anchors.bool().cpu().numpy(); hw = tuple(base.shape[-2:])
        cam_views = []
        for values, unflip in ((cams_original, False), (cams_flip, True)):
            cam_views.append(torch.stack([normalize_cam_tensor(cam, hw, unflip) for cam in values], 1).cpu().numpy())
        # view x batch x branch x class x H x W
        cam_views = np.stack(cam_views)
        for sample, name in enumerate(names):
            selected = source_masks(predictions[sample], labels_np[sample], anchors_np[sample])
            z = pool_masks([row["mask"] for row in selected], item["key4"][sample])
            for row, embedding in zip(selected, z):
                mask = row["mask"].astype(bool)
                ring = ndimage.binary_dilation(mask, structure=STRUCTURE8, iterations=RING_ITERATIONS) & ~mask
                scores = np.zeros((2, 4, 3, 4), np.float32)
                for view in range(2):
                    for branch in range(4):
                        value = cam_views[view, sample, branch]
                        scores[view, branch, 0] = value[:, mask].mean(1)
                        scores[view, branch, 1] = value[:, ring].mean(1) if ring.any() else 0.0
                        scores[view, branch, 2] = scores[view, branch, 0] - scores[view, branch, 1]
                summary = evidence_summary(scores, labels_np[sample].astype(bool), 2)
                geometry = bbox_geometry(mask); index = len(metadata)
                metadata.append({"source_index": index, "image_id": name, "region_id": f"{name}:c{row['class_id']}:r{row['component_id']}",
                                 "base_class": row["class_id"], "region_area": row["area"],
                                 "component_id": row["component_id"], "anchor_count": row["anchor_count"],
                                 "image_level_labels": "".join(map(str, labels_np[sample].astype(int).tolist())),
                                 "ring_area": int(ring.sum()), **geometry,
                                 "contrast_consensus_class": summary["consensus"],
                                 "contrast_agreement": summary["agreement"], "contrast_margin": summary["margin"],
                                 "view_consistent": summary["view_consistent"]})
                embeddings.append(embedding); packed_masks.append(np.packbits(mask.reshape(-1)))
                score_blocks.append(scores)
        if batch_index % 100 == 0:
            print(json.dumps({"event": "extract", "images": min(batch_index * 20, len(dataset)),
                              "sources": len(metadata), "elapsed_s": time.perf_counter() - started}), flush=True)
    frame = pd.DataFrame(metadata); z = np.stack(embeddings).astype(np.float32); masks = np.stack(packed_masks)
    scores = np.stack(score_blocks).astype(np.float32)
    # The decision path compares against the original GT-free CIRV source archive.
    prior = pd.read_parquet(Path(args.phase0_output) / "phase0/train_source_regions.parquet")
    keys_old = prior[["image_id", "class_id", "component_id", "area", "anchor_count"]].rename(
        columns={"class_id": "base_class", "area": "region_area"}).sort_values(["image_id", "base_class"]).reset_index(drop=True)
    order = np.lexsort((frame.base_class.to_numpy(), frame.image_id.to_numpy()))
    keys_new = frame.iloc[order][["image_id", "base_class", "component_id", "region_area", "anchor_count"]].reset_index(drop=True)
    z_columns = [f"z_{i:03d}" for i in range(256)]
    prior_sorted = prior.sort_values(["image_id", "class_id"]).reset_index(drop=True)
    source_identity = keys_old.equals(keys_new)
    embedding_diff = float(np.max(np.abs(prior_sorted[z_columns].to_numpy(np.float32) - z[order]))) if source_identity else None
    if not source_identity or embedding_diff is None or embedding_diff > 1.0e-6:
        raise AssertionError(f"Bank A source reproduction failed: {source_identity=}, {embedding_diff=}")
    frame.to_csv(output / "source_evidence/source_table.csv", index=False)
    np.savez_compressed(output / "source_evidence/cam_scores.npz", scores=scores)
    np.savez_compressed(output / "source_evidence/embeddings.npz", K4=z)
    np.savez_compressed(output / "source_evidence/source_masks.npz", packed=masks, shape=np.asarray(hw))
    view = frame[["source_index", "image_id", "base_class", "contrast_consensus_class",
                  "contrast_agreement", "contrast_margin", "view_consistent"]]
    view.to_csv(output / "source_evidence/view_consistency.csv", index=False)
    marker = {"status": "SOURCE_EVIDENCE_FROZEN", "gt_accessed": False, "validation_accessed": False,
              "source_identity": source_identity, "embedding_max_abs_diff": embedding_diff,
              "sources": len(frame), "cam_scores_sha256": sha256(output / "source_evidence/cam_scores.npz"),
              "embeddings_sha256": sha256(output / "source_evidence/embeddings.npz"),
              "source_table_sha256": sha256(output / "source_evidence/source_table.csv"),
              "seconds": time.perf_counter() - started}
    write_json(output / "provenance/source_evidence_frozen.json", marker)
    print(json.dumps(marker, indent=2), flush=True)


def unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed, count=shape[0] * shape[1]).reshape(shape).astype(bool)


def source_policy_metrics(decision: pd.DataFrame, truth: np.ndarray, base: np.ndarray,
                          areas: np.ndarray) -> dict:
    accepted = decision.action != "REJECT"; changed = decision.action == "RELABEL"
    final = decision.final_class.to_numpy(int); base_correct = base == truth; final_correct = final == truth
    rejected = ~accepted
    return {
        "sources": int(len(decision)), "accepted": int(accepted.sum()), "coverage": float(accepted.mean()),
        "source_accuracy_component": float(final_correct[accepted].mean()) if accepted.any() else 0.0,
        "source_accuracy_area": float(np.sum(areas[accepted] * final_correct[accepted]) / max(areas[accepted].sum(), 1)),
        "changed": int(changed.sum()),
        "changed_region_precision": float(final_correct[changed].mean()) if changed.any() else 0.0,
        "changed_region_precision_area": float(np.sum(areas[changed] * final_correct[changed]) / max(areas[changed].sum(), 1)),
        "harm_rate": float(np.sum(base_correct & ~final_correct & accepted) / max(base_correct.sum(), 1)),
        "recovery_rate": float(np.sum(~base_correct & final_correct & accepted) / max((~base_correct).sum(), 1)),
        "reject_count": int(rejected.sum()), "reject_error_rate": float((~base_correct[rejected]).mean()) if rejected.any() else 0.0,
        "kept_error_rate": float((~base_correct[accepted]).mean()) if accepted.any() else 0.0,
    }


def safe_auroc(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float | None:
    return float(roc_auc_score(y, score, sample_weight=weight)) if len(np.unique(y)) == 2 else None


def append_population(metadata: list[dict], embeddings: list[np.ndarray], population: str,
                      image_id: str, class_id: int, embedding: np.ndarray,
                      area_fraction: float, aspect: float, context_ratio: float) -> None:
    metadata.append({"population": population, "image_id": image_id, "class_id": int(class_id),
                     "embedding_norm": float(np.linalg.norm(embedding)), "area_fraction": float(area_fraction),
                     "log_area_fraction": float(np.log(max(area_fraction, EPS))), "aspect_ratio": float(aspect),
                     "context_ratio": float(context_ratio)})
    embeddings.append(np.asarray(embedding, np.float32))


def run_source_evaluate(args, output: Path) -> None:
    marker_path = output / "provenance/source_evidence_frozen.json"
    marker = json.loads(marker_path.read_text())
    if marker["gt_accessed"] or marker["validation_accessed"]: raise RuntimeError("Source firewall already crossed")
    frame = pd.read_csv(output / "source_evidence/source_table.csv")
    scores = np.load(output / "source_evidence/cam_scores.npz")["scores"]
    embeddings = np.load(output / "source_evidence/embeddings.npz")["K4"]
    mask_archive = np.load(output / "source_evidence/source_masks.npz"); packed, shape = mask_archive["packed"], tuple(mask_archive["shape"])
    present = np.stack([[int(value) for value in text] for text in frame.image_level_labels.astype(str).str.zfill(4)]).astype(bool)
    base = frame.base_class.to_numpy(int)

    # Policies and banks are frozen before the first GT file is opened.
    decisions = {name: policy_decisions(scores, base, present, name) for name in POLICIES}
    ablations = {name: ablation_decisions(scores, base, present, name) for name in ABLATIONS}
    banks, reports, hashes = {}, {}, {}
    prior_bank = np.load(Path(args.phase0_output) / "phase0/static_prototype_bank.npy")
    if sha256(Path(args.phase0_output) / "phase0/static_prototype_bank.npy") != BANK_A_SHA256: raise AssertionError("Bank A mismatch")
    banks["A"] = prior_bank; hashes["A"] = save_bank(output / "prototype_banks/bank_a.npz", prior_bank)
    for name, decision in decisions.items():
        accepted = decision.action != "REJECT"; labels = decision.final_class.to_numpy(int)
        banks[name], reports[name] = build_bank(embeddings[accepted], labels[accepted])
        hashes[name] = save_bank(output / f"prototype_banks/bank_{name.lower()}.npz", banks[name])
        decision.to_csv(output / f"policies/{name.lower()}_results.csv", index=False)
    ablation_banks = {}
    for name, decision in ablations.items():
        labels = decision.final_class.to_numpy(int); ablation_banks[name], reports[name] = build_bank(embeddings, labels)
    np.savez_compressed(output / "prototype_banks/ablation_banks.npz", **ablation_banks)
    bank_marker = {"status": "BANKS_FROZEN_BEFORE_GT_AND_VALIDATION", "gt_accessed": False,
                   "validation_accessed": False, "hashes": hashes, "reports": reports,
                   "policy_config_sha256": sha256(ROOT / "configs/audits/pscr_v1.yaml")}
    write_json(output / "prototype_banks/banks_frozen.json", bank_marker)

    # Evaluation path starts here. GT never flows back into decisions or banks.
    truth_class = np.empty(len(frame), np.int64); purity = np.empty(len(frame), np.float32)
    compatibility_meta, compatibility_z = [], []; d_geometry = {}
    frame_groups = frame.groupby("image_id").groups
    for image_id, indices_value in frame_groups.items():
        indices = list(indices_value); truth_full = np.asarray(Image.open(Path(args.train_gt_root) / f"{image_id}.png"))
        truth56 = F.interpolate(torch.from_numpy(truth_full.copy())[None, None].float(), size=shape,
                                mode="nearest")[0, 0].numpy().astype(np.uint8)
        for index in indices:
            mask = unpack_mask(packed[index], shape); anatomy = gt_anatomy(mask, truth56)
            truth_class[index] = anatomy["gt_majority_class"]; purity[index] = anatomy["gt_majority_purity"]
            append_population(compatibility_meta, compatibility_z, "HQMR_SOURCE_A", image_id,
                              truth_class[index], embeddings[index], mask.mean(),
                              frame.iloc[index].aspect_ratio, frame.iloc[index].context_ratio)
            for name in POLICIES:
                if decisions[name].iloc[index].action != "REJECT":
                    append_population(compatibility_meta, compatibility_z, f"VERIFIED_{name}", image_id,
                                      truth_class[index], embeddings[index], mask.mean(),
                                      frame.iloc[index].aspect_ratio, frame.iloc[index].context_ratio)
        d_masks, d_classes = largest_gt_regions(truth_full)
        for mask, cls in zip(d_masks, d_classes):
            mask56 = F.interpolate(torch.from_numpy(mask.copy())[None, None].float(), size=shape, mode="area")[0, 0].numpy() >= .5
            d_geometry[(image_id, cls)] = (bbox_geometry(mask56), float(mask.mean()))

    eval_labels = pd.DataFrame({"source_index": frame.source_index, "gt_class": truth_class, "gt_purity": purity,
                                "base_correct": base == truth_class})
    eval_labels.to_csv(output / "source_evidence/source_evaluation_labels.csv", index=False)
    areas = frame.region_area.to_numpy(int)
    metric_payload, per_class = {}, []
    baseline = pd.DataFrame({"action": "KEEP", "final_class": base})
    metric_payload["A"] = source_policy_metrics(baseline, truth_class, base, areas)
    for name, decision in decisions.items():
        metric_payload[name] = source_policy_metrics(decision, truth_class, base, areas)
        for cls in range(4):
            select = base == cls; changed = (decision.action.to_numpy() == "RELABEL") & select
            accepted = (decision.action.to_numpy() != "REJECT") & select
            final = decision.final_class.to_numpy(int); correct = final == truth_class
            per_class.append({"policy": name, "base_class": cls, "sources": int(select.sum()),
                              "source_accuracy": float(correct[accepted].mean()) if accepted.any() else 0.0,
                              "relabel_precision": float(correct[changed].mean()) if changed.any() else 0.0,
                              "relabel_recall_of_wrong": float(np.sum(changed & correct) / max(np.sum(select & (base != truth_class)), 1)),
                              "kept_source_count": int(accepted.sum()), "rejected_source_count": int(np.sum(select & ~accepted))})
    write_json(output / "metrics/source_relabel_metrics.json", metric_payload)
    write_csv(output / "metrics/per_class_metrics.csv", per_class)
    confusion_rows = []
    for name, decision in decisions.items():
        accepted = decision.action.to_numpy() != "REJECT"; final = decision.final_class.to_numpy(int)
        matrix = np.zeros((4, 4), np.int64)
        for true, pred in zip(truth_class[accepted], final[accepted]): matrix[true, pred] += 1
        for cls in range(4): confusion_rows.append({"policy": name, "gt_class": cls, **{f"pred_{j}": int(matrix[cls,j]) for j in range(4)}})
    write_csv(output / "metrics/confusion_matrices.csv", confusion_rows)
    error = (base != truth_class).astype(int); reference = decisions["P2"]
    auroc = {
        "agreement_base": {"component": safe_auroc(error, 1-reference.agreement_base.to_numpy()),
                           "area": safe_auroc(error, 1-reference.agreement_base.to_numpy(), areas)},
        "p_cam_base": {"component": safe_auroc(error, 1-reference.p_cam_base.to_numpy()),
                       "area": safe_auroc(error, 1-reference.p_cam_base.to_numpy(), areas)},
        "base_margin": {"component": safe_auroc(error, -reference.base_margin.to_numpy()),
                        "area": safe_auroc(error, -reference.base_margin.to_numpy(), areas)},
        "interpretation": "AUROC<0.70 means source errors cannot be reliably identified"
    }
    write_json(output / "metrics/source_error_auroc.json", auroc)

    # C/D oracle populations are evaluation-only and cannot alter any bank above.
    anatomy = Path(args.anatomy_output); c_frame = pd.read_parquet(anatomy / "sources/bankC_gt_clean_pred_sources.parquet")
    d_frame = pd.read_parquet(anatomy / "sources/bankD_gt_tissue_sources.parquet")
    z_cols = [f"z_{i:03d}" for i in range(256)]
    c_map = {(row.image_id, int(row.class_id)): row for row in c_frame.itertuples()}
    for index, row in frame.iterrows():
        if purity[index] < .5: continue
        key = (row.image_id, int(row.base_class)); item = c_map.get(key)
        if item is None: continue
        mask = unpack_mask(packed[index], shape)
        truth_full = np.asarray(Image.open(Path(args.train_gt_root) / f"{row.image_id}.png"))
        truth56 = F.interpolate(torch.from_numpy(truth_full.copy())[None,None].float(), size=shape, mode="nearest")[0,0].numpy().astype(np.uint8)
        clean = mask & (truth56 == truth_class[index]); geometry = bbox_geometry(clean)
        z = np.asarray([getattr(item, col) for col in z_cols], np.float32)
        append_population(compatibility_meta, compatibility_z, "GT_CLEAN_HQMR_SHAPED_C", row.image_id,
                          truth_class[index], z, clean.mean(), geometry["aspect_ratio"], geometry["context_ratio"])
    for item in d_frame.itertuples():
        key = (item.image_id, int(item.bank_class))
        if key not in d_geometry:
            truth_full = np.asarray(Image.open(Path(args.train_gt_root) / f"{item.image_id}.png"))
            for mask, cls in zip(*largest_gt_regions(truth_full)):
                mask56 = F.interpolate(torch.from_numpy(mask.copy())[None, None].float(), size=shape,
                                       mode="area")[0, 0].numpy() >= .5
                d_geometry[(item.image_id, cls)] = (bbox_geometry(mask56), float(mask.mean()))
        geometry, area_fraction = d_geometry[key]
        z = np.asarray([getattr(item, col) for col in z_cols], np.float32)
        append_population(compatibility_meta, compatibility_z, "LARGEST_GT_TISSUE_D", item.image_id,
                          item.bank_class, z, area_fraction, geometry["aspect_ratio"], geometry["context_ratio"])
    z_frame = pd.DataFrame(np.stack(compatibility_z), columns=[f"z_{i:03d}" for i in range(256)])
    pd.concat([pd.DataFrame(compatibility_meta), z_frame], axis=1).to_parquet(
        output / "compatibility/source_populations.parquet", index=False, compression="zstd")
    marker["gt_accessed"] = True; write_json(marker_path, marker)
    bank_marker["gt_accessed"] = True; write_json(output / "prototype_banks/banks_frozen.json", bank_marker)
    print(json.dumps({"source_metrics": metric_payload, "auroc": auroc}, indent=2), flush=True)


def target_metrics(frame: pd.DataFrame, prediction: str) -> dict:
    return classification_metrics(frame, prediction, purity_min=.5)


def decision_from_recovery(recovery: float, accuracy: float, precision: float,
                           architecture_checks: dict) -> str:
    if recovery >= .75 and precision >= .70 and all(architecture_checks.values()): return "ARCHITECTURE_READY"
    if recovery >= .60: return "STRONG_GO"
    if recovery >= .40: return "GO"
    if recovery >= .25: return "WEAK"
    return "NOGO"


def compatibility_finalize(output: Path, target_meta: list[dict], target_z: list[np.ndarray]) -> dict:
    source = pd.read_parquet(output / "compatibility/source_populations.parquet")
    z_cols = [f"z_{i:03d}" for i in range(256)]
    target = pd.concat([pd.DataFrame(target_meta), pd.DataFrame(np.stack(target_z), columns=z_cols)], axis=1)
    combined = pd.concat([source, target], ignore_index=True)
    clean = combined[combined.population == "GT_CLEAN_HQMR_SHAPED_C"]
    centroids = {cls: l2_normalize(clean.loc[clean.class_id == cls, z_cols].to_numpy(np.float32).mean(0)) for cls in range(4)}
    z = combined[z_cols].to_numpy(np.float32)
    combined["within_class_centroid_distance"] = [1.0 - float(l2_normalize(v) @ centroids[int(cls)]) for v, cls in zip(z, combined.class_id)]
    combined.to_parquet(output / "compatibility/source_target_embeddings.parquet", index=False, compression="zstd")
    stats = {}
    summary_rows = []
    for (population, cls), group in combined.groupby(["population", "class_id"]):
        stats[f"{population}/C{cls}"] = {column: {"mean": float(group[column].mean()), "median": float(group[column].median())}
                                           for column in ("within_class_centroid_distance", "embedding_norm", "area_fraction", "aspect_ratio", "context_ratio")}
        summary_rows.append({"population": population, "class_id": int(cls), "count": len(group),
                             **{f"{column}_mean": float(group[column].mean()) for column in ("within_class_centroid_distance", "embedding_norm", "area_fraction", "aspect_ratio", "context_ratio")},
                             **{f"{column}_median": float(group[column].median()) for column in ("within_class_centroid_distance", "embedding_norm", "area_fraction", "aspect_ratio", "context_ratio")}})
    write_csv(output / "compatibility/source_target_shift.csv", summary_rows)
    c_dist = combined.loc[combined.population == "GT_CLEAN_HQMR_SHAPED_C", "within_class_centroid_distance"].mean()
    d_dist = combined.loc[combined.population == "LARGEST_GT_TISSUE_D", "within_class_centroid_distance"].mean()
    t_dist = combined.loc[combined.population == "M1_TARGET", "within_class_centroid_distance"].mean()
    result = {"statistics": stats, "summary": {"C_distance": float(c_dist), "D_distance": float(d_dist),
              "M1_distance": float(t_dist), "largest_GT_vs_HQMR_shaped_shift": float(d_dist-c_dist),
              "M1_vs_HQMR_shaped_shift": float(t_dist-c_dist),
              "source_target_compatibility_flag": bool((d_dist-c_dist) > .05 or (t_dist-c_dist) > .05)}}
    write_json(output / "compatibility/embedding_statistics.json", result)
    return result


@torch.no_grad()
def generate_visualizations(args, output: Path, hqmr: HQMRNet, best_policy: str) -> dict:
    source = pd.read_csv(output / "source_evidence/source_table.csv")
    labels = pd.read_csv(output / "source_evidence/source_evaluation_labels.csv")
    best = pd.read_csv(output / f"policies/{best_policy.lower()}_results.csv")
    relabel_candidates = {name: pd.read_csv(output / f"policies/{name.lower()}_results.csv") for name in POLICIES}
    relabel_policy = max(POLICIES, key=lambda name: int((relabel_candidates[name].action == "RELABEL").sum()))
    relabel_decision = relabel_candidates[relabel_policy]
    p3 = pd.read_csv(output / "policies/p3_results.csv")
    frame = source.merge(labels, on="source_index").merge(best, on="source_index", suffixes=("", "_decision"))
    relabel_frame = source.merge(labels, on="source_index").merge(relabel_decision, on="source_index", suffixes=("", "_decision"))
    p3_frame = source.merge(labels, on="source_index").merge(p3, on="source_index", suffixes=("", "_decision"))
    categories = {
        "successful_relabel": relabel_frame[(relabel_frame.action == "RELABEL") & (relabel_frame.final_class == relabel_frame.gt_class)].sort_values(["margin", "region_area"], ascending=False).head(20),
        "harmful_relabel": relabel_frame[(relabel_frame.action == "RELABEL") & (relabel_frame.final_class != relabel_frame.gt_class)].sort_values(["margin", "region_area"], ascending=False).head(20),
        "correct_keep": frame[(frame.action == "KEEP") & (frame.final_class == frame.gt_class)].sort_values(["agreement", "region_area"], ascending=False).head(20),
        "ambiguous_reject": p3_frame[p3_frame.action == "REJECT"].sort_values(["margin", "region_area"], ascending=[True, False]).head(20),
    }
    sshr = SSHRCAM(4).cuda(); sshr.load_state_dict(load_state(Path(args.sshr_checkpoint)), strict=True); sshr.eval()
    for parameter in sshr.parameters(): parameter.requires_grad_(False)
    dataset = Stage1_InferDataset(args.trainroot, img_size=224)
    index_for = {Path(path).stem: index for index, path in enumerate(dataset.object)}
    mask_archive = np.load(output / "source_evidence/source_masks.npz"); packed = mask_archive["packed"]
    shape = tuple(mask_archive["shape"]); rows = []
    for category, selected in categories.items():
        for rank, row in enumerate(selected.itertuples(), 1):
            _, tensor = dataset[index_for[row.image_id]]; image_tensor = tensor[None].cuda()
            image_label = torch.from_numpy(parse_label(row.image_id)[None]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hqmr_output = hqmr(image_tensor, image_label, step=29275, hqmr_mode="full")
                cam_output = sshr.forward_cam(image_tensor)[:4]
            prediction = hqmr_output["stages"][2]["hqmr"]["mixture"][0].argmax(0).cpu().numpy()
            cams = [normalize_cam_tensor(cam, shape)[0].cpu().numpy() for cam in cam_output]
            mask = unpack_mask(packed[int(row.source_index)], shape)
            original = np.asarray(Image.open(Path(args.trainroot) / f"{row.image_id}.png").convert("RGB"))
            truth = np.asarray(Image.open(Path(args.train_gt_root) / f"{row.image_id}.png"))
            mask224 = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((224, 224), Image.Resampling.NEAREST)) > 0
            overlay = original.copy(); overlay[mask224] = (.4 * overlay[mask224] + .6 * np.asarray([255, 255, 0])).astype(np.uint8)
            candidate = int(row.candidate_class); votes = row.branch_votes_original
            fig, axes = plt.subplots(2, 4, figsize=(13, 7))
            panels = [(original, "原图", None), (truth, f"GT={int(row.gt_class)}", "tab10"),
                      (prediction, f"HQMR/base={int(row.base_class)}", "tab10"),
                      (overlay, "source component", None)]
            for axis, (value, title, cmap) in zip(axes[0], panels): axis.imshow(value, cmap=cmap, vmin=0 if cmap else None, vmax=4 if cmap else None); axis.set_title(title); axis.axis("off")
            for axis, cam, branch in zip(axes[1], cams, BRANCHES):
                axis.imshow(cam[candidate], cmap="magma", vmin=0, vmax=1); axis.set_title(f"{branch} / c{candidate}"); axis.axis("off")
            fig.suptitle(f"{category} | action={row.action} candidate={candidate} votes={votes} margin={float(row.margin):.3f}")
            fig.tight_layout(); filename = f"{category}_{rank:02d}_s{int(row.source_index):05d}.png"
            fig.savefig(output / "visualizations" / filename, dpi=135); plt.close(fig)
            rows.append({"category": category, "rank": rank, "source_index": int(row.source_index),
                         "image_id": row.image_id, "file": filename, "action": row.action,
                         "base_class": int(row.base_class), "candidate_class": candidate,
                         "gt_class": int(row.gt_class), "margin": float(row.margin)})
    write_csv(output / "visualizations/visualization_manifest.csv", rows)
    summary = {name: int(len(value)) for name, value in categories.items()}
    summary.update({"best_policy": best_policy, "relabel_case_source_policy": relabel_policy})
    write_json(output / "visualizations/visualization_summary.json", summary)
    return summary


@torch.no_grad()
def run_target(args, output: Path) -> None:
    verify_paths(args)
    marker_path = output / "prototype_banks/banks_frozen.json"; marker = json.loads(marker_path.read_text())
    if not marker["gt_accessed"] or marker["validation_accessed"]: raise RuntimeError("Validation firewall invalid")
    banks = {"A": np.load(output / "prototype_banks/bank_a.npz")["prototypes"]}
    for name in POLICIES: banks[name] = np.load(output / f"prototype_banks/bank_{name.lower()}.npz")["prototypes"]
    ablations = dict(np.load(output / "prototype_banks/ablation_banks.npz"))
    model = HQMRNet().cuda(); model.load_state_dict(load_state(Path(args.hqmr_checkpoint)), strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    valroot = Path(args.val_root); loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    targets, compatibility_meta, compatibility_z, hist = [], [], [], []
    identity_rows = {name: [] for name in banks}; started = time.perf_counter()
    for image_index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; original = Image.open(valroot / "img" / f"{image_id}.png")
        truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png"))
        result = infer_hqmr_cirv_inputs(model, image.cuda(non_blocking=True), (original.height, original.width))
        base = result["prediction"]; hist.append(foreground_confusion(truth, base))
        from network.cirv import extract_regions
        for region in extract_regions(base):
            mask = region["mask"]; valid = truth[mask]; valid = valid[valid < 4]
            if not len(valid) or np.any(mask & (truth == region["class_id"])): continue
            counts = np.bincount(valid, minlength=4); majority = int(np.argmax(counts)); purity = counts[majority] / region["area"]
            embedding = pool_masks([mask], result["key4"])[0]; geometry = bbox_geometry(mask)
            row = {"image_id": image_id, "base_class": region["class_id"], "component_id": region["component_id"],
                   "area": region["area"], "gt_majority_class": majority, "purity": purity,
                   "stratum": purity_stratum(purity)}
            for name, bank in banks.items(): row[f"pred_{name}"] = classify(embedding, bank)
            for name, bank in ablations.items(): row[f"pred_{name}"] = classify(embedding, bank)
            targets.append(row)
            append_population(compatibility_meta, compatibility_z, "M1_TARGET", image_id, majority,
                              embedding, mask.mean(), geometry["aspect_ratio"], geometry["context_ratio"])
        gt_masks, gt_classes, _ = all_gt_regions(truth)
        if gt_masks:
            z = pool_masks(gt_masks, result["key4"])
            for embedding, cls, mask in zip(z, gt_classes, gt_masks):
                for name, bank in banks.items(): identity_rows[name].append((cls, int(mask.sum()), classify(embedding, bank)))
        if image_index % 100 == 0:
            print(json.dumps({"event": "target", "images": image_index, "m1": len(targets),
                              "elapsed_s": time.perf_counter() - started}), flush=True)
    frame = pd.DataFrame(targets); hist = np.stack(hist); base_score = scores_from_confusion(hist.sum(0))
    metrics = {name: target_metrics(frame, f"pred_{name}") for name in banks}
    ablation_metrics = {name: target_metrics(frame, f"pred_{name}") for name in ablations}
    for name in banks: write_json(output / f"target_eval/m1_bank_{name.lower()}.json", metrics[name])
    write_json(output / "target_eval/ablation_metrics.json", ablation_metrics)
    source_metrics = json.loads((output / "metrics/source_relabel_metrics.json").read_text())
    recovery = {name: (metrics[name]["area_weighted_accuracy"] - metrics["A"]["area_weighted_accuracy"]) /
                (ORACLE_BANK_B - metrics["A"]["area_weighted_accuracy"]) for name in POLICIES}
    best = max(POLICIES, key=lambda name: recovery[name]); best_policy = pd.read_csv(output / f"policies/{best.lower()}_results.csv")
    counts = best_policy[best_policy.action != "REJECT"].final_class.value_counts().to_dict()
    original_counts = pd.read_csv(output / "source_evidence/source_table.csv").base_class.value_counts().to_dict()
    tissue_identity = {}
    for name, rows in identity_rows.items():
        values = pd.DataFrame(rows, columns=["true", "area", "pred"]); ok = values.true == values.pred
        tissue_identity[name] = float(np.sum(values.area * ok) / values.area.sum())
    per_class_gain = []
    for cls in range(4):
        subset = frame[(frame.purity >= .5) & (frame.gt_majority_class == cls)]
        area = subset.area.sum(); a = float(np.sum(subset.area * (subset.pred_A == cls)) / max(area, 1))
        b = float(np.sum(subset.area * (subset[f"pred_{best}"] == cls)) / max(area, 1)); per_class_gain.append(b-a)
    checks = {"all_classes_have_sources": all(counts.get(cls, 0) >= 4 for cls in range(4)),
              "no_class_reduction_gt_80pct": all(counts.get(cls, 0) >= .20*original_counts.get(cls, 1) for cls in range(4)),
              "not_single_class_contribution": sum(value > 0 for value in per_class_gain) >= 3,
              "clean_tissue_identity_not_down_gt_3pp": tissue_identity[best] >= tissue_identity["A"] - .03}
    precision = source_metrics[best]["changed_region_precision"]
    decision = decision_from_recovery(recovery[best], metrics[best]["area_weighted_accuracy"], precision, checks)
    reproduction = {"hqmr_miou": base_score["mIoU"], "num_m1_components": len(frame),
                    "m1_area": int(frame.area.sum()), "bank_a_area_accuracy": metrics["A"]["area_weighted_accuracy"],
                    "checkpoint_sha256": HQMR_SHA256, "config_sha256": sha256(ROOT / "configs/audits/pscr_v1.yaml")}
    reproduction["pass"] = (reproduction["hqmr_miou"] == EXPECTED_HQMR and len(frame) == EXPECTED_M1_COUNT
                            and int(frame.area.sum()) == EXPECTED_M1_AREA and
                            abs(reproduction["bank_a_area_accuracy"]-EXPECTED_BANK_A) <= 1e-12)
    if not reproduction["pass"]: decision = "ENGINEERING_BLOCKED"
    write_json(output / "00_reproduction_gate.json", reproduction)
    gap = {"bank_A": metrics["A"]["area_weighted_accuracy"], "oracle_bank_B": ORACLE_BANK_B,
           "policies": {name: {"accuracy": metrics[name]["area_weighted_accuracy"], "recovery_ratio": recovery[name]} for name in POLICIES},
           "best_policy": best, "best_recovery_ratio": recovery[best], "decision": decision,
           "changed_region_precision": precision, "architecture_ready_checks": checks,
           "clean_tissue_identity": tissue_identity, "per_class_gain": per_class_gain}
    write_json(output / "metrics/bank_b_gap_recovery.json", gap)
    frame.to_csv(output / "target_eval/m1_targets.csv", index=False)
    compatibility_result = compatibility_finalize(output, compatibility_meta, compatibility_z)
    result = {"decision": decision, "best_policy": best, "reproduction": reproduction,
              "target_metrics": metrics, "ablation_metrics": ablation_metrics,
              "source_metrics": source_metrics, "bank_b_gap": gap,
              "source_error_auroc": json.loads((output / "metrics/source_error_auroc.json").read_text()),
              "compatibility": compatibility_result, "zero_training": True,
              "parameter_updates": 0, "source_commit": commit()}
    result["visualizations"] = generate_visualizations(args, output, model, best)
    write_json(output / "pscr_v1_result.json", result)
    write_report(output, result)
    marker["validation_accessed"] = True; marker["validation_completed"] = True; marker["decision"] = decision
    write_json(marker_path, marker)
    print(json.dumps({"decision": decision, "best_policy": best, "recovery": recovery,
                      "accuracy": {k:v["area_weighted_accuracy"] for k,v in metrics.items()},
                      "changed_precision": precision}, indent=2), flush=True)


def write_report(output: Path, result: dict) -> None:
    r, gap, source = result["reproduction"], result["bank_b_gap"], result["source_metrics"]
    best = result["best_policy"]; target = result["target_metrics"]
    rows = "\n".join([f"| {name} | {100*target[name]['area_weighted_accuracy']:.2f}% | {100*gap['policies'][name]['recovery_ratio']:.2f}% | {100*source[name]['changed_region_precision']:.2f}% | {100*source[name]['harm_rate']:.2f}% | {100*source[name]['recovery_rate']:.2f}% |" for name in POLICIES])
    ablation = "\n".join([f"| {name} | {100*value['area_weighted_accuracy']:.2f}% |" for name, value in result["ablation_metrics"].items()])
    auroc = result["source_error_auroc"]; compat = result["compatibility"]["summary"]
    report = f"""# PSCR-v1 Prototype-Source Class Reliability Audit Report

## 1. Executive Decision

**DECISION = {result['decision']}**  
最佳策略：**{best}**。本轮为零训练审计，parameter update = 0。

## 2. Reproduction Gate

PASS={r['pass']}；HQMR={100*r['hqmr_miou']:.4f}%；M1={r['num_m1_components']} / {r['m1_area']} pixels；Bank A={100*r['bank_a_area_accuracy']:.2f}%。

## 3. Frozen Protocol

冻结 CCRA/HQMR、SSHR 四尺度 CAM、K4 与 spherical k-means seed42。GT 在 policy 与 prototype bank 冻结后才由 evaluation path 读取；P1/P2/P3、margin=0.20、3/4/4/4 votes 与 horizontal-flip consistency 均预注册且未调参。

## 4. Source Error Prevalence

Bank A source component accuracy={100*source['A']['source_accuracy_component']:.2f}%，area accuracy={100*source['A']['source_accuracy_area']:.2f}%。

## 5. Evidence Family Analysis

Ablation 只包含预注册 A0-A4：

| Ablation | M1 area accuracy |
|---|---:|
{ablation}

## 6. Multi-Scale CAM Agreement

四个冻结分支：CAM56、CAM28_1、CAM28_2、CAMdeep；候选类别由训练图 image-level labels hard gate。完整 vote 与 margin 见 `source_evidence/` 与 `policies/`。

## 7. Source Error AUROC

- Agreement(base): component={auroc['agreement_base']['component']:.4f}，area={auroc['agreement_base']['area']:.4f}
- Pcam(base): component={auroc['p_cam_base']['component']:.4f}，area={auroc['p_cam_base']['area']:.4f}
- Base margin: component={auroc['base_margin']['component']:.4f}，area={auroc['base_margin']['area']:.4f}

AUROC<0.70 表示无法稳定识别应干预的错误 source。

## 8. P1/P2/P3 Relabel Metrics

| Policy | M1 area accuracy | Bank-B gap recovery | Changed precision | Harm | Recovery |
|---|---:|---:|---:|---:|---:|
{rows}

## 9. Changed-Region Precision

最佳策略 {best} 的 changed-region precision={100*source[best]['changed_region_precision']:.2f}%。

## 10. Harm / Recovery

最佳策略 harm={100*source[best]['harm_rate']:.2f}%，recovery={100*source[best]['recovery_rate']:.2f}%。

## 11. Per-Class Results

见 `metrics/per_class_metrics.csv` 与 `metrics/confusion_matrices.csv`。

## 12. Verified Prototype Banks

Bank A/P1/P2/P3 均为 K=4 max-cosine bank；hash 与 occupancy 见 `prototype_banks/banks_frozen.json`。

## 13. M1 Area-Weighted Accuracy

Bank A={100*target['A']['area_weighted_accuracy']:.2f}%；P1={100*target['P1']['area_weighted_accuracy']:.2f}%；P2={100*target['P2']['area_weighted_accuracy']:.2f}%；P3={100*target['P3']['area_weighted_accuracy']:.2f}%。

## 14. Bank-B Gap Recovery Ratio

最佳 {best} recovery ratio={100*gap['best_recovery_ratio']:.2f}%；oracle Bank B={100*gap['oracle_bank_B']:.2f}%。

## 15. Source-Target Compatibility

C centroid distance={compat['C_distance']:.4f}；largest-GT D={compat['D_distance']:.4f}；M1 target={compat['M1_distance']:.4f}；shift flag={compat['source_target_compatibility_flag']}。

## 16. Representative Cases

成功 relabel、harmful relabel、correct KEEP 与 ambiguous/REJECT 案例保存在 `visualizations/`（若实际类别少于20则保存全部可用案例）。

## 17. Failure Anatomy

源标签 oracle 根因是否能由冻结 weak-supervision evidence兑现，由 Bank-B gap recovery 与 source-error AUROC 联合解释；compatibility 结果只做只读诊断，不反馈 policy。

## 18. GO / NOGO Decision

**{result['decision']}**。阈值：<25% NOGO，25–40% WEAK，40–60% GO，60–75% STRONG_GO，满足全部安全条件后才可 ARCHITECTURE_READY。

## 19. Exact Next Step

{"若为 GO/STRONG_GO/ARCHITECTURE_READY，下一轮仅允许准备 RPSC memory construction；不得恢复 CIRV product fusion。" if result['decision'] in ('GO','STRONG_GO','ARCHITECTURE_READY') else "冻结 GT-free source verification 路线，不进行 threshold sweep、learned verifier 或 Full25。"}

> **GT-free evidence 将 Bank A 的 {100*target['A']['area_weighted_accuracy']:.2f}% 推进到 {100*target[best]['area_weighted_accuracy']:.2f}%，恢复 Bank-B label headroom 的 {100*gap['best_recovery_ratio']:.2f}%；最终判决为 {result['decision']}。**
"""
    (output / "PSCR_v1_Prototype_Source_Class_Reliability_Audit_Report.md").write_text(report, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("extract", "source-evaluate", "target", "visualize"))
    parser.add_argument("--trainroot", required=True); parser.add_argument("--train-gt-root", required=True)
    parser.add_argument("--train-gt-audit", required=True); parser.add_argument("--val-root", required=True)
    parser.add_argument("--hqmr-checkpoint", required=True); parser.add_argument("--sshr-checkpoint", required=True)
    parser.add_argument("--phase0-output", required=True); parser.add_argument("--anatomy-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); output = Path(args.output_dir).resolve()
    if args.mode == "extract": run_extract(args, output)
    elif args.mode == "source-evaluate": run_source_evaluate(args, output)
    elif args.mode == "target": run_target(args, output)
    else:
        result_path = output / "pscr_v1_result.json"; result = json.loads(result_path.read_text())
        model = HQMRNet().cuda(); model.load_state_dict(load_state(Path(args.hqmr_checkpoint)), strict=True); model.eval()
        result["visualizations"] = generate_visualizations(args, output, model, result["best_policy"])
        write_json(result_path, result); write_report(output, result)


if __name__ == "__main__": main()
