#!/usr/bin/env python3
"""Fresh BCSS Seed42 Full25 training for frozen CCRA allocation plus HQMR."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import train_sshr as official
from network.gcqm_net import GCQMNet
from network.hqmr_net import HQMRNet
from network.hqrf_targets import FULL25_STEPS
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.run_gcqm_full25_bcss_seed42 import _canonical_hash, _finite_model, _git, _gradient_health, _save_recovery
from train_cqrf_phase0 import MonitorDataset, Tee
from train_gcqm_phase0 import load_cohort


INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 25, 1171, 29275
MILESTONES = {5, 10, 15, 20, 25}
PARAMETER_DELTA = 990_208
CONFIG = {
    "experiment": "CCRA HQMR BCSS Seed42 Full25", "dataset": "BCSS training only", "seed": 42,
    "epochs": 25, "batch_size": 20, "effective_batch_size": 20, "image_size": 224,
    "precision": "bf16", "base_lr": .01, "weight_decay": .0005, "poly_power": .9,
    "steps_per_epoch": 1171, "total_steps": 29275, "locality_denominator": 29275,
    "loss": ".50*deep+.25*PCA+.25*mask", "stage_mask_weights": [.20, .30, .50],
    "hqmr_dimension": 256, "feature_hierarchy": ["CHPF(F5)", "CHPF(F4)", "raw F3"],
    "actual_shapes_224": {"H5": [256, 28, 28], "H4": [128, 28, 28], "H3": [256, 56, 56]},
    "normalization": "GroupNorm32 + LayerNorm", "region_pool_accumulation": "FP32", "epsilon": 1e-8,
    "reconstruction": "logit residual, bilinear align_corners=False", "class_weights": "original GCQM w detached",
    "parameter_delta": PARAMETER_DELTA, "new_auxiliary_loss": 0, "propagation": False,
    "checkpoint_selection": "fixed Epoch25 FINAL only", "validation_during_training": False,
}


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _js_rows(weights: torch.Tensor) -> list[float]:
    distribution = weights.float().transpose(1, 2).clamp_min(1e-8)
    rows = []
    for left in range(distribution.shape[1]):
        for right in range(left + 1, distribution.shape[1]):
            p, q = distribution[:, left], distribution[:, right]; m = .5 * (p + q)
            rows.extend((.5 * ((p * (p / m).log()).sum(-1) + (q * (q / m).log()).sum(-1))).cpu().tolist())
    return rows


def _basis_health(stage: dict, targets: torch.Tensor) -> dict:
    decoded = stage["hqmr"]; basis = decoded["basis"].float(); weights = decoded["weights"].float()
    mixture = decoded["mixture"].float(); area = basis.mean((-2, -1)); flat = (basis > .5).flatten(2).float()
    score = weights.mean(-1); selected = torch.argsort(score, dim=1, descending=True, stable=True)[:, :32]
    picked = flat.gather(1, selected[..., None].expand(-1, -1, flat.shape[-1]))
    intersection = torch.bmm(picked, picked.transpose(1, 2)); mass = picked.sum(-1)
    union = mass[:, :, None] + mass[:, None, :] - intersection
    eye = torch.eye(picked.shape[1], device=basis.device, dtype=torch.bool)[None]
    pair_iou = (intersection / union.clamp_min(1))[~eye.expand(intersection.shape)]
    peaks = basis.flatten(2).argmax(-1)
    distinct = [float(torch.unique(row).numel() / row.numel()) for row in peaks]
    target = F.interpolate(targets.float(), mixture.shape[-2:], mode="nearest").to(torch.int8)
    bmax = basis.max(1).values > .5; coverage, top5, top10, recall = [], [], [], []
    for batch in range(basis.shape[0]):
        for cls in range(weights.shape[-1]):
            positive = target[batch, cls] == 1
            if not bool(positive.any()): continue
            order = torch.argsort(weights[batch, :, cls], descending=True, stable=True)
            coverage.append(float(bmax[batch][positive].float().mean()))
            top5.append(float((basis[batch, order[:5]].max(0).values > .5)[positive].float().mean()))
            top10.append(float((basis[batch, order[:10]].max(0).values > .5)[positive].float().mean()))
            recall.append(float((mixture[batch, cls] > .5)[positive].float().mean()))
    rolled = torch.roll(basis, 1, dims=1)
    permuted = torch.einsum("bqc,bqhw->bchw", weights, rolled)
    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)
    identical = bool((basis - basis[:, :1]).abs().amax() <= 1e-6)
    return {
        "Bmax_reliable_positive_coverage": _mean(coverage), "top5_union_reliable_positive_coverage": _mean(top5),
        "top10_union_reliable_positive_coverage": _mean(top10), "class_weighted_F_reliable_positive_recall": _mean(recall),
        "mean_basis_area": float(area.mean()), "basis_area_p10": float(torch.quantile(area, .1)),
        "basis_area_p50": float(torch.quantile(area, .5)), "basis_area_p90": float(torch.quantile(area, .9)),
        "empty_fraction": float((area < .01).float().mean()), "near_full_fraction": float((area > .90).float().mean()),
        "logit_abs_mean": float(decoded["basis_logits"].float().abs().mean()),
        "logit_abs_max": float(decoded["basis_logits"].float().abs().max()),
        "query_pairwise_mask_iou": float(pair_iou.mean()), "distinct_peak_fraction": _mean(distinct),
        "ccra_js": _mean(_js_rows(weights)), "D_perm": float((mixture - permuted).abs().mean()),
        "Neff": float(entropy.exp().mean()), "all_query_masks_identical": identical,
        "mid_direct_abs_mean": float(decoded["direct4"].float().abs().mean()) if decoded["direct4"] is not None else 0.0,
        "fine_direct_abs_mean": float(decoded["direct3"].float().abs().mean()) if decoded["direct3"] is not None else 0.0,
    }


@torch.no_grad()
def mechanism_snapshot(model, loader, epoch: int) -> list[dict]:
    was_training = model.training; model.eval(); batches = {2: [], 3: []}
    for _, images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=epoch * STEPS_PER_EPOCH)
        for stage_index in (2, 3): batches[stage_index].append(_basis_health(result["stages"][stage_index - 1], result["targets"]))
    rows = []
    for stage_index in (2, 3):
        keys = batches[stage_index][0]
        rows.append({"snapshot": f"epoch{epoch}", "stage": stage_index,
                     **{key: _mean([row[key] for row in batches[stage_index]]) if key != "all_query_masks_identical"
                        else all(row[key] for row in batches[stage_index]) for key in keys}})
    if was_training: model.train()
    return rows


def collapse_gate(rows: list[dict]) -> dict:
    near_full = max(row["near_full_fraction"] for row in rows)
    empty = max(row["empty_fraction"] for row in rows)
    identical = any(row["all_query_masks_identical"] for row in rows)
    finite = all(all(np.isfinite(value) for value in row.values() if isinstance(value, (float, int))) for row in rows)
    blocked = near_full > .50 or empty > .80 or identical or not finite
    return {"decision": "HQMR_ENGINEERING_OR_COLLAPSE_BLOCKED" if blocked else "CONTINUE_FULL25_UNCHANGED",
            "near_full_fraction_max": near_full, "empty_fraction_max": empty,
            "all_query_masks_identical": identical, "all_finite": finite}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True); parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--cohort-json", required=True)
    parser.add_argument("--coverage-audit-json", required=True); parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, choices=(0, 2), default=0)
    return parser.parse_args()


def main():
    args = parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("HQMR Full25 requires RTX4090D with native BF16")
    output, weights = Path(args.output_dir).resolve(), Path(args.weights).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing populated output: {output}")
    if not args.smoke_steps and _git("status", "--porcelain"): raise AssertionError("Formal step0 requires clean source")
    audit = json.loads(Path(args.coverage_audit_json).read_text(encoding="utf-8"))
    upstream_decision = audit.get("decision", {}).get("decision") if isinstance(audit.get("decision"), dict) else audit.get("decision")
    if upstream_decision != "QUERY_MASK_COVERAGE_LIMIT": raise AssertionError("Frozen coverage-audit evidence mismatch")
    if sha256(weights) != INIT_SHA256 or weights.suffix != ".params": raise AssertionError("Official initialization mismatch")
    for name in ("provenance", "tests", "logs", "metrics", "mechanism", "checkpoints", "recovery", "visualizations", "evaluation", "ablation", "report", "smoke"):
        (output / name).mkdir(parents=True, exist_ok=True)
    log = (output / "logs/hqmr_full25_train.log").open("w", buffering=1); sys.stdout = Tee(sys.__stdout__, log); sys.stderr = Tee(sys.__stderr__, log)
    accesses = install_train_access_guard(); official.set_seed(42); source = _git("rev-parse", "HEAD")
    config = {**CONFIG, "source_commit": source, "trainroot": str(Path(args.trainroot).resolve()),
              "weights": str(weights), "smoke_steps": args.smoke_steps}
    write_json(output / "provenance/hqmr_config.json", config)
    (output / "provenance/hqmr_config_sha256.txt").write_text(_canonical_hash(config) + "\n")
    (output / "provenance/hqmr_source_commit.txt").write_text(source + "\n")
    (output / "provenance/hqmr_git_diff.patch").write_text(subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True))
    (output / "provenance/hqmr_environment.txt").write_text(f"python\t{sys.version.replace(chr(10), ' ')}\ntorch\t{torch.__version__}\ncuda\t{torch.version.cuda}\ngpu\t{torch.cuda.get_device_name(0)}\nplatform\t{platform.platform()}\n")
    write_json(output / "provenance/hqmr_upstream_coverage_audit.json", {"sha256": sha256(args.coverage_audit_json), "payload": audit})
    write_json(output / "provenance/hqmr_protocol_audit.json", {"protected_sources": protected_sources(ROOT), "fresh_official_init": True,
        "seed": 42, "epochs": 25, "steps": 29275, "effective_batch": 20, "validation_during_training": False,
        "old_trained_checkpoint_loaded": False, "old_decoders_preserved_for_ablation": True})

    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, worker_init_fn=official.seed_worker, generator=generator)
    if len(dataset) != 23422 or len(loader) != STEPS_PER_EPOCH or FULL25_STEPS != TOTAL_STEPS: raise AssertionError("Frozen cardinality mismatch")
    names = sorted(Path(path).name for path, _ in dataset.object)
    write_json(output / "provenance/hqmr_dataset.json", {"samples": len(dataset),
        "filename_manifest_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(), "training_only": True,
        "validation_accessed": False, "test_accessed": False})
    model = HQMRNet(); initialization = model.backbone.load_official_initialization(str(weights))
    initialization.update({"sha256": sha256(weights), "fresh_official_initialization": True, "trained_checkpoint_loaded": False})
    write_json(output / "provenance/hqmr_init_identity.json", initialization)
    base_count = sum(p.numel() for p in GCQMNet().parameters()); count = sum(p.numel() for p in model.parameters())
    if count - base_count != PARAMETER_DELTA: raise AssertionError(f"Unexpected HQMR parameter delta: {count - base_count}")
    write_json(output / "provenance/hqmr_parameter_counts.json", {"gcqm": base_count, "hqmr": count, "delta": count - base_count})
    model = model.cuda(); groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([{"params": group, "lr": .01 * multiplier, "weight_decay": decay}
        for group, multiplier, decay in zip(groups, (1, 2, 10, 20), (.0005, 0, .0005, 0))],
        lr=.01, weight_decay=.0005, max_step=TOTAL_STEPS)
    cohort_payload, cohort = load_cohort(args.cohort_json); write_json(output / "provenance/hqmr_monitor_cohort.json", cohort_payload)
    monitor_loader = DataLoader(MonitorDataset(cohort), batch_size=8, num_workers=4, pin_memory=True)
    losses, epochs, mechanism = [], [], []; started = time.perf_counter(); peak = 0.; target = 2 if args.smoke_steps else TOTAL_STEPS
    gradient_contract = None; torch.cuda.reset_peak_memory_stats(); print("HQMR_FULL25_PROTOCOL " + json.dumps(config, sort_keys=True), flush=True)
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); sums = {}; batches = 0
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=optimizer.global_step)
            loss = result["losses"]["loss"]
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Non-finite HQMR loss")
            if optimizer.global_step == 0:
                watched = {"query": result["stages"][2]["hqmr"]["query0"], "H5": result["query_detail"]["context_feature"],
                           "H4": result["pixel_detail"]["F4_context"], "H3": result["features"]["F3"]}
                for tensor in watched.values(): tensor.retain_grad()
            loss.backward()
            if optimizer.global_step == 0:
                hqmr_grads = {name: float(parameter.grad.float().abs().sum()) if parameter.grad is not None else 0.
                              for name, parameter in model.named_parameters() if name.startswith("hqmr.")}
                gradient_contract = {"hqmr_nonzero_gradient_tensors": sum(value > 0 for value in hqmr_grads.values()),
                    "query_gradient": float(watched["query"].grad.float().abs().sum()),
                    "feature_gradients": {name: float(tensor.grad.float().abs().sum()) for name, tensor in watched.items() if name != "query"},
                    "w_requires_grad": bool(result["stages"][2]["hqmr"]["weights"].requires_grad),
                    "stage2_loss_finite": bool(torch.isfinite(result["losses"]["loss_mask_stage2"])),
                    "stage3_loss_finite": bool(torch.isfinite(result["losses"]["loss_mask_stage3"])),
                    "all_hqmr_gradient_sums": hqmr_grads}
                if gradient_contract["hqmr_nonzero_gradient_tensors"] == 0 or gradient_contract["query_gradient"] <= 0 or any(v <= 0 for v in gradient_contract["feature_gradients"].values()) or gradient_contract["w_requires_grad"]:
                    raise AssertionError(f"HQMR gradient contract failed: {gradient_contract}")
                write_json(output / "tests/hqmr_gradient_contract.json", gradient_contract)
            health = _gradient_health(model) if optimizer.global_step == 0 or (optimizer.global_step + 1) % 100 == 0 else {}
            optimizer.step(); batches += 1
            if not _finite_model(model): raise FloatingPointError("Non-finite HQMR parameter")
            for key, value in result["losses"].items(): sums[key] = sums.get(key, 0.) + float(value.detach())
            if optimizer.global_step % 100 == 0 or optimizer.global_step == target:
                row = {"epoch": epoch, "step": optimizer.global_step, **{key: value / batches for key, value in sums.items()},
                       "lr": optimizer.param_groups[0]["lr"], **health}; losses.append(row)
                write_csv(output / "metrics/train_loss.csv", losses); print("HQMR_FULL25_STEP " + json.dumps(row, sort_keys=True), flush=True)
            if optimizer.global_step >= target: break
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
        epoch_row = {"epoch": epoch, "step": optimizer.global_step, **{key: value / batches for key, value in sums.items()},
                     "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - epoch_started,
                     "peak_cuda_memory_gib": peak}; epochs.append(epoch_row); write_csv(output / "metrics/epoch_summary.csv", epochs)
        print("HQMR_FULL25_EPOCH " + json.dumps(epoch_row, sort_keys=True), flush=True)
        if args.smoke_steps:
            rows = mechanism_snapshot(model, monitor_loader, 0)
            if not all(all(np.isfinite(value) for value in row.values() if isinstance(value, (float, int))) for row in rows): raise AssertionError("HQMR smoke mechanism non-finite")
            write_json(output / "tests/hqmr_smoke_summary.json", {"steps": optimizer.global_step, "finite": True,
                "parameter_delta": PARAMETER_DELTA, "gradient_contract": gradient_contract, "mechanism": rows,
                "validation_accessed": False, "checkpoint_written": False})
            print("HQMR_FULL25_SMOKE_PASS", flush=True); return
        if epoch in MILESTONES:
            rows = mechanism_snapshot(model, monitor_loader, epoch); mechanism.extend(rows)
            write_csv(output / "mechanism/hqmr_basis_health.csv", mechanism)
            checkpoint = output / f"checkpoints/hqmr_epoch{epoch:02d}.pth"; torch.save(model.state_dict(), checkpoint)
            write_json(checkpoint.with_suffix(".json"), {"epoch": epoch, "step": optimizer.global_step, "sha256": sha256(checkpoint), "scientific_endpoint": epoch == 25})
            _save_recovery(output / "recovery/latest.pth", model, optimizer, epoch, generator)
            if epoch == 5:
                gate = collapse_gate(rows); write_json(output / "mechanism/hqmr_epoch5_collapse_gate.json", gate)
                print("HQMR_EPOCH5_GATE " + json.dumps(gate, sort_keys=True), flush=True)
                if gate["decision"] == "HQMR_ENGINEERING_OR_COLLAPSE_BLOCKED":
                    write_json(output / "provenance/hqmr_runtime.json", {"status": gate["decision"], "epoch": 5,
                        "steps": optimizer.global_step, "validation_accessed": False, "gate": gate})
                    print("DECISION = HQMR_ENGINEERING_OR_COLLAPSE_BLOCKED", flush=True); return
    if optimizer.global_step != TOTAL_STEPS or len(epochs) != 25: raise AssertionError("HQMR did not reach E25")
    source_checkpoint = output / "checkpoints/hqmr_epoch25.pth"; final = output / "checkpoints/hqmr_epoch25_final.pth"
    os.replace(source_checkpoint, final); digest = sha256(final); (output / "checkpoints/hqmr_epoch25_final_sha256.txt").write_text(digest + "\n")
    metadata = json.loads((output / "checkpoints/hqmr_epoch25.json").read_text()); metadata.update({"sha256": digest,
        "sealed_before_segmentation_evaluation": True, "selection": "E25 FINAL only"}); write_json(output / "checkpoints/hqmr_epoch25_final.json", metadata)
    (output / "checkpoints/hqmr_epoch25.json").unlink()
    runtime = {"status": "HQMR_FULL25_TRAINING_COMPLETE", "epochs": 25, "steps": optimizer.global_step,
        "train_seconds": time.perf_counter() - started, "peak_cuda_memory_gib": peak, "all_finite": True,
        "validation_accessed": False, "test_accessed": False, "training_paths_accessed": len(accesses),
        "checkpoint": str(final), "checkpoint_sha256": digest}; write_json(output / "provenance/hqmr_runtime.json", runtime)
    print("HQMR_FULL25_TRAINING_COMPLETE " + json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__": main()
