#!/usr/bin/env python3
"""Frozen CCAC BCSS Seed42 Full25 training; validation is strictly post-seal."""
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

import train_sshr as official
from network.ccac_net import CCACNet
from network.gcqm_net import GCQMNet
from network.hqrf_targets import FULL25_STEPS
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_diagnostics import batch_health, summarize
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.run_gcqm_full25_bcss_seed42 import _canonical_hash, _finite_model, _git, _gradient_health, _rows, _save_recovery
from train_cqrf_phase0 import MonitorDataset, Tee
from train_gcqm_phase0 import load_cohort, save_visuals


INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 25, 1171, 29275
MILESTONES = {5, 10, 15, 20, 25}
CONFIG = {
    "experiment": "GCQM CCAC BCSS Seed42 Full25", "dataset": "BCSS training only", "seed": 42,
    "epochs": 25, "batch_size": 20, "effective_batch_size": 20, "image_size": 224,
    "precision": "bf16", "base_lr": .01, "weight_decay": .0005, "poly_power": .9,
    "steps_per_epoch": 1171, "total_steps": 29275, "locality_denominator": 29275,
    "loss": "0.50*deep+0.25*PCA+0.25*mask", "stage_mask_weights": [.20, .30, .50],
    "primary": "Stage3 CCAC^2(GCQM, detached pixel feature)", "ccac_stages": [2, 3],
    "ccac_iterations": 2, "ccac_neighborhood": "3x3 including self", "ccac_affinity": "relu cosine; self=1; row normalized",
    "ccac_update": "fill-only rival-gated", "checkpoint_selection": "fixed Epoch25 FINAL only",
    "validation_during_training": False, "new_parameters": 0, "new_auxiliary_loss": 0,
}


@torch.no_grad()
def mechanism_snapshot(model, loader, epoch: int, output: Path, permutations) -> tuple[dict, list[dict]]:
    training = model.training; model.eval(); batches, pmec_rows, ccac_rows, first = [], [], [], None
    for names, images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=epoch * STEPS_PER_EPOCH, run_pmec=True)
        batches.append(batch_health(result, labels, permutations)); pmec_rows.extend(result["pmec_rows"])
        for stage_index in (2, 3):
            detail = result["stages"][stage_index - 1]["ccac"]; base, restored = detail["base"], detail["restored"]
            row = {"snapshot": f"epoch{epoch}", "stage": stage_index, **detail["diagnostics"],
                   "base_mean": float(base.mean()), "restored_mean": float(restored.mean()),
                   "base_positive_fraction": float((base > .5).float().mean()),
                   "restored_positive_fraction": float((restored > .5).float().mean())}
            ccac_rows.append(row)
        if first is None: first = (names, images.detach(), labels.detach(), result)
    summary = summarize(f"epoch{epoch}", batches, pmec_rows, model, len(permutations))
    aggregated = []
    for stage in (2, 3):
        rows = [r for r in ccac_rows if r["stage"] == stage]
        keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in {"stage", "iterations"}]
        aggregated.append({"snapshot": f"epoch{epoch}", "stage": stage, "iterations": 2,
                           **{k: float(np.mean([r[k] for r in rows])) for k in keys}})
    save_visuals(output, f"epoch{epoch}", *first, permutations)
    if training: model.train()
    return summary, aggregated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True); parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--cohort-json", required=True)
    parser.add_argument("--failure-anatomy-json", required=True); parser.add_argument("--old-gcqm-checkpoint", required=True)
    parser.add_argument("--num-workers", type=int, default=8); parser.add_argument("--smoke-steps", type=int, choices=(0, 2), default=0)
    return parser.parse_args()


def main():
    args = parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("CCAC Full25 requires RTX4090D native BF16")
    output, weights = Path(args.output_dir).resolve(), Path(args.weights).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing populated output: {output}")
    if not args.smoke_steps and _git("status", "--porcelain"): raise AssertionError("Formal step0 requires clean source")
    anatomy = json.loads(Path(args.failure_anatomy_json).read_text(encoding="utf-8"))
    decision = anatomy.get("decision") or anatomy.get("failure_anatomy_decision")
    if decision != "MISSING_SPATIAL_COHERENCE": raise AssertionError(f"Failure-anatomy gate mismatch: {decision}")
    if sha256(weights) != INIT_SHA256 or weights.suffix != ".params": raise AssertionError("Official initialization mismatch")
    for name in ("checkpoints", "logs", "metrics", "mechanism", "visualizations", "evaluation", "provenance", "report", "recovery", "tests", "smoke"):
        (output / name).mkdir(parents=True, exist_ok=True)
    log = (output / "logs/ccac_full25_train.log").open("w", buffering=1); sys.stdout = Tee(sys.__stdout__, log); sys.stderr = Tee(sys.__stderr__, log)
    accesses = install_train_access_guard(); official.set_seed(42); source_commit = _git("rev-parse", "HEAD")
    config = {**CONFIG, "source_commit": source_commit, "trainroot": str(Path(args.trainroot).resolve()),
              "weights": str(weights), "cohort_json": str(Path(args.cohort_json).resolve()), "smoke_steps": args.smoke_steps}
    write_json(output / "provenance/ccac_full25_config.json", config)
    (output / "provenance/ccac_full25_config_sha256.txt").write_text(_canonical_hash(config) + "\n", encoding="utf-8")
    (output / "provenance/ccac_full25_source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    write_json(output / "provenance/ccac_config.json", config)
    (output / "provenance/ccac_config_sha256.txt").write_text(_canonical_hash(config) + "\n", encoding="utf-8")
    (output / "provenance/ccac_source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    (output / "provenance/ccac_git_diff.patch").write_text(subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True), encoding="utf-8")
    (output / "provenance/ccac_full25_environment.txt").write_text(
        f"python\t{sys.version.replace(chr(10),' ')}\ntorch\t{torch.__version__}\ncuda\t{torch.version.cuda}\ngpu\t{torch.cuda.get_device_name(0)}\nplatform\t{platform.platform()}\n", encoding="utf-8")
    (output / "provenance/ccac_environment.txt").write_text((output / "provenance/ccac_full25_environment.txt").read_text(), encoding="utf-8")
    write_json(output / "provenance/gcqm_failure_anatomy_gate.json", anatomy)
    write_json(output / "provenance/ccac_upstream_identity.json", {
        "failure_anatomy_sha256": sha256(args.failure_anatomy_json), "old_gcqm_checkpoint": str(Path(args.old_gcqm_checkpoint).resolve()),
        "old_gcqm_checkpoint_sha256": sha256(args.old_gcqm_checkpoint), "official_init_sha256": sha256(weights)})
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        worker_init_fn=official.seed_worker, generator=generator)
    if len(dataset) != 23422 or len(loader) != STEPS_PER_EPOCH or FULL25_STEPS != TOTAL_STEPS: raise AssertionError("Frozen cardinality mismatch")
    names = sorted(Path(path).name for path, _ in dataset.object)
    write_json(output / "provenance/ccac_full25_dataset_provenance.json", {"trainroot": str(Path(args.trainroot).resolve()),
        "samples": len(dataset), "filename_manifest_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "training_only": True, "validation_accessed": False, "test_accessed": False})
    model = CCACNet(); initialization = model.backbone.load_official_initialization(str(weights))
    initialization.update({"sha256": sha256(weights), "fresh_official_initialization": True, "trained_checkpoint_loaded": False})
    write_json(output / "provenance/ccac_full25_init_identity.json", initialization)
    write_json(output / "provenance/ccac_init_identity.json", initialization)
    gcqm_count, ccac_count = sum(p.numel() for p in GCQMNet().parameters()), sum(p.numel() for p in model.parameters())
    if gcqm_count != ccac_count or list(GCQMNet().state_dict()) != list(model.state_dict()): raise AssertionError("CCAC parameter/state delta")
    write_json(output / "provenance/ccac_parameter_counts.json", {"gcqm": gcqm_count, "ccac": ccac_count, "delta": 0})
    model = model.cuda(); groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([{"params": g, "lr": .01 * m, "weight_decay": d} for g, m, d in zip(groups, (1, 2, 10, 20), (.0005, 0, .0005, 0))], lr=.01, weight_decay=.0005, max_step=TOTAL_STEPS)
    cohort_payload, cohort = load_cohort(args.cohort_json); write_json(output / "provenance/ccac_monitor_cohort.json", cohort_payload)
    monitor_loader = DataLoader(MonitorDataset(cohort), batch_size=8, num_workers=4, pin_memory=True)
    permutations = permutation_payload(196)["permutations"]
    write_json(output / "provenance/ccac_protocol_audit.json", {"failure_anatomy_gate": True, "fresh_official_init": True,
        "seed": 42, "epochs": 25, "steps": 29275, "effective_batch": 20, "no_validation_during_training": True,
        "zero_parameter_delta": True, "zero_auxiliary_loss": True, "ccac_iterations": 2, "protected_sources": protected_sources(ROOT)})
    loss_rows, epoch_rows, summaries, ccac_rows, mechanism_rows, weight_rows, ccra_rows, pca_rows = [], [], [], [], [], [], [], []
    started = time.perf_counter(); peak = 0.; target_steps = 2 if args.smoke_steps else TOTAL_STEPS
    torch.cuda.reset_peak_memory_stats(); print("CCAC_FULL25_PROTOCOL " + json.dumps(config, sort_keys=True), flush=True)
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); sums = {}; batches = 0
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=optimizer.global_step)
            loss = result["losses"]["loss"]
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Non-finite CCAC loss")
            loss.backward(); health = _gradient_health(model) if optimizer.global_step == 0 or (optimizer.global_step + 1) % 100 == 0 else {}
            optimizer.step(); batches += 1
            if not _finite_model(model): raise FloatingPointError("Non-finite CCAC parameter")
            for key, value in result["losses"].items(): sums[key] = sums.get(key, 0.) + float(value.detach())
            if optimizer.global_step % 100 == 0 or optimizer.global_step == target_steps:
                row = {"epoch": epoch, "step": optimizer.global_step, **{k: v / batches for k, v in sums.items()},
                       "lr": optimizer.param_groups[0]["lr"], **health}; loss_rows.append(row)
                write_csv(output / "metrics/ccac_train_loss.csv", loss_rows); print("CCAC_FULL25_STEP " + json.dumps(row, sort_keys=True), flush=True)
            if optimizer.global_step >= target_steps: break
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
        epoch_row = {"epoch": epoch, "step": optimizer.global_step, **{k: v / batches for k, v in sums.items()},
                     "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - epoch_started, "peak_cuda_memory_gib": peak}
        epoch_rows.append(epoch_row); write_csv(output / "metrics/ccac_epoch_summary.csv", epoch_rows)
        print("CCAC_FULL25_EPOCH " + json.dumps(epoch_row, sort_keys=True), flush=True)
        if args.smoke_steps:
            base_summary, functional = mechanism_snapshot(model, monitor_loader, 0, output, permutations)
            if any((not np.isfinite(row["completion_mass_mean"]) or row["changed_fraction"] <= 0 or
                    row["restored_positive_fraction"] >= .99 or row["rival_protected_mass_mean"] <= 0)
                   for row in functional):
                raise AssertionError(f"CCAC functional smoke catastrophic: {functional}")
            write_json(output / "smoke/ccac_functional_smoke.json", {"ccac": functional, "gcqm_base": base_summary})
            write_json(output / "smoke/ccac_smoke_summary.json", {"steps": optimizer.global_step, "finite": True,
                "gradient_finite": True, "parameter_delta": 0, "validation_accessed": False, "checkpoint_written": False,
                "functional_smoke": functional})
            print("CCAC_FULL25_SMOKE_PASS", flush=True); return
        if epoch in MILESTONES:
            summary, ccac = mechanism_snapshot(model, monitor_loader, epoch, output, permutations); summaries.append(summary); ccac_rows.extend(ccac)
            m, w, c, p = _rows(summary); mechanism_rows += m; weight_rows += w; ccra_rows += c; pca_rows += p
            write_json(output / "mechanism/ccac_full25_summary_history.json", summaries); write_csv(output / "mechanism/ccac_health.csv", ccac_rows)
            write_csv(output / "mechanism/gcqm_base_health.csv", mechanism_rows); write_csv(output / "mechanism/gcqm_weight_health.csv", weight_rows)
            write_csv(output / "mechanism/ccra_health.csv", ccra_rows); write_csv(output / "mechanism/pca_deep_health.csv", pca_rows)
            write_csv(output / "mechanism/ccac_train_only_health.csv", ccac_rows); write_csv(output / "mechanism/ccac_completion_health.csv", ccac_rows)
            write_csv(output / "mechanism/ccra_survival.csv", ccra_rows)
            checkpoint = output / f"checkpoints/ccac_full25_epoch{epoch:02d}.pth"; torch.save(model.state_dict(), checkpoint)
            write_json(checkpoint.with_suffix(".json"), {"epoch": epoch, "step": optimizer.global_step, "sha256": sha256(checkpoint), "scientific_endpoint": epoch == 25})
            _save_recovery(output / "recovery/latest.pth", model, optimizer, epoch, generator)
    if optimizer.global_step != TOTAL_STEPS or len(epoch_rows) != EPOCHS: raise AssertionError("CCAC run did not reach E25")
    source = output / "checkpoints/ccac_full25_epoch25.pth"; final = output / "checkpoints/ccac_full25_epoch25_final.pth"; os.replace(source, final)
    digest = sha256(final); (output / "checkpoints/ccac_full25_epoch25_final_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    metadata = json.loads((output / "checkpoints/ccac_full25_epoch25.json").read_text()); metadata.update({"sha256": digest,
        "sealed_before_segmentation_evaluation": True, "selection": "E25 FINAL only"})
    write_json(output / "checkpoints/ccac_full25_epoch25_final.json", metadata); (output / "checkpoints/ccac_full25_epoch25.json").unlink()
    runtime = {"status": "CCAC_FULL25_TRAINING_COMPLETE", "epochs": 25, "steps": optimizer.global_step,
        "train_seconds": time.perf_counter() - started, "peak_cuda_memory_gib": peak, "all_finite": True,
        "validation_accessed": False, "test_accessed": False, "training_paths_accessed": len(accesses),
        "checkpoint": str(final), "checkpoint_sha256": digest}
    write_json(output / "provenance/ccac_full25_runtime.json", runtime); print("CCAC_FULL25_TRAINING_COMPLETE " + json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__": main()
