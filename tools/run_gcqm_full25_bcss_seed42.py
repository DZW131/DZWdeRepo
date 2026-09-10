#!/usr/bin/env python3
"""Frozen GCQM BCSS Seed42 Full25 training; segmentation evaluation is external."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_sshr as official
from network.gcqm_net import GCQMNet
from network.hqrf_targets import FULL25_STEPS
from network.momd_net import MOMDNet
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_diagnostics import batch_health, summarize
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from train_cqrf_phase0 import MonitorDataset, Tee
from train_gcqm_phase0 import load_cohort, save_visuals


ARCHITECTURE_COMMIT = "f327b91bba88bdf45391d991a13b214e9ab101ff"
INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 25, 1171, 29275
MILESTONES = {5, 10, 15, 20, 25}
ARCHITECTURE_FILES = (
    "network/gcqm.py", "network/gcqm_net.py", "network/cqrf_net.py",
    "network/cqrf_query.py", "network/hqrf_backbone.py",
    "network/hqrf_targets.py", "network/momd.py",
)
CONFIG = {
    "experiment": "GCQM Final Validation BCSS Seed42 Full25 E25 FINAL",
    "dataset": "BCSS training only", "seed": 42, "epochs": EPOCHS,
    "batch_size": 20, "effective_batch_size": 20, "image_size": 224,
    "precision": "bf16", "query_grid": "14x14", "query_count": 196,
    "query_dim": 256, "base_lr": 0.01, "weight_decay": 0.0005,
    "optimizer": "frozen PolyOptimizer/SGD", "poly_power": 0.9,
    "steps_per_epoch": STEPS_PER_EPOCH, "total_steps": TOTAL_STEPS,
    "locality_denominator": FULL25_STEPS,
    "loss": "0.50*deep+0.25*PCA+0.25*mask",
    "stage_mask_weights": [0.20, 0.30, 0.50],
    "primary": "Stage3 F=sum_i w_ic B_i(x)",
    "checkpoint_selection": "fixed Epoch25 FINAL only",
    "validation_during_training": False, "new_parameters": 0,
    "new_auxiliary_loss": 0,
}


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _architecture_manifest() -> dict:
    if subprocess.run(["git", "diff", "--quiet", ARCHITECTURE_COMMIT, "--", *ARCHITECTURE_FILES], cwd=ROOT).returncode:
        raise AssertionError("Frozen GCQM architecture differs from Phase-0 STRONG_GO")
    return {name: sha256(ROOT / name) for name in ARCHITECTURE_FILES}


def _gradient_health(model: torch.nn.Module) -> dict:
    grads = [p.grad.detach().float() for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grads or any(not bool(torch.isfinite(g).all()) for g in grads):
        raise FloatingPointError("Missing or non-finite gradients")
    return {"gradient_tensors": len(grads), "gradient_norm": float(torch.sqrt(sum((g * g).sum() for g in grads)))}


def _finite_model(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


def _save_recovery(path: Path, model, optimizer, epoch: int, generator: torch.Generator) -> None:
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "optimizer_global_step": optimizer.global_step, "epoch": epoch,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "loader_generator_rng": generator.get_state(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def _restore_recovery(path: Path, model, optimizer, generator: torch.Generator) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"])
    optimizer.global_step = int(payload["optimizer_global_step"])
    random.setstate(payload["python_rng"]); np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"]); torch.cuda.set_rng_state_all(payload["cuda_rng"])
    generator.set_state(payload["loader_generator_rng"])
    return int(payload["epoch"])


@torch.no_grad()
def _mechanism_snapshot(model, loader, epoch: int, output: Path, permutations: list[list[int]]) -> dict:
    was_training = model.training; model.eval(); batches, pmec_rows, first = [], [], None
    for names, images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(images, labels, step=epoch * STEPS_PER_EPOCH, run_pmec=True)
        batches.append(batch_health(result, labels, permutations)); pmec_rows.extend(result["pmec_rows"])
        if first is None: first = (names, images.detach(), labels.detach(), result)
    summary = summarize(f"epoch{epoch}", batches, pmec_rows, model, len(permutations))
    save_visuals(output, f"epoch{epoch}", *first, permutations)
    if was_training: model.train()
    return summary


def _rows(summary: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    snapshot = summary["primary_semantic_health"][0]["snapshot"]
    mechanism, weights, ccra, pca_deep = [], [], [], []
    for stage in (2, 3):
        pick = lambda key: next(row for row in summary[key] if row.get("stage") == stage)
        mechanism.append({"snapshot": snapshot, "stage": stage, **{f"primary_{k}": v for k, v in pick("primary_semantic_health").items() if k not in ("snapshot", "stage")}, **{f"perm_{k}": v for k, v in pick("query_identity_sensitivity").items() if k not in ("snapshot", "stage")}, **{f"pca_gain_{k}": v for k, v in pick("vs_pca_gain").items() if k not in ("snapshot", "stage")}, **{f"B_{k}": v for k, v in pick("B_basis_health").items() if k not in ("snapshot", "stage")}})
        weights.append({"snapshot": snapshot, "stage": stage, **{f"conservation_{k}": v for k, v in pick("weight_conservation").items() if k not in ("snapshot", "stage")}, **{f"conditioning_{k}": v for k, v in pick("weight_class_conditioning").items() if k not in ("snapshot", "stage")}, **{f"utilization_{k}": v for k, v in pick("weight_utilization").items() if k not in ("snapshot", "stage")}})
        integrity = next(row for row in summary["responsibility_integrity"] if row["stage"] == stage)
        complement = next(row for row in summary["responsibility_complementarity"] if row["stage"] == stage)
        utilization = next(row for row in summary["responsibility_utilization"] if row["stage"] == stage)
        ccra.append({"snapshot": snapshot, "stage": stage, **{f"integrity_{k}": v for k, v in integrity.items() if k not in ("snapshot", "stage")}, **{f"complement_{k}": v for k, v in complement.items() if k not in ("snapshot", "stage")}, **{f"utilization_{k}": v for k, v in utilization.items() if k not in ("snapshot", "stage")}})
    for row in summary["pca_health"]:
        pca_deep.append({**row, **{f"deep_{k}": v for k, v in summary["deep_gate_health"].items() if k != "snapshot"}})
    return mechanism, weights, ccra, pca_deep


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True); parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--cohort-json", required=True)
    parser.add_argument("--phase0-gate-json", required=True); parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, choices=(0, 2), default=0)
    parser.add_argument("--resume-recovery", default=None)
    return parser.parse_args()


def main():
    args = parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("GCQM Full25 requires RTX4090D with native BF16")
    output, weights = Path(args.output_dir).resolve(), Path(args.weights).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume_recovery:
        raise FileExistsError(f"Refusing populated output: {output}")
    if args.resume_recovery and Path(args.resume_recovery).resolve().parent.parent != output:
        raise AssertionError("Recovery checkpoint must belong to this output")
    if not args.resume_recovery and _git("status", "--porcelain"):
        raise AssertionError("Formal step0 requires a clean source tree")
    gate = json.loads(Path(args.phase0_gate_json).read_text(encoding="utf-8"))
    if gate.get("decision") != "GCQM_PHASE0_STRONG_GO" or gate.get("steps") != 5855 or gate.get("source_commit") != ARCHITECTURE_COMMIT:
        raise AssertionError("Phase-0 STRONG_GO provenance mismatch")
    if sha256(weights) != INIT_SHA256 or weights.suffix != ".params":
        raise AssertionError("Official MXNet initialization mismatch")
    for name in ("checkpoints", "logs", "metrics", "mechanism", "visualizations", "evaluation", "provenance", "report", "recovery"):
        (output / name).mkdir(parents=True, exist_ok=True)
    log = (output / "provenance/gcqm_full25_train.log").open("a" if args.resume_recovery else "w", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    accesses = install_train_access_guard(); official.set_seed(42)
    architecture = _architecture_manifest(); source_commit = _git("rev-parse", "HEAD")
    config = {**CONFIG, "source_commit": source_commit, "frozen_architecture_commit": ARCHITECTURE_COMMIT, "trainroot": str(Path(args.trainroot).resolve()), "weights": str(weights), "cohort_json": str(Path(args.cohort_json).resolve()), "smoke_steps": args.smoke_steps, "resume_recovery": bool(args.resume_recovery)}
    config_hash = _canonical_hash(config)
    write_json(output / "provenance/gcqm_full25_config.json", config)
    (output / "provenance/gcqm_full25_config_sha256.txt").write_text(config_hash + "\n", encoding="utf-8")
    (output / "provenance/gcqm_full25_source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    (output / "provenance/gcqm_full25_git_diff.patch").write_text(subprocess.check_output(["git", "diff", f"{ARCHITECTURE_COMMIT}..HEAD", "--", ".", ":(exclude)docs/GCQM_Phase0_Global_Class_Conditioned_Query_Mixture_BCSS_Seed42_artifacts"], cwd=ROOT, text=True), encoding="utf-8")
    (output / "provenance/gcqm_full25_environment.txt").write_text(f"python\t{sys.version.replace(chr(10),' ')}\ntorch\t{torch.__version__}\ncuda\t{torch.version.cuda}\ngpu\t{torch.cuda.get_device_name(0)}\nplatform\t{platform.platform()}\n", encoding="utf-8")
    write_json(output / "provenance/gcqm_full25_architecture_hashes.json", architecture)
    write_json(output / "provenance/gcqm_phase0_strong_go.json", gate)
    protected = protected_sources(ROOT)
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=official.seed_worker, generator=generator)
    if len(dataset) != 23422 or len(loader) != STEPS_PER_EPOCH or FULL25_STEPS != TOTAL_STEPS:
        raise AssertionError("Frozen BCSS/Full25 cardinality mismatch")
    dataset_names = sorted(Path(path).name for path, _ in dataset.object)
    dataset_hash = hashlib.sha256("\n".join(dataset_names).encode()).hexdigest()
    write_json(output / "provenance/gcqm_full25_dataset_provenance.json", {"trainroot": str(Path(args.trainroot).resolve()), "samples": len(dataset), "filename_manifest_sha256": dataset_hash, "training_only": True, "validation_accessed": False, "test_accessed": False, "luad_accessed": False})
    model = GCQMNet(); initialization = model.backbone.load_official_initialization(str(weights))
    initialization.update({"sha256": sha256(weights), "fresh_official_initialization": True, "trained_checkpoint_loaded": bool(args.resume_recovery)})
    write_json(output / "provenance/gcqm_full25_init_identity.json", initialization)
    if sum(p.numel() for p in model.parameters()) != sum(p.numel() for p in MOMDNet().parameters()):
        raise AssertionError("GCQM must retain zero parameter delta")
    model = model.cuda(); groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([{"params": g, "lr": .01 * mult, "weight_decay": decay} for g, mult, decay in zip(groups, (1, 2, 10, 20), (.0005, 0, .0005, 0))], lr=.01, weight_decay=.0005, max_step=TOTAL_STEPS)
    start_epoch = 0
    if args.resume_recovery:
        start_epoch = _restore_recovery(Path(args.resume_recovery), model, optimizer, generator)
    cohort_payload, cohort = load_cohort(args.cohort_json)
    write_json(output / "provenance/gcqm_full25_monitor_cohort.json", cohort_payload)
    monitor_loader = DataLoader(MonitorDataset(cohort), batch_size=8, num_workers=4, pin_memory=True)
    permutations = permutation_payload(196)["permutations"]
    write_json(output / "provenance/gcqm_full25_protocol_audit.json", {"architecture_frozen": True, "phase0_strong_go": True, "fresh_official_init": not bool(args.resume_recovery), "seed": 42, "epochs": 25, "total_steps": 29275, "effective_batch": 20, "locality_denominator": FULL25_STEPS, "no_validation_during_training": True, "no_resume_at_step0": not bool(args.resume_recovery), "no_new_parameters": True, "no_auxiliary_loss": True, "protected_official_sources": protected})
    loss_rows, epoch_rows, mechanism_rows, weight_rows, ccra_rows, pca_rows, summaries = [], [], [], [], [], [], []
    started = time.perf_counter(); peak = 0.0; target_steps = 2 if args.smoke_steps else TOTAL_STEPS
    print("GCQM_FULL25_PROTOCOL " + json.dumps(config, sort_keys=True), flush=True)
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); sums = {}; batches = 0
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=optimizer.global_step, run_pmec=False)
            loss = result["losses"]["loss"]
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Non-finite GCQM loss")
            loss.backward(); step_health = _gradient_health(model) if optimizer.global_step == 0 or (optimizer.global_step + 1) % 100 == 0 else {}
            optimizer.step(); batches += 1
            if not _finite_model(model): raise FloatingPointError("Non-finite GCQM parameter")
            for key, value in result["losses"].items(): sums[key] = sums.get(key, 0.0) + float(value.detach())
            if optimizer.global_step % 100 == 0 or optimizer.global_step == target_steps:
                row = {"epoch": epoch, "step": optimizer.global_step, **{key: value / batches for key, value in sums.items()}, "lr": optimizer.param_groups[0]["lr"], **step_health}
                loss_rows.append(row); write_csv(output / "metrics/gcqm_train_loss.csv", loss_rows)
                print("GCQM_FULL25_STEP " + json.dumps(row, sort_keys=True), flush=True)
            if optimizer.global_step >= target_steps: break
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024**3)
        epoch_row = {"epoch": epoch, "step": optimizer.global_step, **{key: value / batches for key, value in sums.items()}, "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - epoch_started, "peak_cuda_memory_gib": peak}
        epoch_rows.append(epoch_row); write_csv(output / "metrics/gcqm_epoch_summary.csv", epoch_rows)
        print("GCQM_FULL25_EPOCH " + json.dumps(epoch_row, sort_keys=True), flush=True)
        if args.smoke_steps:
            write_json(output / "provenance/gcqm_full25_smoke.json", {"steps": optimizer.global_step, "finite": True, "validation_accessed": False, "checkpoint_written": False})
            print("GCQM_FULL25_SMOKE_PASS", flush=True); return
        if epoch in MILESTONES:
            summary = _mechanism_snapshot(model, monitor_loader, epoch, output, permutations); summaries.append(summary)
            m, w, c, p = _rows(summary); mechanism_rows += m; weight_rows += w; ccra_rows += c; pca_rows += p
            write_json(output / "mechanism/gcqm_full25_summary_history.json", summaries)
            write_csv(output / "mechanism/gcqm_full25_mechanism_health.csv", mechanism_rows)
            write_csv(output / "mechanism/gcqm_full25_weight_health.csv", weight_rows)
            write_csv(output / "mechanism/gcqm_full25_ccra_health.csv", ccra_rows)
            write_csv(output / "mechanism/gcqm_full25_pca_deep_health.csv", pca_rows)
            milestone = output / f"checkpoints/gcqm_full25_epoch{epoch:02d}.pth"
            torch.save(model.state_dict(), milestone)
            write_json(milestone.with_suffix(".json"), {"epoch": epoch, "step": optimizer.global_step, "sha256": sha256(milestone), "scientific_endpoint": epoch == 25})
            _save_recovery(output / "recovery/latest.pth", model, optimizer, epoch, generator)
    if optimizer.global_step != TOTAL_STEPS or len(epoch_rows) != EPOCHS:
        raise AssertionError("Formal GCQM run did not reach E25")
    milestone25 = output / "checkpoints/gcqm_full25_epoch25.pth"
    final = output / "checkpoints/gcqm_full25_epoch25_final.pth"
    os.replace(milestone25, final); final_hash = sha256(final)
    (output / "checkpoints/gcqm_full25_epoch25_final_sha256.txt").write_text(final_hash + "\n", encoding="utf-8")
    metadata = json.loads((output / "checkpoints/gcqm_full25_epoch25.json").read_text(encoding="utf-8")); metadata.update({"sha256": final_hash, "sealed_before_segmentation_evaluation": True, "selection": "E25 FINAL only"})
    write_json(output / "checkpoints/gcqm_full25_epoch25_final.json", metadata)
    (output / "checkpoints/gcqm_full25_epoch25.json").unlink()
    runtime = {"status": "GCQM_FULL25_TRAINING_COMPLETE", "epochs": EPOCHS, "steps": optimizer.global_step, "train_seconds": time.perf_counter() - started, "peak_cuda_memory_gib": peak, "all_finite": True, "validation_accessed": False, "test_accessed": False, "luad_accessed": False, "training_paths_accessed": len(accesses), "checkpoint": str(final), "checkpoint_sha256": final_hash}
    write_json(output / "provenance/gcqm_full25_runtime.json", runtime)
    print("GCQM_FULL25_TRAINING_COMPLETE " + json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
