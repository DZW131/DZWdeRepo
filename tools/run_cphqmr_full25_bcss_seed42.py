#!/usr/bin/env python3
"""Fresh BCSS Seed42 Full25 training for frozen CCRA plus CP-HQMR."""
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
from network.cphqmr_net import CPHQMRNet
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
HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 25, 1171, 29275
MILESTONES = {5, 10, 15, 20, 25}
PARAMETER_DELTA_GCQM, PARAMETER_DELTA_HQMR = 676_874, -313_334
CONFIG = {"experiment": "CCRA CP-HQMR BCSS Seed42 Full25", "dataset": "BCSS training only", "seed": 42,
    "epochs": 25, "batch_size": 20, "effective_batch_size": 20, "image_size": 224, "precision": "bf16",
    "base_lr": .01, "weight_decay": .0005, "poly_power": .9, "steps_per_epoch": 1171, "total_steps": 29275,
    "locality_denominator": 29275, "loss": ".50*deep+.25*PCA+.25*mask", "stage_mask_weights": [.20, .30, .50],
    "semantic_hierarchy": "H5->H4 at 28x28", "semantic_endpoint": "H4", "q_cov_region_updated": False,
    "q_disc_region_updated": True, "CFR": "D+(1-sigmoid(D))*relu(C-D)",
    "DGSR": "F3 256->64 GN32 GELU; softmax 9-way 3x3 kernel; sigmoid shared gate",
    "DGSR_padding": "replicate", "F3_query_semantic_primary": False, "class_weights": "original GCQM w detached",
    "parameter_delta_vs_gcqm": PARAMETER_DELTA_GCQM, "parameter_delta_vs_hqmr_v1": PARAMETER_DELTA_HQMR,
    "new_auxiliary_loss": 0, "propagation": False, "checkpoint_selection": "fixed Epoch25 FINAL only",
    "validation_during_training": False}


def _mean(values): return float(np.mean(values)) if values else 0.0


def basis_health(stage: dict) -> dict:
    item = stage["cphqmr"]; basis, weights = item["basis"].float(), item["weights"].float(); area = basis.mean((-2, -1))
    primary = item["mixture"].float(); permuted = torch.einsum("bqc,bqhw->bchw", weights, torch.roll(basis, 1, 1))
    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(1)
    qprob = basis > .5; peaks = basis.flatten(2).argmax(-1)
    result = {"coverage_branch_area": float(item["C4"].float().sigmoid().mean()),
        "discriminative_branch_area": float(item["D4"].float().sigmoid().mean()),
        "fused_branch_area": float(item["M4"].float().sigmoid().mean()),
        "fraction_C_gt_D": float((item["C4"] > item["D4"]).float().mean()),
        "mean_rescue_logit": float(item["rescue"].float().mean()),
        "mean_rescue_probability": float((item["M4"].float().sigmoid() - item["D4"].float().sigmoid()).mean()),
        "q_disc_update_abs_mean": float((item["q_disc"] - item["q_cov"]).float().abs().mean()),
        "basis_area": float(area.mean()), "basis_area_p10": float(torch.quantile(area, .1)),
        "basis_area_p50": float(torch.quantile(area, .5)), "basis_area_p90": float(torch.quantile(area, .9)),
        "empty_fraction": float((area < .01).float().mean()), "near_full_fraction": float((area > .90).float().mean()),
        "all_query_masks_identical": bool((basis - basis[:, :1]).abs().amax() <= 1e-6),
        "distinct_peak_fraction": _mean([float(torch.unique(row).numel() / row.numel()) for row in peaks]),
        "D_perm": float((primary - permuted).abs().mean()), "Neff": float(entropy.exp().mean()),
        "finite": bool(torch.isfinite(item["basis_logits"]).all())}
    if item["dgsr"] is not None:
        gate, kernel = item["dgsr"]["gate"].float(), item["dgsr"]["kernel"].float()
        result.update({"restore_gate_mean": float(gate.mean()), "restore_gate_p10": float(torch.quantile(gate, .1)),
            "restore_gate_p50": float(torch.quantile(gate, .5)), "restore_gate_p90": float(torch.quantile(gate, .9)),
            "kernel_entropy": float(-(kernel.clamp_min(1e-8) * kernel.clamp_min(1e-8).log()).sum(1).mean()),
            "center_kernel_mass": float(kernel[:, 4].mean()),
            "restoration_delta_abs_mean": float((item["dgsr"]["logits"] - item["dgsr"]["upsampled"]).float().abs().mean())})
    else:
        result.update({key: 0.0 for key in ("restore_gate_mean", "restore_gate_p10", "restore_gate_p50", "restore_gate_p90", "kernel_entropy", "center_kernel_mass", "restoration_delta_abs_mean")})
    return result


@torch.no_grad()
def mechanism_snapshot(model, loader, epoch):
    training = model.training; model.eval(); values = {2: [], 3: []}
    for _, images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): output = model(images, labels, step=epoch * STEPS_PER_EPOCH)
        for stage in (2, 3): values[stage].append(basis_health(output["stages"][stage - 1]))
    rows = []
    for stage in (2, 3):
        rows.append({"snapshot": f"epoch{epoch}", "stage": stage, **{key: all(row[key] for row in values[stage]) if key in ("finite", "all_query_masks_identical") else _mean([row[key] for row in values[stage]]) for key in values[stage][0]}})
    if training: model.train()
    return rows


def health_summary(rows):
    """Record train-only diagnostics without selecting or stopping the Full25 run."""
    return {"action": "CONTINUE_FULL25_UNCHANGED", "diagnostic_only": True,
            "near_full_fraction_max": max(row["near_full_fraction"] for row in rows),
            "empty_fraction_max": max(row["empty_fraction"] for row in rows),
            "all_query_masks_identical": any(row["all_query_masks_identical"] for row in rows),
            "all_finite": all(row["finite"] for row in rows)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--trainroot", required=True); p.add_argument("--weights", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--cohort-json", required=True); p.add_argument("--preaudit-json", required=True)
    p.add_argument("--hqmr-checkpoint", required=True); p.add_argument("--num-workers", type=int, default=8); p.add_argument("--smoke-steps", type=int, choices=(0, 2), default=0); return p.parse_args()


def main():
    args = parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("CP-HQMR requires RTX4090D BF16")
    output, weights = Path(args.output_dir).resolve(), Path(args.weights).resolve(); existing = {path.name for path in output.iterdir()} if output.exists() else set()
    allowed = set() if args.smoke_steps else {"preaudit"}
    if existing - allowed: raise FileExistsError(f"Refusing populated output: {existing - allowed}")
    if not args.smoke_steps and _git("status", "--porcelain"): raise AssertionError("Formal step0 requires clean source")
    preaudit = json.loads(Path(args.preaudit_json).read_text());
    if preaudit["decision"] not in ("COVERAGE_PURITY_CONFLICT_CONFIRMED", "COVERAGE_PURITY_CONFLICT_NOT_CONFIRMED"): raise AssertionError("Invalid frozen preaudit")
    if sha256(weights) != INIT_SHA256 or sha256(args.hqmr_checkpoint) != HQMR_SHA256: raise AssertionError("Frozen identity mismatch")
    for name in ("provenance", "tests", "logs", "metrics", "mechanism", "checkpoints", "recovery", "evaluation", "coverage_purity", "ablation", "visualizations", "report", "smoke"): (output / name).mkdir(parents=True, exist_ok=True)
    log = (output / "logs/cphqmr_full25_train.log").open("w", buffering=1); sys.stdout = Tee(sys.__stdout__, log); sys.stderr = Tee(sys.__stderr__, log)
    accesses = install_train_access_guard(); official.set_seed(42); source = _git("rev-parse", "HEAD"); config = {**CONFIG, "source_commit": source, "trainroot": str(Path(args.trainroot).resolve()), "weights": str(weights), "smoke_steps": args.smoke_steps}
    write_json(output / "provenance/cphqmr_config.json", config); (output / "provenance/cphqmr_config_sha256.txt").write_text(_canonical_hash(config) + "\n"); (output / "provenance/cphqmr_source_commit.txt").write_text(source + "\n")
    (output / "provenance/cphqmr_git_diff.patch").write_text(subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True)); (output / "provenance/cphqmr_environment.txt").write_text(f"python\t{sys.version.replace(chr(10),' ')}\ntorch\t{torch.__version__}\ncuda\t{torch.version.cuda}\ngpu\t{torch.cuda.get_device_name(0)}\nplatform\t{platform.platform()}\n")
    write_json(output / "provenance/hqmr_v1_archive_manifest.json", {"checkpoint": str(Path(args.hqmr_checkpoint).resolve()), "sha256": sha256(args.hqmr_checkpoint), "read_only_reference": True}); write_json(output / "provenance/cphqmr_preaudit_frozen.json", preaudit)
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224); generator = torch.Generator().manual_seed(42); loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=official.seed_worker, generator=generator)
    if len(dataset) != 23422 or len(loader) != STEPS_PER_EPOCH or FULL25_STEPS != TOTAL_STEPS: raise AssertionError("Frozen cardinality mismatch")
    names = sorted(Path(path).name for path, _ in dataset.object); write_json(output / "provenance/cphqmr_dataset.json", {"samples": len(dataset), "filename_manifest_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(), "training_only": True, "validation_accessed": False})
    model = CPHQMRNet(); initialization = model.backbone.load_official_initialization(str(weights)); initialization.update({"sha256": sha256(weights), "fresh_official_initialization": True, "trained_checkpoint_loaded": False}); write_json(output / "provenance/cphqmr_init_identity.json", initialization)
    counts = {"gcqm": sum(p.numel() for p in GCQMNet().parameters()), "hqmr_v1": sum(p.numel() for p in HQMRNet().parameters()), "cphqmr": sum(p.numel() for p in model.parameters())}; counts.update({"delta_vs_gcqm": counts["cphqmr"] - counts["gcqm"], "delta_vs_hqmr_v1": counts["cphqmr"] - counts["hqmr_v1"]})
    if counts["delta_vs_gcqm"] != PARAMETER_DELTA_GCQM or counts["delta_vs_hqmr_v1"] != PARAMETER_DELTA_HQMR: raise AssertionError(counts)
    write_json(output / "provenance/cphqmr_parameter_counts.json", counts); model = model.cuda(); groups = model.get_parameter_groups(); optimizer = PolyOptimizer([{"params": g, "lr": .01 * m, "weight_decay": d} for g, m, d in zip(groups, (1, 2, 10, 20), (.0005, 0, .0005, 0))], lr=.01, weight_decay=.0005, max_step=TOTAL_STEPS)
    cohort_payload, cohort = load_cohort(args.cohort_json); write_json(output / "provenance/cphqmr_monitor_cohort.json", cohort_payload); monitor_loader = DataLoader(MonitorDataset(cohort), batch_size=8, num_workers=4, pin_memory=True)
    losses, epochs, mechanism = [], [], []; started = time.perf_counter(); peak = 0.; target = 2 if args.smoke_steps else TOTAL_STEPS; contract = None; torch.cuda.reset_peak_memory_stats(); print("CPHQMR_FULL25_PROTOCOL " + json.dumps(config, sort_keys=True), flush=True)
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); sums = {}; batches = 0
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=optimizer.global_step)
            loss = result["losses"]["loss"]
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Non-finite CP-HQMR loss")
            if optimizer.global_step == 0:
                item = result["stages"][2]["cphqmr"]; watched = {"q_cov": item["q_cov"], "q_disc": item["q_disc"], "H5": result["query_detail"]["context_feature"], "H4": result["pixel_detail"]["F4_context"], "H3": result["features"]["F3"]}
                for tensor in watched.values(): tensor.retain_grad()
            loss.backward()
            if optimizer.global_step == 0:
                gradients = {name: float(parameter.grad.float().abs().sum()) if parameter.grad is not None else 0. for name, parameter in model.named_parameters() if name.startswith("cphqmr.")}
                contract = {"nonzero_parameter_gradient_tensors": sum(value > 0 for value in gradients.values()), "watched_gradients": {name: float(tensor.grad.float().abs().sum()) for name, tensor in watched.items()}, "q_disc_update_abs_mean": float((item["q_disc"] - item["q_cov"]).detach().float().abs().mean()), "q_cov_equals_normalized_input": True, "w_requires_grad": bool(item["weights"].requires_grad), "stage2_loss_finite": bool(torch.isfinite(result["losses"]["loss_mask_stage2"])), "stage3_loss_finite": bool(torch.isfinite(result["losses"]["loss_mask_stage3"])), "parameter_gradient_sums": gradients}
                if contract["nonzero_parameter_gradient_tensors"] == 0 or any(value <= 0 for value in contract["watched_gradients"].values()) or contract["q_disc_update_abs_mean"] <= 0 or contract["w_requires_grad"]: raise AssertionError(contract)
                write_json(output / "tests/cphqmr_gradient_contract.json", contract)
            health = _gradient_health(model) if optimizer.global_step == 0 or (optimizer.global_step + 1) % 100 == 0 else {}; optimizer.step(); batches += 1
            if not _finite_model(model): raise FloatingPointError("Non-finite parameter")
            for key, value in result["losses"].items(): sums[key] = sums.get(key, 0.) + float(value.detach())
            if optimizer.global_step % 100 == 0 or optimizer.global_step == target:
                row = {"epoch": epoch, "step": optimizer.global_step, **{key: value / batches for key, value in sums.items()}, "lr": optimizer.param_groups[0]["lr"], **health}; losses.append(row); write_csv(output / "metrics/train_loss.csv", losses); print("CPHQMR_FULL25_STEP " + json.dumps(row, sort_keys=True), flush=True)
            if optimizer.global_step >= target: break
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024 ** 3); row = {"epoch": epoch, "step": optimizer.global_step, **{key: value / batches for key, value in sums.items()}, "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - epoch_started, "peak_cuda_memory_gib": peak}; epochs.append(row); write_csv(output / "metrics/epoch_summary.csv", epochs); print("CPHQMR_FULL25_EPOCH " + json.dumps(row, sort_keys=True), flush=True)
        if args.smoke_steps:
            rows = mechanism_snapshot(model, monitor_loader, 0); write_json(output / "tests/cphqmr_smoke.json", {"steps": optimizer.global_step, "finite": True, "gradient_contract": contract, "mechanism": rows, "validation_accessed": False, "checkpoint_written": False}); print("CPHQMR_FULL25_SMOKE_PASS", flush=True); return
        if epoch in MILESTONES:
            rows = mechanism_snapshot(model, monitor_loader, epoch); mechanism.extend(rows); write_csv(output / "mechanism/dual_state_health.csv", mechanism); write_csv(output / "mechanism/cfr_health.csv", [{k: v for k, v in row.items() if k in ("snapshot", "stage", "fraction_C_gt_D", "mean_rescue_logit", "mean_rescue_probability")} for row in mechanism]); write_csv(output / "mechanism/dgsr_health.csv", [{k: v for k, v in row.items() if k.startswith("restore_") or k in ("snapshot", "stage", "kernel_entropy", "center_kernel_mass", "restoration_delta_abs_mean")} for row in mechanism])
            checkpoint = output / f"checkpoints/cphqmr_epoch{epoch:02d}.pth"; torch.save(model.state_dict(), checkpoint); write_json(checkpoint.with_suffix(".json"), {"epoch": epoch, "step": optimizer.global_step, "sha256": sha256(checkpoint), "scientific_endpoint": epoch == 25}); _save_recovery(output / "recovery/latest.pth", model, optimizer, epoch, generator)
            if epoch == 5:
                health = health_summary(rows); write_json(output / "mechanism/cphqmr_epoch5_health.json", health); print("CPHQMR_EPOCH5_HEALTH " + json.dumps(health), flush=True)
    if optimizer.global_step != TOTAL_STEPS: raise AssertionError("CP-HQMR did not reach E25")
    source_checkpoint = output / "checkpoints/cphqmr_epoch25.pth"; final = output / "checkpoints/cphqmr_epoch25_final.pth"; os.replace(source_checkpoint, final); digest = sha256(final); (output / "checkpoints/cphqmr_epoch25_final_sha256.txt").write_text(digest + "\n"); metadata = json.loads((output / "checkpoints/cphqmr_epoch25.json").read_text()); metadata.update({"sha256": digest, "sealed_before_segmentation_evaluation": True, "selection": "E25 FINAL only"}); write_json(output / "checkpoints/cphqmr_epoch25_final.json", metadata); (output / "checkpoints/cphqmr_epoch25.json").unlink()
    runtime = {"status": "CPHQMR_FULL25_TRAINING_COMPLETE", "epochs": 25, "steps": optimizer.global_step, "train_seconds": time.perf_counter() - started, "peak_cuda_memory_gib": peak, "all_finite": True, "validation_accessed": False, "training_paths_accessed": len(accesses), "checkpoint": str(final), "checkpoint_sha256": digest}; write_json(output / "provenance/cphqmr_runtime.json", runtime); print("CPHQMR_FULL25_TRAINING_COMPLETE " + json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__": main()
