#!/usr/bin/env python3
"""Pre-audit, smoke, and fresh Full25 training for CCRA + HQMR-v1 + CCBP."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_sshr as official
from network.ccbp_net import CCBPNet
from network.hqmr_net import HQMRNet
from network.hqrf_targets import FULL25_STEPS
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.run_gcqm_full25_bcss_seed42 import _canonical_hash, _finite_model, _git, _gradient_health, _save_recovery
from tools.run_hqmr_full25_bcss_seed42 import _basis_health, _mean
from tools.eval_gcqm_full25_bcss_seed42 import load_state
from train_cqrf_phase0 import MonitorDataset, Tee
from train_gcqm_phase0 import load_cohort


INIT_SHA256 = "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16"
HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 25, 1171, 29275
MILESTONES = {5, 10, 15, 20, 25}
CCBP_PARAMETERS = 131_073
CONFIG = {
    "experiment": "CCRA HQMR CCBP BCSS Seed42 Full25", "dataset": "BCSS training only", "seed": 42,
    "epochs": 25, "batch_size": 20, "effective_batch_size": 20, "image_size": 224,
    "precision": "bf16", "base_lr": .01, "weight_decay": .0005, "poly_power": .9,
    "steps_per_epoch": 1171, "total_steps": 29275, "locality_denominator": 29275,
    "base_loss": ".50*deep+.25*PCA+.25*mask", "ccbp_loss_weight": 1.0,
    "stage_mask_weights": [.20, .30, .50], "ccbp_stage": 3,
    "ccbp_dimension": 256, "gamma_initial": 5.0, "gamma_range": [1.0, 20.0],
    "rival": "max other active foreground class", "gate": "exp(-relu(rival-target))",
    "semantic_tensors": {"query": "HQMR query4", "feature": "HQMR key4"},
    "gradient_isolation": True, "all_classes_symmetric": True,
    "parameter_delta_vs_hqmr": CCBP_PARAMETERS, "threshold_tuning": False,
    "checkpoint_selection": "fixed Epoch25 FINAL only", "validation_during_training": False,
}


def class_accuracy(logits: torch.Tensor, positive: torch.Tensor, present: torch.Tensor) -> dict:
    value = F.interpolate(logits.float(), positive.shape[-2:], mode="bilinear", align_corners=False)
    prediction = value.argmax(1); rows = []
    for cls in range(value.shape[1]):
        mask = positive[:, cls].bool() & present[:, cls, None, None].bool()
        if mask.any(): rows.append({"class": cls, "correct": int((prediction[mask] == cls).sum()), "pixels": int(mask.sum())})
    classwise = {str(cls): row["correct"] / row["pixels"] for cls in range(4) for row in rows if row["class"] == cls}
    balanced = _mean(list(classwise.values()))
    return {"balanced_accuracy": balanced, "class_accuracy": classwise, "rows": rows,
            "finite": bool(torch.isfinite(value).all())}


def purifier_health(result: dict) -> dict:
    ccbp = result["stages"][2]["hqmr"]["ccbp"]
    positive = F.interpolate(result["target_detail"]["positive"].float(), ccbp["gate_h4"].shape[-2:], mode="nearest").bool()
    present = result["deep_gate"].new_ones(result["deep_gate"].shape, dtype=torch.bool)
    gate, target_values, rival_values = ccbp["gate_h4"].float(), [], []
    active_values = []
    for cls in range(gate.shape[1]):
        mask = positive[:, cls]
        if mask.any():
            target_values.append(gate[:, cls][mask])
            rivals = torch.cat([gate[:, other][mask] for other in range(gate.shape[1]) if other != cls])
            rival_values.append(rivals)
            active_values.append(gate.permute(0, 2, 3, 1)[mask])
    target = torch.cat(target_values) if target_values else gate.new_empty(0)
    rival = torch.cat(rival_values) if rival_values else gate.new_empty(0)
    active = torch.cat(active_values).flatten() if active_values else gate.new_empty(0)
    accuracy = class_accuracy(ccbp["logits"], result["target_detail"]["positive"], result["deep_gate"].new_ones(result["deep_gate"].shape, dtype=torch.bool))
    return {"loss_ccbp": float(result["losses"]["loss_ccbp"].detach()), "balanced_accuracy": accuracy["balanced_accuracy"],
            **{f"accuracy_C{cls}": accuracy["class_accuracy"].get(str(cls), float("nan")) for cls in range(4)},
            "gamma": float(ccbp["gamma"].detach()), "mean_target_gate": float(target.mean()) if target.numel() else float("nan"),
            "mean_rival_gate": float(rival.mean()) if rival.numel() else float("nan"),
            "fraction_gate_lt_095": float((active < .95).float().mean()) if active.numel() else float("nan"),
            "fraction_gate_lt_075": float((active < .75).float().mean()) if active.numel() else float("nan"),
            "fraction_gate_lt_050": float((active < .50).float().mean()) if active.numel() else float("nan")}


@torch.no_grad()
def run_viability(args, output: Path) -> None:
    if sha256(args.hqmr_checkpoint) != HQMR_SHA256:
        raise AssertionError("Frozen HQMR-v1 identity mismatch")
    official.set_seed(42)
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, drop_last=False, worker_init_fn=official.seed_worker,
                        generator=generator)
    model = CCBPNet()
    missing, unexpected = model.load_state_dict(load_state(Path(args.hqmr_checkpoint)), strict=False)
    expected_missing = {name for name, _ in model.named_parameters() if name.startswith("ccbp.")}
    if set(missing) != expected_missing or unexpected:
        raise AssertionError({"missing": missing, "unexpected": unexpected})
    model = model.cuda().eval()
    correct, pixels = [0] * 4, [0] * 4
    finite, prototype_finite = True, True
    for index, (_, images, labels) in enumerate(loader, 1):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(images, labels, ccbp_mode="raw_space")
        payload = result["stages"][2]["hqmr"]["ccbp"]
        logits = payload["logits"]
        positive = result["target_detail"]["positive"].bool()
        value = F.interpolate(logits.float(), positive.shape[-2:], mode="bilinear", align_corners=False)
        prediction = value.argmax(1)
        for cls in range(4):
            mask = positive[:, cls]
            correct[cls] += int((prediction[mask] == cls).sum())
            pixels[cls] += int(mask.sum())
        finite &= bool(torch.isfinite(logits).all())
        prototype_finite &= bool(torch.isfinite(payload["prototype"]).all())
        if index % 100 == 0 or index == len(loader):
            print(f"CCBP_VIABILITY_PROGRESS={index}/{len(loader)}", flush=True)
    rows = [{"class": cls, "correct": correct[cls], "pixels": pixels[cls],
             "accuracy": correct[cls] / max(pixels[cls], 1)} for cls in range(4)]
    balanced = _mean([row["accuracy"] for row in rows if row["pixels"] > 0])
    if not finite or not prototype_finite or not all(pixels):
        decision = "CCBP_ENGINEERING_BLOCKED"
    else:
        decision = "CCBP_VIABILITY_GO" if balanced >= .55 else "CCBP_VIABILITY_WEAK"
    summary = {"decision": decision, "balanced_accuracy": balanced, "class_accuracy": rows,
               "score_finite": finite, "prototype_finite": prototype_finite,
               "images": len(dataset), "training_split_only": True, "gamma": 5.0,
               "query_projection": "identity", "semantic_projection": "identity", "parameter_updates": 0}
    write_csv(output / "preaudit/raw_space_train_viability.csv", rows)
    write_json(output / "preaudit/raw_space_viability_summary.json", summary)
    print("CCBP_VIABILITY " + json.dumps(summary, sort_keys=True), flush=True)


def gradient_contract(model: CCBPNet, images: torch.Tensor, labels: torch.Tensor) -> dict:
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(images[:2], labels[:2])
    named = dict(model.named_parameters())
    watched_names = [next(name for name in named if name.startswith("backbone.") and named[name].requires_grad),
                     "hqmr.scale4.key.0.weight", "hqmr.scale3.key.0.weight"]
    base = [named[name] for name in watched_names]
    purifier_names = [name for name in named if name.startswith("ccbp.")]
    purifier = [named[name] for name in purifier_names]
    base_grad = torch.autograd.grad(result["losses"]["loss_base"], base, retain_graph=True, allow_unused=True)
    total_grad = torch.autograd.grad(result["losses"]["loss"], base, retain_graph=True, allow_unused=True)
    aux_base = torch.autograd.grad(result["losses"]["loss_ccbp"], base, retain_graph=True, allow_unused=True)
    base_purifier = torch.autograd.grad(result["losses"]["loss_base"], purifier, retain_graph=True, allow_unused=True)
    aux_purifier = torch.autograd.grad(result["losses"]["loss_ccbp"], purifier, allow_unused=True)
    differences = [float((left - right).float().abs().max()) for left, right in zip(base_grad, total_grad)
                   if left is not None and right is not None]
    payload = {"watched_base_parameters": watched_names, "purifier_parameters": purifier_names,
               "base_total_max_abs_difference": max(differences, default=float("inf")),
               "grad_purifier_from_base_zero": all(value is None or not bool(value.detach().abs().any()) for value in base_purifier),
               "grad_base_from_ccbp_zero": all(value is None or not bool(value.detach().abs().any()) for value in aux_base),
               "grad_purifier_from_ccbp_nonzero": all(value is not None and bool(value.detach().abs().any()) for value in aux_purifier)}
    payload["passed"] = payload["base_total_max_abs_difference"] <= 1e-7 and payload["grad_purifier_from_base_zero"] and payload["grad_base_from_ccbp_zero"] and payload["grad_purifier_from_ccbp_nonzero"]
    model.zero_grad(set_to_none=True)
    return payload


@torch.no_grad()
def mechanism_snapshot(model: CCBPNet, loader, epoch: int) -> tuple[list[dict], list[dict]]:
    was_training = model.training; model.eval(); purifier_rows, basis_rows = [], []
    for _, images, labels in loader:
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=epoch * STEPS_PER_EPOCH)
        purifier_rows.append(purifier_health(result))
        basis_rows.append(_basis_health(result["stages"][2], result["targets"]))
    pkeys, bkeys = purifier_rows[0], basis_rows[0]
    psummary = {}
    for key in pkeys:
        finite = [float(row[key]) for row in purifier_rows if np.isfinite(float(row[key]))]
        psummary[key] = _mean(finite) if finite else None
    purifier = [{"snapshot": f"epoch{epoch}", **psummary}]
    basis = [{"snapshot": f"epoch{epoch}", "stage": 3, **{key: _mean([row[key] for row in basis_rows]) if key != "all_query_masks_identical" else all(row[key] for row in basis_rows) for key in bkeys}}]
    if was_training: model.train()
    return purifier, basis


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preaudit", "smoke", "train"), required=True)
    parser.add_argument("--trainroot", required=True); parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--cohort-json", required=True)
    parser.add_argument("--residual-audit-json", required=True); parser.add_argument("--hqmr-checkpoint", required=True)
    parser.add_argument("--viability-json"); parser.add_argument("--smoke-json")
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("CCBP Full25 requires RTX4090D with native BF16")
    output, weights = Path(args.output_dir).resolve(), Path(args.weights).resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing populated output: {output}")
    if sha256(weights) != INIT_SHA256 or weights.suffix != ".params": raise AssertionError("Official initialization mismatch")
    audit = json.loads(Path(args.residual_audit_json).read_text(encoding="utf-8"))
    if audit.get("decision") != "MIXED_CLASS23_BOTTLENECK" or audit.get("primary_bottleneck") != "basis purity":
        raise AssertionError("Frozen residual-audit evidence mismatch")
    for name in ("provenance", "preaudit", "tests", "logs", "metrics", "mechanism", "checkpoints", "recovery", "evaluation", "purity", "morphology", "ablation", "visualizations", "report"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if args.mode == "preaudit":
        run_viability(args, output); return
    if not args.viability_json: raise ValueError("smoke/train requires --viability-json")
    viability = json.loads(Path(args.viability_json).read_text(encoding="utf-8"))
    if viability["decision"] == "CCBP_ENGINEERING_BLOCKED": raise RuntimeError(viability)
    if args.mode == "train":
        if not args.smoke_json: raise ValueError("train requires --smoke-json from an independent fresh-init run")
        smoke = json.loads(Path(args.smoke_json).read_text(encoding="utf-8"))
        required = smoke.get("steps") == 2 and smoke.get("finite") is True and smoke.get("no_oom") is True
        required &= smoke.get("checkpoint_written") is False and smoke.get("gradient_contract", {}).get("passed") is True
        if not required: raise AssertionError({"invalid_smoke_evidence": smoke})
    if args.mode == "train" and _git("status", "--porcelain"):
        raise AssertionError("Formal step0 requires clean source")
    log = (output / "logs/ccbp_full25_train.log").open("w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log); sys.stderr = Tee(sys.__stderr__, log)
    accesses = install_train_access_guard(); official.set_seed(42); source = _git("rev-parse", "HEAD")
    config = {**CONFIG, "source_commit": source, "trainroot": str(Path(args.trainroot).resolve()),
              "weights": str(weights), "mode": args.mode, "viability": viability["decision"]}
    write_json(output / "provenance/ccbp_config.json", config)
    (output / "provenance/ccbp_config_sha256.txt").write_text(_canonical_hash(config) + "\n")
    (output / "provenance/ccbp_source_commit.txt").write_text(source + "\n")
    (output / "provenance/ccbp_git_diff.patch").write_text(subprocess.check_output(["git", "show", "--format=", "--binary", "HEAD"], cwd=ROOT, text=True))
    (output / "provenance/ccbp_environment.txt").write_text(f"python\t{sys.version.replace(chr(10), ' ')}\ntorch\t{torch.__version__}\ncuda\t{torch.version.cuda}\ngpu\t{torch.cuda.get_device_name(0)}\nplatform\t{platform.platform()}\n")
    write_json(output / "provenance/ccbp_residual_audit.json", {"sha256": sha256(args.residual_audit_json), "payload": audit})
    write_json(output / "provenance/ccbp_protocol_audit.json", {
        "protected_sources": protected_sources(ROOT), "fresh_official_init": True,
        "fresh_hqmr": True, "fresh_identity_initialized_ccbp": True,
        "seed": 42, "epochs": 25, "steps": 29275, "effective_batch": 20,
        "validation_during_training": False, "trained_checkpoint_loaded": False,
        "stage3_only": True, "suppress_only": True, "no_threshold_tuning": True})
    write_json(output / "preaudit/raw_space_viability_summary.json", viability)
    if args.mode == "train": write_json(output / "tests/ccbp_smoke.json", smoke)
    unit = subprocess.run([sys.executable, "-m", "pytest", "tests/test_ccbp.py", "tests/test_hqmr.py", "-q",
                           "--basetemp", "/tmp/ccbp-unit"], cwd=ROOT, text=True, capture_output=True)
    write_json(output / "tests/ccbp_unit_tests.json", {"passed": unit.returncode == 0, "returncode": unit.returncode,
                                                        "stdout": unit.stdout, "stderr": unit.stderr})
    if unit.returncode: raise AssertionError(unit.stdout + unit.stderr)
    regression = subprocess.run([sys.executable, "-m", "pytest", "-q", "--basetemp", "/tmp/ccbp-regression"],
                                cwd=ROOT, text=True, capture_output=True)
    write_json(output / "tests/ccbp_regression_tests.json", {"passed": regression.returncode == 0,
        "returncode": regression.returncode, "stdout": regression.stdout, "stderr": regression.stderr})
    if regression.returncode: raise AssertionError(regression.stdout + regression.stderr)
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, worker_init_fn=official.seed_worker, generator=generator)
    if len(dataset) != 23422 or len(loader) != STEPS_PER_EPOCH or FULL25_STEPS != TOTAL_STEPS:
        raise AssertionError("Frozen cardinality mismatch")
    names = sorted(Path(path).name for path, _ in dataset.object)
    write_json(output / "provenance/ccbp_dataset.json", {"samples": len(dataset),
        "filename_manifest_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "training_only": True, "validation_accessed": False, "test_accessed": False})
    model = CCBPNet(); initialization = model.backbone.load_official_initialization(str(weights))
    initialization.update({"sha256": sha256(weights), "fresh_official_initialization": True,
                           "trained_checkpoint_loaded": False, "ccbp_identity_initialized": True})
    write_json(output / "provenance/ccbp_init_identity.json", initialization)
    hqmr_count = sum(p.numel() for p in HQMRNet().parameters()); count = sum(p.numel() for p in model.parameters())
    if count - hqmr_count != CCBP_PARAMETERS: raise AssertionError(f"Unexpected CCBP parameter delta: {count-hqmr_count}")
    write_json(output / "provenance/ccbp_parameter_counts.json", {"hqmr": hqmr_count, "ccbp": count,
        "delta": count - hqmr_count, "under_half_million": count - hqmr_count < 500_000})
    model = model.cuda(); groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([{"params": group, "lr": .01 * multiplier, "weight_decay": decay}
        for group, multiplier, decay in zip(groups, (1, 2, 10, 20), (.0005, 0, .0005, 0))],
        lr=.01, weight_decay=.0005, max_step=TOTAL_STEPS)
    cohort_payload, cohort = load_cohort(args.cohort_json)
    write_json(output / "provenance/ccbp_monitor_cohort.json", cohort_payload)
    monitor_loader = DataLoader(MonitorDataset(cohort), batch_size=8, num_workers=4, pin_memory=True)
    contract_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    first = next(iter(contract_loader)); contract = gradient_contract(model, first[1].cuda(), first[2].cuda())
    write_json(output / "tests/ccbp_gradient_contract.json", contract)
    if not contract["passed"]: raise AssertionError(contract)
    losses, epochs, purifier_rows, hqmr_rows = [], [], [], []
    started = time.perf_counter(); peak = 0.0; target = 2 if args.mode == "smoke" else TOTAL_STEPS
    torch.cuda.reset_peak_memory_stats(); print("CCBP_FULL25_PROTOCOL " + json.dumps(config, sort_keys=True), flush=True)
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter(); model.train(); sums = {}; batches = 0
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(images, labels, step=optimizer.global_step)
            loss = result["losses"]["loss"]
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Non-finite CCBP loss")
            loss.backward()
            health = _gradient_health(model) if optimizer.global_step == 0 or (optimizer.global_step + 1) % 100 == 0 else {}
            optimizer.step(); batches += 1
            if not _finite_model(model): raise FloatingPointError("Non-finite CCBP parameter")
            for key, value in result["losses"].items(): sums[key] = sums.get(key, 0.0) + float(value.detach())
            if optimizer.global_step % 100 == 0 or optimizer.global_step == target:
                row = {"epoch": epoch, "step": optimizer.global_step,
                       **{key: value / batches for key, value in sums.items()},
                       "lr": optimizer.param_groups[0]["lr"], **health}
                losses.append(row); write_csv(output / "metrics/train_loss.csv", losses)
                print("CCBP_FULL25_STEP " + json.dumps(row, sort_keys=True), flush=True)
            if optimizer.global_step >= target: break
        peak = max(peak, torch.cuda.max_memory_allocated() / 1024 ** 3)
        epoch_row = {"epoch": epoch, "step": optimizer.global_step,
                     **{key: value / batches for key, value in sums.items()},
                     "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - epoch_started,
                     "peak_cuda_memory_gib": peak}
        epochs.append(epoch_row); write_csv(output / "metrics/epoch_summary.csv", epochs)
        print("CCBP_FULL25_EPOCH " + json.dumps(epoch_row, sort_keys=True), flush=True)
        if args.mode == "smoke":
            p_rows, h_rows = mechanism_snapshot(model, monitor_loader, 0)
            numeric = [value for rows in (p_rows, h_rows) for row in rows for value in row.values()
                       if isinstance(value, (float, int))]
            finite = all(np.isfinite(value) for value in numeric)
            smoke = {"steps": optimizer.global_step, "finite": finite, "no_oom": True,
                     "parameter_delta": CCBP_PARAMETERS, "gradient_contract": contract,
                     "purifier": p_rows, "hqmr": h_rows, "validation_accessed": False,
                     "checkpoint_written": False,
                     "stage_outputs": [list(result["stages"][0]["mask_logits"].shape),
                                       list(result["stages"][1]["hqmr"]["mixture"].shape),
                                       list(result["stages"][2]["hqmr"]["ccbp"]["mixture"].shape)],
                     "purifier_logits_shape": list(result["stages"][2]["hqmr"]["ccbp"]["logits"].shape)}
            if not finite: raise FloatingPointError(smoke)
            write_json(output / "tests/ccbp_smoke.json", smoke)
            print("CCBP_FULL25_SMOKE_PASS", flush=True); return
        if epoch in MILESTONES:
            p_rows, h_rows = mechanism_snapshot(model, monitor_loader, epoch)
            purifier_rows.extend(p_rows); hqmr_rows.extend(h_rows)
            write_csv(output / "mechanism/ccbp_train_health.csv", purifier_rows)
            write_csv(output / "mechanism/ccbp_gate_stats.csv", purifier_rows)
            write_csv(output / "mechanism/ccbp_hqmr_health.csv", hqmr_rows)
            write_csv(output / "mechanism/ccbp_ccra_health.csv", [{"snapshot": row["snapshot"],
                "ccra_js": row["ccra_js"], "D_perm": row["D_perm"], "Neff": row["Neff"]} for row in hqmr_rows])
            latest = purifier_rows[-1]
            collapse = latest["gamma"] > 19.5 and latest["fraction_gate_lt_050"] > .80
            inactive = epoch == 25 and all(row["fraction_gate_lt_095"] < .01 for row in purifier_rows)
            health_decision = "PURIFIER_COLLAPSE" if collapse else "PURIFIER_INACTIVE" if inactive else "CONTINUE_FULL25_UNCHANGED"
            write_json(output / f"mechanism/ccbp_epoch{epoch:02d}_health.json", {"decision": health_decision,
                "purifier": p_rows, "hqmr": h_rows})
            checkpoint = output / f"checkpoints/ccbp_epoch{epoch:02d}.pth"
            torch.save(model.state_dict(), checkpoint)
            write_json(checkpoint.with_suffix(".json"), {"epoch": epoch, "step": optimizer.global_step,
                "sha256": sha256(checkpoint), "scientific_endpoint": epoch == 25})
            _save_recovery(output / "recovery/latest.pth", model, optimizer, epoch, generator)
            if collapse:
                write_json(output / "provenance/ccbp_runtime.json", {"status": "PURIFIER_COLLAPSE",
                    "epoch": epoch, "steps": optimizer.global_step, "validation_accessed": False})
                print("DECISION = CCBP_ENGINEERING_BLOCKED", flush=True); return
    if optimizer.global_step != TOTAL_STEPS or len(epochs) != 25:
        raise AssertionError("CCBP did not reach E25")
    source_checkpoint = output / "checkpoints/ccbp_epoch25.pth"
    final = output / "checkpoints/ccbp_epoch25_final.pth"
    os.replace(source_checkpoint, final); digest = sha256(final)
    (output / "checkpoints/ccbp_epoch25_final_sha256.txt").write_text(digest + "\n")
    metadata = json.loads((output / "checkpoints/ccbp_epoch25.json").read_text())
    metadata.update({"sha256": digest, "sealed_before_segmentation_evaluation": True,
                     "selection": "E25 FINAL only"})
    write_json(output / "checkpoints/ccbp_epoch25_final.json", metadata)
    (output / "checkpoints/ccbp_epoch25.json").unlink()
    runtime = {"status": "CCBP_FULL25_COMPLETE", "epochs": 25, "steps": optimizer.global_step,
               "seconds": time.perf_counter() - started, "peak_cuda_memory_gib": peak,
               "validation_accessed": False, "parameter_updates": TOTAL_STEPS,
               "train_accesses": len(accesses), "checkpoint": str(final), "sha256": digest}
    write_json(output / "provenance/ccbp_runtime.json", runtime)
    print("CCBP_FULL25_COMPLETE " + json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
