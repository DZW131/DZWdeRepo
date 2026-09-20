#!/usr/bin/env python3
"""Train the three frozen-HQMR RACC-v1 Phase-0 groups for five epochs."""
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
from network.hqmr_net import HQMRNet
from network.racc_net import RACCNet
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.eval_gcqm_full25_bcss_seed42 import THRESHOLDS, load_state
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.run_gcqm_full25_bcss_seed42 import _finite_model, _git
from train_cqrf_phase0 import Tee

HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
EPOCHS, STEPS_PER_EPOCH, TOTAL_STEPS = 5, 1171, 5855
GROUPS = {"P1_RACC_A": (True, False), "P2_RACC_G": (False, True), "P3_RACC_JOINT": (True, True)}


def load_racc(checkpoint: Path) -> RACCNet:
    model = RACCNet()
    missing, unexpected = model.load_state_dict(load_state(checkpoint), strict=False)
    expected = {name for name, _ in model.named_parameters() if name.startswith("racc.")}
    if set(missing) != expected or unexpected: raise AssertionError({"missing": missing, "unexpected": unexpected})
    model.freeze_hqmr()
    return model


def set_group(model: RACCNet, enable_a: bool, enable_g: bool) -> list[torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("racc.arbitration.") and enable_a or
                                 name.startswith("racc.presence.") and enable_g)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def identity_tests(checkpoint: Path, images: torch.Tensor, labels: torch.Tensor) -> dict:
    base = HQMRNet().cuda().eval(); base.load_state_dict(load_state(checkpoint), strict=True)
    model = load_racc(checkpoint).cuda().eval()
    with torch.no_grad():
        reference = base(images, labels, step=29275)["primary_output"].float()
        disabled = model(images, labels, step=29275, enable_arbitration=False, enable_presence=False)
        a_only = model(images, labels, step=29275, enable_arbitration=True, enable_presence=False)
        joint = model(images, labels, step=29275, enable_arbitration=True, enable_presence=True)
    threshold = torch.as_tensor(THRESHOLDS, device=images.device)
    deep = disabled["deep_gate"] > threshold
    rescued = model.racc.rescued_gate(joint["deep_gate"], joint["racc"]["local_presence"]["probability"], threshold)
    tests = {
        "A_disabled_max_abs_error": float((disabled["primary_output"].float() - reference).abs().max()),
        "A_disabled_same_argmax": bool(torch.equal(disabled["primary_output"].argmax(1), reference.argmax(1))),
        "B_arbitration_identity_max_abs_error": float((a_only["primary_output"].float() - disabled["primary_output"].float()).abs().max()),
        "B_alpha_max_abs_error_from_one": float((a_only["racc"]["alpha_stage3"].float() - 1).abs().max()),
        "C_presence_gate_identical": bool(torch.equal(deep, rescued)),
        "D_joint_mask_max_abs_error": float((joint["primary_output"].float() - disabled["primary_output"].float()).abs().max()),
    }
    tests["passed"] = tests["A_disabled_max_abs_error"] < 2e-4 and tests["A_disabled_same_argmax"] and tests["B_arbitration_identity_max_abs_error"] < 1e-6 and tests["B_alpha_max_abs_error_from_one"] < 1e-6 and tests["C_presence_gate_identical"] and tests["D_joint_mask_max_abs_error"] < 1e-6
    return tests


def gradient_test(checkpoint: Path, images: torch.Tensor, labels: torch.Tensor, name: str,
                  enable_a: bool, enable_g: bool) -> dict:
    model = load_racc(checkpoint).cuda().train(); trained = set_group(model, enable_a, enable_g)
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(images, labels, enable_arbitration=enable_a, enable_presence=enable_g)
    result["losses"]["loss"].backward()
    trained_ids = {id(parameter) for parameter in trained}
    old = {n: 0.0 if p.grad is None else float(p.grad.float().abs().sum()) for n, p in model.named_parameters() if not n.startswith("racc.")}
    new = {n: 0.0 if p.grad is None else float(p.grad.float().abs().sum()) for n, p in model.named_parameters() if id(p) in trained_ids}
    payload = {"group": name, "old_gradient_sum": sum(old.values()), "new_gradient_sum": sum(new.values()),
               "new_nonzero_tensors": sum(v > 0 for v in new.values()), "new_tensor_count": len(new),
               "old_parameter_count": len(old), "new_gradients": new}
    payload["passed"] = payload["old_gradient_sum"] == 0 and payload["new_gradient_sum"] > 0
    return payload


@torch.no_grad()
def mechanism_snapshot(model: RACCNet, loader, enable_a: bool, enable_g: bool) -> dict:
    model.eval(); alpha, probability, target = [], [], []
    for index, (_, images, labels) in enumerate(loader):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(images, labels, step=29275, enable_arbitration=enable_a, enable_presence=enable_g)
        if enable_a: alpha.append(result["racc"]["alpha_stage3"].float().cpu().flatten())
        probability.append(result["racc"]["local_presence"]["probability"].float().cpu().flatten())
        target.append(labels.cpu().flatten())
        if index == 4: break
    p, y = torch.cat(probability), torch.cat(target)
    item = {"presence_mean": float(p.mean()), "presence_positive_mean": float(p[y.bool()].mean()),
            "presence_negative_mean": float(p[~y.bool()].mean())}
    if alpha:
        a = torch.cat(alpha); item.update({"alpha_mean": float(a.mean()), "alpha_std": float(a.std()),
            "alpha_lt_0.5": float((a < .5).float().mean()), "alpha_0.5_0.8": float(((a >= .5) & (a < .8)).float().mean()),
            "alpha_0.8_1.2": float(((a >= .8) & (a < 1.2)).float().mean()), "alpha_1.2_2": float(((a >= 1.2) & (a < 2)).float().mean()),
            "alpha_2_3": float(((a >= 2) & (a < 3)).float().mean()), "alpha_gt_3": float((a >= 3).float().mean())})
    return item


def train_group(args, dataset, output: Path, name: str, enable_a: bool, enable_g: bool, smoke: bool) -> dict:
    group_dir = output / name; group_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=args.num_workers, pin_memory=True,
                        drop_last=True, worker_init_fn=official.seed_worker, generator=generator)
    monitor = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    model = load_racc(Path(args.hqmr_checkpoint)).cuda(); params = set_group(model, enable_a, enable_g)
    model.eval()
    frozen_buffers = {name: value.detach().cpu().clone() for name, value in model.named_buffers()
                      if not name.startswith("racc.")}
    weights = [p for n, p in model.named_parameters() if p.requires_grad and not n.endswith("bias")]
    biases = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith("bias")]
    optimizer = PolyOptimizer([{"params": weights, "lr": .1, "weight_decay": .0005},
                               {"params": biases, "lr": .2, "weight_decay": 0}],
                              lr=.01, weight_decay=.0005, max_step=TOTAL_STEPS)
    rows, epochs = [], []; started = time.perf_counter(); target = 2 if smoke else TOTAL_STEPS
    for epoch in range(1, EPOCHS + 1):
        # The entire archived HQMR, including BatchNorm running buffers, is frozen.
        # RACC-v1 has no train/eval-dependent layers, so the complete module stays in eval mode.
        model.eval(); sums = {}; batches = 0; epoch_started = time.perf_counter()
        for _, images, labels in loader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(images, labels, step=optimizer.global_step,
                               enable_arbitration=enable_a, enable_presence=enable_g)
            loss = result["losses"]["loss"]
            if not bool(torch.isfinite(loss)): raise FloatingPointError(f"{name} non-finite loss")
            loss.backward(); optimizer.step(); batches += 1
            if not _finite_model(model): raise FloatingPointError(f"{name} non-finite parameter")
            for key, value in result["losses"].items(): sums[key] = sums.get(key, 0.0) + float(value.detach())
            if optimizer.global_step % 100 == 0 or optimizer.global_step == target:
                row = {"group": name, "epoch": epoch, "step": optimizer.global_step,
                       **{key: value / batches for key, value in sums.items()}, "lr": optimizer.param_groups[0]["lr"]}
                rows.append(row); write_csv(group_dir / "train_loss.csv", rows)
                print("RACC_PHASE0_STEP " + json.dumps(row, sort_keys=True), flush=True)
            if optimizer.global_step >= target: break
        snapshot = mechanism_snapshot(model, monitor, enable_a, enable_g)
        epoch_row = {"group": name, "epoch": epoch, "step": optimizer.global_step,
                     **{key: value / batches for key, value in sums.items()}, **snapshot,
                     "seconds": time.perf_counter() - epoch_started}
        epochs.append(epoch_row); write_csv(group_dir / "epoch_summary.csv", epochs)
        print("RACC_PHASE0_EPOCH " + json.dumps(epoch_row, sort_keys=True), flush=True)
        if smoke or optimizer.global_step >= target: break
    if smoke: return {"group": name, "steps": optimizer.global_step, "finite": True, "mechanism": epochs[-1]}
    if optimizer.global_step != TOTAL_STEPS: raise AssertionError(f"{name} incomplete")
    changed_buffers = [name for name, value in model.named_buffers()
                       if name in frozen_buffers and not torch.equal(value.detach().cpu(), frozen_buffers[name])]
    if changed_buffers: raise AssertionError({"frozen_buffers_changed": changed_buffers})
    checkpoint = group_dir / "racc_epoch05_final.pth"; torch.save(model.state_dict(), checkpoint)
    seal = {"group": name, "epoch": 5, "steps": optimizer.global_step, "sha256": sha256(checkpoint),
            "checkpoint": str(checkpoint), "selection": "fixed Epoch5 only", "validation_accessed": False,
            "seconds": time.perf_counter() - started, "trainable_parameters": sum(p.numel() for p in params),
            "frozen_buffers_unchanged": True}
    write_json(group_dir / "seal.json", seal); return seal


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--mode", choices=("smoke", "train"), required=True)
    p.add_argument("--trainroot", required=True); p.add_argument("--hqmr-checkpoint", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--num-workers", type=int, default=8); return p.parse_args()


def main():
    args = parse_args(); check_train_path(args.trainroot)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("RACC Phase0 requires RTX4090D BF16")
    checkpoint, output = Path(args.hqmr_checkpoint).resolve(), Path(args.output_dir).resolve()
    if sha256(checkpoint) != HQMR_SHA256: raise AssertionError("Frozen HQMR checkpoint identity mismatch")
    if output.exists() and any(output.iterdir()): raise FileExistsError(output)
    for name in ("identity_tests", "gradient_tests", "metrics", "visualizations", "provenance", "logs", "report", "P0_HQMR"):
        (output / name).mkdir(parents=True, exist_ok=True)
    log = (output / "logs/racc_phase0_train.log").open("w", buffering=1); sys.stdout = Tee(sys.__stdout__, log); sys.stderr = Tee(sys.__stderr__, log)
    accesses = install_train_access_guard(); official.set_seed(42); source = _git("rev-parse", "HEAD")
    dataset = Stage1_TrainDataset(args.trainroot, dataset="bcss", img_size=224)
    if len(dataset) != 23422: raise AssertionError(len(dataset))
    probe_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0); _, images, labels = next(iter(probe_loader))
    images, labels = images.cuda(), labels.cuda()
    identity = identity_tests(checkpoint, images, labels); write_json(output / "identity_tests/identity_manifest.json", identity)
    if not identity["passed"]: raise AssertionError(identity)
    gradients = {name: gradient_test(checkpoint, images, labels, name, *flags) for name, flags in GROUPS.items()}
    gradients["passed"] = all(row["passed"] for row in gradients.values()); write_json(output / "gradient_tests/gradient_manifest.json", gradients)
    if not gradients["passed"]: raise AssertionError(gradients)
    base_params = sum(p.numel() for p in HQMRNet().parameters()); probe = RACCNet()
    counts = {"hqmr": base_params, "racc_a": sum(p.numel() for p in probe.racc.arbitration.parameters()),
              "racc_g": sum(p.numel() for p in probe.racc.presence.parameters())}
    counts["racc_total"] = counts["racc_a"] + counts["racc_g"]; counts["percent_of_hqmr"] = 100 * counts["racc_total"] / base_params
    write_json(output / "metrics/parameter_counts.json", counts)
    config = {"experiment": "RACC-v1 Phase0 BCSS Seed42", "source_commit": source, "mode": args.mode,
              "seed": 42, "epochs": 5, "steps_per_epoch": STEPS_PER_EPOCH, "total_steps_per_group": TOTAL_STEPS,
              "batch_size": 20, "image_size": 224, "precision": "bf16", "optimizer": "PolyOptimizer",
              "new_weight_lr": .1, "new_bias_lr": .2, "weight_decay": .0005, "groups": GROUPS,
              "hqmr_checkpoint": str(checkpoint), "hqmr_sha256": sha256(checkpoint), "validation_during_training": False,
              "image_level_labels_only": True, "fixed_epoch5": True, "threshold_sweep": False,
              "hqmr_eval_mode_during_training": True, "frozen_buffers_checked_exactly": True}
    write_json(output / "provenance/racc_phase0_config.json", config)
    write_json(output / "provenance/protocol.json", {"protected_sources": protected_sources(ROOT), "old_parameters_frozen": True,
        "training_samples": len(dataset), "dataset_manifest_sha256": hashlib.sha256("\n".join(sorted(Path(x).name for x, _ in dataset.object)).encode()).hexdigest(),
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0), "platform": platform.platform()}})
    results = {name: train_group(args, dataset, output, name, *flags, smoke=args.mode == "smoke") for name, flags in GROUPS.items()}
    runtime = {"status": "RACC_PHASE0_SMOKE_PASS" if args.mode == "smoke" else "RACC_PHASE0_TRAINING_COMPLETE",
               "groups": results, "validation_accessed": False, "train_path_accesses": len(accesses)}
    write_json(output / "provenance/runtime.json", runtime); print(runtime["status"] + " " + json.dumps(runtime, sort_keys=True), flush=True)


if __name__ == "__main__": main()
