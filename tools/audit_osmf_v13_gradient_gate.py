"""Run gated OSMF-v1.3 8-batch readiness or 128-batch Phase-0S."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.osmf_v13 import (
    affinity_equivariance_error,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_preservation_agreement,
    semantic_preservation_loss,
    structural_affinity_loss,
)
from network.resnet38_cls_osmf_v13 import Net
from tool.GenDataset import Stage1_TrainDataset
from tools.audit_osmf_phase0_128batch import (
    BASELINE_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    _build_optimizer,
    _flip_dimension,
    sha256_file,
)
from tools.audit_osmf_v12_gradient_gate import (
    EXPECTED_MISSING_KEYS,
    _load_checkpoint,
    _parameter_summary,
)
from tools.audit_osmf_v12_phase0m import (
    _build_fixed_probe,
    _fixed_probe_measure,
    _tensor_sha256,
)
from tools.osmf_phase0_audit.gradients import (
    gradient_decomposition,
    parameter_gradient_rows,
    parameter_update_rows,
    snapshot_parameters,
)
from tools.osmf_v13_audit import (
    BATCH_SIZE, FIXED_PROBE_STEPS, IMAGE_SIZE, MORPHOLOGY_PARAMETER_NAMES,
    OBJECTIVE_NAMES, OBJECTIVE_WEIGHTS, PARAMETER_NAMES,
    PHASE0S_AUDIT_STEPS, PHASE0S_BATCHES, PROBE_IMAGES,
    READINESS_AUDIT_STEPS, READINESS_BATCHES, SEED,
)
from tools.osmf_v13_audit.decision import phase0s_decision, readiness_decision
from tools.osmf_v13_audit.report import write_csv, write_report
from train_osmf_v13 import sshr_classification_loss
from train_sshr import seed_worker, set_seed


def _rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt().cpu())


def _forward_objectives(model, images, labels, step, force_structural=False):
    outputs, aux = model.forward_with_aux(images)
    base = sshr_classification_loss(outputs, labels)
    sem_pres = semantic_preservation_loss(
        aux["semantic_student_response"], aux["semantic_teacher_response"]
    )
    orth = orthogonality_loss(aux["semantic"], aux["morphology"])
    rec = reconstruction_loss(aux["reconstruction"], aux["input"])
    scheduled = step > 0 and step % 4 == 0
    struct = struct_semantic = raw_morphology = raw_semantic = None
    if scheduled or force_structural:
        flip_dimension = _flip_dimension(step if step > 0 else 4)
        second = model.forward_osmf_features(torch.flip(images, dims=(flip_dimension,)))
        struct = structural_affinity_loss(aux["morphology"], second["morphology"], flip_dimension)
        struct_semantic = structural_affinity_loss(aux["semantic"], second["semantic"], flip_dimension)
        raw_morphology = (aux["morphology"] - inverse_align_morphology(second["morphology"], flip_dimension)).abs().mean()
        raw_semantic = (aux["semantic"] - inverse_align_morphology(second["semantic"], flip_dimension)).abs().mean()
    struct_total = struct if scheduled else base.new_zeros(())
    total = (
        base + OBJECTIVE_WEIGHTS["sem_pres"] * sem_pres
        + OBJECTIVE_WEIGHTS["struct"] * struct_total
        + OBJECTIVE_WEIGHTS["orth"] * orth
        + OBJECTIVE_WEIGHTS["rec"] * rec
    )
    return {
        "outputs": outputs, "aux": aux, "base": base, "sem_pres": sem_pres,
        "struct": struct, "struct_semantic": struct_semantic,
        "raw_morphology": raw_morphology, "raw_semantic": raw_semantic,
        "orth": orth, "rec": rec, "total": total,
        "scheduled_struct": scheduled,
    }


def _finite(bundle):
    tensors = list(bundle["outputs"]) + list(bundle["aux"].values())
    tensors += [bundle[name] for name in ("base", "sem_pres", "orth", "rec", "total")]
    tensors += [bundle[name] for name in ("struct", "struct_semantic", "raw_morphology", "raw_semantic") if bundle[name] is not None]
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _diagnostic(model, images, labels, step):
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        bundle = _forward_objectives(model, images, labels, step, True)
    model.train(was_training)
    aux = bundle["aux"]
    h_rms = _rms(aux["input"])
    row = {
        "step": step, "h_rms": h_rms,
        "semantic_rms": _rms(aux["semantic"]),
        "morphology_rms": _rms(aux["morphology"]),
        "semantic_morphology_rms_ratio": _rms(aux["semantic"]) / (_rms(aux["morphology"]) + 1e-12),
        "reconstruction_cosine": float(reconstruction_cosine(aux["reconstruction"], aux["input"]).cpu()),
        "residual_ratio": _rms(aux["reconstruction"] - aux["input"]) / (h_rms + 1e-12),
        "cross_covariance": float(cross_subspace_covariance(aux["semantic"], aux["morphology"]).square().mean().sqrt().cpu()),
        "struct_error_morphology": float(bundle["struct"].cpu()),
        "struct_error_semantic": float(bundle["struct_semantic"].cpu()),
        "raw_eq_error_morphology": float(bundle["raw_morphology"].cpu()),
        "raw_eq_error_semantic": float(bundle["raw_semantic"].cpu()),
        "semantic_response_rms_ratio": _rms(aux["semantic_student_response"]) / (_rms(aux["semantic_teacher_response"]) + 1e-12),
        "semantic_agreement": float(semantic_preservation_agreement(aux["semantic_student_response"], aux["semantic_teacher_response"]).cpu()),
        "finite": _finite(bundle),
    }
    identity_error = float((aux["reconstruction"] - aux["input"]).abs().max().cpu())
    return row, identity_error


def _eval_pair(model, images, flip_dimension):
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = model.forward_osmf_features(images)
        second = model.forward_osmf_features(torch.flip(images, dims=(flip_dimension,)))
        result = {
            "struct_morphology": float(structural_affinity_loss(first["morphology"], second["morphology"], flip_dimension).cpu()),
            "affinity_morphology": float(affinity_equivariance_error(first["morphology"], second["morphology"], flip_dimension).cpu()),
            "affinity_semantic": float(affinity_equivariance_error(first["semantic"], second["semantic"], flip_dimension).cpu()),
        }
    model.train(was_training)
    return result


def _proof(path, decision, commit_key, commit):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != decision or payload.get(commit_key) != commit:
        raise RuntimeError(f"Invalid gate proof: {path}")
    return sha256_file(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, choices=("readiness", "phase0s"))
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parity-summary", required=True, type=Path)
    parser.add_argument("--readiness-summary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--num-workers", default=4, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    if any(token in str(args.train_root).lower() for token in ("test", "luad", "val")):
        raise ValueError("Only BCSS training images are authorized")
    if len(args.audit_commit) != 40:
        raise ValueError("--audit-commit must be a full Git SHA")
    parity_sha = _proof(args.parity_summary, "OSMF_V13_PARITY_PASS", "osmf_v13_commit", args.audit_commit)
    readiness_sha = None
    if args.gate == "phase0s":
        if args.readiness_summary is None:
            raise ValueError("Phase-0S requires --readiness-summary")
        readiness_sha = _proof(args.readiness_summary, "OSMF_V13_READINESS_PASS", "audit_commit", args.audit_commit)
    elif args.readiness_summary is not None:
        raise ValueError("Readiness does not accept a readiness proof")
    authorized = READINESS_BATCHES if args.gate == "readiness" else PHASE0S_BATCHES
    audit_steps = READINESS_AUDIT_STEPS if args.gate == "readiness" else PHASE0S_AUDIT_STEPS
    active_steps = tuple(range(4, authorized + 1, 4))
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f"Unexpected checkpoint SHA256: {checkpoint_sha}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config", "tables", "docs"):
        (args.output_dir / name).mkdir()

    set_seed(SEED)
    model = Net(n_class=4).cuda()
    dataset = Stage1_TrainDataset(data_path=str(args.train_root), dataset="bcss", img_size=IMAGE_SIZE)
    if len(dataset) != 23422:
        raise AssertionError(f"Expected 23422 BCSS training samples, got {len(dataset)}")
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True, worker_init_fn=seed_worker, generator=generator)
    optimizer, max_step = _build_optimizer(model, len(dataset))
    missing, unexpected = _load_checkpoint(model, args.checkpoint)
    if set(missing) != EXPECTED_MISSING_KEYS or unexpected:
        raise AssertionError("Checkpoint compatibility changed")
    model.train()
    probe_records = None
    if args.gate == "phase0s":
        probe_records, probe_manifest = _build_fixed_probe(dataset)
        write_csv(args.output_dir / "tables" / "fixed_probe_manifest.csv", probe_manifest)

    contract = {
        "gate": args.gate, "authorized_batches": authorized,
        "audit_steps": audit_steps, "active_steps": active_steps,
        "seed": SEED, "batch_size": BATCH_SIZE, "image_size": IMAGE_SIZE,
        "precision": "bf16", "objective_weights": OBJECTIVE_WEIGHTS,
        "structural_interval": 4, "smooth_l1_beta": 1.0,
        "baseline_commit": BASELINE_COMMIT, "audit_commit": args.audit_commit,
        "checkpoint_sha256": checkpoint_sha,
        "parity_summary_sha256": parity_sha, "readiness_summary_sha256": readiness_sha,
        "missing_keys": missing, "unexpected_keys": unexpected,
        "optimizer": {"class": type(optimizer).__name__, "max_step": max_step, "lr_power": optimizer.lr_power, "groups": [{"lr": float(g["lr"]), "weight_decay": float(g["weight_decay"]), "momentum": float(g["momentum"]), "tensors": len(g["params"])} for g in optimizer.param_groups]},
        "exact_command": " ".join(sys.argv),
        "validation_evaluated": False, "test_evaluated": False,
        "luad_evaluated": False, "segmentation_gt_used": False,
    }
    (args.output_dir / "config" / "frozen_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    iterator = iter(loader)
    names, images, labels = next(iterator)
    images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    start_rep, identity_error = _diagnostic(model, images, labels, 0)
    if identity_error >= 1e-6 or not start_rep["finite"]:
        raise RuntimeError("Start-state parity/finite failure")
    representation_rows, loss_rows = [start_rep], []
    ratio_rows, cosine_rows, causal_rows, morph_gradient_rows = [], [], [], []
    parameter_gradient_rows_all, parameter_update_rows_all = [], []
    fixed_rows = [_fixed_probe_measure(model, probe_records, 0)] if probe_records else []
    initial = snapshot_parameters(model.osmf_28_1, PARAMETER_NAMES)
    finite, processed, peak_memory = True, 0, 0
    started = time.perf_counter()

    for step in range(1, authorized + 1):
        if step > 1:
            names, images, labels = next(iterator)
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        flip_dimension = _flip_dimension(step)
        before_pair = _eval_pair(model, images, flip_dimension) if step in active_steps else None
        if step in audit_steps[1:]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                audit = _forward_objectives(model, images, labels, step, True)
            ratios, cosines = gradient_decomposition(
                {"base": audit["base"], "sem_pres": audit["sem_pres"], "struct": audit["struct"], "orth": audit["orth"], "rec": audit["rec"]},
                audit["aux"]["input"], tuple(model.osmf_28_1.parameters()), OBJECTIVE_WEIGHTS, objective_names=OBJECTIVE_NAMES,
            )
            ratio_rows.extend({"step": step, **row} for row in ratios)
            cosine_rows.extend({"step": step, **row} for row in cosines)
            if step in active_steps:
                selected = dict(model.osmf_28_1.named_parameters())
                grads = torch.autograd.grad(audit["struct"], tuple(selected[name] for name in MORPHOLOGY_PARAMETER_NAMES), retain_graph=False, allow_unused=True)
                morph_gradient_rows.append({"step": step, "p_morph_grad_norm": 0.0 if grads[0] is None else float(grads[0].float().norm().cpu()), "u_morph_grad_norm": 0.0 if grads[1] is None else float(grads[1].float().norm().cpu())})
            finite = finite and _finite(audit) and all(row["finite"] for row in ratios + cosines)
            del audit, ratios, cosines

        optimizer.zero_grad()
        before_parameters = snapshot_parameters(model.osmf_28_1, PARAMETER_NAMES) if step in audit_steps[1:] else None
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bundle = _forward_objectives(model, images, labels, step, False)
        bundle["total"].backward()
        finite = finite and _finite(bundle) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
        if step in audit_steps[1:]:
            parameter_gradient_rows_all.extend({"step": step, **row} for row in parameter_gradient_rows(model.osmf_28_1, PARAMETER_NAMES))
        optimizer.step()
        processed = step
        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        loss_rows.append({"step": step, "loss_total": float(bundle["total"].detach().cpu()), "loss_sshr": float(bundle["base"].detach().cpu()), "loss_sem_pres": float(bundle["sem_pres"].detach().cpu()), "loss_struct": None if bundle["struct"] is None else float(bundle["struct"].detach().cpu()), "loss_orth": float(bundle["orth"].detach().cpu()), "loss_rec": float(bundle["rec"].detach().cpu()), "structural_scheduled": bundle["scheduled_struct"], "finite": _finite(bundle)})
        if step in audit_steps[1:]:
            parameter_update_rows_all.extend({"step": step, **row} for row in parameter_update_rows(model.osmf_28_1, PARAMETER_NAMES, initial, before_parameters))
            rep, _ = _diagnostic(model, images, labels, step)
            representation_rows.append(rep)
            finite = finite and rep["finite"]
        if step in active_steps:
            after_pair = _eval_pair(model, images, flip_dimension)
            delta = after_pair["struct_morphology"] - before_pair["struct_morphology"]
            causal_rows.append({"step": step, "flip_dimension": flip_dimension, "struct_before": before_pair["struct_morphology"], "struct_after": after_pair["struct_morphology"], "delta": delta, "affinity_before": before_pair["affinity_morphology"], "affinity_after": after_pair["affinity_morphology"], "semantic_affinity_before": before_pair["affinity_semantic"], "semantic_affinity_after": after_pair["affinity_semantic"], "input_batch_sha256": hashlib.sha256("".join(_tensor_sha256(image) for image in images).encode()).hexdigest()})
        if probe_records and step in FIXED_PROBE_STEPS:
            fixed_rows.append(_fixed_probe_measure(model, probe_records, step))
        if step in audit_steps or step in active_steps:
            print("[OSMF-v1.3] " + json.dumps({"gate": args.gate, "step": step, "causal_delta": None if step not in active_steps else causal_rows[-1]["delta"]}, sort_keys=True), flush=True)
        if not finite:
            break

    parameter_summary = _parameter_summary(parameter_gradient_rows_all, parameter_update_rows_all)
    morph_active = all(any(float(row[key]) > 1e-12 for row in morph_gradient_rows if row["step"] in (4, 8)) for key in ("p_morph_grad_norm", "u_morph_grad_norm"))
    sshr_values = [float(row["loss_sshr"]) for row in loss_rows]
    sshr_stable = all(math.isfinite(value) for value in sshr_values) and max(sshr_values) < 2.0 * max(sshr_values[0], 1e-6)
    cross_start, cross_end = representation_rows[0]["cross_covariance"], representation_rows[-1]["cross_covariance"]
    cross_healthy = math.isfinite(cross_end) and cross_end <= max(2.0 * cross_start, cross_start + 0.01)
    if args.gate == "readiness":
        decision, reasons, ratio_stats, causal = readiness_decision(finite=finite, ratio_rows=ratio_rows, representation_rows=representation_rows, parameter_summary=parameter_summary, morph_struct_active=morph_active, causal_rows=causal_rows, sshr_loss_stable=sshr_stable)
        fixed_summary = None
    else:
        decision, reasons, ratio_stats, causal, evidence = phase0s_decision(finite=finite, ratio_rows=ratio_rows, representation_rows=representation_rows, parameter_summary=parameter_summary, morph_struct_active=morph_active, causal_rows=causal_rows, fixed_rows=fixed_rows, sshr_loss_stable=sshr_stable, cross_covariance_healthy=cross_healthy)
        fixed_summary = {"images": PROBE_IMAGES, "affinity_morphology_start": fixed_rows[0]["affinity_eq_error_morphology"], "affinity_morphology_end": fixed_rows[-1]["affinity_eq_error_morphology"], "affinity_semantic_start": fixed_rows[0]["affinity_eq_error_semantic"], "affinity_semantic_end": fixed_rows[-1]["affinity_eq_error_semantic"], "raw_morphology_start": fixed_rows[0]["eq_error_morphology"], "raw_morphology_end": fixed_rows[-1]["eq_error_morphology"], **evidence}

    for name, rows in (("loss_trace.csv", loss_rows), ("gradient_ratios.csv", ratio_rows), ("gradient_cosines.csv", cosine_rows), ("morphology_structural_gradients.csv", morph_gradient_rows), ("same_pair_causal.csv", causal_rows), ("representation_health.csv", representation_rows), ("parameter_gradients.csv", parameter_gradient_rows_all), ("parameter_updates.csv", parameter_update_rows_all), ("fixed_probe.csv", fixed_rows)):
        write_csv(args.output_dir / "tables" / name, rows)
    summary = {
        "gate": args.gate, "decision": decision, "decision_reasons": reasons,
        "processed_batches": processed, "authorized_batches": authorized,
        "audit_commit": args.audit_commit, "baseline_commit": BASELINE_COMMIT,
        "checkpoint_sha256": checkpoint_sha, "parity_summary_sha256": parity_sha,
        "readiness_summary_sha256": readiness_sha, "exact_command": contract["exact_command"],
        "same_pair_causal": causal, "gradient_budget": ratio_stats,
        "morphology_structural_path_active": morph_active, "parameter_health": parameter_summary,
        "fixed_probe": fixed_summary,
        "representation": {"semantic_agreement_start": representation_rows[0]["semantic_agreement"], "semantic_agreement_end": representation_rows[-1]["semantic_agreement"], "reconstruction_cosine_start": representation_rows[0]["reconstruction_cosine"], "reconstruction_cosine_end": representation_rows[-1]["reconstruction_cosine"], "cross_covariance_start": cross_start, "cross_covariance_end": cross_end, "cross_covariance_healthy": cross_healthy},
        "finite": finite, "sshr_loss_stable": sshr_stable,
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0)},
        "cost": {"elapsed_seconds": time.perf_counter() - started, "peak_memory_allocated_bytes": peak_memory},
        "checkpoint_saved": False, "validation_evaluated": False, "test_evaluated": False,
        "luad_evaluated": False, "segmentation_gt_used": False, "full_training_started": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
