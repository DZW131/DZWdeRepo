"""Run frozen OSMF-v1.2 8-batch readiness or gated 128-batch Phase 0."""

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
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.osmf_v12 import (
    OSMF_EQUIVARIANCE_INTERVAL,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_preservation_agreement,
    semantic_preservation_loss,
    spatial_equivariance_loss,
)
from network.resnet38_cls_osmf_v12 import Net
from tool.GenDataset import Stage1_TrainDataset
from tools.audit_osmf_phase0_128batch import (
    BASELINE_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    _build_optimizer,
    _flip_dimension,
    sha256_file,
)
from tools.osmf_phase0_audit.gradients import (
    gradient_decomposition,
    max_consecutive,
    parameter_gradient_rows,
    parameter_update_rows,
    snapshot_parameters,
)
from tools.osmf_v12_audit import (
    BATCH_SIZE,
    IMAGE_SIZE,
    OBJECTIVE_NAMES,
    OBJECTIVE_WEIGHTS,
    PARAMETER_NAMES,
    PHASE0_AUDIT_STEPS,
    PHASE0_BATCHES,
    READINESS_AUDIT_STEPS,
    READINESS_BATCHES,
    SEED,
)
from tools.osmf_v12_audit.decision import (
    percentile,
    phase0_decision,
    readiness_decision,
)
from tools.osmf_v12_audit.report import (
    build_main_table,
    make_figures,
    write_csv,
    write_report,
)
from train_osmf_v12 import sshr_classification_loss
from train_sshr import seed_worker, set_seed


EXPECTED_MISSING_KEYS = {
    "osmf_28_1.p_sem.weight",
    "osmf_28_1.p_morph.weight",
    "osmf_28_1.u_sem.weight",
    "osmf_28_1.u_morph.weight",
}


def _load_checkpoint(model: Net, checkpoint: Path):
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    if set(incompatible.missing_keys) != EXPECTED_MISSING_KEYS:
        raise AssertionError(
            f"Unexpected v1.2 missing keys: {sorted(incompatible.missing_keys)}"
        )
    if incompatible.unexpected_keys:
        raise AssertionError(
            f"Unexpected checkpoint keys: {sorted(incompatible.unexpected_keys)}"
        )
    for key, value in state.items():
        if not torch.equal(value.cpu(), model.state_dict()[key].cpu()):
            raise AssertionError(f"Frozen checkpoint key changed: {key}")
    return sorted(incompatible.missing_keys), sorted(incompatible.unexpected_keys)


def _sha256_json_proof(path: Path, expected_decision: str, commit: str) -> str:
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("decision") != expected_decision:
        raise RuntimeError(
            f"Required proof {path} is {proof.get('decision')}, expected {expected_decision}"
        )
    proof_commit = proof.get("osmf_v12_commit", proof.get("audit_commit"))
    if proof_commit != commit:
        raise RuntimeError(
            f"Proof commit {proof_commit} does not match audit commit {commit}"
        )
    return sha256_file(path)


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().cpu())


def _forward_objectives(
    model: Net,
    images: torch.Tensor,
    labels: torch.Tensor,
    step: int,
    force_equivariance: bool,
):
    outputs, aux = model.forward_with_aux(images)
    base = sshr_classification_loss(outputs, labels)
    sem_pres = semantic_preservation_loss(
        aux["semantic_student_response"], aux["semantic_teacher_response"]
    )
    orth = orthogonality_loss(aux["semantic"], aux["morphology"])
    rec = reconstruction_loss(aux["reconstruction"], aux["input"])
    scheduled_eq = step > 0 and step % OSMF_EQUIVARIANCE_INTERVAL == 0
    eq = None
    eq_semantic = None
    if force_equivariance or scheduled_eq:
        flip_dimension = _flip_dimension(step)
        second = model.forward_osmf_features(
            torch.flip(images, dims=(flip_dimension,))
        )
        eq = spatial_equivariance_loss(
            aux["morphology"],
            inverse_align_morphology(second["morphology"], flip_dimension),
        )
        eq_semantic = spatial_equivariance_loss(
            aux["semantic"],
            inverse_align_morphology(second["semantic"], flip_dimension),
        )
    eq_for_total = eq if scheduled_eq else base.new_zeros(())
    total = (
        base
        + OBJECTIVE_WEIGHTS["sem_pres"] * sem_pres
        + OBJECTIVE_WEIGHTS["eq"] * eq_for_total
        + OBJECTIVE_WEIGHTS["orth"] * orth
        + OBJECTIVE_WEIGHTS["rec"] * rec
    )
    return {
        "outputs": outputs,
        "aux": aux,
        "base": base,
        "sem_pres": sem_pres,
        "eq": eq,
        "eq_semantic": eq_semantic,
        "orth": orth,
        "rec": rec,
        "total": total,
        "scheduled_eq": scheduled_eq,
    }


def _all_finite(bundle) -> bool:
    aux = bundle["aux"]
    tensors = list(bundle["outputs"]) + [
        aux["input"],
        aux["semantic"],
        aux["morphology"],
        aux["semantic_reconstruction"],
        aux["morphology_reconstruction"],
        aux["reconstruction"],
        aux["semantic_teacher_response"],
        aux["semantic_student_response"],
        bundle["base"],
        bundle["sem_pres"],
        bundle["orth"],
        bundle["rec"],
        bundle["total"],
    ]
    if bundle["eq"] is not None:
        tensors.extend((bundle["eq"], bundle["eq_semantic"]))
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def _diagnostic_state(model, images, labels, step):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            bundle = _forward_objectives(
                model, images, labels, step=step, force_equivariance=True
            )
    aux = bundle["aux"]
    h_rms = _rms(aux["input"])
    reconstruction_rms = _rms(aux["reconstruction"])
    student_rms = _rms(aux["semantic_student_response"])
    teacher_rms = _rms(aux["semantic_teacher_response"])
    representation = {
        "step": step,
        "h_rms": h_rms,
        "semantic_rms": _rms(aux["semantic"]),
        "morphology_rms": _rms(aux["morphology"]),
        "reconstruction_rms": reconstruction_rms,
        "semantic_morphology_rms_ratio": _rms(aux["semantic"])
        / (_rms(aux["morphology"]) + 1e-12),
        "reconstruction_cosine": float(
            reconstruction_cosine(aux["reconstruction"], aux["input"]).cpu()
        ),
        "residual_ratio": _rms(aux["reconstruction"] - aux["input"])
        / (h_rms + 1e-12),
        "cross_covariance": float(bundle["orth"].cpu()),
        "eq_error_morphology": float(bundle["eq"].cpu()),
        "eq_error_semantic": float(bundle["eq_semantic"].cpu()),
        "semantic_student_response_rms": student_rms,
        "semantic_teacher_response_rms": teacher_rms,
        "semantic_response_rms_ratio": student_rms / (teacher_rms + 1e-12),
        "semantic_agreement": float(
            semantic_preservation_agreement(
                aux["semantic_student_response"],
                aux["semantic_teacher_response"],
            ).cpu()
        ),
        "finite": _all_finite(bundle),
    }
    loss = {
        "step": step,
        "state": "start" if step == 0 else "post_update_diagnostic",
        "loss_total": float(bundle["total"].cpu()),
        "loss_sshr": float(bundle["base"].cpu()),
        "loss_sem_pres": float(bundle["sem_pres"].cpu()),
        "loss_eq": float(bundle["eq"].cpu()),
        "loss_orth": float(bundle["orth"].cpu()),
        "loss_rec": float(bundle["rec"].cpu()),
        "equivariance_scheduled": bool(bundle["scheduled_eq"]),
        "finite": _all_finite(bundle),
    }
    max_identity_error = float(
        (aux["reconstruction"] - aux["input"]).abs().max().float().cpu()
    )
    return loss, representation, max_identity_error


def _training_loss_row(bundle, step):
    return {
        "step": step,
        "state": "training_pre_update",
        "loss_total": float(bundle["total"].detach().cpu()),
        "loss_sshr": float(bundle["base"].detach().cpu()),
        "loss_sem_pres": float(bundle["sem_pres"].detach().cpu()),
        "loss_eq": None
        if bundle["eq"] is None
        else float(bundle["eq"].detach().cpu()),
        "loss_orth": float(bundle["orth"].detach().cpu()),
        "loss_rec": float(bundle["rec"].detach().cpu()),
        "equivariance_scheduled": bool(bundle["scheduled_eq"]),
        "finite": _all_finite(bundle),
    }


def _parameter_summary(gradient_rows, update_rows):
    summary = {}
    for name in PARAMETER_NAMES:
        gradients = [row for row in gradient_rows if row["parameter"] == name]
        updates = [row for row in update_rows if row["parameter"] == name]
        end = updates[-1]
        summary[name] = {
            "grad_nonzero": any(
                float(row["grad_norm"]) > 1e-12
                and float(row["nonzero_grad_fraction"]) > 0.0
                for row in gradients
            ),
            "mean_grad_norm": sum(float(row["grad_norm"]) for row in gradients)
            / len(gradients),
            "end_update_norm": float(end["cumulative_update_norm"]),
            "end_relative_update": float(end["relative_update_norm"]),
            "measurable_update": float(end["cumulative_update_norm"]) > 1e-12,
        }
    return summary


def _hard_stop_reason(gate, ratio_rows, representation_rows, finite):
    if not finite:
        return "NONFINITE_TENSOR_LOSS_OR_GRADIENT"
    objectives = ("sem_pres", "eq") if gate == "readiness" else OBJECTIVE_NAMES
    for objective in objectives:
        values = [
            float(row["ratio"])
            for row in ratio_rows
            if row["objective"] == objective
        ]
        if max_consecutive([value > 0.50 for value in values]) >= 2:
            return f"PERSISTENT_{objective.upper()}_RATIO_GT_0_50"
    branch = [
        float(row["semantic_morphology_rms_ratio"])
        for row in representation_rows
    ]
    response = [
        float(row["semantic_response_rms_ratio"]) for row in representation_rows
    ]
    if max_consecutive([value <= 0.05 or value >= 20 for value in branch]) >= 2:
        return "BRANCH_COLLAPSE"
    if max_consecutive([value < 0.05 for value in response]) >= 2:
        return "SEMANTIC_RESPONSE_COLLAPSE"
    if float(representation_rows[-1]["reconstruction_cosine"]) < 0.90:
        return "RECONSTRUCTION_DESTABILIZED"
    return None


def _source_hashes():
    names = (
        "network/osmf.py",
        "network/osmf_v12.py",
        "network/resnet38_cls.py",
        "network/resnet38_cls_osmf_v12.py",
        "train_osmf_v12.py",
        "tools/audit_osmf_v12_gradient_gate.py",
        "tools/osmf_v12_audit/__init__.py",
        "tools/osmf_v12_audit/decision.py",
        "tools/osmf_v12_audit/report.py",
        "docs/specs/osmf_v12_gradient_budget_preregistered_contract.md",
        "train_sshr.py",
        "tool/GenDataset.py",
        "tool/torchutils.py",
    )
    return {name: sha256_file(REPO_ROOT / name) for name in names}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True, choices=("readiness", "phase0"))
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parity-summary", required=True, type=Path)
    parser.add_argument("--readiness-summary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16",))
    return parser.parse_args()


def main():
    args = parse_args()
    if any(token in str(args.train_root).lower() for token in ("test", "luad", "val")):
        raise ValueError("Gradient gates accept BCSS training data only")
    if len(args.audit_commit) != 40:
        raise ValueError("--audit-commit must be a full Git SHA")
    authorized_batches = READINESS_BATCHES if args.gate == "readiness" else PHASE0_BATCHES
    audit_steps = (
        READINESS_AUDIT_STEPS if args.gate == "readiness" else PHASE0_AUDIT_STEPS
    )
    gradient_steps = audit_steps[1:]
    parity_sha = _sha256_json_proof(
        args.parity_summary, "OSMF_V12_PARITY_PASS", args.audit_commit
    )
    readiness_sha = None
    if args.gate == "phase0":
        if args.readiness_summary is None:
            raise ValueError("Phase 0 requires --readiness-summary")
        readiness_sha = _sha256_json_proof(
            args.readiness_summary,
            "OSMF_V12_READINESS_PASS",
            args.audit_commit,
        )
    elif args.readiness_summary is not None:
        raise ValueError("Readiness gate does not accept a readiness proof")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config", "tables", "figures", "docs"):
        (args.output_dir / name).mkdir()
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f"Unexpected checkpoint SHA256: {checkpoint_sha}")

    set_seed(SEED)
    model = Net(n_class=4).cuda()
    dataset = Stage1_TrainDataset(
        data_path=str(args.train_root),
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset="bcss",
        img_size=IMAGE_SIZE,
    )
    if len(dataset) != 23422:
        raise AssertionError(f"Frozen BCSS train count must be 23422, got {len(dataset)}")
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    optimizer, max_step = _build_optimizer(model, len(dataset))
    missing, unexpected = _load_checkpoint(model, args.checkpoint)
    model.train()
    optimizer_contract = {
        "class": type(optimizer).__name__,
        "max_step": max_step,
        "lr_power": optimizer.lr_power,
        "momentum": float(optimizer.param_groups[0]["momentum"]),
        "parameter_groups": [
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "momentum": float(group["momentum"]),
                "parameter_tensors": len(group["params"]),
            }
            for group in optimizer.param_groups
        ],
    }
    contract = {
        "gate": args.gate,
        "authorized_batches": authorized_batches,
        "audit_steps": audit_steps,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "precision": "bf16",
        "objective_weights": OBJECTIVE_WEIGHTS,
        "equivariance_interval": OSMF_EQUIVARIANCE_INTERVAL,
        "dataset_size": len(dataset),
        "baseline_commit": BASELINE_COMMIT,
        "audit_commit": args.audit_commit,
        "checkpoint_sha256": checkpoint_sha,
        "parity_summary_sha256": parity_sha,
        "readiness_summary_sha256": readiness_sha,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "optimizer": optimizer_contract,
        "source_sha256": _source_hashes(),
        "exact_command": " ".join(sys.argv),
        "validation_evaluated": False,
        "test_evaluated": False,
        "luad_evaluated": False,
        "segmentation_gt_used": False,
    }
    (args.output_dir / "config" / "frozen_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    iterator = iter(loader)
    _, first_images, first_labels = next(iterator)
    first_images = first_images.cuda(non_blocking=True)
    first_labels = first_labels.cuda(non_blocking=True)
    start_loss, start_representation, start_error = _diagnostic_state(
        model, first_images, first_labels, 0
    )
    if start_error >= 1e-6 or not start_loss["finite"]:
        raise RuntimeError(
            f"OSMF_V12_PARITY_NOGO start-state identity/finite failure: {start_error}"
        )

    loss_rows = [start_loss]
    ratio_rows, cosine_rows = [], []
    parameter_gradient_rows_all, parameter_update_rows_all = [], []
    representation_rows = [start_representation]
    cost_rows = []
    initial_parameters = snapshot_parameters(model.osmf_28_1, PARAMETER_NAMES)
    processed_batches = 0
    finite = True
    hard_stop = None
    ic1_aux_gradient_free = True
    ic1_base_gradient_active = False
    peak_memory = 0
    started = time.perf_counter()
    current = (None, first_images, first_labels)

    for step in range(1, authorized_batches + 1):
        if step > 1:
            current = next(iterator)
            _, images, labels = current
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        else:
            _, images, labels = current

        if step in gradient_steps:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    audit_bundle = _forward_objectives(
                        model, images, labels, step, force_equivariance=True
                    )
                ratios, cosines = gradient_decomposition(
                    {
                        "base": audit_bundle["base"],
                        "sem_pres": audit_bundle["sem_pres"],
                        "eq": audit_bundle["eq"],
                        "orth": audit_bundle["orth"],
                        "rec": audit_bundle["rec"],
                    },
                    audit_bundle["aux"]["input"],
                    tuple(model.osmf_28_1.parameters()),
                    OBJECTIVE_WEIGHTS,
                    objective_names=OBJECTIVE_NAMES,
                )
                ic1_aux_grads = torch.autograd.grad(
                    audit_bundle["sem_pres"],
                    (model.ic1.weight, model.ic1.bias),
                    retain_graph=False,
                    allow_unused=True,
                )
            ic1_aux_gradient_free = ic1_aux_gradient_free and all(
                gradient is None or bool(torch.count_nonzero(gradient) == 0)
                for gradient in ic1_aux_grads
            )
            finite = finite and all(row["finite"] for row in ratios + cosines)
            for row in ratios:
                row["step"] = step
                ratio_rows.append(row)
            for row in cosines:
                row["step"] = step
                cosine_rows.append(row)
            # Do not retain the diagnostic graph while profiling the separate
            # official optimizer step. Rows above contain scalar copies only.
            del audit_bundle, ratios, cosines, ic1_aux_grads

        optimizer.zero_grad()
        before_step = (
            snapshot_parameters(model.osmf_28_1, PARAMETER_NAMES)
            if step in gradient_steps
            else None
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        iteration_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bundle = _forward_objectives(
                model, images, labels, step, force_equivariance=False
            )
        bundle["total"].backward()
        if model.ic1.weight.grad is not None:
            ic1_base_gradient_active = ic1_base_gradient_active or bool(
                torch.linalg.vector_norm(model.ic1.weight.grad.detach().float()) > 1e-12
            )
        finite = finite and all(
            parameter.grad is None
            or bool(torch.isfinite(parameter.grad.detach()).all())
            for parameter in model.parameters()
        )
        if step in gradient_steps:
            gradients = parameter_gradient_rows(model.osmf_28_1, PARAMETER_NAMES)
            finite = finite and all(row["finite"] for row in gradients)
            for row in gradients:
                row["step"] = step
                parameter_gradient_rows_all.append(row)
        optimizer.step()
        torch.cuda.synchronize()
        iteration_seconds = time.perf_counter() - iteration_started
        step_peak = int(torch.cuda.max_memory_allocated())
        peak_memory = max(peak_memory, step_peak)
        processed_batches = step
        training_loss = _training_loss_row(bundle, step)
        loss_rows.append(training_loss)
        finite = finite and training_loss["finite"] and ic1_aux_gradient_free
        cost_rows.append(
            {
                "step": step,
                "equivariance_scheduled": bool(bundle["scheduled_eq"]),
                "iteration_seconds": iteration_seconds,
                "peak_memory_allocated_bytes": step_peak,
                "lr_group0": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if step in gradient_steps:
            updates = parameter_update_rows(
                model.osmf_28_1,
                PARAMETER_NAMES,
                initial_parameters,
                before_step,
            )
            for row in updates:
                row["step"] = step
                parameter_update_rows_all.append(row)
            _, representation, _ = _diagnostic_state(
                model, images, labels, step
            )
            representation_rows.append(representation)
            finite = finite and representation["finite"]
            hard_stop = _hard_stop_reason(
                args.gate, ratio_rows, representation_rows, finite
            )
            print(
                "[Audit] "
                + json.dumps(
                    {
                        "gate": args.gate,
                        "step": step,
                        "r_sem_pres": next(
                            row["ratio"]
                            for row in reversed(ratio_rows)
                            if row["objective"] == "sem_pres"
                        ),
                        "semantic_agreement": representation["semantic_agreement"],
                        "semantic_response_rms_ratio": representation[
                            "semantic_response_rms_ratio"
                        ],
                        "reconstruction_cosine": representation[
                            "reconstruction_cosine"
                        ],
                        "hard_stop": hard_stop,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if hard_stop:
                break

    elapsed = time.perf_counter() - started
    parameter_summary = _parameter_summary(
        parameter_gradient_rows_all, parameter_update_rows_all
    )
    morphology_eq_gradient_active = any(
        row["objective"] == "eq"
        and float(row["objective_grad_norm_osmf_parameters"]) > 1e-12
        for row in ratio_rows
    )
    eq_values = [float(row["eq_error_morphology"]) for row in representation_rows]
    eq_responsive = max(eq_values) - min(eq_values) > 1e-6
    sshr_values = [float(row["loss_sshr"]) for row in loss_rows]
    sshr_loss_stable = all(math.isfinite(value) for value in sshr_values) and max(
        sshr_values
    ) < max(10.0, 10.0 * sshr_values[0])
    cross_values = [float(row["cross_covariance"]) for row in representation_rows]
    cross_covariance_healthy = all(math.isfinite(value) for value in cross_values) and max(
        cross_values
    ) < max(1.0, 10.0 * cross_values[0])

    if args.gate == "readiness":
        decision, flags, decision_reasons = readiness_decision(
            finite=finite,
            gradient_ratio_rows=ratio_rows,
            gradient_cosine_rows=cosine_rows,
            representation_rows=representation_rows,
            parameter_summary=parameter_summary,
            morphology_eq_gradient_active=morphology_eq_gradient_active,
            sshr_loss_stable=sshr_loss_stable,
        )
    else:
        decision, flags, decision_reasons = phase0_decision(
            finite=finite,
            gradient_ratio_rows=ratio_rows,
            gradient_cosine_rows=cosine_rows,
            representation_rows=representation_rows,
            parameter_summary=parameter_summary,
            morphology_eq_gradient_active=morphology_eq_gradient_active,
            eq_responsive=eq_responsive,
            sshr_loss_stable=sshr_loss_stable,
            cross_covariance_healthy=cross_covariance_healthy,
        )
    if hard_stop and hard_stop not in decision_reasons:
        decision_reasons = sorted(set(decision_reasons + [hard_stop]))

    write_csv(args.output_dir / "tables" / "loss_trace.csv", loss_rows)
    write_csv(args.output_dir / "tables" / "gradient_ratio.csv", ratio_rows)
    write_csv(args.output_dir / "tables" / "gradient_cosine.csv", cosine_rows)
    write_csv(
        args.output_dir / "tables" / "parameter_gradient_coverage.csv",
        parameter_gradient_rows_all,
    )
    write_csv(
        args.output_dir / "tables" / "parameter_update.csv",
        parameter_update_rows_all,
    )
    write_csv(
        args.output_dir / "tables" / "representation_health.csv",
        representation_rows,
    )
    write_csv(
        args.output_dir / "tables" / "semantic_response.csv",
        [
            {
                "step": row["step"],
                "semantic_student_response_rms": row[
                    "semantic_student_response_rms"
                ],
                "semantic_teacher_response_rms": row[
                    "semantic_teacher_response_rms"
                ],
                "semantic_response_rms_ratio": row[
                    "semantic_response_rms_ratio"
                ],
                "semantic_agreement": row["semantic_agreement"],
            }
            for row in representation_rows
        ],
    )
    write_csv(args.output_dir / "tables" / "compute_cost.csv", cost_rows)
    make_figures(
        args.output_dir,
        loss_rows,
        ratio_rows,
        cosine_rows,
        representation_rows,
        parameter_update_rows_all,
    )
    main_table = build_main_table(
        loss_rows, ratio_rows, cosine_rows, representation_rows
    )
    write_csv(args.output_dir / "tables" / "main_summary.csv", main_table)
    write_csv(
        args.output_dir / "tables" / "parameter_health_summary.csv",
        [
            {"parameter": name, **health}
            for name, health in parameter_summary.items()
        ],
    )

    sem_ratios = [
        float(row["ratio"]) for row in ratio_rows if row["objective"] == "sem_pres"
    ]
    gradient_budget = {}
    for objective in OBJECTIVE_NAMES:
        values = [
            float(row["ratio"])
            for row in ratio_rows
            if row["objective"] == objective
        ]
        gradient_budget[objective] = {
            "mean": sum(values) / len(values),
            "max": max(values),
            "p95": percentile(values, 0.95),
            "values": values,
        }
    response_ratios = [
        float(row["semantic_response_rms_ratio"]) for row in representation_rows
    ]
    timing_rows = [row for row in cost_rows if int(row["step"]) > 4] or cost_rows
    non_eq = [
        float(row["iteration_seconds"])
        for row in timing_rows
        if not row["equivariance_scheduled"]
    ]
    eq = [
        float(row["iteration_seconds"])
        for row in timing_rows
        if row["equivariance_scheduled"]
    ]
    mean_iteration = sum(float(row["iteration_seconds"]) for row in timing_rows) / len(
        timing_rows
    )
    mean_non_eq = sum(non_eq) / len(non_eq) if non_eq else mean_iteration
    mean_eq = sum(eq) / len(eq) if eq else None
    summary = {
        "gate": args.gate,
        "decision": decision,
        "flags": flags,
        "decision_reasons": decision_reasons,
        "hard_stop_reason": hard_stop,
        "processed_batches": processed_batches,
        "authorized_batches": authorized_batches,
        "audit_steps_completed": sorted(
            {int(row["step"]) for row in representation_rows}
            | {int(row["step"]) for row in ratio_rows}
        ),
        "audit_commit": args.audit_commit,
        "baseline_commit": BASELINE_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "parity_summary_sha256": parity_sha,
        "readiness_summary_sha256": readiness_sha,
        "exact_command": contract["exact_command"],
        "optimizer": optimizer_contract,
        "start_state": {
            "max_identity_error": start_error,
            "reconstruction_cosine": start_representation[
                "reconstruction_cosine"
            ],
            "finite": start_loss["finite"],
        },
        "semantic": {
            "v10_ratio_min": 2.480527,
            "v10_ratio_max": 4.106567,
            "ratio_min": min(sem_ratios),
            "ratio_max": max(sem_ratios),
            "ratio_mean": sum(sem_ratios) / len(sem_ratios),
            "agreement_start": representation_rows[0]["semantic_agreement"],
            "agreement_end": representation_rows[-1]["semantic_agreement"],
            "response_ratio_start": response_ratios[0],
            "response_ratio_end": response_ratios[-1],
            "response_non_degenerate": max_consecutive(
                [value < 0.05 for value in response_ratios]
            )
            < 2,
            "semantic_parameters_active": all(
                parameter_summary[name]["grad_nonzero"]
                and parameter_summary[name]["measurable_update"]
                for name in ("p_sem.weight", "u_sem.weight")
            ),
        },
        "gradient_budget": gradient_budget,
        "mechanism": {
            "morphology_eq_gradient_active": morphology_eq_gradient_active,
            "eq_responsive": eq_responsive,
            "eq_error_morphology_start": eq_values[0],
            "eq_error_morphology_end": eq_values[-1],
            "reconstruction_cosine_end": representation_rows[-1][
                "reconstruction_cosine"
            ],
            "semantic_morphology_rms_ratio_end": representation_rows[-1][
                "semantic_morphology_rms_ratio"
            ],
            "cross_covariance_start": cross_values[0],
            "cross_covariance_end": cross_values[-1],
            "sshr_loss_stable": sshr_loss_stable,
            "cross_covariance_healthy": cross_covariance_healthy,
        },
        "parameter_health": parameter_summary,
        "finite": finite,
        "ic1_aux_gradient_free": ic1_aux_gradient_free,
        "ic1_base_gradient_active": ic1_base_gradient_active,
        "cost": {
            "elapsed_seconds": elapsed,
            "mean_iteration_seconds": mean_iteration,
            "mean_non_equivariance_seconds": mean_non_eq,
            "mean_equivariance_seconds": mean_eq,
            "non_equivariance_samples": len(non_eq),
            "equivariance_samples": len(eq),
            "peak_memory_allocated_bytes": peak_memory,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "next_gate_authorized": (
            "phase0_128b"
            if decision == "OSMF_V12_READINESS_PASS"
            else "none"
        ),
        "fresh_run_not_continuation": True,
        "checkpoint_saved": False,
        "formal_training_performed": False,
        "validation_evaluated": False,
        "test_evaluated": False,
        "luad_evaluated": False,
        "segmentation_gt_used": False,
        "phase1_started": False,
    }
    write_report(args.output_dir, summary, main_table, parameter_summary)
    (args.output_dir / "config" / "environment.json").write_text(
        json.dumps(summary["environment"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
