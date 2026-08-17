"""Run the frozen OSMF-v1.0 Phase-0 128-real-batch mechanism audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.osmf import (
    OSMF_EQUIVARIANCE_INTERVAL,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_classification_loss,
    spatial_equivariance_loss,
)
from network.resnet38_cls_osmf import Net
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from train_sshr import get_amp_dtype, seed_worker, set_seed
from tools.osmf_phase0_audit import (
    AUDIT_STEPS,
    BATCH_SIZE,
    FORMAL_EPOCHS_FOR_POLY_SCHEDULE,
    GRADIENT_STEPS,
    IMAGE_SIZE,
    NUM_REAL_BATCHES,
    OBJECTIVE_WEIGHTS,
    PARAMETER_NAMES,
    SEED,
)
from tools.osmf_phase0_audit.decision import decide_phase0
from tools.osmf_phase0_audit.gradients import (
    gradient_decomposition,
    max_consecutive,
    parameter_gradient_rows,
    parameter_update_rows,
    snapshot_parameters,
)
from tools.osmf_phase0_audit.report import (
    build_main_table,
    make_figures,
    stats,
    write_csv,
    write_report,
)


BASELINE_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
PHASE0_PARENT_COMMIT = "5eb7b258f0cdeb4fa8779b65e716c105c9541f9a"
EXPECTED_CHECKPOINT_SHA256 = (
    "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
)
EXPECTED_MISSING_KEYS = {
    "osmf_28_1.p_sem.weight",
    "osmf_28_1.p_morph.weight",
    "osmf_28_1.u_sem.weight",
    "osmf_28_1.u_morph.weight",
    "osmf_28_1.semantic_classifier.weight",
    "osmf_28_1.semantic_classifier.bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    paths = (
        "network/resnet38_cls.py",
        "network/resnet38d.py",
        "tool/GenDataset.py",
        "tool/infer_fun.py",
        "tool/iouutils.py",
        "tool/torchutils.py",
        "train_sshr.py",
        "network/osmf.py",
        "network/resnet38_cls_osmf.py",
        "train_osmf.py",
    )
    return {name: sha256_file(REPO_ROOT / name) for name in paths}


def _load_checkpoint(model: Net, checkpoint: Path) -> tuple[list[str], list[str]]:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != EXPECTED_MISSING_KEYS:
        raise AssertionError(f"Unexpected missing keys: {sorted(missing)}")
    if unexpected:
        raise AssertionError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    for key, value in state.items():
        if not torch.equal(value.cpu(), model.state_dict()[key].cpu()):
            raise AssertionError(f"Frozen checkpoint key changed during load: {key}")
    return sorted(missing), sorted(unexpected)


def _build_optimizer(model: Net, dataset_size: int):
    max_step = (
        dataset_size // BATCH_SIZE
    ) * FORMAL_EPOCHS_FOR_POLY_SCHEDULE
    groups = model.get_parameter_groups()
    parameters = [
        {"params": groups[0], "lr": 0.01, "weight_decay": 5e-4},
        {"params": groups[1], "lr": 0.02, "weight_decay": 0.0},
        {"params": groups[2], "lr": 0.10, "weight_decay": 5e-4},
        {"params": groups[3], "lr": 0.20, "weight_decay": 0.0},
    ]
    optimizer = torchutils.PolyOptimizer(
        parameters,
        lr=0.01,
        weight_decay=5e-4,
        max_step=max_step,
    )
    return optimizer, max_step


def _flip_dimension(step: int) -> int:
    block = max(1, (max(step, 1) + OSMF_EQUIVARIANCE_INTERVAL - 1) // OSMF_EQUIVARIANCE_INTERVAL)
    return 3 if block % 2 == 1 else 2


def _sshr_loss(outputs, labels):
    out_56, out_28_1, out_28_2, out_deep = outputs[:4]
    return (
        0.10 * F.multilabel_soft_margin_loss(out_56, labels)
        + 0.15 * F.multilabel_soft_margin_loss(out_28_1, labels)
        + 0.25 * F.multilabel_soft_margin_loss(out_28_2, labels)
        + 0.50 * F.multilabel_soft_margin_loss(out_deep, labels)
    )


def forward_objectives(model, images, labels, step: int, force_equivariance: bool):
    outputs, aux = model.forward_with_aux(images)
    base = _sshr_loss(outputs, labels)
    sem = semantic_classification_loss(aux["semantic_logits"], labels)
    orth = orthogonality_loss(aux["semantic"], aux["morphology"])
    rec = reconstruction_loss(aux["reconstruction"], aux["input"])
    scheduled_eq = step > 0 and step % OSMF_EQUIVARIANCE_INTERVAL == 0
    aux_b = None
    eq = None
    eq_semantic = None
    if force_equivariance or scheduled_eq:
        flip_dimension = _flip_dimension(step)
        view_b = torch.flip(images, dims=(flip_dimension,))
        aux_b = model.forward_osmf_features(view_b)
        morphology_b = inverse_align_morphology(
            aux_b["morphology"], flip_dimension
        )
        semantic_b = inverse_align_morphology(aux_b["semantic"], flip_dimension)
        eq = spatial_equivariance_loss(aux["morphology"], morphology_b)
        eq_semantic = spatial_equivariance_loss(aux["semantic"], semantic_b)
    eq_for_total = eq if scheduled_eq else base.new_zeros(())
    total = (
        base
        + OBJECTIVE_WEIGHTS["sem"] * sem
        + OBJECTIVE_WEIGHTS["eq"] * eq_for_total
        + OBJECTIVE_WEIGHTS["orth"] * orth
        + OBJECTIVE_WEIGHTS["rec"] * rec
    )
    return {
        "outputs": outputs,
        "aux": aux,
        "aux_b": aux_b,
        "base": base,
        "sem": sem,
        "eq": eq,
        "eq_semantic": eq_semantic,
        "orth": orth,
        "rec": rec,
        "total": total,
        "scheduled_eq": scheduled_eq,
    }


def _rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt().cpu())


def _all_finite(bundle) -> bool:
    tensors = list(bundle["outputs"]) + [
        bundle["aux"]["input"],
        bundle["aux"]["semantic"],
        bundle["aux"]["morphology"],
        bundle["aux"]["reconstruction"],
        bundle["aux"]["semantic_logits"],
        bundle["base"],
        bundle["sem"],
        bundle["orth"],
        bundle["rec"],
        bundle["total"],
    ]
    if bundle["eq"] is not None:
        tensors.extend((bundle["eq"], bundle["eq_semantic"]))
    return all(bool(torch.isfinite(tensor).all()) for tensor in tensors)


def diagnostic_state(model, images, labels, step, amp_dtype):
    cuda_device = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_device]):
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            bundle = forward_objectives(
                model, images, labels, step=step, force_equivariance=True
            )
    aux = bundle["aux"]
    h_rms = _rms(aux["input"])
    semantic_rms = _rms(aux["semantic"])
    morphology_rms = _rms(aux["morphology"])
    reconstruction_rms = _rms(aux["reconstruction"])
    residual_rms = _rms(aux["reconstruction"] - aux["input"])
    logits = aux["semantic_logits"].detach().float()
    loss_row = {
        "step": step,
        "state": "start" if step == 0 else "post_update_diagnostic",
        "loss_total": float(bundle["total"].detach().cpu()),
        "loss_sshr": float(bundle["base"].detach().cpu()),
        "loss_sem": float(bundle["sem"].detach().cpu()),
        "loss_eq": float(bundle["eq"].detach().cpu()),
        "loss_orth": float(bundle["orth"].detach().cpu()),
        "loss_rec": float(bundle["rec"].detach().cpu()),
        "equivariance_scheduled": bool(bundle["scheduled_eq"]),
        "finite": _all_finite(bundle),
    }
    representation_row = {
        "step": step,
        "h_rms": h_rms,
        "semantic_rms": semantic_rms,
        "morphology_rms": morphology_rms,
        "reconstruction_rms": reconstruction_rms,
        "semantic_morphology_rms_ratio": semantic_rms / (morphology_rms + 1e-12),
        "reconstruction_cosine": float(
            reconstruction_cosine(aux["reconstruction"], aux["input"])
            .detach()
            .cpu()
        ),
        "residual_ratio": residual_rms / (h_rms + 1e-12),
        "cross_covariance": float(bundle["orth"].detach().cpu()),
        "eq_error_morphology": float(bundle["eq"].detach().cpu()),
        "eq_error_semantic": float(bundle["eq_semantic"].detach().cpu()),
        "semantic_logit_mean": float(logits.mean().cpu()),
        "semantic_logit_std": float(logits.std(unbiased=False).cpu()),
        "semantic_probability_mean": float(torch.sigmoid(logits).mean().cpu()),
        "finite": _all_finite(bundle),
    }
    feature_error = float(
        (aux["reconstruction"] - aux["input"]).detach().float().abs().max().cpu()
    )
    return loss_row, representation_row, feature_error


def _training_loss_row(bundle, step):
    return {
        "step": step,
        "state": "training_pre_update",
        "loss_total": float(bundle["total"].detach().cpu()),
        "loss_sshr": float(bundle["base"].detach().cpu()),
        "loss_sem": float(bundle["sem"].detach().cpu()),
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
        mean_gradient = sum(float(row["grad_norm"]) for row in gradients) / len(
            gradients
        )
        summary[name] = {
            "grad_nonzero": any(
                float(row["grad_norm"]) > 1e-12
                and float(row["nonzero_grad_fraction"]) > 0.0
                for row in gradients
            ),
            "mean_grad_norm": mean_gradient,
            "end_update_norm": float(end["cumulative_update_norm"]),
            "end_relative_update": float(end["relative_update_norm"]),
            "measurable_update": float(end["cumulative_update_norm"]) > 1e-12,
        }
    return summary


def _hard_stop_reason(ratio_rows, representation_rows, finite):
    if not finite:
        return "NONFINITE_TENSOR_LOSS_OR_GRADIENT"
    grouped = {
        objective: [
            float(row["ratio"])
            for row in ratio_rows
            if row["objective"] == objective
        ]
        for objective in ("sem", "eq", "orth", "rec")
    }
    for objective, values in grouped.items():
        if max_consecutive([value > 0.50 for value in values]) >= 2:
            return f"PERSISTENT_{objective.upper()}_GRADIENT_RATIO_GT_0_50"
    reconstruction = float(representation_rows[-1]["reconstruction_cosine"])
    if reconstruction < 0.90:
        return "RECONSTRUCTION_DESTABILIZED"
    ratios_sm = [
        float(row["semantic_morphology_rms_ratio"])
        for row in representation_rows
    ]
    if max_consecutive([value <= 0.05 or value >= 20.0 for value in ratios_sm]) >= 2:
        return "BRANCH_COLLAPSE"
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--amp-dtype", default="bf16", choices=("bf16",), help="Frozen protocol"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if any(token in str(args.train_root).lower() for token in ("test", "luad")):
        raise ValueError("Phase 0 accepts only the BCSS training path")
    if len(args.audit_commit) != 40:
        raise ValueError("--audit-commit must be a full Git SHA")
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
    missing_keys, unexpected_keys = _load_checkpoint(model, args.checkpoint)
    model.train()
    amp_dtype = get_amp_dtype(SimpleNamespace(amp_dtype=args.amp_dtype))

    optimizer_contract = {
        "class": type(optimizer).__name__,
        "fresh_state_from_a0_checkpoint": True,
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
    exact_command = " ".join(sys.argv)
    contract = {
        "scope": "OSMF-v1.0 Phase 0; exactly 128 real BCSS training batches",
        "baseline_commit": BASELINE_COMMIT,
        "phase0_parent_commit": PHASE0_PARENT_COMMIT,
        "audit_commit": args.audit_commit,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "train_root": str(args.train_root),
        "dataset_size": len(dataset),
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "num_real_batches": NUM_REAL_BATCHES,
        "audit_steps": AUDIT_STEPS,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "equivariance_interval": OSMF_EQUIVARIANCE_INTERVAL,
        "geometric_transforms": ["horizontal_flip", "vertical_flip"],
        "photometric_transform": "identity; frozen A0 has no photometric augmentation",
        "optimizer": optimizer_contract,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "source_sha256": source_hashes(),
        "exact_command": exact_command,
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
    start_loss, start_representation, start_feature_error = diagnostic_state(
        model, first_images, first_labels, step=0, amp_dtype=amp_dtype
    )
    if start_feature_error >= 1e-6 or not start_loss["finite"]:
        raise RuntimeError(
            f"OSMF_PHASE0_NOGO start-state parity/finite failure: {start_feature_error}"
        )

    loss_rows = [start_loss]
    gradient_ratio_rows = []
    gradient_cosine_rows = []
    parameter_gradient_coverage_rows = []
    parameter_update_rows_all = []
    representation_rows = [start_representation]
    compute_cost_rows = []
    initial_parameters = snapshot_parameters(model.osmf_28_1, PARAMETER_NAMES)
    finite = True
    processed_batches = 0
    hard_stop = None
    peak_training_memory = 0
    started = time.perf_counter()

    current = (None, first_images, first_labels)
    for step in range(1, NUM_REAL_BATCHES + 1):
        if step > 1:
            current = next(iterator)
            _, images, labels = current
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        else:
            _, images, labels = current

        if step in GRADIENT_STEPS:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=True
                ):
                    audit_bundle = forward_objectives(
                        model,
                        images,
                        labels,
                        step=step,
                        force_equivariance=True,
                    )
                ratios, cosines = gradient_decomposition(
                    {
                        "base": audit_bundle["base"],
                        "sem": audit_bundle["sem"],
                        "eq": audit_bundle["eq"],
                        "orth": audit_bundle["orth"],
                        "rec": audit_bundle["rec"],
                    },
                    audit_bundle["aux"]["input"],
                    tuple(model.osmf_28_1.parameters()),
                    OBJECTIVE_WEIGHTS,
                )
            for row in ratios:
                row["step"] = step
                gradient_ratio_rows.append(row)
            for row in cosines:
                row["step"] = step
                gradient_cosine_rows.append(row)
            finite = finite and _all_finite(audit_bundle) and all(
                row["finite"] for row in ratios + cosines
            )
            del audit_bundle

        optimizer.zero_grad()
        before_step = (
            snapshot_parameters(model.osmf_28_1, PARAMETER_NAMES)
            if step in GRADIENT_STEPS
            else None
        )
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        iteration_started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            bundle = forward_objectives(
                model,
                images,
                labels,
                step=step,
                force_equivariance=False,
            )
        bundle["total"].backward()
        if step in GRADIENT_STEPS:
            grad_rows = parameter_gradient_rows(model.osmf_28_1, PARAMETER_NAMES)
            for row in grad_rows:
                row["step"] = step
                parameter_gradient_coverage_rows.append(row)
        optimizer.step()
        torch.cuda.synchronize()
        iteration_seconds = time.perf_counter() - iteration_started
        step_peak = int(torch.cuda.max_memory_allocated())
        peak_training_memory = max(peak_training_memory, step_peak)
        training_loss = _training_loss_row(bundle, step)
        loss_rows.append(training_loss)
        finite = finite and bool(training_loss["finite"])
        compute_cost_rows.append(
            {
                "step": step,
                "equivariance_scheduled": bool(bundle["scheduled_eq"]),
                "iteration_seconds": iteration_seconds,
                "peak_memory_allocated_bytes": step_peak,
                "lr_group0": float(optimizer.param_groups[0]["lr"]),
            }
        )
        del bundle
        processed_batches = step

        if step in GRADIENT_STEPS:
            update_rows = parameter_update_rows(
                model.osmf_28_1,
                PARAMETER_NAMES,
                initial=initial_parameters,
                before_step=before_step,
            )
            for row in update_rows:
                row["step"] = step
                parameter_update_rows_all.append(row)
            diagnostic_loss, representation, _ = diagnostic_state(
                model, images, labels, step=step, amp_dtype=amp_dtype
            )
            representation_rows.append(representation)
            finite = finite and diagnostic_loss["finite"] and representation["finite"]
            hard_stop = _hard_stop_reason(
                gradient_ratio_rows, representation_rows, finite
            )
            print(
                "[Audit] "
                + json.dumps(
                    {
                        "step": step,
                        "loss_sshr": training_loss["loss_sshr"],
                        "reconstruction_cosine": representation[
                            "reconstruction_cosine"
                        ],
                        "semantic_morphology_rms_ratio": representation[
                            "semantic_morphology_rms_ratio"
                        ],
                        "eq_error_morphology": representation[
                            "eq_error_morphology"
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
        parameter_gradient_coverage_rows, parameter_update_rows_all
    )
    eq_values = [float(row["eq_error_morphology"]) for row in representation_rows]
    eq_responsive = max(eq_values) - min(eq_values) > 1e-6
    morphology_eq_gradient_active = any(
        row["objective"] == "eq"
        and float(row["objective_grad_norm_osmf_parameters"]) > 1e-12
        for row in gradient_ratio_rows
    )
    sshr_values = [float(row["loss_sshr"]) for row in loss_rows]
    sshr_loss_stable = all(math.isfinite(value) for value in sshr_values) and max(
        sshr_values
    ) < max(10.0, 10.0 * sshr_values[0])
    cross_covariance_finite = all(
        math.isfinite(float(row["cross_covariance"])) for row in representation_rows
    )

    timing_rows = [row for row in compute_cost_rows if int(row["step"]) > 4]
    if not timing_rows:
        timing_rows = list(compute_cost_rows)
    non_eq_times = [
        float(row["iteration_seconds"])
        for row in timing_rows
        if not row["equivariance_scheduled"]
    ]
    eq_times = [
        float(row["iteration_seconds"])
        for row in timing_rows
        if row["equivariance_scheduled"]
    ]
    mean_iteration = sum(float(row["iteration_seconds"]) for row in timing_rows) / len(
        timing_rows
    )
    mean_non_eq = (
        sum(non_eq_times) / len(non_eq_times) if non_eq_times else mean_iteration
    )
    mean_eq = sum(eq_times) / len(eq_times) if eq_times else mean_non_eq
    overhead_percent = (
        100.0 * (mean_iteration / mean_non_eq - 1.0)
        if mean_non_eq > 0.0
        else 0.0
    )

    if hard_stop:
        decision = "OSMF_PHASE0_NOGO"
        flags = []
        decision_reasons = [hard_stop]
    else:
        decision, flags, decision_reasons = decide_phase0(
            finite=finite,
            gradient_ratio_rows=gradient_ratio_rows,
            gradient_cosine_rows=gradient_cosine_rows,
            representation_rows=representation_rows,
            parameter_summary=parameter_summary,
            eq_responsive=eq_responsive,
            morphology_eq_gradient_active=morphology_eq_gradient_active,
            sshr_loss_stable=sshr_loss_stable,
            cross_covariance_finite=cross_covariance_finite,
            cost_overhead_percent=overhead_percent,
        )

    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": elapsed,
        "peak_training_memory_allocated_bytes": peak_training_memory,
    }
    (args.output_dir / "config" / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_csv(args.output_dir / "tables" / "loss_trace.csv", loss_rows)
    write_csv(
        args.output_dir / "tables" / "gradient_ratio.csv", gradient_ratio_rows
    )
    write_csv(
        args.output_dir / "tables" / "gradient_cosine.csv", gradient_cosine_rows
    )
    write_csv(
        args.output_dir / "tables" / "parameter_gradient_coverage.csv",
        parameter_gradient_coverage_rows,
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
        args.output_dir / "tables" / "reconstruction.csv",
        [
            {
                "step": row["step"],
                "reconstruction_cosine": row["reconstruction_cosine"],
                "residual_ratio": row["residual_ratio"],
            }
            for row in representation_rows
        ],
    )
    write_csv(
        args.output_dir / "tables" / "redundancy.csv",
        [
            {
                "step": row["step"],
                "cross_covariance": row["cross_covariance"],
                "semantic_rms": row["semantic_rms"],
                "morphology_rms": row["morphology_rms"],
            }
            for row in representation_rows
        ],
    )
    write_csv(
        args.output_dir / "tables" / "equivariance.csv",
        [
            {
                "step": row["step"],
                "eq_error_morphology": row["eq_error_morphology"],
                "eq_error_semantic": row["eq_error_semantic"],
            }
            for row in representation_rows
        ],
    )
    write_csv(
        args.output_dir / "tables" / "compute_cost.csv", compute_cost_rows
    )
    make_figures(
        args.output_dir,
        loss_rows,
        gradient_ratio_rows,
        gradient_cosine_rows,
        representation_rows,
        parameter_update_rows_all,
    )
    main_table = build_main_table(
        loss_rows,
        gradient_ratio_rows,
        gradient_cosine_rows,
        representation_rows,
    )
    write_csv(args.output_dir / "tables" / "main_summary.csv", main_table)
    parameter_table_rows = [
        {"parameter": name, **health} for name, health in parameter_summary.items()
    ]
    write_csv(
        args.output_dir / "tables" / "parameter_health_summary.csv",
        parameter_table_rows,
    )

    summary = {
        "decision": decision,
        "flags": flags,
        "decision_reasons": decision_reasons,
        "processed_batches": processed_batches,
        "baseline_commit": BASELINE_COMMIT,
        "phase0_parent_commit": PHASE0_PARENT_COMMIT,
        "audit_commit": args.audit_commit,
        "checkpoint_sha256": checkpoint_sha,
        "exact_command": exact_command,
        "start_state": {
            "feature_max_abs": start_feature_error,
            "reconstruction_cosine": start_representation[
                "reconstruction_cosine"
            ],
            "finite": bool(start_loss["finite"]),
        },
        "optimizer": optimizer_contract,
        "mechanism": {
            "end_reconstruction_cosine": representation_rows[-1][
                "reconstruction_cosine"
            ],
            "end_semantic_morphology_rms_ratio": representation_rows[-1][
                "semantic_morphology_rms_ratio"
            ],
            "eq_error_morphology_start": eq_values[0],
            "eq_error_morphology_end": eq_values[-1],
            "cross_covariance_start": representation_rows[0][
                "cross_covariance"
            ],
            "cross_covariance_end": representation_rows[-1][
                "cross_covariance"
            ],
            "eq_responsive": eq_responsive,
            "morphology_eq_gradient_active": morphology_eq_gradient_active,
            "sshr_loss_stable": sshr_loss_stable,
        },
        "parameter_health": parameter_summary,
        "cost": {
            "mean_iteration_seconds": mean_iteration,
            "mean_non_equivariance_seconds": mean_non_eq,
            "mean_equivariance_seconds": mean_eq,
            "overhead_percent": overhead_percent,
            "peak_memory_allocated_bytes": peak_training_memory,
        },
        "environment": environment,
        "training_scope": "exactly 128 batches unless preregistered hard stop",
        "formal_training_performed": False,
        "validation_evaluated": False,
        "test_evaluated": False,
        "luad_evaluated": False,
        "segmentation_gt_used": False,
        "phase1_started": False,
    }
    write_report(args.output_dir, summary, main_table, parameter_summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
