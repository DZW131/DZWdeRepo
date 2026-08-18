"""Run the frozen OSMF-v1.2 Phase-0M morphology causal audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.osmf_v12 import (
    cross_subspace_covariance,
    inverse_align_morphology,
    reconstruction_cosine,
    semantic_preservation_agreement,
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
from tools.audit_osmf_v12_gradient_gate import (
    EXPECTED_MISSING_KEYS,
    _all_finite,
    _diagnostic_state,
    _forward_objectives,
    _load_checkpoint,
    _parameter_summary,
    _training_loss_row,
)
from tools.osmf_phase0_audit.gradients import (
    gradient_decomposition,
    parameter_gradient_rows,
    parameter_update_rows,
    snapshot_parameters,
)
from tools.osmf_v12_audit.report import write_csv
from tools.osmf_v12_phase0m import (
    AUTHORIZED_BATCHES,
    BATCH_SIZE,
    EQ_STEPS,
    FIXED_PROBE_STEPS,
    GRADIENT_STEPS,
    IMAGE_SIZE,
    MORPHOLOGY_PARAMETER_NAMES,
    OBJECTIVE_WEIGHTS,
    PROBE_BATCH_SIZE,
    PROBE_IMAGES,
    REPLICATION_AUDIT_STEPS,
    REPLICATION_REFERENCE,
    SEED,
)
from tools.osmf_v12_phase0m.diagnostics import (
    affinity_equivariance_error,
    all_finite,
    causal_statistics,
    decide_phase0m,
    morphology_gradient_competition,
    replication_deviations,
)
from tools.osmf_v12_phase0m.report import make_figures, write_report
from train_sshr import seed_worker, set_seed


FROZEN_V12_EXECUTED_COMMIT = "92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _normalized_image(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
        )
    tensor = TF.to_tensor(image)
    return TF.normalize(
        tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )


def _apply_dataset_flips(
    tensor: torch.Tensor, horizontal: bool, vertical: bool
) -> torch.Tensor:
    result = tensor
    if horizontal:
        result = torch.flip(result, dims=(2,))
    if vertical:
        result = torch.flip(result, dims=(1,))
    return result


def _infer_dataset_flips(path: str, realized: torch.Tensor) -> tuple[bool, bool]:
    canonical = _normalized_image(path)
    candidates = {
        (False, False): canonical,
        (True, False): _apply_dataset_flips(canonical, True, False),
        (False, True): _apply_dataset_flips(canonical, False, True),
        (True, True): _apply_dataset_flips(canonical, True, True),
    }
    errors = {
        key: float((candidate - realized.cpu()).abs().max())
        for key, candidate in candidates.items()
    }
    choice = min(errors, key=errors.get)
    if errors[choice] > 1e-6:
        raise AssertionError(f"Cannot reproduce realized augmentation for {path}: {errors}")
    return choice


def _build_fixed_probe(dataset: Stage1_TrainDataset):
    rng = random.Random(SEED)
    indices = rng.sample(range(len(dataset)), PROBE_IMAGES)
    records, manifest = [], []
    for position, index in enumerate(indices):
        path, _ = dataset.object[index]
        horizontal = bool(rng.getrandbits(1))
        vertical = bool(rng.getrandbits(1))
        pair_flip_dimension = 3 if bool(rng.getrandbits(1)) else 2
        tensor = _apply_dataset_flips(
            _normalized_image(path), horizontal, vertical
        ).contiguous()
        record = {
            "position": position,
            "dataset_index": index,
            "image_id": Path(path).stem,
            "image_path": path,
            "dataset_hflip": horizontal,
            "dataset_vflip": vertical,
            "photometric_transform": "none; frozen ImageNet normalization",
            "pair_flip_dimension": pair_flip_dimension,
            "pair_flip_type": "horizontal" if pair_flip_dimension == 3 else "vertical",
            "selection_seed": SEED,
            "tensor_sha256": _tensor_sha256(tensor),
            "tensor": tensor,
        }
        records.append(record)
        manifest.append({key: value for key, value in record.items() if key != "tensor"})
    return records, manifest


def _eval_pair(
    model: Net, images: torch.Tensor, flip_dimension: int
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = model.forward_osmf_features(images)
        second = model.forward_osmf_features(
            torch.flip(images, dims=(flip_dimension,))
        )
        morphology_b = inverse_align_morphology(
            second["morphology"], flip_dimension
        )
        semantic_b = inverse_align_morphology(second["semantic"], flip_dimension)
        result = {
            "eq_error_morphology": float(
                spatial_equivariance_loss(first["morphology"], morphology_b).cpu()
            ),
            "eq_error_semantic": float(
                spatial_equivariance_loss(first["semantic"], semantic_b).cpu()
            ),
        }
    model.train(was_training)
    return result


def _fixed_probe_measure(model: Net, records: list[dict], step: int) -> dict:
    totals = {
        "eq_error_morphology": 0.0,
        "eq_error_semantic": 0.0,
        "affinity_eq_error_morphology": 0.0,
        "affinity_eq_error_semantic": 0.0,
    }
    count = 0
    was_training = model.training
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for flip_dimension in (2, 3):
            selected = [
                record for record in records
                if record["pair_flip_dimension"] == flip_dimension
            ]
            for start in range(0, len(selected), PROBE_BATCH_SIZE):
                chunk = selected[start : start + PROBE_BATCH_SIZE]
                images = torch.stack([record["tensor"] for record in chunk]).cuda()
                first = model.forward_osmf_features(images)
                second = model.forward_osmf_features(
                    torch.flip(images, dims=(flip_dimension,))
                )
                morphology_b = inverse_align_morphology(
                    second["morphology"], flip_dimension
                )
                semantic_b = inverse_align_morphology(
                    second["semantic"], flip_dimension
                )
                values = {
                    "eq_error_morphology": spatial_equivariance_loss(
                        first["morphology"], morphology_b
                    ),
                    "eq_error_semantic": spatial_equivariance_loss(
                        first["semantic"], semantic_b
                    ),
                    "affinity_eq_error_morphology": affinity_equivariance_error(
                        first["morphology"], second["morphology"], flip_dimension
                    ),
                    "affinity_eq_error_semantic": affinity_equivariance_error(
                        first["semantic"], second["semantic"], flip_dimension
                    ),
                }
                for name, value in values.items():
                    totals[name] += float(value.cpu()) * len(chunk)
                count += len(chunk)
    model.train(was_training)
    result = {"step": step, "images": count}
    result.update({name: value / count for name, value in totals.items()})
    result["finite"] = all_finite(result[name] for name in totals)
    return result


def _source_hashes() -> dict[str, str]:
    names = (
        "network/osmf_v12.py",
        "network/resnet38_cls_osmf_v12.py",
        "train_osmf_v12.py",
        "tools/audit_osmf_v12_phase0m.py",
        "tools/osmf_v12_phase0m/__init__.py",
        "tools/osmf_v12_phase0m/diagnostics.py",
        "tools/osmf_v12_phase0m/report.py",
        "docs/specs/osmf_v12_phase0m_preregistered_contract.md",
        "tool/GenDataset.py",
        "train_sshr.py",
    )
    return {name: sha256_file(REPO_ROOT / name) for name in names}


def _proof(path: Path) -> str:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("decision") != "OSMF_V12_PHASE0_REVIEW":
        raise RuntimeError("Phase-0M requires the frozen OSMF_V12_PHASE0_REVIEW proof")
    if summary.get("audit_commit") != FROZEN_V12_EXECUTED_COMMIT:
        raise RuntimeError("v1.2 proof does not match the frozen executed commit")
    return sha256_file(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--v12-phase0-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16",))
    return parser.parse_args()


def main():
    args = parse_args()
    lowered = str(args.train_root).lower()
    if any(token in lowered for token in ("test", "luad", "val")):
        raise ValueError("Phase-0M accepts BCSS training data only")
    if len(args.audit_commit) != 40:
        raise ValueError("--audit-commit must be a full Git SHA")
    proof_sha = _proof(args.v12_phase0_summary)
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f"Unexpected checkpoint SHA256: {checkpoint_sha}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name in ("config", "tables", "figures", "docs"):
        (args.output_dir / name).mkdir()

    set_seed(SEED)
    model = Net(n_class=4).cuda()
    dataset = Stage1_TrainDataset(
        data_path=str(args.train_root), dataset="bcss", img_size=IMAGE_SIZE
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
    if set(missing) != EXPECTED_MISSING_KEYS or unexpected:
        raise AssertionError("Frozen checkpoint compatibility changed")
    model.train()

    probe_records, probe_manifest = _build_fixed_probe(dataset)
    write_csv(args.output_dir / "tables" / "fixed_probe_manifest.csv", probe_manifest)
    path_by_id = {Path(path).stem: path for path, _ in dataset.object}
    optimizer_contract = {
        "class": type(optimizer).__name__,
        "max_step": max_step,
        "lr_power": optimizer.lr_power,
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
        "authorized_batches": AUTHORIZED_BATCHES,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "precision": "bf16",
        "probe_images": PROBE_IMAGES,
        "probe_batch_size": PROBE_BATCH_SIZE,
        "eq_steps": EQ_STEPS,
        "fixed_probe_steps": FIXED_PROBE_STEPS,
        "gradient_steps": GRADIENT_STEPS,
        "objective_weights": OBJECTIVE_WEIGHTS,
        "baseline_commit": BASELINE_COMMIT,
        "frozen_v12_executed_commit": FROZEN_V12_EXECUTED_COMMIT,
        "audit_commit": args.audit_commit,
        "checkpoint_sha256": checkpoint_sha,
        "v12_phase0_summary_sha256": proof_sha,
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
    first_names, first_images, first_labels = next(iterator)
    first_images = first_images.cuda(non_blocking=True)
    first_labels = first_labels.cuda(non_blocking=True)
    start_loss, start_representation, identity_error = _diagnostic_state(
        model, first_images, first_labels, 0
    )
    if identity_error >= 1e-6 or not start_loss["finite"]:
        raise RuntimeError("Frozen v1.2 start-state parity failed")

    fixed_probe_rows = [_fixed_probe_measure(model, probe_records, 0)]
    causal_rows, training_pair_manifest = [], []
    gradient_competition_rows = []
    ratio_rows, ratio_cosine_rows = [], []
    representation_rows = [start_representation]
    loss_rows = [start_loss]
    parameter_gradient_rows_all, parameter_update_rows_all = [], []
    initial_parameters = snapshot_parameters(
        model.osmf_28_1,
        ("p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight"),
    )
    finite = True
    processed_batches = 0
    peak_memory = 0
    started = time.perf_counter()
    current = (first_names, first_images, first_labels)

    for step in range(1, AUTHORIZED_BATCHES + 1):
        if step > 1:
            names, images, labels = next(iterator)
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        else:
            names, images, labels = current
        flip_dimension = _flip_dimension(step)

        before_pair = None
        if step in EQ_STEPS:
            before_pair = _eval_pair(model, images, flip_dimension)
            for position, image_id in enumerate(names):
                path = path_by_id[str(image_id)]
                hflip, vflip = _infer_dataset_flips(path, images[position].cpu())
                training_pair_manifest.append(
                    {
                        "step": step,
                        "batch_position": position,
                        "image_id": str(image_id),
                        "image_path": path,
                        "dataset_hflip": hflip,
                        "dataset_vflip": vflip,
                        "photometric_transform": "none; frozen ImageNet normalization",
                        "pair_flip_dimension": flip_dimension,
                        "pair_flip_type": "horizontal" if flip_dimension == 3 else "vertical",
                        "tensor_sha256": _tensor_sha256(images[position]),
                    }
                )

        diagnostic_bundle = None
        if step in REPLICATION_AUDIT_STEPS:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    diagnostic_bundle = _forward_objectives(
                        model, images, labels, step, force_equivariance=True
                    )
                ratios, cosines = gradient_decomposition(
                    {
                        "base": diagnostic_bundle["base"],
                        "sem_pres": diagnostic_bundle["sem_pres"],
                        "eq": diagnostic_bundle["eq"],
                        "orth": diagnostic_bundle["orth"],
                        "rec": diagnostic_bundle["rec"],
                    },
                    diagnostic_bundle["aux"]["input"],
                    tuple(model.osmf_28_1.parameters()),
                    OBJECTIVE_WEIGHTS,
                    objective_names=("sem_pres", "eq", "orth", "rec"),
                )
                for row in ratios:
                    ratio_rows.append({"step": step, **row})
                for row in cosines:
                    ratio_cosine_rows.append({"step": step, **row})
                if step in GRADIENT_STEPS:
                    selected = dict(model.osmf_28_1.named_parameters())
                    morphology_parameters = tuple(
                        selected[name] for name in MORPHOLOGY_PARAMETER_NAMES
                    )
                    competition = morphology_gradient_competition(
                        {
                            "base": diagnostic_bundle["base"],
                            "sem_pres": diagnostic_bundle["sem_pres"],
                            "eq": diagnostic_bundle["eq"],
                            "orth": diagnostic_bundle["orth"],
                            "rec": diagnostic_bundle["rec"],
                        },
                        morphology_parameters,
                    )
                    gradient_competition_rows.append({"step": step, **competition})
            finite = finite and all(
                row["finite"] for row in ratios + cosines
            ) and (diagnostic_bundle is None or _all_finite(diagnostic_bundle))
            del diagnostic_bundle, ratios, cosines

        optimizer.zero_grad()
        before_parameters = (
            snapshot_parameters(
                model.osmf_28_1,
                ("p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight"),
            )
            if step in REPLICATION_AUDIT_STEPS
            else None
        )
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bundle = _forward_objectives(
                model, images, labels, step, force_equivariance=False
            )
        bundle["total"].backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        if step in REPLICATION_AUDIT_STEPS:
            for row in parameter_gradient_rows(
                model.osmf_28_1,
                ("p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight"),
            ):
                parameter_gradient_rows_all.append({"step": step, **row})
        optimizer.step()
        peak_memory = max(peak_memory, int(torch.cuda.max_memory_allocated()))
        processed_batches = step
        loss_rows.append(_training_loss_row(bundle, step))

        if step in REPLICATION_AUDIT_STEPS:
            for row in parameter_update_rows(
                model.osmf_28_1,
                ("p_sem.weight", "p_morph.weight", "u_sem.weight", "u_morph.weight"),
                initial_parameters,
                before_parameters,
            ):
                parameter_update_rows_all.append({"step": step, **row})
            _, representation, _ = _diagnostic_state(model, images, labels, step)
            representation_rows.append(representation)
            finite = finite and representation["finite"]

        if step in EQ_STEPS:
            after_pair = _eval_pair(model, images, flip_dimension)
            delta = (
                after_pair["eq_error_morphology"]
                - before_pair["eq_error_morphology"]
            )
            neutral = abs(delta) < 1e-6
            causal_rows.append(
                {
                    "step": step,
                    "flip_dimension": flip_dimension,
                    "flip_type": "horizontal" if flip_dimension == 3 else "vertical",
                    "eq_before": before_pair["eq_error_morphology"],
                    "eq_after": after_pair["eq_error_morphology"],
                    "delta": delta,
                    "improved": delta < 0 and not neutral,
                    "harmed": delta > 0 and not neutral,
                    "neutral": neutral,
                    "semantic_eq_before": before_pair["eq_error_semantic"],
                    "semantic_eq_after": after_pair["eq_error_semantic"],
                    "input_batch_sha256": hashlib.sha256(
                        "".join(_tensor_sha256(image) for image in images).encode()
                    ).hexdigest(),
                }
            )

        if step in FIXED_PROBE_STEPS:
            fixed_probe_rows.append(_fixed_probe_measure(model, probe_records, step))

        if not finite:
            break
        if step in FIXED_PROBE_STEPS or step in EQ_STEPS:
            print(
                "[Phase0M] "
                + json.dumps(
                    {
                        "step": step,
                        "processed_batches": processed_batches,
                        "causal_delta": None if not causal_rows else causal_rows[-1]["delta"],
                        "fixed_probe": None if step not in FIXED_PROBE_STEPS else fixed_probe_rows[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - started
    if processed_batches != AUTHORIZED_BATCHES:
        finite = False
    causal = causal_statistics(causal_rows)
    fixed_start, fixed_end = fixed_probe_rows[0], fixed_probe_rows[-1]
    raw_delta = (
        fixed_end["eq_error_morphology"]
        - fixed_start["eq_error_morphology"]
    )
    affinity_delta = (
        fixed_end["affinity_eq_error_morphology"]
        - fixed_start["affinity_eq_error_morphology"]
    )
    parameter_summary = _parameter_summary(
        parameter_gradient_rows_all, parameter_update_rows_all
    )
    ratios_by_objective = {}
    for objective in ("sem_pres", "eq", "orth", "rec"):
        values = [
            float(row["ratio"])
            for row in ratio_rows
            if row["objective"] == objective
        ]
        ratios_by_objective[objective] = {
            "mean": sum(values) / len(values),
            "max": max(values),
            "values": values,
        }
    observed = {
        "mean_r_sem": ratios_by_objective["sem_pres"]["mean"],
        "mean_r_eq": ratios_by_objective["eq"]["mean"],
        "semantic_agreement_end": representation_rows[-1]["semantic_agreement"],
        "reconstruction_cosine_end": representation_rows[-1]["reconstruction_cosine"],
        "semantic_morphology_rms_ratio_end": representation_rows[-1]["semantic_morphology_rms_ratio"],
        "cross_covariance_end": representation_rows[-1]["cross_covariance"],
    }
    deviations, replication_instability = replication_deviations(
        observed, REPLICATION_REFERENCE
    )
    mean_eq_base_cosine = sum(
        float(row["cos_eq_base"]) for row in gradient_competition_rows
    ) / len(gradient_competition_rows)
    healthy = bool(
        finite
        and all(
            health["grad_nonzero"] and health["measurable_update"]
            for health in parameter_summary.values()
        )
        and representation_rows[-1]["semantic_response_rms_ratio"] > 0.05
        and representation_rows[-1]["reconstruction_cosine"] >= 0.95
        and 0.05
        < representation_rows[-1]["semantic_morphology_rms_ratio"]
        < 20.0
    )
    decision, flags, reasons = decide_phase0m(
        causal=causal,
        fixed_raw_delta=raw_delta,
        fixed_affinity_delta=affinity_delta,
        healthy=healthy,
        mean_eq_base_cosine=mean_eq_base_cosine,
        replication_instability=replication_instability,
    )

    write_csv(args.output_dir / "tables" / "same_pair_causal.csv", causal_rows)
    write_csv(args.output_dir / "tables" / "same_pair_summary.csv", [causal])
    write_csv(args.output_dir / "tables" / "training_pair_manifest.csv", training_pair_manifest)
    write_csv(args.output_dir / "tables" / "fixed_probe_raw_eq.csv", [
        {
            "step": row["step"],
            "eq_error_morphology": row["eq_error_morphology"],
            "eq_error_semantic": row["eq_error_semantic"],
        }
        for row in fixed_probe_rows
    ])
    write_csv(args.output_dir / "tables" / "fixed_probe_affinity_eq.csv", [
        {
            "step": row["step"],
            "affinity_eq_error_morphology": row["affinity_eq_error_morphology"],
            "affinity_eq_error_semantic": row["affinity_eq_error_semantic"],
        }
        for row in fixed_probe_rows
    ])
    write_csv(args.output_dir / "tables" / "morphology_gradient_cosine.csv", gradient_competition_rows)
    write_csv(args.output_dir / "tables" / "representation_health.csv", representation_rows)
    write_csv(args.output_dir / "tables" / "gradient_budget_replication.csv", ratio_rows)
    write_csv(args.output_dir / "tables" / "loss_trace.csv", loss_rows)
    write_csv(args.output_dir / "tables" / "parameter_gradient_coverage.csv", parameter_gradient_rows_all)
    write_csv(args.output_dir / "tables" / "parameter_update.csv", parameter_update_rows_all)
    write_csv(args.output_dir / "tables" / "parameter_health_summary.csv", [
        {"parameter": name, **health} for name, health in parameter_summary.items()
    ])

    summary = {
        "decision": decision,
        "decision_reasons": reasons,
        "flags": flags,
        "processed_batches": processed_batches,
        "authorized_batches": AUTHORIZED_BATCHES,
        "fresh_restart_from_a0": True,
        "audit_commit": args.audit_commit,
        "frozen_v12_executed_commit": FROZEN_V12_EXECUTED_COMMIT,
        "baseline_commit": BASELINE_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "v12_phase0_summary_sha256": proof_sha,
        "exact_command": contract["exact_command"],
        "same_pair_causal": causal,
        "fixed_probe": {
            "images": PROBE_IMAGES,
            "raw_morphology_start": fixed_start["eq_error_morphology"],
            "raw_morphology_end": fixed_end["eq_error_morphology"],
            "raw_morphology_delta": raw_delta,
            "raw_semantic_start": fixed_start["eq_error_semantic"],
            "raw_semantic_end": fixed_end["eq_error_semantic"],
            "affinity_morphology_start": fixed_start["affinity_eq_error_morphology"],
            "affinity_morphology_end": fixed_end["affinity_eq_error_morphology"],
            "affinity_morphology_delta": affinity_delta,
            "affinity_semantic_start": fixed_start["affinity_eq_error_semantic"],
            "affinity_semantic_end": fixed_end["affinity_eq_error_semantic"],
            "trajectory": fixed_probe_rows,
        },
        "morphology_gradient_competition": {
            "mean_cos_eq_base": mean_eq_base_cosine,
            "mean_cos_eq_sem": sum(float(row["cos_eq_sem"]) for row in gradient_competition_rows) / len(gradient_competition_rows),
            "mean_cos_eq_orth": sum(float(row["cos_eq_orth"]) for row in gradient_competition_rows) / len(gradient_competition_rows),
            "mean_cos_eq_rec": sum(float(row["cos_eq_rec"]) for row in gradient_competition_rows) / len(gradient_competition_rows),
        },
        "gradient_budget_replication": ratios_by_objective,
        "representation": {
            "semantic_agreement_start": representation_rows[0]["semantic_agreement"],
            "semantic_agreement_end": representation_rows[-1]["semantic_agreement"],
            "semantic_response_rms_ratio_end": representation_rows[-1]["semantic_response_rms_ratio"],
            "reconstruction_cosine_end": representation_rows[-1]["reconstruction_cosine"],
            "semantic_morphology_rms_ratio_end": representation_rows[-1]["semantic_morphology_rms_ratio"],
            "cross_covariance_start": representation_rows[0]["cross_covariance"],
            "cross_covariance_end": representation_rows[-1]["cross_covariance"],
            "healthy": healthy,
        },
        "parameter_health": parameter_summary,
        "replication": {
            "reference": REPLICATION_REFERENCE,
            "observed": observed,
            "relative_deviations": deviations,
            "instability": replication_instability,
        },
        "finite": finite,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "cost": {
            "elapsed_seconds": elapsed,
            "peak_memory_allocated_bytes": peak_memory,
        },
        "checkpoint_saved": False,
        "validation_evaluated": False,
        "test_evaluated": False,
        "luad_evaluated": False,
        "segmentation_gt_used": False,
        "phase1_started": False,
        "v13_implemented": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "config" / "environment.json").write_text(
        json.dumps(summary["environment"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_figures(
        args.output_dir,
        causal_rows,
        fixed_probe_rows,
        gradient_competition_rows,
        representation_rows,
    )
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
