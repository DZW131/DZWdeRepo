#!/usr/bin/env python3
"""Run matched A2/A3 continuations for EXP-CBCCH-002."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_cbcch
from research.wdch import contrastive_affinity_loss
from tools.train_wdch_matched import compute_accuracy, make_loader
from tools.wdch_common import (
    EXPECTED_TRAIN,
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    build_optimizer,
    capture_rng_state,
    load_schedule,
    named_optimizer_state,
    optimizer_summary,
    restore_named_optimizer_state,
    restore_rng_state,
    set_seed,
    sha256_file,
    write_json,
)
from tools.wdch_evaluation import evaluate_bcss


CONTRASTIVE_WEIGHT = 0.10
TEMPERATURE = 0.07
HARD_FRACTION = 0.20
LOCAL_KERNEL = 15


def source_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_cbcch_model(common, variant: str):
    model = resnet38_cbcch.Net(4, variant=variant)
    incompat = model.load_state_dict(common["model_state"], strict=False)
    expected = {
        "hfrm_28_1.haar.analysis_filters",
        "hfrm_28_1.haar.synthesis_filters",
    }
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise AssertionError(str(incompat))
    model.hfrm_28_1.set_semantic_probe(model.ic1)
    audit = {
        "expected_missing_fixed_buffers": sorted(expected),
        "unexpected": incompat.unexpected_keys,
        "context_parameter": "hfrm_28_1.context_conv.weight",
        "context_parameter_restored_exactly": bool(
            torch.equal(
                model.hfrm_28_1.context_conv.weight.detach().cpu(),
                common["model_state"]["hfrm_28_1.context_conv.weight"],
            )
        ),
        "new_trainable_parameter_count_vs_C0": 0,
        "semantic_probe": "existing ic1; non-registering shared reference",
        "propagation": "local 15x15 softmax dot-product affinity",
        "variant": variant,
    }
    if not audit["context_parameter_restored_exactly"]:
        raise AssertionError("Epoch20 CH15 parameter did not restore exactly")
    return model.cuda(), audit


def train_epoch(model, optimizer, loader, model_seeds, epoch: int, variant: str):
    model.train()
    started = time.time()
    totals = {
        "official": 0.0,
        "contrastive": 0.0,
        "total": 0.0,
        "exact": 0.0,
        "accuracy": 0.0,
        "valid_anchors": 0.0,
        "valid_anchor_fraction": 0.0,
        "positive_similarity": 0.0,
        "negative_similarity": 0.0,
        "similarity_margin": 0.0,
    }
    samples = 0
    batches = 0
    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0]

    hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
    try:
        for step, (_, image, target) in enumerate(loader):
            seed = int(model_seeds[epoch, step])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(image)
                official_losses = [
                    F.multilabel_soft_margin_loss(output, target)
                    for output in outputs[:4]
                ]
                official = sum(
                    weight * value
                    for weight, value in zip(LOSS_WEIGHTS, official_losses)
                )
                contrastive, stats = contrastive_affinity_loss(
                    captured["feature"],
                    model.hfrm_28_1.last_semantic_logits,
                    target,
                    model.hfrm_28_1.haar,
                    variant=variant,
                    kernel_size=LOCAL_KERNEL,
                    hard_fraction=HARD_FRACTION,
                    temperature=TEMPERATURE,
                )
                total_loss = official + CONTRASTIVE_WEIGHT * contrastive
            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1} step {step}"
                )
            total_loss.backward()
            optimizer.step()
            batch = image.shape[0]
            exact, accuracy = compute_accuracy(
                torch.sigmoid(outputs[3]).detach(), target
            )
            totals["official"] += float(official.detach()) * batch
            totals["contrastive"] += float(contrastive.detach()) * batch
            totals["total"] += float(total_loss.detach()) * batch
            totals["exact"] += exact * batch
            totals["accuracy"] += accuracy * batch
            for key in (
                "valid_anchors",
                "valid_anchor_fraction",
                "positive_similarity",
                "negative_similarity",
                "similarity_margin",
            ):
                totals[key] += float(stats[key])
            samples += batch
            batches += 1
            if (step + 1) % 100 == 0:
                print(
                    f"CBCCH_TRAIN variant={variant} epoch={epoch + 1}/25 "
                    f"batch={step + 1}/1171 global_step={optimizer.global_step}/29275 "
                    f"official={totals['official']/samples:.6f} "
                    f"contrastive={totals['contrastive']/samples:.6f} "
                    f"total={totals['total']/samples:.6f} "
                    f"valid={totals['valid_anchor_fraction']/batches:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True,
                )
    finally:
        hook.remove()
    record = {
        "variant": variant,
        "epoch": epoch + 1,
        "global_step": optimizer.global_step,
        "official_classification_loss": totals["official"] / samples,
        "contrastive_loss": totals["contrastive"] / samples,
        "total_loss": totals["total"] / samples,
        "training_exact_match": totals["exact"] / samples,
        "training_accuracy": totals["accuracy"] / samples,
        "valid_anchors_per_batch": totals["valid_anchors"] / batches,
        "valid_anchor_fraction": totals["valid_anchor_fraction"] / batches,
        "positive_similarity": totals["positive_similarity"] / batches,
        "negative_similarity": totals["negative_similarity"] / batches,
        "similarity_margin": totals["similarity_margin"] / batches,
        "gamma_context": float(model.hfrm_28_1.gamma_context.detach().float()),
        "gamma_veto": float(model.hfrm_28_1.gamma_veto.detach().float()),
        "lr_end": float(optimizer.param_groups[0]["lr"]),
        "seconds": time.time() - started,
        "all_finite": bool(np.isfinite(totals["total"] / samples)),
    }
    return record


def run(args):
    variant = args.variant.upper()
    output = Path(args.output_dir)
    recovery_path = output / "checkpoints" / "recovery_latest.pth"
    if output.exists() and any(output.iterdir()) and not recovery_path.exists():
        raise FileExistsError(
            f"Refusing non-empty CBCCH output without valid recovery: {output}"
        )
    for name in ("validation", "checkpoints", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=True)

    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if common.get("format") != "WDCH_MATCHED_COMMON_V1" or common.get("epoch") != 20:
        raise AssertionError("Invalid common Epoch20 state")
    if common["data_sampler_state"]["schedule_sha256"] != sha256_file(args.schedule):
        raise AssertionError("Schedule differs from common checkpoint")
    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    if len(dataset) != EXPECTED_TRAIN:
        raise AssertionError(f"Expected {EXPECTED_TRAIN}, got {len(dataset)}")

    set_seed(42, deterministic=True)
    model, load_audit = load_cbcch_model(common, variant)
    optimizer = build_optimizer(model)
    restore_audit = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    if restore_audit["skipped"]:
        raise AssertionError(
            f"CBCCH must restore all optimizer state: {restore_audit['skipped']}"
        )
    restore_rng_state(common["rng_state"])

    history = []
    start_epoch = 20
    if recovery_path.exists():
        recovery = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if (
            recovery.get("format") != "CBCCH_MATCHED_RECOVERY_V1"
            or recovery.get("variant") != variant
            or recovery.get("common_checkpoint_sha256") != sha256_file(args.common_checkpoint)
            or recovery.get("schedule_sha256") != sha256_file(args.schedule)
        ):
            raise AssertionError("CBCCH recovery provenance differs")
        model.load_state_dict(recovery["model_state"], strict=True)
        model.hfrm_28_1.set_semantic_probe(model.ic1)
        optimizer = build_optimizer(model)
        resumed_restore = restore_named_optimizer_state(
            model, optimizer, recovery["optimizer_named_state"]
        )
        if resumed_restore["skipped"]:
            raise AssertionError(resumed_restore)
        optimizer.global_step = int(recovery["optimizer_global_step"])
        restore_rng_state(recovery["rng_state"])
        history = list(recovery["history"])
        start_epoch = int(recovery["epoch"])
        if start_epoch < 20 or start_epoch > 25 or len(history) != start_epoch - 20:
            raise AssertionError("Invalid CBCCH recovery epoch/history")
        print(f"CBCCH_RESUME variant={variant} completed_epoch={start_epoch}", flush=True)

    provenance = {
        "experiment_id": "EXP-CBCCH-002",
        "variant": variant,
        "source_commit": source_commit(),
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "common_epoch": common["epoch"],
        "common_global_step": common["optimizer_global_step"],
        "schedule_sha256": sha256_file(args.schedule),
        "model_load_audit": load_audit,
        "optimizer_restore": restore_audit,
        "optimizer_after_restore": optimizer_summary(optimizer),
        "precision": "bf16",
        "loss_weights": LOSS_WEIGHTS,
        "contrastive": {
            "weight": CONTRASTIVE_WEIGHT,
            "temperature": TEMPERATURE,
            "hard_fraction": 1.0 if variant == "A2" else HARD_FRACTION,
            "local_kernel": LOCAL_KERNEL,
            "pairing": "one deterministic positive and negative per valid anchor",
        },
        "checkpoint_rule": "Epoch25 FINAL only",
        "test_used": False,
        "luad_used": False,
    }
    if not (output / "provenance.json").exists():
        write_json(output / "provenance.json", provenance)

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch, 25):
        loader = make_loader(dataset, schedule, epoch, args.num_workers)
        record = train_epoch(
            model, optimizer, loader, schedule["model_seeds"], epoch, variant
        )
        history.append(record)
        write_json(output / "training_history.json", history)
        torch.save(
            {
                "format": "CBCCH_MATCHED_RECOVERY_V1",
                "variant": variant,
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_named_state": named_optimizer_state(model, optimizer),
                "optimizer_global_step": optimizer.global_step,
                "rng_state": capture_rng_state(),
                "history": history,
                "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
                "schedule_sha256": sha256_file(args.schedule),
            },
            recovery_path,
        )
        print("CBCCH_BRANCH_EPOCH " + json.dumps(record, sort_keys=True), flush=True)

    prediction_path = output / "predictions" / "epoch25_validation.npz"
    evaluation = evaluate_bcss(
        model,
        args.val_root,
        num_workers=args.num_workers,
        prediction_output=str(prediction_path),
    )
    evaluation.update(
        {
            "variant": variant,
            "epoch": 25,
            "checkpoint_selection": "none; Epoch25 FINAL only",
        }
    )
    write_json(output / "validation" / "epoch25.json", evaluation)
    checkpoint_path = output / "checkpoints" / "epoch25_final.pth"
    torch.save(model.state_dict(), checkpoint_path)
    completion = {
        "status": "CBCCH_MATCHED_COMPLETE",
        "experiment_id": "EXP-CBCCH-002",
        "variant": variant,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "epochs": [21, 22, 23, 24, 25],
        "global_step": optimizer.global_step,
        "final_validation": evaluation,
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
        "luad_used": False,
    }
    write_json(output / "complete.json", completion)
    if recovery_path.exists():
        recovery_path.unlink()
    print(
        json.dumps(
            {
                "status": completion["status"],
                "variant": variant,
                "mIoU": 100 * evaluation["scores"]["final"]["mIoU"],
                "mDice": 100 * evaluation["scores"]["final"]["mDice"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"CBCCH_MATCHED_COMPLETE variant={variant}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("A2", "A3"), required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
