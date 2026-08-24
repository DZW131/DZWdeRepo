#!/usr/bin/env python3
"""Run the EXP-WDCH-002 W2 matched continuation from common epoch20."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network import resnet38_scwdch
from tools.train_wdch_matched import make_loader, source_commit, train_epoch
from tools.wdch_common import (
    LOSS_WEIGHTS,
    MatchedAugmentationDataset,
    build_optimizer,
    load_schedule,
    optimizer_summary,
    restore_named_optimizer_state,
    restore_rng_state,
    set_seed,
    sha256_file,
    write_json,
)
from tools.wdch_evaluation import evaluate_bcss
from tools.scwdch_constants import (
    EXPECTED_COMMON_EPOCH20_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    EXPERIMENT_ID,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_w2(common, kernel: int, scale: float):
    state = dict(common["model_state"])
    removed = state.pop("hfrm_28_1.context_conv.weight")
    model = resnet38_scwdch.Net(
        4, wdch_kernel_size=kernel, scwdch_scale=scale
    )
    incompat = model.load_state_dict(state, strict=False)
    expected = {
        "hfrm_28_1.wdch.haar.analysis_filters",
        "hfrm_28_1.wdch.haar.synthesis_filters",
        "hfrm_28_1.wdch.ll_context.weight",
        "hfrm_28_1.wdch.scale",
    }
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise AssertionError(
            f"Unexpected W2 load: missing={incompat.missing_keys}, "
            f"unexpected={incompat.unexpected_keys}"
        )
    if not torch.equal(
        model.hfrm_28_1.wdch.scale,
        torch.tensor(scale, dtype=model.hfrm_28_1.wdch.scale.dtype),
    ):
        raise AssertionError("Fixed calibration scale changed during model load")
    audit = {
        "expected_missing": sorted(expected),
        "unexpected": list(incompat.unexpected_keys),
        "replaced_parameter": "hfrm_28_1.context_conv.weight",
        "replaced_shape": list(removed.shape),
        "new_parameter": "hfrm_28_1.wdch.ll_context.weight",
        "new_shape": list(model.hfrm_28_1.wdch.ll_context.weight.shape),
        "new_initialization": f"uniform 1/{kernel**2}",
        "fixed_scale_buffer": scale,
    }
    return model.cuda(), audit


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing non-empty W2 output: {output}")
    for name in ("validation", "checkpoints", "predictions"):
        (output / name).mkdir(parents=True, exist_ok=True)

    common = torch.load(args.common_checkpoint, map_location="cpu", weights_only=False)
    if sha256_file(args.common_checkpoint) != EXPECTED_COMMON_EPOCH20_SHA256:
        raise AssertionError("Common epoch20 SHA256 mismatch")
    if common.get("format") != "WDCH_MATCHED_COMMON_V1":
        raise AssertionError("Unexpected common checkpoint format")
    if common.get("epoch") != 20 or common.get("optimizer_global_step") != 23420:
        raise AssertionError("W2 requires common seed42 epoch20/global_step23420")
    if common["data_sampler_state"]["schedule_sha256"] != sha256_file(args.schedule):
        raise AssertionError("Schedule differs from common checkpoint")
    if sha256_file(args.schedule) != EXPECTED_SCHEDULE_SHA256:
        raise AssertionError("Frozen 25-epoch schedule SHA256 mismatch")

    calibration = read_json(args.calibration)
    if calibration.get("experiment_id") != EXPERIMENT_ID:
        raise AssertionError("Unexpected calibration experiment")
    if calibration.get("checkpoint_sha256") != sha256_file(args.common_checkpoint):
        raise AssertionError("Calibration was not computed from this common checkpoint")
    if calibration.get("seed") != 42 or calibration.get("validation_used"):
        raise AssertionError("Calibration provenance violates the frozen protocol")
    if calibration.get("test_used") or calibration.get("split") != "BCSS training only":
        raise AssertionError("Calibration must be BCSS training-only")
    kernel = int(calibration["kernel"])
    scale = float(calibration["scale"])

    schedule = load_schedule(args.schedule)
    dataset = MatchedAugmentationDataset(args.train_root)
    set_seed(42, deterministic=True)
    model, load_audit = load_w2(common, kernel, scale)
    optimizer = build_optimizer(model)
    restore_audit = restore_named_optimizer_state(
        model, optimizer, common["optimizer_named_state"]
    )
    optimizer.global_step = int(common["optimizer_global_step"])
    skipped_names = [row["name"] for row in restore_audit["skipped"]]
    if skipped_names != ["hfrm_28_1.context_conv.weight"]:
        raise AssertionError(f"Optimizer restore mismatch: {restore_audit['skipped']}")
    if any(parameter is model.hfrm_28_1.wdch.scale for group in optimizer.param_groups for parameter in group["params"]):
        raise AssertionError("Fixed scale buffer entered the optimizer")
    restore_rng_state(common["rng_state"])

    provenance = {
        "experiment_id": EXPERIMENT_ID,
        "branch": "W2",
        "source_commit": source_commit(),
        "common_checkpoint": str(Path(args.common_checkpoint).resolve()),
        "common_checkpoint_sha256": sha256_file(args.common_checkpoint),
        "common_epoch": common["epoch"],
        "common_global_step": common["optimizer_global_step"],
        "schedule": str(Path(args.schedule).resolve()),
        "schedule_sha256": sha256_file(args.schedule),
        "calibration": str(Path(args.calibration).resolve()),
        "calibration_sha256": sha256_file(args.calibration),
        "kernel": kernel,
        "fixed_scale": scale,
        "model_load_audit": load_audit,
        "optimizer_restore_skipped": restore_audit["skipped"],
        "optimizer_after_restore": optimizer_summary(optimizer),
        "precision": "bf16",
        "loss_weights": LOSS_WEIGHTS,
        "checkpoint_selection": "none; epoch25 FINAL only",
        "validation_used_for_selection": False,
        "test_used": False,
    }
    write_json(output / "provenance.json", provenance)

    history = []
    validation_history = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(20, 25):
        loader = make_loader(dataset, schedule, epoch, args.num_workers)
        record = train_epoch(
            model, optimizer, loader, schedule["model_seeds"], epoch, "W2"
        )
        history.append(record)
        write_json(output / "training_history.json", history)
        print("SCWDCH_BRANCH_EPOCH " + json.dumps(record, sort_keys=True), flush=True)
        prediction_path = (
            output / "predictions" / "epoch25_validation.npz"
            if epoch == 24
            else None
        )
        evaluation = evaluate_bcss(
            model,
            args.val_root,
            num_workers=args.num_workers,
            prediction_output=str(prediction_path) if prediction_path else None,
        )
        evaluation.update(
            {
                "branch": "W2",
                "epoch": epoch + 1,
                "checkpoint_selection": "none; epoch25 FINAL is primary",
            }
        )
        validation_history.append(evaluation)
        write_json(output / "validation" / "history.json", validation_history)
        write_json(output / "validation" / f"epoch{epoch + 1}.json", evaluation)
        print(
            f"SCWDCH_VALIDATION branch=W2 epoch={epoch + 1} "
            f"mIoU={100*evaluation['scores']['final']['mIoU']:.4f} "
            f"mDice={100*evaluation['scores']['final']['mDice']:.4f}",
            flush=True,
        )

    checkpoint = output / "checkpoints" / "epoch25_final.pth"
    torch.save(model.state_dict(), checkpoint)
    completion = {
        "status": "SCWDCH_MATCHED_BRANCH_COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "branch": "W2",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epochs": [21, 22, 23, 24, 25],
        "global_step": optimizer.global_step,
        "fixed_scale": scale,
        "final_validation": validation_history[-1],
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "test_used": False,
    }
    write_json(output / "complete.json", completion)
    print(json.dumps({
        "status": completion["status"],
        "branch": "W2",
        "mIoU": 100 * completion["final_validation"]["scores"]["final"]["mIoU"],
        "checkpoint_sha256": completion["checkpoint_sha256"],
    }, indent=2), flush=True)
    print("SCWDCH_MATCHED_BRANCH_COMPLETE branch=W2", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--common-checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
