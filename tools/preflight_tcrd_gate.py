#!/usr/bin/env python3
"""Real batch20 BF16 engineering preflight for the frozen TCRD utility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.tcrd_dynamics import BRANCHES, TCRDDynamics
from tools.tcrd_common import (
    EXPECTED_A0_SHA256, LOSS_WEIGHTS, MatchedAugmentationDataset,
    ScheduleBatchSampler, build_optimizer, dataset_fingerprint,
    load_branch_model, load_schedule, set_deterministic_seed,
    load_state, sha256_file, write_json,
)


def run(args):
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("Preflight path guard rejected test/LUAD")
    if sha256_file(args.a0_checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("A0 SHA mismatch")
    val_root = Path(args.val_root)
    if len(list((val_root / "img").glob("*.png"))) != 3418:
        raise AssertionError("Expected 3418 BCSS validation images")
    if len(list((val_root / "mask").glob("*.png"))) != 3418:
        raise AssertionError("Expected 3418 BCSS validation masks")

    schedule = load_schedule(args.schedule)
    schedule_meta = json.loads(
        Path(args.schedule).with_suffix(".json").read_text(encoding="utf-8")
    )
    if sha256_file(args.schedule) != schedule_meta["schedule_sha256"]:
        raise AssertionError("Schedule SHA mismatch")
    if schedule["indices"].shape != (5, 1171, 20):
        raise AssertionError("Schedule does not enforce 5x1171x20")

    dataset = MatchedAugmentationDataset(args.train_root)
    if len(dataset) != 23422:
        raise AssertionError("Expected 23,422 BCSS training samples")
    if dataset_fingerprint(dataset.base, args.train_root) != schedule_meta["dataset_order_sha256"]:
        raise AssertionError("Dataset fingerprint mismatch")
    sampler = ScheduleBatchSampler(
        schedule["indices"], schedule["augmentation_seeds"], epoch=0
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    _, image, label = next(iter(loader))
    if image.shape != (20, 3, 224, 224) or label.shape != (20, 4):
        raise AssertionError("Real batch20 shape mismatch")

    set_deterministic_seed(42)
    branch_results = {}
    common_parameter_checks = {}
    a0_state = load_state(args.a0_checkpoint)
    for branch in BRANCHES:
        model, incompat = load_branch_model(branch, args.a0_checkpoint, "cuda")
        loaded_state = model.state_dict()
        common_exact = all(
            key in loaded_state and torch.equal(loaded_state[key].cpu(), value)
            for key, value in a0_state.items()
        )
        if not common_exact:
            raise AssertionError(f"Common step0 parameter mismatch in {branch}")
        model.train()
        optimizer, tail_base_lr, group_lrs = build_optimizer(model, 1171, 5)
        new_ids = set() if model.tcrd is None else {id(p) for p in model.tcrd.parameters()}
        group2_ids = [id(p) for p in optimizer.param_groups[2]["params"]]
        coverage = {str(value): group2_ids.count(value) for value in new_ids}
        if any(count != 1 for count in coverage.values()):
            raise AssertionError(f"New parameter optimizer coverage failed for {branch}")

        torch.manual_seed(int(schedule["model_seeds"][0, 0]))
        torch.cuda.manual_seed_all(int(schedule["model_seeds"][0, 0]))
        cuda_image = image.cuda(); cuda_label = label.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(cuda_image, active_labels=cuda_label)
            losses = [F.multilabel_soft_margin_loss(value, cuda_label) for value in output[:4]]
            loss = sum(weight * value for weight, value in zip(LOSS_WEIGHTS, losses))
        loss.backward()
        finite = bool(torch.isfinite(loss) and all(torch.isfinite(value).all() for value in output[:9]))
        if not finite:
            raise AssertionError(f"Non-finite output in {branch}")
        new_gradients = {}
        if model.tcrd is not None:
            for name, parameter in model.tcrd.named_parameters():
                new_gradients[name] = None if parameter.grad is None else float(
                    parameter.grad.detach().float().norm()
                )
        common_parameter_checks[branch] = {
            "common_loaded_exactly": common_exact,
            "candidate_only_missing_keys": incompat.missing_keys,
        }
        branch_results[branch] = {
            "loss": float(loss.detach()), "finite": finite,
            "cam_shapes": [list(value.shape) for value in output[5:9]],
            "new_parameter_gradients": new_gradients,
            "new_parameter_group2_coverage": coverage,
            "tail_base_lr": tail_base_lr, "initial_group_lrs": group_lrs,
            "optimizer_momentum": [group["momentum"] for group in optimizer.param_groups],
        }
        del model, optimizer, cuda_image, cuda_label, output, loss
        torch.cuda.empty_cache()

    diffusion = TCRDDynamics("D").cuda()
    toy_feature = torch.randn(2, 8, 9, 9, device="cuda")
    conductance = diffusion.conductance(toy_feature)
    conductance_error = float((conductance.sum(1) - 1).abs().max())
    reaction = TCRDDynamics("R").cuda()
    matrix = reaction.competition_matrix()
    toy_evidence = torch.randn(2, 4, 9, 9, device="cuda")
    single_active = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]], device="cuda").bool()
    single_output = reaction(toy_evidence, toy_feature, single_active)
    multi_active = torch.tensor([[1, 1, 1, 0], [1, 0, 1, 1]], device="cuda").bool()
    _, reaction_diag = reaction(
        toy_evidence, toy_feature, multi_active, return_diagnostics=True
    )
    zero_sum_errors = []
    for batch_index in range(2):
        active = multi_active[batch_index]
        zero_sum_errors.append(float(
            reaction_diag["reaction_update"][batch_index, active].sum(0).abs().max()
        ))

    report = {
        "decision": "TCRD_V0_PREFLIGHT_PASS",
        "a0_checkpoint_sha256": EXPECTED_A0_SHA256,
        "train_samples": len(dataset), "validation_pairs": 3418,
        "batch_size": 20, "precision": "bf16", "hard_epoch_limit": 5,
        "schedule_shape": list(schedule["indices"].shape),
        "schedule_sha256": schedule_meta["schedule_sha256"],
        "schedule_reused_by_all_branches": True,
        "worker_independent_augmentation": True,
        "common_model_seed_per_step": True,
        "common_parameter_checks": common_parameter_checks,
        "branches": branch_results,
        "t_steps": 3,
        "eta_d_initial": float(diffusion.eta_d.detach()),
        "eta_r_initial": float(reaction.eta_r.detach()),
        "eta_ranges_valid": bool(
            0.05 < diffusion.eta_d < 0.50 and 0.05 < reaction.eta_r < 0.50
        ),
        "kappa_initial": float(diffusion.kappa.detach()),
        "conductance_normalization_max_error": conductance_error,
        "reaction_matrix": matrix.detach().cpu().tolist(),
        "reaction_matrix_symmetric": bool(torch.equal(matrix, matrix.T)),
        "reaction_matrix_positive_offdiagonal": bool((matrix[matrix > 0] > 0).all()),
        "reaction_matrix_zero_diagonal": bool(torch.equal(matrix.diag(), torch.zeros(4, device="cuda"))),
        "reaction_zero_sum_max_errors": zero_sum_errors,
        "active_lt2_exact_identity": bool(torch.equal(single_output, toy_evidence)),
        "training_present_mask_source": "image-level GT labels only",
        "inference_present_mask_source": "official predicted deep presence with fallback",
        "segmentation_gt_enters_training": False,
        "test_used": False, "luad_used": False,
        "optimizer_step": False,
    }
    if conductance_error > 1.0e-6 or max(zero_sum_errors) > 1.0e-6:
        raise AssertionError("Mechanism numerical contract failed")
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print("TCRD_V0_PREFLIGHT_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--a0-checkpoint", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
