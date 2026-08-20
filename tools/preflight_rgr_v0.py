#!/usr/bin/env python3
"""Disposable three-step CUDA preflight for RGR-v0."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_rgr_v0 import (
    GRAPH_MODULES,
    base_state_hashes,
    build_optimizer,
    frozen_mode_ok,
    load_fresh_rgr,
    make_loader,
    module_movements,
    module_snapshots,
    parse_args,
    seed_everything,
    set_frozen_training_mode,
    train_step,
    validate_inputs,
    write_json,
    _deterministic_structure_check,
)


PREFLIGHT_STEPS = 3


def main():
    args = parse_args()
    _, source_hashes = validate_inputs(args)
    output = Path(args.output_dir)
    seed_everything()
    dataset, loader = make_loader(args)
    model, missing, zero = load_fresh_rgr(args.checkpoint)
    model.cuda()
    initial_base = base_state_hashes(model)
    initial_modules = module_snapshots(model)
    optimizer = build_optimizer(model, PREFLIGHT_STEPS)
    iterator = iter(loader)
    first = next(iterator)
    deterministic = _deterministic_structure_check(
        model,
        first[1].cuda(non_blocking=True),
        first[2].cuda(non_blocking=True),
    )
    set_frozen_training_mode(model)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, PREFLIGHT_STEPS + 1):
        _, images, labels = first if step == 1 else next(iterator)
        row = train_step(model, optimizer, images, labels, step)
        rows.append(row)
        print("RGR_PREFLIGHT_STEP", json.dumps(row, sort_keys=True), flush=True)
    movements = module_movements(model, initial_modules)
    final_base = base_state_hashes(model)
    upstream = ("node_projection", "edge_gate", "value_projection", "message_projection")
    checks = {
        "batch_size_20": first[1].shape[0] == 20,
        "zero_identity_and_deterministic": all(deterministic.values()),
        "step1_isolated_head_gradient": rows[0]["grad_isolated_head"] > 0,
        "step1_graph_head_gradient": rows[0]["grad_graph_head"] > 0,
        "upstream_gradients_by_step3": all(
            max(row[f"grad_{name}"] for row in rows) > 0 for name in upstream
        ),
        "all_modules_move": all(
            movements[name]["absolute_update_norm"] > 0 for name in GRAPH_MODULES
        ),
        "message_nonzero": any(row["message_norm"] > 0 for row in rows),
        "all_finite": all(
            row["finite"] == 1 and row["frozen_gradients_clean"] == 1 for row in rows
        ),
        "frozen_base_unchanged": initial_base == final_base,
        "mode_contract": all(frozen_mode_ok(model).values()),
    }
    summary = {
        "decision": "RGR_V0_PREFLIGHT_PASS" if all(checks.values()) else "RGR_V0_PREFLIGHT_NOGO",
        "checks": checks,
        "steps": rows,
        "movements": movements,
        "deterministic": deterministic,
        "parsed_train_samples": len(dataset),
        "missing_keys_expected_rgr_only": missing,
        "zero_initialization": zero,
        "source_hashes": source_hashes,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    write_json(output / "preflight.json", summary)
    print(summary["decision"], json.dumps(checks, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
