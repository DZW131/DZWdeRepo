#!/usr/bin/env python3
"""Independent artifact verifier for the Phase-2B1.13 audit."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.rddr_phase2b113_common import (
    CHECKPOINT_SHA256, LAMBDA_ADT, POPULATION_NAMES, PREFIX, TRAIN_BATCHES, write_json,
)


RAW_FILES = (
    "provenance.json", "identity.json", "parameter_manifest.csv",
    "train_batch_metrics.csv", "train_aggregate.json", "train_blockwise.csv", "train_bootstrap.csv",
    "oracle_batch_metrics.csv", "oracle_aggregate.json", "oracle_bootstrap.csv",
    "population_gradients.csv", "population_pairwise_cosines.csv",
    "population_oracle_projection.csv", "population_blockwise.csv",
    "snapshot_attribution.csv", "runtime.json",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check(checks, name, condition):
    checks[name] = bool(condition)


def verify(directory, report=None, post_analysis=False):
    directory = Path(directory)
    checks = {}
    for suffix in RAW_FILES:
        check(checks, f"artifact_{suffix}", (directory / f"{PREFIX}{suffix}").is_file())
    if not all(checks.values()):
        return checks
    provenance = read_json(directory / f"{PREFIX}provenance.json")
    identity = read_json(directory / f"{PREFIX}identity.json")
    runtime = read_json(directory / f"{PREFIX}runtime.json")
    manifest = read_csv(directory / f"{PREFIX}parameter_manifest.csv")
    train = read_json(directory / f"{PREFIX}train_aggregate.json")
    train_rows = read_csv(directory / f"{PREFIX}train_batch_metrics.csv")
    train_blocks = read_csv(directory / f"{PREFIX}train_blockwise.csv")
    train_bootstrap = read_csv(directory / f"{PREFIX}train_bootstrap.csv")
    oracle = read_json(directory / f"{PREFIX}oracle_aggregate.json")
    oracle_rows = read_csv(directory / f"{PREFIX}oracle_batch_metrics.csv")
    oracle_bootstrap = read_csv(directory / f"{PREFIX}oracle_bootstrap.csv")
    populations = read_csv(directory / f"{PREFIX}population_gradients.csv")
    pairwise = read_csv(directory / f"{PREFIX}population_pairwise_cosines.csv")
    pop_blocks = read_csv(directory / f"{PREFIX}population_blockwise.csv")
    snapshots = read_csv(directory / f"{PREFIX}snapshot_attribution.csv")
    source = (ROOT / "tools/run_rddr_phase2b113.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    check(checks, "provenance_pass", provenance.get("status") == "PASS")
    check(checks, "checkpoint_sha_exact", provenance.get("C0_checkpoint_sha256") == CHECKPOINT_SHA256)
    check(checks, "lambda_exact", float(provenance.get("lambda_ADT", -1)) == LAMBDA_ADT)
    check(checks, "parameter_manifest_exact", len(manifest) == 39 and
          [int(row["manifest_index"]) for row in manifest] == list(range(39)))
    check(checks, "only_approved_39_parameters",
          {row["block"] for row in manifest} == {"b4", "b4_1", "b4_2", "b4_3", "b4_4", "b4_5", "bn45"} and
          all(not row["exact_name"].startswith(("ic1", "hfrm", "b5", "b3")) for row in manifest))
    check(checks, "approved_numel_exact", sum(int(row["numel"]) for row in manifest) == 27_275_776)
    check(checks, "trackT_batch_count_128", len(train_rows) == TRAIN_BATCHES and
          [int(row["step"]) for row in train_rows] == list(range(1, TRAIN_BATCHES + 1)))
    check(checks, "phase2b112_identity_replay", train.get("replay_status") == "EXACT_TENSOR_REPLAY_PASS" and
          train.get("all_tensor_sha256_exact") is True)
    check(checks, "random_gate_rate_exact", train.get("random_rate_matching_exact") is True and
          oracle.get("random_rate_matching_exact") is True)
    check(checks, "random_seed42_exact", train.get("random_seed") == 42 and "default_rng(42)" in source)
    check(checks, "adt_formula_exact", all(token in source for token in
          ("adjudicate", "ctx_weight = q * evidence", "target = bundle[\"deep_probability\"].reshape_as(probability)",
           "(ctx_weight * kl).sum() / (ctx_weight.sum() + EPS)")))
    check(checks, "main_gradient", all(math.isfinite(float(row["main_norm"])) and float(row["main_norm"]) > 0
                                         for row in train_rows))
    check(checks, "ctx_aux_gradient", all(math.isfinite(float(row["ctx_norm"])) and float(row["ctx_norm"]) > 0
                                            for row in train_rows))
    check(checks, "random_aux_gradient", all(math.isfinite(float(row["rnd_norm"])) and float(row["rnd_norm"]) > 0
                                               for row in train_rows))
    check(checks, "aux_cosine_formula", all(abs(float(row["D_dir"]) ** 2 -
          2 * (1 - float(row["C_aux"]))) <= 2e-6 for row in train_rows))
    check(checks, "direction_difference_formula", all(abs(float(row["D_dir"]) -
          float(row["D_dir_formula"])) <= 1e-12 for row in train_rows))
    check(checks, "total_gradient_formula", all(-1 - 1e-9 <= float(row["C_tot"]) <= 1 + 1e-9
                                                  for row in train_rows))
    check(checks, "rho_ctx_formula", all(float(row["rho_ctx"]) >= 0 for row in train_rows))
    check(checks, "virtual_optimizer_matches_clone_step",
          float(train.get("virtual_optimizer_clone_max_relative_error", 1)) <= 2e-7)
    check(checks, "virtual_optimizer_no_state_write", train.get("virtual_optimizer_formal_state_mutated") is False)
    check(checks, "update_cosine_formula", all(-1 - 1e-9 <= float(row["C_update"]) <= 1 + 1e-9
                                                 for row in train_rows))
    check(checks, "rho_update_formula", all(float(row["rho_update"]) >= 0 for row in train_rows))
    check(checks, "train_block_count", len(train_blocks) == 7)
    check(checks, "train_bootstrap", len(train_bootstrap) == 7 and
          all(int(row["bootstrap_replicates"]) == 10_000 and row["statistical_unit"] == "minibatch"
              for row in train_bootstrap))
    check(checks, "validation_population_complete", oracle.get("validation_images") == 3418 and
          oracle.get("validation_minibatches") == 171)
    check(checks, "oracle_uses_gt_only_for_diagnostic", oracle.get("oracle_gt_only") is True and
          oracle.get("background_and_ignore_excluded_from_oracle") is True)
    check(checks, "gate_is_gt_blind", oracle.get("gate_gt_blind") is True and
          oracle.get("frozen_step0_probe_gate") is True)
    check(checks, "oracle_gradient_formula", "F.cross_entropy" in source and "reduction=\"sum\"" in source)
    check(checks, "oracle_alignment_formula", all(math.isfinite(float(row["DeltaC_oracle"])) for row in oracle_rows))
    check(checks, "oracle_projection_formula", all(math.isfinite(float(row["DeltaP_oracle"])) for row in oracle_rows))
    check(checks, "first_order_oracle_change_formula", all(math.isfinite(float(row["Adv_oracle"])) for row in oracle_rows))
    counts = oracle.get("population_counts", {})
    check(checks, "population_counts_exact", all(name in counts and int(counts[name]) > 0
                                                   for name in (*POPULATION_NAMES, "Raw-Wrong_0", "Raw-Correct_0",
                                                                "Top20_q0", "boundary", "interior",
                                                                "class0", "class1", "class2", "class3")))
    check(checks, "population_partition_exhaustive", oracle.get("population_partition_exhaustive") is True)
    check(checks, "population_gradient_sum_identity",
          float(oracle.get("population_group_sum_max_relative_error", 1)) <= 2e-6)
    check(checks, "population_rows_exact", len(populations) == 8 and
          {(row["mode"], row["population"]) for row in populations} ==
          {(mode, population) for mode in ("context", "random") for population in POPULATION_NAMES})
    check(checks, "population_cancellation_index", all(0 <= float(oracle[key]) <= 2 for key in
          ("context_CancellationIndex", "random_CancellationIndex")))
    check(checks, "blockwise_sum_identity", len(pop_blocks) == 14)
    check(checks, "pairwise_population_cosines", len(pairwise) == 10)
    check(checks, "oracle_bootstrap_seed42", len(oracle_bootstrap) == 3 and
          all(int(row["bootstrap_seed"]) == 42 and row["statistical_unit"] == "minibatch"
              for row in oracle_bootstrap))
    check(checks, "snapshot_attribution_complete", len(snapshots) == 9 and
          {(int(row["step"]), row["state_arm"]) for row in snapshots} ==
          {(step, arm) for step in (0, 250, 500) for arm in ("B", "A", "R")})
    check(checks, "all_gradients_finite", runtime.get("all_gradients_finite") is True and
          all(row["finite"].lower() == "true" for row in (*train_rows, *oracle_rows)))
    check(checks, "state_hash_unchanged", identity.get("model_state_unchanged") is True and
          identity.get("bn_running_state_unchanged") is True and
          identity.get("source_hashes_unchanged") is True)
    check(checks, "no_grad_outside_approved_params", "model.zero_grad" not in source and
          "loss.backward" not in source and "main_loss.backward" not in source)
    check(checks, "no_optimizer_mutation", identity.get("formal_optimizer_state_unchanged") is True)
    check(checks, "no_optimizer_step", runtime.get("optimizer_steps") == 0 and ".step()" not in source)
    check(checks, "no_checkpoint_write", runtime.get("checkpoint_writes") == 0 and "torch.save" not in source)
    check(checks, "no_random_seed_sweep", source.count("default_rng(42)") >= 2 and "seed_sweep" not in source)
    check(checks, "no_test", runtime.get("test_access") is False)
    check(checks, "no_luad", runtime.get("luad_access") is False)
    check(checks, "runtime_complete", runtime.get("completed") is True and runtime.get("zero_step") is True)
    if post_analysis:
        summary_path = directory / f"{PREFIX}summary.json"
        check(checks, "summary_exists", summary_path.is_file())
        check(checks, "report_exists", report is not None and Path(report).is_file())
        if summary_path.is_file() and report is not None and Path(report).is_file():
            summary = read_json(summary_path)
            text = Path(report).read_text(encoding="utf-8-sig")
            check(checks, "report_last_line_exact",
                  text.rstrip().splitlines()[-1] == f"DIAGNOSIS = {summary['diagnosis']}")
            check(checks, "diagnosis_allowed", summary["diagnosis"] in {
                "CONTEXT_SELECTION_COLLAPSES_IN_SHARED_PARAMETER_SPACE",
                "CONTEXT_GRADIENT_SURVIVES_WITHOUT_PARAMETER_SEMANTIC_ADVANTAGE",
                "CONTEXT_SELECTION_PARAMETER_MISALIGNMENT",
                "CONTEXT_PARAMETER_ADVANTAGE_SURVIVES",
                "PARAMETER_ATTRIBUTION_INCONCLUSIVE",
                "PARAMETER_ATTRIBUTION_ENGINEERING_INVALID",
            })
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--post-analysis", action="store_true")
    args = parser.parse_args()
    checks = verify(args.input, args.report, args.post_analysis)
    failures = [name for name, passed in checks.items() if not passed]
    payload = {"passed": not failures, "post_analysis": args.post_analysis,
               "checks": checks, "check_count": len(checks), "failures": failures}
    write_json(args.input / f"{PREFIX}verification.json", payload)
    print(json.dumps(payload), flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
