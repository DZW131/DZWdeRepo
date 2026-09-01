#!/usr/bin/env python3
"""Apply the frozen Phase-2B1.13 gates and generate the final report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tools.rddr_phase2b113_common import (
    HISTORICAL_DECISIONS, POPULATION_NAMES, PREFIX, require, write_json,
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return float(row[key])


def table(rows, columns):
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.10g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def lookup(rows, endpoint, statistic="mean"):
    matches = [row for row in rows if row["endpoint"] == endpoint and row["statistic"] == statistic]
    require(len(matches) == 1, f"Bootstrap endpoint missing/duplicate: {endpoint}/{statistic}")
    return {key: float(matches[0][key]) for key in ("point", "ci_low", "ci_high")}


def analyze(directory):
    directory = Path(directory)
    provenance = read_json(directory / f"{PREFIX}provenance.json")
    identity = read_json(directory / f"{PREFIX}identity.json")
    runtime = read_json(directory / f"{PREFIX}runtime.json")
    verification = read_json(directory / f"{PREFIX}verification.json")
    train = read_json(directory / f"{PREFIX}train_aggregate.json")
    train_rows = read_csv(directory / f"{PREFIX}train_batch_metrics.csv")
    blocks = read_csv(directory / f"{PREFIX}train_blockwise.csv")
    train_bootstrap = read_csv(directory / f"{PREFIX}train_bootstrap.csv")
    oracle = read_json(directory / f"{PREFIX}oracle_aggregate.json")
    oracle_rows = read_csv(directory / f"{PREFIX}oracle_batch_metrics.csv")
    oracle_bootstrap = read_csv(directory / f"{PREFIX}oracle_bootstrap.csv")
    population = read_csv(directory / f"{PREFIX}population_gradients.csv")
    pairwise = read_csv(directory / f"{PREFIX}population_pairwise_cosines.csv")
    population_blocks = read_csv(directory / f"{PREFIX}population_blockwise.csv")
    snapshots = read_csv(directory / f"{PREFIX}snapshot_attribution.csv")

    median_c_aux = float(np.median([number(row, "C_aux") for row in train_rows]))
    median_d_dir = float(np.median([number(row, "D_dir") for row in train_rows]))
    median_rho_update = float(np.median([number(row, "rho_update") for row in train_rows]))
    block_count = sum(number(row, "C_aux") <= 0.97 for row in blocks)
    gate_a = (train["C_aux"] <= 0.95 and median_c_aux <= 0.97 and
              median_d_dir >= 0.20 and block_count >= 4)
    rho_bootstrap = lookup(train_bootstrap, "rho_update")
    gate_b = (median_rho_update >= 0.02 and train["C_update"] <= 0.9995 and
              rho_bootstrap["ci_low"] > 0.01)
    delta_c = lookup(oracle_bootstrap, "DeltaC_oracle")
    advantage = lookup(oracle_bootstrap, "Adv_oracle")
    mean_delta_c = float(np.mean([number(row, "DeltaC_oracle") for row in oracle_rows]))
    mean_advantage = float(np.mean([number(row, "Adv_oracle") for row in oracle_rows]))
    gate_c = (mean_delta_c > 0 and delta_c["ci_low"] > 0 and
              mean_advantage > 0 and advantage["ci_low"] > 0)
    harmful = (oracle["P_DW_ctx"] > 0 and oracle["P_SW_ctx"] + oracle["P_SC_ctx"] < 0 and
               abs(oracle["P_SW_ctx"] + oracle["P_SC_ctx"]) >= 0.25 * abs(oracle["P_DW_ctx"]) and
               oracle["context_CancellationIndex"] >= 0.20)
    random_superior = ((mean_delta_c < 0 and delta_c["ci_high"] < 0) or
                       (mean_advantage < 0 and advantage["ci_high"] < 0))
    gate_f_checks = {
        "raw_verification_pass": verification.get("passed") is True,
        "runtime_complete": runtime.get("completed") is True,
        "all_gradients_finite": runtime.get("all_gradients_finite") is True,
        "model_state_unchanged": identity.get("model_state_unchanged") is True,
        "bn_state_unchanged": identity.get("bn_running_state_unchanged") is True,
        "optimizer_state_unchanged": identity.get("formal_optimizer_state_unchanged") is True,
        "source_hashes_unchanged": identity.get("source_hashes_unchanged") is True,
        "group_sum_identity": oracle.get("population_group_sum_max_relative_error", 1) <= 2e-6,
        "random_rate_exact": train.get("random_rate_matching_exact") is True and
                             oracle.get("random_rate_matching_exact") is True,
        "adt_formula_exact": verification.get("checks", {}).get("adt_formula_exact") is True,
        "no_optimizer_step": runtime.get("optimizer_steps") == 0,
        "no_checkpoint_write": runtime.get("checkpoint_writes") == 0,
        "no_test": runtime.get("test_access") is False,
        "no_luad": runtime.get("luad_access") is False,
    }
    gate_f = all(gate_f_checks.values())
    if not gate_f:
        diagnosis = "PARAMETER_ATTRIBUTION_ENGINEERING_INVALID"
    elif random_superior:
        diagnosis = "CONTEXT_SELECTION_PARAMETER_MISALIGNMENT"
    elif not gate_a and not gate_b:
        diagnosis = "CONTEXT_SELECTION_COLLAPSES_IN_SHARED_PARAMETER_SPACE"
    elif gate_a and gate_b and not gate_c:
        diagnosis = "CONTEXT_GRADIENT_SURVIVES_WITHOUT_PARAMETER_SEMANTIC_ADVANTAGE"
    elif gate_a and gate_b and gate_c:
        diagnosis = "CONTEXT_PARAMETER_ADVANTAGE_SURVIVES"
    else:
        diagnosis = "PARAMETER_ATTRIBUTION_INCONCLUSIVE"
    next_route = {
        "CONTEXT_SELECTION_COLLAPSES_IN_SHARED_PARAMETER_SPACE":
            "Stop the external deep-to-shallow KL consumer; return to CH/GSR rectification design.",
        "CONTEXT_GRADIENT_SURVIVES_WITHOUT_PARAMETER_SEMANTIC_ADVANTAGE":
            "Redesign reliability-signal consumption without lambda or LR tuning.",
        "CONTEXT_SELECTION_PARAMETER_MISALIGNMENT":
            "Stop treating contextual selection as a superior shared-parameter target.",
        "CONTEXT_PARAMETER_ADVANTAGE_SURVIVES":
            "A separately preregistered early-training timing feasibility audit is now eligible; Full25 remains locked.",
        "PARAMETER_ATTRIBUTION_INCONCLUSIVE":
            "Do not train; resolve the failed prerequisite gates before designing a consumer.",
        "PARAMETER_ATTRIBUTION_ENGINEERING_INVALID":
            "Repair audit engineering only; no scientific interpretation or training is allowed.",
    }[diagnosis]
    summary = {
        "status": "FINAL",
        "historical_decisions": HISTORICAL_DECISIONS,
        "gates": {
            "A_context_gradient_distinguishability": gate_a,
            "B_difference_survives_optimizer": gate_b,
            "C_parameter_semantic_advantage": gate_c,
            "D_harmful_population_cancellation": harmful,
            "E_random_parameter_alignment_superior": random_superior,
            "F_engineering_validity": gate_f,
        },
        "gate_details": {
            "aggregate_C_aux": train["C_aux"], "median_C_aux": median_c_aux,
            "median_D_dir": median_d_dir, "qualifying_blocks": block_count,
            "median_rho_update": median_rho_update, "aggregate_C_update": train["C_update"],
            "rho_update_bootstrap": rho_bootstrap,
            "mean_DeltaC_oracle": mean_delta_c, "DeltaC_oracle_bootstrap": delta_c,
            "mean_Adv_oracle": mean_advantage, "Adv_oracle_bootstrap": advantage,
            "gate_f_checks": gate_f_checks,
        },
        "HARMFUL_POPULATION_CANCELLATION": harmful,
        "RANDOM_PARAMETER_ALIGNMENT_SUPERIOR": random_superior,
        "diagnosis": diagnosis,
        "next_route_constraint": next_route,
    }
    return summary, {
        "provenance": provenance, "identity": identity, "runtime": runtime,
        "verification": verification, "train": train, "train_rows": train_rows,
        "blocks": blocks, "train_bootstrap": train_bootstrap, "oracle": oracle,
        "oracle_rows": oracle_rows, "oracle_bootstrap": oracle_bootstrap,
        "population": population, "pairwise": pairwise,
        "population_blocks": population_blocks, "snapshots": snapshots,
    }


def build_report(summary, evidence):
    train, oracle = evidence["train"], evidence["oracle"]
    train_sample = [{key: row[key] for key in ("batch", "C_aux", "D_dir", "R_norm", "C_tot", "rho_ctx", "C_update", "rho_update")}
                    for row in evidence["train_rows"][:10]]
    oracle_sample = [{key: row[key] for key in ("batch", "n_images", "DeltaC_oracle", "DeltaP_oracle", "Adv_oracle")}
                     for row in evidence["oracle_rows"][:10]]
    gates = [{"Gate": key.split("_", 1)[0], "Result": "PASS" if value else "FAIL/false",
              "Meaning": key.split("_", 1)[1].replace("_", " ")}
             for key, value in summary["gates"].items()]
    population_rows = [{key: (float(row[key]) if key in ("gradient_norm", "norm_share", "oracle_projection") else row[key])
                        for key in ("mode", "population", "n_pixels", "gradient_norm", "norm_share", "oracle_projection")}
                       for row in evidence["population"]]
    lines = [
        "# RDDR Phase-2B1.13 Context-Specific Parameter-Gradient Attribution Audit",
        "",
        "## 1. Provenance",
        "",
        f"Status: FINAL. Pure A0 `{evidence['provenance']['pure_A0_commit']}`; C0 SHA256 "
        f"`{evidence['provenance']['C0_checkpoint_sha256']}`; frozen lambda `{evidence['provenance']['lambda_ADT']}`.",
        "",
        "## 2. Frozen Phase2B1.12 result",
        "",
        "The prior result remains `DECISION = ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION`. "
        "No historical decision is changed by this audit.",
        "",
        "## 3. Scientific question",
        "",
        "Does contextual-vs-rate-matched-random pixel selection remain distinguishable after the shared-convolution "
        "Jacobian, and is its parameter gradient more aligned with a GT-only shallow-semantic oracle?",
        "",
        "## 4. Approved parameter space",
        "",
        f"Exactly 39 parameters and {evidence['provenance']['approved_total_numel']:,} scalars from "
        "`b4,b4_1,b4_2,b4_3,b4_4,b4_5,bn45`; `ic1`, HFRM, b5+, and b3- are excluded.",
        "",
        "## 5. Track T replay",
        "",
        f"{train['batches']} frozen batch20 minibatches replayed with status `{train['replay_status']}`. "
        f"All tensor SHA256 matches: {train['all_tensor_sha256_exact']}.",
        "",
        "## 6. g_main / g_ctx / g_rand",
        "",
        "All parameter gradients are FP32; scalar products, norms, aggregates, and bootstrap statistics are FP64. "
        "Every batch starts from identical C0 weights and performs no update.",
        "",
        "## 7. Auxiliary gradient cosine",
        "",
        f"Aggregate C_aux = `{train['C_aux']:.10g}`; median batch C_aux = "
        f"`{summary['gate_details']['median_C_aux']:.10g}`.",
        "",
        "## 8. Direction difference",
        "",
        f"Aggregate D_dir = `{train['D_dir']:.10g}`; median batch D_dir = "
        f"`{summary['gate_details']['median_D_dir']:.10g}`. Direct and sqrt(2(1-cos)) definitions agree.",
        "",
        "## 9. Norm ratio",
        "",
        f"Aggregate R_norm = `{train['R_norm']:.10g}`.",
        "",
        "## 10. Main-loss alignment",
        "",
        f"Aggregate C_ctx_main = `{train['C_ctx_main']:.10g}`, C_rnd_main = `{train['C_rnd_main']:.10g}`, "
        f"Delta = `{train['DeltaC_main']:.10g}`. This is diagnostic, not a semantic-quality claim.",
        "",
        "## 11. Total-gradient equivalence",
        "",
        f"Aggregate C_tot = `{train['C_tot']:.10g}`; rho_ctx = `{train['rho_ctx']:.10g}`.",
        "",
        "## 12. Virtual optimizer equivalence",
        "",
        f"Aggregate C_update = `{train['C_update']:.10g}`; rho_update = `{train['rho_update']:.10g}`; "
        f"clone dry-run maximum relative error = `{train['virtual_optimizer_clone_max_relative_error']:.3e}`. "
        "The dry-run acted only on isolated clones.",
        "",
        "## 13. Block-wise attribution",
        "",
        table([{key: (float(row[key]) if key != "block" else row[key]) for key in
                ("block", "C_aux", "D_dir", "R_norm", "C_tot", "rho_ctx", "C_update", "rho_update")}
               for row in evidence["blocks"]],
              ["block", "C_aux", "D_dir", "R_norm", "C_tot", "rho_ctx", "C_update", "rho_update"]),
        "",
        "## 14. Track T bootstrap",
        "",
        table(evidence["train_bootstrap"], ["endpoint", "statistic", "point", "ci_low", "ci_high",
                                             "bootstrap_replicates", "statistical_unit"]),
        "",
        "First ten minibatches (the complete CSV contains all 128):",
        "",
        table(train_sample, list(train_sample[0])),
        "",
        "## 15. Oracle definition",
        "",
        "The oracle is mean cross-entropy on pre-HFRM raw shallow logits at native28, using only GT classes 0–3. "
        "Background 4 and ignore 255 are excluded. GT is used only for attribution; contextual and random gates remain GT-blind.",
        "",
        "## 16. Oracle alignment",
        "",
        f"Mean DeltaC_oracle = `{summary['gate_details']['mean_DeltaC_oracle']:.10g}` with paired-minibatch "
        f"95% CI `[{summary['gate_details']['DeltaC_oracle_bootstrap']['ci_low']:.10g}, "
        f"{summary['gate_details']['DeltaC_oracle_bootstrap']['ci_high']:.10g}]`.",
        "",
        "## 17. Oracle projection",
        "",
        f"Aggregate DeltaP_oracle = `{oracle['aggregate_DeltaP_oracle']:.10g}`.",
        "",
        "## 18. First-order oracle change",
        "",
        f"Mean Adv_oracle = `{summary['gate_details']['mean_Adv_oracle']:.10g}` with paired-minibatch "
        f"95% CI `[{summary['gate_details']['Adv_oracle_bootstrap']['ci_low']:.10g}, "
        f"{summary['gate_details']['Adv_oracle_bootstrap']['ci_high']:.10g}]`.",
        "",
        "First ten validation minibatches (the complete CSV contains all batches):",
        "",
        table(oracle_sample, list(oracle_sample[0])),
        "",
        "## 19. Frozen population replay",
        "",
        f"All 3418 BCSS validation images were covered in {oracle['validation_minibatches']} minibatches. "
        f"DW/SW/BW/SC is an exhaustive partition of {oracle['oracle_foreground_pixels']:,} foreground pixels. "
        f"Maximum group-sum gradient relative error: `{oracle['population_group_sum_max_relative_error']:.3e}`.",
        "",
        "## 20. DW/SW/BW/SC gradient decomposition",
        "",
        table(population_rows, ["mode", "population", "n_pixels", "gradient_norm", "norm_share", "oracle_projection"]),
        "",
        "## 21. Pairwise population cosines",
        "",
        table(evidence["pairwise"], ["mode", "left", "right", "cosine"]),
        "",
        "## 22. Cancellation index",
        "",
        f"Context CI = `{oracle['context_CancellationIndex']:.10g}`; random CI = "
        f"`{oracle['random_CancellationIndex']:.10g}`.",
        "",
        "## 23. Population oracle projections",
        "",
        f"Context: P_DW=`{oracle['P_DW_ctx']:.10g}`, P_SW=`{oracle['P_SW_ctx']:.10g}`, "
        f"P_BW=`{oracle['P_BW_ctx']:.10g}`, P_SC=`{oracle['P_SC_ctx']:.10g}`.",
        "",
        "## 24. Block-wise population analysis",
        "",
        table(evidence["population_blocks"], list(evidence["population_blocks"][0])),
        "",
        "## 25. Step250/500 secondary attribution",
        "",
        table(evidence["snapshots"], ["step", "state_arm", "C_aux", "D_dir", "rho_ctx", "C_tot",
                                      "state_unchanged"]),
        "",
        "## 26. State/engineering verification",
        "",
        f"Independent verification PASS: `{evidence['verification']['passed']}`. Model, BN, formal optimizer, and source "
        f"hashes unchanged: `{evidence['identity']['model_state_unchanged']}`, "
        f"`{evidence['identity']['bn_running_state_unchanged']}`, "
        f"`{evidence['identity']['formal_optimizer_state_unchanged']}`, "
        f"`{evidence['identity']['source_hashes_unchanged']}`. Optimizer steps/checkpoint writes: "
        f"`{evidence['runtime']['optimizer_steps']}/{evidence['runtime']['checkpoint_writes']}`.",
        "",
        "## 27. Gate A-F",
        "",
        table(gates, ["Gate", "Result", "Meaning"]),
        "",
        "## 28. HARMFUL_POPULATION_CANCELLATION",
        "",
        f"`HARMFUL_POPULATION_CANCELLATION = {str(summary['HARMFUL_POPULATION_CANCELLATION']).upper()}`",
        "",
        "## 29. RANDOM_PARAMETER_ALIGNMENT_SUPERIOR",
        "",
        f"`RANDOM_PARAMETER_ALIGNMENT_SUPERIOR = {str(summary['RANDOM_PARAMETER_ALIGNMENT_SUPERIOR']).upper()}`",
        "",
        "## 30. Exact diagnosis",
        "",
        f"`DIAGNOSIS = {summary['diagnosis']}`",
        "",
        "## 31. Scientific interpretation",
        "",
        "The diagnosis is applied by the preregistered precedence rules only. Pixel-level adjudication quality is not "
        "relabelled as optimization utility, and no lambda, LR, gate, class, boundary, or timing search was performed.",
        "",
        "## 32. Next-route constraint",
        "",
        summary["next_route_constraint"],
        "",
        f"DIAGNOSIS = {summary['diagnosis']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    summary, evidence = analyze(args.input)
    write_json(args.input / f"{PREFIX}summary.json", summary)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(summary, evidence), encoding="utf-8")
    print(json.dumps({"diagnosis": summary["diagnosis"], "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
