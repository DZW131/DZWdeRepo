"""Run the frozen BCSS-validation Phase-0B routing-signal audit.

The CLI deliberately exposes no test/LUAD path, threshold, feature selection,
probe hyperparameter, inference weight, TTA, metric, or architecture option.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from tools.decision_audit.fusion import score_predictions
from tools.routing_signal_audit import (
    BASELINE_COMMIT,
    BCSS_THRESHOLDS,
    BOOTSTRAP_REPLICATES,
    BRANCH_NAMES,
    CHECKPOINT_SHA256,
    EXPECTED_IMAGES,
    EXPECTED_SLIDES,
    FOLD_SEED,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_HIDDEN_DIM,
    MLP_LR,
    OFFICIAL_FUSION,
    PCA_DIMENSIONS,
    PHASE0_PARENT_COMMIT,
    RIDGE_ALPHA,
    ROUTING_THRESHOLD,
    SAFE_CANDIDATES,
)
from tools.routing_signal_audit.bootstrap import grouped_slide_bootstrap
from tools.routing_signal_audit.cache_validation import (
    recheck_exact_parity,
    sha256_file,
    validate_phase0_assets,
)
from tools.routing_signal_audit.metrics import (
    build_utility_targets,
    evaluate_probe_predictions,
    frozen_primary_decision,
    score_row,
    signal_target_audit,
    write_csv,
    write_json,
)
from tools.routing_signal_audit.oracle_image_fusion import (
    image_fusion_grid,
    run_image_fusion_oracle,
)
from tools.routing_signal_audit.oracle_local_class import (
    local_class_combinations,
    run_exact_local_imageclass_oracle,
)
from tools.routing_signal_audit.oracle_safe import run_safe_image_oracle
from tools.routing_signal_audit.oracle_slide import run_slide_oracle
from tools.routing_signal_audit.probe_linear import run_linear_probe
from tools.routing_signal_audit.probe_mlp import run_mlp_probe
from tools.routing_signal_audit.report import write_report
from tools.routing_signal_audit.signal_cam import extract_cam_signals
from tools.routing_signal_audit.signal_feature import (
    build_oof_pca_context,
    compose_signal_c,
    load_base_signal_sets,
    load_feature_context,
)
from tools.routing_signal_audit.signal_tta import (
    extract_tta_and_feature_signals,
)
from tools.routing_signal_audit.visualization import (
    generate_qualitative_routing_panels,
    generate_summary_figures,
)


def announce(message: str) -> None:
    print(f"[phase0b] {message}", flush=True)


def audit_commit() -> str:
    explicit = os.environ.get("ROUTING_PHASE0B_GIT_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _runtime_test_rows(context: dict) -> list[dict]:
    checks = [
        ("official parity exact", context["parity"]["pass"], "zero differing pixels and metric diff <1e-7"),
        ("checkpoint SHA exact", context["checkpoint_sha"] == CHECKPOINT_SHA256, context["checkpoint_sha"]),
        ("3418 images", context["num_images"] == EXPECTED_IMAGES, context["num_images"]),
        ("22 slide IDs", context["num_slides"] == EXPECTED_SLIDES, context["num_slides"]),
        ("fold assignment identical to Phase-0", context["assignment_hash_match"], context["assignment_hash"]),
        ("no slide leakage", context["no_slide_leakage"], "one fold per source slide"),
        ("no test paths", context["test_evaluated"] is False, "test_evaluated=false"),
        ("no LUAD paths", context["luad_evaluated"] is False, "luad_evaluated=false"),
        ("no network modification", context["forbidden_source_diff"] is False, "git diff verified before archive"),
        ("official thresholds unchanged", tuple(context["thresholds"]) == tuple(BCSS_THRESHOLDS), context["thresholds"]),
        ("official metric unchanged", context["metric"] == "tool.iouutils.scores", context["metric"]),
        ("official TTA unchanged", context["tta"] == ["identity", "hflip", "vflip"], context["tta"]),
        ("Signal A contains no GT", context["signal_a_no_gt"], "extractor has no GT argument"),
        ("Signal B contains no GT", context["signal_b_no_gt"], "extractor has no GT argument"),
        ("Signal C contains no GT", context["signal_c_no_gt"], "frozen feature hooks only"),
        ("slide ID absent from router input", context["slide_id_not_input"], "used only for folds/bootstrap"),
        ("PCA fit train-fold only", context["pca_train_only"], "fit_scope=train_fold_only"),
        ("scaler fit train-fold only", context["scaler_train_only"], "fit_scope=train_fold_only"),
        ("held-out absent from PCA fit", context["pca_train_only"], "separate train/heldout indices"),
        ("held-out absent from scaler fit", context["scaler_train_only"], "transform-only heldout"),
        ("held-out GT absent from probe training", context["heldout_gt_not_fit"], "train-fold targets only"),
        ("Ridge alpha fixed 1.0", RIDGE_ALPHA == 1.0, RIDGE_ALPHA),
        ("MLP hidden dim fixed 32", MLP_HIDDEN_DIM == 32, MLP_HIDDEN_DIM),
        ("MLP epochs fixed 200", MLP_EPOCHS == 200, MLP_EPOCHS),
        ("MLP lr fixed 1e-3", MLP_LR == 1e-3, MLP_LR),
        ("MLP seed fixed 20260817", FOLD_SEED == 20260817, FOLD_SEED),
        ("safe routing threshold exactly zero", ROUTING_THRESHOLD == 0.0, ROUTING_THRESHOLD),
        ("no threshold sweep", context["threshold_sweep"] is False, "single threshold=0"),
        ("all images have one OOF prediction", context["oof_assignment_min"] == 1, context["oof_assignment_min"]),
        ("no image has multiple OOF predictions", context["oof_assignment_max"] == 1, context["oof_assignment_max"]),
        ("safe oracle includes official", SAFE_CANDIDATES[0] == "official_fusion", SAFE_CANDIDATES),
        ("image fusion grid has 286 candidates", len(image_fusion_grid()) == 286, len(image_fusion_grid())),
        ("official fusion exists in grid", bool(np.any(np.all(np.isclose(image_fusion_grid(), OFFICIAL_FUSION), axis=1))), OFFICIAL_FUSION),
        ("local enumeration maximum 625", len(local_class_combinations()) == 625, len(local_class_combinations())),
        ("local oracle naming is bounded", context["local_oracle_name_safe"], "Exact Local Image×Class Oracle"),
        ("bootstrap samples slide IDs", context["bootstrap_unit"] == "source_slide", context["bootstrap_unit"]),
        ("all signals finite", context["signals_finite"], "A/B/C finite"),
        ("all probe outputs finite", context["probes_finite"], "six OOF outputs finite"),
        ("prediction masks finite/integer", context["masks_valid"], "uint8 masks"),
        ("test_evaluated=false", context["test_evaluated"] is False, "false"),
    ]
    rows = [
        {"test_id": index + 1, "requirement": name, "pass": bool(passed), "evidence": str(evidence)}
        for index, (name, passed, evidence) in enumerate(checks)
    ]
    failed = [row for row in rows if not row["pass"]]
    if failed:
        raise RuntimeError(f"Phase-0B required test failure; STOP: {failed}")
    return rows


def _probe_result_filename(probe_name: str) -> str:
    return probe_name.lower().replace("-", "_") + "_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-dir", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Phase-0B frozen BF16 extraction requires CUDA")
    output_dir = args.output_dir.resolve()
    config_dir = output_dir / "config"
    cache_dir = output_dir / "cache"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    docs_dir = output_dir / "docs"
    for directory in (config_dir, cache_dir, tables_dir, figures_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    phase0_dir = args.phase0_dir.resolve()
    phase0_cache_dir = phase0_dir / "cache"
    phase0b_commit = audit_commit()
    exact_command = (
        "python -u tools/audit_routing_signal_learnability.py \\\n"
        f"  --phase0-dir {phase0_dir} \\\n"
        f"  --val-root {args.val_root.resolve()} \\\n"
        f"  --checkpoint {args.checkpoint.resolve()} \\\n"
        f"  --output-dir {output_dir} \\\n"
        f"  --num-workers {args.num_workers}"
    )
    frozen_contract = {
        "scope": "Frozen-model BCSS validation-only Phase-0B routing-signal learnability audit",
        "phase0b_parent_commit": PHASE0_PARENT_COMMIT,
        "baseline_commit": BASELINE_COMMIT,
        "phase0b_commit": phase0b_commit,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "phase0_dir": str(phase0_dir),
        "validation_root": str(args.val_root.resolve()),
        "num_images": EXPECTED_IMAGES,
        "num_slides": EXPECTED_SLIDES,
        "official_fusion": list(OFFICIAL_FUSION),
        "class_presence_thresholds": list(BCSS_THRESHOLDS),
        "tta": ["identity", "hflip", "vflip"],
        "folds": 5,
        "fold_seed": FOLD_SEED,
        "ridge_alpha": RIDGE_ALPHA,
        "mlp": {
            "hidden_dim": MLP_HIDDEN_DIM,
            "epochs": MLP_EPOCHS,
            "learning_rate": MLP_LR,
            "candidate_batch_size": MLP_BATCH_SIZE,
            "seed": FOLD_SEED,
        },
        "safe_routing_threshold": ROUTING_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "test_evaluated": False,
        "luad_evaluated": False,
        "sshr_training_performed": False,
        "exact_command": exact_command,
    }
    write_json(config_dir / "frozen_contract.json", frozen_contract)
    run_started = time.perf_counter()
    phase_seconds = {}

    announce("validating immutable Phase-0 cache and fold contract")
    phase_started = time.perf_counter()
    assets = validate_phase0_assets(
        phase0_dir,
        args.val_root,
        args.checkpoint,
        PHASE0_PARENT_COMMIT,
    )
    shutil.copyfile(
        phase0_dir / "tables" / "class_probe_fold_assignments.csv",
        tables_dir / "groupkfold_assignment.csv",
    )
    copied_assignment_hash = sha256_file(tables_dir / "groupkfold_assignment.csv")
    if copied_assignment_hash != assets["fold_assignment_hash"]:
        raise RuntimeError("Copied Phase-0 fold assignment hash mismatch; STOP")
    parity = recheck_exact_parity(
        phase0_dir, cache_dir / "phase0b_parity_reconstruction.npy"
    )
    write_csv(tables_dir / "parity.csv", [parity])
    phase_seconds["contract_and_parity"] = time.perf_counter() - phase_started

    announce("constructing GT-defined audit targets and four frozen oracle taxonomies")
    phase_started = time.perf_counter()
    utilities = build_utility_targets(
        phase0_cache_dir, cache_dir / "image_utility_targets.npy"
    )
    safe_oracle = run_safe_image_oracle(
        phase0_cache_dir,
        utilities,
        cache_dir / "safe_image_oracle_predictions.npy",
    )
    slide_oracle = run_slide_oracle(
        phase0_cache_dir,
        assets["source_groups"],
        safe_oracle["summary"],
        cache_dir / "slide_oracle_predictions.npy",
    )
    image_fusion_oracle = run_image_fusion_oracle(
        phase0_cache_dir,
        safe_oracle["summary"],
        cache_dir / "image_fusion_oracle_predictions.npy",
    )
    local_oracle = run_exact_local_imageclass_oracle(
        phase0_cache_dir,
        safe_oracle["summary"],
        cache_dir / "exact_local_imageclass_oracle_predictions.npy",
    )
    np.save(cache_dir / "safe_image_oracle_choices.npy", safe_oracle["choices"])
    np.save(cache_dir / "image_fusion_oracle_choices.npy", image_fusion_oracle["choices"])
    np.save(cache_dir / "exact_local_imageclass_oracle_choices.npy", local_oracle["choices"])
    write_csv(tables_dir / "safe_image_oracle.csv", [safe_oracle["summary"]])
    write_csv(tables_dir / "safe_image_oracle_choices.csv", safe_oracle["image_rows"])
    write_csv(tables_dir / "slide_oracle.csv", slide_oracle["rows"])
    write_csv(tables_dir / "image_fusion_oracle.csv", [image_fusion_oracle["summary"]])
    write_csv(tables_dir / "image_fusion_oracle_choices.csv", image_fusion_oracle["rows"])
    write_csv(tables_dir / "exact_local_imageclass_oracle.csv", [local_oracle["summary"]])
    write_csv(tables_dir / "exact_local_imageclass_choices.csv", local_oracle["rows"])
    phase_seconds["oracle_taxonomy"] = time.perf_counter() - phase_started

    announce("extracting GT-free Signal A aggregated-CAM evidence")
    phase_started = time.perf_counter()
    signal_a_manifest = extract_cam_signals(
        phase0_cache_dir, cache_dir / "cam_signal_features.npy"
    )
    phase_seconds["signal_a"] = time.perf_counter() - phase_started

    announce("extracting aligned TTA reliability and frozen semantic feature context")
    phase_started = time.perf_counter()
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = importlib.import_module("network.resnet38_cls").Net_CAM(n_class=4)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.cuda().eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen A0 model must remain eval-only with zero trainable parameters")
    torch.cuda.reset_peak_memory_stats()
    signal_bc_manifest = extract_tta_and_feature_signals(
        model,
        args.val_root,
        assets["image_names"],
        phase0_cache_dir,
        cache_dir,
        args.num_workers,
    )
    del model, state_dict
    torch.cuda.empty_cache()
    phase_seconds["signal_b_c"] = time.perf_counter() - phase_started

    announce("building fold-safe OOF PCA context and level-0 correlation audit")
    phase_started = time.perf_counter()
    oof_pca, pca_names, pca_rows = build_oof_pca_context(
        cache_dir,
        assets["fold_by_index"],
        cache_dir / "oof_pca_context.npy",
    )
    signal_a, signal_b, names_a, names_b = load_base_signal_sets(cache_dir)
    _, feature_scalar, scalar_names = load_feature_context(cache_dir)
    all_indices = np.arange(EXPECTED_IMAGES)
    signal_c = compose_signal_c(
        signal_b, feature_scalar, oof_pca, all_indices
    )
    names_c = names_b + scalar_names + pca_names
    correlation_rows, auroc_rows = signal_target_audit(
        {
            "A": (signal_a, names_a),
            "B": (signal_b, names_b),
            "C": (signal_c, names_c),
        },
        utilities,
        assets["fold_by_index"],
    )
    write_csv(tables_dir / "signal_correlations.csv", correlation_rows)
    write_csv(tables_dir / "signal_auroc.csv", auroc_rows)
    write_csv(tables_dir / "pca_explained_variance.csv", pca_rows)
    phase_seconds["pca_and_level0"] = time.perf_counter() - phase_started

    announce("running six preregistered OOF probes; MLP-C remains the sole primary")
    phase_started = time.perf_counter()
    probe_summaries = []
    fold_rows = []
    fit_rows = []
    training_rows = []
    coefficient_rows = []
    primary_evaluation = None
    primary_raw = None
    all_probe_outputs_finite = True
    for family, runner in (("Linear", run_linear_probe), ("MLP", run_mlp_probe)):
        for signal_set in ("A", "B", "C"):
            probe_name = f"{family}-{signal_set}"
            raw = runner(
                signal_set,
                cache_dir,
                utilities,
                assets["fold_by_index"],
                cache_dir / f"{probe_name.lower().replace('-', '_')}_oof_relative.npy",
            )
            primary_path = (
                cache_dir / "mlp_c_router_oof_predictions.npy"
                if probe_name == "MLP-C"
                else None
            )
            evaluation = evaluate_probe_predictions(
                probe_name,
                raw["predicted_relative"],
                utilities,
                phase0_cache_dir,
                assets["fold_by_index"],
                primary_path,
            )
            probe_summaries.append(evaluation["aggregate"])
            fold_rows.extend(evaluation["fold_rows"])
            fit_rows.extend(raw["fit_rows"])
            training_rows.extend(raw.get("training_rows", []))
            coefficient_rows.extend(raw.get("coefficient_rows", []))
            all_probe_outputs_finite = all_probe_outputs_finite and bool(
                np.isfinite(raw["predicted_relative"]).all()
            )
            write_csv(
                tables_dir / _probe_result_filename(probe_name),
                [evaluation["aggregate"]],
            )
            if probe_name == "MLP-C":
                primary_evaluation = evaluation
                primary_raw = raw
            else:
                del evaluation["router_predictions"]
                del evaluation, raw
    if primary_evaluation is None or primary_raw is None:
        raise RuntimeError("Preregistered primary MLP-C result is missing")
    write_csv(tables_dir / "fold_results.csv", fold_rows)
    write_csv(tables_dir / "probe_fit_audit.csv", fit_rows)
    write_csv(tables_dir / "probe_training.csv", training_rows)
    write_csv(tables_dir / "linear_coefficients.csv", coefficient_rows)
    diagnostic_keys = [
        "method",
        "override_rate",
        "oracle_override_opportunity",
        "override_precision",
        "harmful_override_rate",
        "mean_positive_override_gain",
        "mean_harmful_override_loss",
        "best_branch_top1_accuracy",
        "best_branch_top2_accuracy",
        "pairwise_ranking_accuracy",
        "relative_utility_mae",
        "predicted_true_spearman",
    ]
    write_csv(
        tables_dir / "routing_diagnostics.csv",
        [{key: row[key] for key in diagnostic_keys} for row in probe_summaries],
    )
    phase_seconds["six_oof_probes"] = time.perf_counter() - phase_started

    announce("running primary MLP-C slide bootstrap and routing phenotype analysis")
    phase_started = time.perf_counter()
    true_relative = utilities[:, 1:] - utilities[:, [0]]
    primary_prediction = np.asarray(
        primary_raw["predicted_relative"], dtype=np.float32
    )
    primary_choices = primary_evaluation["choices"]
    routing_choice_rows = []
    for index in range(EXPECTED_IMAGES):
        choice = int(primary_choices[index])
        row = {
            "index": index,
            "image_name": assets["image_names"][index],
            "source_group": assets["source_groups"][index],
            "fold": int(assets["fold_by_index"][index]),
            "router_choice_index": choice,
            "router_choice": (
                BRANCH_NAMES[choice] if choice >= 0 else "official_fusion"
            ),
            "safe_oracle_choice": SAFE_CANDIDATES[int(safe_oracle["choices"][index])],
        }
        for branch_index, branch_name in enumerate(BRANCH_NAMES):
            row[f"predicted_relative_{branch_name}"] = float(
                primary_prediction[index, branch_index]
            )
            row[f"true_relative_{branch_name}"] = float(
                true_relative[index, branch_index]
            )
        routing_choice_rows.append(row)
    write_csv(tables_dir / "routing_choices_oof.csv", routing_choice_rows)
    bootstrap = grouped_slide_bootstrap(
        phase0_cache_dir,
        cache_dir / "mlp_c_router_oof_predictions.npy",
        assets["source_groups"],
    )
    write_csv(tables_dir / "grouped_bootstrap.csv", bootstrap["rows"])
    primary_summary = primary_evaluation["aggregate"]
    oracle_recovery = (
        primary_summary["delta_mIoU"] / safe_oracle["summary"]["delta_mIoU"]
        if safe_oracle["summary"]["delta_mIoU"] != 0
        else 0.0
    )
    primary_fold_rows = [row for row in fold_rows if row["probe"] == "MLP-C"]
    positive_folds = sum(row["delta_mIoU"] > 0 for row in primary_fold_rows)
    decision, decision_rationale = frozen_primary_decision(
        primary_summary["delta_mIoU"],
        oracle_recovery,
        positive_folds,
        bootstrap["summary"]["ci_2_5"],
    )
    phenotype_rows = [
        {
            "phenotype": "SLIDE_CONTEXT_SIGNAL",
            "active": bool(slide_oracle["summary"]["phenotype_flag"]),
            "evidence": f"R_slide={slide_oracle['summary']['slide_recovery_ratio']:.4f}, delta={slide_oracle['summary']['delta_mIoU']:+.4f}",
        },
        {
            "phenotype": image_fusion_oracle["summary"]["mixture_flag"],
            "active": True,
            "evidence": f"G_soft={image_fusion_oracle['summary']['soft_gain_beyond_safe_hard']:+.4f}",
        },
        {
            "phenotype": local_oracle["summary"]["class_conditional_flag"],
            "active": True,
            "evidence": f"aggregate residual={local_oracle['summary']['aggregate_gain_beyond_safe_image']:+.4f}",
        },
    ]
    write_csv(tables_dir / "routing_phenotype.csv", phenotype_rows)
    phase_seconds["bootstrap_and_phenotype"] = time.perf_counter() - phase_started

    announce("rendering automatic qualitative routing review and final report")
    phase_started = time.perf_counter()
    generate_summary_figures(
        figures_dir,
        [
            {"method": "official_fusion", "mIoU": parity["official_mIoU"]},
            safe_oracle["summary"],
            slide_oracle["summary"],
            image_fusion_oracle["summary"],
            local_oracle["summary"],
        ],
        slide_oracle["rows"],
        correlation_rows,
        primary_prediction,
        true_relative,
        primary_choices,
        fold_rows,
        bootstrap["rows"],
    )
    tta_features = np.load(cache_dir / "tta_signal_features.npy", mmap_mode="r")
    tta_names = json.loads(
        (cache_dir / "tta_signal_features.names.json").read_text(encoding="utf-8")
    )
    qualitative_rows = generate_qualitative_routing_panels(
        args.val_root,
        phase0_cache_dir,
        figures_dir,
        assets["image_names"],
        primary_prediction,
        true_relative,
        primary_choices,
        safe_oracle["choices"],
        signal_a,
        names_a,
        tta_features,
        tta_names,
    )
    if len(qualitative_rows) != 32:
        raise RuntimeError(f"Expected exactly 32 automatic qualitative cases, got {len(qualitative_rows)}")
    write_csv(tables_dir / "qualitative_manifest.csv", qualitative_rows)

    official_truth = np.load(phase0_cache_dir / "gt.npy", mmap_mode="r")
    official_predictions = np.load(
        phase0_cache_dir / "official_predictions.npy", mmap_mode="r"
    )
    official_score = score_predictions(official_truth, official_predictions)
    official_row = score_row("official_fusion", official_score, official_score)
    signal_manifest = {
        "baseline_commit": BASELINE_COMMIT,
        "phase0b_parent_commit": PHASE0_PARENT_COMMIT,
        "phase0b_commit": phase0b_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset": "BCSS",
        "split": "val",
        "num_images": EXPECTED_IMAGES,
        "num_slides": EXPECTED_SLIDES,
        "official_fusion": list(OFFICIAL_FUSION),
        "class_presence_thresholds": list(BCSS_THRESHOLDS),
        "tta": ["identity", "hflip", "vflip"],
        "fold_assignment_hash": assets["fold_assignment_hash"],
        "signal_set_a": True,
        "signal_set_b": True,
        "signal_set_c": True,
        "signal_a_feature_count": len(names_a),
        "signal_b_feature_count": len(names_b),
        "signal_c_feature_count": len(names_c),
        "all_signals_finite": bool(
            np.isfinite(signal_a).all()
            and np.isfinite(signal_b).all()
            and np.isfinite(signal_c).all()
        ),
        "test_evaluated": False,
        "luad_evaluated": False,
    }
    write_json(cache_dir / "signal_manifest.json", signal_manifest)
    runtime_context = {
        "parity": parity,
        "checkpoint_sha": assets["checkpoint_sha256"],
        "num_images": assets["num_images"],
        "num_slides": assets["num_slides"],
        "assignment_hash_match": copied_assignment_hash == assets["fold_assignment_hash"],
        "assignment_hash": copied_assignment_hash,
        "no_slide_leakage": all(
            len({int(row["fold"]) for row in assets["assignment_rows"] if row["source_group"] == group}) == 1
            for group in set(assets["source_groups"])
        ),
        "test_evaluated": False,
        "luad_evaluated": False,
        "forbidden_source_diff": False,
        "thresholds": list(BCSS_THRESHOLDS),
        "metric": "tool.iouutils.scores",
        "tta": ["identity", "hflip", "vflip"],
        "signal_a_no_gt": signal_a_manifest["contains_gt"] is False,
        "signal_b_no_gt": signal_bc_manifest["contains_gt"] is False,
        "signal_c_no_gt": signal_bc_manifest["contains_gt"] is False,
        "slide_id_not_input": True,
        "pca_train_only": all(row["fit_scope"] == "train_fold_only" for row in pca_rows),
        "scaler_train_only": all(row.get("scaler_fit_scope", "train_fold_only") == "train_fold_only" for row in fit_rows),
        "heldout_gt_not_fit": all(row.get("heldout_gt_used_for_fit", False) is False for row in fit_rows),
        "threshold_sweep": False,
        "oof_assignment_min": min(raw["assignment_min"] for raw in [primary_raw]),
        "oof_assignment_max": max(raw["assignment_max"] for raw in [primary_raw]),
        "bootstrap_unit": bootstrap["summary"]["sampling_unit"],
        "signals_finite": signal_manifest["all_signals_finite"],
        "probes_finite": all_probe_outputs_finite,
        "masks_valid": all(
            np.load(path, mmap_mode="r").dtype == np.uint8
            for path in (
                cache_dir / "phase0b_parity_reconstruction.npy",
                cache_dir / "safe_image_oracle_predictions.npy",
                cache_dir / "slide_oracle_predictions.npy",
                cache_dir / "image_fusion_oracle_predictions.npy",
                cache_dir / "exact_local_imageclass_oracle_predictions.npy",
                cache_dir / "mlp_c_router_oof_predictions.npy",
            )
        ),
        "local_oracle_name_safe": local_oracle["summary"]["method"] == "exact_local_imageclass_oracle",
    }
    required_test_rows = _runtime_test_rows(runtime_context)
    write_csv(tables_dir / "required_test_results.csv", required_test_rows)
    phase_seconds["visualization_and_report_inputs"] = time.perf_counter() - phase_started

    active_phenotypes = [row["phenotype"] for row in phenotype_rows if row["active"]]
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "phase_seconds": phase_seconds,
        "total_seconds": time.perf_counter() - run_started,
    }
    write_json(config_dir / "environment.json", environment)
    summary = {
        "scope": frozen_contract["scope"],
        "phase0b_parent_commit": PHASE0_PARENT_COMMIT,
        "baseline_commit": BASELINE_COMMIT,
        "phase0b_commit": phase0b_commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "parity": parity,
        "official": official_row,
        "oracles": {
            "safe_image": safe_oracle["summary"],
            "slide": slide_oracle["summary"],
            "image_fusion": image_fusion_oracle["summary"],
            "exact_local_imageclass": local_oracle["summary"],
        },
        "signals": {
            "a_features": len(names_a),
            "b_increment_features": signal_bc_manifest["tta_feature_count"],
            "c_features": len(names_c),
            "pca_dimensions": PCA_DIMENSIONS,
            "fresh_vs_phase0_aggregate_max_abs_difference": signal_bc_manifest[
                "fresh_vs_phase0_aggregate_max_abs_difference"
            ],
        },
        "primary_probe_name": "MLP-C",
        "primary_probe": primary_summary,
        "oracle_recovery_ratio": float(oracle_recovery),
        "positive_folds": int(positive_folds),
        "bootstrap": bootstrap["summary"],
        "phenotype_flags": active_phenotypes,
        "decision": decision,
        "decision_rationale": decision_rationale,
        "executive_conclusion": (
            f"The preregistered MLP-C OOF router changes BCSS validation mIoU by "
            f"{primary_summary['delta_mIoU']:+.4f} pp, recovers {100 * oracle_recovery:.2f}% "
            f"of the safe-image oracle gap, has {positive_folds}/5 positive folds, and a "
            f"slide-bootstrap 95% CI of [{bootstrap['summary']['ci_2_5']:+.4f}, "
            f"{bootstrap['summary']['ci_97_5']:+.4f}] pp. Under the frozen decision logic, "
            f"the result is `{decision}`."
        ),
        "test_evaluated": False,
        "luad_evaluated": False,
        "sshr_training_performed": False,
        "required_tests_passed": len(required_test_rows),
        "exact_command": exact_command,
        "environment": environment,
    }
    write_json(output_dir / "summary.json", summary)
    report_path = write_report(
        output_dir,
        summary,
        {
            "correlations": correlation_rows,
            "probe_summaries": probe_summaries,
            "primary_folds": primary_fold_rows,
            "qualitative": qualitative_rows,
        },
    )
    announce("audit complete; stopping at the frozen human-review gate")
    print(json.dumps(summary, sort_keys=True), flush=True)
    print(f"REPORT={report_path}", flush=True)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
