"""Run the frozen BCSS validation-only SSHR Decision Bottleneck audit.

The CLI intentionally exposes no test root, threshold, fusion weight,
temperature, oracle mode, probe learning rate, or architecture option.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.decision_audit import (
    BCSS_THRESHOLDS,
    BRANCH_NAMES,
    FOLD_SEED,
    OFFICIAL_FUSION,
)
from tools.decision_audit.calibration import calibration_audit
from tools.decision_audit.cam_cache import (
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_SAMPLES,
    cache_frozen_cams,
    capture_released_inference,
    exact_official_parity,
    sha256_file,
)
from tools.decision_audit.class_probe import (
    PROBE_BATCH_SIZE,
    PROBE_LR,
    PROBE_STEPS,
    run_class_probe,
)
from tools.decision_audit.complementarity import (
    build_branch_predictions,
    class_preference,
    complementarity_tables,
    individual_metrics,
)
from tools.decision_audit.fusion import evaluate_static_grid, simplex_weights
from tools.decision_audit.oracle import run_oracles
from tools.decision_audit.report import write_report
from tools.decision_audit.visualization import (
    generate_qualitative_examples,
    generate_summary_figures,
)


BASE_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(json_ready(value), output, indent=2, sort_keys=True)
        output.write("\n")


def audit_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("DECISION_AUDIT_GIT_COMMIT", "unavailable")


def save_table(path: Path, rows: list[dict]):
    pd.DataFrame(rows).to_csv(path, index=False)


def frozen_decision(
    probe_delta: float,
    image_class_delta: float,
    pixel_delta: float,
    class_dependent_preference: bool,
):
    if (
        probe_delta >= 0.5
        and image_class_delta >= 1.5
        and class_dependent_preference
    ):
        return (
            "DECISION_BOTTLENECK_STRONG_GO",
            "The OOF linear probe, image-class oracle, and class-dependent preference "
            "all meet the frozen Strong-Go criteria.",
        )
    if probe_delta >= 0.3 and image_class_delta >= 1.0:
        return (
            "DECISION_BOTTLENECK_GO",
            "The OOF probe and image-class oracle meet the frozen Go criteria.",
        )
    if image_class_delta >= 1.5 and probe_delta < 0.3:
        return (
            "NONLINEAR_ROUTING_REVIEW",
            "The image-class oracle is large but the frozen 16-scalar OOF probe is weak; "
            "only a human review of low-capacity nonlinear routing is permitted.",
        )
    if image_class_delta < 0.5 and pixel_delta >= 1.5:
        return (
            "SPATIAL_ROUTING_SIGNAL",
            "Image-class routing is weak while the pixel oracle is large, indicating "
            "predominantly spatial complementarity.",
        )
    if probe_delta < 0.1 and image_class_delta < 0.5 and pixel_delta < 1.5:
        return (
            "DECISION_BOTTLENECK_NOGO",
            "The OOF probe, image-class oracle, and pixel oracle all remain below the "
            "frozen decision-bottleneck thresholds.",
        )
    return (
        "NONLINEAR_ROUTING_REVIEW",
        "The result lies in the preregistered grey zone: it does not authorize a model, "
        "but it is not sufficiently negative for Decision-Bottleneck NOGO.",
    )


def executive_conclusion(decision, probe_delta, image_class_delta, pixel_delta):
    return (
        f"Using the same frozen A0 checkpoint, the formal five-fold OOF class probe "
        f"changes mIoU by {probe_delta:+.4f} pp, the image-class oracle by "
        f"{image_class_delta:+.4f} pp, and the pixel oracle by {pixel_delta:+.4f} pp. "
        f"Under the frozen hierarchy, the outcome is `{decision}`. This is a scientific "
        "diagnosis only and does not make any oracle or validation-GT-fitted weight a "
        "deployable weakly supervised method."
    )


def announce_phase(message):
    print(f"[decision-audit] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The official BF16 frozen audit requires CUDA")

    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / "cache"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    docs_dir = output_dir / "docs"
    for directory in (output_dir, cache_dir, tables_dir, figures_dir, docs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    exact_command = (
        f"python -u tools/audit_decision_bottleneck.py \\\n"
        f"  --val-root {args.val_root.resolve()} \\\n"
        f"  --checkpoint {args.checkpoint.resolve()} \\\n"
        f"  --output-dir {output_dir} \\\n"
        f"  --num-workers {args.num_workers}"
    )
    config = {
        "scope": "BCSS validation-only frozen-model decision bottleneck audit",
        "base_commit": BASE_COMMIT,
        "audit_git_commit": audit_git_commit(),
        "validation_root": str(args.val_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
        "expected_images": EXPECTED_SAMPLES,
        "official_fusion": list(OFFICIAL_FUSION),
        "class_presence_thresholds": list(BCSS_THRESHOLDS),
        "static_grid_step": 0.05,
        "probe": {
            "folds": 5,
            "fold_method": "GroupKFold by filename prefix before _xmin",
            "seed": FOLD_SEED,
            "trainable_scalars": 16,
            "optimizer": "Adam",
            "learning_rate": PROBE_LR,
            "steps": PROBE_STEPS,
            "batch_size": PROBE_BATCH_SIZE,
            "weight_decay": 0.0,
        },
        "test_evaluated": False,
        "sshr_training_performed": False,
        "exact_command": exact_command,
    }
    write_json(output_dir / "config.json", config)

    announce_phase("validating frozen inputs and loading A0 checkpoint")
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA mismatch: expected={EXPECTED_CHECKPOINT_SHA256}, "
            f"actual={checkpoint_sha}"
        )
    image_count = len(list((args.val_root / "img").glob("*.png")))
    mask_count = len(list((args.val_root / "mask").glob("*.png")))
    if image_count != EXPECTED_SAMPLES or mask_count != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"BCSS validation count mismatch: images={image_count}, masks={mask_count}"
        )

    state_dict = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model = importlib.import_module("network.resnet38_cls").Net_CAM(n_class=4)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False)
    model.cuda().eval()
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen A0 model must be eval-only with no trainable parameters")
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    phase_seconds = {}

    announce_phase("running released inference for the exact-parity reference")
    released_score, released_capture, phase_seconds["released_inference"] = (
        capture_released_inference(model, args.val_root, args.num_workers)
    )
    announce_phase("caching the four frozen CAM branches")
    phase_started = time.perf_counter()
    cache_manifest = cache_frozen_cams(
        model,
        args.val_root,
        cache_dir,
        args.checkpoint,
        checkpoint_sha,
        config["audit_git_commit"],
        args.num_workers,
    )
    phase_seconds["cam_cache"] = time.perf_counter() - phase_started
    parity = exact_official_parity(released_score, released_capture, cache_dir)
    write_json(output_dir / "parity.json", parity)
    del released_capture
    if not parity["pass"]:
        raise RuntimeError(f"Official inference parity failed; STOP: {parity}")
    announce_phase("exact released-inference parity passed")

    announce_phase("auditing individual branches and complementarity")
    phase_started = time.perf_counter()
    build_branch_predictions(cache_dir)
    individual_rows = individual_metrics(cache_dir)
    complementarity = complementarity_tables(cache_dir)
    class_preference_rows = class_preference(individual_rows)
    phase_seconds["individual_and_complementarity"] = time.perf_counter() - phase_started

    announce_phase("evaluating the frozen 1,771-point static fusion grid")
    phase_started = time.perf_counter()
    weights = simplex_weights(0.05)
    if len(weights) != 1771:
        raise RuntimeError(f"Frozen simplex grid must contain 1771 candidates, got {len(weights)}")
    static_rows = evaluate_static_grid(cache_dir, weights, device="cuda")
    static_top10 = sorted(
        static_rows,
        key=lambda row: (row["mIoU"], row["mDice"]),
        reverse=True,
    )[:10]
    phase_seconds["static_fusion"] = time.perf_counter() - phase_started

    announce_phase("computing frozen diagnostic oracles")
    phase_started = time.perf_counter()
    oracle = run_oracles(cache_dir)
    phase_seconds["oracles"] = time.perf_counter() - phase_started

    announce_phase("training the five-fold out-of-fold 16-scalar diagnostic probe")
    phase_started = time.perf_counter()
    probe = run_class_probe(cache_dir, output_dir, device="cuda")
    phase_seconds["class_probe"] = time.perf_counter() - phase_started

    announce_phase("auditing calibration, confidence, and error geometry")
    phase_started = time.perf_counter()
    calibration = calibration_audit(cache_dir)
    phase_seconds["calibration"] = time.perf_counter() - phase_started

    save_table(tables_dir / "individual_cam_metrics.csv", individual_rows)
    save_table(tables_dir / "global_fusion_sweep.csv", static_rows)
    save_table(tables_dir / "global_fusion_top10.csv", static_top10)
    save_table(tables_dir / "class_preference.csv", class_preference_rows)
    save_table(
        tables_dir / "pairwise_complementarity.csv", complementarity["pairwise"]
    )
    save_table(tables_dir / "unique_correct.csv", complementarity["unique"])
    save_table(
        tables_dir / "official_recoverability.csv",
        complementarity["recoverability"],
    )
    save_table(
        tables_dir / "error_overlap_matrix.csv", complementarity["error_overlap"]
    )
    save_table(tables_dir / "oracle_results.csv", oracle["rows"])
    save_table(
        tables_dir / "best_branch_frequency.csv", oracle["preference_rows"]
    )
    save_table(
        tables_dir / "class_probe_fold_results.csv", probe["fold_rows"]
    )
    save_table(tables_dir / "class_probe_weights.csv", probe["weight_rows"])
    save_table(
        tables_dir / "class_probe_training.csv", probe["training_rows"]
    )
    save_table(
        tables_dir / "class_probe_fold_assignments.csv",
        probe["assignment_rows"],
    )
    save_table(
        tables_dir / "confidence_accuracy.csv", calibration["confidence_rows"]
    )
    save_table(
        tables_dir / "calibration_summary.csv", calibration["summary_rows"]
    )

    announce_phase("rendering deterministic summary and qualitative figures")
    phase_started = time.perf_counter()
    generate_summary_figures(
        figures_dir,
        individual_rows,
        complementarity["error_overlap"],
        oracle["preference_rows"],
        oracle["rows"],
        calibration["confidence_rows"],
    )
    qualitative_rows = generate_qualitative_examples(
        args.val_root, cache_dir, figures_dir
    )
    if len(qualitative_rows) < 24:
        raise RuntimeError(
            "The preregistered qualitative audit requires at least 24 automatically "
            f"ranked cases, but only {len(qualitative_rows)} were available"
        )
    save_table(tables_dir / "qualitative_manifest.csv", qualitative_rows)
    phase_seconds["visualization"] = time.perf_counter() - phase_started

    oracle_by_name = {row["method"]: row for row in oracle["rows"]}
    probe_delta = probe["summary"]["delta_mIoU"]
    image_class_delta = oracle_by_name["image_class_oracle"]["delta_vs_official"]
    pixel_delta = oracle_by_name["pixel_oracle"]["delta_vs_official"]
    class_dependent = len(
        set(row["best_branch"] for row in class_preference_rows)
    ) >= 2
    decision, rationale = frozen_decision(
        probe_delta, image_class_delta, pixel_delta, class_dependent
    )
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
    write_json(output_dir / "environment.json", environment)
    summary = {
        "scope": config["scope"],
        "base_commit": BASE_COMMIT,
        "audit_git_commit": config["audit_git_commit"],
        "checkpoint_sha256": checkpoint_sha,
        "num_images": image_count,
        "num_masks": mask_count,
        "test_evaluated": False,
        "sshr_training_performed": False,
        "cache_manifest": cache_manifest,
        "parity": parity,
        "class_probe": probe["summary"],
        "pixel_oracle": oracle["pixel_summary"],
        "class_dependent_preference": class_dependent,
        "decision": decision,
        "decision_rationale": rationale,
        "executive_conclusion": executive_conclusion(
            decision, probe_delta, image_class_delta, pixel_delta
        ),
        "exact_command": exact_command,
        "environment": environment,
    }
    write_json(output_dir / "summary.json", summary)
    report_tables = {
        "individual": individual_rows,
        "static_top10": static_top10,
        "class_preference": class_preference_rows,
        "unique": complementarity["unique"],
        "recoverability": complementarity["recoverability"],
        "error_overlap": complementarity["error_overlap"],
        "oracle": oracle["rows"],
        "calibration": calibration["summary_rows"],
        "qualitative": qualitative_rows,
    }
    report_path = write_report(output_dir, summary, report_tables)
    announce_phase("audit complete")
    print(json.dumps(json_ready(summary), sort_keys=True), flush=True)
    print(f"REPORT={report_path}", flush=True)
    print(decision, flush=True)


if __name__ == "__main__":
    main()
