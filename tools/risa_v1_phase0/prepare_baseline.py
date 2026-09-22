"""Seal the already-completed, pre-modification HQMR replay as RISA artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.risa_v1_phase0.common import (
    REFERENCE_CLASS_IOU, REFERENCE_MDICE, REFERENCE_MIOU, sha256, write_csv, write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--hqmr", type=Path, required=True)
    parser.add_argument("--umrf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads((args.replay / "evaluation/hqmr_metrics_all_modes.json").read_text())["A_full"]
    result = json.loads((args.replay / "evaluation/hqmr_final_result.json").read_text())
    dataset = json.loads((args.replay / "provenance/hqmr_dataset.json").read_text())
    config_path = args.replay / "provenance/hqmr_config.json"
    config = json.loads(config_path.read_text())
    frame = pd.read_parquet(args.umrf / "component_evidence_with_gt.parquet")
    hard_m1 = (
        frame.evaluable
        & (frame.sequential5_pred != frame.true_class)
        & (frame.sequential4_pred != frame.true_class)
        & (frame.sequential3_pred != frame.true_class)
    )
    delta = 100 * (metrics["mIoU"] - REFERENCE_MIOU)
    checks = {
        "mIoU_within_0.01pp": abs(delta) <= .01,
        "mDice_exact": abs(metrics["mDice"] - REFERENCE_MDICE) <= 1.e-12,
        "per_class_iou_exact": all(abs(metrics["class_iou"][key] - value) <= 1.e-12 for key, value in REFERENCE_CLASS_IOU.items()),
        "validation_images_3418": result["provenance"]["validation_images"] == 3418,
        "hard_m1_manifest_present": int(hard_m1.sum()) == 5037,
        "m1_manifest_present": int(frame.m1.sum()) == 4402,
    }
    payload = {
        "BASELINE_REPLAY": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed": {"mIoU": metrics["mIoU"], "mDice": metrics["mDice"], "class_iou": metrics["class_iou"]},
        "reference": {"mIoU": REFERENCE_MIOU, "mDice": REFERENCE_MDICE, "class_iou": REFERENCE_CLASS_IOU},
        "delta_mIoU_pp": delta,
        "validation_images": 3418,
        "hard_m1_components": int(hard_m1.sum()),
        "m1_components": int(frame.m1.sum()),
        "git_commit": result["provenance"]["evaluation_source_commit"],
        "training_source_commit": config["source_commit"],
        "git_diff": (args.replay / "provenance/hqmr_git_diff.patch").read_text(errors="replace"),
        "checkpoint_path": str(args.hqmr.resolve()),
        "checkpoint_sha256": sha256(args.hqmr),
        "dataset_manifest_hash": dataset["filename_manifest_sha256"],
        "seed": config["seed"],
        "CUDA_PyTorch_GPU": (args.replay / "provenance/hqmr_environment.txt").read_text(),
        "baseline_config_hash": sha256(config_path),
        "replay_directory": str(args.replay.resolve()),
        "frozen_manifest": str((args.umrf / "component_evidence_with_gt.parquet").resolve()),
    }
    output = args.output / "artifacts/risa_v1_phase0"
    write_json(output / "baseline_replay.json", payload)
    write_csv(output / "baseline_replay_metrics.csv", [{
        "variant": "B0_Frozen_HQMR", "mIoU": metrics["mIoU"], "mDice": metrics["mDice"],
        **{f"IoU_C{key}": value for key, value in metrics["class_iou"].items()},
        "validation_images": 3418, "delta_reference_pp": delta,
    }])
    print(json.dumps({"event": "RISA_BASELINE_SEALED", "status": payload["BASELINE_REPLAY"], "delta_pp": delta}))
    if payload["BASELINE_REPLAY"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
