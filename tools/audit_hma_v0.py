#!/usr/bin/env python3
"""Run the frozen SSHR-HMA-v0 mechanism autopsy (no optimizer, no updates)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tool.GenDataset import Stage1_TrainDataset
from tools.hma_v0 import (
    A0_COMMIT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_TRAIN_IMAGES,
    EXPECTED_VAL_IMAGES,
    EXPECTED_VAL_SLIDES,
)
from tools.hma_v0.evidence import build_mechanism_map
from tools.hma_v0.figures import generate_figures
from tools.hma_v0.gradient_audit import run_gradient_audit
from tools.hma_v0.kernels import audit_context_kernels
from tools.hma_v0.provenance import (
    gamma_autopsy,
    load_model,
    sha256_file,
    source_contract_manifest,
)
from tools.hma_v0.report import render_report
from tools.hma_v0.validation import run_instrumentation_parity, run_validation_audit


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, encoding="utf-8"
    ).strip()


def verify_frozen_source():
    audit_commit = _git("rev-parse", "HEAD")
    if _git("rev-parse", A0_COMMIT) != A0_COMMIT:
        raise AssertionError("Frozen A0 commit is unavailable")
    subprocess.check_call(
        [
            "git", "diff", "--quiet", A0_COMMIT, "--",
            "network", "train_sshr.py", "tool/GenDataset.py",
            "tool/infer_fun.py", "tool/iouutils.py",
        ],
        cwd=REPO_ROOT,
    )
    return audit_commit


def _slide_id(stem):
    parts = stem.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 and stem.startswith("TCGA-") else stem.split("_")[0]


def verify_dataset(val_root, train_root):
    val_root, train_root = Path(val_root), Path(train_root)
    lower_paths = f"{val_root} {train_root}".lower()
    if "test" in lower_paths or "luad" in lower_paths:
        raise AssertionError("HMA-v0 scope forbids test or LUAD paths")
    val_images = sorted((val_root / "img").glob("*.png"))
    val_masks = sorted((val_root / "mask").glob("*.png"))
    if len(val_images) != EXPECTED_VAL_IMAGES or len(val_masks) != EXPECTED_VAL_IMAGES:
        raise AssertionError(
            f"BCSS validation count mismatch: images={len(val_images)}, masks={len(val_masks)}"
        )
    if [p.stem for p in val_images] != [p.stem for p in val_masks]:
        raise AssertionError("BCSS validation image/mask names differ")
    slide_ids = sorted({_slide_id(path.stem) for path in val_images})
    if len(slide_ids) != EXPECTED_VAL_SLIDES:
        raise AssertionError(
            f"Expected {EXPECTED_VAL_SLIDES} BCSS validation slides, inferred {len(slide_ids)}"
        )
    raw_train = sorted(
        path for path in train_root.rglob("*") if path.suffix.lower() in (".png", ".jpg")
    )
    parsed_train = Stage1_TrainDataset(str(train_root), dataset="bcss", img_size=224)
    if len(raw_train) != EXPECTED_TRAIN_IMAGES or len(parsed_train) != EXPECTED_TRAIN_IMAGES:
        raise AssertionError(
            f"BCSS train count mismatch: raw={len(raw_train)}, parsed={len(parsed_train)}"
        )
    return {
        "validation_images": len(val_images),
        "validation_masks": len(val_masks),
        "validation_slides": len(slide_ids),
        "validation_slide_ids": slide_ids,
        "training_files": len(raw_train),
        "training_parsed_samples": len(parsed_train),
    }


def make_output_tree(root):
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {root}")
    directories = (
        "provenance", "parameter_autopsy", "kernels", "gates", "paired_causal",
        "standalone_cam", "error_taxonomy", "inference_decomposition",
        "gradient_audit", "figures", "docs",
    )
    for directory in directories:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=("bf16",), default="bf16")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal HMA-v0 requires CUDA with BF16 support")
    start = time.time()
    output = make_output_tree(args.output_dir)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(
            f"Checkpoint SHA mismatch: {checkpoint_sha} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    audit_commit = verify_frozen_source()
    dataset_manifest = verify_dataset(args.val_root, args.train_root)
    torch.cuda.reset_peak_memory_stats()
    model = load_model(checkpoint)
    source_contract = source_contract_manifest(model, REPO_ROOT)
    gamma = gamma_autopsy(model)
    kernel_rows, kernels = audit_context_kernels(model)

    provenance = {
        "a0_commit": A0_COMMIT,
        "audit_commit": audit_commit,
        "repository": str(REPO_ROOT),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        **dataset_manifest,
        "precision": "BF16 autocast",
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "command": " ".join(sys.argv),
        "scope": {
            "training": False, "optimizer_constructed": False,
            "optimizer_step": False, "test": False, "luad": False,
        },
    }
    write_json(output / "provenance" / "manifest.json", provenance)
    write_json(output / "provenance" / "source_contract.json", source_contract)
    write_json(output / "parameter_autopsy" / "gamma_autopsy.json", gamma)
    pd.DataFrame(kernel_rows).to_csv(output / "kernels" / "kernel_channel_metrics.csv", index=False)
    write_json(output / "kernels" / "kernel_summary.json", kernels)

    # The released ResNet38 overrides train()/eval() and returns None instead
    # of self, so the two calls must not be chained or assigned.
    model = model.cuda()
    model.eval()
    try:
        parity = run_instrumentation_parity(model, args.val_root, amp_dtype=args.amp_dtype)
    except RuntimeError as error:
        parity = error.args[0] if error.args and isinstance(error.args[0], dict) else {"error": str(error)}
        write_json(output / "provenance" / "instrumentation_parity.json", parity)
        (output / "docs" / "sshr_hfrm_mechanism_autopsy.md").write_text(
            "# SSHR-HMA-v0\n\n**SSHR_HMA_INSTRUMENTATION_NOGO**\n\n"
            "The hard same-process parity gate failed; all downstream analyses were stopped.\n",
            encoding="utf-8",
        )
        raise
    write_json(output / "provenance" / "instrumentation_parity.json", parity)

    validation_result = run_validation_audit(
        model, args.val_root, num_workers=args.num_workers, amp_dtype=args.amp_dtype
    )
    validation = validation_result["summary"]
    validation_result["response_frame"].to_csv(output / "gates" / "gsr_response_rows.csv", index=False)
    np.savez_compressed(output / "gates" / "gate_vectors.npz", **validation_result["gate_vectors"])
    pd.DataFrame(validation_result["gate_semantic_rows"]).to_csv(
        output / "gates" / "gate_semantic_separability.csv", index=False
    )
    write_json(output / "gates" / "gate_statistics.json", validation["gate_statistics"])
    write_json(output / "gates" / "gsr_response_summary.json", validation["gsr_response"])
    write_json(output / "paired_causal" / "final_variants.json", validation["final_variants"])
    write_json(output / "paired_causal" / "ch_spatial_effect.json", validation["ch_spatial_effect"])
    write_json(output / "paired_causal" / "complementarity.json", validation["complementarity"])
    write_json(output / "paired_causal" / "present_confusion.json", validation["present_confusion"])
    write_json(output / "standalone_cam" / "standalone_cam.json", validation["standalone_cam"])
    write_json(output / "error_taxonomy" / "error_taxonomy.json", validation["error_taxonomy"])
    write_json(output / "inference_decomposition" / "pipeline.json", {
        "metrics": validation["pipeline_decomposition"],
        "class_gate_contribution": validation["class_gate_contribution"],
        "normalization": validation["gsr_response"]["normalization"],
    })

    gradient_result = run_gradient_audit(
        model, args.train_root, num_workers=args.num_workers, amp_dtype=args.amp_dtype
    )
    pd.DataFrame(gradient_result["rows"]).to_csv(
        output / "gradient_audit" / "gradient_rows.csv", index=False
    )
    pd.DataFrame(gradient_result["component_rows"]).to_csv(
        output / "gradient_audit" / "hfrm_component_rows.csv", index=False
    )
    gradient = gradient_result["summary"]
    write_json(output / "gradient_audit" / "gradient_summary.json", gradient)

    provenance["runtime_seconds"] = float(time.time() - start)
    provenance["peak_cuda_memory_gib"] = float(torch.cuda.max_memory_allocated() / 1024**3)
    summary = {
        "provenance": provenance,
        "source_contract": source_contract,
        "parity": parity,
        "gamma": gamma,
        "kernels": kernels,
        "validation": validation,
        "gradient": gradient,
    }
    summary["mechanism_map"] = build_mechanism_map(summary)
    write_json(output / "provenance" / "manifest.json", provenance)
    write_json(output / "mechanism_map.json", summary["mechanism_map"])
    write_json(output / "hma_v0_summary.json", summary)
    figure_names = generate_figures(
        summary, pd.DataFrame(kernel_rows), pd.DataFrame(gradient_result["rows"]),
        output / "figures",
    )
    summary["provenance"]["figures"] = figure_names
    write_json(output / "hma_v0_summary.json", summary)
    render_report(summary, output / "docs" / "sshr_hfrm_mechanism_autopsy.md")
    print("HFRM_MECHANISM_MAP_COMPLETE", flush=True)
    print(output / "docs" / "sshr_hfrm_mechanism_autopsy.md", flush=True)


if __name__ == "__main__":
    main()
