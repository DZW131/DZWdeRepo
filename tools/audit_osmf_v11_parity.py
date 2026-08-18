"""OSMF-v1.1 Phase -1.1 exact BCSS validation parity recheck."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net_CAM as A0NetCAM
from network.resnet38_cls_osmf_v11 import Net_CAM as V11NetCAM
from tool.GenDataset import Stage1_InferDataset
from tools.audit_osmf_phase_minus1 import (
    EXPECTED_CHECKPOINT_SHA256,
    FROZEN_A0_COMMIT,
    load_checkpoint,
    official_inference_with_capture,
    sha256_file,
    tensor_parity,
)
from train_sshr import set_seed


EXPECTED_MISSING_KEYS = {
    "osmf_28_1.p_sem.weight",
    "osmf_28_1.p_morph.weight",
    "osmf_28_1.u_sem.weight",
    "osmf_28_1.u_morph.weight",
}


def load_models(checkpoint: Path):
    state = load_checkpoint(checkpoint)
    baseline = A0NetCAM(n_class=4)
    baseline.load_state_dict(state, strict=True)
    v11 = V11NetCAM(n_class=4)
    incompatible = v11.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != EXPECTED_MISSING_KEYS:
        raise AssertionError(f"Unexpected v1.1 missing keys: {sorted(missing)}")
    if unexpected:
        raise AssertionError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    for key, value in state.items():
        if not torch.equal(value.cpu(), v11.state_dict()[key].cpu()):
            raise AssertionError(f"Frozen A0 key changed while loading: {key}")
    baseline = baseline.cuda()
    v11 = v11.cuda()
    baseline.eval()
    v11.eval()
    return baseline, v11, sorted(missing), sorted(unexpected)


def write_report(output_dir: Path, summary: dict) -> None:
    prediction = summary["prediction_parity"]
    lines = [
        "# OSMF-v1.1 Phase -1.1 Parity Recheck",
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        "No optimizer or training step was executed.",
        "",
        "## Frozen provenance",
        "",
        f"- Frozen A0 commit: `{summary['frozen_a0_commit']}`",
        f"- OSMF-v1.1 commit: `{summary['osmf_v11_commit']}`",
        f"- A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Exact command: `{summary['exact_command']}`",
        "- Precision: official BF16 with A0 TF32; OSMF identity projections use local IEEE FP32.",
        "",
        "## Tensor parity",
        "",
    ]
    for scope, values in summary["tensor_parity"].items():
        lines.extend((f"### {scope}", ""))
        lines.extend(f"- {key}: `{value:.10g}`" for key, value in values.items())
        lines.append("")
    lines.extend(
        [
            "## Full BCSS validation parity",
            "",
            f"- Images: {prediction['images']}",
            f"- Differing prediction pixels: {prediction['differing_pixels']}",
            f"- A0/v1.1 mIoU: {prediction['a0_mIoU']:.10f} / {prediction['v11_mIoU']:.10f}",
            f"- Absolute mIoU difference: {prediction['mIoU_absolute_difference']:.10g}",
            f"- A0/v1.1 mDice: {prediction['a0_mDice']:.10f} / {prediction['v11_mDice']:.10f}",
            f"- Absolute mDice difference: {prediction['mDice_absolute_difference']:.10g}",
            "",
            "## Parameter compatibility",
            "",
            f"- New trainable tensors: {summary['parameters']['new_trainable_tensors']}",
            f"- Parameter delta: {summary['parameters']['delta']:,}",
            f"- Overhead: {summary['parameters']['overhead_percent']:.6f}%",
            "- Missing keys are exactly p_sem/p_morph/u_sem/u_morph weights.",
            "- No semantic auxiliary classifier exists.",
            "",
            "## Boundary",
            "",
            "Only a PASS authorizes the separate 8-real-batch readiness audit. "
            "No 128-batch audit or Phase 1 was started by this tool.",
            "",
            summary["decision"],
        ]
    )
    (output_dir / "osmf_v11_parity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--osmf-v11-commit", required=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16",))
    return parser.parse_args()


def main():
    args = parse_args()
    if any(token in str(args.val_root).lower() for token in ("test", "luad")):
        raise ValueError("Parity accepts BCSS validation only")
    if len(args.osmf_v11_commit) != 40:
        raise ValueError("--osmf-v11-commit must be a full Git SHA")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f"Unexpected checkpoint SHA256: {checkpoint_sha}")
    if not (args.val_root / "img").is_dir() or not (
        args.val_root / "mask"
    ).is_dir():
        raise FileNotFoundError("BCSS validation root must contain img/ and mask/")

    set_seed(20260817)
    baseline, v11, missing, unexpected = load_models(args.checkpoint)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260817)
    random_image = torch.randn(1, 3, 224, 224, generator=generator, device="cuda")
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    _, real_image = dataset[0]
    real_image = real_image.unsqueeze(0).cuda()
    tensor_results = {
        "random_input": tensor_parity(
            baseline, v11, random_image, torch.bfloat16
        ),
        "real_validation_input": tensor_parity(
            baseline, v11, real_image, torch.bfloat16
        ),
    }
    exact_output_keys = (
        "cam56_max_abs",
        "cam28_1_max_abs",
        "cam28_2_max_abs",
        "camdeep_max_abs",
        "classification_probability_max_abs",
    )
    tensor_pass = all(
        values["osmf_input_vs_reconstruction_max_abs"] < 1e-6
        and values["baseline_vs_osmf_hfrm_max_abs"] == 0.0
        and values["baseline_vs_osmf_reconstruction_max_abs"] == 0.0
        and all(values[key] == 0.0 for key in exact_output_keys)
        for values in tensor_results.values()
    )

    infer_args = SimpleNamespace(
        dataset="bcss", img_size=224, num_workers=args.num_workers, amp_dtype="bf16"
    )
    a0_score, a0_capture = official_inference_with_capture(
        baseline, args.val_root, infer_args
    )
    v11_score, v11_capture = official_inference_with_capture(
        v11, args.val_root, infer_args
    )
    if not np.array_equal(a0_capture["gt"], v11_capture["gt"]):
        raise AssertionError("Validation GT order changed between parity runs")
    differing_pixels = int(
        np.count_nonzero(a0_capture["prediction"] != v11_capture["prediction"])
    )
    a0_miou, v11_miou = float(a0_score["Mean IoU"]), float(v11_score["Mean IoU"])
    a0_mdice, v11_mdice = float(a0_score["Mean Dice"]), float(v11_score["Mean Dice"])
    miou_diff, mdice_diff = abs(a0_miou - v11_miou), abs(a0_mdice - v11_mdice)
    parity_pass = bool(
        tensor_pass
        and differing_pixels == 0
        and miou_diff < 1e-7
        and mdice_diff < 1e-7
    )

    a0_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    v11_parameters = sum(parameter.numel() for parameter in v11.parameters())
    summary = {
        "decision": "OSMF_V11_PARITY_PASS" if parity_pass else "OSMF_V11_PARITY_NOGO",
        "frozen_a0_commit": FROZEN_A0_COMMIT,
        "osmf_v11_commit": args.osmf_v11_commit,
        "checkpoint_sha256": checkpoint_sha,
        "exact_command": " ".join(sys.argv),
        "tensor_parity": tensor_results,
        "prediction_parity": {
            "images": int(a0_capture["prediction"].shape[0]),
            "differing_pixels": differing_pixels,
            "a0_mIoU": a0_miou,
            "v11_mIoU": v11_miou,
            "mIoU_absolute_difference": miou_diff,
            "a0_mDice": a0_mdice,
            "v11_mDice": v11_mdice,
            "mDice_absolute_difference": mdice_diff,
        },
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "parameters": {
            "a0_total": a0_parameters,
            "v11_total": v11_parameters,
            "delta": v11_parameters - a0_parameters,
            "overhead_percent": 100.0
            * (v11_parameters - a0_parameters)
            / a0_parameters,
            "new_trainable_tensors": len(tuple(v11.osmf_28_1.parameters())),
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "training_performed": False,
        "validation_metrics_used_for_selection": False,
        "test_evaluated": False,
        "luad_evaluated": False,
        "readiness_started": False,
        "phase0_started": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "parity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary["prediction_parity"].keys())
        writer.writeheader()
        writer.writerow(summary["prediction_parity"])
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not parity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
