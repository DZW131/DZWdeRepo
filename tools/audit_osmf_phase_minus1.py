"""OSMF-v1.0 Phase -1 feature and released-inference parity audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net_CAM as A0NetCAM
from network.resnet38_cls_osmf import Net_CAM as OSMFNetCAM
from tool import infer_fun
from tool.GenDataset import Stage1_InferDataset


FROZEN_A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
EXPECTED_CHECKPOINT_SHA256 = (
    "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
)
EXPECTED_OSMF_MISSING_KEYS = {
    "osmf_28_1.p_sem.weight",
    "osmf_28_1.p_morph.weight",
    "osmf_28_1.u_sem.weight",
    "osmf_28_1.u_morph.weight",
    "osmf_28_1.semantic_classifier.weight",
    "osmf_28_1.semantic_classifier.bias",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def load_checkpoint(path: Path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a state_dict mapping")
    return state


def load_models(checkpoint: Path):
    state = load_checkpoint(checkpoint)
    baseline = A0NetCAM(n_class=4)
    baseline.load_state_dict(state, strict=True)
    osmf = OSMFNetCAM(n_class=4)
    incompatible = osmf.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != EXPECTED_OSMF_MISSING_KEYS:
        raise AssertionError(f"Unexpected OSMF missing keys: {sorted(missing)}")
    if unexpected:
        raise AssertionError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    for key, value in state.items():
        if not torch.equal(value.cpu(), osmf.state_dict()[key].cpu()):
            raise AssertionError(f"Frozen A0 key did not load exactly: {key}")
    baseline = baseline.cuda()
    osmf = osmf.cuda()
    # The released ResNet38 ``train`` override returns None, so ``eval`` is not
    # chainable even though it correctly changes module state.
    baseline.eval()
    osmf.eval()
    return baseline, osmf, sorted(missing), sorted(unexpected)


def _capture_hfrm_and_osmf(model, image, amp_dtype, is_osmf):
    captured = {}

    def hfrm_hook(_module, _inputs, output):
        captured["post_hfrm"] = output.detach().float().cpu()

    handles = [model.hfrm_28_1.register_forward_hook(hfrm_hook)]
    if is_osmf:
        def osmf_hook(_module, inputs, output):
            captured["osmf_input"] = inputs[0].detach().float().cpu()
            captured["osmf_reconstruction"] = output[0].detach().float().cpu()

        handles.append(model.osmf_28_1.register_forward_hook(osmf_hook))
    try:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
        ):
            cams = tuple(t.detach().float().cpu() for t in model.forward_cam(image))
    finally:
        for handle in handles:
            handle.remove()
    return captured, cams


def tensor_parity(baseline, osmf, image, amp_dtype):
    base_capture, base_cams = _capture_hfrm_and_osmf(
        baseline, image, amp_dtype, is_osmf=False
    )
    osmf_capture, osmf_cams = _capture_hfrm_and_osmf(
        osmf, image, amp_dtype, is_osmf=True
    )
    metrics = {
        "baseline_vs_osmf_hfrm_max_abs": float(
            (base_capture["post_hfrm"] - osmf_capture["post_hfrm"]).abs().max()
        ),
        "osmf_input_vs_reconstruction_max_abs": float(
            (
                osmf_capture["osmf_input"]
                - osmf_capture["osmf_reconstruction"]
            )
            .abs()
            .max()
        ),
        "baseline_vs_osmf_reconstruction_max_abs": float(
            (
                base_capture["post_hfrm"]
                - osmf_capture["osmf_reconstruction"]
            )
            .abs()
            .max()
        ),
    }
    for index, name in enumerate(
        ("cam56", "cam28_1", "cam28_2", "camdeep", "classification_probability")
    ):
        metrics[f"{name}_max_abs"] = float(
            (base_cams[index] - osmf_cams[index]).abs().max()
        )
    return metrics


def official_inference_with_capture(model, val_root, args):
    captured = {}
    released_scores = infer_fun.iouutils.scores

    def capture_scores(label_trues, label_preds, n_class):
        captured["gt"] = np.asarray(label_trues, dtype=np.uint8)
        captured["prediction"] = np.asarray(label_preds, dtype=np.uint8)
        return released_scores(label_trues, label_preds, n_class=n_class)

    with mock.patch.object(infer_fun.iouutils, "scores", side_effect=capture_scores):
        score = infer_fun.infer(
            model,
            str(val_root),
            n_class=4,
            args=args,
            thr=None,
            cam_weights=(0.6, 0.2, 0.2),
        )
    if score is None or "prediction" not in captured:
        raise RuntimeError("Released inference did not return predictions")
    return score, captured


def write_report(output_dir: Path, summary: dict):
    parity = summary["prediction_parity"]
    lines = [
        "# OSMF-v1.0 Phase -1 Implementation Parity Report",
        "",
        "## 1. Decision",
        "",
        f"Final decision: **{summary['decision']}**.",
        "",
        "This phase implemented OSMF-v1.0 at post-HFRM H28_1 and performed "
        "initialization parity only. No optimization step or SSHR training was run.",
        "",
        "## 2. Frozen contract",
        "",
        f"- Frozen A0 commit: `{summary['frozen_a0_commit']}`.",
        f"- OSMF audit commit: `{summary['osmf_commit']}`.",
        f"- Checkpoint SHA256: `{summary['checkpoint_sha256']}`.",
        "- Dataset/split: BCSS validation only.",
        "- CAM56, CAM28_2, CAMdeep, official fusion, thresholds, TTA, and metric: unchanged.",
        "- Test evaluated: false. LUAD evaluated: false. Training performed: false.",
        f"- Exact command: `{summary['exact_command']}`.",
        "",
        "## 3. Implementation",
        "",
        "- Factorization point: 512-channel post-HFRM H28_1.",
        "- Semantic/morphology channels: 256/256.",
        "- New inference path: P_sem/P_morph then U_sem/U_morph before the original ic1 head.",
        "- Complementary channel-selection/placement initialization reconstructs the identity.",
        "- The semantic auxiliary head and all specialization losses are training-only.",
        "",
        "## 4. Feature and CAM parity",
        "",
    ]
    for scope in ("random_input", "real_validation_input"):
        lines.append(f"### {scope.replace('_', ' ').title()}")
        lines.append("")
        for key, value in summary["tensor_parity"][scope].items():
            lines.append(f"- {key}: `{value:.10g}`")
        lines.append("")
    lines.extend(
        [
            "## 5. Full validation released-inference parity",
            "",
            f"- Images: {parity['images']}.",
            f"- Differing pixels: {parity['differing_pixels']}.",
            f"- A0 mIoU: {parity['a0_mIoU']:.10f}.",
            f"- OSMF-init mIoU: {parity['osmf_mIoU']:.10f}.",
            f"- Absolute mIoU difference: {parity['mIoU_absolute_difference']:.10g}.",
            f"- A0 mDice: {parity['a0_mDice']:.10f}.",
            f"- OSMF-init mDice: {parity['osmf_mDice']:.10f}.",
            f"- Absolute mDice difference: {parity['mDice_absolute_difference']:.10g}.",
            "",
            "## 6. Pretrained compatibility",
            "",
            "The baseline checkpoint loaded strictly into A0. Every frozen A0 tensor "
            "loaded bit-exactly into OSMF. Missing keys were restricted to the newly "
            "introduced factorizer and semantic auxiliary classifier:",
            "",
        ]
    )
    lines.extend(f"- `{key}`" for key in summary["missing_keys"])
    lines.extend(
        [
            "",
            f"Unexpected keys: `{summary['unexpected_keys']}`.",
            "",
            "## 7. Cost at initialization",
            "",
            f"- A0 parameters: {summary['parameters']['a0_total']:,}.",
            f"- OSMF parameters: {summary['parameters']['osmf_total']:,}.",
            f"- Parameter delta: {summary['parameters']['delta']:,} "
            f"({summary['parameters']['overhead_percent']:.4f}%).",
            "",
            "## 8. Phase boundary",
            "",
            "Phase 0 (128 real BCSS training batches) has **not** been started. "
            "The loss weights, 128-batch gradient audit, 3-epoch pilot, and formal "
            "training remain gated by human review.",
            "",
            summary["decision"],
        ]
    )
    (output_dir / "osmf_phase_minus1_readiness_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--osmf-commit",
        default=None,
        help="Exact source commit for immutable archive executions without .git",
    )
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument(
        "--amp-dtype", default="bf16", choices=("none", "bf16", "fp16")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = sha256_file(args.checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise AssertionError(f"Unexpected checkpoint SHA256: {checkpoint_sha}")
    if not (args.val_root / "img").is_dir() or not (args.val_root / "mask").is_dir():
        raise FileNotFoundError("BCSS validation root must contain img/ and mask/")

    amp_dtype = {
        "none": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.amp_dtype]
    baseline, osmf, missing, unexpected = load_models(args.checkpoint)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260817)
    random_image = torch.randn(
        1, 3, 224, 224, generator=generator, device="cuda"
    )
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    _, real_image = dataset[0]
    real_image = real_image.unsqueeze(0).cuda()
    tensor_results = {
        "random_input": tensor_parity(baseline, osmf, random_image, amp_dtype),
        "real_validation_input": tensor_parity(
            baseline, osmf, real_image, amp_dtype
        ),
    }
    max_feature_error = max(
        result["osmf_input_vs_reconstruction_max_abs"]
        for result in tensor_results.values()
    )
    if max_feature_error >= 1e-6:
        raise AssertionError(f"Feature parity failed: {max_feature_error}")

    infer_args = SimpleNamespace(
        dataset="bcss",
        img_size=224,
        num_workers=args.num_workers,
        amp_dtype=args.amp_dtype,
    )
    a0_score, a0_capture = official_inference_with_capture(
        baseline, args.val_root, infer_args
    )
    osmf_score, osmf_capture = official_inference_with_capture(
        osmf, args.val_root, infer_args
    )
    if not np.array_equal(a0_capture["gt"], osmf_capture["gt"]):
        raise AssertionError("Validation GT order changed between parity runs")
    differing_pixels = int(
        np.count_nonzero(
            a0_capture["prediction"] != osmf_capture["prediction"]
        )
    )
    miou_diff = abs(a0_score["Mean IoU"] - osmf_score["Mean IoU"])
    mdice_diff = abs(a0_score["Mean Dice"] - osmf_score["Mean Dice"])
    parity_pass = differing_pixels == 0 and miou_diff < 1e-7 and mdice_diff < 1e-7

    a0_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    osmf_parameters = sum(parameter.numel() for parameter in osmf.parameters())
    osmf_commit = args.osmf_commit or git_commit()
    if len(osmf_commit) != 40 or any(
        character not in "0123456789abcdef" for character in osmf_commit.lower()
    ):
        raise ValueError("--osmf-commit must be a full 40-character Git SHA")
    summary = {
        "scope": "OSMF-v1.0 Phase -1 implementation parity; BCSS validation only",
        "decision": "OSMF_PHASE_MINUS1_PASS" if parity_pass else "OSMF_PHASE_MINUS1_STOP",
        "frozen_a0_commit": FROZEN_A0_COMMIT,
        "osmf_commit": osmf_commit,
        "exact_command": " ".join(sys.argv),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "tensor_parity": tensor_results,
        "prediction_parity": {
            "pass": parity_pass,
            "images": int(a0_capture["prediction"].shape[0]),
            "differing_pixels": differing_pixels,
            "a0_mIoU": a0_score["Mean IoU"],
            "osmf_mIoU": osmf_score["Mean IoU"],
            "mIoU_absolute_difference": miou_diff,
            "a0_mDice": a0_score["Mean Dice"],
            "osmf_mDice": osmf_score["Mean Dice"],
            "mDice_absolute_difference": mdice_diff,
        },
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "parameters": {
            "a0_total": a0_parameters,
            "osmf_total": osmf_parameters,
            "delta": osmf_parameters - a0_parameters,
            "overhead_percent": 100.0
            * (osmf_parameters - a0_parameters)
            / a0_parameters,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "training_performed": False,
        "test_evaluated": False,
        "luad_evaluated": False,
        "phase0_started": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "parity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "images",
                "differing_pixels",
                "a0_mIoU",
                "osmf_mIoU",
                "mIoU_absolute_difference",
                "a0_mDice",
                "osmf_mDice",
                "mDice_absolute_difference",
                "pass",
            ),
        )
        writer.writeheader()
        writer.writerow(summary["prediction_parity"])
    write_report(args.output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not parity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
