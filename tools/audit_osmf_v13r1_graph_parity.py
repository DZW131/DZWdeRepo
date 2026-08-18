"""OSMF-v1.3-R1 Phase -1.1 exact BCSS validation parity recheck."""

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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls import Net_CAM as A0NetCAM
from network.resnet38_cls_osmf_v13 import Net_CAM as V13NetCAM
from tool.GenDataset import Stage1_InferDataset, Stage1_TrainDataset
from tools.audit_osmf_phase_minus1 import (
    EXPECTED_CHECKPOINT_SHA256,
    FROZEN_A0_COMMIT,
    load_checkpoint,
    official_inference_with_capture,
    sha256_file,
    tensor_parity,
)
from tools.audit_osmf_v13r1_gradient_gate import _forward_objectives
from train_sshr import seed_worker, set_seed


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
    v13 = V13NetCAM(n_class=4)
    incompatible = v13.load_state_dict(state, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing != EXPECTED_MISSING_KEYS:
        raise AssertionError(f"Unexpected v1.3 missing keys: {sorted(missing)}")
    if unexpected:
        raise AssertionError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
    for key, value in state.items():
        if not torch.equal(value.cpu(), v13.state_dict()[key].cpu()):
            raise AssertionError(f"Frozen A0 key changed while loading: {key}")
    baseline = baseline.cuda()
    v13 = v13.cuda()
    baseline.eval()
    v13.eval()
    return baseline, v13, sorted(missing), sorted(unexpected)


def write_report(output_dir: Path, summary: dict) -> None:
    prediction = summary["prediction_parity"]
    lines = [
        "# OSMF-v1.3-R1 Phase -1.1 Parity Recheck",
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
        f"- OSMF-v1.3-R1 commit: `{summary['osmf_v13r1_commit']}`",
        f"- A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`",
        f"- Exact command: `{summary['exact_command']}`",
        "- Precision: official BF16 with A0 TF32; OSMF identity projections use local IEEE FP32.",
        "",
        "## Tensor parity",
        "",
    ]
    graph = summary["graph_connectivity"]
    lines.extend([
        "## Corrected graph connectivity", "",
        f"- grad(L_struct, p_morph): `{graph['struct_p_morph_grad_norm']:.10g}`",
        f"- grad(L_struct, u_morph): `{graph['struct_u_morph_grad_norm']:.10g}` (expected zero)",
        f"- grad(L_total, p_morph): `{graph['total_p_morph_grad_norm']:.10g}`",
        f"- grad(L_total, u_morph): `{graph['total_u_morph_grad_norm']:.10g}`",
        f"- Graph expectation pass: `{graph['pass']}`", "",
    ])
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
            f"- A0/v1.3 mIoU: {prediction['a0_mIoU']:.10f} / {prediction['v13_mIoU']:.10f}",
            f"- Absolute mIoU difference: {prediction['mIoU_absolute_difference']:.10g}",
            f"- A0/v1.3 mDice: {prediction['a0_mDice']:.10f} / {prediction['v13_mDice']:.10f}",
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
    (output_dir / "osmf_v13r1_graph_parity_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--osmf-v13r1-commit", required=True)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16",))
    return parser.parse_args()


def main():
    args = parse_args()
    if any(token in str(args.val_root).lower() for token in ("test", "luad")):
        raise ValueError("Parity accepts BCSS validation only")
    if any(token in str(args.train_root).lower() for token in ("test", "luad", "val")):
        raise ValueError("Graph audit accepts BCSS training data only")
    if len(args.osmf_v13r1_commit) != 40:
        raise ValueError("--osmf-v13r1-commit must be a full Git SHA")
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
    baseline, v13, missing, unexpected = load_models(args.checkpoint)
    train_dataset = Stage1_TrainDataset(
        data_path=str(args.train_root), dataset="bcss", img_size=224
    )
    train_generator = torch.Generator().manual_seed(20260817)
    train_loader = DataLoader(
        train_dataset, batch_size=20, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=train_generator,
    )
    _, graph_images, graph_labels = next(iter(train_loader))
    graph_images = graph_images.cuda(non_blocking=True)
    graph_labels = graph_labels.cuda(non_blocking=True)
    v13.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        graph_bundle = _forward_objectives(
            v13, graph_images, graph_labels, step=4, force_structural=True
        )
    named = dict(v13.osmf_28_1.named_parameters())
    graph_parameters = (named["p_morph.weight"], named["u_morph.weight"])
    struct_gradients = torch.autograd.grad(
        graph_bundle["struct"], graph_parameters,
        retain_graph=True, allow_unused=True,
    )
    total_gradients = torch.autograd.grad(
        graph_bundle["total"], graph_parameters,
        retain_graph=False, allow_unused=True,
    )
    def gradient_norm(gradient):
        return 0.0 if gradient is None else float(gradient.float().norm().cpu())
    graph_connectivity = {
        "struct_p_morph_grad_norm": gradient_norm(struct_gradients[0]),
        "struct_u_morph_grad_norm": gradient_norm(struct_gradients[1]),
        "total_p_morph_grad_norm": gradient_norm(total_gradients[0]),
        "total_u_morph_grad_norm": gradient_norm(total_gradients[1]),
    }
    graph_connectivity["pass"] = bool(
        graph_connectivity["struct_p_morph_grad_norm"] > 1e-12
        and graph_connectivity["struct_u_morph_grad_norm"] <= 1e-12
        and graph_connectivity["total_p_morph_grad_norm"] > 1e-12
        and graph_connectivity["total_u_morph_grad_norm"] > 1e-12
    )
    v13.eval()
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260817)
    random_image = torch.randn(1, 3, 224, 224, generator=generator, device="cuda")
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    _, real_image = dataset[0]
    real_image = real_image.unsqueeze(0).cuda()
    tensor_results = {
        "random_input": tensor_parity(
            baseline, v13, random_image, torch.bfloat16
        ),
        "real_validation_input": tensor_parity(
            baseline, v13, real_image, torch.bfloat16
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
    v13_score, v13_capture = official_inference_with_capture(
        v13, args.val_root, infer_args
    )
    if not np.array_equal(a0_capture["gt"], v13_capture["gt"]):
        raise AssertionError("Validation GT order changed between parity runs")
    differing_pixels = int(
        np.count_nonzero(a0_capture["prediction"] != v13_capture["prediction"])
    )
    a0_miou, v13_miou = float(a0_score["Mean IoU"]), float(v13_score["Mean IoU"])
    a0_mdice, v13_mdice = float(a0_score["Mean Dice"]), float(v13_score["Mean Dice"])
    miou_diff, mdice_diff = abs(a0_miou - v13_miou), abs(a0_mdice - v13_mdice)
    parity_pass = bool(
        tensor_pass
        and differing_pixels == 0
        and miou_diff < 1e-7
        and mdice_diff < 1e-7
    )
    graph_parity_pass = parity_pass and graph_connectivity["pass"]

    a0_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    v13_parameters = sum(parameter.numel() for parameter in v13.parameters())
    summary = {
        "decision": "OSMF_V13R1_GRAPH_PARITY_PASS" if graph_parity_pass else "OSMF_V13R1_GRAPH_PARITY_NOGO",
        "frozen_a0_commit": FROZEN_A0_COMMIT,
        "osmf_v13r1_commit": args.osmf_v13r1_commit,
        "checkpoint_sha256": checkpoint_sha,
        "exact_command": " ".join(sys.argv),
        "tensor_parity": tensor_results,
        "graph_connectivity": graph_connectivity,
        "prediction_parity": {
            "images": int(a0_capture["prediction"].shape[0]),
            "differing_pixels": differing_pixels,
            "a0_mIoU": a0_miou,
            "v13_mIoU": v13_miou,
            "mIoU_absolute_difference": miou_diff,
            "a0_mDice": a0_mdice,
            "v13_mDice": v13_mdice,
            "mDice_absolute_difference": mdice_diff,
        },
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "parameters": {
            "a0_total": a0_parameters,
            "v13_total": v13_parameters,
            "delta": v13_parameters - a0_parameters,
            "overhead_percent": 100.0
            * (v13_parameters - a0_parameters)
            / a0_parameters,
            "new_trainable_tensors": len(tuple(v13.osmf_28_1.parameters())),
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
    if not graph_parity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


