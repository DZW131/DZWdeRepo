#!/usr/bin/env python3
"""RSBR-v0 Stage -1R deterministic parity-harness audit.

This file contains an orchestrator and an internal inference worker.  Every
full-validation repeat is launched as a fresh Python process.  It performs no
training and never constructs an optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls import Net_CAM as A0NetCAM
from network.resnet38_cls_rsbr import Net_CAM as RSBRNetCAM
from tool import infer_fun, infer_utils, iouutils
from tool.GenDataset import Stage1_InferDataset
from tool.infer_fun import _get_class_thresholds, _tta_transforms
from tool.infer_rsbr_v0 import infer_rsbr_validation


SEED = 42
N_CLASS = 4
EXPECTED_VAL = 3418
A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
SELF_RUNS = 3
FIXED_SUBSET = 32
MODES = ("production", "deterministic", "fp32", "tf32_off")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rsbr-commit", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=("a0", "rsbr"), help=argparse.SUPPRESS)
    parser.add_argument("--scope", choices=("full", "subset"), help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, default="production", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def metric_record(metrics):
    return {
        "mIoU": float(metrics["Mean IoU"]),
        "mDice": float(metrics["Mean Dice"]),
        "class_iou": {str(key): float(value) for key, value in metrics["Class IoU"].items()},
        "class_dice": {
            str(key): float(value) for key, value in metrics["Dice Coefficients"].items()
        },
    }


def configure_mode(mode):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp_dtype = "bf16"
    if mode == "deterministic":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif mode == "fp32":
        amp_dtype = "none"
    elif mode == "tf32_off":
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        except AttributeError:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return amp_dtype


def environment_record(mode, amp_dtype):
    try:
        matmul_mode = torch.backends.cuda.matmul.fp32_precision
        convolution_mode = torch.backends.cudnn.conv.fp32_precision
    except AttributeError:
        matmul_mode = bool(torch.backends.cuda.matmul.allow_tf32)
        convolution_mode = bool(torch.backends.cudnn.allow_tf32)
    return {
        "mode": mode,
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "autocast_dtype": amp_dtype,
        "tf32_matmul": str(matmul_mode),
        "tf32_convolution": str(convolution_mode),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_model(model_name, checkpoint):
    state = load_state(checkpoint)
    if model_name == "a0":
        model = A0NetCAM(n_class=N_CLASS)
        model.load_state_dict(state, strict=True)
    else:
        model = RSBRNetCAM(n_class=N_CLASS)
        incompat = model.load_state_dict(state, strict=False)
        expected = {key for key in model.state_dict() if key.startswith("rsbr.")}
        if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
            raise RuntimeError({
                "missing": incompat.missing_keys,
                "unexpected": incompat.unexpected_keys,
            })
    model.eval()
    return model.cuda()


def fixed_image_ids(val_root):
    names = sorted(
        path.stem for path in (Path(val_root) / "img").iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if len(names) != EXPECTED_VAL:
        raise RuntimeError(f"BCSS validation count {len(names)} != {EXPECTED_VAL}")
    return names[:FIXED_SUBSET]


def image_tensor(path):
    image = Image.open(path).convert("RGB")
    if image.size != (224, 224):
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.normalize(
        TF.to_tensor(image),
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return np.asarray(Image.open(path).convert("RGB")), tensor[None].cuda()


def normalize(cam):
    array = cam.detach().float().cpu().numpy()
    minimum = array.min(axis=(1, 2), keepdims=True)
    maximum = array.max(axis=(1, 2), keepdims=True)
    return (array - minimum) / (maximum - minimum + 1e-8)


def prediction_from_cams(cams, probability, image):
    thresholds = _get_class_thresholds(SimpleNamespace(dataset="bcss"), None, 4)
    label = (probability > thresholds).astype(np.float32)
    if label.sum() == 0:
        label[int(np.argmax(probability))] = 1.0
    fusion = 0.6 * normalize(cams[1]) + 0.2 * normalize(cams[2]) + 0.2 * normalize(cams[3])
    fusion *= label.reshape(4, 1, 1)
    cam_dict = infer_utils.cam_npy_to_cam_dict(fusion, label)
    cam_score, _ = infer_utils.dict2npy(cam_dict, label, image)
    prediction = infer_utils.cam_npy_to_label_map(cam_score).astype(np.uint8)
    return prediction, label


def subset_inference(model_name, model, val_root, amp_dtype):
    names = fixed_image_ids(val_root)
    predictions, ground_truth = [], []
    autocast_dtype = torch.bfloat16 if amp_dtype == "bf16" else None
    with torch.no_grad():
        for name in names:
            image, tensor = image_tensor(Path(val_root) / "img" / f"{name}.png")
            records, probabilities = [], []
            for input_flip_dims, _ in _tta_transforms():
                current = torch.flip(tensor, dims=input_flip_dims) if input_flip_dims else tensor
                with torch.autocast(
                    device_type="cuda", dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    if model_name == "a0":
                        output = model.forward_cam(current)
                        records.append(output[:4])
                        probabilities.append(output[4])
                    else:
                        base = model.forward_cam_base(current)
                        records.append(base)
                        probabilities.append(base[4])
            probability = torch.stack(probabilities).mean(0).float().cpu().numpy()[0]
            thresholds = _get_class_thresholds(SimpleNamespace(dataset="bcss"), None, 4)
            presence_np = (probability > thresholds).astype(np.float32)
            if presence_np.sum() == 0:
                presence_np[int(np.argmax(probability))] = 1.0
            presence = torch.from_numpy(presence_np)[None].cuda()
            tta_cams = [[], [], [], []]
            for record, (_, flip_dims) in zip(records, _tta_transforms()):
                if model_name == "a0":
                    cams = record
                else:
                    with torch.autocast(
                        device_type="cuda", dtype=autocast_dtype,
                        enabled=autocast_dtype is not None,
                    ):
                        result = model.refine_from_base(record, presence)
                        cams = (
                            F.relu(record[0]), F.relu(result.refined_cam),
                            F.relu(record[2]), F.relu(record[3]),
                        )
                for index, cam in enumerate(cams):
                    upsampled = F.interpolate(
                        cam, image.shape[:2], mode="bilinear", align_corners=False
                    )[0]
                    if flip_dims:
                        upsampled = torch.flip(upsampled, dims=flip_dims)
                    tta_cams[index].append(upsampled)
            averaged = [torch.stack(items).mean(0) for items in tta_cams]
            prediction, _ = prediction_from_cams(averaged, probability, image)
            predictions.append(prediction)
            ground_truth.append(np.asarray(
                Image.open(Path(val_root) / "mask" / f"{name}.png"), dtype=np.uint8
            ))
    prediction_array = np.stack(predictions)
    ground_truth_array = np.stack(ground_truth)
    metrics = iouutils.scores(
        [item.copy() for item in ground_truth_array],
        [item.copy() for item in prediction_array], n_class=4,
    )
    return names, prediction_array, metric_record(metrics)


def official_full_capture(model, val_root, args, amp_dtype):
    captured = {}
    original = infer_fun.iouutils.scores

    def capture(ground_truth, predictions, n_class):
        captured["predictions"] = np.stack([item.copy() for item in predictions])
        return original(ground_truth, predictions, n_class)

    infer_fun.iouutils.scores = capture
    runtime_args = SimpleNamespace(
        dataset="bcss", img_size=224, num_workers=args.num_workers,
        amp_dtype=amp_dtype,
    )
    try:
        metrics = infer_fun.infer(
            model, val_root, 4, runtime_args,
            thr=None, cam_weights=(0.6, 0.2, 0.2),
        )
    finally:
        infer_fun.iouutils.scores = original
    dataset = Stage1_InferDataset(str(Path(val_root) / "img"), img_size=224)
    names = [Path(path).stem for path in dataset.object]
    if metrics is None or "predictions" not in captured or len(names) != EXPECTED_VAL:
        raise RuntimeError("Official A0 inference capture failed")
    return names, captured["predictions"], metric_record(metrics)


def rsbr_full_capture(model, val_root, args, amp_dtype):
    runtime_args = SimpleNamespace(
        dataset="bcss", img_size=224, num_workers=args.num_workers,
        amp_dtype=amp_dtype,
    )
    result = infer_rsbr_validation(model, val_root, runtime_args)
    return result.names, result.predictions, metric_record(result.metrics)


def worker_main(args):
    output = Path(args.worker_output)
    output.mkdir(parents=True, exist_ok=False)
    amp_dtype = configure_mode(args.mode)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = load_model(args.model, args.checkpoint)
    if args.scope == "full":
        if args.mode != "production":
            raise ValueError("Full validation repeats are production mode only")
        if args.model == "a0":
            names, predictions, metrics = official_full_capture(
                model, args.val_root, args, amp_dtype
            )
        else:
            names, predictions, metrics = rsbr_full_capture(
                model, args.val_root, args, amp_dtype
            )
    else:
        names, predictions, metrics = subset_inference(
            args.model, model, args.val_root, amp_dtype
        )
    np.savez_compressed(output / "predictions.npz", prediction=predictions)
    write_json(output / "names.json", names)
    payload = {
        "model": args.model,
        "scope": args.scope,
        "mode": args.mode,
        "seed": SEED,
        "count": len(names),
        "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "metrics": metrics,
        "environment": environment_record(args.mode, amp_dtype),
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "command": " ".join(sys.argv),
    }
    write_json(output / "result.json", payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    if value.dtype == torch.bfloat16:
        value = value.view(torch.int16)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def same_process_identity(args, output):
    configure_mode("production")
    model = load_model("rsbr", args.checkpoint)
    names = fixed_image_ids(args.val_root)
    rows, prediction_differences = [], 0
    max_cam_difference = 0.0
    all_zero_core = all_zero_transition = True
    with torch.no_grad():
        for name in names:
            image, tensor = image_tensor(Path(args.val_root) / "img" / f"{name}.png")
            bases, probabilities = [], []
            for input_flip_dims, _ in _tta_transforms():
                current = torch.flip(tensor, dims=input_flip_dims) if input_flip_dims else tensor
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    base = model.forward_cam_base(current)
                bases.append(base)
                probabilities.append(base[4])
            probability = torch.stack(probabilities).mean(0).float().cpu().numpy()[0]
            thresholds = _get_class_thresholds(SimpleNamespace(dataset="bcss"), None, 4)
            presence_np = (probability > thresholds).astype(np.float32)
            if presence_np.sum() == 0:
                presence_np[int(np.argmax(probability))] = 1.0
            presence = torch.from_numpy(presence_np)[None].cuda()
            base_tta, refined_tta = [[], [], [], []], [[], [], [], []]
            image_max = 0.0
            canonical = None
            for branch, (base, (_, flip_dims)) in enumerate(zip(bases, _tta_transforms())):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.refine_from_base(base, presence)
                difference = float((base[1] - result.refined_cam).float().abs().max().item())
                image_max = max(image_max, difference)
                max_cam_difference = max(max_cam_difference, difference)
                all_zero_core = all_zero_core and torch.count_nonzero(result.delta_core).item() == 0
                all_zero_transition = (
                    all_zero_transition
                    and torch.count_nonzero(result.delta_transition).item() == 0
                )
                base_cams = tuple(F.relu(item) for item in base[:4])
                refined_cams = (
                    base_cams[0], F.relu(result.refined_cam), base_cams[2], base_cams[3]
                )
                for index, (base_cam, refined_cam) in enumerate(zip(base_cams, refined_cams)):
                    base_up = F.interpolate(
                        base_cam, image.shape[:2], mode="bilinear", align_corners=False
                    )[0]
                    refined_up = F.interpolate(
                        refined_cam, image.shape[:2], mode="bilinear", align_corners=False
                    )[0]
                    if flip_dims:
                        base_up = torch.flip(base_up, dims=flip_dims)
                        refined_up = torch.flip(refined_up, dims=flip_dims)
                    base_tta[index].append(base_up)
                    refined_tta[index].append(refined_up)
                if branch == 0:
                    canonical = {
                        "base_cam28_1_hash": tensor_hash(base[1]),
                        "delta_core_hash": tensor_hash(result.delta_core),
                        "delta_transition_hash": tensor_hash(result.delta_transition),
                        "refined_cam28_1_hash": tensor_hash(result.refined_cam),
                        "base_dtype": str(base[1].dtype),
                        "refined_dtype": str(result.refined_cam.dtype),
                        "delta_core_dtype": str(result.delta_core.dtype),
                        "delta_transition_dtype": str(result.delta_transition.dtype),
                        "base_contiguous": bool(base[1].is_contiguous()),
                        "refined_contiguous": bool(result.refined_cam.is_contiguous()),
                        "delta_core_contiguous": bool(result.delta_core.is_contiguous()),
                        "delta_transition_contiguous": bool(result.delta_transition.is_contiguous()),
                    }
            base_average = [torch.stack(items).mean(0) for items in base_tta]
            refined_average = [torch.stack(items).mean(0) for items in refined_tta]
            base_prediction, _ = prediction_from_cams(base_average, probability, image)
            refined_prediction, _ = prediction_from_cams(refined_average, probability, image)
            differing = int(np.count_nonzero(base_prediction != refined_prediction))
            prediction_differences += differing
            rows.append({
                "image_id": name,
                "maximum_cam28_1_difference": image_max,
                "differing_prediction_pixels": differing,
                "merge_path": canonical,
            })
    passed = (
        max_cam_difference == 0.0 and prediction_differences == 0
        and all_zero_core and all_zero_transition
    )
    payload = {
        "pass": passed,
        "image_ids": names,
        "image_count": len(names),
        "maximum_cam28_1_difference": max_cam_difference,
        "differing_prediction_pixels": prediction_differences,
        "delta_core_exact_zero": all_zero_core,
        "delta_transition_exact_zero": all_zero_transition,
        "environment": environment_record("production", "bf16"),
        "rows": rows,
    }
    write_json(output / "same_process_identity" / "summary.json", payload)
    return payload


def launch_worker(args, output, model, scope, mode):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--model", model, "--scope", scope, "--mode", mode,
        "--val-root", args.val_root, "--checkpoint", args.checkpoint,
        "--output-dir", args.output_dir, "--worker-output", str(output),
        "--rsbr-commit", args.rsbr_commit,
        "--num-workers", str(args.num_workers),
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(SEED)
    log_path = output.parent / f"{output.name}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


def load_prediction(directory):
    with np.load(directory / "predictions.npz") as archive:
        return archive["prediction"]


def comparison(left_dir, right_dir, label):
    left_result = json.loads((left_dir / "result.json").read_text())
    right_result = json.loads((right_dir / "result.json").read_text())
    left_names = json.loads((left_dir / "names.json").read_text())
    right_names = json.loads((right_dir / "names.json").read_text())
    if left_names != right_names:
        raise RuntimeError(f"DataLoader/order mismatch for {label}")
    left_prediction, right_prediction = load_prediction(left_dir), load_prediction(right_dir)
    differing = int(np.count_nonzero(left_prediction != right_prediction))
    total = int(left_prediction.size)
    left_metrics, right_metrics = left_result["metrics"], right_result["metrics"]
    return {
        "comparison": label,
        "differing_pixels": differing,
        "differing_pixel_fraction": differing / float(total),
        "miou_diff_pp": 100.0 * abs(left_metrics["mIoU"] - right_metrics["mIoU"]),
        "mdice_diff_pp": 100.0 * abs(left_metrics["mDice"] - right_metrics["mDice"]),
        "per_class_iou_max_abs_diff_pp": 100.0 * max(
            abs(left_metrics["class_iou"][key] - right_metrics["class_iou"][key])
            for key in left_metrics["class_iou"]
        ),
    }


def envelope(rows):
    return {
        "pixels": max(row["differing_pixels"] for row in rows),
        "miou_pp": max(row["miou_diff_pp"] for row in rows),
        "mdice_pp": max(row["mdice_diff_pp"] for row in rows),
        "per_class_iou_pp": max(row["per_class_iou_max_abs_diff_pp"] for row in rows),
    }


def diagnose_source(mode_rows):
    production = mode_rows["production"]
    deterministic = mode_rows["deterministic"]
    fp32 = mode_rows["fp32"]
    tf32_off = mode_rows["tf32_off"]
    conclusions = []
    if deterministic["maximum_differing_pixels"] < production["maximum_differing_pixels"]:
        conclusions.append("Deterministic algorithms reduce the observed subset discrepancy; CUDA/cuDNN algorithm selection is implicated.")
    if fp32["maximum_differing_pixels"] < production["maximum_differing_pixels"]:
        conclusions.append("Disabling autocast reduces the discrepancy; BF16 rounding/accumulation is implicated.")
    if tf32_off["maximum_differing_pixels"] < production["maximum_differing_pixels"]:
        conclusions.append("Disabling TF32 reduces the discrepancy; TF32 execution is implicated.")
    if not conclusions:
        conclusions.append("The four 32-image modes do not isolate a single source; interpolation/reduction and concurrent execution remain unresolved contributors.")
    conclusions.append("Data order and TTA accumulation order are identical by construction, and the same-process zero residual merge is checked separately.")
    return conclusions


def write_report(output, summary):
    run_rows = []
    for model_name in ("a0", "rsbr"):
        for index, record in enumerate(summary["full_run_results"][model_name], start=1):
            metrics = record["metrics"]
            run_rows.append(
                f"| {model_name.upper()}-{index} | {100 * metrics['mIoU']:.8f} | "
                f"{100 * metrics['mDice']:.8f} | "
                f"{100 * metrics['class_iou']['0']:.8f} | "
                f"{100 * metrics['class_iou']['1']:.8f} | "
                f"{100 * metrics['class_iou']['2']:.8f} | "
                f"{100 * metrics['class_iou']['3']:.8f} | "
                f"{record['runtime_seconds']:.2f} | `{record['prediction_sha256'][:12]}` |"
            )
    table_rows = []
    for row in summary["executive_comparisons"]:
        table_rows.append(
            f"| {row['comparison']} | {row['differing_pixels']:,} | "
            f"{row['miou_diff_pp']:.8f} | {row['mdice_diff_pp']:.8f} | "
            f"{row['per_class_iou_max_abs_diff_pp']:.8f} |"
        )
    diagnostic_rows = []
    for mode, record in summary["diagnostic_modes"].items():
        diagnostic_rows.append(
            f"| {mode} | {record['a0_repeat']['differing_pixels']:,} | "
            f"{record['rsbr_repeat']['differing_pixels']:,} | "
            f"{record['cross']['differing_pixels']:,} | "
            f"{record['maximum_miou_diff_pp']:.8f} |"
        )
    source_lines = "\n".join(f"- {item}" for item in summary["source_diagnosis"])
    text = f"""# RSBR-v0 Stage -1R Deterministic Parity Harness Audit

## 1. Executive conclusion

**{summary['decision']}**

This was an audit-only BCSS validation experiment. No optimizer, training,
Stage 0, three-epoch pilot, test split, LUAD split, threshold change, loss
change, or RSBR model change was used.

## 2. Frozen provenance and environment

- A0 commit: `{summary['a0_commit']}`
- RSBR commit: `{summary['rsbr_commit']}`
- Checkpoint: `{summary['checkpoint']}`
- Checkpoint SHA256: `{summary['checkpoint_sha256']}`
- Environment: `{json.dumps(summary['environment'], sort_keys=True)}`
- Fixed subset: the lexicographically first 32 BCSS validation image IDs,
  frozen in `same_process_identity/summary.json`.

## 3. Same-process structural identity

- Images: {summary['same_process_identity']['image_count']}
- Maximum base/refined CAM28_1 difference: {summary['same_process_identity']['maximum_cam28_1_difference']:.3e}
- Differing base/refined prediction pixels: {summary['same_process_identity']['differing_prediction_pixels']}
- Delta-core exact zero: {summary['same_process_identity']['delta_core_exact_zero']}
- Delta-transition exact zero: {summary['same_process_identity']['delta_transition_exact_zero']}
- Structural identity: **{'PASS' if summary['same_process_identity']['pass'] else 'FAIL'}**

Per-image hashes, dtypes, autocast state, and tensor contiguity are retained in
`same_process_identity/summary.json` and summarized under `merge_path/summary.json`.

## 4. Absolute full-validation results

| Run | mIoU (%) | mDice (%) | C0 IoU | C1 IoU | C2 IoU | C3 IoU | Runtime (s) | Prediction hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(run_rows)}

## 5. Full-validation repeat and cross-model table

| Comparison | Differing Pixels | mIoU Diff (pp) | mDice Diff (pp) | Max class-IoU Diff (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## 6. Numerical envelopes and decision rule

- A0 self envelope: `{json.dumps(summary['a0_self_envelope'], sort_keys=True)}`
- RSBR self envelope: `{json.dumps(summary['rsbr_self_envelope'], sort_keys=True)}`
- Cross-model envelope: `{json.dumps(summary['cross_model_envelope'], sort_keys=True)}`
- Frozen mIoU allowance: {summary['decision_rule']['miou_allowance_pp']:.8f} pp
- Frozen pixel allowance: {summary['decision_rule']['pixel_allowance']:,}
- mIoU rule pass: {summary['decision_rule']['miou_pass']}
- Pixel rule pass: {summary['decision_rule']['pixel_pass']}

## 7. Diagnostic modes on the fixed 32-image subset

| Mode | A0 repeat pixels | RSBR repeat pixels | A0-vs-RSBR pixels | Maximum mIoU diff (pp) |
|---|---:|---:|---:|---:|
{chr(10).join(diagnostic_rows)}

## 8. Source localization

{source_lines}

These mode tests are diagnostic only and do not alter the production SSHR or
RSBR protocol.

## 9. Residual merge path

The canonical branch records show zero-filled `delta_core` and
`delta_transition`, equal base/refined CAM hashes, and the recorded BF16 dtype
and contiguity state. Full per-image records are in
`same_process_identity/summary.json`.

## 10. Commands and artifacts

Exact top-level command:

```bash
{summary['command']}
```

Each independent inference command and stdout/stderr is stored beside its run
directory. Prediction masks are retained as compressed NPZ files so every
pixel-count comparison is reproducible.

## 11. Stop decision

Stage -1R stops here. This report does not authorize Stage 0. Under the frozen
specification, a revised parity harness may be implemented only after human
review of a `{summary['decision']}` result.
"""
    (output / "docs" / "rsbr_v0_stage1r_parity_audit.md").write_text(text, encoding="utf-8")


def validate_orchestrator(args):
    combined = " ".join((args.val_root, args.checkpoint, args.output_dir)).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Stage -1R is BCSS validation only")
    val_root = Path(args.val_root)
    if val_root.name.lower() != "val" or not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise ValueError("--val-root must point exactly to BCSS val")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    for name in (
        "same_process_identity", "a0_self_repeat", "rsbr_self_repeat",
        "cross_model", "diagnostic_modes", "merge_path", "docs", "config",
    ):
        (output / name).mkdir(parents=True, exist_ok=True)


def orchestrator_main(args):
    validate_orchestrator(args)
    output = Path(args.output_dir)
    checkpoint_hash = sha256_file(args.checkpoint)
    write_json(output / "config" / "frozen_contract.json", {
        "scope": "BCSS validation-only audit",
        "seed": SEED,
        "a0_commit": A0_COMMIT,
        "rsbr_commit": args.rsbr_commit,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "full_repeats_per_model": SELF_RUNS,
        "fixed_subset_images": FIXED_SUBSET,
        "diagnostic_modes": list(MODES),
        "training_forbidden": True,
        "test_forbidden": True,
        "luad_forbidden": True,
        "command": " ".join(sys.argv),
    })
    identity = same_process_identity(args, output)
    if not identity["pass"]:
        decision = "RSBR_V0_STAGE1R_MODEL_IDENTITY_NOGO"
        summary = {
            "decision": decision, "a0_commit": A0_COMMIT,
            "rsbr_commit": args.rsbr_commit, "checkpoint": args.checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "same_process_identity": identity,
            "command": " ".join(sys.argv),
        }
        write_json(output / "summary.json", summary)
        print(decision, flush=True)
        return

    full_dirs = {"a0": [], "rsbr": []}
    full_results = {"a0": [], "rsbr": []}
    for model_name, parent in (("a0", "a0_self_repeat"), ("rsbr", "rsbr_self_repeat")):
        for run_index in range(1, SELF_RUNS + 1):
            directory = output / parent / f"run_{run_index}"
            print(f"STAGE1R_FULL_START {model_name} {run_index}/{SELF_RUNS}", flush=True)
            result = launch_worker(args, directory, model_name, "full", "production")
            full_dirs[model_name].append(directory)
            full_results[model_name].append(result)
            print(f"STAGE1R_FULL_DONE {model_name} {run_index}/{SELF_RUNS}", flush=True)

    a0_pairs, rsbr_pairs, cross_pairs = [], [], []
    for left, right in combinations(range(SELF_RUNS), 2):
        a0_pairs.append(comparison(
            full_dirs["a0"][left], full_dirs["a0"][right],
            f"A0-{left + 1} vs A0-{right + 1}",
        ))
        rsbr_pairs.append(comparison(
            full_dirs["rsbr"][left], full_dirs["rsbr"][right],
            f"RSBR-{left + 1} vs RSBR-{right + 1}",
        ))
    for index in range(SELF_RUNS):
        cross_pairs.append(comparison(
            full_dirs["a0"][index], full_dirs["rsbr"][index],
            f"A0-{index + 1} vs RSBR-{index + 1}",
        ))

    a0_envelope, rsbr_envelope, cross_envelope = (
        envelope(a0_pairs), envelope(rsbr_pairs), envelope(cross_pairs)
    )
    self_pixels = max(a0_envelope["pixels"], rsbr_envelope["pixels"])
    miou_allowance = max(a0_envelope["miou_pp"], rsbr_envelope["miou_pp"]) + 0.0005
    pixel_allowance = self_pixels + max(1000, int(np.ceil(0.05 * self_pixels)))
    miou_pass = cross_envelope["miou_pp"] <= miou_allowance
    pixel_pass = cross_envelope["pixels"] <= pixel_allowance

    diagnostic_modes = {}
    for mode in MODES:
        mode_dirs = {"a0": [], "rsbr": []}
        for model_name in ("a0", "rsbr"):
            for repeat in (1, 2):
                directory = output / "diagnostic_modes" / mode / f"{model_name}_{repeat}"
                launch_worker(args, directory, model_name, "subset", mode)
                mode_dirs[model_name].append(directory)
        a0_repeat = comparison(mode_dirs["a0"][0], mode_dirs["a0"][1], f"{mode}: A0 repeat")
        rsbr_repeat = comparison(mode_dirs["rsbr"][0], mode_dirs["rsbr"][1], f"{mode}: RSBR repeat")
        cross = comparison(mode_dirs["a0"][0], mode_dirs["rsbr"][0], f"{mode}: A0 vs RSBR")
        diagnostic_modes[mode] = {
            "a0_repeat": a0_repeat,
            "rsbr_repeat": rsbr_repeat,
            "cross": cross,
            "maximum_differing_pixels": max(
                a0_repeat["differing_pixels"], rsbr_repeat["differing_pixels"],
                cross["differing_pixels"],
            ),
            "maximum_miou_diff_pp": max(
                a0_repeat["miou_diff_pp"], rsbr_repeat["miou_diff_pp"],
                cross["miou_diff_pp"],
            ),
        }

    a0_exact = (
        a0_envelope["pixels"] == 0
        and a0_envelope["miou_pp"] == 0.0
        and a0_envelope["mdice_pp"] == 0.0
    )
    rsbr_or_cross_nonzero = rsbr_envelope["pixels"] > 0 or cross_envelope["pixels"] > 0
    if a0_exact and rsbr_or_cross_nonzero:
        decision = "RSBR_V0_CROSS_RUN_MODEL_PATH_REVIEW"
    elif miou_pass and pixel_pass:
        decision = "RSBR_V0_PARITY_HARNESS_NONDETERMINISM_CONFIRMED"
    else:
        decision = "RSBR_V0_STAGE1R_INCONCLUSIVE"

    merge_rows = [row["merge_path"] for row in identity["rows"]]
    merge_summary = {
        "all_base_refined_hashes_equal": all(
            row["base_cam28_1_hash"] == row["refined_cam28_1_hash"] for row in merge_rows
        ),
        "delta_core_unique_hashes": sorted({row["delta_core_hash"] for row in merge_rows}),
        "delta_transition_unique_hashes": sorted({row["delta_transition_hash"] for row in merge_rows}),
        "dtypes": sorted({
            row[key] for row in merge_rows
            for key in ("base_dtype", "refined_dtype", "delta_core_dtype", "delta_transition_dtype")
        }),
        "all_tensors_contiguous": all(
            row[key] for row in merge_rows
            for key in (
                "base_contiguous", "refined_contiguous", "delta_core_contiguous",
                "delta_transition_contiguous",
            )
        ),
    }
    write_json(output / "merge_path" / "summary.json", merge_summary)
    write_json(output / "a0_self_repeat" / "comparisons.json", a0_pairs)
    write_json(output / "rsbr_self_repeat" / "comparisons.json", rsbr_pairs)
    write_json(output / "cross_model" / "comparisons.json", cross_pairs)
    write_json(output / "diagnostic_modes" / "summary.json", diagnostic_modes)

    summary = {
        "decision": decision,
        "scope": "BCSS validation-only audit; no training",
        "a0_commit": A0_COMMIT,
        "rsbr_commit": args.rsbr_commit,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": checkpoint_hash,
        "environment": full_results["a0"][0]["environment"],
        "same_process_identity": identity,
        "executive_comparisons": a0_pairs + rsbr_pairs + cross_pairs,
        "a0_self_envelope": a0_envelope,
        "rsbr_self_envelope": rsbr_envelope,
        "cross_model_envelope": cross_envelope,
        "decision_rule": {
            "miou_allowance_pp": miou_allowance,
            "pixel_allowance": pixel_allowance,
            "miou_pass": miou_pass,
            "pixel_pass": pixel_pass,
            "a0_self_exact": a0_exact,
        },
        "diagnostic_modes": diagnostic_modes,
        "source_diagnosis": diagnose_source(diagnostic_modes),
        "merge_path": merge_summary,
        "full_run_results": full_results,
        "command": " ".join(sys.argv),
        "training_performed": False,
        "test_evaluated": False,
        "luad_accessed": False,
    }
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    print(decision, json.dumps({
        "a0": a0_envelope, "rsbr": rsbr_envelope,
        "cross": cross_envelope, "rule": summary["decision_rule"],
    }, sort_keys=True), flush=True)


def main():
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        orchestrator_main(args)


if __name__ == "__main__":
    main()
