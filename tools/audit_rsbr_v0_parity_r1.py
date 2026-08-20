#!/usr/bin/env python3
"""Corrected two-layer RSBR-v0 parity audit; evaluation only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
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
from tools.rsbr_parity_r1_contract import (
    MIOU_ALLOWANCE_PP,
    MODEL_IDENTITY_NOGO,
    PARITY_PASS,
    PIXEL_ALLOWANCE,
    decide_parity_r1,
)


SEED = 42
N_CLASS = 4
EXPECTED_VAL = 3418
FIXED_SUBSET = 32
A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
FROZEN_MODEL_HASHES = {
    "network/rsbr_v0.py": "b13ff51e0b73816fa3ffbf241764f2f50bfcda5d2de39951f165cf86a2e0a80a",
    "network/resnet38_cls_rsbr.py": "6af680e5be3b509ed4ef87d48e118e050fa1445b6a87234c657f88fb3ddf2765",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audit-commit", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=("a0", "rsbr"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path):
    """Hash source text after canonical LF newline conversion."""

    text = Path(path).read_text(encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def configure_production():
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


def environment_record():
    try:
        matmul_mode = torch.backends.cuda.matmul.fp32_precision
        convolution_mode = torch.backends.cudnn.conv.fp32_precision
    except AttributeError:
        matmul_mode = bool(torch.backends.cuda.matmul.allow_tf32)
        convolution_mode = bool(torch.backends.cudnn.allow_tf32)
    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "autocast_dtype": "bf16",
        "tf32_matmul": str(matmul_mode),
        "tf32_convolution": str(convolution_mode),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def production_flags_ok(record):
    return (
        record["autocast_dtype"] == "bf16"
        and record["cudnn_benchmark"] is True
        and record["cudnn_deterministic"] is False
        and record["deterministic_algorithms"] is False
        and record["tf32_matmul"] == "tf32"
        and record["tf32_convolution"] == "tf32"
    )


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_model(name, checkpoint):
    state = load_state(checkpoint)
    if name == "a0":
        model = A0NetCAM(n_class=N_CLASS)
        model.load_state_dict(state, strict=True)
        missing = []
    else:
        model = RSBRNetCAM(n_class=N_CLASS)
        incompat = model.load_state_dict(state, strict=False)
        expected = {key for key in model.state_dict() if key.startswith("rsbr.")}
        if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
            raise RuntimeError({
                "missing": incompat.missing_keys,
                "unexpected": incompat.unexpected_keys,
                "expected": sorted(expected),
            })
        missing = sorted(expected)
    model.eval()
    return model.cuda(), missing


def metric_record(metrics):
    return {
        "mIoU": float(metrics["Mean IoU"]),
        "mDice": float(metrics["Mean Dice"]),
        "class_iou": {str(k): float(v) for k, v in metrics["Class IoU"].items()},
        "class_dice": {
            str(k): float(v) for k, v in metrics["Dice Coefficients"].items()
        },
    }


def fixed_image_ids(val_root):
    names = sorted(
        path.stem for path in (Path(val_root) / "img").iterdir()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if len(names) != EXPECTED_VAL:
        raise RuntimeError(f"BCSS validation count {len(names)} != {EXPECTED_VAL}")
    return names[:FIXED_SUBSET]


def find_image_path(val_root, name):
    for extension in (".png", ".jpg", ".jpeg"):
        candidate = Path(val_root) / "img" / f"{name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def image_tensor(path):
    image = Image.open(path).convert("RGB")
    original = np.asarray(image)
    if image.size != (224, 224):
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
    tensor = TF.normalize(
        TF.to_tensor(image),
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return original, tensor[None].cuda()


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
    return infer_utils.cam_npy_to_label_map(cam_score).astype(np.uint8), label


def same_process_identity(args, output):
    configure_production()
    model, missing = load_model("rsbr", args.checkpoint)
    names = fixed_image_ids(args.val_root)
    rows = []
    max_cam_difference = 0.0
    differing_predictions = 0
    delta_core_exact_zero = True
    delta_transition_exact_zero = True
    with torch.no_grad():
        for name in names:
            image, tensor = image_tensor(find_image_path(args.val_root, name))
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
            for base, (_, flip_dims) in zip(bases, _tta_transforms()):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model.refine_from_base(base, presence)
                difference = float((base[1] - result.refined_cam).float().abs().max().item())
                image_max = max(image_max, difference)
                max_cam_difference = max(max_cam_difference, difference)
                delta_core_exact_zero &= torch.count_nonzero(result.delta_core).item() == 0
                delta_transition_exact_zero &= torch.count_nonzero(result.delta_transition).item() == 0
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
            base_prediction, _ = prediction_from_cams(
                [torch.stack(items).mean(0) for items in base_tta], probability, image
            )
            refined_prediction, _ = prediction_from_cams(
                [torch.stack(items).mean(0) for items in refined_tta], probability, image
            )
            pixel_difference = int(np.count_nonzero(base_prediction != refined_prediction))
            differing_predictions += pixel_difference
            rows.append({
                "image_id": name,
                "maximum_cam_difference": image_max,
                "differing_prediction_pixels": pixel_difference,
            })
    environment = environment_record()
    payload = {
        "image_count": len(names),
        "image_ids": names,
        "maximum_cam_difference": max_cam_difference,
        "delta_core_exact_zero": bool(delta_core_exact_zero),
        "delta_transition_exact_zero": bool(delta_transition_exact_zero),
        "differing_prediction_pixels": differing_predictions,
        "missing_keys_expected_rsbr_only": missing,
        "production_flags_unchanged": production_flags_ok(environment),
        "environment": environment,
        "rows": rows,
    }
    payload["pass"] = decide_parity_r1(
        max_cam_difference=max_cam_difference,
        delta_core_exact_zero=bool(delta_core_exact_zero),
        delta_transition_exact_zero=bool(delta_transition_exact_zero),
        same_process_prediction_differences=differing_predictions,
        production_miou_difference_pp=0.0,
        production_prediction_differences=0,
    ) == PARITY_PASS
    write_json(output / "same_process_identity" / "summary.json", payload)
    return payload


def official_capture(model, val_root, args):
    captured = {}
    original = infer_fun.iouutils.scores

    def capture(ground_truth, predictions, n_class):
        captured["predictions"] = np.stack([item.copy() for item in predictions])
        return original(ground_truth, predictions, n_class)

    infer_fun.iouutils.scores = capture
    runtime_args = SimpleNamespace(
        dataset="bcss", img_size=224, num_workers=args.num_workers, amp_dtype="bf16"
    )
    try:
        metrics = infer_fun.infer(
            model, args.val_root, 4, runtime_args,
            thr=None, cam_weights=(0.6, 0.2, 0.2),
        )
    finally:
        infer_fun.iouutils.scores = original
    dataset = Stage1_InferDataset(str(Path(val_root) / "img"), img_size=224)
    names = [Path(path).stem for path in dataset.object]
    if metrics is None or "predictions" not in captured or len(names) != EXPECTED_VAL:
        raise RuntimeError("Official A0 validation capture failed")
    return names, captured["predictions"], metric_record(metrics)


def rsbr_capture(model, val_root, args):
    runtime_args = SimpleNamespace(
        dataset="bcss", img_size=224, num_workers=args.num_workers, amp_dtype="bf16"
    )
    result = infer_rsbr_validation(model, val_root, runtime_args)
    return result.names, result.predictions, metric_record(result.metrics)


def worker_main(args):
    output = Path(args.worker_output)
    output.mkdir(parents=True, exist_ok=False)
    configure_production()
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    model, missing = load_model(args.model, args.checkpoint)
    if args.model == "a0":
        names, predictions, metrics = official_capture(model, args.val_root, args)
    else:
        names, predictions, metrics = rsbr_capture(model, args.val_root, args)
    environment = environment_record()
    np.savez_compressed(output / "predictions.npz", prediction=predictions)
    write_json(output / "names.json", names)
    write_json(output / "result.json", {
        "model": args.model,
        "count": len(names),
        "prediction_sha256": hashlib.sha256(predictions.tobytes()).hexdigest(),
        "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "metrics": metrics,
        "missing_keys_expected_rsbr_only": missing,
        "environment": environment,
        "production_flags_unchanged": production_flags_ok(environment),
        "runtime_seconds": time.time() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "command": " ".join(sys.argv),
    })


def launch_worker(args, output, model_name):
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--model", model_name,
        "--val-root", args.val_root,
        "--checkpoint", args.checkpoint,
        "--output-dir", args.output_dir,
        "--worker-output", str(output),
        "--audit-commit", args.audit_commit,
        "--num-workers", str(args.num_workers),
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(SEED)
    log_path = output.parent / f"{model_name}.log"
    with open(log_path, "w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


def compare_workers(a0_dir, rsbr_dir):
    a0_names = json.loads((a0_dir / "names.json").read_text(encoding="utf-8"))
    rsbr_names = json.loads((rsbr_dir / "names.json").read_text(encoding="utf-8"))
    if a0_names != rsbr_names:
        raise RuntimeError("A0 and RSBR validation order differs")
    with np.load(a0_dir / "predictions.npz") as archive:
        a0_predictions = archive["prediction"]
    with np.load(rsbr_dir / "predictions.npz") as archive:
        rsbr_predictions = archive["prediction"]
    a0 = json.loads((a0_dir / "result.json").read_text(encoding="utf-8"))
    rsbr = json.loads((rsbr_dir / "result.json").read_text(encoding="utf-8"))
    return {
        "image_count": len(a0_names),
        "differing_prediction_pixels": int(np.count_nonzero(a0_predictions != rsbr_predictions)),
        "mIoU_difference_pp": 100.0 * abs(a0["metrics"]["mIoU"] - rsbr["metrics"]["mIoU"]),
        "mDice_difference_pp": 100.0 * abs(a0["metrics"]["mDice"] - rsbr["metrics"]["mDice"]),
        "maximum_class_iou_difference_pp": 100.0 * max(
            abs(a0["metrics"]["class_iou"][key] - rsbr["metrics"]["class_iou"][key])
            for key in a0["metrics"]["class_iou"]
        ),
        "a0_metrics": a0["metrics"],
        "rsbr_metrics": rsbr["metrics"],
        "a0_prediction_sha256": a0["prediction_sha256"],
        "rsbr_prediction_sha256": rsbr["prediction_sha256"],
        "production_flags_unchanged": (
            a0["production_flags_unchanged"] and rsbr["production_flags_unchanged"]
        ),
        "a0_runtime_seconds": a0["runtime_seconds"],
        "rsbr_runtime_seconds": rsbr["runtime_seconds"],
    }


def validate_args(args):
    combined = " ".join((args.val_root, args.checkpoint, args.output_dir)).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Parity R1 is BCSS validation only")
    val_root = Path(args.val_root)
    if val_root.name.lower() != "val" or not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise ValueError("--val-root must point exactly to BCSS val")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    for name in ("same_process_identity", "production_a0", "production_rsbr", "docs", "config"):
        (output / name).mkdir(parents=True, exist_ok=True)


def write_report(output, summary):
    identity = summary["same_process_identity"]
    production = summary.get("production_comparison")
    if production:
        production_text = f"""
- A0 mIoU / mDice: {100 * production['a0_metrics']['mIoU']:.6f} / {100 * production['a0_metrics']['mDice']:.6f}
- RSBR-zero mIoU / mDice: {100 * production['rsbr_metrics']['mIoU']:.6f} / {100 * production['rsbr_metrics']['mDice']:.6f}
- Cross-process mIoU difference: {production['mIoU_difference_pp']:.8f} pp (allowance {MIOU_ALLOWANCE_PP:.8f} pp)
- Cross-process differing pixels: {production['differing_prediction_pixels']:,} (allowance {PIXEL_ALLOWANCE:,})
- Production flags unchanged: {production['production_flags_unchanged']}
"""
    else:
        production_text = "\nLayer 2 was not run because the hard identity gate failed.\n"
    text = f"""# RSBR-v0 Corrected Parity R1 Audit

## Decision

**{summary['decision']}**

## Layer 1: same-process hard identity

- Fixed validation images: {identity['image_count']}
- Maximum CAM28_1 difference: {identity['maximum_cam_difference']:.3e}
- Delta-core exact zero: {identity['delta_core_exact_zero']}
- Delta-transition exact zero: {identity['delta_transition_exact_zero']}
- Base/refined differing pixels: {identity['differing_prediction_pixels']}
- Production flags unchanged: {identity['production_flags_unchanged']}

## Layer 2: frozen production numerical envelope
{production_text}

## Provenance and scope

- A0 commit: `{A0_COMMIT}`
- Audit commit: `{summary['audit_commit']}`
- A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`
- RSBR model hashes unchanged: {summary['model_source_hashes_unchanged']}
- Evaluation-only audit: true
- Test accessed: false
- LUAD accessed: false

Exact command:

```bash
{summary['command']}
```
"""
    (output / "docs" / "rsbr_v0_parity_r1.md").write_text(text, encoding="utf-8")


def orchestrator_main(args):
    validate_args(args)
    output = Path(args.output_dir)
    actual_hashes = {name: sha256_source(ROOT / name) for name in FROZEN_MODEL_HASHES}
    hashes_ok = actual_hashes == FROZEN_MODEL_HASHES
    if not hashes_ok:
        raise RuntimeError({"model_source_hash_mismatch": actual_hashes})
    write_json(output / "config" / "frozen_contract.json", {
        "seed": SEED,
        "dataset": "BCSS validation",
        "fixed_subset_images": FIXED_SUBSET,
        "a0_commit": A0_COMMIT,
        "audit_commit": args.audit_commit,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "mIoU_allowance_pp": MIOU_ALLOWANCE_PP,
        "pixel_allowance": PIXEL_ALLOWANCE,
        "model_source_hashes": actual_hashes,
        "evaluation_only": True,
        "test_forbidden": True,
        "luad_forbidden": True,
        "command": " ".join(sys.argv),
    })
    identity = same_process_identity(args, output)
    decision = decide_parity_r1(
        max_cam_difference=identity["maximum_cam_difference"],
        delta_core_exact_zero=identity["delta_core_exact_zero"],
        delta_transition_exact_zero=identity["delta_transition_exact_zero"],
        same_process_prediction_differences=identity["differing_prediction_pixels"],
        production_miou_difference_pp=0.0,
        production_prediction_differences=0,
    )
    production = None
    if decision != MODEL_IDENTITY_NOGO:
        print("RSBR_PARITY_R1_LAYER2_A0_START", flush=True)
        launch_worker(args, output / "production_a0" / "run", "a0")
        print("RSBR_PARITY_R1_LAYER2_RSBR_START", flush=True)
        launch_worker(args, output / "production_rsbr" / "run", "rsbr")
        production = compare_workers(
            output / "production_a0" / "run", output / "production_rsbr" / "run"
        )
        decision = decide_parity_r1(
            max_cam_difference=identity["maximum_cam_difference"],
            delta_core_exact_zero=identity["delta_core_exact_zero"],
            delta_transition_exact_zero=identity["delta_transition_exact_zero"],
            same_process_prediction_differences=identity["differing_prediction_pixels"],
            production_miou_difference_pp=production["mIoU_difference_pp"],
            production_prediction_differences=production["differing_prediction_pixels"],
        )
    summary = {
        "decision": decision,
        "a0_commit": A0_COMMIT,
        "audit_commit": args.audit_commit,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_source_hashes": actual_hashes,
        "model_source_hashes_unchanged": hashes_ok,
        "same_process_identity": identity,
        "production_comparison": production,
        "command": " ".join(sys.argv),
        "evaluation_only": True,
        "test_accessed": False,
        "luad_accessed": False,
    }
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    print(decision, json.dumps({
        "identity": identity["pass"],
        "production": production,
    }, sort_keys=True), flush=True)


def main():
    args = parse_args()
    if args.worker:
        worker_main(args)
    else:
        orchestrator_main(args)


if __name__ == "__main__":
    main()
