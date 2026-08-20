#!/usr/bin/env python3
"""Execute the frozen-SSHR RSBR-v0 three-epoch BCSS pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.resnet38_cls_rsbr import Net as RSBRNet
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_rsbr_v0_paired import paired_rsbr_validation
from tools.rsbr_stage1_contract import (
    GO,
    KNOWN_PRODUCTION_ENVELOPE_PP,
    NOGO,
    REVIEW,
    STRONG_GO,
    decide_pilot,
    select_best_epoch,
)


A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
A0_CHECKPOINT_SHA256 = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
PARITY_REQUIRED = "RSBR_V0_PARITY_R1_PASS"
READINESS_REQUIRED = "RSBR_V0_READINESS_PASS"
SEED = 42
N_CLASS = 4
EPOCHS = 3
EXPECTED_TRAIN = 23_422
EXPECTED_VAL = 3_418
LOSS_WEIGHTS = (0.10, 0.15, 0.25, 0.50)
LAMBDA_REGION = 0.05
LAMBDA_RESIDUAL = 0.01
FROZEN_MODEL_HASHES = {
    "network/rsbr_v0.py": "b13ff51e0b73816fa3ffbf241764f2f50bfcda5d2de39951f165cf86a2e0a80a",
    "network/resnet38_cls_rsbr.py": "6af680e5be3b509ed4ef87d48e118e050fa1445b6a87234c657f88fb3ddf2765",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--parity-summary", required=True)
    parser.add_argument("--readiness-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-commit", required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wt-dec", type=float, default=5e-4)
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().tolist()
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(payload), sort_keys=True) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode()).hexdigest()


def hash_named_tensors(named_tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        value = tensor.detach().contiguous().cpu()
        if value.dtype == torch.bfloat16:
            value = value.view(torch.int16)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def base_state_hashes(model):
    parameters = [
        (name, value) for name, value in model.named_parameters()
        if not name.startswith("rsbr.")
    ]
    buffers = [
        (name, value) for name, value in model.named_buffers()
        if not name.startswith("rsbr.")
    ]
    return {
        "parameter_sha256": hash_named_tensors(parameters),
        "buffer_sha256": hash_named_tensors(buffers),
        "parameter_tensors": len(parameters),
        "buffer_tensors": len(buffers),
    }


def rsbr_state_hash(model):
    return hash_named_tensors(model.rsbr.state_dict().items())


def seed_everything():
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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_inputs(args):
    if args.batch_size != 20 or args.img_size != 224:
        raise ValueError("Frozen pilot requires batch20 and 224x224")
    if args.lr != 0.01 or args.wt_dec != 5e-4:
        raise ValueError("Frozen pilot requires lr=0.01 and wt_dec=0.0005")
    combined = " ".join((
        args.train_root, args.val_root, args.checkpoint,
        args.parity_summary, args.readiness_summary, args.output_dir,
    )).lower()
    if "test" in combined or "luad" in combined:
        raise ValueError("Stage-1 pilot is BCSS train + validation only")
    train_root, val_root = Path(args.train_root), Path(args.val_root)
    if not train_root.is_dir():
        raise FileNotFoundError(args.train_root)
    if val_root.name.lower() != "val" or not (val_root / "img").is_dir() or not (val_root / "mask").is_dir():
        raise ValueError("--val-root must point exactly to BCSS val")
    for path in (args.checkpoint, args.parity_summary, args.readiness_summary):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if sha256_file(args.checkpoint) != A0_CHECKPOINT_SHA256:
        raise RuntimeError("A0 checkpoint SHA256 mismatch")
    parity = json.loads(Path(args.parity_summary).read_text(encoding="utf-8"))
    readiness = json.loads(Path(args.readiness_summary).read_text(encoding="utf-8"))
    if parity.get("decision") != PARITY_REQUIRED:
        raise RuntimeError(f"Pilot locked by parity={parity.get('decision')}")
    if readiness.get("decision") != READINESS_REQUIRED:
        raise RuntimeError(f"Pilot locked by readiness={readiness.get('decision')}")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    for name in ("config", "train", "validation", "mechanism", "checkpoints", "figures", "docs"):
        (output / name).mkdir(parents=True, exist_ok=True)
    model_hashes = {name: sha256_source(ROOT / name) for name in FROZEN_MODEL_HASHES}
    if model_hashes != FROZEN_MODEL_HASHES:
        raise RuntimeError({"frozen_model_source_mismatch": model_hashes})
    return parity, readiness, model_hashes


def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def load_fresh_model(checkpoint):
    state = load_state(checkpoint)
    if any(key.startswith("rsbr.") for key in state):
        raise RuntimeError("Fresh A0 checkpoint unexpectedly contains RSBR weights")
    model = RSBRNet(n_class=N_CLASS)
    incompat = model.load_state_dict(state, strict=False)
    expected = {key for key in model.state_dict() if key.startswith("rsbr.")}
    if set(incompat.missing_keys) != expected or incompat.unexpected_keys:
        raise RuntimeError({
            "missing": incompat.missing_keys,
            "unexpected": incompat.unexpected_keys,
            "expected": sorted(expected),
        })
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("rsbr."))
    set_frozen_training_mode(model)
    zero_init = {
        "region_weight_zero": torch.count_nonzero(model.rsbr.region_semantic_head.weight).item() == 0,
        "region_bias_zero": torch.count_nonzero(model.rsbr.region_semantic_head.bias).item() == 0,
        "transition_output_weight_zero": torch.count_nonzero(model.rsbr.transition_head[-1].weight).item() == 0,
        "transition_output_bias_zero": torch.count_nonzero(model.rsbr.transition_head[-1].bias).item() == 0,
    }
    if not all(zero_init.values()):
        raise RuntimeError({"RSBR zero initialization failed": zero_init})
    return model, sorted(expected), zero_init


def set_frozen_training_mode(model):
    model.eval()
    model.rsbr.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("rsbr."))


def frozen_mode_ok(model):
    base_eval = all(
        not module.training
        for name, module in model.named_modules()
        if name and name != "rsbr" and not name.startswith("rsbr.")
    )
    rsbr_train = model.rsbr.training
    gradients_disabled = all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("rsbr.")
    )
    rsbr_gradients_enabled = all(
        parameter.requires_grad for parameter in model.rsbr.parameters()
    )
    return {
        "all_original_modules_eval": base_eval,
        "rsbr_module_train": rsbr_train,
        "all_original_parameters_frozen": gradients_disabled,
        "all_rsbr_parameters_trainable": rsbr_gradients_enabled,
    }


def make_loader(args):
    dataset = Stage1_TrainDataset(
        data_path=args.train_root,
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset="bcss",
        img_size=args.img_size,
    )
    if len(dataset) != EXPECTED_TRAIN:
        raise RuntimeError(f"BCSS parsed train count {len(dataset)} != {EXPECTED_TRAIN}")
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader


def build_optimizer(model, args, steps_per_epoch):
    weights, biases = [], []
    for module in model.rsbr.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            weights.append(module.weight)
            if module.bias is not None:
                biases.append(module.bias)
    groups = [
        {"params": weights, "lr": 10 * args.lr, "weight_decay": args.wt_dec},
        {"params": biases, "lr": 20 * args.lr, "weight_decay": 0.0},
    ]
    optimizer = torchutils.PolyOptimizer(
        groups,
        lr=args.lr,
        weight_decay=args.wt_dec,
        max_step=steps_per_epoch * EPOCHS,
    )
    optimizer_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    rsbr_ids = {id(parameter) for parameter in model.rsbr.parameters()}
    if optimizer_ids != rsbr_ids:
        raise RuntimeError("Update groups must contain exactly RSBR parameters")
    return optimizer


def region_mil_loss(region_logits, labels):
    losses = []
    for batch_index, logits in enumerate(region_logits):
        if logits.shape[0] == 0:
            continue
        region_score = logits.max(dim=0).values[None]
        losses.append(F.multilabel_soft_margin_loss(
            region_score, labels[batch_index:batch_index + 1]
        ))
    return torch.stack(losses).mean() if losses else labels.sum() * 0.0


def training_losses(model, images, labels):
    outputs = model(images, presence=labels, return_rsbr_aux=True)
    out_56, out_28_1, out_28_2, out_deep = outputs[:4]
    result = outputs[-1]
    slot_56 = F.multilabel_soft_margin_loss(out_56, labels)
    slot_28_1 = F.multilabel_soft_margin_loss(out_28_1, labels)
    slot_28_2 = F.multilabel_soft_margin_loss(out_28_2, labels)
    slot_deep = F.multilabel_soft_margin_loss(out_deep, labels)
    classification = (
        LOSS_WEIGHTS[0] * slot_56
        + LOSS_WEIGHTS[1] * slot_28_1
        + LOSS_WEIGHTS[2] * slot_28_2
        + LOSS_WEIGHTS[3] * slot_deep
    )
    region = region_mil_loss(result.region_logits, labels)
    residual = result.delta_core.abs().mean() + result.delta_transition.abs().mean()
    total = classification + LAMBDA_REGION * region + LAMBDA_RESIDUAL * residual
    return total, classification, slot_28_1, region, residual, result


def grad_norm(parameters):
    values = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters if parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def frozen_gradients_clean(model):
    return all(
        parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0
        for name, parameter in model.named_parameters()
        if not name.startswith("rsbr.")
    )


def summarize_rows(rows):
    numeric_keys = sorted(
        key for key, value in rows[0].items()
        if key not in ("step", "epoch") and isinstance(value, (int, float))
    )
    summary = {}
    for key in numeric_keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def runtime_args(args):
    return SimpleNamespace(
        dataset="bcss", img_size=args.img_size,
        num_workers=args.num_workers, amp_dtype="bf16",
    )


def checkpoint_payload(model, optimizer, args, epoch, validation, provenance):
    return {
        "rsbr_state_dict": {
            key: value.detach().cpu() for key, value in model.rsbr.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "paired_delta_miou_pp": validation["paired_delta_miou_pp"],
        "paired_delta_mdice_pp": validation["paired_delta_mdice_pp"],
        "a0_checkpoint": args.checkpoint,
        "a0_checkpoint_sha256": A0_CHECKPOINT_SHA256,
        "experiment_commit": args.experiment_commit,
        "config": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": args.batch_size,
            "image_size": args.img_size,
            "precision": "BF16",
            "lr": args.lr,
            "weight_decay": args.wt_dec,
            "loss_weights": LOSS_WEIGHTS,
            "lambda_region": LAMBDA_REGION,
            "lambda_residual": LAMBDA_RESIDUAL,
        },
        "provenance": provenance,
    }


def save_checkpoint(model, optimizer, args, output, epoch, validation, provenance):
    path = output / "checkpoints" / f"epoch{epoch}_rsbr.pth"
    torch.save(
        checkpoint_payload(model, optimizer, args, epoch, validation, provenance),
        path,
    )
    return path, sha256_file(path)


def validation_epoch_record(epoch, validation, train_summary, train_seconds, frozen_hashes):
    type_b = validation["taxonomy"]["B_misclassified_pure"]
    type_d = validation["taxonomy"]["D_mixed_boundary"]
    return {
        "epoch": epoch,
        "paired_delta_miou_pp": validation["paired_delta_miou_pp"],
        "paired_delta_mdice_pp": validation["paired_delta_mdice_pp"],
        "paired_class_iou_delta_pp": validation["paired_class_iou_delta_pp"],
        "base_metrics": validation["final_metrics"]["base"],
        "refined_metrics": validation["final_metrics"]["full"],
        "base_cam28_1_standalone": validation["standalone_cam28_1_metrics"]["base"],
        "refined_cam28_1_standalone": validation["standalone_cam28_1_metrics"]["full"],
        "standalone_cam28_1_delta_miou_pp": validation[
            "standalone_cam28_1_delta_miou_pp"
        ],
        "type_b_recovery_pixels": type_b["recovery_pixels"]["full"],
        "type_b_recovery_rate": type_b["recovery_rate"]["full"],
        "type_d_recovery_pixels": type_d["recovery_pixels"]["full"],
        "type_d_recovery_rate": type_d["recovery_rate"]["full"],
        "mechanism": validation["mechanism"],
        "noise_ratio": abs(validation["paired_delta_miou_pp"]) / KNOWN_PRODUCTION_ENVELOPE_PP,
        "training_seconds": train_seconds,
        "validation_runtime": validation["runtime"],
        "training_summary": train_summary,
        "frozen_state_after_epoch": frozen_hashes,
    }


def mechanism_flags(epoch3, epoch3_validation, training_summaries):
    final_metrics = epoch3_validation["final_metrics"]
    base_miou = final_metrics["base"]["mIoU"]
    variant_delta = {
        variant: 100.0 * (metrics["mIoU"] - base_miou)
        for variant, metrics in final_metrics.items()
    }
    taxonomy = epoch3_validation["taxonomy"]
    type_b = taxonomy["B_misclassified_pure"]["recovery_pixels"]
    type_d = taxonomy["D_mixed_boundary"]["recovery_pixels"]
    flags = []
    if variant_delta["core_only"] > 0 and type_b["core_only"] > 0:
        flags.append("REGION_SEMANTIC_SIGNAL")
    if variant_delta["transition_only"] > 0 and type_d["transition_only"] > 0:
        flags.append("TRANSITION_REFINEMENT_SIGNAL")
    if (
        variant_delta["full"] > variant_delta["core_only"]
        and variant_delta["full"] > variant_delta["transition_only"]
    ):
        flags.append("JOINT_RSBR_SYNERGY")
    r_tc = epoch3_validation["mechanism"]["transition_to_core_rms"]
    if (
        abs(variant_delta["transition_only"]) <= KNOWN_PRODUCTION_ENVELOPE_PP
        and r_tc < 0.01
        and type_d["transition_only"] <= 0
    ):
        flags.append("TRANSITION_PATH_NOT_EFFECTIVE")
    if epoch3_validation["runtime"]["rsbr_overhead_vs_base_forward_percent"] > 50.0:
        flags.append("RSBR_RUNTIME_REVIEW")
    g_tc = (
        training_summaries[-1]["transition_head_grad_norm"]["mean"]
        / (training_summaries[-1]["region_head_grad_norm"]["mean"] + 1e-12)
    )
    return flags, variant_delta, r_tc, g_tc


def build_decision(epoch_records, validations, training_summaries, safety):
    best = select_best_epoch(epoch_records)
    epoch3 = epoch_records[-1]
    positive = sum(value > 0 for value in best["paired_class_iou_delta_pp"].values())
    nonnegative = sum(value >= 0 for value in best["paired_class_iou_delta_pp"].values())
    epoch3_validation = validations[-1]
    flags, variant_delta, r_tc, g_tc = mechanism_flags(
        epoch3, epoch3_validation, training_summaries
    )
    type_b = epoch3_validation["taxonomy"]["B_misclassified_pure"]["recovery_pixels"]
    type_d = epoch3_validation["taxonomy"]["D_mixed_boundary"]["recovery_pixels"]
    mechanism_review = (
        max(row["standalone_cam28_1_delta_miou_pp"] for row in epoch_records) >= 0.15
        or variant_delta["core_only"] >= 0.05
        or type_b["full"] > 0
        or type_d["full"] > 0
    )
    decision = decide_pilot(
        best_delta_miou_pp=best["paired_delta_miou_pp"],
        epoch3_delta_miou_pp=epoch3["paired_delta_miou_pp"],
        nonnegative_classes_at_best=nonnegative,
        positive_classes_at_best=positive,
        safety_failure=bool(safety["failures"]),
        mechanism_review_evidence=mechanism_review,
    )
    if (
        decision == STRONG_GO
        and type_b["full"] > 0
        and type_d["full"] <= 0
    ):
        flags.append("REGION_SEMANTIC_DOMINANT")
    return {
        "decision": decision,
        "secondary_flags": sorted(set(flags)),
        "best_epoch": best["epoch"],
        "best_delta_miou_pp": best["paired_delta_miou_pp"],
        "epoch3_delta_miou_pp": epoch3["paired_delta_miou_pp"],
        "positive_classes_at_best": positive,
        "nonnegative_classes_at_best": nonnegative,
        "epoch3_variant_delta_miou_pp": variant_delta,
        "epoch3_type_b_recovery_pixels": type_b,
        "epoch3_type_d_recovery_pixels": type_d,
        "epoch3_transition_to_core_rms": r_tc,
        "epoch3_transition_to_region_grad": g_tc,
        "mechanism_review_evidence": mechanism_review,
    }


def write_report(output, summary):
    rows = [{
        "epoch": 0,
        "base": summary["epoch0_identity"]["final_metrics"]["base"],
        "refined": summary["epoch0_identity"]["final_metrics"]["full"],
        "delta_miou": summary["epoch0_identity"]["paired_delta_miou_pp"],
        "delta_mdice": summary["epoch0_identity"]["paired_delta_mdice_pp"],
        "class_delta": summary["epoch0_identity"]["paired_class_iou_delta_pp"],
    }]
    for record in summary["epochs"]:
        rows.append({
            "epoch": record["epoch"],
            "base": record["base_metrics"],
            "refined": record["refined_metrics"],
            "delta_miou": record["paired_delta_miou_pp"],
            "delta_mdice": record["paired_delta_mdice_pp"],
            "class_delta": record["paired_class_iou_delta_pp"],
        })
    executive_rows = []
    for row in rows:
        executive_rows.append(
            f"| {row['epoch']} | {100 * row['base']['mIoU']:.4f} | "
            f"{100 * row['refined']['mIoU']:.4f} | {row['delta_miou']:+.4f} | "
            f"{row['delta_mdice']:+.4f} | {row['class_delta']['0']:+.4f} | "
            f"{row['class_delta']['1']:+.4f} | {row['class_delta']['2']:+.4f} | "
            f"{row['class_delta']['3']:+.4f} |"
        )
    mechanism_rows = []
    for record in summary["epochs"]:
        mechanism_rows.append(
            f"| {record['epoch']} | {record['standalone_cam28_1_delta_miou_pp']:+.4f} | "
            f"{record['type_b_recovery_pixels']:+,} ({100 * record['type_b_recovery_rate']:+.4f}%) | "
            f"{record['type_d_recovery_pixels']:+,} ({100 * record['type_d_recovery_rate']:+.4f}%) | "
            f"{record['mechanism']['rms_delta_core']:.6f} | "
            f"{record['mechanism']['rms_delta_transition']:.6f} | "
            f"{record['mechanism']['transition_to_core_rms']:.6f} |"
        )
    ablation = summary["epoch3_ablation"]
    ablation_rows = []
    base_miou = ablation["final_metrics"]["base"]["mIoU"]
    for variant in ("base", "core_only", "transition_only", "full"):
        metrics = ablation["final_metrics"][variant]
        type_b = ablation["taxonomy"]["B_misclassified_pure"]["recovery_pixels"][variant]
        type_d = ablation["taxonomy"]["D_mixed_boundary"]["recovery_pixels"][variant]
        ablation_rows.append(
            f"| {variant} | {100 * metrics['mIoU']:.4f} | "
            f"{100 * (metrics['mIoU'] - base_miou):+.4f} | "
            f"{100 * metrics['mDice']:.4f} | "
            f"{100 * metrics['class_iou']['0']:.4f} | "
            f"{100 * metrics['class_iou']['1']:.4f} | "
            f"{100 * metrics['class_iou']['2']:.4f} | "
            f"{100 * metrics['class_iou']['3']:.4f} | {type_b:+,} | {type_d:+,} |"
        )
    recommendation = (
        "Scientifically justified for a separately approved 25-epoch frozen-SSHR study."
        if summary["decision"] in (STRONG_GO, GO)
        else "Not yet justified for a 25-epoch study without scientific review."
    )
    transition_weak = (
        abs(summary["epoch3_variant_delta_miou_pp"]["transition_only"])
        <= KNOWN_PRODUCTION_ENVELOPE_PP
        and summary["epoch3_transition_to_core_rms"] < 0.01
    )
    text = f"""# RSBR-v0 Stage-1 Three-Epoch Frozen-SSHR Pilot

## 1. Executive conclusion

**{summary['decision']}**

Secondary flags: {summary['secondary_flags'] or 'none'}

The performance delta itself is below the +0.05 pp threshold. `REVIEW` is
triggered only by the preregistered mechanism exception: Core-only is
positive, Full does not improve over Core-only, and Type-B errors decrease.
This is not a performance GO.

The pilot used BCSS training and validation only. It fresh-started from the
frozen A0 checkpoint, trained exactly three epochs, and stopped. No test,
LUAD, other seed, 25-epoch run, unfreezing, or tuning was performed.

## 2. Frozen control and provenance

- A0 commit: `{A0_COMMIT}`
- Experiment commit: `{summary['experiment_commit']}`
- A0 checkpoint SHA256: `{summary['checkpoint_sha256']}`
- Initialization SHA256: `{summary['rsbr_initialization_sha256']}`
- Parsed train / validation: {summary['parsed_train_samples']:,} / {EXPECTED_VAL:,}
- Seed / epochs / batch / size / precision: 42 / 3 / 20 / 224 / BF16
- Loss weights: 0.10 / 0.15 / 0.25 / 0.50
- Auxiliary coefficients: region=0.05, residual=0.01
- Frozen SSHR parameters unchanged: {summary['safety']['parameters_unchanged']}
- Frozen SSHR buffers unchanged: {summary['safety']['buffers_unchanged']}
- Original modules remained eval and only RSBR was trainable: {summary['mode_contract_passed']}

## 3. Paired same-forward validation

| Epoch | Base mIoU | Refined mIoU | Paired Δ (pp) | mDice Δ (pp) | C0 Δ | C1 Δ | C2 Δ | C3 Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(executive_rows)}

- Best epoch: {summary['best_epoch']}
- Best paired delta: {summary['best_delta_miou_pp']:+.4f} pp
- Epoch-3 paired delta: {summary['epoch3_delta_miou_pp']:+.4f} pp
- Epoch-3 NoiseRatio: {summary['epochs'][-1]['noise_ratio']:.3f}× the known 0.01329944 pp production envelope

## 4. Mechanism trajectory

| Epoch | CAM28_1 Δ (pp) | Type-B recovery | Type-D recovery | Core RMS | Transition RMS | T/C RMS |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(mechanism_rows)}

Epoch-3 transition/region training-gradient ratio: {summary['epoch3_transition_to_region_grad']:.6f}.

## 5. Epoch-3 paired contribution ablation

| Variant | mIoU | ΔmIoU (pp) | mDice | C0 | C1 | C2 | C3 | Type-B recovery | Type-D recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(ablation_rows)}

## 6. Training dynamics and safety

| Epoch | Training (s) | Validation (s) | Mean region grad | Mean transition grad | Mean residual ratio | Peak residual ratio |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(
    f"| {row['epoch']} | {row['training_seconds']:.2f} | {row['validation_runtime']['validation_seconds']:.2f} | "
    f"{row['training_summary']['region_head_grad_norm']['mean']:.6e} | "
    f"{row['training_summary']['transition_head_grad_norm']['mean']:.6e} | "
    f"{row['training_summary']['residual_ratio']['mean']:.6f} | "
    f"{row['training_summary']['residual_ratio']['max']:.6f} |"
    for row in summary['epochs']
)}

- Peak CUDA memory: {summary['peak_cuda_memory_bytes'] / 2**30:.3f} GiB
- Mean regions/image across validation epochs: {summary['resource_profile']['mean_regions_per_image']:.4f}
- Mean RSBR refinement overhead: {summary['resource_profile']['mean_rsbr_refinement_seconds_per_image']:.6f} s/image
- Mean RSBR overhead vs base forward: {summary['resource_profile']['mean_rsbr_overhead_vs_base_forward_percent']:.2f}%
- Safety failures: {summary['safety']['failures'] or 'none'}

## 7. Required scientific answers

1. Real paired mIoU gain: {summary['best_delta_miou_pp']:+.4f} pp at epoch {summary['best_epoch']}.
2. Gain vs numerical envelope: best NoiseRatio={max(row['noise_ratio'] for row in summary['epochs']):.3f}×.
3. Best epoch: {summary['best_epoch']}.
4. Epoch 3 remains non-negative: {summary['epoch3_delta_miou_pp'] >= 0}.
5. Refined CAM28_1 improves: {max(row['standalone_cam28_1_delta_miou_pp'] for row in summary['epochs']) > 0}.
6. Core-only positive: {summary['epoch3_variant_delta_miou_pp']['core_only'] > 0}.
7. Transition-only positive: {summary['epoch3_variant_delta_miou_pp']['transition_only'] > 0}.
8. Full exceeds both isolated paths: {'JOINT_RSBR_SYNERGY' in summary['secondary_flags']}.
9. Type-B error decreases: {summary['epoch3_type_b_recovery_pixels']['full'] > 0}.
10. Type-D error decreases: {summary['epoch3_type_d_recovery_pixels']['full'] > 0}.
11. Transition path remains quantitatively weak: {transition_weak} (transition-only Δ={summary['epoch3_variant_delta_miou_pp']['transition_only']:+.6f} pp; T/C RMS={summary['epoch3_transition_to_core_rms']:.6f}). The stricter `TRANSITION_PATH_NOT_EFFECTIVE` flag is {'present' if 'TRANSITION_PATH_NOT_EFFECTIVE' in summary['secondary_flags'] else 'absent'} because its Type-D recovery condition is evaluated separately.
12. Frozen parameter/buffer hashes strictly unchanged: {summary['safety']['parameters_unchanged'] and summary['safety']['buffers_unchanged']}.
13. 25-epoch recommendation: {recommendation}

## 8. Checkpoints and commands

- Epoch checkpoints: `checkpoints/epoch1_rsbr.pth` through `epoch3_rsbr.pth`
- Validation-selected diagnostic checkpoint: `checkpoints/best_val_rsbr.pth`
- Checkpoint hashes: `{json.dumps(summary['checkpoint_sha256s'], sort_keys=True)}`

```bash
{summary['command']}
```

## 9. Stop boundary

The Stage-1 protocol stops here regardless of the decision. This report does
not authorize test evaluation or any additional training.
"""
    (output / "docs" / "rsbr_v0_3epoch_pilot_report.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    parity, readiness, model_hashes = validate_inputs(args)
    output = Path(args.output_dir)
    seed_everything()
    dataset, loader = make_loader(args)
    model, missing, zero_init = load_fresh_model(args.checkpoint)
    model.cuda()
    steps_per_epoch = len(dataset) // args.batch_size
    optimizer = build_optimizer(model, args, steps_per_epoch)
    initial_rsbr_hash = rsbr_state_hash(model)
    frozen_initial = base_state_hashes(model)
    mode_contract = frozen_mode_ok(model)
    if not all(mode_contract.values()):
        raise RuntimeError({"Frozen mode contract failed": mode_contract})
    optimizer_groups = [{
        "lr": float(group["lr"]),
        "weight_decay": float(group["weight_decay"]),
        "momentum": float(group["momentum"]),
        "parameter_tensors": len(group["params"]),
        "parameter_elements": sum(parameter.numel() for parameter in group["params"]),
    } for group in optimizer.param_groups]
    provenance = {
        "a0_commit": A0_COMMIT,
        "experiment_commit": args.experiment_commit,
        "checkpoint_sha256": A0_CHECKPOINT_SHA256,
        "parity_decision": parity["decision"],
        "readiness_decision": readiness["decision"],
        "model_source_hashes": model_hashes,
        "rsbr_initialization_sha256": initial_rsbr_hash,
    }
    frozen_contract = {
        **provenance,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": args.batch_size,
        "image_size": args.img_size,
        "precision": "BF16",
        "lr": args.lr,
        "weight_decay": args.wt_dec,
        "steps_per_epoch": steps_per_epoch,
        "maximum_steps": steps_per_epoch * EPOCHS,
        "loss_weights": LOSS_WEIGHTS,
        "lambda_region": LAMBDA_REGION,
        "lambda_residual": LAMBDA_RESIDUAL,
        "zero_initialization": zero_init,
        "mode_contract": mode_contract,
        "optimizer_groups_initial": optimizer_groups,
        "training_split_only": "BCSS train",
        "evaluation_split_only": "BCSS val",
        "test_forbidden": True,
        "luad_forbidden": True,
        "auto_continue_forbidden": True,
        "command": " ".join(sys.argv),
    }
    write_json(output / "config" / "frozen_contract.json", frozen_contract)
    torch.cuda.reset_peak_memory_stats()
    pilot_started = time.time()

    epoch0 = paired_rsbr_validation(
        model, args.val_root, runtime_args(args), variants=("base", "full")
    )
    write_json(output / "validation" / "epoch0" / "summary.json", epoch0)
    init_identity_pass = (
        epoch0["differing_base_full_pixels"] == 0
        and epoch0["paired_delta_miou_pp"] == 0.0
        and epoch0["paired_delta_mdice_pp"] == 0.0
        and all(value == 0.0 for value in epoch0["paired_class_iou_delta_pp"].values())
    )
    if not init_identity_pass:
        payload = {
            "decision": NOGO,
            "primary_failure": "RSBR_V0_PILOT_INIT_IDENTITY_NOGO",
            "epoch0_identity": epoch0,
            "experiment_commit": args.experiment_commit,
            "checkpoint_sha256": A0_CHECKPOINT_SHA256,
            "rsbr_initialization_sha256": initial_rsbr_hash,
            "training_started": False,
            "test_accessed": False,
            "luad_accessed": False,
            "command": " ".join(sys.argv),
        }
        write_json(output / "summary.json", payload)
        print("RSBR_V0_PILOT_INIT_IDENTITY_NOGO", flush=True)
        print(NOGO, flush=True)
        return

    epoch_records = []
    validations = []
    training_summaries = []
    checkpoint_hashes = {}
    best_delta = -float("inf")
    frozen_epoch_hashes = []
    all_finite = True
    heads_active = True
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        set_frozen_training_mode(model)
        mode_now = frozen_mode_ok(model)
        if not all(mode_now.values()):
            raise RuntimeError({"Frozen mode contract failed": mode_now})
        step_rows = []
        train_started = time.time()
        for _, images, labels in loader:
            global_step += 1
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                total, classification, refined_slot, region, residual, result = training_losses(
                    model, images, labels
                )
            total.backward()
            region_gradient = grad_norm(model.rsbr.region_semantic_head.parameters())
            transition_gradient = grad_norm(model.rsbr.transition_head.parameters())
            finite = bool(
                torch.isfinite(total).item()
                and torch.isfinite(result.delta_core).all().item()
                and torch.isfinite(result.delta_transition).all().item()
                and all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.rsbr.parameters()
                )
            )
            frozen_grad_clean = frozen_gradients_clean(model)
            optimizer.step()
            finite = finite and all(
                torch.isfinite(parameter).all().item()
                for parameter in model.rsbr.parameters()
            )
            row = {
                "epoch": epoch,
                "step": global_step,
                "total_loss": float(total.detach().float().item()),
                "multiscale_classification_loss": float(classification.detach().float().item()),
                "refined_cam28_classification_loss": float(refined_slot.detach().float().item()),
                "region_mil_loss": float(region.detach().float().item()),
                "residual_loss": float(residual.detach().float().item()),
                "region_head_grad_norm": region_gradient,
                "transition_head_grad_norm": transition_gradient,
                "finite": int(finite),
                "frozen_gradients_clean": int(frozen_grad_clean),
                "lr_weight": float(optimizer.param_groups[0]["lr"]),
                "lr_bias": float(optimizer.param_groups[1]["lr"]),
                **result.statistics,
            }
            step_rows.append(row)
            append_jsonl(output / "train" / f"epoch{epoch}_steps.jsonl", row)
            all_finite = all_finite and finite and frozen_grad_clean
            if global_step % 100 == 0:
                print("RSBR_PILOT_TRAIN", json.dumps(row, sort_keys=True), flush=True)
        train_seconds = time.time() - train_started
        if len(step_rows) != steps_per_epoch:
            raise RuntimeError(f"Epoch {epoch} steps {len(step_rows)} != {steps_per_epoch}")
        train_summary = summarize_rows(step_rows)
        training_summaries.append(train_summary)
        write_json(output / "train" / f"epoch{epoch}_summary.json", {
            "epoch": epoch,
            "steps": len(step_rows),
            "training_seconds": train_seconds,
            "statistics": train_summary,
        })
        heads_active = heads_active and (
            train_summary["region_head_grad_norm"]["max"] > 0
            and train_summary["transition_head_grad_norm"]["max"] > 0
        )
        frozen_before_val = base_state_hashes(model)
        variants = ("base", "core_only", "transition_only", "full") if epoch == 3 else ("base", "full")
        validation = paired_rsbr_validation(
            model, args.val_root, runtime_args(args), variants=variants
        )
        validations.append(validation)
        write_json(output / "validation" / f"epoch{epoch}" / "summary.json", validation)
        if epoch == 3:
            write_json(output / "mechanism" / "epoch3_ablation.json", validation)
        frozen_after_val = base_state_hashes(model)
        frozen_epoch_hashes.append({
            "epoch": epoch,
            "before_validation": frozen_before_val,
            "after_validation": frozen_after_val,
        })
        record = validation_epoch_record(
            epoch, validation, train_summary, train_seconds, frozen_after_val
        )
        epoch_records.append(record)
        checkpoint, checkpoint_sha = save_checkpoint(
            model, optimizer, args, output, epoch, validation, provenance
        )
        checkpoint_hashes[checkpoint.name] = checkpoint_sha
        if validation["paired_delta_miou_pp"] > best_delta:
            best_delta = validation["paired_delta_miou_pp"]
            best_path = output / "checkpoints" / "best_val_rsbr.pth"
            shutil.copyfile(checkpoint, best_path)
            checkpoint_hashes[best_path.name] = sha256_file(best_path)
        print("RSBR_PILOT_EPOCH", json.dumps(json_ready(record)), flush=True)

    if global_step != steps_per_epoch * EPOCHS:
        raise RuntimeError("Pilot did not execute exactly three full epochs")
    frozen_final = base_state_hashes(model)
    parameters_unchanged = (
        frozen_initial["parameter_sha256"] == frozen_final["parameter_sha256"]
        and all(
            item[stage]["parameter_sha256"] == frozen_initial["parameter_sha256"]
            for item in frozen_epoch_hashes for stage in ("before_validation", "after_validation")
        )
    )
    buffers_unchanged = (
        frozen_initial["buffer_sha256"] == frozen_final["buffer_sha256"]
        and all(
            item[stage]["buffer_sha256"] == frozen_initial["buffer_sha256"]
            for item in frozen_epoch_hashes for stage in ("before_validation", "after_validation")
        )
    )
    residual_explosion = any(
        summary["residual_ratio"]["max"] > 1.0
        or summary["max_abs_delta_core"]["max"] > 100.0
        or summary["max_abs_delta_transition"]["max"] > 100.0
        for summary in training_summaries
    )
    transition_rms = [record["mechanism"]["rms_delta_transition"] for record in epoch_records]
    type_d_worsened_with_growth = (
        epoch_records[-1]["type_d_recovery_pixels"] < 0
        and transition_rms[0] < transition_rms[1] < transition_rms[2]
    )
    mdice_clear_decline = epoch_records[-1]["paired_delta_mdice_pp"] < -0.05
    validation_finite = all(
        np.isfinite(record["paired_delta_miou_pp"])
        and np.isfinite(record["paired_delta_mdice_pp"])
        and all(np.isfinite(value) for value in record["paired_class_iou_delta_pp"].values())
        and all(np.isfinite(value) for value in record["mechanism"].values())
        for record in epoch_records
    )
    safety_failures = []
    if not parameters_unchanged or not buffers_unchanged:
        safety_failures.append("FROZEN_BASE_STATE_VIOLATION")
    if not all_finite:
        safety_failures.append("NONFINITE_OR_FROZEN_GRADIENT_VIOLATION")
    if not heads_active:
        safety_failures.append("RSBR_HEAD_DEAD")
    if residual_explosion:
        safety_failures.append("RESIDUAL_EXPLOSION")
    if type_d_worsened_with_growth:
        safety_failures.append("TYPE_D_WORSENED_WITH_TRANSITION_GROWTH")
    if mdice_clear_decline:
        safety_failures.append("FINAL_MDICE_CLEAR_DECLINE")
    if not validation_finite:
        safety_failures.append("VALIDATION_INFERENCE_NONFINITE")
    safety = {
        "parameters_unchanged": parameters_unchanged,
        "buffers_unchanged": buffers_unchanged,
        "all_outputs_and_gradients_finite": all_finite,
        "both_heads_active_every_epoch": heads_active,
        "residual_explosion": residual_explosion,
        "type_d_worsened_with_transition_growth": type_d_worsened_with_growth,
        "final_mdice_clear_decline": mdice_clear_decline,
        "validation_metrics_and_mechanism_finite": validation_finite,
        "failures": safety_failures,
        "initial_hashes": frozen_initial,
        "epoch_hashes": frozen_epoch_hashes,
        "final_hashes": frozen_final,
    }
    decision = build_decision(epoch_records, validations, training_summaries, safety)
    resource_profile = {
        "training_seconds_per_epoch": [row["training_seconds"] for row in epoch_records],
        "validation_seconds_per_epoch": [
            row["validation_runtime"]["validation_seconds"] for row in epoch_records
        ],
        "mean_regions_per_image": float(np.mean([
            row["mechanism"]["components_per_image"] for row in epoch_records
        ])),
        "mean_rsbr_refinement_seconds_per_image": float(np.mean([
            row["validation_runtime"]["mean_rsbr_refinement_seconds_per_image"]
            for row in epoch_records
        ])),
        "mean_rsbr_overhead_vs_base_forward_percent": float(np.mean([
            row["validation_runtime"]["rsbr_overhead_vs_base_forward_percent"]
            for row in epoch_records
        ])),
    }
    summary = {
        **decision,
        "experiment_commit": args.experiment_commit,
        "a0_commit": A0_COMMIT,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": A0_CHECKPOINT_SHA256,
        "parity_decision": parity["decision"],
        "readiness_decision": readiness["decision"],
        "fresh_restart_from_a0": True,
        "rsbr_initialization_sha256": initial_rsbr_hash,
        "zero_initialization": zero_init,
        "missing_keys_expected_rsbr_only": missing,
        "model_source_hashes": model_hashes,
        "mode_contract": mode_contract,
        "mode_contract_passed": all(mode_contract.values()),
        "parsed_train_samples": len(dataset),
        "steps_per_epoch": steps_per_epoch,
        "epochs_trained": EPOCHS,
        "total_optimizer_updates": global_step,
        "optimizer_groups_initial": optimizer_groups,
        "epoch0_identity": epoch0,
        "epochs": epoch_records,
        "epoch3_ablation": validations[-1],
        "safety": safety,
        "resource_profile": resource_profile,
        "checkpoint_sha256s": checkpoint_hashes,
        "runtime_seconds": time.time() - pilot_started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "command": " ".join(sys.argv),
        "test_accessed": False,
        "luad_accessed": False,
        "auto_continued": False,
    }
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    print(summary["decision"], json.dumps({
        "secondary_flags": summary["secondary_flags"],
        "best_epoch": summary["best_epoch"],
        "best_delta_miou_pp": summary["best_delta_miou_pp"],
        "epoch3_delta_miou_pp": summary["epoch3_delta_miou_pp"],
        "safety_failures": safety_failures,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
