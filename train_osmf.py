"""OSMF-v1.0 training entry point built on the frozen SSHR A0 protocol.

This entry point intentionally has no test-set evaluation path. Phase-specific
audits and human review gate any later formal experiment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from network.osmf import (
    OSMF_EQUIVARIANCE_INTERVAL,
    OSMF_LAMBDA_MORPH,
    OSMF_LAMBDA_ORTH,
    OSMF_LAMBDA_REC,
    OSMF_LAMBDA_SEM,
    cross_subspace_covariance,
    inverse_align_morphology,
    orthogonality_loss,
    reconstruction_cosine,
    reconstruction_loss,
    semantic_classification_loss,
    spatial_equivariance_loss,
)
from tool import torchutils
from tool.GenDataset import Stage1_TrainDataset
from train_sshr import get_amp_dtype, seed_worker, set_seed, test_phase


def _load_weights(model: torch.nn.Module, path: str):
    if path.endswith(".params"):
        weights = importlib.import_module("network.resnet38d").convert_mxnet_to_torch(
            path
        )
    elif path.endswith(".pth"):
        weights = torch.load(path, map_location="cpu")
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
    else:
        raise ValueError("Weights must use .params or .pth")
    incompatible = model.load_state_dict(weights, strict=False)
    print(
        "[WeightLoad] "
        + json.dumps(
            {
                "missing_keys": incompatible.missing_keys,
                "unexpected_keys": incompatible.unexpected_keys,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return incompatible


def _build_optimizer(model, args, max_step):
    groups = model.get_parameter_groups()
    optim_params = [
        {"params": groups[0], "lr": args.lr, "weight_decay": args.wt_dec},
        {"params": groups[1], "lr": 2 * args.lr, "weight_decay": 0},
        {"params": groups[2], "lr": 10 * args.lr, "weight_decay": args.wt_dec},
        {"params": groups[3], "lr": 20 * args.lr, "weight_decay": 0},
    ]
    return torchutils.PolyOptimizer(
        optim_params,
        lr=args.lr,
        weight_decay=args.wt_dec,
        max_step=max_step,
    )


def _rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().mean().sqrt()


def compute_step_losses(model, images, labels, optimizer_step):
    outputs, aux = model.forward_with_aux(images)
    out_56, out_28_1, out_28_2, out_deep = outputs[:4]
    loss_sshr = (
        0.10 * F.multilabel_soft_margin_loss(out_56, labels)
        + 0.15 * F.multilabel_soft_margin_loss(out_28_1, labels)
        + 0.25 * F.multilabel_soft_margin_loss(out_28_2, labels)
        + 0.50 * F.multilabel_soft_margin_loss(out_deep, labels)
    )
    loss_sem = semantic_classification_loss(aux["semantic_logits"], labels)
    loss_orth = orthogonality_loss(aux["semantic"], aux["morphology"])
    loss_rec = reconstruction_loss(aux["reconstruction"], aux["input"])

    compute_equivariance = (optimizer_step + 1) % OSMF_EQUIVARIANCE_INTERVAL == 0
    if compute_equivariance:
        # The frozen A0 training pipeline has no photometric transform. OSMF-v1
        # therefore adds no new color augmentation: view_a is the existing A0
        # sample and view_b differs only by the permitted deterministic flip.
        flip_dimension = 3 if ((optimizer_step + 1) // OSMF_EQUIVARIANCE_INTERVAL) % 2 else 2
        view_b = torch.flip(images, dims=(flip_dimension,))
        morphology_b = model.forward_morphology(view_b)
        morphology_b = inverse_align_morphology(morphology_b, flip_dimension)
        loss_eq = spatial_equivariance_loss(aux["morphology"], morphology_b)
    else:
        loss_eq = loss_sshr.new_zeros(())

    total = (
        loss_sshr
        + OSMF_LAMBDA_SEM * loss_sem
        + OSMF_LAMBDA_MORPH * loss_eq
        + OSMF_LAMBDA_ORTH * loss_orth
        + OSMF_LAMBDA_REC * loss_rec
    )
    diagnostics = {
        "loss_total": total.detach(),
        "loss_sshr": loss_sshr.detach(),
        "loss_sem": loss_sem.detach(),
        "loss_eq": loss_eq.detach(),
        "loss_orth": loss_orth.detach(),
        "loss_rec": loss_rec.detach(),
        "semantic_rms": _rms(aux["semantic"]).detach(),
        "morphology_rms": _rms(aux["morphology"]).detach(),
        "reconstruction_cosine": reconstruction_cosine(
            aux["reconstruction"], aux["input"]
        ).detach(),
        "cross_covariance": cross_subspace_covariance(
            aux["semantic"], aux["morphology"]
        )
        .square()
        .mean()
        .sqrt()
        .detach(),
        "equivariance_computed": compute_equivariance,
    }
    return total, diagnostics


def train(args):
    set_seed(args.seed)
    model = getattr(importlib.import_module(args.network), "Net")(
        n_class=args.n_class
    ).cuda()
    _load_weights(model, args.weights)

    dataset = Stage1_TrainDataset(
        data_path=args.trainroot,
        transform=transforms.Compose([transforms.ToTensor()]),
        dataset=args.dataset,
        img_size=args.img_size,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
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
    max_step = (len(dataset) // args.batch_size) * args.max_epoches
    optimizer = _build_optimizer(model, args, max_step=max_step)
    amp_dtype = get_amp_dtype(args)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp_dtype == "fp16")

    output_dir = Path(args.save_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history.jsonl"
    best_val_miou = None
    started = time.perf_counter()

    for epoch in range(1, args.max_epoches + 1):
        model.train()
        sums = {}
        count = 0
        for _, images, labels in loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                loss, diagnostics = compute_step_losses(
                    model, images, labels, optimizer.global_step
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            count += 1
            for key, value in diagnostics.items():
                if key == "equivariance_computed":
                    continue
                sums[key] = sums.get(key, 0.0) + float(value.cpu())

        torch.save(model.state_dict(), output_dir / "last.pth")
        record = {
            "epoch": epoch,
            **{key: value / count for key, value in sums.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            score = test_phase(
                args,
                dataroot=args.valroot,
                split_name="val",
                state_dict=model.state_dict(),
            )
            record["val_mIoU"] = score["Mean IoU"]
            record["val_mDice"] = score["Mean Dice"]
            if best_val_miou is None or record["val_mIoU"] > best_val_miou:
                best_val_miou = record["val_mIoU"]
                torch.save(model.state_dict(), output_dir / "best_val.pth")
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print("[Epoch] " + json.dumps(record, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=3, type=int)
    parser.add_argument("--network", default="network.resnet38_cls_osmf")
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument(
        "--weights",
        default="init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params",
    )
    parser.add_argument("--trainroot", required=True)
    parser.add_argument("--valroot", required=True)
    parser.add_argument("--dataset", default="bcss", choices=["bcss"])
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--save_folder", required=True)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument(
        "--amp_dtype",
        "--amp-dtype",
        default="bf16",
        choices=["none", "bf16", "fp16"],
    )
    # Frozen inference and loss settings are explicit for test_phase parity.
    parser.add_argument("--infer_thr", default=None, type=float)
    parser.add_argument("--cam_w_28_1", default=0.6, type=float)
    parser.add_argument("--cam_w_28_2", default=0.2, type=float)
    parser.add_argument("--cam_w_deep", default=0.2, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
