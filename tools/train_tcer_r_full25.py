#!/usr/bin/env python3
"""Fresh 25-epoch BCSS seed42 training for frozen R-only TCER."""

from __future__ import annotations

import argparse
import importlib
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from network.resnet38_cls_tcrd_gate import Net
from tool.GenDataset import Stage1_TrainDataset
from tools.tcer_r_full25_common import (
    BATCH_SIZE, EPOCHS, EXPECTED_PRETRAINED_SHA256, EXPERIMENT_NAME,
    MAX_STEPS, SEED, STEPS_PER_EPOCH, build_official_optimizer,
    official_classification_loss, seed_worker, set_official_seed,
    sha256_file, write_json,
)


def classification_accuracy(probability, truth, threshold=0.2):
    exact = 0
    jaccard = 0.0
    for prediction, target in zip(probability, truth):
        predicted = set(np.where(prediction > threshold)[0].tolist())
        actual = set(np.where(target == 1)[0].tolist())
        exact += int(predicted == actual)
        union = predicted | actual
        jaccard += len(predicted & actual) / len(union) if union else 1.0
    return exact, jaccard


def mechanism_state(model):
    matrix = model.tcrd.competition_matrix().detach().float().cpu()
    return {
        "eta_r": float(model.tcrd.eta_r.detach()),
        "competition_matrix": matrix.tolist(),
        "competition_offdiag_min": float(matrix[matrix > 0].min()),
        "competition_offdiag_max": float(matrix.max()),
    }


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_resume(path, epoch, model, optimizer, generator, history):
    atomic_torch_save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_global_step": optimizer.global_step,
        "data_generator_state": generator.get_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "history": history,
    }, path)


def restore_resume(path, model, optimizer, generator):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    optimizer.global_step = int(state["optimizer_global_step"])
    generator.set_state(state["data_generator_state"])
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["epoch"]), list(state["history"])


def run(args):
    if args.num_workers != 4:
        raise AssertionError("Frozen exploratory protocol requires num_workers=4")
    if sha256_file(args.pretrained) != EXPECTED_PRETRAINED_SHA256:
        raise AssertionError("Pretrained ResNet38 SHA256 mismatch")
    output = Path(args.output_dir)
    resume_path = output / "resume_latest.pth"
    if output.exists() and not args.resume:
        # Shell redirection creates run.log before this process begins.
        allowed_bootstrap_files = {"preflight.json", "run.log"}
        unexpected = [
            path.name for path in output.iterdir()
            if path.name not in allowed_bootstrap_files
        ]
        if unexpected:
            raise FileExistsError(
                f"Refusing to overwrite experiment {output}; existing={unexpected}"
            )
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    output.mkdir(parents=True, exist_ok=True)

    set_official_seed(SEED)
    model = Net(4, branch="R").cuda()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    transform_train = transforms.Compose([transforms.ToTensor()])
    dataset = Stage1_TrainDataset(
        data_path=args.train_root, transform=transform_train,
        dataset="bcss", img_size=224,
    )
    if len(dataset) != 23422:
        raise AssertionError(f"Expected 23,422 BCSS samples, found {len(dataset)}")
    data_generator = torch.Generator()
    data_generator.manual_seed(SEED)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker, generator=data_generator,
    )
    if len(loader) != STEPS_PER_EPOCH:
        raise AssertionError(f"Expected 1,171 steps/epoch, found {len(loader)}")
    optimizer = build_official_optimizer(model)

    converted = importlib.import_module("network.resnet38d").convert_mxnet_to_torch(
        args.pretrained
    )
    incompat = model.load_state_dict(converted, strict=False)
    if incompat.unexpected_keys:
        raise AssertionError(f"Unexpected pretrained keys: {incompat.unexpected_keys}")
    for key, value in converted.items():
        if not torch.equal(model.state_dict()[key].cpu(), value):
            raise AssertionError(f"Pretrained backbone tensor did not load: {key}")

    start_epoch, history = 0, []
    if args.resume:
        start_epoch, history = restore_resume(
            resume_path, model, optimizer, data_generator
        )
    provenance = {
        "experiment": EXPERIMENT_NAME,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "pretrained_path": str(Path(args.pretrained).resolve()),
        "pretrained_sha256": EXPECTED_PRETRAINED_SHA256,
        "pretrained_missing_keys": incompat.missing_keys,
        "pretrained_unexpected_keys": incompat.unexpected_keys,
        "dataset": "BCSS", "train_samples": len(dataset),
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "steps_per_epoch": STEPS_PER_EPOCH, "max_steps": MAX_STEPS,
        "precision": "bf16", "image_size": 224,
        "model": "frozen TCER R-only, T=3",
        "optimizer": "released PolyOptimizer/SGD",
        "optimizer_momentum": [g["momentum"] for g in optimizer.param_groups],
        "initial_group_lrs": [g["lr"] for g in optimizer.param_groups],
        "group_weight_decay": [g["weight_decay"] for g in optimizer.param_groups],
        "poly_power": 0.9,
        "loss_weights": [0.10, 0.15, 0.25, 0.50],
        "official_nondeterministic_gpu_policy": True,
        "validation_during_training": False,
        "test_used": False, "luad_used": False,
        "checkpoint_selection": "none; epoch25 FINAL only",
    }
    write_json(output / "provenance.json", provenance)

    started = time.time()
    print(f"TCER_R25_START epoch={start_epoch + 1} max_steps={MAX_STEPS}", flush=True)
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        epoch_started = time.time()
        loss_sum = 0.0
        exact_sum = 0
        jaccard_sum = 0.0
        sample_count = 0
        for step, (_, image, label) in enumerate(loader):
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(image, active_labels=label)
                loss = official_classification_loss(outputs, label)
            loss.backward()
            optimizer.step()
            probability = outputs[4].detach().float().cpu().numpy()
            truth = label.detach().float().cpu().numpy()
            exact, jaccard = classification_accuracy(probability, truth)
            batch = image.shape[0]
            loss_sum += float(loss.detach()) * batch
            exact_sum += exact
            jaccard_sum += jaccard
            sample_count += batch
            if (step + 1) % 100 == 0:
                print(
                    f"TCER_R25_TRAIN epoch={epoch + 1}/25 "
                    f"batch={step + 1}/{STEPS_PER_EPOCH} "
                    f"global_step={optimizer.global_step}/{MAX_STEPS} "
                    f"loss={loss_sum / sample_count:.6f} "
                    f"exact_match={exact_sum / sample_count:.6f} "
                    f"accuracy={jaccard_sum / sample_count:.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f}", flush=True,
                )
        record = {
            "epoch": epoch + 1,
            "global_step": optimizer.global_step,
            "classification_loss": loss_sum / sample_count,
            "training_exact_match": exact_sum / sample_count,
            "training_accuracy": jaccard_sum / sample_count,
            "lr_end": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "mechanism": mechanism_state(model),
            "all_finite": bool(np.isfinite(loss_sum / sample_count)),
        }
        history.append(record)
        write_json(output / "training_history.json", history)
        save_resume(
            resume_path, epoch + 1, model, optimizer, data_generator, history
        )
        print("TCER_R25_EPOCH " + str(record), flush=True)

    final_checkpoint = output / "stage1_last.pth"
    atomic_torch_save(model.state_dict(), final_checkpoint)
    completion = {
        "status": "TCER_R25_TRAINING_COMPLETE",
        "epochs": EPOCHS, "steps": optimizer.global_step,
        "checkpoint": str(final_checkpoint),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "total_seconds_this_process": time.time() - started,
        "final_training_record": history[-1],
        "validation_run": False, "test_used": False,
    }
    write_json(output / "training_complete.json", completion)
    print("TCER_R25_TRAINING_COMPLETE", flush=True)
    print(completion, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
