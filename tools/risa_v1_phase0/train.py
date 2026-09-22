"""Fixed five-epoch RISA-v1 Phase-0 training on image-level labels only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_sshr as official
from network.risa_v1 import RISAAdapter, risa_losses
from tools.pdsr_vpca_phase0.common import CommonAugmentTrainDataset
from tools.risa_v1_phase0.common import (
    ACCUMULATION, CHECKPOINT_SHA256, EPOCHS, MICRO_BATCH, STEPS_PER_EPOCH, TOTAL_STEPS,
    append_csv, binary_auc, set_seed, sha256, write_json,
)


def tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.inference_mode()
def spatial_snapshot(model: RISAAdapter, raw: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for prefix, layer in (("ccra2", model.base.ccra2), ("ccra3", model.base.ccra3)):
        hooks.append(layer.k_projection.register_forward_hook(
            lambda _m, _i, out, name=f"{prefix}_K": captured.__setitem__(name, out.detach().float().cpu())
        ))
        hooks.append(layer.v_projection.register_forward_hook(
            lambda _m, _i, out, name=f"{prefix}_V": captured.__setitem__(name, out.detach().float().cpu())
        ))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.frozen_forward(raw, labels)
    for hook in hooks:
        hook.remove()
    final = output["stages"][2]["hqmr"]
    captured.update({
        "responsibility": output["stages"][2]["detail"]["responsibility_class"].detach().float().cpu(),
        "hqmr_query0": final["query0"].detach().float().cpu(),
        "L5": final["logits5"].detach().float().cpu(),
        "q5": final["query5"].detach().float().cpu(),
        "L4": final["logits4"].detach().float().cpu(),
        "q4": final["query4"].detach().float().cpu(),
        "L3": final["logits3"].detach().float().cpu(),
    })
    return captured


def gradient_state(model: RISAAdapter) -> dict:
    base_bad = [name for name, parameter in model.base.named_parameters() if parameter.grad is not None]
    adapter = {name: None if parameter.grad is None else float(parameter.grad.detach().float().abs().sum())
               for name, parameter in model.risa.named_parameters()}
    return {"frozen_base_grad_none": not base_bad, "unexpected_base_gradients": base_bad,
            "adapter_gradient_abs_sum": adapter}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    artifact = args.output / "artifacts/risa_v1_phase0"
    baseline = json.loads((artifact / "baseline_replay.json").read_text())
    if baseline["BASELINE_REPLAY"] != "PASS":
        raise AssertionError("A passing pre-modification baseline replay is mandatory")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise AssertionError("Frozen HQMR checkpoint hash mismatch")
    set_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dataset = CommonAugmentTrainDataset(args.train_root)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(
        dataset, batch_size=MICRO_BATCH, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, worker_init_fn=official.seed_worker,
        generator=generator, persistent_workers=args.num_workers > 0,
    )
    if len(dataset) != 23422 or len(loader) != 4684:
        raise AssertionError("BCSS training manifest changed")
    model = RISAAdapter(args.checkpoint).cuda().train()
    trainable = [parameter for parameter in model.risa.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=3.e-4, weight_decay=1.e-4)
    target_steps = args.smoke_steps or TOTAL_STEPS
    fixed_name, fixed_raw, fixed_label = dataset[0]
    del fixed_name
    fixed_raw = fixed_raw[None].cuda()
    fixed_label = fixed_label[None].cuda()
    before = spatial_snapshot(model, fixed_raw, fixed_label)
    before_hash = {name: tensor_digest(value) for name, value in before.items()}
    parameter_before = {name: tensor_digest(parameter) for name, parameter in model.base.named_parameters()}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    gradients = {}
    for epoch in range(1, EPOCHS + 1):
        hard = epoch >= 2
        sums = {key: 0. for key in ("loss", "loss_mil", "loss_rank", "loss_hard", "entropy", "margin", "gate")}
        observations = 0
        histogram = torch.zeros(4, dtype=torch.long)
        presence_rows, label_rows = [], []
        epoch_started = time.perf_counter()
        for micro_step, (_names, raw, labels) in enumerate(loader, 1):
            raw, labels = raw.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(raw, labels, hard_refinement=hard)
                losses = risa_losses(result["risa"], labels, hard)
                scaled = losses["loss"] / ACCUMULATION
            if not torch.isfinite(scaled):
                raise FloatingPointError("Non-finite RISA loss")
            scaled.backward()
            if epoch == 1 and "epoch1_first_backward" not in gradients:
                gradients["epoch1_first_backward"] = gradient_state(model)
            if epoch == 2 and "epoch2_first_backward" not in gradients:
                gradients["epoch2_first_backward"] = gradient_state(model)
            output = result["risa"]
            batch = raw.shape[0]
            observations += batch
            for key in ("loss", "loss_mil", "loss_rank", "loss_hard"):
                sums[key] += float(losses[key].detach()) * batch
            sums["entropy"] += float(output["identity_entropy"].detach().mean()) * batch
            probability = output["identity_prob"].detach().float()
            top = probability.topk(2, dim=-1)
            sums["margin"] += float((top.values[..., 0] - top.values[..., 1]).mean()) * batch
            sums["gate"] += float(output["hard_gate"].detach().float().mean()) * batch
            histogram += torch.bincount(top.indices[..., 0].cpu().flatten(), minlength=4)
            presence_rows.append(output["presence_prob"].detach().float().cpu())
            label_rows.append(labels.detach().float().cpu())
            if micro_step % ACCUMULATION:
                continue
            gradient_norm = float(clip_grad_norm_(trainable, 1.0))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % 100 == 0 or global_step == target_steps:
                print(json.dumps({"event": "RISA_TRAIN_STEP", "epoch": epoch, "step": global_step,
                                  "loss": sums["loss"] / observations, "gradient_norm": gradient_norm,
                                  "hard_refinement": hard}), flush=True)
            if global_step >= target_steps:
                break
        presence = torch.cat(presence_rows).numpy()
        targets = torch.cat(label_rows).numpy()
        auc = [binary_auc(targets[:, class_id], presence[:, class_id]) for class_id in range(4)]
        class_fraction = (histogram.float() / histogram.sum().clamp_min(1)).tolist()
        row = {
            "epoch": epoch, "optimizer_step": global_step, "hard_refinement": hard,
            "L_total": sums["loss"] / observations, "L_MIL": sums["loss_mil"] / observations,
            "L_rank": sums["loss_rank"] / observations, "L_hard": sums["loss_hard"] / observations,
            "mean_identity_entropy": sums["entropy"] / observations,
            "mean_top1_top2_margin": sums["margin"] / observations,
            "mean_hard_gate": sums["gate"] / observations,
            **{f"class_hist_C{index}": value for index, value in enumerate(class_fraction)},
            **{f"presence_AUROC_C{index}": value for index, value in enumerate(auc)},
            "presence_AUROC_macro": float(np.nanmean(auc)),
            "seconds": time.perf_counter() - epoch_started,
        }
        append_csv(artifact / "training_epoch_metrics.csv", row)
        print("RISA_EPOCH_COMPLETE " + json.dumps(row), flush=True)
        if global_step >= target_steps:
            break
    status = "SMOKE_COMPLETE" if args.smoke_steps else "TRAINING_COMPLETE"
    if not args.smoke_steps and (epoch != 5 or global_step != TOTAL_STEPS):
        raise AssertionError(f"Incomplete fixed training: {epoch=} {global_step=}")
    after = spatial_snapshot(model, fixed_raw, fixed_label)
    tensor_audit = {}
    for name in before:
        delta = (before[name] - after[name]).abs()
        tensor_audit[name] = {"before_sha256": before_hash[name], "after_sha256": tensor_digest(after[name]),
                              "max_abs": float(delta.max()), "rms": float(delta.square().mean().sqrt())}
    parameter_after = {name: tensor_digest(parameter) for name, parameter in model.base.named_parameters()}
    changed_parameters = [name for name in parameter_before if parameter_before[name] != parameter_after[name]]
    isolation = {
        "ISOLATION_PASS": all(item["max_abs"] <= 1.e-7 for item in tensor_audit.values()) and not changed_parameters,
        "tensor_audit": tensor_audit, "changed_frozen_parameters": changed_parameters,
        "frozen_checkpoint_sha256": sha256(args.checkpoint), "gradient_audit": gradients,
    }
    if not isolation["ISOLATION_PASS"]:
        raise AssertionError(isolation)
    write_json(artifact / "gradient_causal_isolation_audit.json", isolation)
    checkpoint = artifact / "risa_v1_epoch5.pt"
    torch.save({"state_dict": model.trainable_state_dict(), "epoch": epoch, "optimizer_steps": global_step,
                "config": {"dimension": 128, "temperature": .10, "delta": .15,
                           "gate_temperature": .05, "topk_fraction": .20}}, checkpoint)
    replay = RISAAdapter(args.checkpoint).cuda().eval()
    replay.load_trainable_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        first = model(fixed_raw, fixed_label, True)["risa"]["identity_prob"]
        second = replay(fixed_raw, fixed_label, True)["risa"]["identity_prob"]
    checkpoint_replay_delta = float((first.float() - second.float()).abs().max())
    if checkpoint_replay_delta > 1.e-7:
        raise AssertionError(f"Checkpoint replay mismatch: {checkpoint_replay_delta}")
    trainable_count = sum(parameter.numel() for parameter in model.risa.parameters())
    base_count = sum(parameter.numel() for parameter in model.base.parameters())
    runtime = {
        "status": status, "epochs": epoch, "optimizer_steps": global_step,
        "micro_batch": MICRO_BATCH, "gradient_accumulation": ACCUMULATION,
        "effective_batch": MICRO_BATCH * ACCUMULATION, "train_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_replay_max_abs": checkpoint_replay_delta,
        "risa_parameters": trainable_count, "baseline_parameters": base_count,
        "additional_percent": 100 * trainable_count / base_count,
        "segmentation_gt_accessed": False, "validation_accessed": False,
        "image_level_supervision_only": True,
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "platform": platform.platform()},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write_json(artifact / "training_runtime.json", runtime)
    print("RISA_TRAINING_DONE " + json.dumps(runtime), flush=True)


if __name__ == "__main__":
    main()
