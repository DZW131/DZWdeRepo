"""Train-only FQRF focus-preservation gate on BCSS Seed42."""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import train_sshr as official
from network.fqrf_net import FQRFNet, STAGE_WEIGHTS
from network.hqrf_targets import FULL25_STEPS
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.fqrf_diagnostics import apply_epoch2_screen, apply_final_gate, batch_health, summarize
from tools.fqrf_report import render_report
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json


ROOT = Path(__file__).resolve().parent
PHASE0_STEPS = 3513
STEPS_PER_EPOCH = 1171
SNAPSHOTS = {250: "step0250", 500: "step0500", 1000: "step1000", 1171: "epoch1", 2342: "epoch2", 3513: "epoch3"}
VISUAL_SNAPSHOTS = {500, 1000, 2342, 3513}
CONFIG = {
    "experiment": "FQRF-Net Innovation 1 v3 Phase-0 Focus Preservation Gate",
    "dataset": "BCSS training only", "seed": 42, "gpu": "RTX4090D", "batch": 20,
    "epochs_max": 3, "steps_per_epoch": STEPS_PER_EPOCH, "phase0_steps_max": PHASE0_STEPS,
    "full25_schedule_denominator": FULL25_STEPS, "amp": "bf16", "image_size": 224,
    "query_grid": [14, 14], "query_count": 196, "dimension": 256, "patch_size": 16,
    "decoder_order": ["cross_attention", "norm", "self_attention", "norm", "ffn", "norm"],
    "decoder_stages": 3, "heads": 8, "ffn_hidden": 1024, "dropout": .1, "activation": "GELU",
    "memory_position": "parameter-free normalized 2-D sine/cosine on each 28x28 memory grid",
    "dynamic_focus": "head-average attention, FP32 renormalization, detached A, h(AKp+B)",
    "memory_detach": "detach CNN-side FD/F5/F4-context before trainable memory projection",
    "attention_mask": {"threshold": .15, "source": "previous-stage sigmoid mask", "detached": True, "empty_row_fallback": "global"},
    "stage_loss_weights": list(STAGE_WEIGHTS), "loss_weights": {"deep": .50, "PCA": .25, "mask": .25},
    "pseudo": {"positive_floor": .60, "top_ratio": .15, "class_margin": .10, "background_ceiling": .10, "positive_dilation": 1},
    "locality": {"radii": [1, 5], "denominator": FULL25_STEPS, "query_to_mask_scale": 4},
    "pmec": {"tau_bin": .70, "tau_low": .40, "tau_high": .50, "T": 5, "stage": 3},
    "optimizer": {"base_lr": .01, "weight_decay": .0005, "multipliers": [1, 2, 10, 20], "poly_max_step": PHASE0_STEPS},
    "monitor_snapshots": list(SNAPSHOTS), "validation_access": False,
    "hqrf_reference": {"median_iou": .9554, "fraction_iou_gt_090": .6104, "embedding_cosine_p90": .9965},
}


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


class MonitorDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        from PIL import Image
        path, label = self.rows[index]
        image = Image.open(path).convert("RGB").resize((224, 224))
        value = transforms.functional.to_tensor(image)
        value = transforms.functional.normalize(value, [.485, .456, .406], [.229, .224, .225])
        return Path(path).stem, value, label.float()


def select_monitor_cohort(rows):
    single = [row for row in rows if int(row[1].sum()) == 1]
    multi = [row for row in rows if int(row[1].sum()) >= 2]
    rng = random.Random(42)
    rng.shuffle(single); rng.shuffle(multi)
    chosen = single[:8] + multi[:16]
    used = {str(row[0]) for row in chosen}
    remainder = [row for row in rows if str(row[0]) not in used]
    rng.shuffle(remainder); chosen += remainder[:32 - len(chosen)]
    if len(chosen) != 32 or sum(int(row[1].sum()) == 1 for row in chosen) < 8 or sum(int(row[1].sum()) >= 2 for row in chosen) < 16:
        raise RuntimeError("Unable to freeze the preregistered 32-image cohort")
    return chosen


def gradient_norms(model):
    names = ("backbone", "deep", "query", "focus", "pixel", "mask", "pca")
    groups = {name: [] for name in names}
    for name, parameter in model.named_parameters():
        if parameter.grad is None: continue
        if name.startswith("backbone."): group = "backbone"
        elif name.startswith("deep_head."): group = "deep"
        elif name.startswith(("focus_update1.", "focus_update2.")): group = "focus"
        elif name.startswith("pixel_decoder."): group = "pixel"
        elif name.startswith("mask_embeddings."): group = "mask"
        elif name.startswith("pca_heads."): group = "pca"
        else: group = "query"
        groups[group].append(parameter.grad.detach().float().square().sum())
    return {key: float(torch.stack(values).sum().sqrt()) if values else 0.0 for key, values in groups.items()}


def _rgb(images):
    mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
    std = images.new_tensor([.229, .224, .225])[None, :, None, None]
    return (images * std + mean).clamp(0, 1).permute(0, 2, 3, 1).float().cpu().numpy()


def save_visualizations(directory, snapshot, names, images, labels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target = Path(directory) / "visualizations" / snapshot
    target.mkdir(parents=True, exist_ok=True)
    rgb = _rgb(images)
    for image in range(min(8, len(names))):
        panels = [(rgb[image], "input", None), (output["normalized_cam"][image].max(0).values.float().cpu().numpy(), "deep CAM", None)]
        top_indices = []
        for stage_index, stage in enumerate(output["stages"], start=1):
            score = stage["confidence"]["joint"][image].float().max(-1).values
            top = torch.argsort(score, descending=True, stable=True)[:5]
            top_indices.append(top)
            probability = stage["mask_logits"][image, top].float().sigmoid().cpu().numpy()
            panels.extend((probability[rank], f"S{stage_index} mask {rank+1}", None) for rank in range(5))
        for stage_index, stage in enumerate(output["stages"], start=1):
            index = int(top_indices[stage_index - 1][0])
            height, width = stage["memory_hw"]
            attention = stage["attention"]["cross_attention"][image, index].float().reshape(height, width).cpu().numpy()
            panels.append((attention, f"S{stage_index} top1 attention", None))
        for source_stage, title in ((0, "Qp2 focus centroids"), (1, "Qp3 focus centroids")):
            stage = output["stages"][source_stage]
            height, width = stage["memory_hw"]
            attention = stage["attention"]["cross_attention"][image, top_indices[source_stage]].float()
            yy, xx = torch.meshgrid(torch.arange(height, device=attention.device), torch.arange(width, device=attention.device), indexing="ij")
            norm = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
            cx = (norm * xx.flatten()).sum(-1).cpu().numpy() / max(width - 1, 1) * 223
            cy = (norm * yy.flatten()).sum(-1).cpu().numpy() / max(height - 1, 1) * 223
            panels.append((rgb[image], title, (cx, cy)))
        assignment = output["stages"][2]["confidence"]["p_class"][image].argmax(-1).reshape(14, 14).float().cpu().numpy()
        panels += [(assignment, "S3 PCA assignment", None), (output["pmec_region"][image].amax(0).float().cpu().numpy(), "S3 PMEC", None)]
        figure, axes = plt.subplots(4, 6, figsize=(18, 12))
        for axis, (value, title, scatter) in zip(axes.flat, panels):
            axis.imshow(value, cmap=None if value.ndim == 3 else "viridis")
            if scatter is not None: axis.scatter(scatter[0], scatter[1], c=np.arange(5), cmap="tab10", s=30, edgecolors="white")
            axis.set_title(title); axis.axis("off")
        figure.suptitle(f"{names[image]} | present={torch.where(labels[image].bool())[0].tolist()}")
        figure.tight_layout(); figure.savefig(target / f"{names[image]}.png", dpi=120); plt.close(figure)


@torch.no_grad()
def monitor(model, loader, step, snapshot, histories, output_dir, visualize=False):
    was_training = model.training; model.eval(); batches = []; pmec_rows = []; first = None
    for names, images, labels in loader:
        images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=step, run_pmec=True)
        batches.append(batch_health(result, labels)); pmec_rows.extend(result["pmec_rows"])
        if first is None: first = (names, images.detach(), labels.detach(), result)
    summary = summarize(snapshot, batches, pmec_rows, model)
    mapping = {
        "stagewise_mask_area": "fqrf_stagewise_mask_area.csv",
        "stagewise_query_redundancy": "fqrf_stagewise_query_redundancy.csv",
        "stagewise_embedding_diversity": "fqrf_stagewise_embedding_diversity.csv",
        "dynamic_focus": "fqrf_dynamic_focus.csv", "attention_focus": "fqrf_attention_focus.csv",
        "masked_attention_health": "fqrf_masked_attention_health.csv", "semantic_selectivity": "fqrf_semantic_selectivity.csv",
        "pca_health": "fqrf_pca_health.csv", "pmec_health": "fqrf_pmec_health.csv", "chpf_health": "fqrf_chpf_health.csv",
    }
    for key, filename in mapping.items():
        rows = summary[key] if isinstance(summary[key], list) else [summary[key]]
        histories[key].extend(rows); write_csv(Path(output_dir) / filename, histories[key])
    if visualize and first is not None: save_visualizations(output_dir, snapshot, *first)
    print("FQRF_MONITOR " + json.dumps(summary, allow_nan=False), flush=True)
    if was_training: model.train()
    return summary


def checkpoint(model, output):
    path = Path(output) / "fqrf_phase0_endpoint.pth"
    torch.save(model.state_dict(), path)
    digest = sha256(path)
    (Path(output) / "fqrf_phase0_endpoint_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    return path, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True); parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--smoke-steps", type=int, choices=(0, 2), default=0)
    args = parser.parse_args(); check_train_path(args.trainroot); output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("Native CUDA BF16 required")
    if "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("Registered RTX4090D required")
    protected = protected_sources(ROOT); accesses = install_train_access_guard(); official.set_seed(42)
    output.mkdir(parents=True); log_handle = (output / "fqrf_phase0_train.log").open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_handle); sys.stderr = Tee(sys.__stderr__, log_handle)
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {**CONFIG, "source_commit": source_commit, "smoke_steps": args.smoke_steps, "trainroot": str(Path(args.trainroot).resolve())}
    write_json(output / "fqrf_phase0_config.json", config)
    (output / "fqrf_phase0_source_commit.txt").write_text(source_commit + "\n", encoding="utf-8")
    model = FQRFNet().cuda(); init = model.backbone.load_official_initialization(args.weights)
    write_json(output / "fqrf_phase0_init_identity.json", init)
    dataset = Stage1_TrainDataset(args.trainroot, transform=transforms.Compose([transforms.ToTensor()]), dataset="bcss", img_size=224)
    if len(dataset) != 23422: raise RuntimeError("Expected 23,422 BCSS training images")
    cohort = select_monitor_cohort(dataset.object)
    write_json(output / "fqrf_phase0_monitor_cohort.json", {"seed": 42, "images": [{"path": str(Path(path).resolve()), "label": [int(v) for v in label.tolist()]} for path, label in cohort]})
    monitor_loader = DataLoader(MonitorDataset(cohort), batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, worker_init_fn=official.seed_worker, generator=generator)
    if len(loader) != STEPS_PER_EPOCH: raise RuntimeError("Expected 1,171 steps per epoch")
    groups = model.get_parameter_groups()
    optimizer = PolyOptimizer([{"params": group, "lr": .01 * multiplier, "weight_decay": decay} for group, multiplier, decay in zip(groups, (1, 2, 10, 20), (.0005, 0, .0005, 0))], lr=.01, weight_decay=.0005, max_step=PHASE0_STEPS)
    write_json(output / "fqrf_phase0_provenance.json", {"protected_sources": protected, "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}, "argv": sys.argv})
    print("FQRF_PROTOCOL " + json.dumps(config, allow_nan=False), flush=True)

    history_keys = ("stagewise_mask_area", "stagewise_query_redundancy", "stagewise_embedding_diversity", "dynamic_focus", "attention_focus", "masked_attention_health", "semantic_selectivity", "pca_health", "pmec_health", "chpf_health")
    histories = {key: [] for key in history_keys}; losses = []; final_summary = None; gate = None
    completed = 0; last_grad = {}; started = time.perf_counter(); all_finite = True
    torch.cuda.reset_peak_memory_stats()
    try:
        model.train()
        for epoch in range(1, 4):
            epoch_losses = []
            for _, images, labels in loader:
                images = images.cuda(non_blocking=True); labels = labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16): result = model(images, labels, step=optimizer.global_step, run_pmec=False)
                loss = result["losses"]["loss"]
                if not bool(torch.isfinite(loss)): raise FloatingPointError(f"Nonfinite loss at step {optimizer.global_step + 1}")
                loss.backward(); last_grad = gradient_norms(model)
                if not all(np.isfinite(value) for value in last_grad.values()): raise FloatingPointError("Nonfinite gradient")
                optimizer.step(); step = optimizer.global_step
                row = {"epoch": epoch, "step": step, **{key: float(value.detach()) for key, value in result["losses"].items()}, "lr": optimizer.param_groups[0]["lr"], **{f"grad_{key}": value for key, value in last_grad.items()}}
                epoch_losses.append(row)
                if step % 100 == 0 or step in SNAPSHOTS:
                    losses.append(row); write_csv(output / "fqrf_losses.csv", losses)
                    print("FQRF_STEP " + json.dumps({**row, "peak_memory": torch.cuda.max_memory_allocated(), "elapsed_seconds": time.perf_counter() - started}), flush=True)
                if step in SNAPSHOTS:
                    final_summary = monitor(model, monitor_loader, step, SNAPSHOTS[step], histories, output, visualize=step in VISUAL_SNAPSHOTS)
                if args.smoke_steps and step >= args.smoke_steps: break
            completed = epoch
            mean_row = {key: float(np.mean([row[key] for row in epoch_losses])) for key in ("loss", "loss_deep", "loss_pca", "loss_mask")}
            print("FQRF_EPOCH " + json.dumps({"epoch": epoch, "step": optimizer.global_step, **mean_row, "gradients": last_grad}), flush=True)
            if args.smoke_steps: break
            if epoch == 2:
                gate = apply_epoch2_screen(final_summary); write_json(output / "fqrf_phase0_epoch2_screen.json", gate)
                if gate["decision"] == "FQRF_FOCUS_NOGO": break

        elapsed = time.perf_counter() - started
        if args.smoke_steps:
            runtime = {"smoke": True, "steps": optimizer.global_step, "epochs": completed, "all_finite": True, "train_seconds": elapsed, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "gradients": last_grad, "validation_accessed": False, "training_paths_opened_parent": len(accesses)}
            write_json(output / "fqrf_phase0_runtime.json", runtime); print("FQRF_SMOKE_PASS " + json.dumps(runtime), flush=True); return
        endpoint, digest = checkpoint(model, output)
        if gate and gate["decision"] == "FQRF_FOCUS_NOGO": decision = "FQRF_FOCUS_NOGO"
        else: gate = apply_final_gate(final_summary); decision = gate["decision"]
        runtime = {"smoke": False, "steps": optimizer.global_step, "epochs": completed, "all_finite": all_finite and final_summary["all_finite"], "train_seconds": elapsed, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(), "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024**3, "gradients": last_grad, "validation_accessed": False, "test_accessed": False, "luad_accessed": False, "training_samples_consumed": optimizer.global_step * 20, "training_paths_opened_parent": len(accesses), "decision": decision, "checkpoint": str(endpoint), "checkpoint_sha256": digest}
        write_json(output / "fqrf_phase0_runtime.json", runtime)
        result = {**runtime, "source_commit": source_commit, "final_summary": final_summary, "gate": gate}
        write_json(output / "fqrf_phase0_gate_result.json", result)
        report = render_report(output, result)
        print("FQRF_FINAL " + json.dumps({"decision": decision, "report": str(report), "gate": gate}), flush=True)
        print(f"DECISION = {decision}", flush=True)
    except Exception as error:
        failure = {"decision": "FQRF_ENGINEERING_BLOCKED", "error": repr(error), "source_commit": source_commit, "epochs": completed, "steps": optimizer.global_step, "all_finite": False, "final_summary": final_summary or {}, "gate": {}}
        write_json(output / "fqrf_phase0_engineering_failure.json", failure); render_report(output, failure)
        print("DECISION = FQRF_ENGINEERING_BLOCKED", flush=True); raise


if __name__ == "__main__": main()

