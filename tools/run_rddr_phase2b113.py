#!/usr/bin/env python3
"""Run the frozen zero-step Phase-2B1.13 parameter-gradient audit.

This executable never calls ``optimizer.step`` on the formal model, never
writes a checkpoint, and only reads BCSS training/validation data.  It replays
the first 128 Phase-2B1.12 transformed minibatches by manifest and uses the
unchanged C0 state for all primary Track-T and Track-V gradients.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch
import random
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from network.resnet38_cls import Net, Net_CAM
from tool.GenDataset import Stage1_InferDataset, Stage1_TrainDataset
from train_sshr import set_seed
from tools.rddr_phase2b112_common import (
    EPS, adjudicate, auxiliary_forward, make_optimizer, random_gate, restore_rng,
)
from tools.rddr_phase2b113_common import (
    A0, BOOTSTRAP_REPLICATES, CHECKPOINT_SHA256, DIAGNOSTIC_BATCHES,
    HISTORICAL_DECISIONS, LAMBDA_ADT, POPULATION_NAMES, PREFIX, TRAIN_BATCHES,
    accumulate, add_gradients, approved, block_name, bootstrap,
    cancellation_index, cpu_grad_map, dry_run_clone_error, max_relative_error,
    oracle_metrics, optimizer_specs, parameter_manifest, relative_difference,
    require, scale_gradients, sha256, subset, tensor_digest, vector_cosine,
    vector_dot, vector_metrics, vector_norm, virtual_fresh_update, write_csv,
    write_json, zero_aggregate,
)


C0_RUN = Path("/home/duyanhong/sshr-official-25ep-final-retry2-20260815")
C0_CHECKPOINT = C0_RUN / "runs/bcss_seed42/checkpoints/stage1_last.pth"
DATA_ROOT = Path("/home/duyanhong/reseg-data/raw/BCSS-WSSS")
PHASE112 = Path("/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2")
NATIVE = Path("/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz")
REQUIRED_112 = (
    "rddr_phase2b112_optimizer_provenance.json",
    "rddr_phase2b112_identity_step0.json",
    "rddr_phase2b112_batch_manifest.json",
    "rddr_phase2b112_training_curve.csv",
    "rddr_phase2b112_verification.json",
    "rddr_phase2b112_summary.json",
    "rddr_phase2b112_runtime.json",
    "rddr_phase2b112_lambda_calibration.json",
    "snapshot_0000_B.npz",
    "checkpoint_step0000_shared.pth",
    "checkpoint_step0250_B.pth", "checkpoint_step0250_A.pth", "checkpoint_step0250_R.pth",
    "checkpoint_step0500_B.pth", "checkpoint_step0500_A.pth", "checkpoint_step0500_R.pth",
)


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _state_hash(model) -> str:
    return tensor_digest(model.state_dict().items())


def _bn_hash(model) -> str:
    return tensor_digest((name, value) for name, value in model.state_dict().items()
                         if "running_" in name or "num_batches_tracked" in name)


def _save_flags(model):
    return ([(module, module.training) for module in model.modules()],
            [(parameter, parameter.requires_grad) for parameter in model.parameters()])


def _restore_flags(flags):
    modules, parameters = flags
    for module, training in modules:
        module.training = training
    for parameter, requires_grad in parameters:
        parameter.requires_grad_(requires_grad)


def _load_model(checkpoint: Path = C0_CHECKPOINT):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=checkpoint == C0_CHECKPOINT)
    state = payload if checkpoint == C0_CHECKPOINT else payload["model"]
    model = Net_CAM(4)
    loaded = model.load_state_dict(state, strict=True)
    require(not loaded.missing_keys and not loaded.unexpected_keys, f"Strict load failed: {checkpoint}")
    return model, payload


def _main_gradients(model, loss, names):
    named = dict(model.named_parameters())
    active = [(name, named[name]) for name in names if named[name].requires_grad]
    gradients = torch.autograd.grad(loss, tuple(parameter for _, parameter in active), allow_unused=True)
    by_name = {name: gradient for (name, _), gradient in zip(active, gradients)}
    ordered = [by_name.get(name) for name in names]
    return cpu_grad_map(names, ordered)


def _aux_gradients(loss, leaves, names, retain_graph):
    require(list(leaves) == list(names), "Auxiliary leaf order differs from manifest")
    gradients = torch.autograd.grad(loss, tuple(leaves.values()), retain_graph=retain_graph,
                                    allow_unused=False)
    return cpu_grad_map(names, gradients)


def _classification_loss(outputs, labels):
    return sum(weight * F.multilabel_soft_margin_loss(output, labels, weight=None)
               for weight, output in zip((0.10, 0.15, 0.25, 0.50), outputs[:4]))


def _forward_graph(model, images, labels, names, training):
    flags = _save_flags(model)
    model.train(training)
    capture = {}
    handles = [
        model.b4.bn_branch2a.register_forward_pre_hook(
            lambda _module, args: capture.update(feat56=args[0].detach())),
        model.hfrm_28_1.register_forward_hook(
            lambda _module, args, _output: capture.update(raw=args[0].detach())),
    ]
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = Net.forward(model, images)
            main_loss = _classification_loss(outputs, labels)
            auxiliary_logits, leaves = auxiliary_forward(model, capture["feat56"])
            observed_raw = F.conv2d(capture["raw"], model.ic1.weight.detach(), model.ic1.bias.detach())
        require(torch.equal(auxiliary_logits.detach(), observed_raw.detach()),
                "Auxiliary raw-shallow probe parity failed")
        deep_probability = outputs[8].detach().float().softmax(1)
        raw_probability = auxiliary_logits.detach().float().softmax(1)
        main_gradients = _main_gradients(model, main_loss, names)
        return {
            "main_gradients": main_gradients,
            "main_loss": float(main_loss.detach()),
            "auxiliary_logits": auxiliary_logits,
            "leaves": leaves,
            "raw_probability": raw_probability,
            "deep_probability": deep_probability,
            "flags": flags,
            "handles": handles,
        }
    except Exception:
        for handle in handles:
            handle.remove()
        _restore_flags(flags)
        raise


def _close_graph(bundle):
    for handle in bundle["handles"]:
        handle.remove()
    _restore_flags(bundle["flags"])


def _training_triplet(model, images, labels, names, random_rng):
    bundle = _forward_graph(model, images, labels, names, training=True)
    try:
        evidence = adjudicate(bundle["raw_probability"], bundle["deep_probability"])
        counts = evidence["gate"].sum(1).cpu().numpy()
        random_mask = random_gate(counts, random_rng, images.device)
        require(np.array_equal(random_mask.sum(1).cpu().numpy(), counts), "Random rate mismatch")
        probability = bundle["auxiliary_logits"].float().softmax(1)
        target = bundle["deep_probability"].reshape_as(probability)
        q = evidence["q"].detach().reshape(probability.shape[0], *probability.shape[2:])
        kl = (target * ((target + EPS).log() - (probability + EPS).log())).sum(1)
        ctx_weight = q * evidence["gate"].reshape_as(q)
        rnd_weight = q * random_mask.reshape_as(q)
        ctx_loss = (ctx_weight * kl).sum() / (ctx_weight.sum() + EPS)
        rnd_loss = (rnd_weight * kl).sum() / (rnd_weight.sum() + EPS)
        ctx = _aux_gradients(ctx_loss, bundle["leaves"], names, retain_graph=True)
        rnd = _aux_gradients(rnd_loss, bundle["leaves"], names, retain_graph=False)
        return bundle["main_gradients"], ctx, rnd, {
            "main_loss": bundle["main_loss"],
            "ctx_aux_loss": float(ctx_loss.detach()),
            "rnd_aux_loss": float(rnd_loss.detach()),
            "ctx_active_pixels": int(counts.sum()),
            "rnd_active_pixels": int(random_mask.sum()),
            "active_fraction": float(counts.sum() / evidence["gate"].numel()),
            "random_rate_exact": bool(np.array_equal(random_mask.sum(1).cpu().numpy(), counts)),
        }
    finally:
        _close_graph(bundle)


def _manual_training_batch(dataset, item):
    names, images, labels = [], [], []
    require(len(item["names"]) == len(item["augmentation"]) == 20, "Manifest batch is not batch20")
    for expected_name, augmentation in zip(item["names"], item["augmentation"]):
        index = int(augmentation["index"])
        choices = [1.0 if augmentation["horizontal_flip"] else 0.0,
                   1.0 if augmentation["vertical_flip"] else 0.0]
        with patch("tool.GenDataset.random.random", side_effect=choices):
            name, image, label = dataset[index]
        require(name == expected_name, f"Manifest index/name mismatch: {index}/{name}/{expected_name}")
        names.append(name); images.append(image); labels.append(label)
    image_tensor, label_tensor = torch.stack(images), torch.stack(labels)
    actual = tensor_digest((("image", image_tensor), ("label", label_tensor)))
    require(actual == item["tensor_sha256"],
            f"TRACK_T_REPLAY_BLOCKED step={item['step']} expected={item['tensor_sha256']} actual={actual}")
    return names, image_tensor, label_tensor, actual


def _initialize_aggregates(*gradients):
    return [zero_aggregate(gradient) for gradient in gradients]


def _aggregate_metrics(g_main, g_ctx, g_rnd, update_a, update_r):
    total_a = add_gradients(g_main, g_ctx, LAMBDA_ADT)
    total_r = add_gradients(g_main, g_rnd, LAMBDA_ADT)
    from tools.rddr_phase2b113_common import direction_difference, norm_ratio
    d_dir, formula = direction_difference(g_ctx, g_rnd)
    return {
        "C_aux": vector_cosine(g_ctx, g_rnd),
        "D_dir": d_dir,
        "D_dir_formula": formula,
        "R_norm": norm_ratio(g_ctx, g_rnd),
        "C_ctx_main": vector_cosine(g_ctx, g_main),
        "C_rnd_main": vector_cosine(g_rnd, g_main),
        "DeltaC_main": vector_cosine(g_ctx, g_main) - vector_cosine(g_rnd, g_main),
        "C_tot": vector_cosine(total_a, total_r),
        "rho_ctx": vector_norm(scale_gradients(add_gradients(g_ctx, g_rnd, -1.0), LAMBDA_ADT)) /
                   (0.5 * (vector_norm(total_a) + vector_norm(total_r)) + EPS),
        "C_update": vector_cosine(update_a, update_r),
        "rho_update": relative_difference(update_a, update_r),
    }


def run_track_t(model, names, parameters, specs, manifest, output):
    training = manifest["training"][:TRAIN_BATCHES]
    require(len(training) == TRAIN_BATCHES and [item["step"] for item in training] == list(range(1, 129)),
            "Frozen Track-T manifest does not contain exact step1-128")
    dataset = Stage1_TrainDataset(str(DATA_ROOT / "training"), dataset="bcss", img_size=224)
    require(len(dataset) == 23_422, "BCSS training size changed")
    step0 = torch.load(PHASE112 / "checkpoint_step0000_shared.pth", map_location="cpu", weights_only=False)
    require(int(step0["step"]) == 0 and float(step0["lambda_value"]) == LAMBDA_ADT,
            "Step0 checkpoint provenance changed")
    restore_rng(step0["rng"])
    random_rng = np.random.default_rng(42)
    rows = []
    agg_main = agg_ctx = agg_rnd = agg_update_a = agg_update_r = None
    clone_errors = []
    for batch_number, item in enumerate(training, 1):
        batch_names, images, labels, digest = _manual_training_batch(dataset, item)
        main, ctx, rnd, detail = _training_triplet(
            model, images.cuda(non_blocking=True), labels.cuda(non_blocking=True), names, random_rng)
        metrics, update_a, update_r = vector_metrics(main, ctx, rnd, parameters, specs)
        if agg_main is None:
            agg_main, agg_ctx, agg_rnd, agg_update_a, agg_update_r = _initialize_aggregates(
                main, ctx, rnd, update_a, update_r)
        for aggregate, gradient in ((agg_main, main), (agg_ctx, ctx), (agg_rnd, rnd),
                                    (agg_update_a, update_a), (agg_update_r, update_r)):
            accumulate(aggregate, gradient)
        if batch_number == 1:
            total_a = add_gradients(main, ctx, LAMBDA_ADT)
            total_r = add_gradients(main, rnd, LAMBDA_ADT)
            clone_errors = [dry_run_clone_error(parameters, total_a, specs),
                            dry_run_clone_error(parameters, total_r, specs)]
            require(max(clone_errors) <= 2e-7, f"Virtual optimizer clone mismatch: {clone_errors}")
        rows.append({"batch": batch_number, "step": int(item["step"]), "tensor_sha256": digest,
                     **detail, **metrics, "finite": True})
        if batch_number % 8 == 0:
            write_csv(output / f"{PREFIX}train_batch_metrics.csv", rows)
            print(json.dumps({"phase": "track_t", "batch": batch_number, "total": TRAIN_BATCHES,
                              "C_aux": metrics["C_aux"], "rho_update": metrics["rho_update"]}), flush=True)
        del images, labels, main, ctx, rnd, update_a, update_r
        torch.cuda.empty_cache()
    aggregate = _aggregate_metrics(agg_main, agg_ctx, agg_rnd, agg_update_a, agg_update_r)
    block_rows = []
    for block in ("b4", "b4_1", "b4_2", "b4_3", "b4_4", "b4_5", "bn45"):
        block_rows.append({"block": block, **_aggregate_metrics(
            subset(agg_main, block), subset(agg_ctx, block), subset(agg_rnd, block),
            subset(agg_update_a, block), subset(agg_update_r, block))})
    bootstrap_rows = bootstrap(rows, (
        ("C_aux", "mean"), ("C_aux", "median"), ("D_dir", "mean"),
        ("rho_ctx", "mean"), ("C_tot", "mean"), ("rho_update", "mean"),
        ("DeltaC_main", "mean"),
    ))
    aggregate.update({
        "batches": TRAIN_BATCHES,
        "batch_size": 20,
        "replay_status": "EXACT_TENSOR_REPLAY_PASS",
        "all_tensor_sha256_exact": True,
        "lambda_ADT": LAMBDA_ADT,
        "virtual_optimizer_clone_max_relative_error": max(clone_errors),
        "virtual_optimizer_formal_state_mutated": False,
        "random_seed": 42,
        "random_rate_matching_exact": all(row["random_rate_exact"] for row in rows),
    })
    write_csv(output / f"{PREFIX}train_batch_metrics.csv", rows)
    write_json(output / f"{PREFIX}train_aggregate.json", aggregate)
    write_csv(output / f"{PREFIX}train_blockwise.csv", block_rows)
    write_csv(output / f"{PREFIX}train_bootstrap.csv", bootstrap_rows)
    return aggregate, rows, block_rows


def _parse_image_labels(names):
    labels = []
    for name in names:
        encoded = name.split("[", 1)[1].split("]", 1)[0]
        require(len(encoded) >= 4 and set(encoded[:4]) <= {"0", "1"}, f"Invalid BCSS image label: {name}")
        labels.append([int(value) for value in encoded[:4]])
    return torch.tensor(labels, dtype=torch.float32)


def _frozen_populations(snapshot):
    truth = snapshot["truth"]
    valid = truth < 4
    raw_ok = snapshot["ps"].argmax(1) == truth
    deep_ok = snapshot["pd"].argmax(1) == truth
    populations = {
        "Deep-Win_0": valid & ~raw_ok & deep_ok,
        "Shallow-Win_0": valid & raw_ok & ~deep_ok,
        "Both-Wrong_0": valid & ~raw_ok & ~deep_ok,
        "Stable-Correct_0": valid & raw_ok & deep_ok,
        "Raw-Wrong_0": valid & ~raw_ok,
        "Raw-Correct_0": valid & raw_ok,
        "Top20_q0": valid & snapshot["top20"],
        "boundary": valid & snapshot["boundary"],
        "interior": valid & ~snapshot["boundary"],
    }
    populations.update({f"class{index}": truth == index for index in range(4)})
    partition = sum(populations[name].astype(np.int8) for name in POPULATION_NAMES)
    require(np.array_equal(partition, valid.astype(np.int8)), "Frozen DW/SW/BW/SC partition is not exhaustive")
    return populations


def _load_snapshot():
    with np.load(PHASE112 / "snapshot_0000_B.npz", allow_pickle=False) as archive:
        required = ("names", "truth", "ps", "pd", "q", "delta", "boundary", "top20")
        require(set(required).issubset(archive.files), "Step0 snapshot fields missing")
        result = {key: archive[key] for key in required}
    require(result["names"].shape == (3418,) and result["truth"].shape == (3418, 784),
            "Frozen validation population shape changed")
    require(result["names"].astype(str).tolist() == sorted(result["names"].astype(str).tolist()),
            "Frozen validation names are not sorted")
    result["names"] = result["names"].astype(str)
    result["boundary"] = result["boundary"].astype(bool)
    result["top20"] = result["top20"].astype(bool)
    return result


def _population_rows(mode, gradients, oracle, counts):
    denominator = vector_dot(oracle, oracle)
    norms = {name: vector_norm(gradients[name]) for name in POPULATION_NAMES}
    total_norm = sum(norms.values())
    return [{
        "mode": mode,
        "population": name,
        "n_pixels": int(counts[name]),
        "gradient_norm": norms[name],
        "norm_share": norms[name] / (total_norm + EPS),
        "oracle_projection": vector_dot(gradients[name], oracle) / (denominator + EPS),
    } for name in POPULATION_NAMES]


def run_track_v(model, names, parameters, specs, output):
    snapshot = _load_snapshot()
    populations = _frozen_populations(snapshot)
    counts = {name: int(mask.sum()) for name, mask in populations.items()}
    dataset = Stage1_InferDataset(str(DATA_ROOT / "val" / "img"), img_size=224)
    dataset.object = sorted(dataset.object)
    actual_names = [Path(path).stem for path in dataset.object]
    require(actual_names == snapshot["names"].tolist(), "Canonical validation names differ from frozen step0")
    loader = DataLoader(dataset, batch_size=20, shuffle=False, num_workers=0, pin_memory=True)
    random_rng = np.random.default_rng(42)
    oracle_rows = []
    population_aggregates = {mode: {name: None for name in POPULATION_NAMES} for mode in ("context", "random")}
    aggregate_oracle_numerator = aggregate_ctx_numerator = aggregate_rnd_numerator = None
    oracle_pixel_total = 0
    ctx_denominator_total = rnd_denominator_total = 0.0
    max_group_error = 0.0
    max_raw_probability_difference = max_deep_probability_difference = 0.0
    start = 0
    for batch_number, (batch_names, images) in enumerate(loader, 1):
        batch_names = list(batch_names)
        end = start + len(batch_names)
        require(batch_names == actual_names[start:end], "Validation batch order changed")
        labels = _parse_image_labels(batch_names)
        bundle = _forward_graph(model, images.cuda(non_blocking=True), labels.cuda(non_blocking=True), names,
                                training=False)
        try:
            shape = (len(batch_names), 4, 28, 28)
            frozen_pd = torch.from_numpy(snapshot["pd"][start:end].reshape(shape)).cuda()
            frozen_q = torch.from_numpy(snapshot["q"][start:end].reshape(len(batch_names), 28, 28)).cuda()
            frozen_gate = torch.from_numpy((snapshot["delta"][start:end] > 0).reshape(len(batch_names), 28, 28)).cuda()
            current_raw = bundle["raw_probability"].reshape(shape)
            current_deep = bundle["deep_probability"].reshape(shape)
            max_raw_probability_difference = max(max_raw_probability_difference,
                float((current_raw - torch.from_numpy(snapshot["ps"][start:end].reshape(shape)).cuda()).abs().max()))
            max_deep_probability_difference = max(max_deep_probability_difference,
                float((current_deep - frozen_pd).abs().max()))
            active_counts = frozen_gate.flatten(1).sum(1).cpu().numpy()
            frozen_random = random_gate(active_counts, random_rng, frozen_gate.device).reshape_as(frozen_gate)
            require(np.array_equal(frozen_random.flatten(1).sum(1).cpu().numpy(), active_counts),
                    "Validation random gate is not exact rate matched")
            probability = bundle["auxiliary_logits"].float().softmax(1)
            kl = (frozen_pd * ((frozen_pd + EPS).log() - (probability + EPS).log())).sum(1)
            ctx_weight, rnd_weight = frozen_q * frozen_gate, frozen_q * frozen_random
            ctx_numerator, rnd_numerator = (ctx_weight * kl).sum(), (rnd_weight * kl).sum()
            ctx_denominator, rnd_denominator = float(ctx_weight.sum()), float(rnd_weight.sum())
            ctx_numerator_gradient = _aux_gradients(ctx_numerator, bundle["leaves"], names, True)
            rnd_numerator_gradient = _aux_gradients(rnd_numerator, bundle["leaves"], names, True)
            g_ctx = scale_gradients(ctx_numerator_gradient, 1.0 / (ctx_denominator + EPS))
            g_rnd = scale_gradients(rnd_numerator_gradient, 1.0 / (rnd_denominator + EPS))
            truth = torch.from_numpy(snapshot["truth"][start:end].reshape(len(batch_names), 28, 28)).cuda().long()
            valid = truth < 4
            require(bool(valid.any()), "Validation minibatch has no foreground oracle pixels")
            oracle_logits = bundle["auxiliary_logits"].float()
            oracle_numerator = F.cross_entropy(oracle_logits.permute(0, 2, 3, 1)[valid], truth[valid],
                                               reduction="sum")
            oracle_numerator_gradient = _aux_gradients(oracle_numerator, bundle["leaves"], names, True)
            g_oracle = scale_gradients(oracle_numerator_gradient, 1.0 / int(valid.sum()))
            group_sums = {mode: zero_aggregate(g_ctx) for mode in ("context", "random")}
            for population_index, population_name in enumerate(POPULATION_NAMES):
                mask = torch.from_numpy(populations[population_name][start:end].reshape(len(batch_names), 28, 28)).cuda()
                for mode, weight in (("context", ctx_weight), ("random", rnd_weight)):
                    numerator = (weight * kl * mask).sum()
                    gradient = _aux_gradients(numerator, bundle["leaves"], names, True)
                    accumulate(group_sums[mode], gradient)
                    if population_aggregates[mode][population_name] is None:
                        population_aggregates[mode][population_name] = zero_aggregate(gradient)
                    accumulate(population_aggregates[mode][population_name], gradient)
                    del gradient, numerator
            foreground = valid
            ctx_fg = _aux_gradients((ctx_weight * kl * foreground).sum(), bundle["leaves"], names, True)
            rnd_fg = _aux_gradients((rnd_weight * kl * foreground).sum(), bundle["leaves"], names, False)
            max_group_error = max(max_group_error, max_relative_error(group_sums["context"], ctx_fg),
                                  max_relative_error(group_sums["random"], rnd_fg))
            metrics, update_a, update_r = vector_metrics(
                bundle["main_gradients"], g_ctx, g_rnd, parameters, specs)
            oracle = oracle_metrics(g_ctx, g_rnd, g_oracle, update_a, update_r)
            oracle_rows.append({
                "batch": batch_number, "start_index": start, "end_index_exclusive": end,
                "n_images": len(batch_names), "oracle_foreground_pixels": int(valid.sum()),
                "ctx_active_pixels": int(frozen_gate.sum()), "rnd_active_pixels": int(frozen_random.sum()),
                "random_rate_exact": bool(torch.equal(frozen_gate.flatten(1).sum(1), frozen_random.flatten(1).sum(1))),
                **oracle, **{key: metrics[key] for key in ("C_aux", "D_dir", "R_norm", "C_tot", "rho_ctx")},
                "finite": True,
            })
            if aggregate_oracle_numerator is None:
                aggregate_oracle_numerator, aggregate_ctx_numerator, aggregate_rnd_numerator = _initialize_aggregates(
                    oracle_numerator_gradient, ctx_numerator_gradient, rnd_numerator_gradient)
            accumulate(aggregate_oracle_numerator, oracle_numerator_gradient)
            accumulate(aggregate_ctx_numerator, ctx_numerator_gradient)
            accumulate(aggregate_rnd_numerator, rnd_numerator_gradient)
            oracle_pixel_total += int(valid.sum())
            ctx_denominator_total += ctx_denominator
            rnd_denominator_total += rnd_denominator
            if batch_number % 10 == 0:
                write_csv(output / f"{PREFIX}oracle_batch_metrics.csv", oracle_rows)
                print(json.dumps({"phase": "track_v", "batch": batch_number, "images": end,
                                  "total_images": 3418, "DeltaC_oracle": oracle["DeltaC_oracle"],
                                  "Adv_oracle": oracle["Adv_oracle"]}), flush=True)
        finally:
            _close_graph(bundle)
        start = end
        del images, labels
        torch.cuda.empty_cache()
    require(start == 3418, f"Validation coverage incomplete: {start}")
    require(max_group_error <= 2e-6, f"Population gradient identity failed: {max_group_error}")
    aggregate_oracle = scale_gradients(aggregate_oracle_numerator, 1.0 / oracle_pixel_total)
    aggregate_ctx = scale_gradients(aggregate_ctx_numerator, 1.0 / (ctx_denominator_total + EPS))
    aggregate_rnd = scale_gradients(aggregate_rnd_numerator, 1.0 / (rnd_denominator_total + EPS))
    population_rows = []
    for mode in ("context", "random"):
        population_rows.extend(_population_rows(mode, population_aggregates[mode], aggregate_oracle, counts))
    pairwise_rows = []
    pairs = (("Deep-Win_0", "Shallow-Win_0"), ("Deep-Win_0", "Both-Wrong_0"),
             ("Deep-Win_0", "Stable-Correct_0"), ("Shallow-Win_0", "Stable-Correct_0"),
             ("Both-Wrong_0", "Stable-Correct_0"))
    for mode in ("context", "random"):
        for left, right in pairs:
            pairwise_rows.append({"mode": mode, "left": left, "right": right,
                                  "cosine": vector_cosine(population_aggregates[mode][left],
                                                          population_aggregates[mode][right])})
    projection_rows = list(population_rows)
    block_rows = []
    oracle_sq = vector_dot(aggregate_oracle, aggregate_oracle)
    for mode in ("context", "random"):
        for block in ("b4", "b4_1", "b4_2", "b4_3", "b4_4", "b4_5", "bn45"):
            vectors = [subset(population_aggregates[mode][name], block) for name in POPULATION_NAMES]
            row = {"mode": mode, "block": block, "CancellationIndex": cancellation_index(vectors)}
            for name, vector in zip(POPULATION_NAMES, vectors):
                block_oracle = subset(aggregate_oracle, block)
                row[f"P_{name}"] = (
                    vector_dot(vector, block_oracle) /
                    (vector_dot(block_oracle, block_oracle) + EPS)
                )
            block_rows.append(row)
    context_ci = cancellation_index([population_aggregates["context"][name] for name in POPULATION_NAMES])
    random_ci = cancellation_index([population_aggregates["random"][name] for name in POPULATION_NAMES])
    projections = {row["population"]: row["oracle_projection"]
                   for row in population_rows if row["mode"] == "context"}
    aggregate = {
        "validation_images": 3418,
        "validation_minibatches": len(oracle_rows),
        "batch_size": 20,
        "last_batch_size": oracle_rows[-1]["n_images"],
        "canonical_input": True,
        "frozen_step0_probe_gate": True,
        "gate_gt_blind": True,
        "oracle_gt_only": True,
        "oracle_foreground_pixels": oracle_pixel_total,
        "background_and_ignore_excluded_from_oracle": True,
        "population_counts": counts,
        "population_partition_exhaustive": True,
        "population_group_sum_max_relative_error": max_group_error,
        "random_rate_matching_exact": all(row["random_rate_exact"] for row in oracle_rows),
        "max_batch20_vs_frozen_batch1_raw_probability_abs_difference": max_raw_probability_difference,
        "max_batch20_vs_frozen_batch1_deep_probability_abs_difference": max_deep_probability_difference,
        "aggregate_C_aux": vector_cosine(aggregate_ctx, aggregate_rnd),
        "aggregate_DeltaC_oracle": vector_cosine(aggregate_ctx, aggregate_oracle) -
                                           vector_cosine(aggregate_rnd, aggregate_oracle),
        "aggregate_DeltaP_oracle": (vector_dot(aggregate_ctx, aggregate_oracle) -
                                         vector_dot(aggregate_rnd, aggregate_oracle)) / (oracle_sq + EPS),
        "context_CancellationIndex": context_ci,
        "random_CancellationIndex": random_ci,
        "P_DW_ctx": projections["Deep-Win_0"],
        "P_SW_ctx": projections["Shallow-Win_0"],
        "P_BW_ctx": projections["Both-Wrong_0"],
        "P_SC_ctx": projections["Stable-Correct_0"],
    }
    bootstrap_rows = bootstrap(oracle_rows, (("DeltaC_oracle", "mean"),
                                             ("DeltaP_oracle", "mean"),
                                             ("Adv_oracle", "mean")))
    write_csv(output / f"{PREFIX}oracle_batch_metrics.csv", oracle_rows)
    write_json(output / f"{PREFIX}oracle_aggregate.json", aggregate)
    write_csv(output / f"{PREFIX}oracle_bootstrap.csv", bootstrap_rows)
    write_csv(output / f"{PREFIX}population_gradients.csv", population_rows)
    write_csv(output / f"{PREFIX}population_pairwise_cosines.csv", pairwise_rows)
    write_csv(output / f"{PREFIX}population_oracle_projection.csv", projection_rows)
    write_csv(output / f"{PREFIX}population_blockwise.csv", block_rows)
    return aggregate, oracle_rows


def _snapshot_states():
    shared = PHASE112 / "checkpoint_step0000_shared.pth"
    yield 0, "B", shared
    yield 0, "A", shared
    yield 0, "R", shared
    for step in (250, 500):
        for arm in ("B", "A", "R"):
            yield step, arm, PHASE112 / f"checkpoint_step{step:04d}_{arm}.pth"


def run_snapshot_attribution(names, manifest, output):
    diagnostic = manifest["training"][:DIAGNOSTIC_BATCHES]
    dataset = Stage1_TrainDataset(str(DATA_ROOT / "training"), dataset="bcss", img_size=224)
    cached_batches = [_manual_training_batch(dataset, item) for item in diagnostic]
    rows = []
    shared_result = None
    for step, arm, checkpoint in _snapshot_states():
        if step == 0 and shared_result is not None:
            rows.append({**shared_result, "state_arm": arm})
            continue
        model, payload = _load_model(checkpoint)
        model.cuda()
        flags = _save_flags(model)
        restore_rng(payload["rng"])
        random_rng = np.random.default_rng(42)
        if step > 0 and "random_gate_rng" in payload:
            random_rng.bit_generator.state = copy.deepcopy(payload["random_gate_rng"])
        state_before = _state_hash(model)
        parameters = {name: parameter.detach().cpu().clone()
                      for name, parameter in model.named_parameters() if approved(name)}
        optimizer = make_optimizer(model)
        manifest_rows, state_names = parameter_manifest(model, optimizer.param_groups)
        specs = optimizer_specs(manifest_rows)
        agg_main = agg_ctx = agg_rnd = None
        for batch_names, images, labels, _digest in cached_batches:
            main, ctx, rnd, _detail = _training_triplet(
                model, images.cuda(non_blocking=True), labels.cuda(non_blocking=True), state_names, random_rng)
            if agg_main is None:
                agg_main, agg_ctx, agg_rnd = _initialize_aggregates(main, ctx, rnd)
            for aggregate, gradient in ((agg_main, main), (agg_ctx, ctx), (agg_rnd, rnd)):
                accumulate(aggregate, gradient)
            del main, ctx, rnd
        result = {"step": step, "state_arm": arm, "checkpoint": str(checkpoint),
                  "checkpoint_sha256": sha256(checkpoint), "diagnostic_minibatches": DIAGNOSTIC_BATCHES,
                  **{key: value for key, value in _aggregate_metrics(
                      agg_main, agg_ctx, agg_rnd,
                      virtual_fresh_update(parameters, add_gradients(agg_main, agg_ctx, LAMBDA_ADT), specs),
                      virtual_fresh_update(parameters, add_gradients(agg_main, agg_rnd, LAMBDA_ADT), specs)).items()
                     if key in ("C_aux", "D_dir", "rho_ctx", "C_tot")},
                  "state_unchanged": _state_hash(model) == state_before}
        require(result["state_unchanged"], f"Snapshot attribution changed {step}/{arm}")
        rows.append(result)
        if step == 0:
            shared_result = dict(result)
        _restore_flags(flags)
        del model, optimizer
        torch.cuda.empty_cache()
        print(json.dumps({"phase": "snapshot_attribution", "step": step, "arm": arm}), flush=True)
    write_csv(output / f"{PREFIX}snapshot_attribution.csv", rows)
    return rows


def preflight(output):
    require(not subprocess.check_output(["git", "diff", A0, "--", "network", "tool", "train_sshr.py"], cwd=ROOT),
            "Frozen A0 model/training/inference sources changed")
    require(C0_CHECKPOINT.is_file() and sha256(C0_CHECKPOINT) == CHECKPOINT_SHA256,
            "C0 checkpoint missing or SHA changed")
    for name in REQUIRED_112:
        require((PHASE112 / name).is_file(), f"Missing frozen Phase2B1.12 evidence: {name}")
    runtime112 = _json(PHASE112 / "rddr_phase2b112_runtime.json")
    verification112 = _json(PHASE112 / "rddr_phase2b112_verification.json")
    summary112 = _json(PHASE112 / "rddr_phase2b112_summary.json")
    calibration112 = _json(PHASE112 / "rddr_phase2b112_lambda_calibration.json")
    require(runtime112["completed"] and runtime112["steps_per_arm"] == {"B": 500, "A": 500, "R": 500},
            "Phase2B1.12 formal run incomplete")
    require(verification112["passed"] and verification112.get("post_analysis") is True,
            "Phase2B1.12 independent verification is not final PASS")
    require(summary112["decision"] == HISTORICAL_DECISIONS["phase2b112"], "Phase2B1.12 decision changed")
    require(float(calibration112["lambda_value"]) == LAMBDA_ADT, "Frozen lambda changed")
    frozen_hashes = {name: sha256(PHASE112 / name) for name in REQUIRED_112}
    source_paths = [ROOT / path for path in ("network/resnet38_cls.py", "network/resnet38d.py",
                                             "tool/GenDataset.py", "tool/torchutils.py",
                                             "train_sshr.py")]
    model, _payload = _load_model()
    formal_optimizer = make_optimizer(model)
    require(not formal_optimizer.state, "Formal fresh optimizer state is not empty")
    manifest_rows, names = parameter_manifest(model, formal_optimizer.param_groups)
    write_csv(output / f"{PREFIX}parameter_manifest.csv", manifest_rows)
    provenance = {
        "status": "PASS",
        "pure_A0_commit": A0,
        "C0_checkpoint": str(C0_CHECKPOINT),
        "C0_checkpoint_sha256": CHECKPOINT_SHA256,
        "phase2b112_directory": str(PHASE112),
        "phase2b112_artifact_sha256": frozen_hashes,
        "phase2b112_decision": summary112["decision"],
        "historical_decisions": HISTORICAL_DECISIONS,
        "lambda_ADT": LAMBDA_ADT,
        "optimizer_provenance_sha256": frozen_hashes["rddr_phase2b112_optimizer_provenance.json"],
        "batch_manifest_sha256": frozen_hashes["rddr_phase2b112_batch_manifest.json"],
        "training_curve_sha256": frozen_hashes["rddr_phase2b112_training_curve.csv"],
        "step0_identity_sha256": frozen_hashes["rddr_phase2b112_identity_step0.json"],
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "approved_parameter_count": len(names),
        "approved_total_numel": sum(row["numel"] for row in manifest_rows),
        "formal_optimizer_state_empty": True,
        "no_training": True,
        "test_access": False,
        "luad_access": False,
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    write_json(output / f"{PREFIX}provenance.json", provenance)
    return model, formal_optimizer, manifest_rows, names, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {output}")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    access = set()
    phase = ["preflight"]

    def audit(event, event_args):
        if event == "open" and isinstance(event_args[0], (str, bytes)):
            path = os.fsdecode(event_args[0]).replace("\\", "/").lower()
            if "/reseg-data/" in path:
                require("/bcss-wsss/" in path and "/test/" not in path and "luad" not in path,
                        f"Unauthorized dataset access: {path}")
                if phase[0] == "track_t":
                    require("/training/" in path, f"Track T accessed non-training data: {path}")
                access.add(path)
    sys.addaudithook(audit)
    model, formal_optimizer, manifest_rows, names, provenance = preflight(output)
    if args.preflight_only:
        write_json(output / f"{PREFIX}runtime.json", {
            "completed": False, "preflight_only": True, "optimizer_steps": 0,
            "checkpoint_writes": 0, "test_access": False, "luad_access": False,
        })
        return
    require(torch.cuda.is_available(), "CUDA is required for the formal BF16 audit")
    parameters = {name: parameter.detach().cpu().clone()
                  for name, parameter in model.named_parameters() if approved(name)}
    specs = optimizer_specs(manifest_rows)
    formal_state_before = copy.deepcopy(formal_optimizer.state_dict())
    initial_state = _state_hash(model)
    initial_bn = _bn_hash(model)
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    manifest = _json(PHASE112 / "rddr_phase2b112_batch_manifest.json")
    phase[0] = "track_t"
    track_t, _train_rows, _blocks = run_track_t(model, names, parameters, specs, manifest, output)
    model.cpu(); torch.cuda.empty_cache()
    require(_state_hash(model) == initial_state and _bn_hash(model) == initial_bn,
            "Track T changed formal C0 state")
    del model
    phase[0] = "track_v"
    model_v, _payload = _load_model()
    state_v_before, bn_v_before = _state_hash(model_v), _bn_hash(model_v)
    model_v.cuda()
    track_v, _oracle_rows = run_track_v(model_v, names, parameters, specs, output)
    model_v.cpu(); torch.cuda.empty_cache()
    require(_state_hash(model_v) == state_v_before == initial_state and _bn_hash(model_v) == bn_v_before == initial_bn,
            "Track V changed formal C0 state")
    del model_v
    phase[0] = "snapshot"
    snapshots = run_snapshot_attribution(names, manifest, output)
    require(formal_optimizer.state_dict() == formal_state_before, "Formal optimizer state mutated")
    source_hashes_after = {path: sha256(path) for path in provenance["source_sha256"]}
    require(source_hashes_after == provenance["source_sha256"], "Frozen source hashes changed during audit")
    identity = {
        "model_state_hash_before": initial_state,
        "model_state_hash_after_track_t": initial_state,
        "model_state_hash_after_track_v": state_v_before,
        "model_state_unchanged": True,
        "bn_running_state_hash_before": initial_bn,
        "bn_running_state_hash_after": initial_bn,
        "bn_running_state_unchanged": True,
        "formal_optimizer_state_before": formal_state_before,
        "formal_optimizer_state_after": formal_optimizer.state_dict(),
        "formal_optimizer_state_unchanged": True,
        "source_hashes_before": provenance["source_sha256"],
        "source_hashes_after": source_hashes_after,
        "source_hashes_unchanged": True,
        "checkpoint_writes": 0,
        "optimizer_steps": 0,
    }
    write_json(output / f"{PREFIX}identity.json", identity)
    runtime = {
        "completed": True,
        "zero_step": True,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "track_t_batches": track_t["batches"],
        "track_v_images": track_v["validation_images"],
        "track_v_minibatches": track_v["validation_minibatches"],
        "snapshot_states": len(snapshots),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "precision": "BF16 network forward; FP32 parameter gradients; FP64 scalar/aggregate statistics",
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "all_gradients_finite": True,
        "state_unchanged": True,
        "test_access": False,
        "luad_access": False,
        "dataset_paths_accessed": len(access),
        "command": shlex.join([sys.executable, *sys.argv]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    write_json(output / f"{PREFIX}runtime.json", runtime)
    print(json.dumps({"phase": "raw_audit_complete", "seconds": runtime["seconds"],
                      "track_t_C_aux": track_t["C_aux"],
                      "track_v_DeltaC_oracle": track_v["aggregate_DeltaC_oracle"]}), flush=True)


if __name__ == "__main__":
    main()
