"""Fixed 32-batch, no-step training-gradient audit for frozen SSHR."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tool.GenDataset import Stage1_TrainDataset
from tools.hma_v0 import (
    GRADIENT_BATCHES,
    GRADIENT_BATCH_SIZE,
    IMAGE_SIZE,
    LOSS_WEIGHTS,
    SEED,
    STAGES,
)


BRANCHES = ("56", "28_1", "28_2", "deep")
PARAMETER_GROUPS = (
    "shared_early", "mid_backbone", "late_backbone", "fc8_deep_head",
    "hfrm_56", "hfrm_28_1", "hfrm_28_2",
)


def hash_named_tensors(items):
    digest = hashlib.sha256()
    for name, tensor in sorted(items, key=lambda item: item[0]):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_hashes(model):
    return {
        "parameter_sha256": hash_named_tensors(model.named_parameters()),
        "buffer_sha256": hash_named_tensors(model.named_buffers()),
    }


def _parameter_group(name):
    if name.startswith(("conv1a.", "b2.", "b2_", "b3.", "b3_")):
        return "shared_early"
    if name.startswith(("b4.", "b4_", "bn45.")):
        return "mid_backbone"
    if name.startswith(("b5.", "b5_", "bn52.", "b6.", "b6_", "b7.", "bn7.")):
        return "late_backbone"
    if name.startswith("fc8."):
        return "fc8_deep_head"
    if name.startswith("hfrm_56."):
        return "hfrm_56"
    if name.startswith("hfrm_28_1."):
        return "hfrm_28_1"
    if name.startswith("hfrm_28_2."):
        return "hfrm_28_2"
    return None


def parameter_inventory(model):
    items, groups = [], defaultdict(list)
    for name, parameter in model.named_parameters():
        # The released train()/eval() freezes the early stem and every BN.
        # autograd.grad must receive only the parameters that the official
        # training protocol actually leaves trainable.
        if not parameter.requires_grad:
            continue
        group = _parameter_group(name)
        if group is not None:
            index = len(items)
            items.append((name, parameter, group))
            groups[group].append(index)
    if any(not groups[name] for name in PARAMETER_GROUPS):
        missing = [name for name in PARAMETER_GROUPS if not groups[name]]
        raise AssertionError(f"Empty gradient parameter groups: {missing}")
    return items, groups


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _cosine_from_maps(left, right):
    dot = norm_left = norm_right = 0.0
    for name in left:
        if name not in right:
            continue
        lvalue, rvalue = left[name], right[name]
        dot += float((lvalue * rvalue).sum().item())
        norm_left += float((lvalue * lvalue).sum().item())
        norm_right += float((rvalue * rvalue).sum().item())
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return float(dot / np.sqrt(norm_left * norm_right))


def _cosine_tensor(left, right):
    left = left.reshape(-1).float()
    right = right.reshape(-1).float()
    denominator = left.norm() * right.norm()
    return float((left @ right / (denominator + 1e-20)).item())


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _component_norms(branch, named_gradients):
    output = {}
    for stage in STAGES:
        prefix = f"hfrm_{stage}."
        selected = {
            name[len(prefix):]: gradient
            for name, gradient in named_gradients.items()
            if name.startswith(prefix) and gradient is not None
        }

        def norm_for(predicate):
            total = sum(
                float((gradient.float() ** 2).sum().item())
                for name, gradient in selected.items() if predicate(name)
            )
            return float(np.sqrt(total))

        output[stage] = {
            "loss_branch": branch,
            "gamma_veto_grad_abs": norm_for(lambda name: name == "gamma_veto"),
            "gamma_context_grad_abs": norm_for(lambda name: name == "gamma_context"),
            "veto_mlp_grad_norm": norm_for(lambda name: name.startswith("veto_mlp.")),
            "context_conv_grad_norm": norm_for(lambda name: name.startswith("context_conv.")),
        }
    return output


def run_gradient_audit(model, train_root, num_workers=4, amp_dtype="bf16"):
    """Measure gradients without optimizer construction or parameter updates."""

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dataset = Stage1_TrainDataset(
        data_path=train_root, dataset="bcss", img_size=IMAGE_SIZE
    )
    if len(dataset) < GRADIENT_BATCHES * GRADIENT_BATCH_SIZE:
        raise AssertionError("Training dataset is too small for fixed gradient audit")
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        dataset,
        batch_size=GRADIENT_BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    parameter_items, group_indices = parameter_inventory(model)
    parameters = [item[1] for item in parameter_items]
    names = [item[0] for item in parameter_items]
    before = model_hashes(model)
    buffer_snapshot = {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }
    model.train()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else None
    rows, component_rows = [], []
    early_cosines = defaultdict(list)
    deep_cosines = defaultdict(list)
    deep_norms = defaultdict(list)

    for batch_index, (_, images, labels) in enumerate(loader):
        if batch_index >= GRADIENT_BATCHES:
            break
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=dtype, enabled=dtype is not None
        ):
            audit = model.forward_hfrm_audit(images, apply_deep_dropout=True)
            losses = {
                stage: LOSS_WEIGHTS[stage] * F.multilabel_soft_margin_loss(
                    audit["pooled_logits"]["full"][stage], labels
                )
                for stage in STAGES
            }
            losses["deep"] = LOSS_WEIGHTS["deep"] * F.multilabel_soft_margin_loss(
                audit["deep_pooled"], labels
            )

        early_gradients, feature_gradients = {}, {}
        for branch_index, branch in enumerate(BRANCHES):
            gradients = torch.autograd.grad(
                losses[branch],
                [*parameters, audit["feat_deep"]],
                retain_graph=branch_index < len(BRANCHES) - 1,
                allow_unused=True,
            )
            parameter_gradients = gradients[:-1]
            feature_gradient = gradients[-1]
            named_gradients = {
                name: gradient
                for name, gradient in zip(names, parameter_gradients)
                if gradient is not None
            }
            group_norms = {}
            for group, indices in group_indices.items():
                squared = 0.0
                for index in indices:
                    gradient = parameter_gradients[index]
                    if gradient is not None:
                        squared += float((gradient.float() ** 2).sum().item())
                group_norms[group] = float(np.sqrt(squared))
            total_norm = float(np.sqrt(sum(value * value for value in group_norms.values())))
            for group in PARAMETER_GROUPS:
                rows.append({
                    "batch": batch_index + 1,
                    "loss_branch": branch,
                    "loss_weight": LOSS_WEIGHTS[branch],
                    "weighted_loss": float(losses[branch].detach().float().item()),
                    "parameter_group": group,
                    "gradient_norm": group_norms[group],
                    "relative_norm": float(group_norms[group] / (total_norm + 1e-20)),
                })
            components = _component_norms(branch, named_gradients)
            for stage, values in components.items():
                component_rows.append({"batch": batch_index + 1, "stage": stage, **values})
            early_gradients[branch] = {
                name: gradient.detach().float().cpu()
                for name, gradient in named_gradients.items()
                if _parameter_group(name) == "shared_early"
            }
            if feature_gradient is None:
                feature_gradient = torch.zeros_like(audit["feat_deep"])
            # Keep the four branch maps on CPU while computing pairwise cosine;
            # retaining all [B,4096,H,W] FP32 maps on the GPU is unnecessary.
            feature_gradients[branch] = feature_gradient.detach().float().cpu()
            deep_norms[branch].append(float(feature_gradient.detach().float().norm().item()))

        for left_index, left in enumerate(BRANCHES):
            for right in BRANCHES[left_index:]:
                key = f"{left}__{right}"
                early_cosines[key].append(
                    _cosine_from_maps(early_gradients[left], early_gradients[right])
                )
                deep_cosines[key].append(
                    _cosine_tensor(feature_gradients[left], feature_gradients[right])
                )

        with torch.no_grad():
            current_buffers = dict(model.named_buffers())
            for name, snapshot in buffer_snapshot.items():
                current_buffers[name].copy_(snapshot)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("autograd.grad unexpectedly populated parameter .grad")
        print(f"GRADIENT_PROGRESS {batch_index + 1}/{GRADIENT_BATCHES}", flush=True)

    if len({row["batch"] for row in rows}) != GRADIENT_BATCHES:
        raise AssertionError("Gradient audit did not consume exactly 32 batches")
    with torch.no_grad():
        current_buffers = dict(model.named_buffers())
        for name, snapshot in buffer_snapshot.items():
            current_buffers[name].copy_(snapshot)
    model.eval()
    after = model_hashes(model)
    if before != after:
        raise AssertionError(f"Frozen model changed during gradient audit: {before} != {after}")

    group_summary = {}
    for branch in BRANCHES:
        group_summary[branch] = {}
        for group in PARAMETER_GROUPS:
            selected = [
                row for row in rows
                if row["loss_branch"] == branch and row["parameter_group"] == group
            ]
            group_summary[branch][group] = {
                "gradient_norm": _distribution([row["gradient_norm"] for row in selected]),
                "relative_norm": _distribution([row["relative_norm"] for row in selected]),
            }
    component_summary = {}
    for branch in BRANCHES:
        component_summary[branch] = {}
        for stage in STAGES:
            selected = [
                row for row in component_rows
                if row["loss_branch"] == branch and row["stage"] == stage
            ]
            component_summary[branch][stage] = {
                key: _distribution([row[key] for row in selected])
                for key in (
                    "gamma_veto_grad_abs", "gamma_context_grad_abs",
                    "veto_mlp_grad_norm", "context_conv_grad_norm",
                )
            }
    return {
        "rows": rows,
        "component_rows": component_rows,
        "summary": {
            "batches": GRADIENT_BATCHES,
            "batch_size": GRADIENT_BATCH_SIZE,
            "seed": SEED,
            "optimizer_constructed": False,
            "optimizer_step": False,
            "parameter_hash_before": before["parameter_sha256"],
            "parameter_hash_after": after["parameter_sha256"],
            "buffer_hash_before": before["buffer_sha256"],
            "buffer_hash_after": after["buffer_sha256"],
            "group_summary": group_summary,
            "hfrm_component_summary": component_summary,
            "feat_deep_gradient_norm": {
                branch: _distribution(values) for branch, values in deep_norms.items()
            },
            "shared_early_gradient_cosine": {
                key: _distribution(values) for key, values in early_cosines.items()
            },
            "feat_deep_gradient_cosine": {
                key: _distribution(values) for key, values in deep_cosines.items()
            },
        },
    }
