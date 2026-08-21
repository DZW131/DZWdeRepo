"""Checkpoint loading and executable source-contract assertions for HMA-v0."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from network.resnet38_cls import HFRM
from tools.hma_v0 import (
    BCSS_THRESHOLDS,
    CAM_WEIGHTS,
    CONTEXT_KERNEL,
    LOSS_WEIGHTS,
    STAGE_CHANNELS,
    STAGES,
    TTA_TRANSFORMS,
)
from tools.hma_v0.instrumentation import HMAAuditNet


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = HMAAuditNet(n_class=4)
    incompat = model.load_state_dict(state, strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise AssertionError(f"Checkpoint incompatibility: {incompat}")
    return model


def _assert_source_text(root):
    train_text = (Path(root) / "train_sshr.py").read_text(encoding="utf-8")
    infer_text = (Path(root) / "tool" / "infer_fun.py").read_text(encoding="utf-8")
    required_train = (
        "loss_w_56 * F.multilabel_soft_margin_loss",
        "loss_w_28_1 * F.multilabel_soft_margin_loss",
        "loss_w_28_2 * F.multilabel_soft_margin_loss",
        "loss_w_deep * F.multilabel_soft_margin_loss",
        "'lr': 10*args.lr",
        "'lr': 20*args.lr",
        'parser.add_argument("--loss_w_56", default=0.1',
        'parser.add_argument("--loss_w_28_1", default=0.15',
        'parser.add_argument("--loss_w_28_2", default=0.25',
        'parser.add_argument("--loss_w_deep", default=0.5',
    )
    required_infer = (
        "return np.asarray([0.8, 0.9, 0.8, 0.6]",
        "return (((), ()), ((3,), (2,)), ((2,), (1,)))",
        "cam_weights = (0.6, 0.2, 0.2)",
        "c_min = np.min(cam_np",
        "c_max = np.max(cam_np",
    )
    missing = [item for item in required_train if item not in train_text]
    missing += [item for item in required_infer if item not in infer_text]
    if missing:
        raise AssertionError(f"Released source contract changed: {missing}")


def source_contract_manifest(model, root):
    _assert_source_text(root)
    modules = {
        "56": model.hfrm_56,
        "28_1": model.hfrm_28_1,
        "28_2": model.hfrm_28_2,
    }
    architecture = {}
    for stage in STAGES:
        module = modules[stage]
        first, second = module.veto_mlp[0], module.veto_mlp[2]
        context = module.context_conv
        expected_channels = STAGE_CHANNELS[stage]
        facts = {
            "channels": expected_channels,
            "deep_channels": int(first.in_features),
            "hidden_channels": int(first.out_features),
            "gate_output_channels": int(second.out_features),
            "context_kernel": list(context.kernel_size),
            "context_groups": int(context.groups),
            "context_bias": context.bias is not None,
        }
        if not (
            first.in_features == 4096
            and first.out_features == 512
            and second.out_features == expected_channels
            and context.in_channels == context.out_channels == context.groups == expected_channels
            and context.kernel_size == (CONTEXT_KERNEL, CONTEXT_KERNEL)
            and context.bias is None
        ):
            raise AssertionError(f"HFRM architecture mismatch at {stage}: {facts}")
        architecture[stage] = facts

    initialization = {}
    for stage in STAGES:
        fresh = HFRM(STAGE_CHANNELS[stage], deep_channels=4096, context_kernel=CONTEXT_KERNEL)
        expected = 1.0 / CONTEXT_KERNEL**2
        facts = {
            "context_uniform_exact": bool(torch.all(fresh.context_conv.weight == expected)),
            "context_value": float(fresh.context_conv.weight.flatten()[0].item()),
            "gamma_veto_zero": bool(torch.equal(fresh.gamma_veto, torch.zeros_like(fresh.gamma_veto))),
            "gamma_context_zero": bool(torch.equal(fresh.gamma_context, torch.zeros_like(fresh.gamma_context))),
        }
        if not all((facts["context_uniform_exact"], facts["gamma_veto_zero"], facts["gamma_context_zero"])):
            raise AssertionError(f"Fresh HFRM initialization mismatch: {facts}")
        initialization[stage] = facts
        del fresh

    groups = model.get_parameter_groups()
    group_ids = [set(map(id, values)) for values in groups]
    scratch_expectations = {
        "ic_56.weight": model.ic_56.weight,
        "ic1.weight": model.ic1.weight,
        "ic2.weight": model.ic2.weight,
        "fc8.weight": model.fc8.weight,
        "hfrm_56.context_conv.weight": model.hfrm_56.context_conv.weight,
        "hfrm_28_1.context_conv.weight": model.hfrm_28_1.context_conv.weight,
        "hfrm_28_2.context_conv.weight": model.hfrm_28_2.context_conv.weight,
        "hfrm_56.gamma_veto": model.hfrm_56.gamma_veto,
        "hfrm_56.gamma_context": model.hfrm_56.gamma_context,
        "hfrm_28_1.gamma_veto": model.hfrm_28_1.gamma_veto,
        "hfrm_28_1.gamma_context": model.hfrm_28_1.gamma_context,
        "hfrm_28_2.gamma_veto": model.hfrm_28_2.gamma_veto,
        "hfrm_28_2.gamma_context": model.hfrm_28_2.gamma_context,
    }
    scratch_membership = {
        name: [index for index, values in enumerate(group_ids) if id(parameter) in values]
        for name, parameter in scratch_expectations.items()
    }
    if any(indices != [2] for indices in scratch_membership.values()):
        raise AssertionError(f"Scratch parameter grouping mismatch: {scratch_membership}")

    return {
        "architecture": architecture,
        "fresh_initialization": initialization,
        "loss_weights": LOSS_WEIGHTS,
        "optimizer_lr_multipliers": {
            "backbone_weight": 1,
            "backbone_bias": 2,
            "scratch_weight_and_gamma": 10,
            "scratch_bias": 20,
        },
        "scratch_membership": scratch_membership,
        "inference": {
            "cam56_fused": False,
            "cam_weights": CAM_WEIGHTS,
            "class_presence_thresholds": list(BCSS_THRESHOLDS),
            "tta": [[list(left), list(right)] for left, right in TTA_TRANSFORMS],
            "per_stage_per_class_minmax": True,
            "gate_source": "deep classifier probability",
        },
        "effective_input_span": {
            "56_stride_approximately_4": 60,
            "28_stride_approximately_8": 120,
            "claim": "structural fact only; not evidence that K=15 is wrong",
        },
    }


def gamma_autopsy(model):
    modules = {
        "56": model.hfrm_56,
        "28_1": model.hfrm_28_1,
        "28_2": model.hfrm_28_2,
    }
    result = {}
    for stage, module in modules.items():
        veto = float(module.gamma_veto.detach().float().item())
        context = float(module.gamma_context.detach().float().item())
        result[stage] = {
            "gamma_veto": veto,
            "gamma_context": context,
            "sign_veto": "positive" if veto > 0 else "negative" if veto < 0 else "zero",
            "sign_context": "positive" if context > 0 else "negative" if context < 0 else "zero",
            "absolute_veto_context_ratio": float(abs(veto) / (abs(context) + 1e-20)),
            "mathematical_gsr_action": (
                "amplification_modulation" if veto > 0 else "direct_attenuation" if veto < 0 else "inactive"
            ),
        }
    return result
