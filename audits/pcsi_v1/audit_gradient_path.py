"""Prove HQMR loss reaches pre-CCRA PDSR through frozen differentiable modules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.ucrf_v1.gate import load_model
from audits.pcsi_v1.test_semantic_injection import normalized_residual
from network.pdsr import PathologyDenseSemanticReconstructor
from network.plip_adapter import FrozenPLIPAdapter
from tools.pdsr_vpca_phase0.common import CommonAugmentTrainDataset, write_json


def gradient_sum(parameter):
    return None if parameter.grad is None else float(parameter.grad.detach().float().abs().sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plip", type=Path, required=True)
    parser.add_argument("--p2-adapter", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--training-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.output / "phaseA_decision.json"
    if (args.output / "gradient_path.json").exists(): raise FileExistsError("Gradient audit is write-once")
    decision = json.loads(decision_path.read_text())
    if decision["PRE_CCRA_SITE"] != "I1_PRE_CONTEXT" or not decision["CAUSAL_GO"]:
        raise AssertionError("Only the causally passed common pre-CCRA site is eligible")
    base = load_model(args.checkpoint)
    plip = FrozenPLIPAdapter(args.plip).cuda().eval()
    pdsr = PathologyDenseSemanticReconstructor(plip.hidden_size).cuda()
    state = torch.load(args.p2_adapter, map_location="cpu", weights_only=False)
    pdsr.load_state_dict({key.removeprefix("pdsr."): value for key, value in state.items() if key.startswith("pdsr.")}, strict=True)
    concept = torch.load(args.concepts, map_location="cpu", weights_only=False)["embeddings"].cuda()
    gamma = torch.nn.Parameter(torch.full((256,), 1e-3, device="cuda"))
    images = CommonAugmentTrainDataset(args.training_images)
    batch = [images[index] for index in range(5)]
    raw = torch.stack([item[1] for item in batch]).cuda()
    labels = torch.stack([item[2] for item in batch]).cuda()
    mean = raw.new_tensor([.485, .456, .406])[None, :, None, None]
    std = raw.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens = plip.dense_tokens(raw)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        semantic, _ = pdsr(tokens, concept, torch.full((len(batch), 32), 1 / 32, device="cuda"))
    retained = {}

    def inject(_module, _inputs, feature):
        residual = normalized_residual(semantic, feature)
        fused = (feature.float() + gamma[None, :, None, None] * residual.float()).to(feature.dtype)
        fused.retain_grad()
        retained["fused_feature"] = fused
        return fused

    handle = base.f5_chpf.register_forward_hook(inject)
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = base((raw - mean) / std, labels, step=29275)
            # No semantic loss: nonzero PDSR gradients can only come through
            # frozen CCRA/HQMR and their original objective.
            loss = output["losses"]["loss"]
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite HQMR gradient-audit loss")
        loss.backward()
    finally:
        handle.remove()
    base_bad = [name for name, parameter in base.named_parameters() if parameter.grad is not None]
    plip_bad = [name for name, parameter in plip.named_parameters() if parameter.grad is not None]
    pdsr_grads = {name: gradient_sum(parameter) for name, parameter in pdsr.named_parameters()}
    gamma_grad = gradient_sum(gamma)
    feature_grad = float(retained["fused_feature"].grad.detach().float().abs().sum())
    required = [name for name in pdsr_grads if name.startswith("projections.") or name.startswith("semantic_projection.")]
    passed = (not base_bad and not plip_bad and gamma_grad is not None and gamma_grad > 0
              and feature_grad > 0 and bool(required)
              and all(pdsr_grads[name] is not None and pdsr_grads[name] > 0 for name in required))
    result = {"GRADIENT_PATH_PASS": bool(passed), "loss_used": "HQMR original loss only",
              "no_semantic_auxiliary_loss_in_test": True, "gamma_value": 1e-3,
              "gamma_grad_abs_sum": gamma_grad, "pre_ccra_fused_feature_grad_abs_sum": feature_grad,
              "pdsr_projection_grad_abs_sum": pdsr_grads,
              "hqmr_ccra_parameter_grads_none": not base_bad,
              "plip_parameter_grads_none": not plip_bad,
              "unexpected_frozen_grads": {"hqmr_ccra": base_bad, "plip": plip_bad},
              "parameter_updates": 0, "training_samples": [item[0] for item in batch],
              "segmentation_gt_opened": False}
    write_json(args.output / "gradient_path.json", result)
    decision["GRADIENT_PATH_PASS"] = bool(passed)
    decision["PHASE_A_DECISION"] = "GO" if passed and decision["CAUSAL_GO"] and decision["IDENTITY_PASS"] else "NOGO"
    write_json(decision_path, decision)
    print(json.dumps({"event": "PCSI_GRADIENT_AUDIT_COMPLETE", "phase_a": decision["PHASE_A_DECISION"],
                      "gamma_grad": gamma_grad, "feature_grad": feature_grad}), flush=True)
    if not passed: raise AssertionError(f"Frozen differentiable path failed: {result}")


if __name__ == "__main__":
    main()
