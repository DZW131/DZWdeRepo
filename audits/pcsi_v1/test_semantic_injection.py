"""GT-free causal probe of the frozen P2 semantic residual at exact forward sites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.ucrf_v1.gate import load_model
from network.pdsr import PathologyDenseSemanticReconstructor
from network.plip_adapter import FrozenPLIPAdapter
from tools.pdsr_vpca_phase0.common import CommonEvalDataset, sha256, write_json


SITES = ("I0_LATE_H5", "I1_PRE_CONTEXT", "I2_CCRA_K_ONLY", "I3_CCRA_V_ONLY")
KEYS = ("pre_feature", "ccra_k5", "ccra_v5", "ccra_responsibility", "ccra_query2",
        "hqmr_q0", "hqmr_k5", "hqmr_v5", "L5", "q5", "L4", "q4", "L3", "final")


def normalized_residual(semantic: torch.Tensor, feature: torch.Tensor) -> torch.Tensor:
    if feature.ndim == 3:
        value = semantic.float().flatten(2).transpose(1, 2)
    else:
        value = semantic.float()
    if value.shape != feature.shape:
        raise AssertionError(f"Residual/site shape mismatch: {value.shape} vs {feature.shape}")
    dims = tuple(range(1, value.ndim))
    rms_z = value.square().mean(dims, keepdim=True).sqrt().clamp_min(1e-8)
    rms_f = feature.float().square().mean(dims, keepdim=True).sqrt()
    return (value / rms_z * rms_f).to(feature.dtype)


class FrozenStaticP2:
    def __init__(self, plip_path: Path, p2_path: Path, concepts_path: Path):
        self.plip = FrozenPLIPAdapter(plip_path).cuda().eval()
        self.pdsr = PathologyDenseSemanticReconstructor(self.plip.hidden_size).cuda().eval()
        checkpoint = torch.load(p2_path, map_location="cpu", weights_only=False)
        state = {key.removeprefix("pdsr."): value for key, value in checkpoint.items() if key.startswith("pdsr.")}
        self.pdsr.load_state_dict(state, strict=True)
        self.concepts = torch.load(concepts_path, map_location="cpu", weights_only=False)["embeddings"].cuda()
        for module in (self.plip, self.pdsr):
            for parameter in module.parameters(): parameter.requires_grad_(False)

    @torch.inference_mode()
    def semantic(self, raw: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = self.plip.dense_tokens(raw)
            weights = torch.full((raw.shape[0], 32), 1 / 32, device=raw.device)
            semantic, _ = self.pdsr(tokens, self.concepts, weights)
        return semantic.detach()


def run(model, image, labels, semantic, site: str, alpha: float, baseline_feature=None):
    captured = {"hqmr_k5": [], "hqmr_v5": []}
    hooks = []

    def perturb(_module, _inputs, output):
        if alpha == 0:
            return output
        return (output.float() + alpha * normalized_residual(semantic, output).float()).to(output.dtype)

    if site == "I1_PRE_CONTEXT":
        hooks.append(model.f5_chpf.register_forward_hook(perturb))
    elif site == "I2_CCRA_K_ONLY":
        hooks.append(model.ccra2.k_projection.register_forward_hook(perturb))
    elif site == "I3_CCRA_V_ONLY":
        hooks.append(model.ccra2.v_projection.register_forward_hook(perturb))
    elif site != "I0_LATE_H5":
        raise ValueError(site)

    def save(name):
        def capture(_module, _inputs, output):
            if isinstance(captured.get(name), list): captured[name].append(output)
            else: captured[name] = output
        return capture

    hooks.extend([
        model.ccra2.k_projection.register_forward_hook(save("ccra_k5")),
        model.ccra2.v_projection.register_forward_hook(save("ccra_v5")),
        model.hqmr.scale5.key.register_forward_hook(save("hqmr_k5")),
        model.hqmr.scale5.value.register_forward_hook(save("hqmr_v5")),
    ])
    late = None
    if site == "I0_LATE_H5" and alpha != 0:
        if baseline_feature is None: raise ValueError("Late control requires the baseline feature")
        late = (alpha * normalized_residual(semantic, baseline_feature).float()).to(baseline_feature.dtype)
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(image, labels, step=29275, h5_residual=late)
    finally:
        for hook in hooks: hook.remove()
    stage = output["stages"][2]
    payload = {
        "pre_feature": output["query_detail"]["context_feature"],
        "ccra_k5": captured["ccra_k5"], "ccra_v5": captured["ccra_v5"],
        "ccra_responsibility": output["stages"][1]["detail"]["responsibility_class"],
        "ccra_query2": output["stages"][1]["query"],
        "pre_query1": output["stages"][0]["query"],
        "hqmr_q0": stage["hqmr"]["query0"],
        "hqmr_k5": captured["hqmr_k5"][-1], "hqmr_v5": captured["hqmr_v5"][-1],
        "L5": stage["hqmr"]["logits5"], "q5": stage["hqmr"]["query5"],
        "L4": stage["hqmr"]["logits4"], "q4": stage["hqmr"]["query4"],
        "L3": stage["hqmr"]["logits3"], "final": output["primary_output"],
    }
    return {key: value.detach().float() for key, value in payload.items()}


def rms_delta(before, after):
    return float((after - before).square().mean().sqrt())


def js_probability(p, q, dim=1):
    p = p.float().clamp_min(1e-8); q = q.float().clamp_min(1e-8); m = .5 * (p + q)
    return float((.5 * ((p * (p.log() - m.log())).sum(dim) + (q * (q.log() - m.log())).sum(dim))).mean())


def compare(base, changed, site, alpha, image_id):
    row = {"image_id": image_id, "site": site, "alpha": alpha}
    for key in (*KEYS, "pre_query1"):
        row[f"delta_{key}_rms"] = rms_delta(base[key], changed[key])
        if alpha == 0:
            row[f"max_abs_{key}"] = float((changed[key] - base[key]).abs().max())
    p, q = base["ccra_responsibility"], changed["ccra_responsibility"]
    row["ccra_responsibility_js"] = js_probability(p, q)
    old_top, new_top = p.argmax(1), q.argmax(1)
    row["top_query_change_rate"] = float((old_top != new_top).float().mean())
    row["spatial_any_class_change_rate"] = float((old_top != new_top).any(-1).float().mean())
    old_entropy = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(1).mean()
    new_entropy = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(1).mean()
    row["ccra_entropy_change"] = float(new_entropy - old_entropy)
    row["L5_query_softmax_js"] = js_probability(F.softmax(base["L5"], dim=1), F.softmax(changed["L5"], dim=1))
    row["final_argmax_change_rate"] = float((base["final"].argmax(1) != changed["final"].argmax(1)).float().mean())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plip", type=Path, required=True)
    parser.add_argument("--p2-adapter", type=Path, required=True)
    parser.add_argument("--concepts", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "causal_sensitivity.csv").exists():
        raise FileExistsError("Causal audit is write-once")
    model = load_model(args.checkpoint)
    probe = FrozenStaticP2(args.plip, args.p2_adapter, args.concepts)
    data = CommonEvalDataset(args.images)
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    labels = torch.ones((1, 4), device="cuda")
    rows = []
    for index in range(min(args.num_images, len(data))):
        image_id, raw = data[index]; raw = raw[None].cuda(); image = (raw - mean) / std
        semantic = probe.semantic(raw)
        baseline = run(model, image, labels, semantic, "I0_LATE_H5", 0.)
        for site in SITES:
            for alpha in (0., .05, .10, .20):
                # Secondary scales are GT-free sensitivity only, never a selection sweep.
                current = run(model, image, labels, semantic, site, alpha, baseline["pre_feature"])
                row = compare(baseline, current, site, alpha, image_id)
                rows.append(row)
        if (index + 1) % 8 == 0:
            print(json.dumps({"event": "PCSI_CAUSAL_PROGRESS", "images": index + 1}), flush=True)
    frame = pd.DataFrame(rows)
    identity = {}
    for site, group in frame[frame.alpha == 0].groupby("site"):
        columns = [key for key in group.columns if key.startswith("max_abs_")]
        maximum = max(float(group[column].max()) for column in columns)
        identity[site] = {"images": len(group), "max_abs_drift": maximum,
                          "argmax_equal": bool((group.final_argmax_change_rate == 0).all()),
                          "pass": maximum < 1e-6 and bool((group.final_argmax_change_rate == 0).all())}
    frame.to_csv(args.output / "causal_sensitivity.csv", index=False)
    write_json(args.output / "identity_tests.json", {"tests": identity, "all_pass": all(x["pass"] for x in identity.values())})
    summaries = []
    for site in SITES:
        value = frame[(frame.site == site) & (frame.alpha == .1)]
        summary = {"site": site, "structurally_pre_ccra": site != "I0_LATE_H5",
                   "common_kv_source": site == "I1_PRE_CONTEXT", "primary_scale": .1,
                   "images": len(value), "identity_pass": identity[site]["pass"]}
        for column in ("delta_pre_feature_rms", "delta_ccra_k5_rms", "delta_ccra_v5_rms",
                       "delta_ccra_responsibility_rms", "delta_hqmr_q0_rms", "delta_hqmr_k5_rms",
                       "delta_hqmr_v5_rms", "delta_L5_rms", "delta_q5_rms", "delta_L4_rms",
                       "delta_q4_rms", "delta_L3_rms", "ccra_responsibility_js", "L5_query_softmax_js",
                       "top_query_change_rate", "spatial_any_class_change_rate",
                       "ccra_entropy_change", "final_argmax_change_rate", "delta_pre_query1_rms"):
            summary[column] = float(value[column].mean())
        summary["causal_go"] = bool(summary["identity_pass"] and summary["structurally_pre_ccra"]
                                    and summary["delta_L5_rms"] > 1e-4
                                    and summary["spatial_any_class_change_rate"] > .01
                                    and summary["delta_L3_rms"] > 1e-4
                                    and summary["delta_pre_query1_rms"] == 0)
        summaries.append(summary)
    pd.DataFrame(summaries).to_csv(args.output / "injection_candidates.csv", index=False)
    eligible = [x for x in summaries if x["causal_go"]]
    priority = ("I1_PRE_CONTEXT", "I2_CCRA_K_ONLY", "I3_CCRA_V_ONLY")
    chosen = next((site for site in priority if any(x["site"] == site for x in eligible)), None)
    decision = {"PRE_CCRA_SITE": chosen, "STRUCTURAL_CAUSALITY": bool(chosen),
                "IDENTITY_PASS": all(x["pass"] for x in identity.values()),
                "CAUSAL_GO": bool(chosen), "GRADIENT_PATH_PASS": None,
                "PHASE_A_DECISION": "PENDING_GRADIENT" if chosen else "NOGO",
                "candidate_priority": list(priority), "I4_KV_SEPARATE": "not applicable: a common source exists",
                "probe_p2_adapter_sha256": sha256(args.p2_adapter),
                "no_segmentation_gt_access": True, "primary_alpha": .1,
                "secondary_alphas": [.05, .20], "sample_count": min(args.num_images, len(data))}
    write_json(args.output / "phaseA_decision.json", decision)
    print(json.dumps({"event": "PCSI_CAUSAL_COMPLETE", **decision}), flush=True)


if __name__ == "__main__":
    main()
