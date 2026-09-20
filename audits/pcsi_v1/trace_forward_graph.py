"""Trace the untouched HQMR forward before any pre-CCRA model edit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.ucrf_v1.gate import load_model
from tools.pdsr_vpca_phase0.common import CommonEvalDataset, write_json


def tensor_meta(value):
    if isinstance(value, torch.Tensor):
        shape = list(value.shape)
        channels = shape[1] if value.ndim == 4 else shape[-1] if value.ndim >= 2 else None
        spatial = list(shape[-2:]) if value.ndim == 4 else None
        return {"shape": shape, "dtype": str(value.dtype), "requires_grad": bool(value.requires_grad),
                "channels": channels, "spatial_resolution": spatial}
    if isinstance(value, (tuple, list)):
        return [tensor_meta(item) for item in value]
    if isinstance(value, dict):
        return {key: tensor_meta(item) for key, item in value.items() if isinstance(item, torch.Tensor)}
    return str(type(value).__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "forward_dependency_graph.json").exists():
        raise FileExistsError("Forward graph is write-once")
    model = load_model(args.checkpoint)
    name, raw = CommonEvalDataset(args.images)[0]
    raw = raw[None].cuda()
    mean = raw.new_tensor([.485, .456, .406])[None, :, None, None]
    std = raw.new_tensor([.229, .224, .225])[None, :, None, None]
    image = (raw - mean) / std
    labels = torch.ones((1, 4), device=image.device)
    modules = {
        "backbone": model.backbone,
        "patch_queries": model.patch_queries,
        "semantic_projection": model.semantic_projection,
        "decoder1": model.decoder1,
        "context_projection": model.context_projection,
        "f5_chpf": model.f5_chpf,
        "ccra2_q_projection": model.ccra2.q_projection,
        "ccra2_k_projection": model.ccra2.k_projection,
        "ccra2_v_projection": model.ccra2.v_projection,
        "ccra2": model.ccra2,
        "f4_memory_projection": model.f4_memory_projection,
        "ccra3": model.ccra3,
        "hqmr_query_norm": model.hqmr.query_norm,
        "hqmr_scale5_key": model.hqmr.scale5.key,
        "hqmr_scale5_value": model.hqmr.scale5.value,
        "hqmr_update5": model.hqmr.update5,
        "hqmr_scale4_key": model.hqmr.scale4.key,
        "hqmr_scale4_value": model.hqmr.scale4.value,
        "hqmr_update4": model.hqmr.update4,
        "hqmr_scale3_key": model.hqmr.scale3.key,
        "hqmr_scale3_value": model.hqmr.scale3.value,
    }
    events = []
    hooks = []
    for key, module in modules.items():
        def capture(_module, _inputs, result, label=key):
            events.append({"order": len(events) + 1, "tensor_name": label, "producer_module": label,
                           "runtime": tensor_meta(result)})
        hooks.append(module.register_forward_hook(capture))
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(image, labels, step=29275)
    finally:
        for hook in hooks:
            hook.remove()
    for stage_index, stage in enumerate(output["stages"], 1):
        events.append({"order": len(events) + 1, "tensor_name": f"stage{stage_index}_query",
                       "producer_module": "GCQMNet.forward", "runtime": tensor_meta(stage["query"])})
        if stage_index >= 2:
            for key in ("responsibility_class", "responsibility", "query_delta"):
                events.append({"order": len(events) + 1, "tensor_name": f"stage{stage_index}_{key}",
                               "producer_module": f"ccra{stage_index}", "runtime": tensor_meta(stage["detail"][key])})
            for key in ("query0", "logits5", "query5", "logits4", "query4", "logits3"):
                value = stage["hqmr"].get(key)
                if value is not None:
                    events.append({"order": len(events) + 1, "tensor_name": f"stage{stage_index}_hqmr_{key}",
                                   "producer_module": "HQMR.forward", "runtime": tensor_meta(value)})
    for name_in_output, key in (("context_raw", "context_raw"), ("context_feature", "context_feature")):
        events.append({"order": len(events) + 1, "tensor_name": key,
                       "producer_module": "GCQMNet.forward", "runtime": tensor_meta(output["query_detail"][name_in_output])})
    events.append({"order": len(events) + 1, "tensor_name": "final_class_projection",
                   "producer_module": "class_mixture", "runtime": tensor_meta(output["primary_output"])})
    edges = [
        ("backbone.F3", "pixel_decoder"), ("backbone.F4", "pixel_decoder"),
        ("backbone.F5.detach", "context_projection"), ("context_projection", "f5_chpf"),
        ("f5_chpf/context_feature", "ccra2.memory_norm"),
        ("ccra2.memory_norm", "ccra2.k_projection"), ("ccra2.memory_norm", "ccra2.v_projection"),
        ("patch_queries", "decoder1"), ("backbone.FD.detach", "semantic_projection"),
        ("semantic_projection", "decoder1"), ("decoder1/query1", "ccra2.q_projection"),
        ("ccra2.q_projection", "ccra2.affinity"), ("ccra2.k_projection", "ccra2.affinity"),
        ("ccra2.affinity", "ccra2.responsibility_class"),
        ("ccra2.responsibility_class", "ccra2.query2"),
        ("ccra2.v_projection", "ccra2.query2"),
        ("ccra2.query2", "ccra3"), ("ccra2.query2", "hqmr.query_norm/q0"),
        ("f5_chpf/context_feature", "hqmr.scale5.key/k5"),
        ("f5_chpf/context_feature", "hqmr.scale5.value/v5"),
        ("hqmr.query_norm/q0", "hqmr.logits5/L5"), ("hqmr.scale5.key/k5", "hqmr.logits5/L5"),
        ("hqmr.logits5/L5", "hqmr.update5/q5"), ("hqmr.scale5.value/v5", "hqmr.update5/q5"),
        ("hqmr.update5/q5", "hqmr.logits4/L4"), ("pixel_decoder.F4_context", "hqmr.scale4.key/k4"),
        ("hqmr.scale4.key/k4", "hqmr.logits4/L4"), ("hqmr.logits4/L4", "hqmr.update4/q4"),
        ("pixel_decoder.F4_context", "hqmr.scale4.value/v4"), ("hqmr.scale4.value/v4", "hqmr.update4/q4"),
        ("hqmr.update4/q4", "hqmr.logits3/L3"), ("backbone.F3", "hqmr.scale3.key/k3"),
        ("hqmr.scale3.key/k3", "hqmr.logits3/L3"), ("hqmr.logits3/L3", "final_class_projection"),
    ]
    graph = {
        "sample_image": name, "checkpoint": str(args.checkpoint), "model_parameters_untouched": True,
        "events": events, "edges": [{"from": a, "to": b} for a, b in edges],
        "pre_ccra_common_visual_source": "f5_chpf/context_feature",
        "late_residual_path": "HQMRNet.forward calls GCQMNet.forward first; only after it returns is h5_residual added to context_feature for HQMR.scale5",
        "query_naming_disambiguation": "decoder1/query1 is unchanged by a context_feature injection, but HQMR query0 is downstream of ccra2 and is expected to change",
        "responsibility_detach_note": "CCRALayer returns a detached diagnostic responsibility_class, while its internal pooled update remains differentiable through responsibility and v",
    }
    write_json(args.output / "forward_dependency_graph.json", graph)
    lines = ["# PCSI-v1 exact HQMR forward dependency graph", "",
             f"Sample: `{name}`. Checkpoint: `{args.checkpoint}`. No model file or parameter was modified.", "",
             "The previous late residual enters only after `GCQMNet.forward` has completed `ccra2` and `ccra3`:", "",
             "`F5 → context_projection → CHPF/context_feature → CCRA2(K,V,responsibility,query2) → CCRA3 → return to HQMRNet.forward → old h5 + residual → HQMR.scale5(K,V) → L5 → q5 → L4 → q4 → L3 → class mixture`", "",
             "Thus `context_feature` immediately after `f5_chpf` is the common spatial visual source before both CCRA2 K/V and HQMR K/V. `decoder1/query1` is the frozen CCRA input query. HQMR's internally named `query0` is downstream of CCRA2 and may change under a valid pre-CCRA injection.", "",
             "| Execution | Tensor/module | Shape | Channels | Spatial | Requires grad (frozen baseline) |", "|---:|---|---|---:|---|---|"]
    for event in events:
        runtime = event["runtime"]
        if isinstance(runtime, dict) and "shape" in runtime:
            lines.append(f"| {event['order']} | `{event['tensor_name']}` | `{runtime['shape']}` | {runtime['channels']} | `{runtime['spatial_resolution']}` | {runtime['requires_grad']} |")
    lines.extend(["", "## Causal edge list", ""])
    lines.extend(f"- `{source}` → `{target}`" for source, target in edges)
    lines.extend(["", "## Detach boundary", "", "The CCRA diagnostic `responsibility_class` is returned detached. The internal CCRA pooled value/update that forms `query2` is not wrapped in `torch.no_grad`; frozen parameter tensors may still pass gradients to an injected input. This must be tested explicitly in the gradient audit.", ""])
    (args.output / "forward_dependency_graph.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "PCSI_FORWARD_GRAPH_COMPLETE", "sample": name,
                      "pre_ccra_source": graph["pre_ccra_common_visual_source"],
                      "events": len(events)}), flush=True)


if __name__ == "__main__":
    main()
