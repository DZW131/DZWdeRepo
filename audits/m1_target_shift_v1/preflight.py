"""Read-only preflight for the frozen M1 representation-shift audit.

This deliberately does not turn archival PSCR numbers into a fresh reproduction
gate.  A fresh checkpoint/validation run is required before the audit proceeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "checkpoint_sha256": "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb",
    "hqmr_miou": 0.6557244403737567,
    "m1_components": 4440,
    "m1_pixels": 8750254,
    "distance_C": 0.12075555730415839,
    "distance_D": 0.09994883691017588,
    "distance_M1": 0.3159886015000174,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def hook_manifest() -> dict:
    """Map proposed stage names to tensors actually present in frozen HQMR."""
    return {
        "S0_H5_pre": {
            "tensor": 'output["features"]["F5"]',
            "kind": "spatial_backbone_feature",
            "region_pooling": True,
            "note": "Backbone F5 before context projection and CCRA query allocation.",
        },
        "S1_H5_cond": {
            "tensor": 'output["query_detail"]["context_feature"]',
            "kind": "spatial_CHPF_feature",
            "region_pooling": True,
            "note": "CHPF(context_projection(F5.detach())); not itself CCRA-conditioned. CCRA changes query2, not H5 map.",
        },
        "S2_H4_recon": {
            "tensor": 'output["stages"][2]["hqmr"]["logits4"]',
            "kind": "query_mask_logits_not_feature_map",
            "region_pooling": False,
            "note": "Actual H5→H4 reconstruction is interpolate(logits5)+direct4. No separate H4_recon tissue feature exists.",
            "available_auxiliary": ['output["pixel_detail"]["F4_context"]',
                                    'output["stages"][2]["hqmr"]["direct4"]'],
        },
        "S3_H4_query": {
            "tensor": 'output["stages"][2]["hqmr"]["query4"]',
            "kind": "per_query_vector_not_spatial_feature",
            "region_pooling": False,
            "note": "QueryRegionUpdate changes query4; no region-specific spatial H4_query map is materialized.",
        },
        "S4_K4": {
            "tensor": 'output["stages"][2]["hqmr"]["key4"]',
            "kind": "spatial_semantic_key",
            "region_pooling": True,
            "note": "Identical Stage3 HQMR key4 tensor used by CIRV/PSCR; projected from F4_context before query update.",
        },
        "valid_primary_stagewise_manifold_comparison": ["S0_H5_pre", "S1_H5_cond", "S4_K4"],
        "invalid_without_architecture_change": [
            "H4_recon spatial tissue manifold", "H4_query spatial tissue manifold",
            "H5_cond as a CCRA-conditioned H5 map", "raw cross-stage cosine comparison",
        ],
        "parameter_updates": 0,
    }


def archival_gate(reproduction: dict, compatibility: dict) -> dict:
    summary = compatibility["summary"]
    checks = {
        "hqmr_miou": abs(reproduction["hqmr_miou"] - EXPECTED["hqmr_miou"]) <= 1e-12,
        "m1_components": reproduction["num_m1_components"] == EXPECTED["m1_components"],
        "m1_pixels": reproduction["m1_area"] == EXPECTED["m1_pixels"],
        "checkpoint_sha256": reproduction["checkpoint_sha256"] == EXPECTED["checkpoint_sha256"],
        **{f"distance_{key}": abs(summary[f"{key}_distance"] - EXPECTED[f"distance_{key}"]) <= 1e-9
           for key in ("C", "D", "M1")},
    }
    return {
        "status": "ARCHIVAL_PASS_RUNTIME_PENDING" if all(checks.values()) else "ARCHIVAL_STOP",
        "archival_checks": checks,
        "source": "Frozen PSCR-v1 report artifacts; not a fresh inference run",
        "runtime_reproduction_pass": False,
        "parameter_updates": 0,
        "values": {"hqmr_miou": reproduction["hqmr_miou"],
                   "m1_components": reproduction["num_m1_components"],
                   "m1_pixels": reproduction["m1_area"],
                   **{f"distance_{key}": summary[f"{key}_distance"] for key in ("C", "D", "M1")}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pscr-reproduction", type=Path, required=True)
    parser.add_argument("--pscr-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reproduction = json.loads(args.pscr_reproduction.read_text(encoding="utf-8"))
    compatibility = json.loads(args.pscr_compatibility.read_text(encoding="utf-8"))
    gate = archival_gate(reproduction, compatibility)
    if args.checkpoint:
        gate["current_checkpoint_sha256"] = digest(args.checkpoint)
        gate["current_checkpoint_hash_pass"] = gate["current_checkpoint_sha256"] == EXPECTED["checkpoint_sha256"]
        if not gate["current_checkpoint_hash_pass"]:
            gate["status"] = "CHECKPOINT_STOP"
    manifest = hook_manifest()
    (args.output / "00_reproduction_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    (args.output / "hook_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"reproduction": gate["status"], "hook_map": "written"}))


if __name__ == "__main__":
    main()
