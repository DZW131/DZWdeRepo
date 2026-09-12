#!/usr/bin/env python3
"""Read-only HQMR-v1 Full-vs-no-update coverage/purity conflict audit."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.audit_semantic_coverage_reachability import resize_label
from tools.eval_gcqm_full25_bcss_seed42 import THRESHOLDS, load_state
from tools.eval_hqmr_full25_bcss_seed42 import predict_hqmr_modes, coverage_one
from tools.hqrf_phase0_io import sha256, write_csv, write_json


HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args(); valroot, checkpoint, output = map(lambda value: Path(value).resolve(), (args.val_root, args.checkpoint, args.output_dir))
    if sha256(checkpoint) != HQMR_SHA256: raise AssertionError("HQMR-v1 checkpoint identity mismatch")
    if len(list((valroot / "img").glob("*.png"))) != 3418: raise AssertionError("Expected frozen BCSS validation split")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise FileExistsError(output)
    model = HQMRNet().cuda(); model.load_state_dict(load_state(checkpoint), strict=True); model.eval()
    loader = DataLoader(Stage1_InferDataset(str(valroot / "img"), img_size=224), batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    totals = {mode: defaultdict(float) for mode in ("A_full", "D_no_query_update")}; rows = []
    for index, (names, image) in enumerate(loader, 1):
        image_id = names[0]; truth = np.asarray(Image.open(valroot / "mask" / f"{image_id}.png")); image = image.cuda(non_blocking=True)
        bundle = predict_hqmr_modes(model, image, truth.shape)
        for mode in totals:
            payload = bundle["modes"][mode]; gt = resize_label(truth, payload["basis"].shape[-2:])
            coverage, _ = coverage_one(payload["basis"], bundle["weights"], payload["evidence_grid"], truth, payload["prediction_grid"])
            for key in ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall"):
                totals[mode][key] += coverage[key] * coverage["gt_pixels"]
            totals[mode]["gt_pixels"] += coverage["gt_pixels"]
            for cls in range(4):
                class_map = np.einsum("q,qhw->hw", bundle["weights"][:, cls], payload["basis"])
                target, rival, background = gt == cls, (gt < 4) & (gt != cls), gt == 4
                mass = float(class_map.sum()) + 1e-8
                values = {"weighted_purity": float(class_map[target].sum() / mass),
                          "rival_mass": float(class_map[rival].sum() / mass),
                          "background_mass": float(class_map[background].sum() / mass),
                          "prediction_gt_area_ratio": float((payload["prediction_grid"] == cls).sum() / max(target.sum(), 1))}
                pixels = int(target.sum()); rows.append({"image_id": image_id, "mode": mode, "class": cls, "gt_pixels": pixels, **values})
                for key, value in values.items(): totals[mode][key] += value * pixels
        if index % 200 == 0 or index == len(loader): print(f"CPHQMR_PREAUDIT_PROGRESS={index}/{len(loader)}", flush=True)
    summary = {}
    for mode, values in totals.items():
        denominator = max(values["gt_pixels"], 1)
        summary[mode] = {key: values[key] / denominator for key in ("basis_max_coverage", "class_basis_uncovered", "oracle_top10_recall", "weighted_purity", "rival_mass", "background_mass", "prediction_gt_area_ratio")}
    full, broad = summary["A_full"], summary["D_no_query_update"]
    conflict = broad["basis_max_coverage"] > full["basis_max_coverage"] and broad["class_basis_uncovered"] < full["class_basis_uncovered"]
    purity_lower = broad["weighted_purity"] < full["weighted_purity"] or broad["rival_mass"] > full["rival_mass"]
    decision = "COVERAGE_PURITY_CONFLICT_CONFIRMED" if conflict and purity_lower else "COVERAGE_PURITY_CONFLICT_NOT_CONFIRMED"
    result = {"decision": decision, "hqmr_v1_checkpoint_sha256": sha256(checkpoint), "validation_images": len(loader),
              "definitions": {"weighted_purity": "sum(w_ic*B_i on GT class c)/sum(w_ic*B_i)", "rival_mass": "mass on other foreground classes", "background_mass": "mass on BCSS label 4", "aggregation": "GT-pixel-weighted across image/class rows"},
              "summary": summary, "delta_no_update_minus_full": {key: broad[key] - full[key] for key in full},
              "interpretation": "Query-update benefit may reflect calibration/semantic concentration rather than a simple purity gain." if not purity_lower else "No-query-update expands support but reduces purity or increases rival mass."}
    write_csv(output / "hqmr_v1_coverage_purity_conflict.csv", rows); write_json(output / "hqmr_v1_conflict_summary.json", result)
    print(json.dumps(result, indent=2)); print(f"DECISION = {decision}")


if __name__ == "__main__": main()
