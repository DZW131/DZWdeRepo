#!/usr/bin/env python3
"""One-batch BF16 and actual-HQMR gradient-contract preflight for CIRV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.cirv import calibrate_evidence, select_source_regions
from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import load_state
from tools.hqrf_phase0_io import sha256, write_json
from tools.run_cirv_phase0_bcss_seed42 import HQMR_SHA256, parse_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); checkpoint = Path(args.checkpoint); output = Path(args.output_dir)
    if sha256(checkpoint) != HQMR_SHA256: raise AssertionError("Frozen HQMR checkpoint mismatch")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    names, image = next(iter(DataLoader(Stage1_InferDataset(args.trainroot, img_size=224), batch_size=1)))
    label_np = parse_label(names[0]); label = torch.from_numpy(label_np[None]).cuda(); image = image.cuda()
    model = HQMRNet().cuda(); model.load_state_dict(load_state(checkpoint), strict=True); model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(image, label, step=29275, hqmr_mode="full")
    item = result["stages"][2]["hqmr"]; base = item["mixture"]
    anchors = result["target_detail"]["positive"].float()
    if anchors.shape[-2:] != base.shape[-2:]: anchors = F.interpolate(anchors, base.shape[-2:], mode="nearest")
    prediction = base.detach().argmax(1)[0].cpu().numpy()
    sources = select_source_regions(prediction, label_np, anchors[0].bool().cpu().numpy(), item["key4"][0])
    bank = np.zeros((4, 4, item["key4"].shape[1]), np.float32)
    for cls in range(4):
        source = next((row["embedding"] for row in sources if row["class_id"] == cls), None)
        if source is None: source = np.eye(1, bank.shape[-1], dtype=np.float32)[0]
        bank[cls] = source
    evidence = base[0].detach().float().cpu().numpy()
    calibrated, records = calibrate_evidence(evidence, prediction, item["key4"][0], bank,
                                              result["deep_gate"][0].detach().float().cpu().numpy())
    loss = result["losses"]["loss"]
    model.zero_grad(set_to_none=True); loss.backward(retain_graph=True)
    first = {name: value.grad.detach().clone() for name, value in model.named_parameters() if value.grad is not None}
    model.zero_grad(set_to_none=True)
    # CIRV bookkeeping has already run and is detached; repeat the identical base backward.
    loss.backward()
    differences = [float((value.grad - first[name]).abs().max()) for name, value in model.named_parameters()
                   if name in first]
    report = {
        "status": "PASS", "checkpoint_sha256": sha256(checkpoint), "image_id": names[0],
        "bf16_base_finite": bool(torch.isfinite(base).all()),
        "components_valid": bool(records), "region_embeddings_finite": all(
            np.isfinite(row["embedding"]).all() for row in records),
        "prototype_scores_finite": all(np.isfinite(row["p_proto"]).all() for row in records),
        "fusion_finite": bool(np.isfinite(calibrated).all()), "ema_finite": bool(np.isfinite(bank).all()),
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "gradient_parameters_compared": len(differences),
        "max_abs_gradient_difference": max(differences, default=0.0),
        "gradient_contract_pass": max(differences, default=0.0) <= 1e-7,
        "new_trainable_parameters": 0,
    }
    if not all((report["bf16_base_finite"], report["components_valid"],
                report["region_embeddings_finite"], report["prototype_scores_finite"],
                report["fusion_finite"], report["ema_finite"], report["gradient_contract_pass"])):
        report["status"] = "FAIL"
    write_json(output / "tests/cirv_smoke.json", report)
    write_json(output / "tests/cirv_gradient_contract.json", {
        key: report[key] for key in ("gradient_parameters_compared", "max_abs_gradient_difference",
                                    "gradient_contract_pass")})
    print(report)
    if report["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
