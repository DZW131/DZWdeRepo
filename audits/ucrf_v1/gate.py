"""Fresh checkpoint, formula, hook-integrity, and BCSS reproduction gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.cirv import extract_regions
from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import foreground_confusion, load_state, scores_from_confusion
from tools.run_cirv_phase0_bcss_seed42 import infer_hqmr_cirv_inputs

EXPECTED_SHA = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
EXPECTED_MIOU = 0.6557244403737567
EXPECTED_M1 = 4440
EXPECTED_PIXELS = 8750254


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(checkpoint: Path) -> HQMRNet:
    if digest(checkpoint) != EXPECTED_SHA:
        raise AssertionError("Frozen HQMR checkpoint hash mismatch")
    model = HQMRNet().cuda()
    model.load_state_dict(load_state(checkpoint), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def sample_integrity(model: HQMRNet, image: torch.Tensor) -> dict:
    dummy = torch.ones((1, 4), device=image.device)
    names = ("logits5", "direct4", "logits4", "query4", "logits3", "direct3")
    def forward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(image, dummy, step=29275, hqmr_mode="full")
        item = output["stages"][2]["hqmr"]
        payload = {name: item[name].detach().float().clone() for name in names}
        payload["final_output"] = output["primary_output"].detach().float().clone()
        payload["CAM"] = output["deep_cam_logits"].detach().float().clone()
        payload["prediction"] = output["primary_output"].argmax(1).clone()
        return payload, output
    before, output = forward()
    calls = {name: 0 for name in ("scale5_key", "scale4_key", "scale3_key", "update4")}
    def count(name):
        def callback(_module, _args, _result):
            calls[name] += 1
        return callback
    hooks = [model.hqmr.scale5.key.register_forward_hook(count("scale5_key")),
             model.hqmr.scale4.key.register_forward_hook(count("scale4_key")),
             model.hqmr.scale3.key.register_forward_hook(count("scale3_key")),
             model.hqmr.update4.register_forward_hook(count("update4"))]
    try:
        after, _ = forward()
    finally:
        for hook in hooks:
            hook.remove()
    differences = {name: float((before[name] - after[name]).abs().max())
                   for name in (*names, "final_output", "CAM")}
    prediction_equal = bool(torch.equal(before["prediction"], after["prediction"]))
    item = output["stages"][2]["hqmr"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        reconstructed4 = F.interpolate(item["logits5"], size=item["direct4"].shape[-2:],
                                       mode="bilinear", align_corners=False) + item["direct4"]
        reconstructed3 = F.interpolate(item["logits4"], size=item["direct3"].shape[-2:],
                                       mode="bilinear", align_corners=False) + item["direct3"]
    formula_error = {"logits4": float((reconstructed4.float()-item["logits4"].float()).abs().max()),
                     "logits3": float((reconstructed3.float()-item["logits3"].float()).abs().max())}
    passed = all(value < 1e-6 for value in differences.values()) and prediction_equal
    passed &= all(value < 1e-6 for value in formula_error.values())
    passed &= calls == {"scale5_key": 2, "scale4_key": 2, "scale3_key": 1, "update4": 2}
    return {"pass": bool(passed), "max_abs_difference": differences,
            "prediction_equal": prediction_equal, "formula_max_abs_error": formula_error,
            "hook_calls": calls}


@torch.inference_mode()
def run(args) -> None:
    np.random.seed(42)
    torch.manual_seed(42)
    config = ROOT / "configs/audits/ucrf_v1.yaml"
    model = load_model(args.checkpoint)
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    if len(dataset) != 3418:
        raise AssertionError(f"Expected 3418 validation images, got {len(dataset)}")
    _, first = dataset[0]
    integrity = sample_integrity(model, first[None].cuda())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "hook_manifest.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if not integrity["pass"]:
        raise AssertionError("Forward formula/hook integrity gate failed")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    counts = {name: 0 for name in ("scale5_key", "scale4_key", "scale3_key", "update4")}
    def count(name):
        def callback(_module, _args, _result):
            counts[name] += 1
        return callback
    hooks = [model.hqmr.scale5.key.register_forward_hook(count("scale5_key")),
             model.hqmr.scale4.key.register_forward_hook(count("scale4_key")),
             model.hqmr.scale3.key.register_forward_hook(count("scale3_key")),
             model.hqmr.update4.register_forward_hook(count("update4"))]
    histograms, m1_count, m1_pixels = [], 0, 0
    started = time.perf_counter()
    try:
        for index, (names, image) in enumerate(loader, 1):
            image_id = names[0]
            original = Image.open(args.val_root / "img" / f"{image_id}.png")
            truth = np.asarray(Image.open(args.val_root / "mask" / f"{image_id}.png"))
            result = infer_hqmr_cirv_inputs(model, image.cuda(non_blocking=True),
                                            (original.height, original.width))
            prediction = result["prediction"]
            histograms.append(foreground_confusion(truth, prediction))
            for region in extract_regions(prediction):
                mask = region["mask"]
                valid = truth[mask]
                valid = valid[valid < 4]
                if len(valid) and not np.any(mask & (truth == region["class_id"])):
                    m1_count += 1
                    m1_pixels += int(region["area"])
            if index % 100 == 0:
                print(json.dumps({"event": "gate_progress", "images": index,
                                  "M1": m1_count, "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    miou = float(scores_from_confusion(np.stack(histograms).sum(0))["mIoU"])
    expected_calls = {"scale5_key": 6*len(dataset), "scale4_key": 6*len(dataset),
                      "scale3_key": 3*len(dataset), "update4": 6*len(dataset)}
    gate = {"pass": bool(abs(miou-EXPECTED_MIOU) <= 1e-12 and m1_count == EXPECTED_M1 and
                         m1_pixels == EXPECTED_PIXELS and integrity["pass"] and counts == expected_calls),
            "hqmr_miou": miou, "m1_components": m1_count, "m1_pixels": m1_pixels,
            "checkpoint_sha256": EXPECTED_SHA, "config_sha256": digest(config),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(), "seed": 42,
            "parameter_updates": 0, "validation_images": len(dataset),
            "hook_integrity": integrity, "full_validation_hook_calls": counts,
            "expected_hook_calls": expected_calls}
    (args.output / "00_reproduction_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"event": "gate_done", "pass": gate["pass"], "mIoU": miou,
                      "M1": m1_count, "M1_pixels": m1_pixels}), flush=True)
    if not gate["pass"]:
        raise AssertionError("Frozen checkpoint reproduction failed; downstream audit stopped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
