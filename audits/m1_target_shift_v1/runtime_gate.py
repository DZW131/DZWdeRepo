"""Fresh zero-update HQMR reproduction and non-mutating hook-safety gate."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.m1_target_shift_v1.preflight import EXPECTED, digest
from network.cirv import extract_regions
from network.hqmr_net import HQMRNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import foreground_confusion, load_state, scores_from_confusion
from tools.run_cirv_phase0_bcss_seed42 import infer_hqmr_cirv_inputs


@torch.inference_mode()
def verify_hooks(model: HQMRNet, sample: torch.Tensor) -> dict:
    dummy = torch.ones((1, 4), device=sample.device)

    def forward() -> dict[str, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample, dummy, step=29275, hqmr_mode="full")
        return {
            "logits": output["stages"][2]["hqmr"]["basis_logits"].float().clone(),
            "CAM": output["deep_cam_logits"].float().clone(),
            "primary_output": output["primary_output"].float().clone(),
            "prediction": output["primary_output"].argmax(1).clone(),
        }

    before = forward()
    captured: dict[str, list[tuple[int, ...]]] = {name: [] for name in ("k5", "k4", "update4")}
    handles = [
        model.hqmr.scale5.key.register_forward_hook(
            lambda _module, _input, value: captured["k5"].append(tuple(value.shape))),
        model.hqmr.scale4.key.register_forward_hook(
            lambda _module, _input, value: captured["k4"].append(tuple(value.shape))),
        model.hqmr.update4.register_forward_hook(
            lambda _module, _input, value: captured["update4"].append(tuple(value.shape))),
    ]
    try:
        after = forward()
    finally:
        for handle in handles:
            handle.remove()
    differences = {name: float((before[name] - after[name]).abs().max())
                   for name in ("logits", "CAM", "primary_output")}
    prediction_equal = bool(torch.equal(before["prediction"], after["prediction"]))
    passed = all(value < 1e-6 for value in differences.values()) and prediction_equal
    passed &= all(len(values) == 2 for values in captured.values())
    return {"pass": bool(passed), "max_abs_difference": differences,
            "prediction_equal": prediction_equal, "captured_shapes": captured,
            "mIoU_unchanged_by_hooks": prediction_equal,
            "note": "Sample outputs are compared with/without hooks; the full validation run reinstalls read-only hooks and compares mIoU/M1 to the frozen no-hook baseline."}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--pscr-compatibility", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = digest(args.checkpoint)
    if checkpoint_hash != EXPECTED["checkpoint_sha256"]:
        raise AssertionError(f"Checkpoint SHA256 mismatch: {checkpoint_hash}")
    stats = json.loads(args.pscr_compatibility.read_text(encoding="utf-8"))["summary"]
    distances = {key: float(stats[f"{key}_distance"]) for key in ("C", "D", "M1")}
    if any(abs(distances[key] - EXPECTED[f"distance_{key}"]) > 1e-9 for key in distances):
        raise AssertionError("Frozen PSCR distance anchors do not match")
    model = HQMRNet().cuda()
    model.load_state_dict(load_state(args.checkpoint), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
    first_name, first_image = dataset[0]
    hook_safety = verify_hooks(model, first_image[None].cuda())
    (args.output / "hook_safety.json").write_text(json.dumps(hook_safety, indent=2), encoding="utf-8")
    if not hook_safety["pass"]:
        raise AssertionError("Hooks change HQMR output; audit stopped")
    histograms = []
    m1_count = 0
    m1_pixels = 0
    full_hook_calls = {"k5": 0, "k4": 0, "update4": 0}
    def count_hook(name):
        def callback(_module, _input, _output):
            full_hook_calls[name] += 1
        return callback
    full_handles = [model.hqmr.scale5.key.register_forward_hook(count_hook("k5")),
                    model.hqmr.scale4.key.register_forward_hook(count_hook("k4")),
                    model.hqmr.update4.register_forward_hook(count_hook("update4"))]
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
                print(json.dumps({"images": index, "m1_components": m1_count,
                                  "elapsed_s": round(time.perf_counter() - started, 1)}), flush=True)
    finally:
        for handle in full_handles:
            handle.remove()
    miou = float(scores_from_confusion(np.stack(histograms).sum(0))["mIoU"])
    gate = {
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": digest(ROOT / "configs" / "audits" / "m1_target_shift_v1.yaml"),
        "hqmr_miou": miou,
        "m1_components": m1_count,
        "m1_pixels": m1_pixels,
        "distance_C": distances["C"],
        "distance_D": distances["D"],
        "distance_M1": distances["M1"],
        "distance_source": "PSCR-v1 frozen source_target_embeddings statistics (historical anchor)",
        "parameter_updates": 0,
        "images": len(dataset),
        "hook_safety_pass": hook_safety["pass"],
        "full_validation_with_hooks": True,
        "full_validation_hook_calls": full_hook_calls,
    }
    gate["pass"] = (abs(miou - EXPECTED["hqmr_miou"]) <= 1e-12 and
                    m1_count == EXPECTED["m1_components"] and
                    m1_pixels == EXPECTED["m1_pixels"] and hook_safety["pass"] and
                    all(value == 6 * len(dataset) for value in full_hook_calls.values()))
    (args.output / "00_reproduction_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"reproduction_pass": gate["pass"], "mIoU": miou,
                      "M1": m1_count, "M1_pixels": m1_pixels}), flush=True)
    if not gate["pass"]:
        raise AssertionError("Frozen HQMR reproduction gate failed; audit stopped")


if __name__ == "__main__":
    main()
