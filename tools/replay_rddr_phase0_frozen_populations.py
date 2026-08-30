"""Replay the original Phase-0 diagnostic process, verifying every saved count.

Run as a fresh process: do not import the Phase-2A TF32/benchmark setup. This
produces immutable diagnostic populations, not model predictions for selection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase0-repo", required=True)
    parser.add_argument("--phase0-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo, source, output = Path(args.phase0_repo), Path(args.phase0_dir), Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if "test" in args.val_root.lower() or "luad" in args.val_root.lower():
        raise AssertionError("BCSS validation only")
    sys.path.insert(0, str(repo))
    from network.resnet38_cls import Net
    from tool.GenDataset import Stage1_InferDataset
    from tools.rddr_phase0_common import (
        CHECKPOINT_SHA256, canonical_predictions, diagnostic_forward,
        probability_scores, sha256_file,
    )
    assert sha256_file(args.checkpoint) == CHECKPOINT_SHA256
    summary = json.loads((source / "rddr_phase0_summary.json").read_text())
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() == summary["commit"]
    with (source / "rddr_phase0_per_image.csv").open(newline="") as handle:
        expected = {row["image_id"]: row for row in csv.DictReader(handle)}
    dataset = Stage1_InferDataset(str(Path(args.val_root) / "img"), img_size=224)
    dataset.object = sorted(dataset.object)
    assert len(dataset) == len(expected) == 3418
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = Net(4).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False), strict=True)
    model.eval()
    output.mkdir(parents=True)
    threshold = summary["thresholds"]["S_JS"]["0.2"]
    totals = {name: 0 for name in ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct", "Top20")}
    manifest = []
    with torch.no_grad():
        for index, (names, image) in enumerate(loader, 1):
            image_id = names[0]
            truth = np.asarray(Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"), dtype=np.uint8)
            foreground = truth < 4
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, diag = diagnostic_forward(model, image.cuda(non_blocking=True))
                prediction = canonical_predictions(diag, truth.shape)
                _, _, scores = probability_scores(prediction["raw_logits"], prediction["deep_logits"])
                _, _, feature_scores = probability_scores(diag["CAM28_raw_logits"], diag["CAMdeep_logits"])
            raw = prediction["raw"][0].cpu().numpy().astype(np.uint8)
            rect = prediction["rect"][0].cpu().numpy().astype(np.uint8)
            js = scores["S_JS"][0].float().cpu().numpy()
            top = foreground & (js >= threshold)
            masks = {
                "Corrected_by_CH": foreground & (raw != truth) & (rect == truth),
                "Still_Wrong": foreground & (raw != truth) & (rect != truth),
                "Harmed_by_CH": foreground & (raw == truth) & (rect != truth),
                "Stable_Correct": foreground & (raw == truth) & (rect == truth),
            }
            for name, mask in masks.items():
                count = int(mask.sum())
                assert count == int(expected[image_id][f"ch_{name}_count"]), (image_id, name, count, expected[image_id][f"ch_{name}_count"])
                totals[name] += count
            assert int(top.sum()) == int(expected[image_id]["S_JS_top20_flagged"]), (image_id, "Top20")
            totals["Top20"] += int(top.sum())
            cache_path = output / f"{image_id}.npz"
            np.savez_compressed(cache_path, raw=raw, rect=rect, top20=top,
                                q_full=js / math.log(2),
                                q_feature=feature_scores["S_JS"][0].float().cpu().numpy() / math.log(2))
            manifest.append({"image_id": image_id, "sha256": sha256_file(cache_path)})
            if index % 100 == 0 or index == len(dataset):
                print(f"FROZEN_PHASE0_REPLAY {index}/{len(dataset)} exact per-image count parity", flush=True)
    metadata = {
        "status": "PASS", "images": len(dataset), "source_commit": summary["commit"],
        "checkpoint_sha256": CHECKPOINT_SHA256, "counts": totals,
        "per_image_all_four_ch_counts_and_top20_exact": True,
        "tf32_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
        "tf32_conv_precision": torch.backends.cudnn.conv.fp32_precision,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "phase0_threshold_js": threshold, "files": manifest,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: value for key, value in metadata.items() if key != "files"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
