#!/usr/bin/env python3
"""Frozen-checkpoint C0/W0/W1 causal intervention for WD-CH."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wdch import WaveletDecoupledContext
from tool.GenDataset import Stage1_InferDataset
from tools.wdch_common import (
    CAM_WEIGHTS,
    EXPECTED_A0_SHA256,
    OfficialMetricAccumulator,
    PairedComponentAccumulator,
    PairedZoneAccumulator,
    TTA_TRANSFORMS,
    component_thresholds,
    load_a0_model,
    minmax_normalize,
    presence_from_probability,
    set_seed,
    sha256_file,
    verify_validation_root,
    write_json,
)


VARIANTS = ("C0", "W0", "W1")
STAGES = ("56", "28_1", "28_2", "deep", "final")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def backbone_features(model, x):
    x = model.conv1a(x)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)
    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    feat_56 = x
    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x)
    x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    feat_28_1 = F.relu(model.bn45(x))
    x, _ = model.b5(x, get_x_bn_relu=True)
    x = model.b5_1(x); x = model.b5_2(x)
    feat_28_2 = F.relu(model.bn52(x))
    x, _ = model.b6(x, get_x_bn_relu=True); x = model.b7(x)
    feat_deep = F.relu(model.bn7(x))
    return feat_56, feat_28_1, feat_28_2, feat_deep


def hfrm28_variants(module, feature, deep, wdch):
    batch, channels = feature.shape[:2]
    pooled = F.adaptive_avg_pool2d(deep, 1).view(batch, -1)
    gate = module.veto_mlp(pooled).view(batch, channels, 1, 1)
    vetoed = feature * gate
    c0_context = module.context_conv(feature)
    reconstructed = wdch.identity(feature)
    w0_context = module.context_conv(reconstructed)
    w1_context, bands, ll_rectified = wdch.forward_with_bands(feature)
    contexts = {"C0": c0_context, "W0": w0_context, "W1": w1_context}
    rectified = {
        name: feature + module.gamma_veto * vetoed + module.gamma_context * context
        for name, context in contexts.items()
    }
    return rectified, contexts, bands, ll_rectified, reconstructed


def forward_variants(model, wdch, x):
    feat56, feat28, feat28_2, deep = backbone_features(model, x)
    rect28, contexts, bands, ll_rectified, reconstructed = hfrm28_variants(
        model.hfrm_28_1, feat28, deep, wdch
    )
    cam56 = F.relu(model.ic_56(model.hfrm_56(feat56, deep)))
    cam28_2 = F.relu(model.ic2(model.hfrm_28_2(feat28_2, deep)))
    camdeep = F.relu(model.fc8(deep))
    raw_deep = model.fc8(deep)
    probability = torch.sigmoid(F.adaptive_avg_pool2d(raw_deep, 1).flatten(1))
    cams = {}
    for variant in VARIANTS:
        cams[variant] = {
            "56": cam56,
            "28_1": F.relu(model.ic1(rect28[variant])),
            "28_2": cam28_2,
            "deep": camdeep,
        }
    return {
        "cams": cams,
        "probability": probability,
        "feature": feat28,
        "contexts": contexts,
        "bands": bands,
        "ll_rectified": ll_rectified,
        "reconstructed": reconstructed,
    }


def resize_unflip(cam, size, output_dims):
    cam = F.interpolate(cam, size, mode="bilinear", align_corners=False)[0]
    return torch.flip(cam, dims=output_dims) if output_dims else cam


def diagnostic_row(image_id, result):
    feature = result["feature"]
    input_rms = rms(feature)
    row = {"image": image_id, "input_rms": input_rms}
    for variant in VARIANTS:
        context = result["contexts"][variant]
        row[f"{variant}_output_rms"] = rms(context)
        row[f"{variant}_output_input_rms"] = rms(context) / max(input_rms, 1.0e-12)
        row[f"{variant}_rectification_rms"] = rms(context - feature) / max(
            input_rms, 1.0e-12
        )
    row["W0_reconstruction_relative_rms"] = rms(
        result["reconstructed"] - feature
    ) / max(input_rms, 1.0e-12)
    return row


def run(args):
    verify_validation_root(args.val_root)
    if sha256_file(args.checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("A0 checkpoint mismatch")
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))
    if phase0["phase0_status"] != "PASS":
        raise AssertionError("Phase 0 has not passed")
    kernel = int(phase0["selected_kernel"])
    output = Path(args.output_dir)
    if (output / "wdch_frozen_intervention_metrics.json").exists():
        raise FileExistsError(f"Frozen-intervention output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(42, deterministic=True)
    model = load_a0_model(args.checkpoint, cam=True, device="cuda")
    model.eval()
    wdch = WaveletDecoupledContext(512, kernel).cuda()
    wdch.eval()
    dataset = Stage1_InferDataset(str(Path(args.val_root) / "img"), img_size=224)
    if len(dataset) != 3418:
        raise AssertionError(len(dataset))
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )
    metrics = {
        variant: {stage: OfficialMetricAccumulator() for stage in STAGES}
        for variant in VARIANTS
    }
    zones = PairedZoneAccumulator()
    thresholds = component_thresholds(args.val_root)
    components = PairedComponentAccumulator(thresholds)
    feature_rows = []
    w0_difference_sum = 0.0
    w0_difference_count = 0
    w0_difference_max = 0.0
    w0_prediction_differences = 0
    torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        for index, (name_tuple, image) in enumerate(loader, start=1):
            image_id = name_tuple[0]
            truth = np.asarray(
                Image.open(Path(args.val_root) / "mask" / f"{image_id}.png"),
                dtype=np.uint8,
            )
            original_size = truth.shape
            image = image.cuda(non_blocking=True)
            views = {
                variant: {stage: [] for stage in STAGES[:-1]}
                for variant in VARIANTS
            }
            probabilities = []
            first_result = None
            for view_index, (input_dims, output_dims) in enumerate(TTA_TRANSFORMS):
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = forward_variants(model, wdch, augmented)
                if view_index == 0:
                    first_result = result
                probabilities.append(result["probability"][0])
                for variant in VARIANTS:
                    for stage in STAGES[:-1]:
                        views[variant][stage].append(
                            resize_unflip(
                                result["cams"][variant][stage], original_size, output_dims
                            )
                        )
            feature_rows.append(diagnostic_row(image_id, first_result))
            probability = (
                torch.stack(probabilities).mean(0).detach().float().cpu().numpy()
            )
            presence = presence_from_probability(probability)
            predictions = {}
            normalized = {}
            for variant in VARIANTS:
                normalized[variant] = {
                    stage: minmax_normalize(
                        torch.stack(values).mean(0).detach().float().cpu().numpy()
                    )
                    for stage, values in views[variant].items()
                }
                predictions[variant] = {}
                for stage in STAGES[:-1]:
                    response = normalized[variant][stage] * presence.reshape(4, 1, 1)
                    prediction = response.argmax(0).astype(np.uint8)
                    predictions[variant][stage] = prediction
                    metrics[variant][stage].update(truth, prediction)
                fused = sum(
                    CAM_WEIGHTS[stage] * normalized[variant][stage]
                    for stage in STAGES[:-1]
                )
                fused *= presence.reshape(4, 1, 1)
                prediction = fused.argmax(0).astype(np.uint8)
                predictions[variant]["final"] = prediction
                metrics[variant]["final"].update(truth, prediction)
            c0_cam = normalized["C0"]["28_1"]
            w0_cam = normalized["W0"]["28_1"]
            difference = np.abs(c0_cam - w0_cam)
            w0_difference_sum += float(difference.sum())
            w0_difference_count += int(difference.size)
            w0_difference_max = max(w0_difference_max, float(difference.max()))
            w0_prediction_differences += int(
                np.count_nonzero(predictions["C0"]["final"] != predictions["W0"]["final"])
            )
            zones.update(
                truth, predictions["C0"]["final"], predictions["W1"]["final"]
            )
            components.update(
                truth, predictions["C0"]["final"], predictions["W1"]["final"]
            )
            if index % 200 == 0:
                print(f"WDCH_FROZEN_PROGRESS {index}/{len(dataset)}", flush=True)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    scores = {
        variant: {stage: accumulator.result() for stage, accumulator in stages.items()}
        for variant, stages in metrics.items()
    }
    zone_result = zones.result()
    component_result = components.result()
    c0 = scores["C0"]["final"]
    w0 = scores["W0"]["final"]
    w1 = scores["W1"]["final"]
    delta_miou_pp = 100.0 * (w1["mIoU"] - c0["mIoU"])
    delta_mdice_pp = 100.0 * (w1["mDice"] - c0["mDice"])
    boundary_delta = zone_result["boundary_le_7"]["delta_pp"]
    interior_delta = zone_result["interior_ge_8"]["delta_pp"]
    catastrophic = (
        delta_miou_pp < -1.0 and boundary_delta <= 0.0 and interior_delta < 0.0
    )
    total_pixels = 3418 * 224 * 224
    w0_delta_pp = 100.0 * (w0["mIoU"] - c0["mIoU"])
    w0_parity = (
        abs(w0_delta_pp) < 0.001
        and w0_prediction_differences / total_pixels < 1.0e-5
    )
    phase1_status = "PASS" if w0_parity and not catastrophic else "FAIL"
    summary = {
        "phase1_status": phase1_status,
        "selected_kernel": kernel,
        "scores": scores,
        "delta_W1_minus_C0_pp": {
            "mIoU": delta_miou_pp,
            "mDice": delta_mdice_pp,
            "class_iou": {
                str(i): 100.0 * (w1["class_iou"][str(i)] - c0["class_iou"][str(i)])
                for i in range(4)
            },
            "boundary": boundary_delta,
            "interior": interior_delta,
        },
        "W0_regression": {
            "mIoU_delta_pp": w0_delta_pp,
            "normalized_cam28_1_mean_abs": w0_difference_sum / max(w0_difference_count, 1),
            "normalized_cam28_1_max_abs": w0_difference_max,
            "different_final_pixels": w0_prediction_differences,
            "different_final_fraction": w0_prediction_differences / total_pixels,
            "parity_pass": w0_parity,
        },
        "zones": zone_result,
        "component_thresholds": thresholds,
        "components": component_result,
        "catastrophic_failure": catastrophic,
        "training_performed": False,
        "test_used": False,
        "runtime_seconds": elapsed,
        "checkpoint_sha256": EXPECTED_A0_SHA256,
    }
    write_json(output / "wdch_frozen_intervention_metrics.json", summary)
    write_csv(output / "wdch_frozen_feature_diagnostics.csv", feature_rows)
    write_csv(output / "wdch_frozen_component_size.csv", component_result)
    write_csv(
        output / "wdch_frozen_multiscale.csv",
        [
            {
                "variant": variant,
                "stage": stage,
                "mIoU": scores[variant][stage]["mIoU"],
                "mDice": scores[variant][stage]["mDice"],
            }
            for variant in VARIANTS
            for stage in STAGES
        ],
    )
    lines = [
        "# WD-CH Frozen Intervention Report",
        "",
        f"- Checkpoint SHA256: `{EXPECTED_A0_SHA256}`",
        f"- Phase-0 selected kernel: `k*={kernel}` (performance was not used).",
        "- Scope: BCSS validation only; BF16; official three-view TTA, thresholds, fusion and metric.",
        "- No training, optimizer step, test set, LUAD or parameter selection was used.",
        "",
        "## Overall",
        "",
        "| Variant | mIoU | mDice | Boundary accuracy | Interior accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| C0 | {100*c0['mIoU']:.4f} | {100*c0['mDice']:.4f} | {100*zone_result['boundary_le_7']['base_accuracy']:.4f} | {100*zone_result['interior_ge_8']['base_accuracy']:.4f} |",
        f"| W0 | {100*w0['mIoU']:.4f} | {100*w0['mDice']:.4f} | — | — |",
        f"| W1 | {100*w1['mIoU']:.4f} | {100*w1['mDice']:.4f} | {100*zone_result['boundary_le_7']['candidate_accuracy']:.4f} | {100*zone_result['interior_ge_8']['candidate_accuracy']:.4f} |",
        f"| W1−C0 (pp) | {delta_miou_pp:+.4f} | {delta_mdice_pp:+.4f} | {boundary_delta:+.4f} | {interior_delta:+.4f} |",
        "",
        "## W0 regression control",
        "",
        f"- mIoU delta: {w0_delta_pp:+.6f} pp",
        f"- differing final pixels: {w0_prediction_differences} ({w0_prediction_differences/total_pixels:.3e})",
        f"- normalized CAM28_1 mean/max absolute difference: {w0_difference_sum/max(w0_difference_count,1):.3e} / {w0_difference_max:.3e}",
        f"- W0 parity: {w0_parity}",
        "",
        "## Screening",
        "",
        f"- Catastrophic failure condition: {catastrophic}",
        f"- PHASE1_STATUS = {phase1_status}",
    ]
    (output / "wdch_frozen_intervention_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "phase1_status": phase1_status,
        "kernel": kernel,
        "delta_mIoU_pp": delta_miou_pp,
        "boundary_delta_pp": boundary_delta,
        "interior_delta_pp": interior_delta,
        "catastrophic": catastrophic,
    }, indent=2), flush=True)
    print(f"PHASE1_STATUS = {phase1_status}", flush=True)
    if phase1_status != "PASS":
        raise RuntimeError("Frozen intervention failed; matched training is forbidden")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
