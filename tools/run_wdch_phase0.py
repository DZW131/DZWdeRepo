#!/usr/bin/env python3
"""Run the zero-training WD-CH engineering audit and lock k* by RF matching."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.wdch import FixedHaarDWT2D, WaveletDecoupledContext
from tools.wdch_common import (
    A0_COMMIT,
    EXPECTED_A0_SHA256,
    load_a0_model,
    set_seed,
    sha256_file,
    verify_validation_root,
    write_json,
)


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().square().mean().sqrt())


def reconstruction_row(shape, dtype, device):
    generator = torch.Generator(device=device).manual_seed(20260824 + shape[-1])
    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    x = x.to(dtype)
    transform = FixedHaarDWT2D().to(device)
    reconstructed = transform.reconstruct(x)
    error = reconstructed.float() - x.float()
    input_rms = rms(x)
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "device": str(device),
        "shape": "x".join(str(value) for value in shape),
        "input_rms": input_rms,
        "reconstruction_rms": rms(reconstructed),
        "mae": float(error.abs().mean()),
        "rmse": rms(error),
        "max_abs_error": float(error.abs().max()),
        "relative_error": rms(error) / max(input_rms, 1.0e-12),
        "shape_exact": tuple(reconstructed.shape) == tuple(x.shape),
    }


def energy_row(x: torch.Tensor, source: str, image: str = ""):
    transform = FixedHaarDWT2D().to(x.device)
    bands = transform.dwt(x)
    energies = [float(band.detach().float().square().sum()) for band in bands]
    total = float(x.detach().float().square().sum())
    band_sum = sum(energies)
    return {
        "source": source,
        "image": image,
        "dtype": str(x.dtype).removeprefix("torch."),
        "input_energy": total,
        "LL_energy": energies[0],
        "LH_energy": energies[1],
        "HL_energy": energies[2],
        "HH_energy": energies[3],
        "LL_ratio": energies[0] / max(band_sum, 1.0e-12),
        "LH_ratio": energies[1] / max(band_sum, 1.0e-12),
        "HL_ratio": energies[2] / max(band_sum, 1.0e-12),
        "HH_ratio": energies[3] / max(band_sum, 1.0e-12),
        "relative_energy_error": abs(total - band_sum) / max(total, 1.0e-12),
    }


def correction_metrics(correction: torch.Tensor):
    response = correction.detach().float().cpu().numpy()[0, 0]
    energy = np.square(response.astype(np.float64))
    height, width = response.shape
    yy, xx = np.mgrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    threshold = max(float(np.abs(response).max()) * 1.0e-7, 1.0e-12)
    support = np.abs(response) > threshold
    support_radius = float(radius[support].max()) if support.any() else 0.0
    order = np.argsort(radius.ravel())
    sorted_radius = radius.ravel()[order]
    sorted_energy = energy.ravel()[order]
    cumulative = np.cumsum(sorted_energy)
    total = max(float(cumulative[-1]), 1.0e-20)

    def quantile_radius(fraction):
        index = int(np.searchsorted(cumulative, fraction * total, side="left"))
        return float(sorted_radius[min(index, len(sorted_radius) - 1)])

    second_moment = float((energy * radius**2).sum() / total)
    return {
        "support_radius": support_radius,
        "energy_radius_50": quantile_radius(0.50),
        "energy_radius_90": quantile_radius(0.90),
        "spatial_second_moment": second_moment,
        "spatial_rms_radius": math.sqrt(max(second_moment, 0.0)),
        "correction_rms": float(np.sqrt(energy.mean())),
        "response": response,
        "radius": radius,
        "energy": energy,
    }


def receptive_field_audit(output_dir: Path):
    impulse = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    impulse[..., 14, 14] = 1.0
    original = nn.Conv2d(1, 1, 15, padding=7, groups=1, bias=False)
    nn.init.constant_(original.weight, 1.0 / 225.0)
    reference = correction_metrics(original(impulse) - impulse)
    results = {"CH15": reference}
    rows = []
    reference_vector = np.asarray(
        [
            reference["support_radius"],
            reference["energy_radius_50"],
            reference["energy_radius_90"],
            reference["spatial_rms_radius"],
        ],
        dtype=np.float64,
    )
    for kernel in (5, 7, 9):
        operator = WaveletDecoupledContext(1, kernel)
        metrics = correction_metrics(operator(impulse) - impulse)
        vector = np.asarray(
            [
                metrics["support_radius"],
                metrics["energy_radius_50"],
                metrics["energy_radius_90"],
                metrics["spatial_rms_radius"],
            ]
        )
        normalized = (vector - reference_vector) / np.maximum(
            np.abs(reference_vector), 1.0
        )
        metrics["distance"] = float(np.sqrt(np.mean(normalized**2)))
        results[f"WDCH{kernel}"] = metrics
    selected = min((results[f"WDCH{k}"]["distance"], k) for k in (5, 7, 9))[1]
    for name, metrics in results.items():
        rows.append(
            {
                "operator": name,
                "kernel": 15 if name == "CH15" else int(name.removeprefix("WDCH")),
                "support_radius": metrics["support_radius"],
                "energy_radius_50": metrics["energy_radius_50"],
                "energy_radius_90": metrics["energy_radius_90"],
                "spatial_second_moment": metrics["spatial_second_moment"],
                "spatial_rms_radius": metrics["spatial_rms_radius"],
                "correction_rms": metrics["correction_rms"],
                "distance_to_ch15": 0.0 if name == "CH15" else metrics["distance"],
                "selected": name == f"WDCH{selected}",
            }
        )
    write_csv(output_dir / "wdch_receptive_field_matching.csv", rows)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), constrained_layout=True)
    maximum = max(np.abs(value["response"]).max() for value in results.values())
    for axis, (name, value) in zip(axes, results.items()):
        image = axis.imshow(value["response"], cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        axis.set_title(name + (" (k*)" if name == f"WDCH{selected}" else ""))
        axis.axis("off")
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    figure_path = output_dir / "wdch_receptive_field_visualization.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return selected, rows, figure_path


def extract_hfrm28_1_input(model, x):
    x = model.conv1a(x)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)
    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x)
    x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    return F.relu(model.bn45(x))


def load_image(path: Path):
    image = Image.open(path).convert("RGB")
    if image.size != (224, 224):
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
    return TF.normalize(
        TF.to_tensor(image),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).unsqueeze(0)


def feature_row(image_id, name, value, input_rms):
    value32 = value.detach().float()
    channel_rms = value32.square().mean(dim=(-2, -1)).sqrt().cpu().numpy().ravel()
    value_rms = rms(value32)
    return {
        "image": image_id,
        "operator": name,
        "mean": float(value32.mean()),
        "std": float(value32.std(unbiased=False)),
        "min": float(value32.min()),
        "max": float(value32.max()),
        "rms": value_rms,
        "output_input_rms": value_rms / max(input_rms, 1.0e-12),
        "channel_rms_mean": float(channel_rms.mean()),
        "channel_rms_std": float(channel_rms.std()),
        "channel_rms_p10": float(np.quantile(channel_rms, 0.10)),
        "channel_rms_p50": float(np.quantile(channel_rms, 0.50)),
        "channel_rms_p90": float(np.quantile(channel_rms, 0.90)),
    }


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal Phase 0 requires BF16-capable CUDA")
    verify_validation_root(args.val_root)
    if sha256_file(args.checkpoint) != EXPECTED_A0_SHA256:
        raise AssertionError("A0 checkpoint mismatch")
    output = Path(args.output_dir)
    if (output / "wdch_phase0_summary.json").exists():
        raise FileExistsError(f"Phase-0 output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    set_seed(42, deterministic=True)

    reconstruction = []
    for shape in ((2, 8, 28, 28), (1, 8, 56, 56)):
        reconstruction.append(reconstruction_row(shape, torch.float32, "cuda"))
        reconstruction.append(reconstruction_row(shape, torch.bfloat16, "cuda"))
    write_csv(output / "wdch_reconstruction_metrics.csv", reconstruction)

    random_feature = torch.randn(2, 8, 28, 28, device="cuda")
    energy_rows = [energy_row(random_feature, "fixed_random")]
    selected_kernel, receptive_rows, figure_path = receptive_field_audit(output)

    model = load_a0_model(args.checkpoint, cam=True, device="cuda")
    model.eval()
    operator = WaveletDecoupledContext(512, selected_kernel).cuda()
    operator.eval()
    image_paths = sorted((Path(args.val_root) / "img").glob("*.png"))[: args.sample_count]
    feature_rows = []
    with torch.no_grad():
        for index, image_path in enumerate(image_paths, start=1):
            image = load_image(image_path).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feature = extract_hfrm28_1_input(model, image)
                original_context = model.hfrm_28_1.context_conv(feature)
                wdch_context = operator(feature)
            input_rms = rms(feature)
            feature_rows.extend(
                (
                    feature_row(image_path.stem, "input", feature, input_rms),
                    feature_row(image_path.stem, "CH15", original_context, input_rms),
                    feature_row(image_path.stem, f"WDCH{selected_kernel}", wdch_context, input_rms),
                )
            )
            energy_rows.append(energy_row(feature, "BCSS_validation", image_path.stem))
            if index % 16 == 0 or index == len(image_paths):
                print(f"PHASE0_FEATURE_PROGRESS {index}/{len(image_paths)}", flush=True)
    write_csv(output / "wdch_band_energy.csv", energy_rows)
    write_csv(output / "wdch_feature_statistics.csv", feature_rows)

    fp32_ok = all(
        row["relative_error"] < 1.0e-6 and row["shape_exact"]
        for row in reconstruction
        if row["dtype"] == "float32"
    )
    bf16_ok = all(
        row["relative_error"] < 2.0e-2 and row["shape_exact"]
        for row in reconstruction
        if row["dtype"] == "bfloat16"
    )
    random_energy_ok = energy_rows[0]["relative_energy_error"] < 1.0e-6
    finite_features = all(
        np.isfinite(row[key])
        for row in feature_rows
        for key in ("mean", "std", "min", "max", "rms", "output_input_rms")
    )
    amplitude_safe = all(
        1.0e-6 < row["output_input_rms"] < 100.0
        for row in feature_rows
        if row["operator"] != "input"
    )
    status = "PASS" if all((fp32_ok, bf16_ok, random_energy_ok, finite_features, amplitude_safe)) else "FAIL"
    summary = {
        "phase0_status": status,
        "selected_kernel": selected_kernel,
        "selection_rule": (
            "minimum RMS normalized discrepancy over support radius, 50%/90% "
            "energy radii, and spatial RMS radius; validation performance unused"
        ),
        "checks": {
            "fp32_reconstruction": fp32_ok,
            "bf16_reconstruction": bf16_ok,
            "orthonormal_energy": random_energy_ok,
            "finite_real_features": finite_features,
            "amplitude_safe": amplitude_safe,
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": EXPECTED_A0_SHA256,
        "a0_commit": A0_COMMIT,
        "samples": len(image_paths),
        "test_used": False,
        "training_performed": False,
    }
    write_json(output / "wdch_phase0_summary.json", summary)
    diff = subprocess.check_output(
        ["git", "diff", "--stat", A0_COMMIT], cwd=REPO_ROOT, text=True
    ).strip()
    ratios = {
        band: np.asarray([row[f"{band}_ratio"] for row in energy_rows[1:]])
        for band in ("LL", "LH", "HL", "HH")
    }
    lines = [
        "# WD-CH Phase-0 Engineering Audit",
        "",
        f"- A0 source: `{A0_COMMIT}`",
        "- Baseline location: `network/resnet38_cls.py`, HFRM28_1 receives `[B,512,28,28]` and applies depth-wise CH15.",
        f"- A0 checkpoint SHA256: `{EXPECTED_A0_SHA256}`",
        f"- Real BCSS validation feature sample: first {len(image_paths)} lexicographically sorted images; no GT used for selection.",
        "",
        "## Reconstruction and shape",
        "",
        "| dtype | shape | relative error | max abs | shape exact |",
        "|---|---|---:|---:|---|",
    ]
    for row in reconstruction:
        lines.append(
            f"| {row['dtype']} | {row['shape']} | {row['relative_error']:.3e} | "
            f"{row['max_abs_error']:.3e} | {row['shape_exact']} |"
        )
    lines += ["", "## Real-feature band energy", ""]
    for band, values in ratios.items():
        lines.append(f"- {band}: {values.mean():.6f} ± {values.std():.6f}")
    lines += [
        "",
        "## Receptive-field matching",
        "",
        f"Selected `k*={selected_kernel}` solely by impulse-response distance; no mIoU was computed.",
        f"Visualization: `{figure_path.name}`.",
        "",
        "| operator | support | r50 | r90 | RMS radius | distance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in receptive_rows:
        lines.append(
            f"| {row['operator']} | {row['support_radius']:.4f} | "
            f"{row['energy_radius_50']:.4f} | {row['energy_radius_90']:.4f} | "
            f"{row['spatial_rms_radius']:.4f} | {row['distance_to_ch15']:.6f} |"
        )
    lines += [
        "",
        "## Feature-amplitude safety",
        "",
        f"- Finite: {finite_features}",
        f"- No >100× explosion or <1e-6× collapse: {amplitude_safe}",
        "- Per-image and channel-RMS statistics are in `wdch_feature_statistics.csv`.",
        "",
        "## Source isolation",
        "",
        "The released A0 model, loss and inference files are unchanged. Current diff summary:",
        "",
        "```text",
        diff,
        "```",
        "",
        f"PHASE0_STATUS = {status}",
    ]
    (output / "wdch_phase0_engineering_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"PHASE0_STATUS = {status}", flush=True)
    if status != "PASS":
        raise RuntimeError("WD-CH Phase 0 failed; training remains forbidden")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
