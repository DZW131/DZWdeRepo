"""Spatial and frequency-domain autopsy of trained CH depthwise kernels."""

from __future__ import annotations

import numpy as np

from tools.hma_v0 import CONTEXT_KERNEL, STAGES


def _anisotropy_and_center_of_mass(kernel):
    size = kernel.shape[0]
    coordinates = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    weights = np.abs(kernel).astype(np.float64)
    total = weights.sum() + 1e-12
    center_y = float((weights * yy).sum() / total)
    center_x = float((weights * xx).sum() / total)
    dy, dx = yy - center_y, xx - center_x
    covariance = np.asarray([
        [(weights * dy * dy).sum(), (weights * dy * dx).sum()],
        [(weights * dy * dx).sum(), (weights * dx * dx).sum()],
    ]) / total
    eigenvalues = np.linalg.eigvalsh(covariance)
    anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / (eigenvalues.sum() + 1e-12))
    return center_y, center_x, float(np.hypot(center_y, center_x)), anisotropy


def _frequency_energy(kernel, fft_size=64):
    spectrum = np.fft.fftshift(np.fft.fft2(kernel, s=(fft_size, fft_size)))
    energy = np.abs(spectrum) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(fft_size))
    fy, fx = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(fx * fx + fy * fy) / 0.5
    masks = {
        "low_frequency_energy": radius <= 0.20,
        "mid_frequency_energy": (radius > 0.20) & (radius <= 0.50),
        "high_frequency_energy": radius > 0.50,
    }
    total = energy.sum() + 1e-12
    values = {name: float(energy[mask].sum() / total) for name, mask in masks.items()}
    values["hf_lf_ratio"] = float(
        values["high_frequency_energy"] / (values["low_frequency_energy"] + 1e-12)
    )
    return values


def channel_kernel_metrics(stage, kernels):
    kernels = np.asarray(kernels, dtype=np.float64)
    if kernels.ndim == 4:
        kernels = kernels[:, 0]
    if kernels.shape[1:] != (CONTEXT_KERNEL, CONTEXT_KERNEL):
        raise ValueError(f"Unexpected {stage} context kernel shape {kernels.shape}")
    uniform = np.full((CONTEXT_KERNEL, CONTEXT_KERNEL), 1.0 / CONTEXT_KERNEL**2)
    uniform_norm = np.linalg.norm(uniform)
    outer_mask = np.zeros_like(uniform, dtype=bool)
    outer_mask[[0, -1], :] = True
    outer_mask[:, [0, -1]] = True
    center_mask = np.zeros_like(uniform, dtype=bool)
    middle = CONTEXT_KERNEL // 2
    center_mask[middle - 1:middle + 2, middle - 1:middle + 2] = True
    rows = []
    for channel, kernel in enumerate(kernels):
        center_mean = float(kernel[center_mask].mean())
        outer_mean = float(kernel[outer_mask].mean())
        center_y, center_x, center_distance, anisotropy = _anisotropy_and_center_of_mass(kernel)
        row = {
            "stage": stage,
            "channel": channel,
            "mean_weight": float(kernel.mean()),
            "dc_gain": float(kernel.sum()),
            "negative_fraction": float((kernel < 0).mean()),
            "positive_fraction": float((kernel > 0).mean()),
            "l1_norm": float(np.abs(kernel).sum()),
            "l2_norm": float(np.linalg.norm(kernel)),
            "uniform_cosine": float(
                np.sum(kernel * uniform)
                / (np.linalg.norm(kernel) * uniform_norm + 1e-12)
            ),
            "center_weight_mean": center_mean,
            "outer_ring_weight_mean": outer_mean,
            "center_outer_ratio": float(center_mean / (outer_mean + 1e-12)),
            "center_of_mass_y": center_y,
            "center_of_mass_x": center_x,
            "center_of_mass_distance": center_distance,
            "anisotropy": anisotropy,
            **_frequency_energy(kernel),
        }
        rows.append(row)
    return rows


def distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def summarize_kernel_rows(rows):
    metrics = (
        "mean_weight", "dc_gain", "negative_fraction", "positive_fraction",
        "l1_norm", "l2_norm", "uniform_cosine", "center_outer_ratio",
        "center_of_mass_distance", "anisotropy", "low_frequency_energy",
        "mid_frequency_energy", "high_frequency_energy", "hf_lf_ratio",
    )
    result = {}
    for stage in STAGES:
        selected = [row for row in rows if row["stage"] == stage]
        result[stage] = {
            "channels": len(selected),
            **{
                metric: distribution_summary([row[metric] for row in selected])
                for metric in metrics
            },
        }
        median_cosine = result[stage]["uniform_cosine"]["median"]
        median_negative = result[stage]["negative_fraction"]["median"]
        median_hf_lf = result[stage]["hf_lf_ratio"]["median"]
        if median_cosine >= 0.80 and median_negative <= 0.10 and median_hf_lf <= 0.25:
            behavior = "CH_BEHAVES_AS_HOMOGENIZER"
        elif median_cosine < 0.50 or median_negative > 0.20 or median_hf_lf > 1.0:
            behavior = "CH_FREE_FILTER_BEHAVIOR"
        else:
            behavior = "CH_MIXED_FILTER_BEHAVIOR"
        result[stage]["behavior"] = behavior
    return result


def audit_context_kernels(model):
    modules = {
        "56": model.hfrm_56,
        "28_1": model.hfrm_28_1,
        "28_2": model.hfrm_28_2,
    }
    rows = []
    for stage in STAGES:
        rows.extend(channel_kernel_metrics(
            stage, modules[stage].context_conv.weight.detach().float().cpu().numpy()
        ))
    return rows, summarize_kernel_rows(rows)
