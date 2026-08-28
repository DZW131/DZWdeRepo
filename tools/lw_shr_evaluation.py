"""Official BCSS validation and observation-only LW-SHR diagnostics."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from tool.GenDataset import Stage1_InferDataset
from tools.lw_shr_common import (
    CAM_WEIGHTS,
    TTA_TRANSFORMS,
    ComponentMetricAccumulator,
    OfficialMetricAccumulator,
    ZoneMetricAccumulator,
    component_thresholds,
    minmax_normalize,
    official_histogram,
    presence_from_probability,
    verify_validation_root,
)


STAGES = ("56", "28_1", "28_2", "deep")


def _backbone_features(model, x):
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
    x, _ = model.b6(x, get_x_bn_relu=True)
    x = model.b7(x)
    feat_deep = F.relu(model.bn7(x))
    return feat_56, feat_28_1, feat_28_2, feat_deep


def forward_cam_with_diagnostics(model, x, diagnostics=False):
    feat_56, feat_28_1, feat_28_2, feat_deep = _backbone_features(model, x)
    feat_56_rectified = model.hfrm_56(feat_56, feat_deep)
    if diagnostics:
        feat_28_1_rectified, mechanism = model.hfrm_28_1(
            feat_28_1,
            feat_deep,
            wavelet_bank=model.wavelet_bank,
            return_diagnostics=True,
        )
    else:
        feat_28_1_rectified = model.hfrm_28_1(
            feat_28_1, feat_deep, wavelet_bank=model.wavelet_bank
        )
        mechanism = None
    feat_28_2_rectified = model.hfrm_28_2(feat_28_2, feat_deep)

    cam_56 = F.relu(model.ic_56(feat_56_rectified))
    cam_28_1 = F.relu(model.ic1(feat_28_1_rectified))
    cam_28_2 = F.relu(model.ic2(feat_28_2_rectified))
    raw_deep = model.fc8(feat_deep)
    cam_deep = F.relu(raw_deep)
    probability = torch.sigmoid(F.adaptive_avg_pool2d(raw_deep, 1).flatten(1))
    return cam_56, cam_28_1, cam_28_2, cam_deep, probability, mechanism


def _resize_unflip(cam, size, output_dims):
    cam = F.interpolate(cam, size, mode="bilinear", align_corners=False)[0]
    return torch.flip(cam, dims=output_dims) if output_dims else cam


def _rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def _masked_rms(energy_map, mask):
    values = np.asarray(energy_map, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else float("nan")


class GateDistribution:
    """Exact moments and 0.001-resolution global quantiles over gate values."""

    def __init__(self, bins=2000):
        self.bins = int(bins)
        self.histogram = np.zeros(self.bins, dtype=np.int64)
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")

    def update(self, tensor):
        values = tensor.detach().float()
        self.count += values.numel()
        self.total += float(values.double().sum())
        self.square_total += float(values.double().square().sum())
        self.minimum = min(self.minimum, float(values.min()))
        self.maximum = max(self.maximum, float(values.max()))
        hist = torch.histc(values, bins=self.bins, min=0.0, max=2.0)
        self.histogram += hist.to(dtype=torch.int64).cpu().numpy()

    def _quantile(self, probability):
        target = probability * max(self.histogram.sum() - 1, 0)
        index = int(np.searchsorted(np.cumsum(self.histogram), target, side="right"))
        index = min(max(index, 0), self.bins - 1)
        return 2.0 * (index + 0.5) / self.bins

    def result(self):
        mean = self.total / max(self.count, 1)
        variance = max(self.square_total / max(self.count, 1) - mean * mean, 0.0)
        return {
            "count": int(self.count),
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "p05": self._quantile(0.05),
            "p25": self._quantile(0.25),
            "p50": self._quantile(0.50),
            "p75": self._quantile(0.75),
            "p95": self._quantile(0.95),
            "min": self.minimum,
            "max": self.maximum,
            "quantile_resolution": 2.0 / self.bins,
        }


class MechanismAccumulator:
    def __init__(self, model):
        self.model = model
        self.gate_distribution = GateDistribution()
        self.rows = []

    @staticmethod
    def _correlation(left, right):
        left = left.detach().float().reshape(-1)
        right = right.detach().float().reshape(-1)
        left = left - left.mean()
        right = right - right.mean()
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if float(denominator) <= 1.0e-12:
            return 1.0 if torch.equal(left, right) else 0.0
        return float(torch.dot(left, right) / denominator)

    def update(self, details, truth):
        gate = details["wavelet_gate"].detach().float()
        self.gate_distribution.update(gate)
        wavelet_gate = 2.0 * torch.sigmoid(details["wavelet_logits"].detach().float())
        raw_context = details["raw_context"].detach().float()
        gated_context = details["gated_context"].detach().float()

        gate_map = gate.mean(dim=1, keepdim=True)
        gate_map = F.interpolate(
            gate_map, size=truth.shape, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        raw_map = raw_context.square().mean(dim=1, keepdim=True).sqrt()
        gated_map = gated_context.square().mean(dim=1, keepdim=True).sqrt()
        raw_map = F.interpolate(
            raw_map, size=truth.shape, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        gated_map = F.interpolate(
            gated_map, size=truth.shape, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        from tools.lw_shr_common import foreground_boundary_distance

        zones = foreground_boundary_distance(truth)
        row = {
            "spatial_std": float(gate.std(dim=(-2, -1), unbiased=False).mean()),
            "channel_std": float(gate.std(dim=1, unbiased=False).mean()),
            "boundary_gate_mean": float(gate_map[zones["boundary_le_7"]].mean()),
            "interior_gate_mean": float(gate_map[zones["interior_ge_8"]].mean()),
            "raw_context_rms": _rms(raw_context),
            "gated_context_rms": _rms(gated_context),
            "context_rms_ratio": _rms(gated_context) / max(_rms(raw_context), 1.0e-12),
            "boundary_context_ratio": _masked_rms(
                gated_map, zones["boundary_le_7"]
            ) / max(_masked_rms(raw_map, zones["boundary_le_7"]), 1.0e-12),
            "interior_context_ratio": _masked_rms(
                gated_map, zones["interior_ge_8"]
            ) / max(_masked_rms(raw_map, zones["interior_ge_8"]), 1.0e-12),
            "wave_joint_gate_correlation": self._correlation(wavelet_gate, gate),
        }
        semantic_logits = details["semantic_logits"]
        if semantic_logits is None:
            row["semantic_logit_contribution_mean_abs"] = 0.0
        else:
            contribution = self.model.hfrm_28_1.lambda_sf.detach() * semantic_logits.detach()
            row["semantic_logit_contribution_mean_abs"] = float(
                contribution.float().abs().mean()
            )

        subbands = details["wavelet_details"]["subbands"]
        energies = {
            name: float(value.detach().float().square().mean())
            for name, value in subbands.items()
        }
        total_energy = max(sum(energies.values()), 1.0e-12)
        for name, value in subbands.items():
            channel_rms = value.detach().float().square().mean(dim=(-2, -1)).sqrt()
            row[f"{name}_rms_mean"] = float(channel_rms.mean())
            row[f"{name}_rms_std"] = float(channel_rms.std(unbiased=False))
            row[f"{name}_energy_ratio"] = energies[name] / total_energy
        self.rows.append(row)

    def result(self):
        summary = {"gate": self.gate_distribution.result()}
        keys = sorted(self.rows[0]) if self.rows else []
        for key in keys:
            values = np.asarray([row[key] for row in self.rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[key] = {
                "mean": float(values.mean()) if values.size else float("nan"),
                "std": float(values.std()) if values.size else float("nan"),
            }
        module = self.model.hfrm_28_1
        summary.update(
            {
                "gamma_veto": float(module.gamma_veto.detach().float()),
                "gamma_context": float(module.gamma_context.detach().float()),
                "lambda_sf": None
                if module.lambda_sf is None
                else float(module.lambda_sf.detach().float()),
                "filters": self.model.wavelet_bank.diagnostics(),
                "images": len(self.rows),
            }
        )
        return summary


def evaluate_bcss(model, val_root, num_workers=4, prediction_output=None):
    verify_validation_root(val_root)
    model = model.cuda()
    model.eval()
    dataset = Stage1_InferDataset(os.path.join(val_root, "img"), img_size=224)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    metrics = {stage: OfficialMetricAccumulator() for stage in (*STAGES, "final")}
    zones = ZoneMetricAccumulator()
    components = ComponentMetricAccumulator(component_thresholds(val_root))
    mechanism = MechanismAccumulator(model)
    image_ids, predictions, truths, histograms = [], [], [], []

    torch.cuda.synchronize()
    started = time.time()
    with torch.no_grad():
        for index, (name_tuple, image) in enumerate(loader, start=1):
            image_id = name_tuple[0]
            truth = np.asarray(
                Image.open(Path(val_root) / "mask" / f"{image_id}.png"),
                dtype=np.uint8,
            )
            image = image.cuda(non_blocking=True)
            views = {stage: [] for stage in STAGES}
            probabilities = []
            canonical_details = None
            for view_index, (input_dims, output_dims) in enumerate(TTA_TRANSFORMS):
                augmented = torch.flip(image, dims=input_dims) if input_dims else image
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cam56, cam28, cam28_2, camdeep, probability, details = (
                        forward_cam_with_diagnostics(
                            model, augmented, diagnostics=view_index == 0
                        )
                    )
                if view_index == 0:
                    canonical_details = details
                for stage, value in (
                    ("56", cam56),
                    ("28_1", cam28),
                    ("28_2", cam28_2),
                    ("deep", camdeep),
                ):
                    views[stage].append(_resize_unflip(value, truth.shape, output_dims))
                probabilities.append(probability[0])

            mechanism.update(canonical_details, truth)
            probability = torch.stack(probabilities).mean(0).float().cpu().numpy()
            presence = presence_from_probability(probability)
            normalized = {
                stage: minmax_normalize(
                    torch.stack(values).mean(0).float().cpu().numpy()
                )
                for stage, values in views.items()
            }
            for stage in STAGES:
                response = normalized[stage] * presence.reshape(4, 1, 1)
                metrics[stage].update(truth, response.argmax(0).astype(np.uint8))
            fused = sum(CAM_WEIGHTS[stage] * normalized[stage] for stage in STAGES)
            fused *= presence.reshape(4, 1, 1)
            prediction = fused.argmax(0).astype(np.uint8)
            metrics["final"].update(truth, prediction)
            zones.update(truth, prediction)
            components.update(truth, prediction)
            histogram = official_histogram(truth, prediction)
            histograms.append(histogram)
            if prediction_output is not None:
                image_ids.append(image_id)
                predictions.append(prediction)
                truths.append(truth)
            if index % 200 == 0:
                print(f"LW_SHR_EVAL_PROGRESS {index}/{len(dataset)}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.time() - started
    if prediction_output is not None:
        target = Path(prediction_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            image_ids=np.asarray(image_ids),
            predictions=np.stack(predictions),
            truths=np.stack(truths),
            histograms=np.stack(histograms),
        )
    return {
        "scores": {stage: accumulator.result() for stage, accumulator in metrics.items()},
        "structural": {
            "zones": zones.result(),
            "components": components.result(),
        },
        "mechanism": mechanism.result(),
        "runtime": {
            "images": len(dataset),
            "seconds": elapsed,
            "seconds_per_image": elapsed / len(dataset),
            "precision": "bf16",
            "tta": TTA_TRANSFORMS,
            "fusion": [0.0, 0.6, 0.2, 0.2],
        },
        "test_used": False,
    }
