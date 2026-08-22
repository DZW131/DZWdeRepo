"""Pure-PyTorch sparse adaptive context correction for MATR-v1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ANCHORS = (
    (-7.0, -7.0), (-7.0, 0.0), (-7.0, 7.0),
    (0.0, -7.0), (0.0, 0.0), (0.0, 7.0),
    (7.0, -7.0), (7.0, 0.0), (7.0, 7.0),
)


class SparseAdaptiveContext(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.channels = channels
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 27, kernel_size=3, padding=1, bias=True),
        )
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        self.a_logits = nn.Parameter(torch.zeros(channels, 9))
        self.beta_adapt = nn.Parameter(torch.tensor(-4.0))
        self.register_buffer("anchors", torch.tensor(ANCHORS, dtype=torch.float32))

    @property
    def gamma_adapt(self) -> torch.Tensor:
        return F.softplus(self.beta_adapt)

    @staticmethod
    def _grid(y: torch.Tensor, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x_norm = 2.0 * x / max(width - 1, 1) - 1.0
        y_norm = 2.0 * y / max(height - 1, 1) - 1.0
        return torch.stack((x_norm, y_norm), dim=-1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, channels, height, width = features.shape
        if channels != self.channels:
            raise ValueError("SACR channel count mismatch")
        prediction = self.predictor(features)
        offset_raw = prediction[:, :18].view(batch, 9, 2, height, width)
        mask_raw = prediction[:, 18:].view(batch, 9, height, width)

        with torch.autocast(device_type=features.device.type, enabled=False):
            source = features.float()
            offsets = 2.0 * torch.tanh(offset_raw.float())
            modulation = 2.0 * torch.sigmoid(mask_raw.float())
            sparse_weights = F.softmax(self.a_logits.float(), dim=1)
            yy, xx = torch.meshgrid(
                torch.arange(height, device=features.device, dtype=torch.float32),
                torch.arange(width, device=features.device, dtype=torch.float32),
                indexing="ij",
            )
            reference = torch.zeros_like(source)
            dynamic = torch.zeros_like(source)
            for anchor_index, (anchor_y, anchor_x) in enumerate(self.anchors):
                base_y = yy + anchor_y
                base_x = xx + anchor_x
                reference_grid = self._grid(base_y, base_x, height, width)
                reference_grid = reference_grid[None].expand(batch, -1, -1, -1)
                dynamic_y = base_y[None] + offsets[:, anchor_index, 0]
                dynamic_x = base_x[None] + offsets[:, anchor_index, 1]
                dynamic_grid = self._grid(dynamic_y, dynamic_x, height, width)
                sampled_reference = F.grid_sample(
                    source, reference_grid, mode="bilinear",
                    padding_mode="zeros", align_corners=True,
                )
                sampled_dynamic = F.grid_sample(
                    source, dynamic_grid, mode="bilinear",
                    padding_mode="zeros", align_corners=True,
                )
                channel_weight = sparse_weights[:, anchor_index].view(1, channels, 1, 1)
                reference = reference + channel_weight * sampled_reference
                dynamic = dynamic + channel_weight * modulation[:, anchor_index, None] * sampled_dynamic
            delta = dynamic - reference
            delta_rms = delta.square().mean().sqrt()
            reference_rms = reference.square().mean().sqrt()
            diagnostics = {
                "mean_abs_offset": offsets.abs().mean().detach(),
                "p95_abs_offset": torch.quantile(offsets.abs().reshape(-1), 0.95).detach(),
                "mean_modulation": modulation.mean().detach(),
                "delta_rms": delta_rms.detach(),
                "reference_rms": reference_rms.detach(),
                "delta_reference_ratio": (delta_rms / reference_rms.clamp_min(1.0e-12)).detach(),
                "gamma_adapt": self.gamma_adapt.detach().float(),
            }
        return delta.to(features.dtype), diagnostics
