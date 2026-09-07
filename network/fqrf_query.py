"""Focus-preserved query primitives for FQRF Phase-0."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from network.hqrf_query import CHPF, DualConfidenceAllocator, MaskEmbedding, PixelDecoder, Projection


class FocusPatchQueries(nn.Module):
    """Return image-patch content queries and the independent base position B."""

    def __init__(self, image_size=224, patch_size=16, dimension=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.projection = nn.Conv2d(3, dimension, patch_size, stride=patch_size)
        self.base_position = nn.Parameter(torch.zeros(1, self.grid_size ** 2, dimension))
        nn.init.trunc_normal_(self.base_position, std=0.02)

    def forward(self, image):
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("FQRF Phase-0 requires 224x224 input")
        content = self.projection(image).flatten(2).transpose(1, 2)
        return content, self.base_position.expand(image.shape[0], -1, -1)


def sine_position_2d(batch: int, height: int, width: int, dimension: int,
                     device, dtype=torch.float32) -> torch.Tensor:
    """Parameter-free normalized 2-D sine/cosine memory position Kp."""
    if dimension % 4:
        raise ValueError("2-D sine position dimension must be divisible by four")
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32),
        torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    frequencies = torch.arange(dimension // 4, device=device, dtype=torch.float32)
    frequencies = 2.0 * math.pi * (10000.0 ** (-frequencies / max(dimension // 4, 1)))
    y = y[..., None] * frequencies
    x = x[..., None] * frequencies
    position = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1)
    return position.reshape(1, height * width, dimension).expand(batch, -1, -1).to(dtype=dtype)


class CrossFirstFocusDecoder(nn.Module):
    """Cross-attention-first post-norm decoder with optional per-query visibility."""

    operation_order = ("cross_attention", "norm", "self_attention", "norm", "ffn", "norm")

    def __init__(self, dimension=256, heads=8, hidden=1024, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dimension)
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dimension), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dimension)

    def forward(self, query, memory, visible=None):
        attention_mask = None
        if visible is not None:
            if visible.shape != (query.shape[0], query.shape[1], memory.shape[1]):
                raise ValueError("Visible-memory mask does not match query and memory")
            attention_mask = (~visible).repeat_interleave(self.heads, dim=0)
        attended, cross_weights = self.cross_attention(
            query, memory, memory, attn_mask=attention_mask,
            need_weights=True, average_attn_weights=True,
        )
        query = self.norm1(query + attended)
        attended, self_weights = self.self_attention(query, query, query, need_weights=True)
        query = self.norm2(query + attended)
        query = self.norm3(query + self.ffn(query))
        return query, {"cross_attention": cross_weights, "self_attention": self_weights}


class DynamicFocusPosition(nn.Module):
    """DFPQ update h(A Kp + B), with detached FP32-normalized attention."""

    def __init__(self, dimension=256):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, dimension), nn.ReLU(), nn.Linear(dimension, dimension))

    def forward(self, attention, memory_position, base_position):
        with torch.autocast(device_type=attention.device.type, enabled=False):
            normalized = attention.detach().float()
            normalized = normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            focused = torch.bmm(normalized, memory_position.detach().float())
            value = focused + base_position.float()
        return self.layers(value)


@torch.no_grad()
def previous_mask_visibility(mask_logits, memory_hw, threshold=0.15):
    """Detached previous-mask visibility with global fallback for empty rows."""
    if threshold != 0.15:
        raise ValueError("FQRF attention-mask threshold is frozen at 0.15")
    probability = mask_logits.detach().float().sigmoid()
    resized = F.interpolate(probability, memory_hw, mode="bilinear", align_corners=False)
    visible = resized.flatten(2) >= threshold
    all_masked = ~visible.any(dim=-1)
    visible = visible | all_masked[..., None]
    return visible, {
        "visible_ratio": visible.float().mean(dim=-1),
        "all_masked_before_fallback": all_masked,
        "fallback_to_global": all_masked.clone(),
    }


__all__ = [
    "CHPF", "CrossFirstFocusDecoder", "DualConfidenceAllocator", "DynamicFocusPosition",
    "FocusPatchQueries", "MaskEmbedding", "PixelDecoder", "Projection",
    "previous_mask_visibility", "sine_position_2d",
]
