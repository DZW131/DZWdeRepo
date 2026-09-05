"""PCRE-style patch-query decoder, pixel decoder, mask MLP and PCA."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchQueries(nn.Module):
    def __init__(self, image_size=224, patch_size=16, dimension=256):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.projection = nn.Conv2d(3, dimension, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.grid_size ** 2, dimension))
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, image):
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError("HQRF Phase-0 requires 224x224 input")
        query = self.projection(image).flatten(2).transpose(1, 2)
        return query + self.position


class QueryDecoderLayer(nn.Module):
    """Post-norm DETR layer matching the PCRE operation order."""
    def __init__(self, dimension=256, heads=8, hidden=1024, dropout=0.1):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dimension)
        self.cross_attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dimension), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(dimension)

    def forward(self, query, memory):
        attended, self_weights = self.self_attention(query, query, query, need_weights=True)
        query = self.norm1(query + attended)
        attended, cross_weights = self.cross_attention(query, memory, memory, need_weights=True)
        query = self.norm2(query + attended)
        query = self.norm3(query + self.ffn(query))
        return query, {"self_attention": self_weights, "cross_attention": cross_weights}


class Projection(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )


class CHPF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.context = nn.Conv2d(channels, channels, 15, padding=7, groups=channels, bias=False)
        nn.init.constant_(self.context.weight, 1.0 / 225.0)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, value):
        return value + self.gamma * self.context(value)


class PixelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.f4_projection = Projection(512, 128)
        self.f4_chpf = CHPF(128)
        self.f3_projection = Projection(256, 128)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
        )

    def forward(self, f4, f3):
        raw4 = self.f4_projection(f4)
        context4 = self.f4_chpf(raw4)
        raw3 = self.f3_projection(f3)
        context4_up = F.interpolate(context4, raw3.shape[-2:], mode="bilinear", align_corners=False)
        pixel = self.fusion(torch.cat((context4_up, raw3), dim=1))
        return pixel, {"F4_raw": raw4, "F4_context": context4, "F3_raw": raw3}


class MaskEmbedding(nn.Module):
    def __init__(self, dimension=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(),
            nn.Linear(dimension, dimension), nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, query):
        return self.layers(query)


class DualConfidenceAllocator(nn.Module):
    def __init__(self, dimension=256, classes=4):
        super().__init__()
        self.class_head = nn.Linear(dimension, classes)
        self.patch_head = nn.Linear(dimension, classes)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.xavier_uniform_(self.patch_head.weight)
        nn.init.zeros_(self.class_head.bias)
        nn.init.zeros_(self.patch_head.bias)

    def forward(self, query):
        class_logits = self.class_head(query)
        patch_logits = self.patch_head(query)
        with torch.autocast(device_type=query.device.type, enabled=False):
            p_class = F.softmax(class_logits.float(), dim=-1)
            p_patch = F.softmax(patch_logits.float(), dim=1)
            joint = p_class * p_patch
            image_probability = joint.sum(dim=1).clamp(1.0e-6, 1.0 - 1.0e-6)
        return {
            "class_logits": class_logits,
            "patch_logits": patch_logits,
            "p_class": p_class,
            "p_patch": p_patch,
            "joint": joint,
            "image_probability": image_probability,
        }
