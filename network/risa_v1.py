"""RISA-v1: responsibility-conditioned identity semantic assignment.

The adapter consumes detached HQMR responsibilities and detached backbone
features.  It has no access to segmentation targets or component manifests.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from network.hqmr_net import HQMRNet


class IdentityProjection(nn.Module):
    def __init__(self, in_channels: int, dimension: int = 128):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, dimension, 1, bias=False),
            nn.GroupNorm(16, dimension),
            nn.GELU(),
        )
        nn.init.xavier_uniform_(self.block[0].weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class RISA(nn.Module):
    """Small, GT-free region identity adapter used only after frozen HQMR L3."""

    def __init__(self, dimension: int = 128, classes: int = 4,
                 temperature: float = .10, gate_delta: float = .15,
                 gate_temperature: float = .05, topk_fraction: float = .20):
        super().__init__()
        self.dimension = dimension
        self.classes = classes
        self.temperature = temperature
        self.gate_delta = gate_delta
        self.gate_temperature = gate_temperature
        self.topk_fraction = topk_fraction
        self.project3 = IdentityProjection(256, dimension)
        self.project4 = IdentityProjection(512, dimension)
        self.project5 = IdentityProjection(1024, dimension)
        self.fusion = nn.Conv2d(3 * dimension, dimension, 1, bias=False)
        self.mean_norm = nn.LayerNorm(dimension)
        self.hard_projection = nn.Linear(dimension, dimension, bias=False)
        self.final_norm = nn.LayerNorm(dimension)
        self.identity_embeddings = nn.Parameter(torch.empty(classes, dimension))
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.xavier_uniform_(self.hard_projection.weight)
        nn.init.xavier_uniform_(self.identity_embeddings)

    def identity_feature(self, f3: torch.Tensor, f4: torch.Tensor,
                         f5: torch.Tensor) -> torch.Tensor:
        f3, f4, f5 = f3.detach(), f4.detach(), f5.detach()
        z3 = self.project3(f3)
        z4 = F.interpolate(self.project4(f4), z3.shape[-2:], mode="bilinear", align_corners=False)
        z5 = F.interpolate(self.project5(f5), z3.shape[-2:], mode="bilinear", align_corners=False)
        return self.fusion(torch.cat((z3, z4, z5), dim=1))

    def _logits(self, descriptors: torch.Tensor) -> torch.Tensor:
        descriptor = F.normalize(descriptors.float(), dim=-1, eps=1.e-8)
        identity = F.normalize(self.identity_embeddings.float(), dim=-1, eps=1.e-8)
        return torch.einsum("bqd,cd->bqc", descriptor, identity) / self.temperature

    def forward(self, f3: torch.Tensor, f4: torch.Tensor, f5: torch.Tensor,
                responsibility: torch.Tensor, hard_refinement: bool = True) -> dict:
        feature = self.identity_feature(f3, f4, f5)
        responsibility = responsibility.detach().float()
        if responsibility.shape[-2:] != feature.shape[-2:]:
            responsibility = F.interpolate(
                responsibility, feature.shape[-2:], mode="bilinear", align_corners=False,
            )
        batch, queries, height, width = responsibility.shape
        tokens = feature.float().flatten(2).transpose(1, 2)
        weights = responsibility.flatten(2)
        region_mean = torch.einsum("bqn,bnd->bqd", weights, tokens)
        region_mean = region_mean / weights.sum(-1, keepdim=True).clamp_min(1.e-8)
        region_mean = self.mean_norm(region_mean)

        initial_logits = self._logits(region_mean)
        initial_probability = initial_logits.softmax(-1)
        top_probability, top_class = initial_probability.topk(2, dim=-1)
        initial_margin = top_probability[..., 0] - top_probability[..., 1]
        hard_gate = torch.sigmoid(
            (self.gate_delta - initial_margin) / self.gate_temperature,
        )

        identity = F.normalize(self.identity_embeddings.float(), dim=-1, eps=1.e-8)
        rival_direction = identity[top_class[..., 0]] - identity[top_class[..., 1]]
        normalized_tokens = F.normalize(tokens, dim=-1, eps=1.e-8)
        rival_evidence = torch.einsum("bqd,bnd->bqn", rival_direction, normalized_tokens).abs()
        selection_score = weights * rival_evidence
        k = max(1, int(height * width * self.topk_fraction))
        selected_score, selected_index = selection_score.topk(k, dim=-1, sorted=False)
        del selected_score
        selected_evidence = rival_evidence.gather(-1, selected_index)
        batch_index = torch.arange(batch, device=tokens.device)[:, None, None].expand_as(selected_index)
        selected_tokens = tokens[batch_index, selected_index]
        attention = (selected_evidence / .10).softmax(-1)
        region_hard = torch.einsum("bqk,bqkd->bqd", attention, selected_tokens)
        hard_logits = self._logits(region_hard)

        if hard_refinement:
            descriptor = self.final_norm(
                region_mean + hard_gate[..., None] * self.hard_projection(region_hard),
            )
        else:
            descriptor = region_mean
        identity_logits = self._logits(descriptor)
        identity_probability = identity_logits.softmax(-1)
        local_presence = identity_logits.sigmoid()
        mil_attention = (identity_logits / .10).softmax(dim=1)
        presence_probability = (mil_attention * local_presence).sum(dim=1)
        class_map = torch.einsum(
            "bqc,bqhw->bchw", identity_probability, responsibility,
        ).clamp(0., 1.)
        entropy = -(identity_probability.clamp_min(1.e-8).log() * identity_probability).sum(-1)
        return {
            "identity_feature": feature,
            "identity_logits": identity_logits,
            "identity_prob": identity_probability,
            "initial_identity_prob": initial_probability,
            "region_mean": region_mean,
            "region_hard": region_hard,
            "hard_identity_logits": hard_logits,
            "hard_gate": hard_gate,
            "initial_margin": initial_margin,
            "identity_entropy": entropy,
            "presence_prob": presence_probability,
            "class_map": class_map,
            "rival_evidence": rival_evidence,
            "selection_score": selection_score,
            "selected_index": selected_index,
            "responsibility": responsibility,
            "hard_refinement": hard_refinement,
        }


def risa_losses(output: dict, labels: torch.Tensor, hard_refinement: bool) -> dict:
    with torch.autocast(device_type=output["identity_logits"].device.type, enabled=False):
        labels = labels.float()
        presence = output["presence_prob"].float().clamp(1.e-6, 1. - 1.e-6)
        loss_mil = F.binary_cross_entropy(presence, labels)
        logits = output["identity_logits"].float()
        maximum = logits.max(dim=1).values
        present = labels.bool()
        absent = ~present
        absent_max = maximum.masked_fill(~absent, -torch.inf).max(-1).values
        valid = present.any(-1) & absent.any(-1)
        ranking = F.relu(.2 - maximum + absent_max[:, None])
        ranking = ranking.masked_fill(~present, 0.)
        loss_rank = (ranking.sum(-1) / present.sum(-1).clamp_min(1)).masked_select(valid).mean() if valid.any() else maximum.sum() * 0.
        if hard_refinement:
            target = output["identity_prob"].detach().float()
            hard_log_probability = output["hard_identity_logits"].float().log_softmax(-1)
            divergence = F.kl_div(hard_log_probability, target, reduction="none").sum(-1)
            loss_hard = (output["hard_gate"].float() * divergence).mean()
        else:
            loss_hard = logits.sum() * 0.
        total = loss_mil + .2 * loss_rank + .05 * loss_hard
    return {"loss": total, "loss_mil": loss_mil, "loss_rank": loss_rank, "loss_hard": loss_hard}


class RISAAdapter(nn.Module):
    """Frozen HQMR plus a trainable RISA adapter."""

    def __init__(self, checkpoint: str | Path):
        super().__init__()
        self.base = HQMRNet()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        self.base.load_state_dict(state, strict=True)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.risa = RISA()
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def frozen_forward(self, raw_rgb: torch.Tensor, labels: torch.Tensor) -> dict:
        with torch.no_grad():
            normalized = (raw_rgb - self.image_mean.to(raw_rgb)) / self.image_std.to(raw_rgb)
            return self.base(normalized, labels, step=29275)

    def forward_from_base(self, base_output: dict, hard_refinement: bool = True) -> dict:
        features = base_output["features"]
        responsibility = base_output["stages"][2]["hqmr"]["basis"]
        return self.risa(
            features["F3"], features["F4"], features["F5"], responsibility,
            hard_refinement=hard_refinement,
        )

    def forward(self, raw_rgb: torch.Tensor, labels: torch.Tensor,
                hard_refinement: bool = True) -> dict:
        base = self.frozen_forward(raw_rgb, labels)
        return {"base": base, "risa": self.forward_from_base(base, hard_refinement)}

    def trainable_state_dict(self) -> dict:
        return {key: value.detach().cpu() for key, value in self.risa.state_dict().items()}

    def load_trainable_state_dict(self, state: dict):
        self.risa.load_state_dict(state, strict=True)


__all__ = ["IdentityProjection", "RISA", "RISAAdapter", "risa_losses"]
