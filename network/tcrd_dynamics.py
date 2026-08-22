"""Frozen TCRD-v0 evidence dynamics used by the matched utility gate."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


BRANCHES = ("C0", "D", "R", "DR")
NEIGHBOR_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def inverse_bounded_step(value: float) -> float:
    probability = (value - 0.05) / 0.45
    return math.log(probability / (1.0 - probability))


class TCRDDynamics(nn.Module):
    """Three shared reaction/diffusion steps on the CAM28_1 evidence field."""

    def __init__(self, branch: str, n_class: int = 4, steps: int = 3):
        super().__init__()
        if branch not in BRANCHES:
            raise ValueError(f"Unknown TCRD utility branch: {branch}")
        if steps != 3:
            raise ValueError("The frozen TCRD-v0 contract requires exactly T=3")
        self.branch = branch
        self.n_class = n_class
        self.steps = steps
        self.use_diffusion = branch in ("D", "DR")
        self.use_reaction = branch in ("R", "DR")

        if self.use_diffusion:
            self.theta_kappa = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
            self.beta_d = nn.Parameter(torch.tensor(inverse_bounded_step(0.10)))
        else:
            self.register_parameter("theta_kappa", None)
            self.register_parameter("beta_d", None)

        if self.use_reaction:
            self.pair_raw = nn.Parameter(
                torch.full((n_class * (n_class - 1) // 2,), inverse_softplus(1.0))
            )
            self.beta_r = nn.Parameter(torch.tensor(inverse_bounded_step(0.10)))
        else:
            self.register_parameter("pair_raw", None)
            self.register_parameter("beta_r", None)

    @property
    def eta_d(self) -> Optional[torch.Tensor]:
        if not self.use_diffusion:
            return None
        return 0.05 + 0.45 * torch.sigmoid(self.beta_d)

    @property
    def eta_r(self) -> Optional[torch.Tensor]:
        if not self.use_reaction:
            return None
        return 0.05 + 0.45 * torch.sigmoid(self.beta_r)

    @property
    def kappa(self) -> Optional[torch.Tensor]:
        if not self.use_diffusion:
            return None
        return F.softplus(self.theta_kappa)

    def competition_matrix(self) -> Optional[torch.Tensor]:
        if not self.use_reaction:
            return None
        matrix = self.pair_raw.new_zeros((self.n_class, self.n_class))
        values = F.softplus(self.pair_raw)
        cursor = 0
        for row in range(self.n_class):
            for column in range(row + 1, self.n_class):
                value = values[cursor]
                matrix[row, column] = value
                matrix[column, row] = value
                cursor += 1
        return matrix

    @staticmethod
    def _shift(tensor: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        padded = F.pad(tensor, (1, 1, 1, 1), mode="constant", value=0.0)
        return padded[..., 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]

    def conductance(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.use_diffusion:
            raise RuntimeError("Conductance requested for a non-diffusion branch")
        feature = F.normalize(feature.float(), dim=1, eps=1.0e-6)
        valid_base = feature.new_ones((feature.shape[0], 1, *feature.shape[-2:]))
        conductances = []
        for dy, dx in NEIGHBOR_OFFSETS:
            neighbor = self._shift(feature, dy, dx)
            valid = self._shift(valid_base, dy, dx)
            distance = (1.0 - (feature * neighbor).sum(dim=1, keepdim=True)).clamp(0.0, 2.0)
            conductances.append(torch.exp(-self.kappa * distance) * valid)
        weights = torch.cat(conductances, dim=1)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)

    def _diffusion_update(
        self, evidence: torch.Tensor, weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        neighbors = torch.stack(
            [self._shift(evidence, dy, dx) for dy, dx in NEIGHBOR_OFFSETS], dim=2
        )
        neighbor_average = (neighbors * weights.unsqueeze(1)).sum(dim=2)
        update = self.eta_d * (neighbor_average - evidence)
        return evidence + update, update

    def _reaction_update(
        self, evidence: torch.Tensor, active_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if active_classes is None:
            raise ValueError("Reaction requires an explicit active-class mask")
        active_classes = active_classes.to(device=evidence.device, dtype=torch.bool)
        base_matrix = self.competition_matrix()
        batch_updates = []
        for batch_index in range(evidence.shape[0]):
            active = torch.nonzero(active_classes[batch_index], as_tuple=False).flatten()
            if active.numel() < 2:
                batch_updates.append(torch.zeros_like(evidence[batch_index]))
                continue
            probabilities = torch.softmax(evidence[batch_index, active], dim=0)
            matrix = base_matrix.index_select(0, active).index_select(1, active)
            competitor_count = float(active.numel() - 1)
            matrix = matrix * (
                competitor_count / matrix.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
            )
            competitors = torch.einsum("ij,jhw->ihw", matrix, probabilities)
            competitors = competitors / matrix.sum(dim=1).view(-1, 1, 1).clamp_min(1.0e-8)
            reaction = probabilities * (probabilities - competitors)
            reaction = reaction - reaction.mean(dim=0, keepdim=True)
            active_update = self.eta_r * reaction
            update = torch.zeros_like(evidence[batch_index]).index_copy(
                0, active, active_update
            )
            batch_updates.append(update)
        update = torch.stack(batch_updates, dim=0)
        return evidence + update, update

    def forward(
        self,
        evidence: torch.Tensor,
        feature: torch.Tensor,
        active_classes: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ):
        if self.branch == "C0":
            diagnostics: Dict[str, Optional[torch.Tensor]] = {
                "z0": evidence.detach(), "zt": evidence.detach(),
                "conductance": None,
                "diffusion_update": torch.zeros_like(evidence).detach(),
                "reaction_update": torch.zeros_like(evidence).detach(),
            }
            return (evidence, diagnostics) if return_diagnostics else evidence

        output_dtype = evidence.dtype
        evidence_fp32 = evidence.float()
        feature_fp32 = feature.float()
        diffusion_total = torch.zeros_like(evidence_fp32)
        reaction_total = torch.zeros_like(evidence_fp32)
        weights = self.conductance(feature_fp32) if self.use_diffusion else None

        for _ in range(self.steps):
            if self.use_reaction:
                evidence_fp32, update = self._reaction_update(evidence_fp32, active_classes)
                reaction_total = reaction_total + update
            if self.use_diffusion:
                evidence_fp32, update = self._diffusion_update(evidence_fp32, weights)
                diffusion_total = diffusion_total + update

        result = evidence_fp32.to(output_dtype)
        if not return_diagnostics:
            return result
        diagnostics = {
            "z0": evidence.detach().float(),
            "zt": result.detach().float(),
            "conductance": None if weights is None else weights.detach(),
            "diffusion_update": diffusion_total.detach(),
            "reaction_update": reaction_total.detach(),
            "active_classes": None if active_classes is None else active_classes.detach().bool(),
        }
        return result, diagnostics
