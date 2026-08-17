"""Six-scalar selective gates for the original SSHR HFRM residuals."""

import math

import torch
import torch.nn as nn


class SelectiveRectificationGate(nn.Module):
    """One semantic and one context selectivity scalar for a target stage."""

    def __init__(self, alpha_init: float = 0.10):
        super().__init__()
        if not 0.0 < alpha_init < 1.0:
            raise ValueError(
                f"alpha_init must be strictly between 0 and 1, got {alpha_init}"
            )
        initial_logit = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_sem_logit = nn.Parameter(torch.tensor(initial_logit))
        self.alpha_ctx_logit = nn.Parameter(torch.tensor(initial_logit))

    @property
    def alpha_sem(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_sem_logit)

    @property
    def alpha_ctx(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_ctx_logit)

    def logit_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.alpha_sem_logit, self.alpha_ctx_logit

    def forward(
        self, need_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if need_map.ndim != 4 or need_map.shape[1] != 1:
            raise ValueError(
                "need_map must have shape [B, 1, H, W], got "
                f"{tuple(need_map.shape)}"
            )
        semantic_gate = 1.0 - self.alpha_sem * (1.0 - need_map)
        context_gate = 1.0 - self.alpha_ctx * (1.0 - need_map)
        return semantic_gate, context_gate
