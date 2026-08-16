"""Multi-band frequency selection and morphology-sensitivity estimation."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBandFrequencySelector(nn.Module):
    """Decompose features into telescoping bands and reweight them spatially.

    The final logit convolution is zero-initialized.  Consequently all four
    band weights are exactly one at initialization, and the residual form used
    for ``x_fs`` makes the initial selector an exact tensor identity.
    """

    def __init__(
        self,
        lowpass_kernels: Tuple[int, int, int] = (3, 7, 15),
        band_hidden: int = 16,
        morphology_eps: float = 1e-6,
        morphology_smooth_kernel: int = 3,
    ):
        super().__init__()
        if tuple(lowpass_kernels) != (3, 7, 15):
            raise ValueError("FA-MPR fixes lowpass_kernels to (3, 7, 15)")
        if band_hidden != 16:
            raise ValueError("FA-MPR fixes band_hidden to 16")
        if morphology_smooth_kernel != 3:
            raise ValueError("FA-MPR fixes morphology_smooth_kernel to 3")

        self.lowpass_kernels = tuple(lowpass_kernels)
        self.morphology_eps = float(morphology_eps)
        self.morphology_smooth_kernel = int(morphology_smooth_kernel)
        self.band_weight_network = nn.Sequential(
            nn.Conv2d(4, band_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(band_hidden, 4, kernel_size=1),
        )
        final_conv = self.band_weight_network[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)

    @staticmethod
    def _replicate_lowpass(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        padded = F.pad(x, (padding, padding, padding, padding), mode="replicate")
        return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)

    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        low_3 = self._replicate_lowpass(x, 3)
        low_7 = self._replicate_lowpass(x, 7)
        low_15 = self._replicate_lowpass(x, 15)
        return (
            x - low_3,
            low_3 - low_7,
            low_7 - low_15,
            low_15,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bands = self.decompose(x)
        energies = torch.cat(
            [band.abs().mean(dim=1, keepdim=True) for band in bands], dim=1
        )
        logits = self.band_weight_network(energies)
        weights = 2.0 * torch.sigmoid(logits)

        # Residual parameterization preserves x bit-for-bit when weights == 1.
        selected_residual = sum(
            (weights[:, index : index + 1] - 1.0) * band
            for index, band in enumerate(bands)
        )
        x_fs = x + selected_residual

        high_energy = (
            weights[:, 0:1] * energies[:, 0:1]
            + weights[:, 1:2] * energies[:, 1:2]
        )
        low_energy = (
            weights[:, 2:3] * energies[:, 2:3]
            + weights[:, 3:4] * energies[:, 3:4]
        )
        morphology = high_energy / (
            high_energy + low_energy + self.morphology_eps
        )
        padding = self.morphology_smooth_kernel // 2
        morphology = F.avg_pool2d(
            F.pad(
                morphology,
                (padding, padding, padding, padding),
                mode="replicate",
            ),
            kernel_size=self.morphology_smooth_kernel,
            stride=1,
        ).clamp(0.0, 1.0)

        diagnostics = {
            "band_energy": energies,
            "band_weights": weights,
        }
        return x_fs, morphology, diagnostics
