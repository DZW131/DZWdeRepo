"""RSBR-v0 region semantic and transition refinement.

The proposal path is deliberately analytical and detached.  Only the two
residual heads are trainable; no dense labels enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASS = 4
MIN_REGION_AREA = 2
INTERNAL_TRANSITION_FRACTION = 0.25
CONNECTIVITY = 8


@dataclass
class RSBRResult:
    refined_cam: torch.Tensor
    delta_core: torch.Tensor
    delta_transition: torch.Tensor
    region_logits: List[torch.Tensor]
    statistics: Dict[str, float]
    structures: Optional[List[Tuple[Tuple[int, ...], ...]]] = None
    per_image_component_counts: Optional[List[int]] = None


def _spatial_normalize(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(2)
    minimum = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    maximum = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def _resize_like(cam: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if cam.shape[-2:] == reference.shape[-2:]:
        return cam
    return F.interpolate(cam, reference.shape[-2:], mode="bilinear", align_corners=False)


def _deterministic_top_fraction(
    flat_indices: np.ndarray,
    deviations: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Highest deviations with flattened-index ascending as the tie-break."""

    count = max(1, int(np.ceil(len(flat_indices) * fraction)))
    order = np.lexsort((flat_indices, -deviations))
    return flat_indices[order[:count]]


def _component_records(label_map: np.ndarray) -> Sequence[Tuple[int, int, np.ndarray]]:
    records = []
    for predicted_class in range(N_CLASS):
        count, components = cv2.connectedComponents(
            np.asarray(label_map == predicted_class, dtype=np.uint8),
            connectivity=CONNECTIVITY,
        )
        for component_id in range(1, count):
            indices = np.flatnonzero(components.reshape(-1) == component_id)
            records.append((predicted_class, component_id, indices))
    return records


class RSBRRefinement(nn.Module):
    """Minimum trainable RSBR-v0 module for the post-HFRM H28_1 feature."""

    def __init__(self, feature_channels: int = 512, n_class: int = N_CLASS):
        super().__init__()
        if feature_channels != 512 or n_class != N_CLASS:
            raise ValueError("RSBR-v0 is frozen to H28_1=512 and four BCSS classes")
        self.feature_channels = feature_channels
        self.n_class = n_class
        self.region_semantic_head = nn.Linear(feature_channels, n_class)
        self.transition_head = nn.Sequential(
            nn.Conv1d(3 * feature_channels + 1 + n_class, 128, kernel_size=1),
            nn.ReLU(inplace=False),
            nn.Conv1d(128, n_class, kernel_size=1),
        )
        nn.init.zeros_(self.region_semantic_head.weight)
        nn.init.zeros_(self.region_semantic_head.bias)
        nn.init.zeros_(self.transition_head[-1].weight)
        nn.init.zeros_(self.transition_head[-1].bias)

    def proposal_labels(
        self,
        cam_56: torch.Tensor,
        cam_28_1: torch.Tensor,
        cam_28_2: torch.Tensor,
        cam_deep: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        """Detached exact released-fusion foreground proposal at H28_1.

        The released helper computes a white-tissue background mask but does
        not concatenate it into ``cam_score`` before argmax.  Consequently the
        exact released prediction is a four-channel foreground argmax; RSBR
        preserves that behavior and does not introduce a new threshold.
        """

        with torch.no_grad():
            reference = cam_28_1
            normalized = [
                _spatial_normalize(F.relu(_resize_like(cam, reference).detach()).float())
                for cam in (cam_56, cam_28_1, cam_28_2, cam_deep)
            ]
            fused = (
                0.0 * normalized[0]
                + 0.6 * normalized[1]
                + 0.2 * normalized[2]
                + 0.2 * normalized[3]
            )
            fused = fused * presence.detach().float().view(-1, self.n_class, 1, 1)
            return fused.argmax(dim=1)

    def forward(
        self,
        feature: torch.Tensor,
        cam_56: torch.Tensor,
        cam_28_1: torch.Tensor,
        cam_28_2: torch.Tensor,
        cam_deep: torch.Tensor,
        presence: torch.Tensor,
        collect_structures: bool = False,
    ) -> RSBRResult:
        if feature.shape[1] != self.feature_channels:
            raise ValueError(f"Expected {self.feature_channels} H28_1 channels")
        if presence.shape != (feature.shape[0], self.n_class):
            raise ValueError("presence must have shape [batch, 4]")

        proposal = self.proposal_labels(
            cam_56, cam_28_1, cam_28_2, cam_deep, presence
        ).detach().cpu().numpy().astype(np.uint8)
        batch, _, height, width = cam_28_1.shape
        delta_core_images = []
        delta_transition_images = []
        region_logits: List[torch.Tensor] = []
        all_structures = [] if collect_structures else None

        total_components = 0
        total_semantic_regions = 0
        total_tiny = 0
        total_core_pixels = 0
        total_transition_pixels = 0
        images_without_regions = 0
        token_norm_sum = 0.0
        per_image_component_counts = []

        for batch_index in range(batch):
            feature_flat = feature[batch_index].reshape(self.feature_channels, -1)
            cam_flat = cam_28_1[batch_index].reshape(self.n_class, -1)
            component_records = _component_records(proposal[batch_index])
            total_components += len(component_records)
            per_image_component_counts.append(len(component_records))

            semantic_tokens = []
            semantic_base_logits = []
            core_indices = []
            transition_inputs = []
            transition_indices = []
            image_structures = []

            for predicted_class, component_id, indices_np in component_records:
                area = int(len(indices_np))
                indices = torch.as_tensor(indices_np, device=feature.device, dtype=torch.long)
                pixels = feature_flat.index_select(1, indices)
                token = pixels.mean(dim=1)
                normalized_pixels = F.normalize(pixels.float(), dim=0, eps=1e-8)
                normalized_token = F.normalize(token.float(), dim=0, eps=1e-8)
                deviation = 1.0 - (normalized_pixels * normalized_token[:, None]).sum(dim=0)

                region_mask = np.zeros((height, width), dtype=np.uint8)
                region_mask.reshape(-1)[indices_np] = 1
                eroded = cv2.erode(region_mask, np.ones((3, 3), np.uint8), iterations=1)
                outer_np = np.flatnonzero((region_mask - eroded).reshape(-1) > 0)
                internal_np = _deterministic_top_fraction(
                    indices_np,
                    deviation.detach().cpu().numpy().astype(np.float64),
                    INTERNAL_TRANSITION_FRACTION,
                )
                transition_np = np.union1d(outer_np, internal_np).astype(np.int64)
                core_np = np.setdiff1d(indices_np, transition_np, assume_unique=True)

                transition_index = torch.as_tensor(
                    transition_np, device=feature.device, dtype=torch.long
                )
                transition_feature = feature_flat.index_select(1, transition_index)
                transition_cam = cam_flat.index_select(1, transition_index)
                local_position = np.searchsorted(indices_np, transition_np)
                transition_deviation = deviation.index_select(
                    0,
                    torch.as_tensor(local_position, device=feature.device, dtype=torch.long),
                ).to(dtype=transition_feature.dtype)
                token_column = token[:, None].expand(-1, transition_feature.shape[1])
                transition_inputs.append(torch.cat(
                    (
                        transition_feature,
                        token_column,
                        transition_feature - token_column,
                        transition_deviation[None],
                        transition_cam,
                    ),
                    dim=0,
                ))
                transition_indices.append(transition_index)
                total_transition_pixels += int(len(transition_np))

                if area >= MIN_REGION_AREA:
                    semantic_tokens.append(token)
                    semantic_base_logits.append(cam_flat.index_select(1, indices).mean(dim=1))
                    core_indices.append(torch.as_tensor(
                        core_np, device=feature.device, dtype=torch.long
                    ))
                    total_semantic_regions += 1
                    total_core_pixels += int(len(core_np))
                    token_norm_sum += float(token.detach().float().norm().item())
                else:
                    total_tiny += 1

                if collect_structures:
                    image_structures.append((
                        int(predicted_class), int(component_id),
                        tuple(int(item) for item in indices_np),
                        tuple(int(item) for item in core_np),
                        tuple(int(item) for item in transition_np),
                    ))

            core_flat = cam_28_1.new_zeros((self.n_class, height * width))
            if semantic_tokens:
                tokens = torch.stack(semantic_tokens)
                semantic_delta = self.region_semantic_head(tokens)
                base_logits = torch.stack(semantic_base_logits)
                region_logits.append(base_logits + semantic_delta)
                for region_index, target_indices in enumerate(core_indices):
                    if target_indices.numel() > 0:
                        values = semantic_delta[region_index][:, None].expand(
                            -1, target_indices.numel()
                        )
                        core_flat = core_flat.index_copy(1, target_indices, values)
            else:
                images_without_regions += 1
                region_logits.append(cam_28_1.new_empty((0, self.n_class)))

            transition_flat = cam_28_1.new_zeros((self.n_class, height * width))
            if transition_inputs:
                inputs = torch.cat(transition_inputs, dim=1).unsqueeze(0)
                values = self.transition_head(inputs).squeeze(0)
                targets = torch.cat(transition_indices)
                transition_flat = transition_flat.index_copy(1, targets, values)

            delta_core_images.append(core_flat.reshape(self.n_class, height, width))
            delta_transition_images.append(
                transition_flat.reshape(self.n_class, height, width)
            )
            if collect_structures:
                all_structures.append(tuple(image_structures))

        delta_core = torch.stack(delta_core_images)
        delta_transition = torch.stack(delta_transition_images)
        refined_cam = cam_28_1 + delta_core + delta_transition
        pixels = float(batch * height * width)
        residual = delta_core + delta_transition
        base_rms = cam_28_1.detach().float().square().mean().sqrt()
        residual_rms = residual.detach().float().square().mean().sqrt()
        statistics = {
            "components_per_image": total_components / float(batch),
            "semantic_regions_per_image": total_semantic_regions / float(batch),
            "core_fraction": total_core_pixels / pixels,
            "transition_fraction": total_transition_pixels / pixels,
            "tiny_region_fraction": total_tiny / float(max(total_components, 1)),
            "no_region_fraction": images_without_regions / float(batch),
            "mean_region_token_norm": token_norm_sum / float(max(total_semantic_regions, 1)),
            "rms_delta_core": float(delta_core.detach().float().square().mean().sqrt().item()),
            "rms_delta_transition": float(
                delta_transition.detach().float().square().mean().sqrt().item()
            ),
            "max_abs_delta_core": float(delta_core.detach().float().abs().max().item()),
            "max_abs_delta_transition": float(
                delta_transition.detach().float().abs().max().item()
            ),
            "residual_ratio": float((residual_rms / (base_rms + 1e-8)).item()),
            "mean_abs_delta_core": float(delta_core.detach().float().abs().mean().item()),
            "mean_abs_delta_transition": float(
                delta_transition.detach().float().abs().mean().item()
            ),
        }
        return RSBRResult(
            refined_cam=refined_cam,
            delta_core=delta_core,
            delta_transition=delta_transition,
            region_logits=region_logits,
            statistics=statistics,
            structures=all_structures,
            per_image_component_counts=per_image_component_counts,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
