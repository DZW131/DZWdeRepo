"""RGR-v0 minimal region-graph semantic reasoning.

Region construction is analytical and detached.  The module uses no dense
labels: it builds a complete directed graph over coarse predicted regions and
broadcasts learned isolated/relational semantic residuals to semantic cores.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_CLASS = 4
FEATURE_CHANNELS = 512
LATENT_CHANNELS = 128
MIN_REGION_AREA = 2
INTERNAL_AMBIGUITY_FRACTION = 0.25
CONNECTIVITY = 8


@dataclass
class RGRResult:
    refined_cam: torch.Tensor
    delta_iso: torch.Tensor
    delta_graph: torch.Tensor
    region_logits: List[torch.Tensor]
    statistics: Dict[str, float]
    structures: Optional[List[Tuple[Tuple[int, ...], ...]]] = None
    per_image_region_counts: Optional[List[int]] = None


def _spatial_normalize(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(2)
    minimum = flat.min(dim=2, keepdim=True).values.unsqueeze(-1)
    maximum = flat.max(dim=2, keepdim=True).values.unsqueeze(-1)
    return (cam - minimum) / (maximum - minimum + 1e-8)


def _resize_like(cam: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if cam.shape[-2:] == reference.shape[-2:]:
        return cam
    return F.interpolate(cam, reference.shape[-2:], mode="bilinear", align_corners=False)


def _deterministic_top_fraction(indices: np.ndarray, values: np.ndarray) -> np.ndarray:
    count = max(1, int(np.ceil(len(indices) * INTERNAL_AMBIGUITY_FRACTION)))
    order = np.lexsort((indices, -values))
    return indices[order[:count]]


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


def _core_indices(
    indices: np.ndarray,
    feature_pixels: torch.Tensor,
    token: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    normalized_pixels = F.normalize(feature_pixels.float(), dim=0, eps=1e-8)
    normalized_token = F.normalize(token.float(), dim=0, eps=1e-8)
    deviation = 1.0 - (normalized_pixels * normalized_token[:, None]).sum(dim=0)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask.reshape(-1)[indices] = 1
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    outer = np.flatnonzero((mask - eroded).reshape(-1) > 0)
    internal = _deterministic_top_fraction(
        indices,
        deviation.detach().cpu().numpy().astype(np.float64),
    )
    transition = np.union1d(outer, internal).astype(np.int64)
    core = np.setdiff1d(indices, transition, assume_unique=True)
    return core, transition


def _edge_geometry(
    region_masks: Sequence[np.ndarray],
    centroids: torch.Tensor,
    tokens: torch.Tensor,
    coarse_classes: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return source, target and exact 4-D features for all i!=j edges."""

    node_count = len(region_masks)
    device = tokens.device
    if node_count <= 1:
        empty_index = torch.empty(0, dtype=torch.long, device=device)
        empty_feature = tokens.new_empty((0, 4))
        return empty_index, empty_index, empty_feature

    normalized_tokens = F.normalize(tokens.float(), dim=1, eps=1e-8)
    dilated = [
        cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        for mask in region_masks
    ]
    sources, targets, features = [], [], []
    diagonal = float(np.sqrt(height * height + width * width))
    for target in range(node_count):
        for source in range(node_count):
            if source == target:
                continue
            cosine = (normalized_tokens[target] * normalized_tokens[source]).sum()
            distance = torch.linalg.vector_norm(centroids[target] - centroids[source]) / diagonal
            touch = float(np.any(dilated[target] & region_masks[source].astype(bool)))
            same = float(coarse_classes[target].item() == coarse_classes[source].item())
            sources.append(source)
            targets.append(target)
            features.append(torch.stack((
                cosine.to(dtype=tokens.dtype),
                distance.to(dtype=tokens.dtype),
                tokens.new_tensor(touch),
                tokens.new_tensor(same),
            )))
    return (
        torch.as_tensor(sources, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
        torch.stack(features),
    )


class RGRRefinement(nn.Module):
    """One-layer complete region graph with decomposed semantic residuals."""

    def __init__(
        self,
        feature_channels: int = FEATURE_CHANNELS,
        latent_channels: int = LATENT_CHANNELS,
        n_class: int = N_CLASS,
    ):
        super().__init__()
        if feature_channels != FEATURE_CHANNELS or n_class != N_CLASS:
            raise ValueError("RGR-v0 is frozen to H28_1=512 and four BCSS classes")
        self.feature_channels = feature_channels
        self.latent_channels = latent_channels
        self.n_class = n_class
        self.node_projection = nn.Linear(feature_channels, latent_channels)
        self.edge_gate = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(inplace=False),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.value_projection = nn.Linear(latent_channels, latent_channels, bias=False)
        self.message_projection = nn.Linear(latent_channels, latent_channels)
        self.isolated_head = nn.Linear(feature_channels, n_class)
        self.graph_head = nn.Linear(latent_channels, n_class)
        nn.init.zeros_(self.isolated_head.weight)
        nn.init.zeros_(self.isolated_head.bias)
        nn.init.zeros_(self.graph_head.weight)
        nn.init.zeros_(self.graph_head.bias)

    def proposal_labels(
        self,
        cam_56: torch.Tensor,
        cam_28_1: torch.Tensor,
        cam_28_2: torch.Tensor,
        cam_deep: torch.Tensor,
        presence: torch.Tensor,
    ) -> torch.Tensor:
        """Detached released 0/0.6/0.2/0.2 foreground proposal."""

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

    def _graph_reasoning(
        self,
        tokens: torch.Tensor,
        base_logits: torch.Tensor,
        masks: Sequence[np.ndarray],
        centroids: torch.Tensor,
        height: int,
        width: int,
    ):
        node_count = tokens.shape[0]
        node_started = time.perf_counter()
        hidden = F.relu(self.node_projection(tokens), inplace=False)
        coarse_classes = base_logits.argmax(dim=1)
        node_encoding_seconds = time.perf_counter() - node_started
        graph_started = time.perf_counter()
        source, target, edge_features = _edge_geometry(
            masks, centroids, tokens, coarse_classes, height, width
        )
        graph_construction_seconds = time.perf_counter() - graph_started
        message_started = time.perf_counter()
        if node_count <= 1:
            message = hidden.new_zeros(hidden.shape)
            context = hidden.new_zeros(hidden.shape)
            graph_delta = base_logits.new_zeros(base_logits.shape)
            gates = hidden.new_empty((0,))
        else:
            gates = self.edge_gate(edge_features).squeeze(1)
            values = self.value_projection(hidden).index_select(0, source)
            weighted = values * gates[:, None]
            numerator = hidden.new_zeros(hidden.shape).index_add(0, target, weighted)
            denominator = hidden.new_zeros((node_count,)).index_add(0, target, gates)
            message = numerator / (denominator[:, None] + 1e-8)
            context = F.relu(self.message_projection(message), inplace=False)
            contextualized = hidden + context
            graph_delta = self.graph_head(contextualized)

        isolated_delta = self.isolated_head(tokens)
        stats = self._graph_statistics(
            gates, edge_features, message, hidden + context, node_count
        )
        stats.update({
            "node_encoding_seconds": node_encoding_seconds,
            "graph_construction_seconds": graph_construction_seconds,
            "message_passing_seconds": time.perf_counter() - message_started,
        })
        return isolated_delta, graph_delta, edge_features, source, target, stats

    @staticmethod
    def _graph_statistics(gates, edge_features, message, contextualized, node_count):
        def selected_mean(values, mask):
            return float(values[mask].detach().float().mean().item()) if mask.any() else 0.0

        if gates.numel() == 0:
            contextual_norm = (
                float(contextualized.detach().float().norm(dim=1).mean().item())
                if contextualized.shape[0]
                else 0.0
            )
            return {
                "nodes": float(node_count),
                "edges": 0.0,
                "gate_mean": 0.0,
                "gate_std": 0.0,
                "gate_min": 0.0,
                "gate_max": 0.0,
                "touch_gate_mean": 0.0,
                "nontouch_gate_mean": 0.0,
                "same_class_gate_mean": 0.0,
                "different_class_gate_mean": 0.0,
                "top_gate_feature_similarity": 0.0,
                "top_gate_spatial_distance": 0.0,
                "message_norm": 0.0,
                "contextual_token_norm": contextual_norm,
                "multi_node": 0.0,
                "multi_node_message_nonzero": 0.0,
            }
        touch = edge_features[:, 2] > 0.5
        same = edge_features[:, 3] > 0.5
        top_count = max(1, int(np.ceil(gates.numel() * 0.25)))
        top = torch.topk(gates.detach().float(), k=top_count, largest=True).indices
        message_norm = message.detach().float().norm(dim=1).mean()
        return {
            "nodes": float(node_count),
            "edges": float(gates.numel()),
            "gate_mean": float(gates.detach().float().mean().item()),
            "gate_std": float(gates.detach().float().std(unbiased=False).item()),
            "gate_min": float(gates.detach().float().min().item()),
            "gate_max": float(gates.detach().float().max().item()),
            "touch_gate_mean": selected_mean(gates, touch),
            "nontouch_gate_mean": selected_mean(gates, ~touch),
            "same_class_gate_mean": selected_mean(gates, same),
            "different_class_gate_mean": selected_mean(gates, ~same),
            "top_gate_feature_similarity": float(edge_features[top, 0].detach().float().mean().item()),
            "top_gate_spatial_distance": float(edge_features[top, 1].detach().float().mean().item()),
            "message_norm": float(message_norm.item()),
            "contextual_token_norm": float(contextualized.detach().float().norm(dim=1).mean().item()),
            "multi_node": 1.0,
            "multi_node_message_nonzero": float(message_norm.item() > 0.0),
        }

    def forward(
        self,
        feature: torch.Tensor,
        cam_56: torch.Tensor,
        cam_28_1: torch.Tensor,
        cam_28_2: torch.Tensor,
        cam_deep: torch.Tensor,
        presence: torch.Tensor,
        collect_structures: bool = False,
    ) -> RGRResult:
        if feature.shape[1] != self.feature_channels:
            raise ValueError(f"Expected {self.feature_channels} H28_1 channels")
        if presence.shape != (feature.shape[0], self.n_class):
            raise ValueError("presence must have shape [batch, 4]")

        proposal = self.proposal_labels(
            cam_56, cam_28_1, cam_28_2, cam_deep, presence
        ).detach().cpu().numpy().astype(np.uint8)
        batch, _, height, width = cam_28_1.shape
        iso_images, graph_images, region_logits = [], [], []
        structures = [] if collect_structures else None
        per_image_region_counts = []
        graph_statistics = []
        total_components = total_regions = total_tiny = total_core = total_transition = 0
        token_norm_sum = 0.0

        for batch_index in range(batch):
            region_started = time.perf_counter()
            feature_flat = feature[batch_index].reshape(self.feature_channels, -1)
            cam_flat = cam_28_1[batch_index].reshape(self.n_class, -1)
            records = _component_records(proposal[batch_index])
            total_components += len(records)
            tokens, base_logits, cores, masks, centroids = [], [], [], [], []
            image_structures = []

            for predicted_class, component_id, indices_np in records:
                area = int(len(indices_np))
                if area < MIN_REGION_AREA:
                    total_tiny += 1
                    continue
                indices = torch.as_tensor(indices_np, device=feature.device, dtype=torch.long)
                pixels = feature_flat.index_select(1, indices)
                token = pixels.mean(dim=1)
                core_np, transition_np = _core_indices(
                    indices_np, pixels, token, height, width
                )
                mask = np.zeros((height, width), dtype=np.uint8)
                mask.reshape(-1)[indices_np] = 1
                ys, xs = np.unravel_index(indices_np, (height, width))
                tokens.append(token)
                base_logits.append(cam_flat.index_select(1, indices).mean(dim=1))
                cores.append(torch.as_tensor(core_np, device=feature.device, dtype=torch.long))
                masks.append(mask)
                centroids.append((float(np.mean(ys)), float(np.mean(xs))))
                total_regions += 1
                total_core += int(len(core_np))
                total_transition += int(len(transition_np))
                token_norm_sum += float(token.detach().float().norm().item())
                if collect_structures:
                    image_structures.append((
                        int(predicted_class),
                        int(component_id),
                        tuple(int(item) for item in indices_np),
                        tuple(int(item) for item in core_np),
                        tuple(int(item) for item in transition_np),
                    ))

            iso_flat = cam_28_1.new_zeros((self.n_class, height * width))
            graph_flat = cam_28_1.new_zeros((self.n_class, height * width))
            node_count = len(tokens)
            per_image_region_counts.append(node_count)
            region_extraction_seconds = time.perf_counter() - region_started
            if node_count:
                token_tensor = torch.stack(tokens)
                base_tensor = torch.stack(base_logits)
                centroid_tensor = token_tensor.new_tensor(centroids)
                iso_delta, graph_delta, edge_features, source, target, stats = (
                    self._graph_reasoning(
                        token_tensor, base_tensor, masks, centroid_tensor, height, width
                    )
                )
                region_logits.append(base_tensor + iso_delta + graph_delta)
                for region_index, target_indices in enumerate(cores):
                    if target_indices.numel() == 0:
                        continue
                    iso_flat = iso_flat.index_copy(
                        1,
                        target_indices,
                        iso_delta[region_index][:, None].expand(-1, target_indices.numel()),
                    )
                    graph_flat = graph_flat.index_copy(
                        1,
                        target_indices,
                        graph_delta[region_index][:, None].expand(-1, target_indices.numel()),
                    )
                if collect_structures:
                    edges = tuple(
                        (int(target[k].item()), int(source[k].item()))
                        for k in range(source.numel())
                    )
                    image_structures.append(("edges", edges))
            else:
                region_logits.append(cam_28_1.new_empty((0, self.n_class)))
                stats = self._graph_statistics(
                    cam_28_1.new_empty((0,)),
                    cam_28_1.new_empty((0, 4)),
                    cam_28_1.new_empty((0, self.latent_channels)),
                    cam_28_1.new_empty((0, self.latent_channels)),
                    0,
                )
                stats.update({
                    "node_encoding_seconds": 0.0,
                    "graph_construction_seconds": 0.0,
                    "message_passing_seconds": 0.0,
                })

            stats["region_extraction_seconds"] = region_extraction_seconds

            graph_statistics.append(stats)
            iso_images.append(iso_flat.reshape(self.n_class, height, width))
            graph_images.append(graph_flat.reshape(self.n_class, height, width))
            if collect_structures:
                structures.append(tuple(image_structures))

        delta_iso = torch.stack(iso_images)
        delta_graph = torch.stack(graph_images)
        residual = delta_iso + delta_graph
        refined = cam_28_1 + residual
        pixels = float(batch * height * width)
        base_rms = cam_28_1.detach().float().square().mean().sqrt()
        residual_rms = residual.detach().float().square().mean().sqrt()

        def mean_stat(name):
            return float(np.mean([row[name] for row in graph_statistics]))

        statistics = {
            "components_per_image": total_components / float(batch),
            "regions_per_image": total_regions / float(batch),
            "edges_per_image": mean_stat("edges"),
            "core_fraction": total_core / pixels,
            "transition_fraction": total_transition / pixels,
            "tiny_region_fraction": total_tiny / float(max(total_components, 1)),
            "mean_region_token_norm": token_norm_sum / float(max(total_regions, 1)),
            "gate_mean": mean_stat("gate_mean"),
            "gate_std": mean_stat("gate_std"),
            "gate_min": min(row["gate_min"] for row in graph_statistics),
            "gate_max": max(row["gate_max"] for row in graph_statistics),
            "touch_gate_mean": mean_stat("touch_gate_mean"),
            "nontouch_gate_mean": mean_stat("nontouch_gate_mean"),
            "same_class_gate_mean": mean_stat("same_class_gate_mean"),
            "different_class_gate_mean": mean_stat("different_class_gate_mean"),
            "top_gate_feature_similarity": mean_stat("top_gate_feature_similarity"),
            "top_gate_spatial_distance": mean_stat("top_gate_spatial_distance"),
            "message_norm": mean_stat("message_norm"),
            "contextual_token_norm": mean_stat("contextual_token_norm"),
            "region_extraction_seconds": mean_stat("region_extraction_seconds"),
            "node_encoding_seconds": mean_stat("node_encoding_seconds"),
            "graph_construction_seconds": mean_stat("graph_construction_seconds"),
            "message_passing_seconds": mean_stat("message_passing_seconds"),
            "multi_node_fraction": mean_stat("multi_node"),
            "multi_node_message_nonzero_fraction": mean_stat("multi_node_message_nonzero"),
            "rms_delta_iso": float(delta_iso.detach().float().square().mean().sqrt().item()),
            "rms_delta_graph": float(delta_graph.detach().float().square().mean().sqrt().item()),
            "graph_to_isolated_rms": float(
                (
                    delta_graph.detach().float().square().mean().sqrt()
                    / (delta_iso.detach().float().square().mean().sqrt() + 1e-8)
                ).item()
            ),
            "max_abs_delta_iso": float(delta_iso.detach().float().abs().max().item()),
            "max_abs_delta_graph": float(delta_graph.detach().float().abs().max().item()),
            "residual_ratio": float((residual_rms / (base_rms + 1e-8)).item()),
            "mean_abs_residual": float(residual.detach().float().abs().mean().item()),
        }
        return RGRResult(
            refined_cam=refined,
            delta_iso=delta_iso,
            delta_graph=delta_graph,
            region_logits=region_logits,
            statistics=statistics,
            structures=structures,
            per_image_region_counts=per_image_region_counts,
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
