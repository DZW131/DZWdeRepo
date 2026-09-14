"""Cross-image multi-prototype region verification (CIRV).

The implementation is intentionally parameter-free.  Connected components,
region embeddings, prototype verification, and EMA memory updates are all
executed outside autograd so the frozen HQMR training path is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


EPS = 1.0e-8
N_CLASSES = 4
N_PROTOTYPES = 4
COSINE_SCALE = 5.0
EMA_MOMENTUM = 0.99
RATIO_MIN = 0.25
RATIO_MAX = 4.0
STRUCTURE8 = np.ones((3, 3), dtype=np.uint8)


def l2_normalize(value: np.ndarray, axis: int = -1) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=axis, keepdims=True), EPS)


def extract_regions(prediction: np.ndarray) -> list[dict]:
    """Return deterministic 8-connected foreground class regions."""
    prediction = np.asarray(prediction)
    if prediction.ndim != 2:
        raise ValueError("prediction must be HxW")
    rows: list[dict] = []
    for cls in range(N_CLASSES):
        labels, count = ndimage.label(prediction == cls, structure=STRUCTURE8)
        for component_id in range(1, count + 1):
            mask = labels == component_id
            rows.append({"class_id": cls, "component_id": component_id,
                         "area": int(mask.sum()), "mask": mask})
    return rows


@torch.no_grad()
def region_embedding(mask: np.ndarray | torch.Tensor, key4: torch.Tensor) -> torch.Tensor:
    """Soft-mask pool a stop-gradient H4 semantic key and L2 normalize it."""
    if key4.ndim == 4:
        if key4.shape[0] != 1:
            raise ValueError("region_embedding accepts one image at a time")
        key4 = key4[0]
    if key4.ndim != 3:
        raise ValueError("key4 must be DxHxW")
    region = torch.as_tensor(mask, device=key4.device, dtype=torch.float32)[None, None]
    soft = F.interpolate(region, size=key4.shape[-2:], mode="area")[0, 0]
    pooled = (key4.detach().float() * soft).sum((1, 2)) / soft.sum().clamp_min(EPS)
    return F.normalize(pooled, dim=0, eps=EPS).detach()


@torch.no_grad()
def select_source_regions(prediction: np.ndarray, image_label: np.ndarray,
                          anchors: np.ndarray, key4: torch.Tensor) -> list[dict]:
    """Select exactly one max-anchor region per present class and image."""
    image_label = np.asarray(image_label).astype(bool)
    anchors = np.asarray(anchors).astype(bool)
    if image_label.shape != (N_CLASSES,) or anchors.shape != (N_CLASSES, *prediction.shape):
        raise ValueError("source label/anchor shape mismatch")
    selected: list[dict] = []
    all_regions = extract_regions(prediction)
    for cls in range(N_CLASSES):
        if not image_label[cls]:
            continue
        candidates = []
        for region in all_regions:
            if region["class_id"] != cls:
                continue
            count = int(np.sum(region["mask"] & anchors[cls]))
            if count > 0:
                candidates.append((count, region))
        if not candidates:
            continue
        # Stable component id resolves equal anchor counts; area is never used.
        count, chosen = sorted(candidates, key=lambda item: (-item[0], item[1]["component_id"]))[0]
        selected.append({"class_id": cls, "component_id": chosen["component_id"],
                         "area": chosen["area"], "anchor_count": count, "mask": chosen["mask"]})
    if selected:
        feature = key4[0] if key4.ndim == 4 else key4
        masks = torch.as_tensor(np.stack([row.pop("mask") for row in selected]),
                                device=feature.device, dtype=torch.float32)[:, None]
        soft = F.interpolate(masks, size=feature.shape[-2:], mode="area")[:, 0]
        pooled = torch.einsum("nhw,dhw->nd", soft, feature.detach().float())
        pooled /= soft.sum((1, 2), keepdim=False)[:, None].clamp_min(EPS)
        embeddings = F.normalize(pooled, dim=1, eps=EPS).cpu().numpy()
        for row, embedding in zip(selected, embeddings):
            row["embedding"] = embedding
    return selected


def spherical_kmeans(values: np.ndarray, k: int = N_PROTOTYPES, seed: int = 42,
                     max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic spherical k-means with seeded farthest-first initialization."""
    x = l2_normalize(np.asarray(values, dtype=np.float32))
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("spherical_kmeans requires non-empty NxD input")
    rng = np.random.default_rng(seed)
    unique = min(k, len(x))
    first = int(rng.integers(len(x)))
    indices = [first]
    while len(indices) < unique:
        similarity = x @ x[indices].T
        distance = 1.0 - similarity.max(1)
        distance[np.asarray(indices)] = -np.inf
        indices.append(int(np.argmax(distance)))
    centers = x[indices].copy()
    assignment = np.zeros(len(x), dtype=np.int64)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        new_assignment = np.argmax(x @ centers.T, axis=1)
        if iterations > 1 and np.array_equal(new_assignment, assignment):
            break
        assignment = new_assignment
        updated = centers.copy()
        for cluster in range(unique):
            members = x[assignment == cluster]
            if len(members):
                updated[cluster] = l2_normalize(members.mean(0))
        centers = updated
    replicated = 0
    if unique < k:
        original = centers.copy()
        while len(centers) < k:
            centers = np.concatenate((centers, original[len(centers) % unique][None]), axis=0)
            replicated += 1
    final_assignment = np.argmax(x @ centers.T, axis=1)
    report = {"samples": int(len(x)), "k": int(k), "seed": int(seed),
              "iterations": int(iterations), "replicated_centers": int(replicated),
              "occupancy": np.bincount(final_assignment, minlength=k).astype(int).tolist()}
    return l2_normalize(centers), final_assignment, report


def prototype_posterior(embedding: np.ndarray, prototypes: np.ndarray,
                        g_prior: np.ndarray, scale: float = COSINE_SCALE) -> tuple[np.ndarray, np.ndarray]:
    z = l2_normalize(np.asarray(embedding, dtype=np.float32))
    bank = l2_normalize(np.asarray(prototypes, dtype=np.float32))
    prior = np.asarray(g_prior, dtype=np.float32)
    if bank.shape[:2] != (N_CLASSES, N_PROTOTYPES) or prior.shape != (N_CLASSES,):
        raise ValueError("prototype bank/prior shape mismatch")
    similarity = np.einsum("d,ckd->ck", z, bank).max(1)
    logits = float(scale) * similarity + np.log(np.maximum(prior, EPS))
    logits -= logits.max()
    posterior = np.exp(logits)
    posterior /= posterior.sum()
    return posterior.astype(np.float32), similarity.astype(np.float32)


def base_region_posterior(evidence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float32)
    pixel = (evidence + EPS) / (evidence.sum(0, keepdims=True) + N_CLASSES * EPS)
    posterior = pixel[:, np.asarray(mask).astype(bool)].mean(1)
    return (posterior / np.maximum(posterior.sum(), EPS)).astype(np.float32)


def product_fusion(base: np.ndarray, proto: np.ndarray) -> np.ndarray:
    result = np.asarray(base, dtype=np.float32) * np.asarray(proto, dtype=np.float32)
    return (result / np.maximum(result.sum(), EPS)).astype(np.float32)


def calibration_ratio(base: np.ndarray, proto: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(proto, dtype=np.float32) / (np.asarray(base, dtype=np.float32) + EPS)
    return np.clip(raw, RATIO_MIN, RATIO_MAX).astype(np.float32), raw


@torch.no_grad()
def calibrate_evidence(evidence: np.ndarray, prediction: np.ndarray, key4: torch.Tensor,
                       prototypes: np.ndarray, g_prior: np.ndarray,
                       mode: str = "full") -> tuple[np.ndarray, list[dict]]:
    """Calibrate all base regions without deletion or background creation."""
    base_evidence = np.asarray(evidence, dtype=np.float32)
    if mode == "off":
        return base_evidence.copy(), []
    bank = np.asarray(prototypes, dtype=np.float32)
    if mode == "single":
        mean = l2_normalize(bank.mean(1))
        bank = np.repeat(mean[:, None], N_PROTOTYPES, axis=1)
    calibrated = base_evidence.copy()
    records = []
    for region in extract_regions(prediction):
        z = region_embedding(region["mask"], key4).cpu().numpy()
        prior = np.ones(N_CLASSES, np.float32) if mode == "no_g" else g_prior
        proto, similarity = prototype_posterior(z, bank, prior)
        base = base_region_posterior(base_evidence, region["mask"])
        fused = proto if mode == "proto_only" else product_fusion(base, proto)
        # The registered pixel calibration uses the prototype posterior ratio.
        ratio, raw = calibration_ratio(base, proto)
        calibrated[:, region["mask"]] *= ratio[:, None]
        records.append({**{k: v for k, v in region.items() if k != "mask"},
                        "mask": region["mask"], "embedding": z, "p_base": base,
                        "p_proto": proto, "p_fuse": fused, "similarity": similarity,
                        "ratio": ratio, "ratio_raw": raw,
                        "base_class": int(np.argmax(base)), "proto_class": int(np.argmax(proto)),
                        "fused_class": int(np.argmax(fused))})
    calibrated /= calibrated.sum(0, keepdims=True) + EPS
    return calibrated.astype(np.float32), records


@dataclass
class PrototypeBank:
    prototypes: np.ndarray
    momentum: float = EMA_MOMENTUM

    def __post_init__(self):
        value = np.asarray(self.prototypes, dtype=np.float32)
        if value.ndim != 3 or value.shape[:2] != (N_CLASSES, N_PROTOTYPES):
            raise ValueError("bank must have shape 4x4xD")
        self.prototypes = l2_normalize(value)

    def verify(self, embedding: np.ndarray, class_id: int) -> tuple[int, np.ndarray]:
        similarity = self.prototypes[class_id] @ l2_normalize(embedding)
        return int(np.argmax(similarity)), similarity

    def update(self, embedding: np.ndarray, class_id: int) -> int:
        # Callers must verify before this non-gradient memory update.
        assignment, _ = self.verify(embedding, class_id)
        old = self.prototypes[class_id, assignment]
        self.prototypes[class_id, assignment] = l2_normalize(
            self.momentum * old + (1.0 - self.momentum) * l2_normalize(embedding))
        return assignment


__all__ = ["PrototypeBank", "base_region_posterior", "calibrate_evidence", "calibration_ratio",
           "extract_regions", "l2_normalize", "product_fusion", "prototype_posterior",
           "region_embedding", "select_source_regions", "spherical_kmeans"]
