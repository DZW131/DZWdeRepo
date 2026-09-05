"""Detached PCRE-style Progressive Mask Expansion and Combination diagnostics."""
from __future__ import annotations

import torch


TAU_BIN = 0.70
TAU_LOW = 0.40
TAU_HIGH = 0.50
MAX_REFERENCE_GROUPS = 5


def reference_overlap_ratio(candidates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """PCRE ROR: intersection(candidate, reference) / area(reference)."""
    intersection = (candidates & reference[None]).flatten(1).sum(dim=1).float()
    denominator = reference.sum().float().clamp_min(1.0)
    return intersection / denominator


@torch.no_grad()
def pmec(mask_logits: torch.Tensor, joint: torch.Tensor, p_class: torch.Tensor,
         present: torch.Tensor, locality: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
    probability = mask_logits.float().sigmoid() * locality[None].float()
    binary = probability >= TAU_BIN
    assigned = p_class.argmax(dim=-1)
    batch, _, height, width = probability.shape
    regions = torch.zeros((batch, 4, height, width), device=probability.device)
    rows: list[dict] = []
    for image in range(batch):
        for cls in range(4):
            if not bool(present[image, cls]):
                continue
            candidates = torch.where((assigned[image] == cls) & binary[image].flatten(1).any(dim=1))[0]
            if candidates.numel():
                confidence = joint[image, candidates, cls]
                candidates = candidates[torch.argsort(confidence, descending=True, stable=True)]
            candidate_count = int(candidates.numel())
            remaining = candidates
            selected_count = redundancy = groups = 0
            reference_count = 0
            top1 = binary[image, candidates[0]].clone() if candidate_count else torch.zeros((height, width), dtype=torch.bool, device=probability.device)
            combined = torch.zeros_like(top1)
            while remaining.numel() and groups < MAX_REFERENCE_GROUPS:
                reference_index = remaining[0]
                reference = binary[image, reference_index]
                ratios = reference_overlap_ratio(binary[image, remaining], reference)
                expansion = (ratios >= TAU_LOW) & (ratios <= TAU_HIGH)
                redundant = ratios > TAU_HIGH
                members = torch.cat((reference_index[None], remaining[expansion]))
                members = torch.unique(members, sorted=False)
                combined |= binary[image, members].any(dim=0)
                selected_count += int(members.numel())
                redundancy += int(redundant.sum()) - 1  # reference itself has ROR=1
                groups += 1
                reference_count += 1
                remaining = remaining[ratios < TAU_LOW]
            regions[image, cls] = probability[image, candidates].amax(dim=0) * combined if candidate_count else 0
            rows.append({
                "image": image,
                "class": cls,
                "candidate_masks": candidate_count,
                "selected_masks": selected_count,
                "region_groups": groups,
                "selected_region_area": float(combined.float().mean()),
                "redundancy_rejections": redundancy,
                "new_references": reference_count,
                "differs_from_top1": float(bool((combined != top1).any())),
            })
    return regions, rows
