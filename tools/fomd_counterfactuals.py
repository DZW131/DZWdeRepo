"""Detached counterfactuals for FOMD factorization diagnostics.

The training graph remains the archived MOMD-v1 graph.  These helpers are
used only on monitor batches and deliberately detach every input/output.
"""
from __future__ import annotations

import hashlib
import json
import random

import torch


PERMUTATION_SEEDS = tuple(range(1001, 1009))


def fixed_derangement(size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    values = list(range(size))
    while True:
        rng.shuffle(values)
        if all(index != value for index, value in enumerate(values)):
            return values.copy()


def permutation_payload(size: int = 196) -> dict:
    permutations = [fixed_derangement(size, seed) for seed in PERMUTATION_SEEDS]
    canonical = json.dumps(permutations, separators=(",", ":"), ensure_ascii=True).encode()
    return {
        "query_count": size,
        "seeds": list(PERMUTATION_SEEDS),
        "permutations": permutations,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "all_derangements": all(all(i != v for i, v in enumerate(p)) for p in permutations),
    }


@torch.no_grad()
def materialize(stage: dict, permutations: list[list[int]]) -> dict:
    """Return F_full/perm/global/PCA/uniform without touching the main graph."""
    momd = stage["momd"]
    a = momd["routing"].detach().float()          # B,Q,C,H,W
    b = momd["base_probability"].detach().float() # B,Q,H,W
    full = momd["mixture"].detach().float()

    perm = []
    for values in permutations:
        index = torch.as_tensor(values, device=a.device, dtype=torch.long)
        perm.append((a[:, index] * b[:, :, None]).sum(1))

    global_a = a.mean((-2, -1), keepdim=True)
    global_a = global_a / global_a.sum(1, keepdim=True).clamp_min(1.0e-8)
    global_mix = (global_a * b[:, :, None]).sum(1)

    joint = stage["confidence"]["joint"].detach().float()
    alpha = joint / joint.sum(1, keepdim=True).clamp_min(1.0e-8)
    pca = (alpha[..., None, None] * b[:, :, None]).sum(1)
    uniform = b.mean(1, keepdim=True).expand(-1, full.shape[1], -1, -1)

    result = {
        "full": full,
        "perm": torch.stack(perm, 0),
        "global": global_mix,
        "pca": pca,
        "uniform": uniform,
        "global_A": global_a,
    }
    if not all(not value.requires_grad for value in result.values()):
        raise RuntimeError("FOMD counterfactual gradient leakage")
    if not torch.equal(full, momd["mixture"].detach().float()):
        raise RuntimeError("FOMD changed the archived MOMD full mixture")
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise FloatingPointError("Nonfinite FOMD counterfactual")
    return result


__all__ = ["PERMUTATION_SEEDS", "fixed_derangement", "permutation_payload", "materialize"]
