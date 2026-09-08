"""Detached references for GCQM diagnostics."""
from __future__ import annotations

import torch


@torch.no_grad()
def materialize(stage,permutations):
    g=stage["gcqm"]; w=g["weights"].detach().float(); b=g["base_probability"].detach().float()
    primary=g["mixture"].detach().float(); perms=[]
    for values in permutations:
        index=torch.as_tensor(values,device=w.device,dtype=torch.long)
        perms.append(torch.einsum("bqc,bqhw->bchw",w[:,index],b))
    result={"primary":primary,"perm":torch.stack(perms),"pca":g["pca_reference"].detach().float(),
            "pixel":g["pixel_reference"].detach().float(),
            "uniform":b.mean(1,keepdim=True).expand(-1,primary.shape[1],-1,-1),"weights":w}
    if not all(not x.requires_grad for x in result.values()): raise RuntimeError("GCQM reference gradient leakage")
    if not torch.equal(primary,g["mixture"].detach().float()): raise RuntimeError("GCQM primary changed in diagnostic path")
    if not all(bool(torch.isfinite(x).all()) for x in result.values()): raise FloatingPointError("Nonfinite GCQM reference")
    return result


__all__=["materialize"]
