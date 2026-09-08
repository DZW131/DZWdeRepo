"""Zero-parameter Global Class-Conditioned Query Mixture decoding."""
from __future__ import annotations

import torch

from network.momd import EPS, route_one_class


def gcqm_decode(base_logits: torch.Tensor, responsibility_class: torch.Tensor,
                memory_hw: tuple[int, int], locality: torch.Tensor,
                materialize: bool = False) -> dict:
    """Pool frozen MOMD A spatially into detached global w and decode sum(w*B)."""
    with torch.autocast(device_type=base_logits.device.type, enabled=False):
        base=base_logits.float().sigmoid()
        routes=[route_one_class(responsibility_class,memory_hw,base.shape[-2:],locality,cls)
                for cls in range(responsibility_class.shape[-1])]
        routing=torch.stack([x["routing"] for x in routes],dim=2)
        weights=routing.mean((-2,-1))
        weights=(weights/weights.sum(1,keepdim=True).clamp_min(EPS)).detach()
        final=torch.einsum("bqc,bqhw->bchw",weights,base).clamp(0,1)
        w_error=(weights.sum(1)-1).abs()
        payload={"base_probability":base,"weights":weights,"mixture":final,"primary_output":final,
                 "weight_sum_error_max":float(w_error.max()),"weight_sum_error_mean":float(w_error.mean()),
                 "routing_sum_error_max":max(x["sum_error_max"] for x in routes),
                 "routing_sum_error_mean":sum(x["sum_error_mean"] for x in routes)/len(routes),
                 "fallback_fraction":float(torch.stack([x["fallback"] for x in routes],1).float().mean()),
                 "weight_sidepath_detached":True}
        if materialize:
            contribution=weights[...,None,None]*base[:,:,None]
            error=(contribution.sum(1)-final).abs()
            pixel=(routing*base[:,:,None]).sum(1).detach()
            payload.update({"routing":routing.detach(),"contribution":contribution,
                            "contribution_sum_error_max":float(error.max().detach()),
                            "pixel_reference":pixel,
                            "responsibility_resized":torch.stack([x["responsibility"] for x in routes],dim=2),
                            "fallback":torch.stack([x["fallback"] for x in routes],dim=1)})
        if not torch.isfinite(final).all() or float(w_error.max())>1e-6:
            raise FloatingPointError("GCQM conservation/finite invariant failed")
    return payload


__all__=["gcqm_decode"]
