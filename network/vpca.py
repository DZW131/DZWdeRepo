"""Deterministic visually grounded pathology-concept participation."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class VPCAConceptProvider(nn.Module):
    def __init__(self, classes:int=4, concepts_per_class:int=8, topk_ratio:float=.20, tau:float=.1):
        super().__init__(); self.classes=classes; self.m=concepts_per_class; self.topk_ratio=topk_ratio; self.tau=tau

    def forward(self, final_dense:torch.Tensor, concepts:torch.Tensor):
        # final_dense: B,D,H,W; concepts: 32,D
        u=F.normalize(final_dense.float(),dim=1); t=F.normalize(concepts.float(),dim=-1)
        evidence=torch.einsum("bdhw,kd->bkhw",u,t)
        flat=evidence.flatten(2); k=max(1,int(flat.shape[-1]*self.topk_ratio))
        g=flat.topk(k,dim=-1).values.mean(-1).reshape(-1,self.classes,self.m)
        q=F.softmax(g/self.tau,dim=-1); class_support=torch.logsumexp(g,dim=-1); rho=F.softmax(class_support,dim=-1)
        omega=(rho[...,None]*q).reshape(-1,self.classes*self.m)
        return omega,{"q":q,"rho":rho,"omega":omega,"concept_evidence":evidence,"support":g,
                      "concept_entropy":-(q.clamp_min(1e-8)*q.clamp_min(1e-8).log()).sum(-1)}


class StaticConceptProvider(nn.Module):
    def __init__(self, total:int=32): super().__init__(); self.total=total
    def forward(self,batch:int,device): return torch.full((batch,self.total),1/self.total,device=device,dtype=torch.float32),{}

