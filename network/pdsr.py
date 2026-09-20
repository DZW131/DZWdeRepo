"""Pathology Dense Semantic Reconstruction with fixed three-layer JS consensus."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class PathologyDenseSemanticReconstructor(nn.Module):
    def __init__(self,vision_dim:int,semantic_dim:int=512,output_dim:int=256,tau_semantic:float=.1,tau_layer:float=.1):
        super().__init__(); self.tau_semantic=tau_semantic; self.tau_layer=tau_layer
        self.projections=nn.ModuleList([nn.Linear(vision_dim,semantic_dim) for _ in range(3)])
        self.semantic_projection=nn.Conv2d(semantic_dim,output_dim,1)

    @staticmethod
    def grid(tokens:torch.Tensor)->torch.Tensor:
        side=math.isqrt(tokens.shape[1])
        if side*side!=tokens.shape[1]: raise AssertionError("PDSR token grid is not square")
        return tokens.transpose(1,2).reshape(tokens.shape[0],tokens.shape[2],side,side)

    def forward(self,vision_hidden_states:tuple[torch.Tensor,...],concept_embeddings:torch.Tensor,concept_weights:torch.Tensor):
        concepts=F.normalize(concept_embeddings.float(),dim=-1); us=[]; pis=[]; recon=[]; evidence=[]
        for tokens,projection in zip(vision_hidden_states,self.projections):
            u=self.grid(projection(tokens.float())); us.append(u)
            score=torch.einsum("bdhw,kd->bkhw",F.normalize(u,dim=1),concepts)
            adjusted=score+concept_weights.float().clamp_min(1e-8).log()[...,None,None]
            pi=F.softmax(adjusted/self.tau_semantic,dim=1); r=torch.einsum("bkhw,kd->bdhw",pi,concepts)
            evidence.append(score); pis.append(pi); recon.append(r)
        mean_pi=torch.stack(pis).mean(0).clamp_min(1e-8)
        divergences=torch.stack([(pi.clamp_min(1e-8)*(pi.clamp_min(1e-8).log()-mean_pi.log())).sum(1) for pi in pis],1)
        beta=F.softmax(-divergences/self.tau_layer,dim=1)
        z_layers=[u+r for u,r in zip(us,recon)]
        z_sem=sum(beta[:,i:i+1]*z for i,z in enumerate(z_layers))
        residual=F.interpolate(self.semantic_projection(z_sem),size=(28,28),mode="bilinear",align_corners=False)
        semantic_evidence=torch.einsum("bdhw,kd->bkhw",F.normalize(z_sem,dim=1),concepts)
        flat=semantic_evidence.flatten(2); topk=max(1,int(flat.shape[-1]*.20)); concept_scores=flat.topk(topk,dim=-1).values.mean(-1)
        layer_class_evidence=torch.stack([pi.reshape(z_sem.shape[0],4,8,*z_sem.shape[-2:]).sum(2) for pi in pis],1)
        class_evidence=(sum(beta[:,i:i+1]*pis[i] for i in range(3))).reshape(z_sem.shape[0],4,8,*z_sem.shape[-2:]).sum(2)
        aux={"concept_distributions":torch.stack(pis,1),"layer_js":divergences,"layer_weights":beta,
             "semantic_reconstruction":torch.stack(recon,1),"visual_projections":torch.stack(us,1),
             "semantic_embedding":z_sem,"semantic_evidence":semantic_evidence,"class_semantic_evidence":class_evidence,
             "layer_class_evidence":layer_class_evidence,
             "concept_scores":concept_scores,
             "reconstruction_norm":torch.stack([x.float().norm(dim=1).mean((1,2)) for x in recon],1),
             "visual_norm":torch.stack([x.float().norm(dim=1).mean((1,2)) for x in us],1)}
        return residual,aux
