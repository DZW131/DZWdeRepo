"""Frozen HQMR/PLIP with trainable identity-initialized P1/P2/P3 adapters."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from network.hqmr_net import HQMRNet
from network.pdsr import PathologyDenseSemanticReconstructor
from network.plip_adapter import FrozenPLIPAdapter
from network.vpca import StaticConceptProvider,VPCAConceptProvider


class PDSRVPCAHQMR(nn.Module):
    MODES=("P1","P2","P3")
    def __init__(self,base_checkpoint:str|Path,plip_path:str|Path,concept_embeddings:torch.Tensor,mode:str):
        super().__init__()
        if mode not in self.MODES: raise ValueError(mode)
        self.mode=mode; self.base=HQMRNet(); state=torch.load(base_checkpoint,map_location="cpu",weights_only=False)
        if isinstance(state,dict) and "state_dict" in state: state=state["state_dict"]
        if any(k.startswith("module.") for k in state): state={k.removeprefix("module."):v for k,v in state.items()}
        self.base.load_state_dict(state,strict=True)
        for p in self.base.parameters(): p.requires_grad_(False)
        self.plip=FrozenPLIPAdapter(plip_path)
        self.register_buffer("concept_embeddings",F.normalize(concept_embeddings.float(),dim=-1))
        self.gamma=nn.Parameter(torch.zeros(256))
        if mode=="P1":
            self.last_projection=nn.Linear(self.plip.hidden_size,256); self.pdsr=None; self.provider=None
        else:
            self.last_projection=None
            self.pdsr=PathologyDenseSemanticReconstructor(self.plip.hidden_size)
            self.provider=StaticConceptProvider(32) if mode=="P2" else VPCAConceptProvider()
        self.hqmr_mean=torch.tensor([.485,.456,.406])[None,:,None,None]
        self.hqmr_std=torch.tensor([.229,.224,.225])[None,:,None,None]
        self.eval_frozen()

    def eval_frozen(self): self.base.eval(); self.plip.eval()
    def train(self,mode:bool=True):
        super().train(mode); self.eval_frozen(); return self
    def trainable_state_dict(self):
        prefixes=("gamma","last_projection","pdsr")
        return {k:v.detach().cpu() for k,v in self.state_dict().items() if k.startswith(prefixes)}
    def load_trainable_state_dict(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected: raise AssertionError(unexpected)
        return missing

    def forward(self,raw_rgb:torch.Tensor,labels:torch.Tensor):
        with torch.no_grad(): hidden=self.plip.dense_tokens(raw_rgb)
        provider_aux={}; pdsr_aux={}
        if self.mode=="P1":
            z=self.pdsr_grid(self.last_projection(hidden[-1].float()))
            semantic=F.interpolate(z,size=(28,28),mode="bilinear",align_corners=False)
            semantic_loss=semantic.new_zeros(())
        else:
            if self.mode=="P2": weights,provider_aux=self.provider(raw_rgb.shape[0],raw_rgb.device)
            else:
                u12=self.pdsr.grid(self.pdsr.projections[2](hidden[-1].float()))
                weights,provider_aux=self.provider(u12,self.concept_embeddings)
            semantic,pdsr_aux=self.pdsr(hidden,self.concept_embeddings,weights)
            scores=pdsr_aux["concept_scores"].reshape(-1,4,8)
            log_probability=F.log_softmax(scores.reshape(-1,32)/.1,dim=-1).reshape(-1,4,8)
            if self.mode=="P2":
                positive=labels.float()[...,None]/(8*labels.float().sum(-1,keepdim=True).clamp_min(1)[...,None])
                concept_loss=-(positive*log_probability).sum(-1).sum(-1).mean()
                class_score=torch.logsumexp(scores,dim=-1)
            else:
                omega=provider_aux["omega"].reshape(-1,4,8)
                concept_loss=-(labels.float()[...,None]*omega*log_probability).sum(-1).sum(-1).mean()
                class_score=torch.logsumexp(provider_aux["q"].clamp_min(1e-8).log()+scores,dim=-1)
            semantic_cls=F.binary_cross_entropy_with_logits(class_score,labels.float())
            semantic_loss=concept_loss+semantic_cls
            pdsr_aux.update({"concept_loss":concept_loss,"semantic_classification_loss":semantic_cls,"concept_weights":weights})
        residual=self.gamma[None,:,None,None]*semantic
        mean=self.hqmr_mean.to(raw_rgb); std=self.hqmr_std.to(raw_rgb)
        base=self.base((raw_rgb-mean)/std,labels,step=29275,h5_residual=residual)
        total=base["losses"]["loss"]+.2*semantic_loss
        base["losses"].update({"loss_hqmr":base["losses"]["loss"],"loss_semantic":semantic_loss,"loss":total})
        base["phase0"]={"mode":self.mode,"semantic":semantic,"semantic_residual":residual,"gamma":self.gamma,
                        "pdsr":pdsr_aux,"provider":provider_aux}
        return base

    @staticmethod
    def pdsr_grid(tokens): return PathologyDenseSemanticReconstructor.grid(tokens)

