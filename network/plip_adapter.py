"""Frozen local PLIP adapter with explicit dense-token and preprocessing audits."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class FrozenPLIPAdapter(nn.Module):
    BLOCKS = (4, 8, 12)

    def __init__(self, model_path: str | Path):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizerFast
        path = str(Path(model_path).resolve())
        self.model = CLIPModel.from_pretrained(path, local_files_only=True)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(path, local_files_only=True)
        self.model.eval()
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        v = self.model.config.vision_config
        self.hidden_size = int(v.hidden_size); self.projection_dim = int(self.model.config.projection_dim)
        self.image_size = int(v.image_size); self.patch_size = int(v.patch_size); self.layers = int(v.num_hidden_layers)
        self.register_buffer("image_mean", torch.tensor([.48145466,.4578275,.40821073])[None,:,None,None], persistent=False)
        self.register_buffer("image_std", torch.tensor([.26862954,.26130258,.27577711])[None,:,None,None], persistent=False)

    def train(self, mode: bool=True):
        super().train(False); self.model.eval(); return self

    def preprocess(self, raw_rgb: torch.Tensor) -> torch.Tensor:
        if raw_rgb.ndim != 4 or raw_rgb.shape[1] != 3: raise ValueError("PLIP expects BCHW RGB")
        x = F.interpolate(raw_rgb, (self.image_size,self.image_size), mode="bicubic", align_corners=False)
        return (x-self.image_mean)/self.image_std

    @torch.no_grad()
    def dense_tokens(self, raw_rgb: torch.Tensor) -> tuple[torch.Tensor,...]:
        output = self.model.vision_model(pixel_values=self.preprocess(raw_rgb), output_hidden_states=True, return_dict=True)
        states = output.hidden_states
        if len(states) != self.layers + 1: raise AssertionError(f"PLIP hidden-state ordering changed: {len(states)}")
        chosen=[]
        for block in self.BLOCKS:
            value=states[block]
            patch=value[:,1:]
            side=math.isqrt(patch.shape[1])
            if side*side != patch.shape[1]: raise AssertionError(f"Non-square PLIP dense tokens: {patch.shape}")
            chosen.append(patch.detach())
        return tuple(chosen)

    @torch.no_grad()
    def encode_text(self, concepts: list[str], device: torch.device) -> torch.Tensor:
        tokens=self.tokenizer(concepts,padding=True,truncation=True,return_tensors="pt")
        tokens={k:v.to(device) for k,v in tokens.items()}
        embedding=self.model.get_text_features(**tokens).float()
        return F.normalize(embedding,dim=-1)

    @torch.no_grad()
    def runtime_audit(self, raw_rgb: torch.Tensor) -> dict:
        pixels=self.preprocess(raw_rgb)
        output=self.model.vision_model(pixel_values=pixels,output_hidden_states=True,return_dict=True)
        states=output.hidden_states; dense=self.dense_tokens(raw_rgb)
        return {"model_type":self.model.config.model_type,"hidden_states_length":len(states),
                "hidden_state_shapes":[list(x.shape) for x in states],"embedding_input_at_index_0":len(states)==self.layers+1,
                "block_to_hidden_state_index":{str(x):x for x in self.BLOCKS},"selected_dense_shapes":[list(x.shape) for x in dense],
                "cls_token_index":0,"patch_token_count":dense[-1].shape[1],"patch_grid_side":math.isqrt(dense[-1].shape[1]),
                "vision_hidden_size":self.hidden_size,"vision_projection_input":int(self.model.visual_projection.in_features),
                "vision_projection_output":int(self.model.visual_projection.out_features),"text_projection_dim":self.projection_dim,
                "image_size":self.image_size,"patch_size":self.patch_size,"num_layers":self.layers,
                "dense_grid_safe":all(math.isqrt(x.shape[1])**2==x.shape[1] for x in dense)}
