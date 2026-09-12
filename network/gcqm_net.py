"""Frozen CQRF backbone with GCQM Stage2/3 primary masks."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from network.cqrf_net import CQRFNet, STAGE_WEIGHTS
from network.cqrf_query import sine_position_2d
from network.gcqm import gcqm_decode, gcqm_weights
from network.hqrf_pmec import pmec
from network.hqrf_targets import circular_locality, masked_query_bce, normalize_cam, radius_at_step, tri_state_targets
from network.momd import mixture_class_bce, pca_reference_envelope


class GCQMNet(CQRFNet):
    def forward(self,image,labels,step=0,run_pmec=False,gcqm_weights_only=False):
        if labels is None: raise ValueError("GCQM Phase-0 requires train-split image labels")
        if run_pmec and gcqm_weights_only: raise ValueError("PMEC requires materialized legacy mask logits")
        features=self.backbone(image); deep_cam_logits=self.deep_head(features["FD"])
        deep_logits=F.adaptive_avg_pool2d(deep_cam_logits,1).flatten(1); deep_gate=deep_logits.float().sigmoid()
        normalized_cam=normalize_cam(deep_cam_logits); targets,target_detail=tri_state_targets(normalized_cam.detach(),labels.bool())
        content0,base=self.patch_queries(image); semantic_feature=self.semantic_projection(features["FD"].detach())
        kp1=sine_position_2d(image.shape[0],*semantic_feature.shape[-2:],256,image.device,semantic_feature.dtype)
        query1,attention1=self.decoder1(content0+base,self._memory(semantic_feature)+kp1); confidence1=self.pca_heads[0](query1)
        context_raw=self.context_projection(features["F5"].detach()); context_feature=self.f5_chpf(context_raw)
        kp2=sine_position_2d(image.shape[0],*context_feature.shape[-2:],256,image.device,context_feature.dtype)
        query2,ccra2=self.ccra2(query1,base,self._memory(context_feature),kp2,confidence1["p_class"],deep_gate); confidence2=self.pca_heads[1](query2)
        pixel_feature,pixel_detail=self.pixel_decoder(features["F4"],features["F3"]); f4_memory=self.f4_memory_projection(pixel_detail["F4_context"].detach())
        kp3=sine_position_2d(image.shape[0],*f4_memory.shape[-2:],256,image.device,f4_memory.dtype)
        query3,ccra3=self.ccra3(query2,base,self._memory(f4_memory),kp3,confidence2["p_class"],deep_gate); confidence3=self.pca_heads[2](query3)
        radius=radius_at_step(step); locality=circular_locality(14,pixel_feature.shape[-2:],radius,pixel_feature.device)
        inputs=((query1,confidence1,{"cross_attention":attention1["cross_attention"]},semantic_feature.shape[-2:]),(query2,confidence2,ccra2,context_feature.shape[-2:]),(query3,confidence3,ccra3,f4_memory.shape[-2:]))
        stages=[]
        for stage_index,(query,confidence,detail,memory_hw) in enumerate(inputs,1):
            if gcqm_weights_only and stage_index>=2:
                embedding=base_logits=None
            else:
                embedding=self.mask_embeddings[stage_index-1](query); base_logits=torch.einsum("bqd,bdhw->bqhw",embedding,pixel_feature)
            stage={"query":query,"mask_embedding":embedding,"base_mask_logits":base_logits,"mask_logits":base_logits,
                   "confidence":confidence,"detail":detail,"memory_hw":memory_hw,"gcqm":None}
            if stage_index>=2:
                if gcqm_weights_only:
                    stage["gcqm"]=gcqm_weights(detail["responsibility_class"],memory_hw,pixel_feature.shape[-2:],locality)
                else:
                    stage["gcqm"]=gcqm_decode(base_logits,detail["responsibility_class"],memory_hw,locality,materialize=run_pmec)
                if run_pmec and not gcqm_weights_only:
                    stage["gcqm"]["pca_reference"]=pca_reference_envelope(stage["gcqm"]["base_probability"],confidence["joint"])
            stages.append(stage)
        mask_losses=[]; pca_losses=[]
        for stage_index,stage in enumerate(stages,1):
            if stage_index==1:
                mask_loss,detail=masked_query_bce(stage["base_mask_logits"],targets,labels.bool(),stage["confidence"]["p_class"],stage["confidence"]["joint"],locality)
            elif gcqm_weights_only:
                mask_loss,detail=query.sum()*0.,{"weights_only":True,"legacy_mask_loss_skipped":True}
            else:
                mask_loss,detail=mixture_class_bce(stage["gcqm"]["mixture"],targets,labels.bool())
            stage["mask_detail"]=detail; mask_losses.append(mask_loss.float())
            with torch.autocast(device_type=image.device.type,enabled=False):
                pca_losses.append(F.binary_cross_entropy(stage["confidence"]["image_probability"].float(),labels.float()))
        with torch.autocast(device_type=image.device.type,enabled=False):
            loss_deep=F.multilabel_soft_margin_loss(deep_logits.float(),labels.float())
            loss_pca=sum(w*v for w,v in zip(STAGE_WEIGHTS,pca_losses)); loss_mask=sum(w*v for w,v in zip(STAGE_WEIGHTS,mask_losses))
            total=.50*loss_deep+.25*loss_pca+.25*loss_mask
        output={"features":features,"deep_cam_logits":deep_cam_logits,"deep_logits":deep_logits,"deep_gate":deep_gate,
                "normalized_cam":normalized_cam,"targets":targets,"target_detail":target_detail,"stages":stages,
                "pixel_feature":pixel_feature,"pixel_detail":pixel_detail,"query_detail":{"B":base,"semantic_feature":semantic_feature,"context_raw":context_raw,"context_feature":context_feature,"f4_memory":f4_memory},
                "radius":radius,"locality":locality,"primary_output":stages[-1]["gcqm"]["mixture"],
                "losses":{"loss":total,"loss_deep":loss_deep,"loss_pca":loss_pca,"loss_mask":loss_mask,
                          **{f"loss_pca_stage{i+1}":v for i,v in enumerate(pca_losses)},**{f"loss_mask_stage{i+1}":v for i,v in enumerate(mask_losses)}}}
        if run_pmec:
            final=stages[-1]; region,rows=pmec(final["base_mask_logits"].detach(),final["confidence"]["joint"].detach(),final["confidence"]["p_class"].detach(),labels.bool(),locality)
            output["pmec_region"],output["pmec_rows"]=region,rows; output["pmec_diagnostic_only"]=True
        return output


__all__=["GCQMNet"]
