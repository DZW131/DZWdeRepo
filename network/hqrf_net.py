"""HQRF-Net Phase-0 query-mask health-gate model."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from network.hqrf_backbone import HQRFBackbone
from network.hqrf_pmec import pmec
from network.hqrf_query import (
    CHPF,
    DualConfidenceAllocator,
    MaskEmbedding,
    PatchQueries,
    PixelDecoder,
    Projection,
    QueryDecoderLayer,
)
from network.hqrf_targets import circular_locality, masked_query_bce, normalize_cam, radius_at_step, tri_state_targets


class HQRFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = HQRFBackbone()
        self.deep_head = nn.Conv2d(4096, 4, 1, bias=False)
        nn.init.xavier_uniform_(self.deep_head.weight)
        self.patch_queries = PatchQueries()
        self.semantic_projection = Projection(4096, 256)
        self.decoder1 = QueryDecoderLayer()
        self.context_projection = Projection(1024, 256)
        self.f5_chpf = CHPF(256)
        self.decoder2 = QueryDecoderLayer()
        self.pixel_decoder = PixelDecoder()
        self.mask_embedding = MaskEmbedding()
        self.pca = DualConfidenceAllocator()

    def query_decode(self, image, fd, f5):
        """Only detach CNN memories; their trainable projections remain live."""
        query0 = self.patch_queries(image)
        semantic_feature = self.semantic_projection(fd.detach())
        semantic_memory = semantic_feature.flatten(2).transpose(1, 2)
        query1, attention1 = self.decoder1(query0, semantic_memory)
        context_raw = self.context_projection(f5.detach())
        context_feature = self.f5_chpf(context_raw)
        context_memory = context_feature.flatten(2).transpose(1, 2)
        query2, attention2 = self.decoder2(query1, context_memory)
        return query2, {
            "Q0": query0,
            "Q1": query1,
            "semantic_feature": semantic_feature,
            "context_raw": context_raw,
            "context_feature": context_feature,
            "decoder1_attention": attention1,
            "decoder2_attention": attention2,
        }

    def forward(self, image, labels, step=0, run_pmec=False):
        if labels is None:
            raise ValueError("Phase-0 requires train-split image labels")
        features = self.backbone(image)
        deep_cam_logits = self.deep_head(features["FD"])
        deep_logits = F.adaptive_avg_pool2d(deep_cam_logits, 1).flatten(1)
        normalized_cam = normalize_cam(deep_cam_logits)
        targets, target_detail = tri_state_targets(normalized_cam.detach(), labels.bool())
        query, query_detail = self.query_decode(image, features["FD"], features["F5"])
        pixel_feature, pixel_detail = self.pixel_decoder(features["F4"], features["F3"])
        mask_embedding = self.mask_embedding(query)
        mask_logits = torch.einsum("bqd,bdhw->bqhw", mask_embedding, pixel_feature)
        confidence = self.pca(query)
        radius = radius_at_step(step)
        locality = circular_locality(14, mask_logits.shape[-2:], radius, mask_logits.device)
        mask_loss, mask_detail = masked_query_bce(
            mask_logits, targets, labels.bool(), confidence["p_class"], confidence["joint"], locality
        )
        with torch.autocast(device_type=image.device.type, enabled=False):
            loss_deep = F.multilabel_soft_margin_loss(deep_logits.float(), labels.float())
            loss_pca = F.binary_cross_entropy(confidence["image_probability"], labels.float())
            total = 0.50 * loss_deep + 0.25 * loss_pca + 0.25 * mask_loss.float()
        output = {
            "features": features,
            "deep_cam_logits": deep_cam_logits,
            "deep_logits": deep_logits,
            "normalized_cam": normalized_cam,
            "targets": targets,
            "target_detail": target_detail,
            "query": query,
            "query_detail": query_detail,
            "pixel_feature": pixel_feature,
            "pixel_detail": pixel_detail,
            "mask_embedding": mask_embedding,
            "mask_logits": mask_logits,
            "confidence": confidence,
            "radius": radius,
            "locality": locality,
            "mask_detail": mask_detail,
            "losses": {"loss": total, "loss_deep": loss_deep, "loss_pca": loss_pca, "loss_mask": mask_loss},
        }
        if run_pmec:
            region, rows = pmec(mask_logits.detach(), confidence["joint"].detach(), confidence["p_class"].detach(), labels.bool(), locality)
            output["pmec_region"] = region
            output["pmec_rows"] = rows
        return output

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            scratch = not name.startswith("backbone.")
            bias = name.endswith("bias")
            groups[2 * int(scratch) + int(bias)].append(parameter)
        if len({id(value) for group in groups for value in group}) != sum(len(group) for group in groups):
            raise AssertionError("Duplicate optimizer parameter")
        return groups
