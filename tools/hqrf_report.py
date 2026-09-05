"""Render the fixed 28-section HQRF Phase-0 health-gate report."""
from pathlib import Path


def _fmt(value):
    return f"{100*float(value):.2f}%"


def render_report(output, result):
    output = Path(output); final = result.get("final_summary", {})
    area=final.get("mask_area",{}); logits=final.get("mask_logits",{}); diversity=final.get("query_diversity",{})
    selectivity=final.get("semantic_selectivity",{}); pca=final.get("pca_health",{}); pmec=final.get("pmec_health",{}); chpf=final.get("chpf",{})
    gate=result.get("gate",{}); decision=result["decision"]
    lines=[
        "# HQRF-Net Phase-0 — Query-Mask Health Gate", "",
        "## 1. Executive Decision", "", f"The preregistered train-only decision is `{decision}`. No validation image, mask, mIoU, or mDice was used.", "",
        "## 2. HCRF-v1.1 Failure Recap", "", "HCRF-v1.1 was stopped after local q-k masks expanded toward entire windows. Its worktree and incomplete artifacts were not reused.", "",
        "## 3. PCRE Mechanism Restored", "", "HQRF restores image patch queries, post-norm DETR self/cross attention, a three-layer mask embedding, query-mask dot products, PCA, and detached PMEC. PCRE reference commit: `d89029439c2676ed98035a77bc94157afd0c777c`.", "",
        "## 4. HQRF Architecture", "", "Plain ResNet38 supplies hierarchical semantic/context memories and F4/F3 pixel features; no SSHR HFRM/GSR/multi-CAM and no HCRF local-qk/cross-stage state are present.", "",
        "## 5. Query Stream", "", "A 16×16 stride-16 image patch embedding plus learned 14×14 positional embedding produces 196 queries of dimension 256.", "",
        "## 6. Deep Semantic Decoder", "", "Decoder 1 applies 8-head self-attention, detached-FD-memory cross-attention, and a 1024-wide GELU FFN.", "",
        "## 7. Contextual Decoder", "", "Decoder 2 repeats the DETR operation order against projected CH(F5) memory. Detach is applied to CNN F5/FD tensors before trainable projections, blocking memory-path gradients to the backbone while keeping the projections learnable.", "",
        "## 8. CHPF", "", f"Final gamma5/gamma4: {chpf.get('gamma5',0):.6f}/{chpf.get('gamma4',0):.6f}; F5/F4 residual norms: {chpf.get('F5_context_residual_norm',0):.6f}/{chpf.get('F4_context_residual_norm',0):.6f}.", "",
        "## 9. Pixel Decoder", "", "CH-supported F4 and raw F3 are fused at 56×56 through 3×3 convolution, GroupNorm/GELU, and a final 256-channel 3×3 convolution.", "",
        "## 10. Mask Embedding", "", "Each Q2 query is mapped by Linear–GELU–Linear–GELU–Linear and dotted with the pixel feature. BCE receives raw logits.", "",
        "## 11. PCA", "", "Class softmax is over four classes; spatial softmax is over 196 queries; their FP32 product sums to the image-level probability.", "",
        "## 12. Tri-state Supervision", "", "Positive CAM seeds use 0.60/top15%/0.10 margin. Rival positives and reliable background (≤0.10 outside dilation-1 positives) are negatives; all other pixels are ignored.", "",
        "## 13. Locality Curriculum", "", "The radius uses the Full25 denominator (29,275 steps). Query centers and radii are mapped from the 14×14 query grid to the 56×56 mask grid at the exact 4× spatial scale.", "",
        "## 14. PMEC", "", "Detached PMEC uses mask threshold 0.70, T=5, and PCRE ROR = intersection(candidate, reference) / area(reference), with 0.40/0.50 thresholds.", "",
        "## 15. Engineering Verification", "", f"Unit tests and smoke evidence are recorded in the repository. Source commit: `{result.get('source_commit','')}`; endpoint SHA256: `{result.get('checkpoint_sha256','')}`.", "",
        "## 16. Training Stability", "", f"Completed epochs/steps: {result.get('epochs',0)}/{result.get('steps',0)}. All recorded losses and gradients finite: {result.get('all_finite',False)}. Training-only paths opened: {result.get('training_paths_opened',0)}.", "",
        "## 17. Mask Area", "", f"Local median: {_fmt(area.get('local_median',0))}; local fraction >90%: {_fmt(area.get('local_fraction_gt_090',0))}; global median: {_fmt(area.get('global_median',0))}.", "",
        "## 18. Mask Logit Distribution", "", f"Mean/median/std: {logits.get('mean',0):.4f}/{logits.get('median',0):.4f}/{logits.get('std',0):.4f}; fraction logits >0: {_fmt(logits.get('fraction_gt_zero',0))}.", "",
        "## 19. Query Diversity", "", f"Mean/median pairwise mask IoU: {diversity.get('pair_iou_mean',0):.4f}/{diversity.get('pair_iou_median',0):.4f}; fraction IoU>0.90: {_fmt(diversity.get('pair_iou_fraction_gt_090',0))}; embedding p90 cosine: {diversity.get('embedding_cosine_p90',0):.4f}.", "",
        "## 20. Rival/Background Leakage", "", f"Rival leakage: {_fmt(selectivity.get('rival_seed_leakage',0))}; reliable-background leakage: {_fmt(selectivity.get('background_leakage',0))}.", "",
        "## 21. Positive-Negative Separation", "", f"Positive/negative mean logits: {selectivity.get('mean_positive_logit',0):.4f}/{selectivity.get('mean_negative_logit',0):.4f}; gap: {selectivity.get('positive_negative_logit_gap',0):.4f}.", "",
        "## 22. PCA Health", "", f"Present-class query coverage: {_fmt(pca.get('present_pair_query_coverage',0))}; absent-confidence dominance images: {_fmt(pca.get('absent_confidence_dominance_image_fraction',0))}; top1 confidence: {pca.get('top1_confidence',0):.6f}.", "",
        "## 23. PMEC Health", "", f"Pairs with candidates: {_fmt(pmec.get('candidate_pair_fraction',0))}; final PMEC differs from top1: {_fmt(pmec.get('differs_from_top1_fraction',0))}; groups/pair: {pmec.get('region_groups_per_pair',0):.3f}.", "",
        "## 24. CHPF Health", "", f"Raw/context cosine F5/F4: {chpf.get('F5_raw_context_cosine',0):.6f}/{chpf.get('F4_raw_context_cosine',0):.6f}. CHPF is diagnostic and is not itself required for GO.", "",
        "## 25. Visual Inspection", "", "The fixed eight-image panels are in `visualizations/`; selection uses the preregistered train-only cohort and no segmentation ground truth.", "",
        "## 26. Gate Evaluation", "", "| Criterion | Pass |", "|---|---:|",
    ]
    for key,value in gate.get("checks",{}).items(): lines.append(f"| {key} | {bool(value)} |")
    if gate.get("failed_criteria"): lines += ["", "Epoch-2 catastrophic criteria: " + ", ".join(gate["failed_criteria"]) + "."]
    lines += ["", "## 27. Exact Decision", "", f"`DECISION = {decision}`", "", "## 28. Full25 Recommendation", ""]
    if decision in ("HQRF_PHASE0_GO","HQRF_PHASE0_STRONG_GO"):
        lines.append("Freeze this exact architecture and prepare a separate fresh-initialization Seed42 Full25 run; do not continue from the Phase-0 checkpoint.")
    elif decision == "HQRF_QUERY_REGION_NOGO":
        lines.append("Archive the PCRE-style query-region migration route for this ResNet38 pathology setting; do not tune or rescue Phase-0.")
    else:
        lines.append("Resolve the engineering blocker before making a scientific query-region decision.")
    lines += ["", f"DECISION = {decision}"]
    path=output/"HQRF_Phase0_QueryMask_Health_Report.md"
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return path
