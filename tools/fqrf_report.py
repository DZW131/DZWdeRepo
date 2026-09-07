"""Render the preregistered 34-section FQRF Phase-0 report."""
from pathlib import Path


def _row(summary, key, stage):
    return next((row for row in summary.get(key, []) if row.get("stage") == stage), {})


def _pct(value):
    return f"{100 * float(value):.2f}%"


def render_report(output, result):
    output = Path(output)
    final = result.get("final_summary", {})
    decision = result["decision"]
    red = [_row(final, "stagewise_query_redundancy", stage) for stage in (1, 2, 3)]
    emb = [_row(final, "stagewise_embedding_diversity", stage) for stage in (1, 2, 3)]
    focus = [_row(final, "dynamic_focus", stage) for stage in (2, 3)]
    attention = [_row(final, "attention_focus", stage) for stage in (1, 2, 3)]
    masked = [_row(final, "masked_attention_health", stage) for stage in (2, 3)]
    area = _row(final, "stagewise_mask_area", 3)
    pca = _row(final, "pca_health", 3)
    selectivity = final.get("semantic_selectivity", {})
    pmec = final.get("pmec_health", {})
    chpf = final.get("chpf_health", {})
    gate = result.get("gate", {})
    lines = [
        "# FQRF-Net Phase-0 — Focus Preservation & Query Specialization Gate", "",
        "## 1. Executive Decision", "", f"The preregistered train-only decision is `{decision}`. No validation image, segmentation GT, mIoU, or mDice was used.", "",
        "## 2. HQRF Failure Signal", "", "HQRF produced selective masks but failed query specialization: median pairwise IoU 0.9554, IoU>0.90 fraction 61.04%, and mask-embedding p90 cosine 0.9965.", "",
        "## 3. FQRF Hypothesis", "", "FQRF tests whether dynamic focus-aware position plus previous-mask-guided cross-attention can preserve distinct spatial responsibilities without a diversity loss.", "",
        "## 4. Architectural Delta from HQRF", "", "FQRF replaces self-attention-first decoding with three cross-attention-first stages and explicitly propagates content, position, mask, and cross-attention state.", "",
        "## 5. Backbone / Hierarchy", "", "The hash-locked plain ResNet38 supplies FD/F5/F4/F3. SSHR refinement modules and HCRF region state are absent.", "",
        "## 6. CHPF", "", f"Final gamma5/gamma4: {chpf.get('gamma5', 0):.6f}/{chpf.get('gamma4', 0):.6f}; residual norms F5/F4: {chpf.get('F5_context_residual_norm', 0):.6f}/{chpf.get('F4_context_residual_norm', 0):.6f}.", "",
        "## 7. Pixel Decoder", "", "The unchanged healthy HQRF pixel path fuses CHPF-supported F4 with raw F3 at 56×56.", "",
        "## 8. Query State Definition", "", "Each stage propagates (Qc,Qp,M,A); 196 patch-content queries use a learned base B. Memory Kp is a parameter-free normalized 2-D sine/cosine encoding.", "",
        "## 9. Semantic Focus Decoder", "", "Stage1 performs global cross-attention first over projected detached FD, followed by self-attention and a GELU FFN.", "",
        "## 10. Dynamic Focus Update 1", "", f"Qp2 active fraction/displacement median: {_pct(focus[0].get('active_fraction', 0))}/{focus[0].get('displacement_median', 0):.4f}.", "",
        "## 11. Context Masked Decoder", "", "Stage2 uses detached M1>=0.15 visibility over detached-F5 projected context, with global fallback for empty rows.", "",
        "## 12. Dynamic Focus Update 2", "", f"Qp3 active fraction/displacement median: {_pct(focus[1].get('active_fraction', 0))}/{focus[1].get('displacement_median', 0):.4f}.", "",
        "## 13. Fine Masked Decoder", "", "Stage3 uses detached M2>=0.15 visibility over the projected detached F4-context memory; F3 remains pixel-only.", "",
        "## 14. Stage-wise Mask Prediction", "", "Three independent 256–256–256 mask MLPs dot their embeddings with the shared pixel feature; raw logits feed BCE.", "",
        "## 15. PCA", "", "Independent dual-confidence PCA heads supervise all stages with weights 0.20/0.30/0.50.", "",
        "## 16. Tri-state Supervision", "", "The frozen HQRF CAM thresholds (0.60, top15%, margin 0.10, background 0.10) and Full25 locality denominator are unchanged.", "",
        "## 17. PMEC", "", "Only Stage3 feeds detached PCRE-style PMEC (mask 0.70, T=5, ROR 0.40/0.50).", "",
        "## 18. Engineering Verification", "", f"Source commit: `{result.get('source_commit', '')}`; endpoint SHA256: `{result.get('checkpoint_sha256', '')}`. Unit, protocol, smoke, and endpoint integrity checks are recorded with the artifacts.", "",
        "## 19. Training Stability", "", f"Completed epochs/steps: {result.get('epochs', 0)}/{result.get('steps', 0)}; all diagnostics finite: {result.get('all_finite', False)}; peak CUDA memory: {result.get('peak_cuda_memory_gib', 0):.3f} GiB.", "",
        "## 20. Stage-wise Query Redundancy", "",
        "| Stage | Mean IoU | Median IoU | IoU>0.90 | IoU>0.75 |", "|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(red, start=1):
        lines.append(f"| {index} | {row.get('pair_iou_mean', 0):.4f} | {row.get('pair_iou_median', 0):.4f} | {_pct(row.get('pair_iou_fraction_gt_090', 0))} | {_pct(row.get('pair_iou_fraction_gt_075', 0))} |")
    lines += ["", "## 21. Mask-Embedding Diversity", "",
              "| Stage | Cosine mean | Median | P90 | >0.95 |", "|---:|---:|---:|---:|---:|"]
    for index, row in enumerate(emb, start=1):
        lines.append(f"| {index} | {row.get('cosine_mean', 0):.4f} | {row.get('cosine_median', 0):.4f} | {row.get('cosine_p90', 0):.4f} | {_pct(row.get('cosine_fraction_gt_095', 0))} |")
    lines += [
        "", "## 22. Dynamic Positional Behavior", "", f"Qp2/Qp3 focus cosine p90: {focus[0].get('focus_cosine_p90', 0):.4f}/{focus[1].get('focus_cosine_p90', 0):.4f}.", "",
        "## 23. Attention Focus Behavior", "", f"Stage1/2/3 centroid pairwise distances: {attention[0].get('centroid_pairwise_distance_mean', 0):.4f}/{attention[1].get('centroid_pairwise_distance_mean', 0):.4f}/{attention[2].get('centroid_pairwise_distance_mean', 0):.4f}.", "",
        "## 24. Masked-Attention Health", "", f"Stage2/3 visible means: {_pct(masked[0].get('visible_ratio_mean', 0))}/{_pct(masked[1].get('visible_ratio_mean', 0))}; fallback fractions: {_pct(masked[0].get('fallback_to_global_fraction', 0))}/{_pct(masked[1].get('fallback_to_global_fraction', 0))}.", "",
        "## 25. Semantic Selectivity", "", f"Rival/background leakage: {_pct(selectivity.get('rival_leakage', 0))}/{_pct(selectivity.get('background_leakage', 0))}; positive-negative gap: {selectivity.get('positive_negative_gap', 0):.4f}.", "",
        "## 26. Mask-Area Guardrails", "", f"Stage3 local median/high-area fraction/global median: {_pct(area.get('local_median', 0))}/{_pct(area.get('local_fraction_gt_090', 0))}/{_pct(area.get('global_median', 0))}.", "",
        "## 27. PCA Health", "", f"Stage3 present coverage/absent dominance/top1 confidence: {_pct(pca.get('present_class_query_coverage', 0))}/{_pct(pca.get('absent_class_dominance', 0))}/{pca.get('top1_confidence', 0):.6f}.", "",
        "## 28. PMEC Health", "", f"Candidate coverage/differs from top1/groups per pair: {_pct(pmec.get('candidate_coverage', 0))}/{_pct(pmec.get('differs_from_top1', 0))}/{pmec.get('groups_per_pair', 0):.3f}.", "",
        "## 29. CHPF Health", "", f"Raw/context cosine F5/F4: {chpf.get('F5_raw_context_cosine', 0):.6f}/{chpf.get('F4_raw_context_cosine', 0):.6f}. CHPF is diagnostic, not a GO criterion.", "",
        "## 30. Visual Analysis", "", "Fixed train-cohort panels are stored under `visualizations/`. They contain no segmentation ground truth.", "",
        "## 31. Gate Evaluation", "", "| Criterion | Pass |", "|---|---:|",
    ]
    for key, passed in gate.get("checks", {}).items():
        lines.append(f"| {key} | {bool(passed)} |")
    if gate.get("failed_criteria"):
        lines += ["", "Epoch-2 catastrophic criteria: " + ", ".join(gate["failed_criteria"]) + "."]
    lines += [
        "", "## 32. Scientific Interpretation", "",
        "A GO means architectural focus preservation alone produced selective and specialized queries. A NOGO means dynamic focus plus previous-mask-guided attention was insufficient without adding a prohibited rescue objective.", "",
        "## 33. Exact Decision", "", f"`DECISION = {decision}`", "",
        "## 34. Full25 Recommendation", "",
    ]
    if decision in ("FQRF_PHASE0_GO", "FQRF_PHASE0_STRONG_GO"):
        lines.append("Freeze the exact architecture and prepare a separate fresh-MXNet-initialization Seed42 Full25 run; do not continue this Phase-0 checkpoint.")
    elif decision == "FQRF_FOCUS_NOGO":
        lines.append("Archive the FQRF dynamic-focus plus masked-attention route; do not tune or add diversity/repulsion rescue losses.")
    else:
        lines.append("Resolve the engineering blocker before making a scientific decision.")
    lines += ["", f"DECISION = {decision}"]
    path = output / "FQRF_Phase0_Focus_Preservation_Report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

