"""Render the preregistered 43-section CQRF Phase-0 report."""
from pathlib import Path


def _row(summary, key, stage):
    return next((r for r in summary.get(key, []) if r.get("stage") == stage), {})


def _f(row, key, digits=4):
    value = row.get(key)
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_report(output, result):
    output = Path(output); summary = result.get("final_summary", {}); decision = result["decision"]
    red = _row(summary, "stagewise_query_redundancy", 3); area = _row(summary, "stagewise_mask_area", 3)
    util = _row(summary, "responsibility_utilization", 3); comp = _row(summary, "responsibility_complementarity", 3)
    integ = _row(summary, "responsibility_integrity", 3); update = _row(summary, "query_update_health", 3)
    pca = _row(summary, "pca_health", 3); sem = summary.get("semantic_selectivity", {})
    pmec = summary.get("pmec_health", {}); deep = summary.get("deep_gate_health", {}); gate = result.get("gate", {})
    runtime = result
    sections = [
        ("Executive Decision", f"Preregistered train-only decision: `{decision}`. No validation/test image, segmentation GT, mIoU, or mDice was used."),
        ("Scientific Question", "Whether class-conditioned competition allocates spatial responsibility across patch queries without semantic or mask-health collapse."),
        ("Frozen Boundary", "BCSS training split only; Seed42; official ResNet38 initialization; maximum five epochs."),
        ("Prior FQRF Finding", "FQRF retained semantics and mask health but showed high late-stage query redundancy, motivating direct competitive allocation."),
        ("Mechanism Under Test", "CCRA normalizes class-conditioned pixel responsibility across the 196-query axis."),
        ("Excluded Mechanisms", "No DFPQ, previous-mask attention gate, Stage2/3 self-attention, diversity regularizer, Sinkhorn, hard assignment, or GT-label routing."),
        ("Stage 1", "Global cross-attention first, followed by self-attention and FFN; Q0 plus learned base B."),
        ("Stage 2", "CCRA over detached F5 CHPF memory, using detached PCA1 and predicted deep gate priors."),
        ("Stage 3", "CCRA over detached, coherently transformed F4 CHPF memory, using detached PCA2 and predicted deep gate priors."),
        ("Memory Position", "Parameter-free normalized 2-D sine/cosine Kp, inherited from the audited FQRF interpretation."),
        ("Responsibility Formula", "FP32 affinity plus log class prior; softmax is strictly over the query dimension for each pixel/class."),
        ("Responsibility Integrity", f"Stage3 max/mean absolute query-sum error: {_f(integ,'max_abs_sum_error',8)} / {_f(integ,'mean_abs_sum_error',8)}."),
        ("Class Prior and Fallback", "PCA probabilities and sigmoid deep gates are detached; all-gates-tiny images fall back to normalized PCA."),
        ("Query Update", f"Stage3 projected-update mean {_f(update,'projected_update_norm_mean')}; delta mean {_f(update,'query_delta_norm_mean')}; old/new cosine {_f(update,'old_new_cosine_mean')}."),
        ("Loss Contract", "Stage weights 0.20/0.30/0.50; total loss = 0.50 deep + 0.25 PCA + 0.25 mask."),
        ("Pseudo-label Contract", "Frozen tri-state CAM thresholds: positive .60, top 15%, class margin .10, background .10, minimum four labeled pixels."),
        ("Locality Contract", "Full25 radius schedule denominator is 29,275 steps."),
        ("Monitoring Cohort", "Fixed 32-image Seed42 train cohort with at least 8 single-class and 16 multi-class images."),
        ("Snapshot Schedule", "step250, step500, step1000, and epoch1 through epoch5."),
        ("Numerical Health", f"All finite: {summary.get('all_finite','n/a')}; runtime gate: {runtime.get('all_finite','n/a')}."),
        ("Stagewise Redundancy", "See `cqrf_stagewise_query_redundancy.csv` for all stages and snapshots."),
        ("Stage3 Endpoint Redundancy", f"Median IoU {_f(red,'pair_iou_median')}; IoU>.90 fraction {_f(red,'pair_iou_fraction_gt_090')}; IoU>.75 fraction {_f(red,'pair_iou_fraction_gt_075')}."),
        ("Temporal Rebound", f"No-rebound gate: {gate.get('checks',{}).get('B_no_rebound','n/a')}."),
        ("Embedding Diversity", "See `cqrf_stagewise_embedding_diversity.csv`; this is diagnostic only and has no auxiliary loss."),
        ("Responsibility Utilization", f"Stage3 median dominant share {_f(util,'dominant_share_median')}; median effective queries {_f(util,'effective_queries_median')}; normalized entropy {_f(util,'normalized_entropy_median')}."),
        ("Responsibility Complementarity", f"Top-responsible support IoU {_f(comp,'responsibility_iou_median')}; centroid distance {_f(comp,'centroid_distance_mean')}; distinct peaks {_f(comp,'distinct_peak_fraction')}."),
        ("Mask Area", f"Stage3 local median {_f(area,'local_median')}; >.90 fraction {_f(area,'local_fraction_gt_090')}; <.05 fraction {_f(area,'local_fraction_lt_005')}."),
        ("Over-fragmentation", "Connected-component diagnostics are in `cqrf_over_fragmentation.csv`."),
        ("Semantic Selectivity", f"Rival leakage {_f(sem,'rival_leakage')}; background leakage {_f(sem,'background_leakage')}; positive-negative logit gap {_f(sem,'positive_negative_gap')}."),
        ("Deep Gate Health", f"Present/absent means {_f(deep,'present_mean')} / {_f(deep,'absent_mean')}; gap {_f(deep,'present_absent_gap')}."),
        ("PCA Health", f"Stage3 present-class coverage {_f(pca,'present_class_query_coverage')}; absent dominance {_f(pca,'absent_class_dominance')}."),
        ("PMEC Health", f"Candidate coverage {_f(pmec,'candidate_coverage')}; differs from top1 {_f(pmec,'differs_from_top1')}; groups/pair {_f(pmec,'groups_per_pair')}."),
        ("CHPF Health", "F5/F4 residual and raw-context cosine traces are in `cqrf_chpf_health.csv`."),
        ("Gradient Health", "Per-module mean/p50/p90 gradient RMS, zero fraction, and nonfinite fraction are in `cqrf_gradient_health.csv`."),
        ("Epoch-2 Catastrophic Screen", str(result.get("epoch2_screen", "not reached or unavailable"))),
        ("Final Gate Checks", "\n".join(f"- {k}: {v}" for k,v in gate.get("checks",{}).items()) or "Unavailable."),
        ("Strong-GO Check", f"Strong-GO: {gate.get('strong_go',False)}."),
        ("Training Runtime", f"Steps {runtime.get('steps','n/a')}; epochs {runtime.get('epochs','n/a')}; seconds {runtime.get('train_seconds','n/a')}; peak GiB {runtime.get('peak_cuda_memory_gib','n/a')}."),
        ("Data-access Audit", f"Validation accessed: {runtime.get('validation_accessed',False)}; test accessed: {runtime.get('test_accessed',False)}; LUAD accessed: {runtime.get('luad_accessed',False)}."),
        ("Reproducibility", f"Source commit `{result.get('source_commit','unknown')}`; endpoint SHA256 `{runtime.get('checkpoint_sha256','unavailable')}`."),
        ("Visual Evidence", "Eight fixed train images are rendered at step500, step1000, and epoch2–epoch5 under `visualizations/`."),
        ("Interpretation", "The decision follows the preregistered all-gates rule; individual favorable metrics cannot override a failed required criterion."),
        ("Next Action", "GO permits a fresh-initialization Full25 run; NOGO archives the current query-allocation family without rescue tuning."),
    ]
    assert len(sections) == 43
    lines = ["# CQRF-Net Phase-0 — CCRA Responsibility Allocation Report", ""]
    for index, (title, body) in enumerate(sections, 1): lines += [f"## {index}. {title}", "", body, ""]
    lines.append(f"DECISION = {decision}")
    path = output / "CQRF_Phase0_CCRA_Responsibility_Report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
