"""OSMF-v1.3-R1 decisions with graph-corrected morphology connectivity."""

from __future__ import annotations

from tools.osmf_v13_audit.decision import (
    _ratio_stats,
    _safety_failures,
    causal_statistics,
)


def readiness_decision(*, finite, ratio_rows, representation_rows,
                       parameter_summary, morphology_graph_expected,
                       semantic_path_active, causal_rows, sshr_loss_stable):
    nogo, review = _safety_failures(
        finite, representation_rows, parameter_summary
    ), []
    if not morphology_graph_expected:
        nogo.append("MORPHOLOGY_GRAPH_EXPECTATION_FAILED")
    if not semantic_path_active:
        nogo.append("SEMANTIC_PRESERVATION_PATH_INACTIVE")
    if not sshr_loss_stable:
        nogo.append("SSHR_LOSS_DESTABILIZED")
    causal = causal_statistics(causal_rows)
    if all(float(row["delta"]) > 0 for row in causal_rows):
        nogo.append("STRUCTURAL_UPDATE_HARMS_BOTH_ACTIVE_STEPS")
    elif causal["mean_delta"] >= 0 or causal["num_improved"] < 1:
        review.append("SAME_PAIR_STRUCTURAL_CAUSAL_NOT_FAVORABLE")
    stats = _ratio_stats(ratio_rows)
    for objective in ("sem_pres", "struct"):
        if stats[objective]["mean"] > 0.20 or stats[objective]["max"] > 0.30:
            review.append(f"{objective.upper()}_GRADIENT_BUDGET_NOT_MET")
    if nogo:
        return "OSMF_V13R1_READINESS_NOGO", sorted(set(nogo)), stats, causal
    if review:
        return "OSMF_V13R1_READINESS_REVIEW", sorted(set(review)), stats, causal
    return "OSMF_V13R1_READINESS_PASS", [], stats, causal


def phase0s_decision(*, finite, ratio_rows, representation_rows,
                     parameter_summary, morphology_graph_expected,
                     semantic_path_active, causal_rows, fixed_rows,
                     sshr_loss_stable, cross_covariance_healthy):
    nogo = _safety_failures(finite, representation_rows, parameter_summary)
    stats = _ratio_stats(ratio_rows)
    causal = causal_statistics(causal_rows)
    start, end = fixed_rows[0], fixed_rows[-1]
    m_start = float(start["affinity_eq_error_morphology"])
    m_end = float(end["affinity_eq_error_morphology"])
    s_start = float(start["affinity_eq_error_semantic"])
    s_end = float(end["affinity_eq_error_semantic"])
    m_improve = (m_start - m_end) / (m_start + 1e-12)
    s_improve = (s_start - s_end) / (s_start + 1e-12)
    specificity_gap = m_improve - s_improve
    if not morphology_graph_expected:
        nogo.append("MORPHOLOGY_GRAPH_EXPECTATION_FAILED")
    if not semantic_path_active:
        nogo.append("SEMANTIC_PRESERVATION_PATH_INACTIVE")
    if not sshr_loss_stable:
        nogo.append("SSHR_LOSS_DESTABILIZED")
    if not cross_covariance_healthy:
        nogo.append("CROSS_COVARIANCE_DESTABILIZED")
    if causal["improved_fraction"] < 0.50 or causal["mean_delta"] >= 0:
        nogo.append("SAME_PAIR_STRUCTURAL_CAUSAL_INVALID")
    if m_end - m_start > 0.005:
        nogo.append("FIXED_MORPHOLOGY_AFFINITY_CLEARLY_WORSENED")
    budget_ok = all(
        stats[name]["mean"] <= 0.20 and stats[name]["p95"] <= 0.30
        for name in ("sem_pres", "struct")
    )
    health_ok = (
        float(representation_rows[-1]["semantic_agreement"]) >= 0.90
        and float(representation_rows[-1]["reconstruction_cosine"]) >= 0.95
        and budget_ok
    )
    evidence = {
        "struct_improve_m": m_improve,
        "struct_improve_s": s_improve,
        "specificity_gap": specificity_gap,
        "budget_ok": budget_ok,
        "health_ok": health_ok,
    }
    if nogo:
        return "OSMF_V13R1_PHASE0S_NOGO", sorted(set(nogo)), stats, causal, evidence
    if (
        causal["improved_fraction"] >= 0.75
        and causal["mean_delta"] < 0
        and m_improve >= 0.05
        and health_ok
    ):
        return "OSMF_V13R1_PHASE0S_GO", [], stats, causal, evidence
    return "OSMF_V13R1_PHASE0S_REVIEW", ["GO_CRITERIA_NOT_ALL_MET"], stats, causal, evidence
