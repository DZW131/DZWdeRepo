"""Preregistered OSMF-v1.3 readiness and Phase-0S decisions."""

from __future__ import annotations

import math
from collections import defaultdict

from tools.osmf_v13_audit import PARAMETER_NAMES


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    fraction = position - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def causal_statistics(rows):
    deltas = [float(row["delta"]) for row in rows]
    improved = sum(delta < -1e-6 for delta in deltas)
    harmed = sum(delta > 1e-6 for delta in deltas)
    neutral = len(deltas) - improved - harmed
    return {
        "num_struct_steps": len(deltas),
        "num_improved": improved,
        "num_harmed": harmed,
        "num_neutral": neutral,
        "improved_fraction": improved / len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "median_delta": percentile(deltas, 0.5),
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }


def _ratio_stats(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["objective"]].append(float(row["ratio"]))
    return {
        name: {
            "mean": sum(values) / len(values),
            "max": max(values),
            "p95": percentile(values, 0.95),
            "values": values,
        }
        for name, values in grouped.items()
    }


def _safety_failures(finite, representation_rows, parameter_summary):
    failures = []
    if not finite:
        failures.append("NONFINITE_TENSOR_LOSS_OR_GRADIENT")
    for name in PARAMETER_NAMES:
        health = parameter_summary[name]
        if not health["grad_nonzero"] or not health["measurable_update"]:
            failures.append(f"INACTIVE_PARAMETER_{name}")
    end = representation_rows[-1]
    if float(end["reconstruction_cosine"]) < 0.90:
        failures.append("RECONSTRUCTION_DESTABILIZED")
    if not 0.05 < float(end["semantic_morphology_rms_ratio"]) < 20.0:
        failures.append("BRANCH_COLLAPSE")
    if float(end["semantic_response_rms_ratio"]) < 0.05:
        failures.append("SEMANTIC_RESPONSE_COLLAPSE")
    return failures


def readiness_decision(*, finite, ratio_rows, representation_rows,
                       parameter_summary, morph_struct_active,
                       causal_rows, sshr_loss_stable):
    nogo, review = _safety_failures(
        finite, representation_rows, parameter_summary
    ), []
    if not morph_struct_active:
        nogo.append("MORPHOLOGY_STRUCTURAL_PATH_INACTIVE")
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
        return "OSMF_V13_READINESS_NOGO", sorted(set(nogo)), stats, causal
    if review:
        return "OSMF_V13_READINESS_REVIEW", sorted(set(review)), stats, causal
    return "OSMF_V13_READINESS_PASS", [], stats, causal


def phase0s_decision(*, finite, ratio_rows, representation_rows,
                     parameter_summary, morph_struct_active, causal_rows,
                     fixed_rows, sshr_loss_stable, cross_covariance_healthy):
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
    if not morph_struct_active:
        nogo.append("MORPHOLOGY_STRUCTURAL_PATH_INACTIVE")
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
        return "OSMF_V13_PHASE0S_NOGO", sorted(set(nogo)), stats, causal, evidence
    go = (
        causal["improved_fraction"] >= 0.75
        and causal["mean_delta"] < 0
        and m_improve >= 0.05
        and health_ok
    )
    if go:
        return "OSMF_V13_PHASE0S_GO", [], stats, causal, evidence
    return "OSMF_V13_PHASE0S_REVIEW", ["GO_CRITERIA_NOT_ALL_MET"], stats, causal, evidence
