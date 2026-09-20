import numpy as np

from audits.oracle_action_separability_v1.analyze import (
    A_ABLATIONS, G_ABLATIONS, ablation_features, decision_a, decision_g, metrics,
)


def test_operating_points_respect_score_ties():
    result = metrics(np.array([1, 0, 0, 1]), np.array([.9, .9, .9, .1]))
    assert result["recall_p80"] == 0
    assert np.isclose(result["precision_top1"], 1/3)


def test_perfect_separation():
    result = metrics(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8]))
    assert result["auroc"] == result["auprc"] == result["recall_p95"] == 1


def test_ablation_family_complete():
    a=["a_disagreement_mean", "a_energy_ratio", "a_deep_entropy", "a_c3_margin",
       "a_tta_class_vote", "a_log_area"]
    g=["g_deep_score", "g_tta_std", "g_c3_top05", "g_c4_c3_corr", "g_query_support"]
    assert set(ablation_features(a, A_ABLATIONS)["A5"]) == set(a)
    assert set(ablation_features(g, G_ABLATIONS)["G4"]) == set(g)


def test_decisions_use_safety_operating_points():
    a={"auroc":.9,"recall_p80":.4,"enrichment":2.5}
    assert decision_a(a,a)=="STRONG_ARBITRATION_GO"
    assert decision_a(a,{**a,"recall_p80":.09})=="ARBITRATION_ACTION_NOT_SEPARABLE"
    g={"auroc":.93,"recall_p80":.3,"recall_p90":.22,"enrichment":3.2}
    assert decision_g(g)=="STRONG_GATE_GO"
    assert decision_g({**g,"recall_p80":.09})=="GATE_RESCUE_NOT_SEPARABLE"
