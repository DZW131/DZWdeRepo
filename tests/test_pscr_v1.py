import numpy as np

from audits.pscr_v1.run_pscr_v1 import (
    ablation_decisions, candidate_softmax, decision_from_recovery,
    evidence_summary, policy_decisions,
)


def scores_for(top_classes, flip_top=None, margin=3.0):
    scores = np.zeros((1, 2, 4, 3, 4), np.float32)
    for branch, cls in enumerate(top_classes): scores[0, 0, branch, 2, cls] = margin
    for branch, cls in enumerate(flip_top or top_classes): scores[0, 1, branch, 2, cls] = margin
    scores[:, :, :, 0] = scores[:, :, :, 2]
    return scores


def test_candidate_softmax_hard_gates_absent_classes():
    value = candidate_softmax(np.array([[2., 9., 1., 8.]], np.float32), np.array([1, 0, 1, 0]))
    assert value[0, 1] == value[0, 3] == 0
    assert np.isclose(value[0].sum(), 1)
    assert value[0].argmax() == 0


def test_evidence_summary_votes_and_view():
    scores = scores_for([1, 1, 1, 0])[0]
    result = evidence_summary(scores, np.ones(4, bool), 2)
    assert result["consensus"] == 1
    assert result["agreement"] == .75
    assert result["view_consistent"]


def test_policies_are_preregistered():
    scores = scores_for([1, 1, 1, 1])
    present = np.ones((1, 4), bool); base = np.array([0])
    assert policy_decisions(scores, base, present, "P1").iloc[0].action == "RELABEL"
    assert policy_decisions(scores, base, present, "P2").iloc[0].action == "RELABEL"
    assert policy_decisions(scores, base, present, "P3").iloc[0].action == "RELABEL"
    ambiguous = scores_for([1, 1, 0, 0])
    assert policy_decisions(ambiguous, base, present, "P3").iloc[0].action == "REJECT"


def test_ablation_flip_only_enters_a4():
    scores = scores_for([1, 1, 1, 1], flip_top=[2, 2, 2, 2])
    present = np.ones((1, 4), bool); base = np.array([0])
    assert ablation_decisions(scores, base, present, "A3").iloc[0].action == "RELABEL"
    assert ablation_decisions(scores, base, present, "A4").iloc[0].action == "KEEP"


def test_decision_thresholds():
    checks = {"a": True}
    assert decision_from_recovery(.24, .1, .9, checks) == "NOGO"
    assert decision_from_recovery(.25, .1, .9, checks) == "WEAK"
    assert decision_from_recovery(.40, .15, .9, checks) == "GO"
    assert decision_from_recovery(.60, .19, .9, checks) == "STRONG_GO"
    assert decision_from_recovery(.75, .22, .70, checks) == "ARCHITECTURE_READY"
