import numpy as np

from tools.audit_semantic_coverage_reachability import (
    affinity_reachable, anchor_category, coverage_confidence, decide_bottleneck, distance_maps,
    geodesic_distance,
)


def test_anchor_categories_are_frozen():
    assert anchor_category(.05) == "anchored"
    assert anchor_category(.049) == "weakly_anchored"
    assert anchor_category(0) == "unanchored"


def test_distance_maps_exact_axes_and_diagonal():
    seed=np.zeros((5,5),bool); seed[2,2]=1
    eu,ma,ch=distance_maps(seed)
    assert np.isclose(eu[3,3], np.sqrt(2))
    assert ma[3,3] == 2
    assert ch[3,3] == 1


def test_no_seed_is_infinite():
    assert np.isinf(distance_maps(np.zeros((3,3),bool))[0]).all()


def test_gt_constrained_geodesic_does_not_cross_gap():
    mask=np.zeros((3,5),bool); mask[1,:2]=1; mask[1,3:]=1
    seed=np.zeros_like(mask); seed[1,0]=1
    distance=geodesic_distance(mask,seed)
    assert distance[1,1] == 1 and distance[1,3] == -1


def test_geodesic_uses_eight_neighbours():
    mask=np.ones((3,3),bool); seed=np.zeros_like(mask); seed[0,0]=1
    assert geodesic_distance(mask,seed)[2,2] == 2


def test_affinity_threshold_blocks_edge():
    mask=np.ones((1,3),bool); seed=np.zeros_like(mask); seed[0,0]=1
    affinity=np.zeros((9,1,3),np.float32); affinity[5,0,0]=.6; affinity[5,0,1]=.4
    reached=affinity_reachable(mask,seed,affinity,.5)
    assert reached.tolist() == [[True,True,False]]


def test_decision_long_range():
    decision,_,_=decide_bottleneck(.8,.95,.97,.7,.05,.8,{"B":.7})
    assert decision == "LONG_RANGE_PROPAGATION_LIMIT"


def test_decision_query_mask_coverage():
    decision,_,_=decide_bottleneck(.6,.80,.80,.4,.25,.5,{"E":.6})
    assert decision == "QUERY_MASK_COVERAGE_LIMIT"


def test_decision_weighting():
    decision,_,_=decide_bottleneck(.9,.95,.98,.3,.05,.7,{"D":.6})
    assert decision == "QUERY_BASIS_WEIGHTING_LIMIT"


def test_decision_mixed():
    decision,_,_=decide_bottleneck(.75,.95,.97,.7,.25,.8,{"C":.6})
    assert decision == "MIXED_COVERAGE_AND_REACHABILITY"


def test_high_confidence_requires_three_analyses_and_three_classes():
    coverage=[{"class":c,"class_basis_uncovered":.7,"oracle_top10_recall":.3} for c in range(4)]
    coverage.append({"class":"overall","class_basis_uncovered":.78,"oracle_top10_recall":.23})
    rescue=[{"class":c,"type_C_fraction":.05,"type_E_fraction":.85} for c in range(4)]
    rescue.append({"class":"overall","type_C_fraction":.04,"type_E_fraction":.90})
    confidence,evidence=coverage_confidence(coverage,rescue)
    assert confidence == "HIGH"
    assert evidence["independent_coverage_analyses"] == 3
