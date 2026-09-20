import numpy as np
from audits.umrf_v1.core import decision, metric_block, rules_from_probabilities

def test_rules_are_frozen():
    p5=np.eye(4,dtype=np.float32)[:,[0,0,1,2]]
    p4=np.eye(4,dtype=np.float32)[:,[1,1,2,3]]
    p3=np.eye(4,dtype=np.float32)[:,[1,2,2,0]]
    out=rules_from_probabilities(p5,p4,p3)
    assert out["r1"].tolist()==[1,0,2,2]
    assert out["r2"].tolist()==[1,0,2,2]
    assert out["r3"].tolist()==[1,0,2,0]

def test_metric_and_decision_boundaries():
    h5=np.array([1,1,0,0]); h4=np.array([0,0,0,1]); h3=np.array([0,0,0,1]); truth=np.array([0,0,0,0])
    m=metric_block(h5,h4,h3,truth)
    assert m["corrected"]==2 and m["harmed"]==1 and m["nce"]==2
    assert decision({"precision":.9,"coverage":.4,"nce":2.1,"h5_correct_harm_rate":.01})=="STRONG_GO"
    assert decision({"precision":.64,"coverage":.4,"nce":2.1,"h5_correct_harm_rate":.01})=="NOGO"
