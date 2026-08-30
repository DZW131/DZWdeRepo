import ast
import inspect
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b1_common import compute_support,neighbors,bootstrap_indices,binary_exact
from tools.rddr_phase2b15_common import (compute_four_way_support_matrix,compute_same_family_bias,
    compute_delta_symmetric,probes,gt_context_diagnostic,third_class_metrics,
    class_status,aggregate_class_status,decide,make_groups)


class BiasAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(42)
        cls.s=torch.randn(1,4,28,28).softmax(1)
        cls.d=torch.randn(1,4,28,28).softmax(1)
        cls.t=probes(cls.s,cls.d)

    def test_phase2b1_delta_exact_reproduction(self):
        old=compute_support(self.s,self.d)
        for a,b in (("T_SS","ss"),("T_DS","sd"),("old","delta"),("ctx_S","ctx")):
            self.assertTrue(torch.equal(self.t[a],old[b]))

    def test_four_way_support_shapes(self):
        for k in ("T_SS","T_SD","T_DS","T_DD"):
            self.assertEqual(self.t[k].shape,(1,784))
            self.assertTrue(((self.t[k]>=0)&(self.t[k]<=1)).all())

    def test_support_scores_no_gt(self):
        self.assertEqual(list(inspect.signature(compute_four_way_support_matrix).parameters),["ps","pd"])

    def test_same_family_bias_formula(self):
        b=compute_same_family_bias(self.t)
        self.assertTrue(torch.equal(b["B_S"],self.t["T_SS"]-self.t["T_SD"]))
        self.assertTrue(torch.equal(b["B_D"],self.t["T_DD"]-self.t["T_DS"]))
        self.assertTrue(torch.equal(b["B_family"],.5*(b["B_S"]+b["B_D"])))

    def test_symmetric_support_formula(self):
        a,b,d=compute_delta_symmetric(self.t)
        self.assertTrue(torch.equal(d,b-a))
        self.assertTrue(torch.equal(a,.5*(self.t["T_SS"]+self.t["T_SD"])))

    def test_delta_sym_no_gt(self):
        self.assertEqual(list(inspect.signature(probes).parameters),["ps","pd"])
        self.assertNotIn("gt_context",inspect.getsource(probes))

    def test_anchor_sym_probability_sum(self):
        torch.testing.assert_close(self.t["anchor_sym"].sum(1),torch.ones(1,784),rtol=0,atol=3e-7)

    def test_15x15_window_exact(self):
        valid=neighbors(torch.ones(1,1,28,28))[:,0];valid[:,112]=0
        self.assertEqual(int(valid[0,:,0].sum()),63)
        self.assertEqual(int(valid[0,:,14*28+14].sum()),224)

    def test_self_edge_excluded(self):
        x=torch.zeros(1,1,28,28); x[0,0,14,14]=1
        source=neighbors(x); source[:,:,112]=0
        self.assertEqual(float(source[0,0,:,14*28+14].sum()),0.)

    def test_context_sym_formula(self):
        self.assertTrue(torch.equal(self.t["ctx_sym"],.5*(self.t["ctx_S"]+self.t["ctx_D"])))

    def test_branch_swap_symmetry(self):
        swapped=probes(self.d,self.s)
        torch.testing.assert_close(swapped["sym"],-self.t["sym"],rtol=0,atol=0)

    def test_third_class_metrics_gt_only_in_analysis(self):
        y=np.array([2,3]); s=np.array([0,0]); d=np.array([1,1]);c=np.array([2,0]);m=np.ones(2,bool)
        r=third_class_metrics(y,s,d,c,m)
        self.assertEqual(r["rescue_rate"],.5);self.assertEqual(r["rescue_precision"],1.)
        y=np.array([0,1]);c=np.array([2,1]);r=third_class_metrics(y,s,d,c,m)
        self.assertEqual(r["intrusion_rate"],r["harm_rate"])

    def test_gt_context_denominator(self):
        y=torch.zeros(1,28,28,dtype=torch.long);y[:,0,0]=4;y[:,0,1]=255
        s=torch.zeros_like(y);d=torch.ones_like(y)
        r=gt_context_diagnostic(y,s,d)
        total=sum(r[k] for k in ("GT_shallow_candidate_fraction","GT_deep_candidate_fraction","GT_other_fraction","GT_background_fraction","GT_ignore_fraction"))
        torch.testing.assert_close(total,torch.ones_like(total),rtol=0,atol=2e-7)
        self.assertAlmostEqual(float(r["GT_background_fraction"][0,0]),0.)

    def test_ordered_pair_partition_exact(self):
        s=np.arange(16)//4;d=np.arange(16)%4
        parts=[(s==a)&(d==b) for a in range(4) for b in range(4) if a!=b]
        np.testing.assert_array_equal(sum(parts),s!=d)

    def test_frozen_group_counts(self):
        y=np.array([[0,1,2,3,4,255]])
        ps=np.eye(4)[[0,0,2,0,0,0]].T[None];pd=np.eye(4)[[0,1,0,1,0,0]].T[None]
        data=dict(truth=y,ps=ps,pd=pd,top20=np.array([[1,0,0,1,0,0]]),boundary=np.zeros_like(y),q_feature=np.zeros_like(y),hfrm=np.zeros_like(y))
        g,win,*_=make_groups(data)
        self.assertEqual(g["all"].sum(),4)
        np.testing.assert_array_equal(g["Both_Correct"]+g["Both_Wrong"]+g["Deep_Win"]+g["Shallow_Win"],g["all"])
        np.testing.assert_array_equal(g["Top20"]+g["Bottom80"],g["all"])
        self.assertEqual(win.sum(),2)

    def test_no_gradient(self):
        t=probes(self.s.requires_grad_(),self.d.requires_grad_())
        self.assertTrue(all(not v.requires_grad for v in t.values()))

    def test_no_optimizer(self):
        src=(ROOT/"tools/run_rddr_phase2b15_bias_decomposition_audit.py").read_text()
        self.assertNotIn("torch.optim.",src); self.assertNotIn(".backward(",src)

    def test_no_checkpoint_write(self):
        src=(ROOT/"tools/run_rddr_phase2b15_bias_decomposition_audit.py").read_text()
        self.assertNotIn("torch.save(",src); self.assertNotIn("torch.load(",src)

    def test_no_test_luad_access(self):
        src=(ROOT/"tools/run_rddr_phase2b15_bias_decomposition_audit.py").read_text()
        self.assertNotIn("/test/",src);self.assertNotIn("LUAD-HistoSeg",src);self.assertNotIn("GenDataset",src)

    def test_bootstrap_reproducible(self):
        a=np.concatenate(list(bootstrap_indices(20,100,42)))
        b=np.random.default_rng(42).integers(0,20,(100,20),dtype=np.int32)
        np.testing.assert_array_equal(a,b)

    def test_exact_auc_ties_and_absence(self):
        self.assertEqual(binary_exact([1,1],[0,1])["auroc"],.5)
        self.assertTrue(np.isnan(binary_exact([1],[1])["auroc"]))

    def test_class_underpower_precedence(self):
        self.assertEqual(class_status(74336,418,.9),"UNDERPOWERED")
        self.assertEqual(aggregate_class_status(["FAIL","UNDERPOWERED"]),"FAIL")
        self.assertEqual(aggregate_class_status(["PASS","UNDERPOWERED"]),"UNDERPOWERED")

    def test_decision_all_branches(self):
        self.assertEqual(decide(False,True,"PASS",True),"SAME_FAMILY_BIAS_HYPOTHESIS_NOT_SUPPORTED")
        self.assertEqual(decide(True,False,"PASS",True),"THIRD_EVIDENCE_REQUIRED_FOR_NEXT_DESIGN")
        self.assertEqual(decide(True,False,"PASS",False),"ADJUDICATION_BIAS_UNRESOLVED")
        self.assertEqual(decide(True,True,"FAIL",True),"ADJUDICATION_BIAS_UNRESOLVED")
        self.assertEqual(decide(True,True,"UNDERPOWERED",True),"SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED")
        self.assertEqual(decide(True,True,"PASS",False),"SYMMETRIC_ADJUDICATION_BIAS_RESOLVED")


if __name__=="__main__":unittest.main()
