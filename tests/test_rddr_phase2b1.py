import ast
import inspect
import math
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import rddr_phase2b1_common as c
from tools.run_rddr_phase2b1_dual_hypothesis_audit import validate_root


class DualHypothesisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(42)
        cls.ps=torch.randn(1,4,28,28).softmax(1)
        cls.pd=torch.randn(1,4,28,28).softmax(1)
        cls.r=c.compute_support(cls.ps,cls.pd)

    def test_phase0_js_exact_match(self):
        p,q=self.ps,self.pd; m=.5*(p+q)
        js=.5*((p*((p+1e-8).log()-(m+1e-8).log())).sum(1)+(q*((q+1e-8).log()-(m+1e-8).log())).sum(1))
        self.assertTrue(torch.equal(js,c.phase0_js(p,q)))

    def test_no_gt_in_support_score(self):
        self.assertEqual(list(inspect.signature(c.compute_support).parameters),["ps","pd"])
        tree=ast.parse(inspect.getsource(c.compute_support))
        names={v.id for v in ast.walk(tree) if isinstance(v,ast.Name)}
        self.assertFalse(names & {"truth","gt","label","labels","reliability"})

    def test_gt_only_used_for_audit_targets(self):
        original={k:v.clone() for k,v in self.r.items()}
        for y in (np.zeros((28,28)),np.full((28,28),4)):
            c.winner_labels(y,self.ps.argmax(1)[0].numpy(),self.pd.argmax(1)[0].numpy())
        for k,v in original.items(): self.assertTrue(torch.equal(v,self.r[k]))

    def test_15x15_window_exact(self):
        self.assertEqual(self.r["valid"][0,:,0].sum(),63)
        self.assertEqual(self.r["valid"][0,:,14*28+14].sum(),224)

    def test_self_edge_excluded(self): self.assertFalse(self.r["valid"][:,112].any())

    def test_support_range_0_1(self):
        for k in ("ss","sd","wd","ws"):
            self.assertTrue(((self.r[k]>=0)&(self.r[k]<=1)).all())

    def test_delta_finite(self):
        self.assertTrue(torch.isfinite(self.r["delta"]).all())
        self.assertTrue(torch.equal(self.r["delta"],self.r["sd"]-self.r["ss"]))

    def test_weights_sum_to_one(self): self.assertTrue(torch.equal(self.r["wd"]+self.r["ws"],torch.ones_like(self.r["wd"])))

    def test_anchor_probability_sum_one(self): self.assertTrue(torch.allclose(self.r["anchor"].sum(1),torch.ones(1,784),atol=1e-6))

    def test_fixed_average_sanity(self):
        average=.5*self.ps+.5*self.pd
        self.assertTrue(torch.allclose(average.sum(1),torch.ones(1,28,28),atol=1e-6))
        r=c.compute_support(self.ps,self.ps)
        self.assertTrue(torch.equal(r["delta"],torch.zeros_like(r["delta"])))
        self.assertTrue(torch.allclose(r["anchor"],self.ps.flatten(2),atol=1e-7))
        self.assertFalse(r["choose_deep"].any())

    def test_frozen_population_counts(self):
        y=np.array([[0,1],[2,3]],np.uint8)
        masks=c.populations(dict(raw=np.array([[1,2],[2,3]]),rect=np.array([[0,3],[1,3]]),top20=np.array([[1,0],[0,0]],bool)),y)
        for k in c.HFRM_GROUPS:
            self.assertEqual(masks[k].sum(),1); self.assertEqual(c.project(masks[k]).sum(),196)

    def test_no_grad(self): self.assertFalse(any(v.requires_grad for v in c.compute_support(self.ps.clone().requires_grad_(),self.pd).values()))

    def runner_calls(self):
        source=(Path(__file__).resolve().parents[1]/"tools/run_rddr_phase2b1_dual_hypothesis_audit.py").read_text(encoding="utf-8")
        return [ast.unparse(v.func) for v in ast.walk(ast.parse(source)) if isinstance(v,ast.Call)]

    def test_no_optimizer(self): self.assertFalse(any("optim." in n or n.endswith(".backward") for n in self.runner_calls()))
    def test_no_checkpoint_write(self): self.assertNotIn("torch.save",self.runner_calls())

    def test_no_test_luad_access(self):
        for p in ("/tmp/BCSS-WSSS/test","/tmp/LUAD-HistoSeg/val","/tmp/BCSS-WSSS/training"):
            with self.assertRaises(ValueError): validate_root(p)

    def test_bootstrap_reproducible(self): self.assertTrue(np.array_equal(np.concatenate(list(c.bootstrap_indices(5,80))),np.concatenate(list(c.bootstrap_indices(5,80)))))

    def test_support_against_explicit_neighbors(self):
        for ty,tx in ((0,0),(14,14),(27,10)):
            shallow,deep,ctx=[],[],[]
            for sy in range(max(0,ty-7),min(28,ty+8)):
                for sx in range(max(0,tx-7),min(28,tx+8)):
                    if (ty,tx)==(sy,sx): continue
                    evidence=self.ps[0,:,sy,sx]
                    shallow.append((1-c.phase0_js(self.ps[0,:,ty,tx],evidence,0)/math.log(2)).clamp(0,1))
                    deep.append((1-c.phase0_js(self.pd[0,:,ty,tx],evidence,0)/math.log(2)).clamp(0,1))
                    ctx.append(evidence)
            index=ty*28+tx
            self.assertAlmostEqual(float(torch.stack(shallow).mean()),float(self.r["ss"][0,index]),places=6)
            self.assertAlmostEqual(float(torch.stack(deep).mean()),float(self.r["sd"][0,index]),places=6)
            self.assertTrue(torch.allclose(torch.stack(ctx).mean(0),self.r["ctx"][0,:,index],atol=1e-6))

    def test_winner_population_excludes_agreement_and_both_wrong(self):
        eligible,y=c.winner_labels(np.array([0,0,0,0,4]),np.array([1,0,1,0,1]),np.array([0,1,2,0,0]))
        self.assertTrue(np.array_equal(eligible,[1,1,0,0,0]))
        self.assertTrue(np.array_equal(y[eligible],[1,0]))

    def test_exact_ties_and_absent_labels(self):
        a=c.binary_exact(np.array([0.,0.,1.,2.]),np.array([0,1,0,1]))
        self.assertEqual(a["auroc"],.625)
        self.assertTrue(np.isnan(c.binary_exact([1.,2.],[1,1])["auroc"]))

    def test_sign_metrics_pooled(self):
        m=c.sign_scores(np.array([[60,40],[20,80]]))
        self.assertAlmostEqual(float(m["balanced_accuracy"]),.7)
        self.assertAlmostEqual(float(m["deep_win_recall"]),.8)
        self.assertAlmostEqual(float(m["shallow_win_recall"]),.6)
        self.assertTrue(np.isnan(c.sign_scores(np.array([[0,0],[0,5]]))["balanced_accuracy"]))

    def test_decision_all_cases_and_safety_precedence(self):
        for a in (False,True):
            for b in (False,True):
                for g in (False,True):
                    self.assertEqual(c.decide(a,b,g,False),"ADJUDICATION_DEEP_WRONG_UNSAFE")
                    expected="RDDR_PHASE2B1_NOGO" if not a or not b else "RDDR_PHASE2B1_GO" if g else "ADJUDICATION_EXISTS_FUSION_UTILITY_FAIL"
                    self.assertEqual(c.decide(a,b,g,True),expected)


if __name__=="__main__": unittest.main(verbosity=2)
