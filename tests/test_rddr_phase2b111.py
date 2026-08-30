import inspect
import unittest
from pathlib import Path
import numpy as np
from tools.rddr_phase2b111_common import *

def softmax(z):
    p=np.exp(z-z.max(1,keepdims=True));return p/p.sum(1,keepdims=True)

class FormulaTests(unittest.TestCase):
    def arrays(self):
        p=np.array([[[.8,.8],[.1,.1],[.05,.05],[.05,.05]]],dtype=np.float32)
        d=np.array([[[.1,.1],[.8,.8],[.05,.05],[.05,.05]]],dtype=np.float32)
        t=np.array([[[.1,.4],[.1,.1],[.7,.4],[.1,.1]]],dtype=np.float32)
        return p,d,t,np.array([[False,False]])
    def test_alt_candidate_formula(self):
        c=candidate(*self.arrays());np.testing.assert_array_equal(c['A_alt'],[[True,False]])
    def test_alt_margin_formula(self):
        c=candidate(*self.arrays());self.assertAlmostEqual(float(c['M_alt'][0,0]),.6,places=6)
    def test_margin_zero_rejected(self):
        p,d,t,md=self.arrays();p[:,0,1]=.1;p[:,1,1]=.8;d[:,1,1]=.1;d[:,3,1]=.8
        t[0,:,1]=[.4,.4,.1,.1];c=candidate(p,d,t,md)
        self.assertTrue(c['a_alt'][0,1]);self.assertEqual(c['M_alt'][0,1],0);self.assertFalse(c['A_alt'][0,1])
    def test_no_gt_in_candidate(self):self.assertEqual(list(inspect.signature(candidate).parameters),['ps','pd','ctx','md'])
    def test_no_gt_in_score(self):self.assertEqual(list(inspect.signature(scores).parameters),['ps','pd','ctx','q','delta','c'])
    def test_third_rescue_label(self):
        c=candidate(*self.arrays());m=masks(np.array([[2,0]]),c,np.array([[.2,0]]));self.assertEqual(m['ThirdRescue'].sum(),1)
    def test_alternative_failure_label(self):
        c=candidate(*self.arrays());m=masks(np.array([[0,0]]),c,np.array([[-.2,0]]));self.assertEqual(m['AlternativeFailure'].sum(),1)
    def test_candidate_precision_formula(self):self.assertEqual(float(divide(3,5)),.6)
    def test_hard_repair_harm_formula(self):
        c={'cs':np.array([[0,0,1,1]]),'cd':np.array([[1,1,0,0]]),'cc':np.array([[2,2,2,2]]),'U_R':np.ones((1,4),bool),'A_alt':np.array([[1,1,1,0]],bool)}
        m=masks(np.array([[2,0,3,1]]),c,np.zeros((1,4)))
        self.assertEqual(m['Repair'].sum(),1);self.assertEqual(m['Harm'].sum(),1);self.assertEqual(m['WrongToWrong'].sum(),1);self.assertEqual(m['StableCorrectActivated'].sum(),0)
    def test_context_kl_gradient_formula(self):
        rng=np.random.default_rng(42);z=rng.normal(size=(2,4,5));p=softmax(z);t=softmax(rng.normal(size=z.shape));active=np.ones((2,5),bool)
        _,g=context_gradient(p,t,active);numeric=np.zeros_like(g);h=1e-5
        for idx in np.ndindex(z.shape):
            plus=z.copy();minus=z.copy();plus[idx]+=h;minus[idx]-=h
            kp=(t*(np.log(t+EPS)-np.log(softmax(plus)+EPS))).sum();km=(t*(np.log(t+EPS)-np.log(softmax(minus)+EPS))).sum()
            numeric[idx]=(kp-km)/(2*h)
        np.testing.assert_allclose(g,numeric,rtol=1e-6,atol=1e-9)
    def test_epsilon_is_not_plain_p_minus_t(self):
        p=softmax(np.array([[[-30.],[0.],[0.],[0.]]]));t=np.array([[[1.],[0.],[0.],[0.]]])
        _,g=context_gradient(p,t,np.ones((1,1),bool));self.assertGreater(np.abs(g-(p-t)).max(),.5)
    def test_no_q_weight_or_aggregation(self):self.assertEqual(list(inspect.signature(context_gradient).parameters),['ps','ctx','active'])
    def test_inactive_exact_zero(self):
        p,d,t,md=self.arrays();k,g=context_gradient(p,t,np.array([[False,True]]));self.assertTrue((g[:,:,0]==0).all());self.assertEqual(k[0,0],0)
    def test_gt_margin_tie_rule(self):
        z=np.array([[[0.],[1.],[1.],[0.]]]);g=np.array([[[-2.],[-1.],[-3.],[0.]]])
        self.assertEqual(margin_direction(z,g,np.array([[0]]))[0,0],-1.)
    def test_gradient_utility_label(self):
        r=utility(np.array([[1e-20,-1e-20,0.]]),np.ones((1,3),bool));self.assertEqual((r['beneficial'],r['harmful'],r['zero']),(1,1,1))
    def test_control_directions_frozen(self):
        p,d,t,md=self.arrays();c=candidate(p,d,t,md);q=np.ones(md.shape)*.4;delta=-q;r=scores(p,d,t,q,delta,c)
        np.testing.assert_array_equal(r['q'],q);np.testing.assert_array_equal(r['Delta_sym'],delta)
        np.testing.assert_allclose(r['E_ctx'],(t.astype(float)*np.log(t.astype(float)+EPS)).sum(1))
        np.testing.assert_array_equal(r['D_hier'],1-np.maximum(p.max(1),d.max(1)))
    def test_no_score_substitution(self):self.assertEqual(SCORES,('M_alt','C_ctx','E_ctx','q','Delta_sym','D_hier'))
    def test_per_class_power(self):
        self.assertEqual(class_power(500,500,30),'POWERED');self.assertEqual(class_power(499,500,30),'UNDERPOWERED')
    def test_bootstrap_reproducible(self):np.testing.assert_array_equal(next(bootstrap_indices(10)),next(bootstrap_indices(10)))
    def test_decision_precedence(self):
        self.assertTrue(decide(True,False,True,False,False,'FAIL').endswith('EXISTS_BUT_NOT_SELECTABLE'))
        self.assertTrue(decide(True,True,False,True,True,'PASS').endswith('EXISTS_BUT_NOT_SELECTABLE'))
        self.assertIsNone(decide(True,True,True,True,True,'UNDERPOWERED'))
        self.assertTrue(decide(True,True,True,True,True,'PASS').endswith('GTBLIND_FEASIBILITY_SUPPORTED'))
    def test_underpowered_not_auto_fail(self):
        rr=[dict(power='POWERED',image_auroc=.7)]*2+[dict(power='UNDERPOWERED',image_auroc=.4)]*2
        self.assertEqual(cross_gate(.7,rr),'UNDERPOWERED');self.assertEqual(cross_gate(.5,rr),'FAIL')

if __name__=='__main__':unittest.main()
