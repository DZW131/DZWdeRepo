import inspect
import itertools
import unittest
from pathlib import Path
import numpy as np
from tools.rddr_phase2b110_common import *

class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.ps=np.array([[[.1,.4],[.2,.3],[.3,.2],[.4,.1]]],np.float32);self.pd=self.ps[:,:,::-1].copy()
        self.q=np.array([[.2,.3]],np.float32);self.ts=[np.array([[.1+i*.1,.2+i*.1]],np.float32) for i in range(4)]
    def test_no_gt_in_scores(self):self.assertEqual(list(inspect.signature(frozen_scores).parameters),['ps','pd','q','tss','tsd','tds','tdd'])
    def test_primary_score_is_sD(self):
        s=frozen_scores(self.ps,self.pd,self.q,*self.ts);self.assertEqual(tuple(s),SCORES);np.testing.assert_array_equal(s['S_D_sym'],.5*(self.ts[2]+self.ts[3]))
    def test_control_score_directions_frozen(self):
        s=frozen_scores(self.ps,self.pd,self.q,*self.ts)
        np.testing.assert_array_equal(s['Delta_sym'],.5*(self.ts[2]+self.ts[3])-.5*(self.ts[0]+self.ts[1]))
        np.testing.assert_array_equal(s['q'],self.q);np.testing.assert_array_equal(s['deep_confidence_advantage'],self.pd.max(1)-self.ps.max(1))
        a=self.ps.astype(float);b=self.pd.astype(float)
        np.testing.assert_array_equal(s['deep_entropy_advantage'],-(a*np.log(a+EPS)).sum(1)+(b*np.log(b+EPS)).sum(1))
    def test_required_gap_exact_formula(self):
        for n in range(1,100):
            for b in range(n+1):self.assertEqual(required_gap(n,b),int(np.ceil(n*.4))-b)
        self.assertEqual(required_gap(708407,252097),31266)
    def pop(self):
        y=np.array([[0,1,2]]);s=np.eye(4)[[1,0,3]].T[None];d=np.eye(4)[[0,3,2]].T[None]
        return population(y,s,d,np.zeros((1,3),bool),np.array([[1.,-1.,0.]]))
    def test_residual_population_formula(self):self.assertEqual(self.pop()['R_RW'].sum(),3)
    def test_residual_beneficial_label(self):np.testing.assert_array_equal(self.pop()['Residual_Beneficial'],[[True,False,False]])
    def test_residual_harmful_label(self):np.testing.assert_array_equal(self.pop()['Residual_Harmful'],[[False,True,False]])
    def test_rejected_bothwrong_formula(self):np.testing.assert_array_equal(self.pop()['Rejected_Both_Wrong'],[[False,True,False]])
    def test_margin_tie_direction(self):
        v=margin_direction(np.array([[[2.],[3.],[3.],[0.]]]),np.array([[[1.],[2.],[-4.],[0.]]]),np.array([[0]]));self.assertEqual(v.item(),-5)
    def test_binary_tied_scores(self):
        r=binary(np.ones(4),[1,0,1,0]);self.assertEqual(r['auroc'],.5);self.assertEqual(r['auprc'],.5)
        r=binary(np.array([1,2,3]),[0,0,1]);self.assertEqual(r['auroc'],1);self.assertEqual(r['auprc'],1)
    def test_quintile_ties_not_split(self):
        a=np.ones((1,20));edges,m=diagnostic_quintiles(a,np.ones_like(a,bool));self.assertEqual(m[0].sum(),20);self.assertEqual(sum(x.sum() for x in m[1:]),0)
    def test_third_class_rescue_formula(self):
        p=np.eye(4)[[2,1]].T[None];y=np.array([[2,2]]);s=np.array([[0,0]]);d=np.array([[1,1]]);m=np.ones((1,2),bool)
        r=context_metrics(p,y,s,d,m);self.assertEqual(r['accuracy'],.5);self.assertEqual(r['rescue_rate'],.5);self.assertEqual(r['rescue_precision'],1)
    def test_cross_stratum_underpowered(self):
        c=[dict(power='POWERED',image_auroc=.7)]*2+[dict(power='UNDERPOWERED',image_auroc=.9)]*2
        self.assertEqual(cross_stratum(.7,c),'UNDERPOWERED');self.assertEqual(cross_stratum(.5,c),'FAIL')
    def test_decision_all_cases(self):
        for a,b,c in itertools.product((False,True),repeat=3):
            for d in ('PASS','FAIL','UNDERPOWERED'):
                for third in (False,True):
                    r=decide(a,b,c,d,third)
                    expected='RESIDUAL_COVERAGE_HEADROOM_INSUFFICIENT' if not a else ('DUAL_RESIDUAL_RECOVERY_SIGNAL_SUPPORTED' if third else 'RESIDUAL_DEEP_RECOVERY_SIGNAL_SUPPORTED') if b and c and d=='PASS' else ('RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED' if third else 'RESIDUAL_COVERAGE_NOT_RECOVERABLE_WITH_FROZEN_EVIDENCE')
                    self.assertEqual(r,expected)
    def test_no_threshold_search(self):
        text=(Path(__file__).resolve().parents[1]/'tools/run_rddr_phase2b110_audit.py').read_text()
        for term in ('GridSearch','LogisticRegression','MLPClassifier','torch.save(','.step(','.backward(','Net('):self.assertNotIn(term,text)
    def test_no_gate_construction(self):self.assertNotIn('gate',inspect.getsource(frozen_scores).split('return')[1])
    def test_bootstrap_reproducible(self):np.testing.assert_array_equal(next(bootstrap_indices(9)),next(bootstrap_indices(9)))

if __name__=='__main__':unittest.main()
