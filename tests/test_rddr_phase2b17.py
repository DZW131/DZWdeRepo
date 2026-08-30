import ast
import inspect
import unittest
from pathlib import Path
import numpy as np
import torch
from tools.rddr_phase2b17_common import acceptance_support,acceptance_loss,rank_metrics,decide
from tools.rddr_phase2b16_common import bootstrap_indices


class AcceptanceUnitTests(unittest.TestCase):
    def test_support_no_gt_and_finite(self):
        self.assertEqual(tuple(inspect.signature(acceptance_support).parameters),('ps','pd','rect','teacher'))
        v=torch.ones(1,4,28,28)/4
        result=acceptance_support(v,v,v,v)
        self.assertTrue(all(torch.isfinite(x).all() for x in result.values()))
        self.assertTrue(torch.equal(result['delta'],torch.zeros_like(result['delta'])))
        self.assertTrue(torch.equal(result['R_S'],torch.ones_like(result['R_S'])))

    def test_15x15_window_self_edge_excluded(self):
        v=torch.ones(1,4,28,28)/4;s=v.clone()
        s[:,:,14,14]=torch.tensor([1.,0,0,0])
        a=acceptance_support(v,v,v,v);b=acceptance_support(s,v,v,v)
        idx=lambda y,x:y*28+x
        self.assertEqual(a['R_S'][0,idx(14,14)],b['R_S'][0,idx(14,14)])
        self.assertLess(b['R_S'][0,idx(14,21)],a['R_S'][0,idx(14,21)])
        self.assertEqual(b['R_S'][0,idx(14,22)],a['R_S'][0,idx(14,22)])

    def test_ha_sa_formula_detach_and_gradient_shape(self):
        torch.manual_seed(42)
        L=torch.randn(2,4,3,3,requires_grad=True);teacher=torch.rand_like(L).softmax(1).detach().requires_grad_()
        q=torch.rand(2,3,3,requires_grad=True);delta=torch.linspace(-1,1,18).reshape(2,3,3).requires_grad_()
        for mode in ('HA','SA'):
            loss,kl=acceptance_loss(L,teacher,q,delta,mode)
            weight=q.detach()*((delta.detach()>0) if mode=='HA' else delta.detach().relu())
            expected=(weight*kl).sum()/(weight.sum()+1e-8)
            self.assertEqual(loss.item(),expected.item())
            loss.backward();self.assertIsNone(teacher.grad);self.assertIsNone(q.grad);self.assertIsNone(delta.grad)
            self.assertEqual(L.grad.shape,L.shape);self.assertTrue(torch.isfinite(L.grad).all())
            self.assertTrue((L.grad.permute(0,2,3,1)[delta.detach()<=0]==0).all());L.grad=None

    def test_all_rejected_has_exact_zero_loss_gradient(self):
        L=torch.randn(1,4,3,3,requires_grad=True);teacher=torch.randn_like(L).softmax(1)
        for mode in ('HA','SA'):
            loss,_=acceptance_loss(L,teacher,torch.ones(1,3,3),-torch.ones(1,3,3),mode)
            loss.backward();self.assertEqual(loss.item(),0.);self.assertTrue((L.grad==0).all());L.grad=None

    def test_no_optimizer_step_or_checkpoint_write(self):
        p=Path(__file__).resolve().parents[1]/'tools/run_rddr_phase2b17_acceptance_audit.py'
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('step','SGD','Adam','AdamW','save','train'))

    def test_rank_ties_and_degenerate_classes(self):
        m=rank_metrics(np.array([0.,0.,1.,1.]),np.array([0,1,0,1]))
        self.assertEqual(m['auroc'],.5);self.assertEqual(m['auprc'],.5)
        self.assertTrue(np.isnan(rank_metrics(np.ones(3),np.ones(3))['auroc']))

    def test_bootstrap_reproducible(self):
        a=list(bootstrap_indices(3418,100));b=list(bootstrap_indices(3418,100))
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))

    def test_decision_precedence(self):
        self.assertEqual(decide(False,False,'FAIL',False),'CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED')
        self.assertEqual(decide(True,True,'PASS',False),'ACCEPTANCE_PROTECTION_CAPACITY_FAIL')
        self.assertEqual(decide(True,True,'UNDERPOWERED',True),'ACCEPTANCE_SIGNAL_CLASS_SAFETY_UNDERPOWERED')
        self.assertEqual(decide(True,True,'PASS',True,False),'ACCEPTANCE_AUDIT_ENGINEERING_NOGO')


if __name__=='__main__':unittest.main()
