import inspect
import unittest
from pathlib import Path
import numpy as np
import torch
from tools.rddr_phase2b16_common import EPS,margin_direction,bootstrap_indices
from tools.rddr_phase2b19_common import directional_loss,direction_gate,random_gate,student_logits,conflict_gradient,decide

class FormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.z=torch.randn(2,4,3,3,requires_grad=True);self.p=torch.randn(2,4,3,3).softmax(1).requires_grad_()
        self.q=torch.rand(2,3,3,requires_grad=True);self.d=torch.randn(2,3,3,requires_grad=True);self.r=torch.rand(2,3,3)>.5
    def formula(self,mode):
        p=self.z.softmax(1);t=self.p.detach();q=self.q.detach();d=self.d.detach()
        gate=torch.ones_like(q) if mode=='UDT' else self.r if mode=='RG' else d>0 if mode=='ADT' else d.relu()
        kl=(t*((t+EPS).log()-(p+EPS).log())).sum(1);w=q*gate
        self.assertTrue(torch.equal(directional_loss(self.z,self.p,self.q,self.d,mode,self.r),(kl*w).sum()/(w.sum()+EPS)))
    def test_adt_formula(self):self.formula('ADT')
    def test_udt_formula(self):self.formula('UDT')
    def test_random_loss_formula(self):self.formula('RG')
    def test_soft_directional_formula(self):self.formula('SDT')
    def test_direction_gate_formula(self):self.assertTrue(torch.equal(direction_gate(self.d),self.d>0))
    def test_tie_preserves_shallow(self):self.assertFalse(direction_gate(torch.tensor([0.])).item())
    def test_q_delta_gate_deep_detached(self):
        for mode in ('UDT','RG','ADT','SDT'):
            directional_loss(self.z,self.p,self.q,self.d,mode,self.r).backward()
            self.assertIsNone(self.q.grad);self.assertIsNone(self.d.grad);self.assertIsNone(self.p.grad)
        self.assertFalse(direction_gate(self.d).requires_grad)
    def test_ic1_detached(self):
        raw=torch.randn(2,8,3,3,requires_grad=True);head=torch.nn.Conv2d(8,4,1);z=student_logits(raw,head)
        self.assertTrue(torch.equal(z,head(raw)));directional_loss(z,self.p,self.q,self.d,'ADT').backward()
        self.assertIsNone(head.weight.grad);self.assertIsNone(head.bias.grad);self.assertGreater(raw.grad.abs().max(),0)
    def test_random_gate_rate_match(self):
        a=np.linspace(-1,1,7840).reshape(10,784);r=random_gate(a)
        self.assertTrue(np.array_equal(r.sum(1),(a>0).sum(1)))
    def test_random_gate_seed42_reproducible(self):
        a=np.linspace(-1,1,7840).reshape(10,784);self.assertTrue(np.array_equal(random_gate(a),random_gate(a)))
    def test_rejected_logit_gradient_zero(self):
        g,=torch.autograd.grad(directional_loss(self.z,self.p,self.q,self.d,'ADT'),self.z)
        self.assertTrue(torch.all(g.permute(0,2,3,1)[self.d<=0]==0))
    def test_all_rejected_zero_loss_gradient(self):
        d=-torch.ones_like(self.d);l=directional_loss(self.z,self.p,self.q,d,'ADT');g,=torch.autograd.grad(l,self.z)
        self.assertEqual(l.item(),0);self.assertTrue(torch.all(g==0))
    def test_active_direction_matches_deep_transfer(self):
        p=self.z.detach().double().softmax(1);d=self.p.detach().double();q=self.q.detach().double();m=(self.d.detach()>0).double();w=q*m
        a=p*d/(p+EPS);expected=(p*a.sum(1,keepdim=True)-a)*(w/(w.sum()+EPS))[:,None]
        z=self.z.detach().double().requires_grad_();pp=z.softmax(1);kl=(d*((d+EPS).log()-(pp+EPS).log())).sum(1)
        g,=torch.autograd.grad((w*kl).sum()/(w.sum()+EPS),z);torch.testing.assert_close(g,expected,atol=1e-15,rtol=1e-12)
    def test_gt_margin_tie_direction(self):
        d,t=margin_direction(np.array([[[2.],[3.],[3.],[0.]]]),np.array([[[1.],[2.],[-4.],[0.]]]),np.array([[0]]))
        self.assertEqual(d.item(),-5);self.assertTrue(t.item())
    def test_dq_direction(self):
        _,g=conflict_gradient(self.z,self.p);self.assertTrue(torch.isfinite(g).all());self.assertIsNone(self.z.grad)
    def test_no_gt_in_loss(self):self.assertEqual(list(inspect.signature(directional_loss).parameters),['logits','deep','q','delta','mode','random'])
    def test_zero_threshold_no_search(self):
        self.assertIn('>0',inspect.getsource(direction_gate));self.assertEqual(len(inspect.signature(direction_gate).parameters),1)
    def test_no_optimizer_step_in_runner(self):
        s=(Path(__file__).resolve().parents[1]/'tools/run_rddr_phase2b19_audit.py').read_text()
        for forbidden in ('.step(', 'torch.save(', 'torch.optim.SGD(', 'torch.optim.Adam('):self.assertNotIn(forbidden,s)
    def test_bootstrap_reproducible(self):self.assertTrue(np.array_equal(np.concatenate(list(bootstrap_indices(9,101))),np.concatenate(list(bootstrap_indices(9,101)))))
    def test_decision_precedence(self):
        self.assertEqual(decide(0,0,0,0,0,0,0),'DIRECTIONAL_TRANSFER_ENGINEERING_NOGO')
        self.assertEqual(decide(0,1,1,1,1,1,1),'SYMMETRIC_ADJUDICATION_REPRODUCTION_FAIL')
        for i in (1,2,3,4):
            gates=[True]*7;gates[i]=False;self.assertEqual(decide(*gates),'ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE')
        self.assertEqual(decide(1,1,1,1,1,0,1),'DIRECTIONAL_TRANSFER_NOT_BETTER_THAN_RANDOM_SELECTION')
        self.assertEqual(decide(1,1,1,1,1,1,1),'RDDR_PHASE2B19_DIRECTIONAL_TRANSFER_GO')

if __name__=='__main__':unittest.main()
