import inspect
import unittest
import numpy as np
import torch
from tools.rddr_phase2b16_common import js,EPS,margin_direction,bootstrap_indices,detached_teacher
from tools.rddr_phase2b18_common import guidance_loss,student_logits,conflict_gradient,hierarchy,decide,groups


class FormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.l=torch.randn(2,4,3,3,requires_grad=True)
        self.t=torch.randn(2,4,3,3).softmax(1).requires_grad_()
        self.f=torch.randn(2,4,3,3).softmax(1).requires_grad_()
        self.q=torch.rand(2,3,3,requires_grad=True)

    def check_formula(self,mode):
        teacher=self.f.detach() if mode=='FAraw' else self.t.detach()
        kl=(teacher*((teacher+EPS).log()-(self.l.softmax(1)+EPS).log())).sum(1)
        wanted=kl.mean() if mode=='Uraw' else (kl*self.q.detach()).sum()/(self.q.detach().sum()+EPS)
        actual=guidance_loss(self.l,self.t,self.f,self.q,mode)
        self.assertTrue(torch.equal(actual,wanted))

    def test_prg_formula(self):self.check_formula('PRG')
    def test_uraw_formula(self):self.check_formula('Uraw')
    def test_faraw_formula(self):self.check_formula('FAraw')

    def test_teacher_q_fixed_detached(self):
        for mode in ('PRG','Uraw','FAraw'):
            self.l.grad=None
            guidance_loss(self.l,self.t,self.f,self.q,mode).backward()
            self.assertIsNone(self.t.grad);self.assertIsNone(self.f.grad);self.assertIsNone(self.q.grad)
            self.assertGreater(self.l.grad.abs().max(),0)

    def test_primary_ic1_detached(self):
        raw=torch.randn(2,8,3,3,requires_grad=True);head=torch.nn.Conv2d(8,4,1)
        original=head(raw);frozen=student_logits(raw,head)
        self.assertTrue(torch.equal(original,frozen))
        guidance_loss(frozen,self.t,self.f,self.q).backward()
        self.assertIsNone(head.weight.grad);self.assertIsNone(head.bias.grad)
        self.assertGreater(raw.grad.abs().max(),0)

    def test_shared_head_same_feature_derivative(self):
        raw=torch.randn(2,8,3,3,requires_grad=True);head=torch.nn.Conv2d(8,4,1)
        a=guidance_loss(student_logits(raw,head),self.t,self.f,self.q)
        b=guidance_loss(student_logits(raw,head,True),self.t,self.f,self.q)
        ga,=torch.autograd.grad(a,raw);gb,gh=torch.autograd.grad(b,(raw,head.weight))
        self.assertTrue(torch.equal(ga,gb));self.assertGreater(gh.abs().max(),0)

    def test_gt_margin_direction(self):
        l=np.array([[[2.],[3.],[3.],[0.]]]);g=np.array([[[1.],[2.],[-4.],[0.]]]);y=np.array([[0]])
        dm,ties=margin_direction(l,g,y)
        self.assertEqual(dm.item(),-5.);self.assertTrue(ties.item())

    def test_q_directional_derivative(self):
        import math
        d=self.t.detach();q,g=conflict_gradient(self.l,d)
        p=self.l.detach().double().softmax(1);dd=d.double();m=.5*(p+dd)
        h=.5*((p+EPS).log()-(m+EPS).log()+p/(p+EPS)-.5*(p+dd)/(m+EPS))/math.log(2)
        exact=p*(h-(p*h).sum(1,keepdim=True))
        torch.testing.assert_close(g.double(),exact,atol=8e-8,rtol=5e-6)
        self.assertFalse(q.requires_grad);self.assertIsNone(self.l.grad)

    def test_collapse_cosine_finite(self):
        a=np.ones((1,4,2));b=np.ones_like(a)
        dq,cos=hierarchy(a,b)
        self.assertTrue(np.all(dq<0));self.assertTrue(np.all(cos<0));self.assertTrue(np.isfinite(cos).all())
        dq,cos=hierarchy(a,np.zeros_like(b));self.assertTrue(np.all(cos==0))

    def test_no_gt_in_loss(self):
        self.assertEqual(list(inspect.signature(guidance_loss).parameters),['logits','teacher','fixed','q','mode'])

    def test_no_optimizer_or_step(self):
        from pathlib import Path
        p=Path(__file__).resolve().parents[1]/'tools/run_rddr_phase2b18_audit.py'
        s=p.read_text();self.assertNotIn('.step(',s);self.assertNotIn('torch.save(',s)

    def test_bootstrap_reproducible(self):
        a=np.concatenate(list(bootstrap_indices(9,101,42)));b=np.concatenate(list(bootstrap_indices(9,101,42)))
        self.assertTrue(np.array_equal(a,b));self.assertEqual(a.shape,(101,9))

    def test_gate_precedence(self):
        self.assertEqual(decide(1,1,1,1,0),'PRERECT_GUIDANCE_ENGINEERING_NOGO')
        self.assertEqual(decide(0,1,1,1,1),'SYMMETRIC_TEACHER_NOT_SUITABLE_FOR_RAW')
        self.assertEqual(decide(1,0,1,1,1),'TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE')
        self.assertEqual(decide(1,1,1,0,1),'PRERECT_GUIDANCE_HIERARCHY_COLLAPSE_RISK')
        self.assertEqual(decide(1,1,1,1,1),'RDDR_PHASE2B18_PRERECT_GUIDANCE_GO')


if __name__=='__main__':unittest.main()
