import ast
import unittest
from pathlib import Path
import numpy as np
import torch
from tools.rddr_phase2b16_common import loss_probe,detached_teacher,margin_direction,bootstrap_indices,decision,PARAMS


class LossTests(unittest.TestCase):
    def test_teacher_detach_q_detach_no_source_gradient(self):
        torch.manual_seed(42)
        s=torch.randn(1,4,28,28,requires_grad=True);d=torch.randn_like(s,requires_grad=True)
        teacher=detached_teacher(s.softmax(1),d.softmax(1))
        self.assertFalse(any(v.requires_grad for v in teacher.values()))
        student=torch.randn_like(s,requires_grad=True)
        loss,_=loss_probe(student,teacher['anchor_sym'],teacher['q'])
        loss.backward()
        self.assertIsNone(s.grad);self.assertIsNone(d.grad)
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(student.grad).all())
        self.assertEqual(student.grad.shape,student.shape)
        # Caller-supplied teacher/q are detached too, independently of builder.
        t=teacher['anchor_sym'].clone().requires_grad_();q=teacher['q'].clone().requires_grad_()
        loss_probe(student,t,q)[0].backward()
        self.assertIsNone(t.grad);self.assertIsNone(q.grad)

    def test_loss_denominator_covers_all_pixels_and_batch(self):
        l=torch.zeros(2,4,3,3,requires_grad=True);t=torch.ones_like(l)/4;q=torch.zeros(2,3,3)
        q[1,2,2]=1;t[1,:,2,2]=torch.tensor([1.,0,0,0])
        for mode in ('CCA','FA'):
            loss,kl=loss_probe(l,t,q,mode)
            self.assertAlmostEqual(loss.item(),kl[1,2,2].item(),places=6)
        loss,kl=loss_probe(l,t,q,'U')
        self.assertAlmostEqual(loss.item(),kl.sum().item()/18,places=7)

    def test_positive_scalar_identity_and_analytic_gradient(self):
        torch.manual_seed(1)
        l=torch.randn(2,4,5,5,requires_grad=True)
        t=torch.randn_like(l).softmax(1);q=torch.rand(2,5,5)
        u=torch.autograd.grad(loss_probe(l,t,q,'U')[0],l)[0]
        g=torch.autograd.grad(loss_probe(l,t,q,'CCA')[0],l)[0]
        self.assertTrue(torch.allclose(g,u*q[:,None]*50/q.sum(),atol=1e-8))
        p=l.softmax(1);a=t*p/(p+1e-8)
        expected=(p*a.sum(1,keepdim=True)-a)*q[:,None]/(q.sum()+1e-8)
        self.assertTrue(torch.allclose(g,expected,atol=1e-8))

    def test_directional_tied_max(self):
        l=np.array([[[0.],[2.],[2.],[1.]]]);g=np.array([[[0.],[1.],[-2.],[0.]]])
        dm,tie=margin_direction(l,g,np.array([[0]]))
        self.assertEqual(dm.item(),-2.);self.assertTrue(tie.item())

    def test_bootstrap_reproducible(self):
        a=list(bootstrap_indices(20,100));b=list(bootstrap_indices(20,100))
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))

    def test_no_optimizer_or_training_api(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'tools/run_rddr_phase2b16_trainability_audit.py').read_text()
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotIn(node.func.attr,('step','SGD','Adam','AdamW','save','save_state_dict','train'))
        self.assertEqual(len(PARAMS),7)

    def test_decision_precedence(self):
        self.assertEqual(decision(False,False,'FAIL',False),'CCA_INTEGRATION_ENGINEERING_NOGO')
        self.assertEqual(decision(True,False,'FAIL',True),'CONFLICT_WEIGHTING_LOCALIZATION_NOT_SUPPORTED')
        self.assertEqual(decision(True,True,'FAIL',True),'TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE')


if __name__=='__main__':unittest.main()
