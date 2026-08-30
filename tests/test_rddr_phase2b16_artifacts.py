"""Integration assertions over the independently verified real-GPU audit, not synthetic substitutes."""
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('RDDR_PHASE2B16_RUN') and os.environ.get('RDDR_PHASE2B16_REPORT'),
                     'set RUN/REPORT to completed real audit; never count a skip as integration PASS')
class IntegrationArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run=Path(os.environ['RDDR_PHASE2B16_RUN']);report=Path(os.environ['RDDR_PHASE2B16_REPORT'])
        def read(root,name):return json.loads((root/('rddr_phase2b16_'+name+'.json')).read_text())
        cls.v=read(report,'verification');cls.r=read(run,'runtime');cls.i=read(run,'identity_audit');cls.b=read(run,'bf16_smoke');cls.d=read(run,'detach_audit')
        assert cls.v['status']=='PASS' and cls.r['images']==3418

    def test_phase2b15_teacher_exact_reproduction(self):self.assertTrue(self.v['checks']['teacher_parity'])
    def test_q_exact_reproduction(self):self.assertLessEqual(self.r['parity_max_abs']['q'],1e-7)
    def test_teacher_detached(self):self.assertTrue(self.d['teacher_detached'])
    def test_q_detached(self):self.assertTrue(self.d['q_detached'])
    def test_no_teacher_gradient(self):self.assertTrue(self.b['ps_teacher_grad_none'] and self.b['pd_teacher_grad_none'])
    def test_no_q_gradient(self):self.assertTrue(self.d['q_source_grad_none'])
    def test_loss_finite(self):self.assertTrue(self.r['numerical_stability']['all_finite'])
    def test_gradient_finite(self):self.assertTrue(self.v['checks']['all_gradients_finite'])
    def test_logit_gradient_shape(self):self.assertTrue(self.v['checks']['analytical_logit_gradient'])
    def test_feature_gradient_nonzero(self):self.assertTrue(self.v['checks']['feature_gradient_nonzero'])
    def test_context_branch_gradient_nonzero(self):self.assertGreater(self.b['parameter_gradients'][0]['sumsq'],0)
    def test_semantic_branch_gradient_nonzero(self):self.assertGreater(self.b['parameter_gradients'][1]['sumsq']+self.b['parameter_gradients'][2]['sumsq'],0)
    def test_ic1_gradient_nonzero(self):self.assertGreater(self.b['parameter_gradients'][5]['sumsq'],0)
    def test_no_optimizer_step(self):self.assertTrue(self.v['checks']['zero_update_state_identity'])
    def test_checkpoint_unchanged(self):self.assertTrue(self.v['checks']['checkpoint_unchanged'])
    def test_inference_off_exact_equivalence(self):self.assertTrue(self.v['checks']['inference_off_exact'])
    def test_batch20_bf16_backward(self):self.assertTrue(self.v['checks']['batch20_bf16_backward'])
    def test_no_test_luad_access(self):self.assertTrue(self.v['checks']['no_test_luad_access'])
    def test_bootstrap_reproducible(self):self.assertTrue(self.v['checks']['bootstrap_reproducible'])


if __name__=='__main__':unittest.main()
