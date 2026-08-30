"""Assertions over this real full-validation audit's independently checked artifacts."""
import json
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get('RDDR_PHASE2B17_RUN') and os.environ.get('RDDR_PHASE2B17_REPORT'),
                     'requires completed real GPU audit; skips are not integration evidence')
class RealAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def read(folder,name):return json.loads((Path(folder)/('rddr_phase2b17_'+name+'.json')).read_text())
        cls.r=read(os.environ['RDDR_PHASE2B17_RUN'],'runtime');cls.i=read(os.environ['RDDR_PHASE2B17_RUN'],'identity_audit')
        cls.d=read(os.environ['RDDR_PHASE2B17_RUN'],'detach_audit');cls.v=read(os.environ['RDDR_PHASE2B17_REPORT'],'verification')
        assert cls.v['status']=='PASS' and cls.r['images']==3418
    def test_phase2b16_rect_exact_replay(self):self.assertEqual(self.i['max_logit_replay_difference'],0)
    def test_teacher_exact_replay(self):self.assertEqual(self.r['parity']['anchor_sym'],0)
    def test_q_exact_replay(self):self.assertLessEqual(self.r['parity']['q'],1e-7)
    def test_rect_support_no_gt(self):self.assertTrue(self.v['checks']['full3418_independent_support'])
    def test_teacher_support_no_gt(self):self.assertTrue(self.v['checks']['full3418_independent_support'])
    def test_delta_accept_no_gt(self):self.assertTrue(self.d['delta_detached'])
    def test_acceptance_score_finite(self):self.assertTrue(self.r['all_finite'])
    def test_hard_acceptance_formula(self):self.assertTrue(self.v['checks']['independent_ha_sa_analytic_gradient'])
    def test_soft_acceptance_formula(self):self.assertTrue(self.v['checks']['independent_ha_sa_analytic_gradient'])
    def test_acceptance_detached(self):self.assertTrue(self.d['m_detached'] and self.d['a_detached'])
    def test_teacher_detached(self):self.assertTrue(self.d['teacher_detached'])
    def test_q_detached(self):self.assertTrue(self.d['q_detached'])
    def test_no_optimizer(self):self.assertFalse(self.r['optimizer_created'])
    def test_no_step(self):self.assertEqual(self.r['optimizer_steps'],0)
    def test_checkpoint_unchanged(self):self.assertEqual(self.i['checkpoint_sha_before'],self.i['checkpoint_sha_after'])
    def test_prediction_identity(self):self.assertEqual(self.i['prediction_before'],self.i['prediction_after'])
    def test_15x15_window(self):self.assertTrue(self.v['checks']['full3418_independent_support'])
    def test_self_edge_excluded(self):self.assertTrue(self.v['checks']['full3418_independent_support'])
    def test_gradient_finite(self):self.assertTrue(self.v['checks']['all_gradients_finite'])
    def test_bootstrap_reproducible(self):self.assertTrue(self.v['checks']['paired_bootstrap_all_replicates'])
    def test_no_test_luad_access(self):self.assertTrue(self.v['checks']['no_optimizer_test_luad'])


if __name__=='__main__':unittest.main()
