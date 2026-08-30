"""Required integration assertions against the full real audit, never synthetic substitutes."""
import inspect
import json
import os
import unittest
from pathlib import Path
from tools.rddr_phase2b18_common import guidance_loss

RUN=os.environ.get('RDDR_PHASE2B18_RUN')
REPORT=os.environ.get('RDDR_PHASE2B18_REPORT')
P='rddr_phase2b18_'


@unittest.skipUnless(RUN and REPORT,'set both full-audit artifact paths')
class RealArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def read(folder,name):return json.loads((Path(folder)/(P+name+'.json')).read_text())
        cls.rt=read(RUN,'runtime');cls.dt=read(RUN,'detach_audit');cls.ident=read(RUN,'identity_audit')
        cls.smoke=read(RUN,'bf16_smoke');cls.v=read(REPORT,'verification');cls.summary=read(REPORT,'summary')
        assert cls.rt['images']==3418 and cls.v['status']=='PASS'

    def test_phase2b15_teacher_exact_replay(self):self.assertEqual(self.rt['parity']['teacher'],0)
    def test_raw_logits_exact_replay(self):
        self.assertEqual(self.rt['parity']['raw_frozen_head_logits'],0)
        self.assertEqual(self.ident['raw_gradient_replay_max_abs'],0)
        self.assertTrue(self.v['checks']['raw_probability_exact'])
    def test_q_exact_replay(self):self.assertLessEqual(self.rt['parity']['q'],1e-7)
    def test_teacher_detached(self):self.assertTrue(self.dt['teacher_detached'])
    def test_q_detached(self):self.assertTrue(self.dt['q_detached'])
    def test_deep_path_detached(self):self.assertTrue(self.dt['deep_source_detached'] and self.smoke['deep_source_grad_none'])
    def test_primary_ic1_detached(self):self.assertTrue(self.dt['primary_ic1_none'] and self.v['checks']['frozen_head_zero'])
    def test_hfrm28_1_no_gradient(self):self.assertTrue(self.dt['hfrm_none'])
    def test_upstream_gradient_nonzero(self):self.assertTrue(self.v['checks']['each_b4_conv_group_active'])
    def test_prg_formula(self):self.assertTrue(self.v['checks']['independent_exact_FP32_losses_gradients'])
    def test_uraw_formula(self):self.assertEqual(self.v['errors']['FP32_loss'],0)
    def test_faraw_formula(self):self.assertTrue(self.v['checks']['FP64_analytic_KL_and_JS'])
    def test_gt_margin_direction(self):self.assertTrue(self.v['checks']['all_strata_margin_hierarchy'])
    def test_q_directional_derivative(self):self.assertTrue(self.v['checks']['independent_exact_FP32_q_derivative'])
    def test_collapse_cosine_finite(self):self.assertTrue(self.v['checks']['all_observation_gradients_finite'] and self.v['checks']['all_strata_margin_hierarchy'])
    def test_brr_hhcr_labels_gt_only(self):self.assertTrue(self.v['checks']['BRR_HHCR_denominators'])
    def test_no_gt_in_loss(self):self.assertEqual(list(inspect.signature(guidance_loss).parameters),['logits','teacher','fixed','q','mode'])
    def test_batch20_bf16_backward(self):self.assertTrue(self.v['checks']['BF16_batch20'])
    def test_no_optimizer(self):self.assertFalse(self.rt['optimizer_created'])
    def test_no_step(self):self.assertEqual(self.rt['optimizer_steps'],0)
    def test_checkpoint_identity(self):self.assertTrue(self.v['checks']['state_bn_checkpoint_identity'])
    def test_inference_identity(self):self.assertTrue(self.v['checks']['official_prediction_identity'])
    def test_no_test_luad(self):self.assertFalse(self.rt['test_access'] or self.rt['luad_access'] or self.rt['training_split_access'])
    def test_bootstrap_reproducible(self):self.assertTrue(self.v['checks']['10000_paired_image_bootstrap'])


if __name__=='__main__':unittest.main()
