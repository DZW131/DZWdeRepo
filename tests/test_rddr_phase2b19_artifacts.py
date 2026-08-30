"""Full3418 integration evidence; absent artifacts SKIP, never synthetic PASS."""
import json
import os
import unittest
from pathlib import Path
P='rddr_phase2b19_'
RUN=os.environ.get('RDDR_PHASE2B19_RUN');REPORT=os.environ.get('RDDR_PHASE2B19_REPORT')

@unittest.skipUnless(RUN and REPORT,'set real run and report paths')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def read(root,name):return json.loads((Path(root)/(P+name+'.json')).read_text())
        cls.rt=read(RUN,'runtime');cls.dt=read(RUN,'detach_audit');cls.smoke=read(RUN,'bf16_smoke')
        cls.ident=read(RUN,'identity_audit');cls.v=read(REPORT,'verification')
        assert cls.rt['images']==3418 and cls.v['status']=='PASS'
    def test_q_detached(self):self.assertTrue(self.dt['q_detached'])
    def test_delta_detached(self):self.assertTrue(self.dt['delta_detached'])
    def test_gate_detached(self):self.assertTrue(self.dt['gate_detached'] and self.smoke['gate_detached'])
    def test_deep_target_detached(self):self.assertTrue(self.dt['deep_source_detached'] and self.smoke['deep_source_grad_none'])
    def test_ic1_detached(self):self.assertTrue(self.dt['primary_ic1_none'])
    def test_hfrm_no_grad(self):self.assertTrue(self.dt['hfrm_none'])
    def test_no_optimizer(self):self.assertFalse(self.rt['optimizer_created'])
    def test_no_step(self):self.assertEqual(self.rt['optimizer_steps'],0)
    def test_bn_identity(self):self.assertEqual(self.ident['bn_before'],self.ident['bn_after'])

CHECKS={
 'phase2b15_delta_sym_exact_replay':'phase2b15_delta_exact',
 'phase2b18_raw_logits_exact_replay':'phase2b18_raw_exact',
 'q_exact_replay':'q_frozen_replay',
 'upstream_grad_nonzero':'each_b4_conv_group_active',
 'adt_formula':'four_losses_exact_FP32',
 'udt_formula':'four_losses_exact_FP32',
 'random_gate_rate_match':'random_per_image_rate_match',
 'random_gate_seed42_reproducible':'random_seed42_exact',
 'soft_directional_formula':'four_losses_exact_FP32',
 'rejected_logit_gradient_zero':'rejected_logit_feature_dQ_zero',
 'active_direction_matches_deep_transfer':'active_direction_analytic_FP64',
 'gt_margin_tie_direction':'all_strata_margin_hierarchy',
 'dq_direction':'q_gradient_exact_FP32',
 'brr_definition':'BRR_HHCR_all_denominators',
 'hhcr_definition':'BRR_HHCR_all_denominators',
 'checkpoint_identity':'state_bn_checkpoint_identity',
 'prediction_identity':'official_prediction_identity',
 'batch20_bf16':'BF16_batch20',
 'no_test_luad':'no_optimizer_test_luad',
 'bootstrap_reproducible':'10000_paired_image_bootstrap',
 'all_gates_independent':'independent_gates_decision',
 'class_power':'all_adjudication_and_power',
 'original_a0_untouched':'original_sources_unchanged',
}
def make_test(key):
    def check(self):self.assertTrue(self.v['checks'][key],key)
    return check
for name,key in CHECKS.items():setattr(ArtifactTests,'test_'+name,make_test(key))

if __name__=='__main__':unittest.main()
