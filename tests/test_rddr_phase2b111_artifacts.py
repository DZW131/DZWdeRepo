import json
import os
from pathlib import Path
import unittest

class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get('RDDR_PHASE2B111_RUN')
        if not path:raise unittest.SkipTest('Set RDDR_PHASE2B111_RUN to verified artifacts')
        root=Path(path);cls.ver=json.loads((root/'rddr_phase2b111_verification.json').read_text());cls.rt=json.loads((root/'rddr_phase2b111_runtime.json').read_text())
    def test_no_optimizer(self):self.assertFalse(self.rt['optimizer_created'])
    def test_no_step(self):self.assertEqual(self.rt['optimizer_steps'],0)
    def test_no_forward_if_cache_complete(self):self.assertFalse(self.rt['model_instantiated'] or self.rt['network_forward'])
    def test_no_backward(self):self.assertFalse(self.rt['backward'] or self.rt['autograd'])
    def test_no_checkpoint_write(self):self.assertFalse(self.rt['checkpoint_written'])
    def test_no_test_luad(self):self.assertFalse(self.rt['test_access'] or self.rt['luad_access'] or self.rt['training_split_access'])
    def test_no_threshold_search(self):self.assertFalse(self.rt['threshold_search'] or self.rt['new_gate_design'])
    def test_no_classifier_fit(self):self.assertFalse(self.rt['classifier_fit'] or self.rt['score_fusion'])

def make(key):
    def check(self):self.assertTrue(self.ver['checks'][key],key)
    return check
for key in ('input_sha_unchanged','full3418_order','mD_exact_replay','q_exact_replay','phase2b110_counts_exact_replay','legacy_third_rescue',
    'candidate_counts_and_ties','candidate_composition_partition','hard_repair_harm_formula','context_kl_gradient_formula','gradient_labels_exact',
    'real128_finite_difference','all_three_task_rankings','controls_frozen_no_substitution','gradient_and_full_active_protection',
    'q_strata_replay','boundary_replay','per_class_power','bootstrap_reproducible','bootstrap_denominators_and_CIs','independent_gates_decision','secondary_flags','identity_unchanged','original_A0_unchanged'):
    setattr(ArtifactTests,'test_'+key,make(key))
ArtifactTests.test_ctx_exact_replay=make('ctx_exact_and_independent_replay')
