"""Required integration tests against real frozen full-validation audit evidence."""
import json
import os
import unittest
from pathlib import Path
RUN=os.environ.get('RDDR_PHASE2B110_RUN');P='rddr_phase2b110_'

@unittest.skipUnless(RUN,'set completed real audit directory; skipped is not validated')
class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def read(name):return json.loads((Path(RUN)/(P+name+'.json')).read_text())
        cls.rt=read('runtime');cls.s=read('summary');cls.v=read('verification');cls.ident=read('identity_audit')
        assert cls.rt['images']==3418 and cls.v['status']=='PASS'
    def test_no_optimizer(self):self.assertFalse(self.rt['optimizer_created'])
    def test_no_step(self):self.assertEqual(self.rt['optimizer_steps'],0)
    def test_no_checkpoint(self):self.assertFalse(self.rt['checkpoint_written'])
    def test_no_test_luad(self):self.assertFalse(self.rt['test_access'] or self.rt['luad_access'] or self.rt['training_split_access'])
    def test_no_gate_construction(self):self.assertFalse(self.rt['new_recovery_gate'])
    def test_no_threshold_search(self):self.assertFalse(self.rt['threshold_search'])

CHECKS={
 'phase2b19_gate_exact_replay':'sD_delta_gate_exact',
 'phase2b19_rawwrong_count':'phase2b19_frozen_counts',
 'phase2b19_rejected_counts':'phase2b19_frozen_counts',
 'required_gap_exact_formula':'exact_integer_gap',
 'sD_exact_replay':'sD_delta_gate_exact',
 'delta_exact_replay':'sD_delta_gate_exact',
 'q_exact_replay':'q_exact_cache_replay',
 'residual_population_formula':'residual_partition',
 'residual_beneficial_label':'counts_all_populations',
 'residual_harmful_label':'counts_all_populations',
 'primary_score_is_sD':'five_scores_no_substitution',
 'ctx_sym_exact_replay':'ctx_exact_primary_and_independent',
 'rejected_bothwrong_formula':'residual_partition',
 'third_class_rescue_formula':'third_rescue_identity',
 'bootstrap_reproducible':'10000_paired_image_bootstrap',
 'identity_unchanged':'identity_evidence_honest',
 'all_ranking_metrics':'all_primary_control_rankings',
 'headroom_denominator':'headroom_bootstrap_denominator',
 'class_power':'cross_stratum_and_power',
 'all_gates_decision':'independent_gates_decision',
 'secondary_flags':'independent_secondary_flags',
}
def make(key):
    def check(self):self.assertTrue(self.v['checks'][key],key)
    return check
for name,key in CHECKS.items():setattr(ArtifactTests,'test_'+name,make(key))

if __name__=='__main__':unittest.main()
