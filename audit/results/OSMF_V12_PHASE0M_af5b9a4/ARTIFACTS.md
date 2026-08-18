# OSMF-v1.2 Phase-0M Artifact Manifest

Executed commit: `af5b9a431e30d26bec36c024447e1b0af93cc197`

## Decision

- Primary: `MORPH_EQ_OBJECTIVE_INVALID`
- Secondary: `SAME_PAIR_CAUSAL_INVALID`
- Later experiments and v1.3: not started

## Integrity

- `audit/summary.json`: `4645ef10ac803f3be0e4131f253edecb1f851bae71877be559adadacbe08e8a5`
- `phase0m.log`: `a279b85628903dda8e22c2c33cd88b1fb7312c82abda7a2205489c6303198372`
- same-pair causal table: `3802d0b4e3b0ec0b03de47a6c1374035f334009f73b7bbb12da1dceac99bf11c`
- fixed-probe manifest: `2ea6dce9309b6c18ca79eae4a9903440e32a8bba0c92ec785300e98679aeb260`
- fixed raw EqErr: `46007c9cdfc5bb6d5b8abebbf834b59b997b9d60ac6c642bb4eb8fdb55f83e5d`
- fixed affinity EqErr: `4a336467e1559c1f7a33ac87294c89ceaf59d1e1243b4672b8ce3264efbb8553`
- morphology gradient cosine: `b9eeea99c2391cc3b5babb8d3879989949147fe0d56a7612b956ec1380ff1047`
- generated audit report: `7f1f870c4e31918cfca09d658187a64a5a47699ebac57a065b8d1d2fc2e5631d`

## Entry points

- [Summary](audit/summary.json)
- [Generated audit report](audit/docs/osmf_v12_phase0m_morphology_objective_causal_audit.md)
- [Same-pair causal rows](audit/tables/same_pair_causal.csv)
- [Same-pair summary](audit/tables/same_pair_summary.csv)
- [Training-pair manifest](audit/tables/training_pair_manifest.csv)
- [Fixed-probe manifest](audit/tables/fixed_probe_manifest.csv)
- [Fixed raw EqErr trajectory](audit/tables/fixed_probe_raw_eq.csv)
- [Fixed affinity trajectory](audit/tables/fixed_probe_affinity_eq.csv)
- [Morphology gradient competition](audit/tables/morphology_gradient_cosine.csv)
- [Safety replication](audit/tables/gradient_budget_replication.csv)
- [Representation health](audit/tables/representation_health.csv)
- [Console log](phase0m.log)

All files are raw downloads from the formal immutable server run. No validation
performance, test data, LUAD data, segmentation ground truth, checkpoint
selection, or hyperparameter tuning was used.
