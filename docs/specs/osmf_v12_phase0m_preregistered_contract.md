# OSMF-v1.2 Phase-0M Preregistered Contract

This implementation follows
`OSMF_v1.2_Phase0M_Morphology_Objective_Causal_Audit_v1.0.md` without changing
the frozen v1.2 model, objective, loss weights, optimizer, scheduler, data
augmentation, inference, or metric.

- Source specification SHA256: `fe9a38e49ed95b51658b804e7d335dbcace4aefd59679c2db13f039129f1b4c7`
- Source specification length: 1,609 lines
- Frozen v1.2 executed commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Seed: `20260817`
- BCSS training batches: exactly `128`
- Batch/image size: `20 / 224`
- Precision: BF16
- Loss weights: `0.05 / 0.05 / 0.05 / 0.10`

At every eq-active step `4,8,...,128`, the audit measures morphology EqErr on
the exact realized pair before and after the one normal joint optimizer update.
It records realized augmentation flips, pair flip, image identity, and exact
tensor SHA256.

A fixed 64-image training-split probe is selected without GT using a local RNG.
Image IDs, dataset flips, pair transforms, seeds, and input hashes remain fixed.
Raw morphology/semantic EqErr and diagnostic-only 8-neighbor local-affinity
EqErr are measured without gradients at `0,4,8,16,32,64,96,128`.

Morphology-parameter objective-gradient cosines are measured at
`4,8,16,32,64,96,128`. They are observational and never alter optimizer
gradients. The v1.2 safety dynamics are replicated at the original audit steps.

The only permitted primary decisions are:

- `MORPH_EQ_OBJECTIVE_VALID`
- `MORPH_EQ_GENERALIZATION_FAILURE`
- `MORPH_EQ_OBJECTIVE_INVALID`
- `MORPH_EQ_METRIC_MISMATCH_REVIEW`

No checkpoint continuation, validation performance, test, LUAD, other seed,
three-epoch pilot, 25-epoch training, loss change, or v1.3 implementation is
authorized.
