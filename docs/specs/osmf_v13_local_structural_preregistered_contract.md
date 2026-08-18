# OSMF-v1.3 Local Structural Morphology Learning — Preregistered Contract

This repository record freezes the user-supplied v1.3 technical specification.

- Scientific parent: frozen OSMF-v1.2 factorizer and its semantic, orthogonality, and reconstruction objectives.
- Sole scientific change: replace pointwise morphology equivariance training loss with channel-normalized, masked, direction-aware 8-neighbour local-affinity SmoothL1 (`beta=1.0`).
- Weights: semantic `0.05`, structural `0.05`, orthogonality `0.05`, reconstruction `0.10`.
- Structural interval: every 4 optimizer steps with the existing alternating flip schedule.
- Stage A: full BCSS validation exact parity; only `OSMF_V13_PARITY_PASS` unlocks Stage B.
- Stage B: fresh A0 restart, 8 real BCSS batches; audit steps `0,1,2,4,8`; only `OSMF_V13_READINESS_PASS` unlocks Stage C.
- Stage C: another fresh A0 restart, exactly 128 real BCSS batches; same-pair causal checks every structural step and fixed 64-image GT-free probe at `0,4,8,16,32,64,96,128`.
- Hard boundary: no checkpoint save, no test/LUAD, no segmentation-GT use, no 3-epoch or 25-epoch training, no v1.4, and no tuning.

The complete authoritative specification remains the attached source document `OSMF_v1.3_Local_Structural_Morphology_Learning_Technical_Spec_v1.0.md`; this file is an execution-focused repository mirror, not a replacement.
