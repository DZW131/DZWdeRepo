# OSMF-v1.2 Preregistered Gradient-Budget Contract

This repository implementation follows
`OSMF_v1.2_Conservative_Gradient_Budgeted_Specialization_Technical_Spec_v1.0.md`.

- Source specification SHA256: `c110d2a638f553941ce6f6c19f3657676b07bb418242f6520cbacf4da7b2039b`
- Source specification length: 1,215 lines
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Frozen OSMF-v1.1 executed commit: `35591791e0bd81edaf53183afbf319358ccb7b81`
- Development parent including the v1.1 archive: `2c30fd67cb1ab1c33d6ed26593fdbac00054e74f`

## Only scientific change

| Objective | v1.1 | v1.2 |
|---|---:|---:|
| semantic preservation | 0.20 | 0.05 |
| morphology equivariance | 0.20 | 0.05 |
| orthogonality | 0.05 | 0.05 |
| reconstruction | 0.10 | 0.10 |

The architecture, insertion point, 256/256 split, objective equations,
pretrained `ic1` teacher/student geometry, optimizer, schedule, augmentation,
inference, fusion, threshold, TTA, and metric remain frozen.

## Stage A: exact parity

Before any optimizer step, require:

- `max|H_hat-H| < 1e-6`;
- exact equality of CAM56, CAM28_1, CAM28_2, CAMdeep, and classification
  probabilities;
- full BCSS validation differing pixels equal zero;
- absolute mIoU and mDice differences below `1e-7`.

Failure returns `OSMF_V12_PARITY_NOGO` and stops.

## Stage B: eight-batch readiness

Start afresh from the frozen A0 checkpoint using BCSS training only, seed
20260817, batch size 20, 224x224, BF16, the released optimizer/schedule, and
equivariance interval 4. Process exactly eight batches and audit steps
`0,1,2,4,8`.

PASS requires both specialization objectives to have maximum weighted gradient
ratio at most `0.30` and mean at most `0.20`. Orthogonality and reconstruction
must never exceed `0.30`. All tensors, paths, parameters, representations, and
reconstruction must satisfy the frozen health criteria.

Two consecutive specialization ratios above `0.50`, a dead path, collapse,
non-finite state, reconstruction cosine below `0.90`, or SSHR loss explosion is
NOGO. Stable budget excess is REVIEW. A strong gradient conflict is at least
REVIEW.

Only `OSMF_V12_READINESS_PASS` authorizes Stage C.

## Stage C: fresh 128-batch Phase 0

Stage C must restart from the same A0 checkpoint and seed; it must not continue
the eight-batch state. Audit steps are
`0,1,2,4,8,16,32,64,96,128`.

GO requires, among all frozen health conditions:

- mean semantic and morphology ratios at most `0.20`;
- p95 semantic and morphology ratios at most `0.30`;
- end semantic agreement at least `0.90`;
- end reconstruction cosine at least `0.95`;
- `0.05 < RMS(S)/RMS(M) < 20`;
- active semantic and morphology paths and measurable movement of all four
  factorization tensors;
- stable SSHR loss and no sustained specialization ratio above `0.50`.

Morphology equivariance and cross-covariance trends are reported explicitly.
An unclear trend produces REVIEW rather than automatic tuning.

## Hard boundary

No gate starts a three-epoch pilot, 25-epoch training, BCSS test evaluation,
LUAD evaluation, another seed, or hyperparameter adjustment. Even Phase-0 GO
stops for human scientific review.
