# OSMF-v1.3-R1 PHASE0S Audit

## Decision

**OSMF_V13R1_PHASE0S_NOGO**

Reasons: `['FIXED_MORPHOLOGY_AFFINITY_NOT_IMPROVED']`
Processed batches: 128/128

## Corrected graph contract

- `grad(L_struct, p_morph) > 0`
- `grad(L_struct, u_morph) = 0` (expected by graph)
- `grad(L_total, p_morph) > 0`
- `grad(L_total, u_morph) > 0` with measurable update
- Graph expectation satisfied: True

## Causal evidence

- Improved/harmed/neutral: 16 / 15 / 1
- Improved fraction: 0.500000
- Mean delta: -7.5250812e-06

## Gradient budgets

- sem_pres: mean=0.161645, max=0.274865, p95=0.254330
- struct: mean=0.003073, max=0.007145, p95=0.005761
- orth: mean=0.044842, max=0.054173, p95=0.053968
- rec: mean=0.012428, max=0.018852, p95=0.018717

## Fixed 64-image probe

- AffinityEqErr(M): 0.010489036 -> 0.013293104
- AffinityEqErr(S): 0.0089395991 -> 0.010492395
- StructImprove(M): -26.733332%
- StructImprove(S): -17.369861%
- SpecificityGap: -9.363471%

## Boundary

No checkpoint, test, LUAD, segmentation-GT, pilot, or full training was run.

OSMF_V13R1_PHASE0S_NOGO
