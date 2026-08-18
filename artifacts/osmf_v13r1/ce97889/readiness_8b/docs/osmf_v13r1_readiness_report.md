# OSMF-v1.3-R1 READINESS Audit

## Decision

**OSMF_V13R1_READINESS_PASS**

Reasons: `[]`
Processed batches: 8/8

## Corrected graph contract

- `grad(L_struct, p_morph) > 0`
- `grad(L_struct, u_morph) = 0` (expected by graph)
- `grad(L_total, p_morph) > 0`
- `grad(L_total, u_morph) > 0` with measurable update
- Graph expectation satisfied: True

## Causal evidence

- Improved/harmed/neutral: 2 / 0 / 0
- Improved fraction: 1.000000
- Mean delta: -0.00011217729

## Gradient budgets

- sem_pres: mean=0.141721, max=0.219207, p95=0.212006
- struct: mean=0.002991, max=0.007158, p95=0.006351
- orth: mean=0.053338, max=0.054170, p95=0.054106
- rec: mean=0.005395, max=0.016911, p95=0.014822

## Boundary

No checkpoint, test, LUAD, segmentation-GT, pilot, or full training was run.

OSMF_V13R1_READINESS_PASS
