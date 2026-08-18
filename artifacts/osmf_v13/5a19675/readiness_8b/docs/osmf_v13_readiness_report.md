# OSMF-v1.3 READINESS Audit

## Decision

**OSMF_V13_READINESS_NOGO**

Reasons: `['MORPHOLOGY_STRUCTURAL_PATH_INACTIVE']`
Processed batches: 8/8

## Frozen contract

- Audit commit: `5a19675e76e60a020892be934936aa19f31b03fa`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Parity proof SHA256: `35b0ce620790de88e4ab1c6ef3edcdf15fc811921389d9b2e580f6f11590c822`
- Readiness proof SHA256: `None`
- BCSS train only; seed 20260817; batch 20; 224x224; BF16.
- Frozen weights sem/struct/orth/rec = 0.05/0.05/0.05/0.10.
- Structural interval = 4; masked SmoothL1 beta = 1.0.
- Exact command: `tools/audit_osmf_v13_gradient_gate.py --gate readiness --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --parity-summary /home/duyanhong/experiments/OSMF_V13_LOCAL_STRUCTURAL_5a19675/parity/summary.json --output-dir /home/duyanhong/experiments/OSMF_V13_LOCAL_STRUCTURAL_5a19675/readiness_8b --audit-commit 5a19675e76e60a020892be934936aa19f31b03fa --num-workers 4`

## Same-pair causal evidence

- Improved/harmed/neutral: 2 / 0 / 0
- Improved fraction: 1.000000
- Mean delta: -0.00011211889

## Gradient budgets and safety

- sem_pres: mean=0.141535, max=0.219207, p95=0.211888
- struct: mean=0.002988, max=0.007145, p95=0.006340
- orth: mean=0.053258, max=0.054173, p95=0.054096
- rec: mean=0.005386, max=0.016878, p95=0.014793
- SemAgree end: 0.936311
- Reconstruction cosine end: 0.999103
- CrossCov start/end: 0.125875 / 0.088994

## Boundary

No checkpoint was saved. Validation, test, LUAD, segmentation GT, three-epoch pilot, and full training were not run.

OSMF_V13_READINESS_NOGO
