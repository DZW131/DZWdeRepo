# OSMF-v1.2 Phase-0M Morphology Objective Causal Audit

## 1. Primary decision

**MORPH_EQ_OBJECTIVE_INVALID**

Decision reasons: `['SAME_PAIR_CAUSAL_INVALID']`.
Secondary flags: `['SAME_PAIR_CAUSAL_INVALID']`.

## 2. Frozen contract

- Audit commit: `af5b9a431e30d26bec36c024447e1b0af93cc197`
- Frozen v1.2 executed commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- v1.2 Phase-0 proof SHA256: `095871dcfc31e967fc842c777b6c9d44edd3826ac2eb084e0123ebc587ab94b6`
- Fresh A0 restart; BCSS train only; seed 20260817; batch 20; 224x224; BF16.
- Loss weights remain 0.05/0.05/0.05/0.10; architecture, objective, optimizer, schedule, and augmentation are unchanged.
- Exact command: `tools/audit_osmf_v12_phase0m.py --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --v12-phase0-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/phase0_128b/summary.json --output-dir /home/duyanhong/experiments/OSMF_V12_PHASE0M_af5b9a4/audit --audit-commit af5b9a431e30d26bec36c024447e1b0af93cc197 --num-workers 4`

## 3. Same-pair causal before/after

- Eq-active steps: 32
- Improved/harmed/neutral: 16 / 16 / 0
- Improved fraction: 0.500000
- Harmed fraction: 0.500000
- Mean/median delta: 4.6008732e-05 / 5.2526593e-07
- P25/P75 delta: -0.0020048004 / 0.0016775038
- Min/max delta: -0.01813969 / 0.021510545

Each before/after comparison reused the exact realized input tensor and flip. The normal full joint v1.2 update was executed once; no eq-only or second optimizer step was used.

## 4. Fixed 64-image probe

- Raw EqErr(M) start/end/delta: 0.063379295 / 0.069951042 / 0.0065717461
- Raw EqErr(S) start/end: 0.04613633 / 0.054438039
- AffinityEqErr(M) start/end/delta: 0.010489176 / 0.012004955 / 0.0015157785
- AffinityEqErr(S) start/end: 0.0089399068 / 0.010873745

The manifest fixes image IDs, dataset flips, pair flip, normalization, selection seed, and exact tensor SHA256. Probe forwards used no gradients and no optimizer update.

## 5. Morphology-parameter gradient competition

- Mean cos(eq, SSHR): -0.005286
- Mean cos(eq, semantic): 0.000000
- Mean cos(eq, orth): -0.037525
- Mean cos(eq, reconstruction): -0.108986

## 6. Safety replication

- Mean r_sem / r_eq: 0.162896 / 0.107691
- SemAgree start/end: 0.856712 / 0.986955
- Semantic response RMS ratio end: 0.770044
- Reconstruction cosine end: 0.998094
- S/M RMS ratio end: 1.355634
- CrossCov start/end: 0.015845 / 0.012998
- Representation healthy: True
- Replication instability: False

## 7. Boundary

- Processed batches: 128/128
- All finite: True
- Checkpoint saved: false
- Validation/test/LUAD/segmentation GT used: false
- Three-epoch pilot and 25-epoch training started: false
- v1.3 implemented: false

This audit stops after the causal decision and waits for human scientific review.

MORPH_EQ_OBJECTIVE_INVALID
