# OSMF-v1.2 Conservative Gradient-Budget Delivery

## Executive decision

**`OSMF_V12_PHASE0_REVIEW`**

OSMF-v1.2 passed exact A0 parity and the preregistered eight-batch readiness
gate. The independently restarted 128-batch Phase-0 run kept both
specialization objectives auxiliary, preserved semantic geometry and exact
reconstruction health, and reduced cross-subspace covariance. It did not show
the required favorable morphology-equivariance trend, so Phase 0 is REVIEW and
all later experiments remain locked.

## What was implemented

OSMF-v1.2 is architecturally identical to OSMF-v1.1. Its only scientific
change is the frozen gradient budget:

| Objective | v1.1 | v1.2 |
|---|---:|---:|
| semantic preservation | 0.20 | 0.05 |
| morphology equivariance | 0.20 | 0.05 |
| orthogonality | 0.05 | 0.05 |
| reconstruction | 0.10 | 0.10 |

The implementation keeps exactly four new trainable tensors and no auxiliary
classifier. Dedicated v1.2 parity, readiness, Phase-0 decision, table, figure,
and report tools enforce the preregistered hard stops. Phase 0 requires a PASS
proof from the same immutable commit and always creates a fresh model,
optimizer, dataloader, and A0 checkpoint state.

## Provenance

- Branch: `research/osmf-v1.2-gradient-budget`
- Executed commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`
- Development parent: `2c30fd67cb1ab1c33d6ed26593fdbac00054e74f`
- Frozen OSMF-v1.1 executed commit: `35591791e0bd81edaf53183afbf319358ccb7b81`
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Source specification SHA256: `c110d2a638f553941ce6f6c19f3657676b07bb418242f6520cbacf4da7b2039b`
- Immutable source archive SHA256: `b22d1890d98559fe2508d2660108b84b2e7908e943b5aff48c8d07995c313ce9`
- Server environment: PyTorch `2.11.0+cu128`, CUDA `12.8`, cuDNN `9.19`, RTX 5090 D v2

## Validation evidence

### Tests and CUDA smoke

- Local full suite: `87 passed, 3 CUDA-only skipped`
- RTX 5090 full suite: `92 passed`
- CUDA coverage includes batch20 BF16 exact identity, finite forward/backward,
  checkpoint compatibility, and optimizer coverage.
- The auxiliary semantic path leaves `ic1` gradient-free; the original SSHR
  loss still updates the live `ic1` parameters.

### Stage A: exact parity

Decision: **`OSMF_V12_PARITY_PASS`**

- Full BCSS validation images: `3,418`
- Differing prediction pixels: `0`
- Random and real `max|H_hat-H|`: `0`
- CAM56/CAM28_1/CAM28_2/CAMdeep difference: `0`
- Classification-probability difference: `0`
- A0/v1.2 mIoU: `0.6732774705 / 0.6732774705`
- A0/v1.2 mDice: `0.8026783768 / 0.8026783768`
- Parameter delta: `524,288` (`0.465167%`), exactly four tensors

Validation was used only to prove exact initialization parity. It was not used
for model selection or scientific tuning.

## Stage B: eight-batch readiness

Decision: **`OSMF_V12_READINESS_PASS`**

The run started from the frozen A0 checkpoint with seed `20260817`, batch size
20, image size 224, BF16, and the released SSHR optimizer/schedule. It processed
exactly eight BCSS training batches and saved no checkpoint.

| Objective | Mean ratio | Maximum | P95 | PASS budget |
|---|---:|---:|---:|---:|
| semantic preservation | 0.136138 | 0.219207 | 0.204784 | mean <=0.20, max <=0.30 |
| morphology equivariance | 0.092112 | 0.131270 | 0.124054 | mean <=0.20, max <=0.30 |
| orthogonality | 0.056316 | 0.071869 | 0.069330 | max <=0.30 |
| reconstruction | 0.003077 | 0.008632 | 0.007628 | max <=0.30 |

All four factorization tensors received finite gradients and measurable
updates. End reconstruction cosine was `0.999692`; end semantic agreement was
`0.948260`; end semantic-response RMS ratio was `0.813332`; end S/M RMS ratio
was `1.531177`. These checks authorized a fresh 128-batch Phase 0.

## Stage C: fresh 128-batch Phase 0

Decision: **`OSMF_V12_PHASE0_REVIEW`**

The formal Phase-0 run restarted from A0 and did not continue the readiness
weights. It processed exactly 128 batches and audited
`0,1,2,4,8,16,32,64,96,128`.

### Gradient budget

| Objective | Mean ratio | Maximum | P95 | Phase-0 target |
|---|---:|---:|---:|---:|
| semantic preservation | 0.162896 | 0.277304 | 0.254950 | mean <=0.20, P95 <=0.30 |
| morphology equivariance | 0.107691 | 0.132036 | 0.131882 | mean <=0.20, P95 <=0.30 |
| orthogonality | 0.052264 | 0.078339 | 0.076215 | auxiliary |
| reconstruction | 0.011950 | 0.023282 | 0.021823 | auxiliary |

Neither specialization objective was dominant or persistently above `0.50`.
Mean gradient-direction cosines remained close to zero; no strong gradient
conflict was detected.

### Representation and mechanism

| Diagnostic | Start | End | Interpretation |
|---|---:|---:|---|
| SemAgree | 0.856712 | 0.986955 | semantic geometry preserved |
| semantic response RMS ratio | 0.821541 | 0.770044 | non-collapsed |
| reconstruction cosine | 1.000000 | 0.998094 | SSHR representation protected |
| S/M RMS ratio | 1.306320 | 1.355634 | both branches healthy |
| CrossCov | 0.015845 | 0.012998 | genuine decorrelation signal |
| EqErr(M) | 0.062573 | 0.081165 | no favorable morphology trend |

The sole REVIEW reason is
`MORPHOLOGY_EQUIVARIANCE_NO_FAVORABLE_TREND`. EqErr(M) ended about 29.7% above
its start value. This prevents GO even though gradient budgeting, semantic
preservation, reconstruction, parameter movement, branch health, and
cross-covariance all met their preregistered criteria.

### Parameter health

| Parameter | Mean grad norm | Absolute update | Relative update |
|---|---:|---:|---:|
| `p_sem.weight` | 0.062458 | 0.172839 | 0.010802 |
| `p_morph.weight` | 0.047836 | 0.114626 | 0.007164 |
| `u_sem.weight` | 0.064953 | 0.145084 | 0.009068 |
| `u_morph.weight` | 0.042640 | 0.115653 | 0.007228 |

## Resource profile

- Full parity elapsed time: `107.02 s`
- Eight-batch audit elapsed time: `3.60 s`
- 128-batch audit elapsed time: `21.21 s`
- Phase-0 mean optimizer iteration: `0.135 s`
- Recorded peak allocated CUDA memory: about `6.14 GiB`

## Exact commands

```bash
python tools/audit_osmf_v12_parity.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-dir /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/parity \
  --osmf-v12-commit 92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4 \
  --num-workers 4

python tools/audit_osmf_v12_gradient_gate.py \
  --gate readiness \
  --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --parity-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/parity/summary.json \
  --output-dir /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/readiness_8b \
  --audit-commit 92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4 \
  --num-workers 4

python tools/audit_osmf_v12_gradient_gate.py \
  --gate phase0 \
  --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --parity-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/parity/summary.json \
  --readiness-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/readiness_8b/summary.json \
  --output-dir /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/phase0_128b \
  --audit-commit 92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4 \
  --num-workers 4
```

## Boundary and remaining work

- Three-epoch mechanism pilot: **not authorized and not run**
- 25-epoch training: not run
- BCSS test: not evaluated
- LUAD/other seeds: not evaluated
- Hyperparameter adjustment: none
- Continuation checkpoint: none

The current version stops at REVIEW. Any further change or experiment requires
separate human scientific review and authorization.

Raw logs, summaries, CSV tables, and figures are indexed in
[`audit/results/OSMF_V12_GRADIENT_BUDGET_92b9c14`](../audit/results/OSMF_V12_GRADIENT_BUDGET_92b9c14/ARTIFACTS.md).
