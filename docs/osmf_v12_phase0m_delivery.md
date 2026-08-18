# OSMF-v1.2 Phase-0M Causal Audit Delivery

## Executive decision

**`MORPH_EQ_OBJECTIVE_INVALID`**

The frozen pointwise morphology-equivariance objective did not show a net
causal benefit under the real joint OSMF-v1.2 optimizer update. Exactly half of
the 32 morphology-active steps improved their own fixed pair and half harmed
it, while the mean causal delta was slightly positive. Both raw-feature and
local-affinity fixed probes also failed to improve. The result is therefore not
explained by random probe batches, generalization alone, gradient conflict, or
a raw-versus-affinity metric mismatch.

## What was implemented

No model, loss, lambda, optimizer, scheduler, augmentation, inference, or
metric was changed. A diagnostic-only Phase-0M runner adds:

- same-pair EqErr before/after the one normal joint update at steps
  `4,8,...,128`;
- exact realized-pair manifests containing image IDs, dataset flips, pair flip,
  normalization metadata, and tensor SHA256;
- a fixed, GT-free 64-image BCSS training probe audited at
  `0,4,8,16,32,64,96,128`;
- raw morphology/semantic feature EqErr;
- direction-aware 8-neighbor local-affinity EqErr, used only for diagnosis;
- objective-gradient competition restricted to `p_morph/u_morph`;
- exact replication of the frozen v1.2 gradient and representation audit;
- deterministic tables, figures, decisions, and a hard stop after the report.

## Provenance

- Branch: `analysis/osmf-v1.2-phase0m`
- Executed Phase-0M commit: `af5b9a431e30d26bec36c024447e1b0af93cc197`
- Frozen v1.2 executed commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- v1.2 Phase-0 proof SHA256: `095871dcfc31e967fc842c777b6c9d44edd3826ac2eb084e0123ebc587ab94b6`
- Source specification SHA256: `fe9a38e49ed95b51658b804e7d335dbcace4aefd59679c2db13f039129f1b4c7`
- Immutable source archive SHA256: `2560a92f081e3be0a26abe35262f3720845269ed7e3333995e2649b129aee76d`
- Environment: PyTorch `2.11.0+cu128`, CUDA `12.8`, cuDNN `9.19`, RTX 5090 D v2

An initial launch completed with an incorrectly transcribed full audit SHA. It
was excluded before analysis and retained only as server-side provenance at
`/home/duyanhong/experiments/OSMF_V12_PHASE0M_af5b9a4_INVALID_PROVENANCE_20260818`.
The formal run restarted from A0 with the verified commit above and wrote to the
standard experiment directory.

## Validation evidence

- Local full suite: `101 passed, 3 CUDA-only skipped`
- RTX 5090 full suite: `104 passed`
- Direction-aware affinity inverse alignment is exact for both horizontal and
  vertical feature flips.
- Formal run processed `128/128` real BCSS training batches and remained finite.
- No continuation checkpoint was loaded or saved.

## Same-pair causal test

| Statistic | Result |
|---|---:|
| Eq-active steps | 32 |
| Improved | 16 |
| Harmed | 16 |
| Neutral | 0 |
| Improved fraction | 0.500000 |
| Mean delta | +0.00004601 |
| Median delta | +0.00000053 |
| P25 / P75 | -0.00200480 / +0.00167750 |
| Minimum / maximum | -0.01813969 / +0.02151055 |

The preregistered invalid criterion is met because mean
`EqErr_after - EqErr_before >= 0`. This measures the net effect of the full
v1.2 update, not an artificial eq-only step.

## Fixed 64-image probe

| Metric | Start | End | Delta | Interpretation |
|---|---:|---:|---:|---|
| Raw EqErr(M) | 0.063379 | 0.069951 | +0.006572 | unfavorable |
| Raw EqErr(S) | 0.046136 | 0.054438 | +0.008302 | semantic control also worsened |
| AffinityEqErr(M) | 0.010489 | 0.012005 | +0.001516 | no affinity improvement |
| AffinityEqErr(S) | 0.008940 | 0.010874 | +0.001934 | semantic control also worsened |

Raw morphology EqErr exceeded the preregistered `+0.005` unfavorable boundary.
Affinity did not improve, so
`LOCAL_GEOMETRY_IMPROVES_DESPITE_RAW_FEATURE_EQ_FAILURE` is not supported.

## Morphology-parameter gradient competition

| Gradient cosine | Mean |
|---|---:|
| cos(eq, SSHR) | -0.005286 |
| cos(eq, semantic) | 0.000000 |
| cos(eq, orthogonality) | -0.037525 |
| cos(eq, reconstruction) | -0.108986 |

The main task cosine is far above the conflict thresholds `-0.30/-0.50`.
Therefore the invalid causal result cannot be attributed to strong morphology
task-gradient conflict.

## Safety replication

The Phase-0M training dynamics reproduced every frozen v1.2 reference exactly:

| Diagnostic | Reference | Phase-0M |
|---|---:|---:|
| mean r_sem | 0.162896 | 0.162896 |
| mean r_eq | 0.107691 | 0.107691 |
| SemAgree end | 0.986955 | 0.986955 |
| reconstruction cosine end | 0.998094 | 0.998094 |
| S/M RMS ratio end | 1.355634 | 1.355634 |
| CrossCov end | 0.012998 | 0.012998 |

All four factorization tensors received finite gradients and measurable
updates. There was no replication instability, response collapse,
reconstruction failure, or branch collapse.

## Resource profile

- Formal audit elapsed time: `28.72 s`
- Recorded peak allocated CUDA memory: about `6.14 GiB`
- Probe images: `64`
- Training-pair manifest rows: `640`
- Checkpoint output: none

## Exact command

```bash
python tools/audit_osmf_v12_phase0m.py \
  --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --v12-phase0-summary /home/duyanhong/experiments/OSMF_V12_GRADIENT_BUDGET_92b9c14/phase0_128b/summary.json \
  --output-dir /home/duyanhong/experiments/OSMF_V12_PHASE0M_af5b9a4/audit \
  --audit-commit af5b9a431e30d26bec36c024447e1b0af93cc197 \
  --num-workers 4
```

## Scientific boundary

Under the preregistered decision tree, the current pointwise morphology
feature-equivariance objective should be stopped. This audit does not authorize
increasing lambda, training longer, adding warmup, entering a three-epoch or
25-epoch run, evaluating test/LUAD, or implementing v1.3. A local-structural or
affinity objective would require a new, independently preregistered version.

Raw artifacts are indexed in
[`audit/results/OSMF_V12_PHASE0M_af5b9a4`](../audit/results/OSMF_V12_PHASE0M_af5b9a4/ARTIFACTS.md).
