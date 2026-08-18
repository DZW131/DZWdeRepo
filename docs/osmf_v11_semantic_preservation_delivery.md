# OSMF-v1.1 Semantic Preservation Delivery

## Executive decision

**`OSMF_V11_SEMANTIC_READINESS_REVIEW`**

OSMF-v1.1 passed exact implementation parity, then completed exactly eight real
BCSS training batches. The semantic-preservation formulation reduced the
semantic gradient ratio by several fold relative to v1.0, but did not satisfy
the preregistered readiness PASS range. Consequently, the 128-batch audit and
all later phases were not started.

## What changed from v1.0

The randomly initialized `GAP(S) -> Linear` classifier was removed. OSMF-v1.1
has exactly four new trainable tensors:

- `p_sem.weight`
- `p_morph.weight`
- `u_sem.weight`
- `u_morph.weight`

The pretrained SSHR `ic1` geometry supplies detached teacher and functional
student responses:

```text
Z_H = functional_ic1(H.detach(), ic1.weight.detach(), ic1.bias.detach())
Z_S = functional_ic1(H_S,        ic1.weight.detach(), ic1.bias.detach())
```

`L_sem_pres` is their class-channel normalized spatial cosine distance. Unit
and real CUDA audits verify that this auxiliary loss cannot update `ic1`, while
the original SSHR loss still updates the live `ic1` parameters.

Everything else remained frozen: 256/256 split, exact reconstruction,
equivariance/orthogonality/reconstruction objectives, weights
`0.20/0.20/0.05/0.10`, optimizer, schedule, augmentation, inference, and metric.

## Provenance

- Stacked parent: OSMF-v1.0 Phase-0 branch at `cd63575d5a78e4d342ef8a844cb32fbe1ff47e1f`
- Executed OSMF-v1.1 commit: `35591791e0bd81edaf53183afbf319358ccb7b81`
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Preregistered source-spec SHA256: `66340fdbef15d0092870a7ad06081134670a094008783523cde28519c6c2ec23`
- Environment: PyTorch `2.11.0+cu128`, CUDA `12.8`, cuDNN `9.19`, RTX 5090 D v2

## Validation evidence

### Local and CUDA tests

- Local: `64 passed, 2 CUDA-only skipped`
- RTX 5090: `66 passed`
- Batch20 BF16 gradient smoke: finite; exact identity; auxiliary `ic1` gradients both `None`
- Synthetic smoke `r_sem_pres`: `0.01989` (diagnostic only, not a decision input)

### Phase -1.1 parity

Final decision: **`OSMF_V11_PARITY_PASS`**

- Random and real-input `max|H_hat-H|`: `0`
- CAM56/CAM28_1/CAM28_2/CAMdeep difference: `0`
- Classification probability difference: `0`
- Full BCSS validation images: `3,418`
- Differing prediction pixels: `0`
- A0/v1.1 mIoU: `0.6732758491 / 0.6732758491`
- A0/v1.1 mDice: `0.8026772813 / 0.8026772813`
- Parameter delta: `524,288` (`0.465167%`), exactly four tensors

The validation split was used only for exact parity. It did not select any
configuration or contribute to the readiness decision.

## Eight-batch readiness result

The run started afresh from the A0 checkpoint with seed `20260817`, batch size
20, image size 224, BF16, and the released optimizer behavior. It processed no
ninth batch and saved no continuation checkpoint.

| Audit step | r_sem_pres | r_eq | r_orth | r_rec | SemAgree | CosRec |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.876828 | 0.332644 | 0.053213 | 0.000000 | 0.877873 | 0.999872 |
| 2 | 0.380616 | 0.289275 | 0.054710 | 0.003346 | 0.871569 | 0.999717 |
| 4 | 0.421914 | 0.331531 | 0.047215 | 0.005342 | 0.946767 | 0.999498 |
| 8 | 0.715869 | 0.599952 | 0.050048 | 0.016271 | 0.978579 | 0.998917 |

Semantic-preservation ratio summary:

- v1.0: `2.480527–4.106567`
- v1.1: `0.380616–0.876828`
- v1.1 mean: `0.598807`
- step 4 and step 8: both above the PASS threshold `0.30`
- two consecutive values above `0.50`: no

Therefore v1.1 is substantially better than v1.0 and avoids NOGO, but it cannot
receive PASS. The exact preregistered outcome is REVIEW.

## Parameter and representation health

| Parameter | Mean grad norm | Absolute update | Relative update |
|---|---:|---:|---:|
| `p_sem.weight` | 0.102299 | 0.050946 | 0.003184 |
| `p_morph.weight` | 0.064801 | 0.028884 | 0.001805 |
| `u_sem.weight` | 0.107887 | 0.046047 | 0.002878 |
| `u_morph.weight` | 0.051811 | 0.030864 | 0.001929 |

- Semantic/morphology total-update ratio is about `1.62`, far below the review threshold 20.
- Semantic response RMS ratio ended at `0.759923`; no response collapse.
- Semantic agreement improved from `0.856712` to `0.978579`.
- Reconstruction cosine ended at `0.998917`.
- S/M RMS ratio ended at `1.37094`; no branch collapse.
- CrossCov changed `0.0158448 -> 0.0100741` with healthy branches.
- All tensors and gradients remained finite.
- No strong gradient-direction conflict was observed.

EqErr(M) changed `0.0625734 -> 0.0993292`; the morphology path was connected and
updated, but eight batches are not used to claim long-term equivariance quality.

## Resource observation

The executed audit reported a peak allocated memory of about 10.88 GiB. That
value includes diagnostic-gradient tensors still referenced when the separate
optimizer-step profile began; it is a conservative audit-process peak, not a
formal training-only memory estimate. The post-run engineering cleanup releases
those diagnostic references before profiling future steps. No scientific data
or decision was recomputed after this cleanup.

## Exact executed commands

```bash
python tools/audit_osmf_v11_parity.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --output-dir /home/duyanhong/experiments/OSMF_V11_SEMANTIC_PRESERVATION_3559179/parity \
  --osmf-v11-commit 35591791e0bd81edaf53183afbf319358ccb7b81 \
  --num-workers 4

python tools/audit_osmf_v11_gradient_gate.py \
  --gate readiness \
  --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --parity-summary /home/duyanhong/experiments/OSMF_V11_SEMANTIC_PRESERVATION_3559179/parity/summary.json \
  --output-dir /home/duyanhong/experiments/OSMF_V11_SEMANTIC_PRESERVATION_3559179/readiness_8b \
  --audit-commit 35591791e0bd81edaf53183afbf319358ccb7b81 \
  --num-workers 4
```

## Boundary and remaining work

- 128-batch Phase 0: **not authorized and not run**
- 3-epoch pilot: not run
- 25-epoch training: not run
- Test/LUAD/other seeds: not run
- Hyperparameter adjustment: none

The current version stops at REVIEW. Any change to lambda, initialization,
teacher formulation, masks, or warmup would require a new preregistered OSMF
version.

Raw artifacts are indexed in
[`audit/results/OSMF_V11_SEMANTIC_PRESERVATION_3559179`](../audit/results/OSMF_V11_SEMANTIC_PRESERVATION_3559179/ARTIFACTS.md).
