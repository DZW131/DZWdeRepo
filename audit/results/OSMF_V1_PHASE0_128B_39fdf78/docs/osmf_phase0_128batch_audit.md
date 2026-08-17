# OSMF-v1.0 Phase 0 — 128-Batch Structural & Gradient Audit

## 1. Executive decision

Final decision: **OSMF_PHASE0_NOGO**.

Processed real BCSS training batches: 2/128.
This is a mechanism-safety audit, not a segmentation-performance experiment.
No validation mIoU was used and no test/LUAD data were accessed.

## 2. Frozen contract

- Phase-0 parent/OSMF implementation commit: `5eb7b258f0cdeb4fa8779b65e716c105c9541f9a`.
- Phase-0 audit commit: `39fdf788aed6d0e31bd42108d87fc502a37d591a`.
- Frozen A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Optimizer: freshly initialized released PolyOptimizer/SGD from the A0 checkpoint.
- Optimizer momentum: 0.0005.
- Poly schedule max steps: 29275 (25-epoch formal scale).
- Batch size: 20; image size: 224; seed: 20260817; BF16.
- Fixed auxiliary weights: 0.20/0.20/0.05/0.10.
- Equivariance interval: 4; transforms: horizontal/vertical flip only.
- Exact command: `tools/audit_osmf_phase0_128batch.py --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/OSMF_V1_PHASE0_128B_39fdf78 --audit-commit 39fdf788aed6d0e31bd42108d87fc502a37d591a --num-workers 4`.

## 3. Start-state safety

- max|H_hat-H|: 0.
- Reconstruction cosine: 1.0000000000.
- All tensors/CAMs/losses finite: True.

## 4. Required main table

| Metric | Start | Mean | End | Min | Max | Status |
|---|---:|---:|---:|---:|---:|---|
| L_SSHR | 0.169584 | 0.187749 | 0.224079 | 0.169584 | 0.224079 | FINITE |
| L_sem | 1.15705 | 1.00664 | 0.705823 | 0.705823 | 1.15705 | FINITE |
| L_eq | 0.0625734 | 0.0625734 | 0.0625734 | 0.0625734 | 0.0625734 | FINITE |
| L_orth | 0.0158448 | 0.0154082 | 0.014535 | 0.014535 | 0.0158448 | FINITE |
| L_rec | 0 | 0.000500361 | 0.00150108 | 0 | 0.00150108 | FINITE |
| r_sem | 4.10657 | 3.29355 | 2.48053 | 2.48053 | 4.10657 | HARD-STOP RANGE OBSERVED |
| r_eq | 0.332644 | 0.325808 | 0.318972 | 0.318972 | 0.332644 | REVIEW RANGE OBSERVED |
| r_orth | 0.053213 | 0.0558216 | 0.0584301 | 0.053213 | 0.0584301 | HEALTHY REFERENCE RANGE |
| r_rec | 2.11865e-08 | 0.0064559 | 0.0129118 | 2.11865e-08 | 0.0129118 | BELOW REFERENCE RANGE |
| cos(base,sem) | -0.0159096 | -0.0214359 | -0.0269622 | -0.0269622 | -0.0159096 | FINITE |
| cos(base,eq) | 0.00561597 | 0.00193491 | -0.00174616 | -0.00174616 | 0.00561597 | FINITE |
| cos(base,orth) | -0.0152036 | -0.00632617 | 0.00255126 | -0.0152036 | 0.00255126 | FINITE |
| cos(base,rec) | 0 | -0.00530014 | -0.0106003 | -0.0106003 | 0 | FINITE |
| RMS(S) | 0.919284 | 0.802674 | 0.7394 | 0.7394 | 0.919284 | FINITE |
| RMS(M) | 0.703722 | 0.667134 | 0.660231 | 0.637449 | 0.703722 | FINITE |
| RMS(S)/RMS(M) | 1.30632 | 1.20058 | 1.11991 | 1.11991 | 1.30632 | HEALTHY |
| Cos(H,H_hat) | 1 | 0.998462 | 0.996959 | 0.996959 | 1 | GO RANGE |
| Residual Ratio | 0 | 0.0446121 | 0.0773233 | 0 | 0.0773233 | FINITE |
| CrossCov(S,M) | 0.0158448 | 0.0152003 | 0.0153385 | 0.0144176 | 0.0158448 | FINITE |
| EqErr(M) | 0.0625734 | 0.0623126 | 0.062208 | 0.0621566 | 0.0625734 | FINITE |
| EqErr(S) | 0.0456588 | 0.0528998 | 0.0601392 | 0.0456588 | 0.0601392 | FINITE |

## 5. Parameter health

| Parameter | Grad nonzero? | Mean grad norm | End absolute update | End relative update | Status |
|---|---:|---:|---:|---:|---|
| `p_sem.weight` | True | 0.638325 | 0.11878 | 0.00742372 | ACTIVE |
| `p_morph.weight` | True | 0.0323273 | 0.00507006 | 0.000316879 | ACTIVE |
| `u_sem.weight` | True | 0.0387019 | 0.00617546 | 0.000385966 | ACTIVE |
| `u_morph.weight` | True | 0.0345828 | 0.00551123 | 0.000344452 | ACTIVE |
| `semantic_classifier.weight` | True | 0.380804 | 0.0725455 | 0.0255646 | ACTIVE |
| `semantic_classifier.bias` | True | 0.029982 | 0.0111959 | 1.11959e+10 | ACTIVE |

## 6. Gradient safety and direction

Flags: `[]`.
Decision reasons: `['PERSISTENT_SEM_GRADIENT_RATIO_GT_0_50']`.
Independent objective gradients were measured with `torch.autograd.grad` at steps 1/2/4/8/16/32/64/96/128 and did not populate optimizer gradients.

## 7. Representation health and early specialization

- End reconstruction cosine: 0.99695915.
- End S/M RMS ratio: 1.11991.
- EqErr(M) start/end: 0.0625734 / 0.062208.
- CrossCov start/end: 0.0158448 / 0.0153385.
- Equivariance response detected: True.
- Morphology equivariance gradient active: True.

## 8. Compute cost

- Mean iteration time: 1.003943 s.
- Mean non-equivariance iteration: 1.003943 s.
- Mean equivariance iteration: 1.003943 s.
- Interval-averaged overhead proxy versus non-equivariance OSMF step: 0.00%.
- Peak training-step GPU memory: 3.687 GiB.

The overhead is an in-run proxy using non-equivariance OSMF steps as the denominator; no additional A0 training batches were run.

## 9. Protocol safety

- Training labels: image-level filename labels only.
- Segmentation GT used in training: false.
- Validation evaluated: false.
- Test evaluated: false.
- LUAD evaluated: false.
- 3-epoch or 25-epoch training: false.
- Lambda/LR/architecture/fusion/threshold/TTA changes: false.

## 10. Phase boundary

This audit now stops. Even a GO only permits human review before a separate 3-epoch mechanism pilot; it does not authorize Phase 1 automatically.

OSMF_PHASE0_NOGO
