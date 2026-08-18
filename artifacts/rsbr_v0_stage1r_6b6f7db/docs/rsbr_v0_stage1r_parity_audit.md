# RSBR-v0 Stage -1R Deterministic Parity Harness Audit

## 1. Executive conclusion

**RSBR_V0_PARITY_HARNESS_NONDETERMINISM_CONFIRMED**

This was an audit-only BCSS validation experiment. No optimizer, training,
Stage 0, three-epoch pilot, test split, LUAD split, threshold change, loss
change, or RSBR model change was used.

## 2. Frozen provenance and environment

- A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- RSBR commit: `98e15df9ba702f0b9f43efc1942287abe02b49c8`
- Checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- Checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Environment: `{"autocast_dtype": "bf16", "cuda_runtime": "12.8", "cudnn": 91900, "cudnn_benchmark": true, "cudnn_deterministic": false, "deterministic_algorithms": false, "gpu": "NVIDIA GeForce RTX 5090 D v2", "mode": "production", "python": "3.10.20", "pytorch": "2.11.0+cu128", "tf32_convolution": "tf32", "tf32_matmul": "tf32"}`
- Fixed subset: the lexicographically first 32 BCSS validation image IDs,
  frozen in `same_process_identity/summary.json`.

## 3. Same-process structural identity

- Images: 32
- Maximum base/refined CAM28_1 difference: 0.000e+00
- Differing base/refined prediction pixels: 0
- Delta-core exact zero: True
- Delta-transition exact zero: True
- Structural identity: **PASS**

Per-image hashes, dtypes, autocast state, and tensor contiguity are retained in
`same_process_identity/summary.json` and summarized under `merge_path/summary.json`.

## 4. Absolute full-validation results

| Run | mIoU (%) | mDice (%) | C0 IoU | C1 IoU | C2 IoU | C3 IoU | Runtime (s) | Prediction hash |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A0-1 | 67.32458600 | 80.26580218 | 76.43939732 | 70.56580695 | 57.82670075 | 64.46643899 | 61.22 | `61aa8f221c7c` |
| A0-2 | 67.32458600 | 80.26580218 | 76.43939732 | 70.56580695 | 57.82670075 | 64.46643899 | 61.63 | `61aa8f221c7c` |
| A0-3 | 67.33203077 | 80.27168515 | 76.43923804 | 70.57038723 | 57.85409011 | 64.46440769 | 60.38 | `b455574b437a` |
| RSBR-1 | 67.32368977 | 80.26527657 | 76.43425493 | 70.56590099 | 57.82756214 | 64.46704102 | 189.20 | `bb8f13854191` |
| RSBR-2 | 67.33652669 | 80.27470637 | 76.44902738 | 70.57652506 | 57.85498143 | 64.46557288 | 189.51 | `2ed8da20a01a` |
| RSBR-3 | 67.33698921 | 80.27506195 | 76.44901660 | 70.57655251 | 57.85584741 | 64.46654032 | 193.33 | `b56ca756a3b0` |

## 5. Full-validation repeat and cross-model table

| Comparison | Differing Pixels | mIoU Diff (pp) | mDice Diff (pp) | Max class-IoU Diff (pp) |
|---|---:|---:|---:|---:|
| A0-1 vs A0-2 | 0 | 0.00000000 | 0.00000000 | 0.00000000 |
| A0-1 vs A0-3 | 36,795 | 0.00744477 | 0.00588297 | 0.02738936 |
| A0-2 vs A0-3 | 36,795 | 0.00744477 | 0.00588297 | 0.02738936 |
| RSBR-1 vs RSBR-2 | 83,626 | 0.01283692 | 0.00942979 | 0.02741929 |
| RSBR-1 vs RSBR-3 | 81,937 | 0.01329944 | 0.00978538 | 0.02828527 |
| RSBR-2 vs RSBR-3 | 28,155 | 0.00046252 | 0.00035558 | 0.00096744 |
| A0-1 vs RSBR-1 | 58,482 | 0.00089623 | 0.00052560 | 0.00514238 |
| A0-2 vs RSBR-2 | 57,792 | 0.01194068 | 0.00890419 | 0.02828068 |
| A0-3 vs RSBR-3 | 39,400 | 0.00495844 | 0.00337680 | 0.00977856 |

## 6. Numerical envelopes and decision rule

- A0 self envelope: `{"mdice_pp": 0.005882968287429513, "miou_pp": 0.007444765208375337, "per_class_iou_pp": 0.027389359881901942, "pixels": 36795}`
- RSBR self envelope: `{"mdice_pp": 0.009785375481807801, "miou_pp": 0.01329944088913626, "per_class_iou_pp": 0.028285273676786904, "pixels": 83626}`
- Cross-model envelope: `{"mdice_pp": 0.008904190103597998, "miou_pp": 0.011940684959466097, "per_class_iou_pp": 0.02828068382003668, "pixels": 58482}`
- Frozen mIoU allowance: 0.01379944 pp
- Frozen pixel allowance: 87,808
- mIoU rule pass: True
- Pixel rule pass: True

## 7. Diagnostic modes on the fixed 32-image subset

| Mode | A0 repeat pixels | RSBR repeat pixels | A0-vs-RSBR pixels | Maximum mIoU diff (pp) |
|---|---:|---:|---:|---:|
| production | 360 | 480 | 282 | 0.00526878 |
| deterministic | 0 | 0 | 0 | 0.00000000 |
| fp32 | 0 | 18 | 0 | 0.00000845 |
| tf32_off | 315 | 333 | 311 | 0.00238177 |

## 8. Source localization

- Deterministic algorithms reduce the observed subset discrepancy; CUDA/cuDNN algorithm selection is implicated.
- Disabling autocast reduces the discrepancy; BF16 rounding/accumulation is implicated.
- Disabling TF32 reduces the discrepancy; TF32 execution is implicated.
- Data order and TTA accumulation order are identical by construction, and the same-process zero residual merge is checked separately.

These mode tests are diagnostic only and do not alter the production SSHR or
RSBR protocol.

## 9. Residual merge path

The canonical branch records show zero-filled `delta_core` and
`delta_transition`, equal base/refined CAM hashes, and the recorded BF16 dtype
and contiguity state. Full per-image records are in
`same_process_identity/summary.json`.

## 10. Commands and artifacts

Exact top-level command:

```bash
tools/audit_rsbr_v0_stage1r.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --output-dir /home/duyanhong/experiments/RSBR_V0_STAGE1R_PARITY_6b6f7db --rsbr-commit 98e15df9ba702f0b9f43efc1942287abe02b49c8 --num-workers 4
```

Each independent inference command and stdout/stderr is stored beside its run
directory. Prediction masks are retained as compressed NPZ files so every
pixel-count comparison is reproducible.

## 11. Stop decision

Stage -1R stops here. This report does not authorize Stage 0. Under the frozen
specification, a revised parity harness may be implemented only after human
review of a `RSBR_V0_PARITY_HARNESS_NONDETERMINISM_CONFIRMED` result.
