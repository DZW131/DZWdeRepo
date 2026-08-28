# LW-SHR Phase-1 LWTformer Transfer Utility Report

## 1. Commit hash

- Implementation commit: `a91f45dd0f343c850f179398a02fab3075fccac0`
- Pure official A0 base: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`

## 2. Exact commands

```bash
python tools/run_lw_shr_phase0.py --common-checkpoint /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/common/common_epoch20.pth --schedule /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/schedule/wdch_25epoch_schedule.npz --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --output-dir /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/phase0
python tools/train_lw_shr_matched.py --variant A1 --common-checkpoint /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/common/common_epoch20.pth --schedule /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/schedule/wdch_25epoch_schedule.npz --phase0-summary /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/phase0/lw_shr_phase0_summary.json --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/matched
python tools/train_lw_shr_matched.py --variant A2 --common-checkpoint /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/common/common_epoch20.pth --schedule /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/schedule/wdch_25epoch_schedule.npz --phase0-summary /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/phase0/lw_shr_phase0_summary.json --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/matched
python tools/train_lw_shr_matched.py --variant A3 --common-checkpoint /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/common/common_epoch20.pth --schedule /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/schedule/wdch_25epoch_schedule.npz --phase0-summary /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/phase0/lw_shr_phase0_summary.json --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/matched
python tools/analyze_lw_shr_phase1.py --mode final --c0-dir /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/C0 --experiment-root /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/matched --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/LW_SHR_PHASE1_a91f45d/report
```

All continuations independently start from the same frozen Epoch-20 state; no test or LUAD data were used.

## 3. Checkpoint SHA256

- C0: `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8`
- A1: `a9db26e24a246874f70116686ed85d8c73d5268e9e39dda9193e2808b8abc5ac`
- A2: `d2c6c283341abf345497a5ca796f1b140d5cabde8d6532dab4f637e242fff659`
- A3: `87cd6a81407f2397e4e52213abbc288ab17fe634a7d1ccc8e2caca6c7ca3a062`

## 4. Baseline-equivalence audit

Phase-0 verified FP32 maximum absolute output difference below `1e-5` for A1/A2/A3 at identity initialization. The original `wavelet_hfrm_mode=none` path remains bitwise identical to A0. A2/A3 filter gradients are expected to be zero at step 1 because the final gate projection is zero-initialized, and were required to become positive by step 2.

## 5. C0/A1/A2/A3 overall comparison

| Variant | mIoU (%) | Delta vs C0 (pp) | mDice (%) |
|---|---:|---:|---:|
| C0 | 66.8555 | +0.0000 | 79.9194 |
| A1 | 66.8787 | +0.0232 | 79.9359 |
| A2 | 66.8813 | +0.0258 | 79.9372 |
| A3 | 66.8685 | +0.0130 | 79.9292 |

## 6. CAM hierarchy

| Variant | CAM56 | CAM28_1 | CAM28_2 | CAMdeep | Final |
|---|---:|---:|---:|---:|---:|
| C0 | 61.0919 | 66.4431 | 66.2999 | 64.5274 | 66.8555 |
| A1 | 61.0898 | 66.4635 | 66.3504 | 64.5741 | 66.8787 |
| A2 | 61.0837 | 66.4686 | 66.3531 | 64.5720 | 66.8813 |
| A3 | 61.0858 | 66.4543 | 66.3471 | 64.5756 | 66.8685 |

## 7. Boundary/interior

| Variant | Boundary accuracy (%) | Boundary restricted mIoU (%) | Interior accuracy (%) | Interior restricted mIoU (%) |
|---|---:|---:|---:|---:|
| C0 | 51.6894 | 31.9958 | 85.5057 | 70.9137 |
| A1 | 51.6915 | 31.9823 | 85.5181 | 70.9417 |
| A2 | 51.7038 | 32.0089 | 85.5215 | 70.9431 |
| A3 | 51.6825 | 31.9801 | 85.5059 | 70.9308 |

## 8. Object size

The frozen historical object-size statistic is pixel-weighted component recall. It is reported under that accurate name; size-restricted mIoU is an additional diagnostic and is not substituted into the preregistered Gate C.

| Variant | Small recall | Small diagnostic mIoU | Medium recall | Medium diagnostic mIoU | Large recall | Large diagnostic mIoU |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 36.6083 | 19.2796 | 68.5264 | 46.1874 | 89.1509 | 77.6281 |
| A1 | 36.4554 | 19.2265 | 68.5547 | 46.2212 | 89.1563 | 77.6530 |
| A2 | 36.5062 | 19.2383 | 68.6065 | 46.2408 | 89.1397 | 77.6462 |
| A3 | 36.5057 | 19.2443 | 68.5358 | 46.2110 | 89.1469 | 77.6440 |

## 9. Filter drift

| Variant | dec_lo | dec_hi | low drift L2 | high drift L2 | low cosine | high cosine |
|---|---|---|---:|---:|---:|---:|
| A1 | `[0.7071067690849304, 0.7071067690849304]` | `[0.7071067690849304, -0.7071067690849304]` | 0.00000000 | 0.00000000 | 1.00000012 | 1.00000012 |
| A2 | `[0.6819543838500977, 0.6819542646408081]` | `[0.6819524765014648, -0.6819534301757812]` | 0.03557093 | 0.03557287 | 1.00000012 | 1.00000012 |
| A3 | `[0.6819545030593872, 0.6819542646408081]` | `[0.6819524765014648, -0.6819534301757812]` | 0.03557085 | 0.03557287 | 1.00000000 | 1.00000012 |

## 10. Gate statistics

| Variant | mean | std | p05 | p25 | p50 | p75 | p95 | min | max | spatial std | channel std | boundary mean | interior mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 1.000816 | 0.004676 | 0.999500 | 0.999500 | 1.000500 | 1.000500 | 1.001500 | 0.998236 | 1.063057 | 0.000017 | 0.004675 | 1.000813 | 1.000815 |
| A2 | 1.000816 | 0.004677 | 0.999500 | 0.999500 | 1.000500 | 1.000500 | 1.001500 | 0.998241 | 1.062905 | 0.000017 | 0.004676 | 1.000813 | 1.000816 |
| A3 | 1.000839 | 0.004705 | 0.999500 | 0.999500 | 1.000500 | 1.000500 | 1.002500 | 0.997771 | 1.067317 | 0.000017 | 0.004704 | 1.000836 | 1.000839 |

## 11. Context residual statistics

| Variant | Raw RMS | Gated RMS | Gated/raw | Boundary ratio | Interior ratio |
|---|---:|---:|---:|---:|---:|
| A1 | 0.373831 | 0.383034 | 1.023699 | 1.023536 | 1.023707 |
| A2 | 0.373766 | 0.382971 | 1.023707 | 1.023544 | 1.023715 |
| A3 | 0.373813 | 0.383269 | 1.024319 | 1.024041 | 1.024331 |

## 12. Per-class IoU

| Variant | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.5672 | 70.1931 | 57.9787 | 62.6833 |
| A1 | 76.5820 | 70.2143 | 57.9613 | 62.7572 |
| A2 | 76.6025 | 70.2121 | 57.9537 | 62.7571 |
| A3 | 76.5597 | 70.1933 | 57.9597 | 62.7615 |

## 13. Paired image bootstrap

10,000 paired image-level resamples (seed 42); each resample sums image confusion matrices and recomputes the official global mIoU.

| Variant | Observed delta (pp) | Bootstrap mean delta (pp) | 95% CI (pp) |
|---|---:|---:|---:|
| A1 | +0.0232 | +0.0236 | [-0.0108, +0.0737] |
| A2 | +0.0258 | +0.0263 | [-0.0096, +0.0771] |
| A3 | +0.0130 | +0.0132 | [-0.0135, +0.0565] |

## 14. Failure analysis

- A1: failed A_overall_utility.
- A2: failed A_overall_utility.
- A3: failed A_overall_utility, C_structural_mechanism.

## 15. GO/NO-GO

- A1: A=False, B=True, C=True, D=True; overall=NO-GO.
- A2: A=False, B=True, C=True, D=True; overall=NO-GO.
- A3: A=False, B=True, C=False, D=True; overall=NO-GO.

Phase-2 is authorized only when at least one executed variant passes all four preregistered gates. No model or threshold was selected using test data.

DECISION = NO_GO

