# RSBR-v0 Stage-1 Three-Epoch Frozen-SSHR Pilot

## 1. Executive conclusion

**RSBR_V0_PILOT_REVIEW**

Secondary flags: ['REGION_SEMANTIC_SIGNAL']

The performance delta itself is below the +0.05 pp threshold. `REVIEW` is
triggered only by the preregistered mechanism exception: Core-only is
positive, Full does not improve over Core-only, and Type-B errors decrease.
This is not a performance GO.

The pilot used BCSS training and validation only. It fresh-started from the
frozen A0 checkpoint, trained exactly three epochs, and stopped. No test,
LUAD, other seed, 25-epoch run, unfreezing, or tuning was performed.

## 2. Frozen control and provenance

- A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Experiment commit: `a266d0129edd52e537807ce45bf6ab58f34a9e29`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Initialization SHA256: `7b384befc6af84f6b8d209a327d6671bc9685b7bc9aa135485595f0637b23381`
- Parsed train / validation: 23,422 / 3,418
- Seed / epochs / batch / size / precision: 42 / 3 / 20 / 224 / BF16
- Loss weights: 0.10 / 0.15 / 0.25 / 0.50
- Auxiliary coefficients: region=0.05, residual=0.01
- Frozen SSHR parameters unchanged: True
- Frozen SSHR buffers unchanged: True
- Original modules remained eval and only RSBR was trainable: True

## 3. Paired same-forward validation

| Epoch | Base mIoU | Refined mIoU | Paired Δ (pp) | mDice Δ (pp) | C0 Δ | C1 Δ | C2 Δ | C3 Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 67.3291 | 67.3291 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| 1 | 67.3291 | 67.3562 | +0.0271 | +0.0221 | -0.0004 | -0.0057 | +0.1244 | -0.0099 |
| 2 | 67.3291 | 67.3540 | +0.0249 | +0.0204 | -0.0014 | -0.0063 | +0.1162 | -0.0088 |
| 3 | 67.3291 | 67.3519 | +0.0228 | +0.0186 | -0.0015 | -0.0063 | +0.1061 | -0.0071 |

- Best epoch: 1
- Best paired delta: +0.0271 pp
- Epoch-3 paired delta: +0.0228 pp
- Epoch-3 NoiseRatio: 1.714× the known 0.01329944 pp production envelope

## 4. Mechanism trajectory

| Epoch | CAM28_1 Δ (pp) | Type-B recovery | Type-D recovery | Core RMS | Transition RMS | T/C RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | +0.0037 | +34,077 (+0.4570%) | +16,670 (+0.0653%) | 0.197106 | 0.000535 | 0.002715 |
| 2 | +0.0025 | +33,453 (+0.4486%) | +16,672 (+0.0653%) | 0.194546 | 0.000646 | 0.003322 |
| 3 | +0.0038 | +31,812 (+0.4266%) | +15,791 (+0.0619%) | 0.182110 | 0.000639 | 0.003508 |

Epoch-3 transition/region training-gradient ratio: 0.067164.

## 5. Epoch-3 paired contribution ablation

| Variant | mIoU | ΔmIoU (pp) | mDice | C0 | C1 | C2 | C3 | Type-B recovery | Type-D recovery |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 67.3291 | +0.0000 | 80.2688 | 76.4494 | 70.5726 | 57.8269 | 64.4676 | +0 | +0 |
| core_only | 67.3520 | +0.0229 | 80.2875 | 76.4481 | 70.5665 | 57.9332 | 64.4604 | +31,835 | +15,707 |
| transition_only | 67.3290 | -0.0001 | 80.2688 | 76.4494 | 70.5724 | 57.8266 | 64.4677 | +190 | +203 |
| full | 67.3519 | +0.0228 | 80.2875 | 76.4479 | 70.5663 | 57.9330 | 64.4605 | +31,812 | +15,791 |

## 6. Training dynamics and safety

| Epoch | Training (s) | Validation (s) | Mean region grad | Mean transition grad | Mean residual ratio | Peak residual ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 129.92 | 65.43 | 6.011543e-02 | 4.681410e-03 | 0.030906 | 0.069053 |
| 2 | 75.54 | 65.57 | 5.864576e-02 | 4.201182e-03 | 0.038873 | 0.060864 |
| 3 | 73.36 | 72.96 | 5.679383e-02 | 3.814528e-03 | 0.038933 | 0.064959 |

- Peak CUDA memory: 1.723 GiB
- Mean regions/image across validation epochs: 2.6515
- Mean RSBR refinement overhead: 0.004080 s/image
- Mean RSBR overhead vs base forward: 46.45%
- Safety failures: none

## 7. Required scientific answers

1. Real paired mIoU gain: +0.0271 pp at epoch 1.
2. Gain vs numerical envelope: best NoiseRatio=2.039×.
3. Best epoch: 1.
4. Epoch 3 remains non-negative: True.
5. Refined CAM28_1 improves: True.
6. Core-only positive: True.
7. Transition-only positive: False.
8. Full exceeds both isolated paths: False.
9. Type-B error decreases: True.
10. Type-D error decreases: True.
11. Transition path remains quantitatively weak: True (transition-only Δ=-0.000063 pp; T/C RMS=0.003508). The stricter `TRANSITION_PATH_NOT_EFFECTIVE` flag is absent because its Type-D recovery condition is evaluated separately.
12. Frozen parameter/buffer hashes strictly unchanged: True.
13. 25-epoch recommendation: Not yet justified for a 25-epoch study without scientific review.

## 8. Checkpoints and commands

- Epoch checkpoints: `checkpoints/epoch1_rsbr.pth` through `epoch3_rsbr.pth`
- Validation-selected diagnostic checkpoint: `checkpoints/best_val_rsbr.pth`
- Checkpoint hashes: `{"best_val_rsbr.pth": "c530486b00683ff9dc24fe4d19ce4d79300cf6301172f9b4dbeb81c875c395ca", "epoch1_rsbr.pth": "c530486b00683ff9dc24fe4d19ce4d79300cf6301172f9b4dbeb81c875c395ca", "epoch2_rsbr.pth": "ecd0d67df5fb2a91164d7c8314daf749378182801ee29601c7bf29c8500bb846", "epoch3_rsbr.pth": "d39358b46de304dfeba052c0b6ddcc22b24fe60af897210903ec6b52dc8bbe36"}`

```bash
tools/run_rsbr_v0_stage1_3ep.py --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --parity-summary /home/duyanhong/experiments/RSBR_V0_PARITY_R1_AND_READINESS_7cbe5aa/parity_r1/summary.json --readiness-summary /home/duyanhong/experiments/RSBR_V0_PARITY_R1_AND_READINESS_7cbe5aa/readiness_32b/summary.json --output-dir /home/duyanhong/experiments/RSBR_V0_PILOT_3EP_a266d01 --experiment-commit a266d0129edd52e537807ce45bf6ab58f34a9e29 --batch-size 20 --img-size 224 --num-workers 4 --lr 0.01 --wt-dec 0.0005
```

## 9. Stop boundary

The Stage-1 protocol stops here regardless of the decision. This report does
not authorize test evaluation or any additional training.
