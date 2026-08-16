# Full FA-MPR BCSS Seed-42 Final-Checkpoint Result

## 1. Executive conclusion

The single frozen Full FA-MPR experiment completed successfully. On BCSS
seed 42, using the epoch-25 final checkpoint and the official SSHR test
inference, Full FA-MPR obtained **70.0390% mIoU and 82.1946% mDice**.

Against the frozen A0 result of 69.9795% mIoU and 82.1400% mDice, the changes
are only **+0.0596 mIoU and +0.0547 mDice percentage points**. This is
effectively a neutral result and does not establish a meaningful segmentation
improvement, especially given the 93.6% increase in observed training time.

## 2. Frozen protocol

| Setting | Value |
|---|---|
| experiment | `EXP_BCSS_FAMPR_SEED42_FINAL25` |
| repository commit | `e4b7b6cb0d9354afc07f9d0348f801340043ffd1` |
| dataset | BCSS-WSSS |
| train / val / test | 23,422 / 3,418 / 4,986 |
| seed | 42 |
| epochs | 25 |
| batch size | 20 |
| image size | 224 |
| precision | BF16 autocast; FA-MPR sampling internally FP32 |
| model switch | `--rectifier hfrm --context-mode fampr` |
| pretrained SHA256 | `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16` |
| checkpoint rule | epoch-25 final checkpoint only |
| inference / metric | official SSHR test inference and metric |
| checkpoint SHA256 | `6dac360ddf5fe883b01dc5ac0a77dbdd84ea57067ac393439b24de20a4f7b1d7` |

No validation- or test-based checkpoint selection was performed. No loss,
optimizer, learning-rate schedule, CAM fusion, threshold, metric, backbone,
GSR, or HFRM residual-gamma setting was changed from A0.

## 3. Final test comparison

| Metric | A0 final-25 | Full FA-MPR final-25 | Delta (pp) |
|---|---:|---:|---:|
| Pixel Accuracy | 85.1452 | 84.9599 | -0.1853 |
| Mean Accuracy | 81.2905 | 81.3162 | +0.0257 |
| Frequency Weighted IoU | 74.2449 | 73.9669 | -0.2780 |
| **mIoU** | **69.9795** | **70.0390** | **+0.0596** |
| **mDice** | **82.1400** | **82.1946** | **+0.0547** |

## 4. Per-class comparison

| BCSS class | A0 IoU | FA-MPR IoU | Delta IoU (pp) | A0 Dice | FA-MPR Dice | Delta Dice (pp) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 79.2939 | 78.9139 | -0.3799 | 88.4513 | 88.2144 | -0.2369 |
| 1 | 73.0588 | 72.7006 | -0.3582 | 84.4324 | 84.1926 | -0.2397 |
| 2 | 60.5091 | 60.5014 | -0.0078 | 75.3965 | 75.3905 | -0.0060 |
| 3 | 67.0560 | 68.0403 | +0.9843 | 80.2796 | 80.9809 | +0.7013 |

The mean improvement comes entirely from class 3. Classes 0 and 1 decline,
and class 2 is unchanged. The result therefore suggests a redistribution of
class behavior rather than a broad improvement.

## 5. Training dynamics and health

The run finished with exit code 0. All 25 epoch-level FA-MPR diagnostic records
were written. No NaN, Inf, OOM, traceback, killed process, or non-finite
FA-MPR output/gradient was detected.

| Final training quantity | A0 | Full FA-MPR |
|---|---:|---:|
| classification loss | 0.2247 | 0.2242 |
| exact match | 72.08% | 71.43% |
| training accuracy | 87.19% | 86.87% |
| elapsed training time | 45.19 min | 87.50 min |

At epoch 25, the three FA-MPR CH anchors were approximately 0.342, 0.325, and
0.367. All frequency predictor, base-kernel, anchor, and context residual
gradients remained finite and nonzero. The learned branch was active and
numerically stable, so the neutral result is not explained by a dormant path or
training failure.

## 6. Interpretation and decision

This controlled run does not support Full FA-MPR v1.0 as a successful new
Innovation 1 in its current form:

- the mIoU gain is only 0.0596 percentage points;
- two of four classes become worse;
- observed training time rises by 93.6%;
- the result remains 1.7810 mIoU points below the paper's 71.82 result and
  0.9210 points below the authors' recent H100 rerun mean of 70.96.

Because the effect is essentially zero, expanding this exact configuration to
LUAD or multiple seeds is not justified as the immediate next experiment. The
recommended next step is a targeted architectural diagnosis of why adaptive
frequency sampling mainly benefits class 3 while harming classes 0 and 1,
followed by a new reviewed design. Do not tune this frozen run on the test set.

## 7. Artifact locations

Server experiment root:

```text
/home/duyanhong/fampr-bcss-seed42-25ep-final-20260816
```

Key artifacts:

```text
checkpoints/stage1_last.pth
checkpoints/experiment_config.json
checkpoints/fampr_diagnostics.jsonl
checkpoint.sha256
train.log
test.log
status.tsv
```
