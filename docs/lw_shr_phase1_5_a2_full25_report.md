# LW-SHR Phase-1.5 A2 Full-25 Report

## 1. Implementation commit

- Diagnostic/training commit: `8f6da267a76b0c7e573616c9bd9c5fbeff4a577c`
- Frozen A2 architecture commit: `a91f45dd0f343c850f179398a02fab3075fccac0`
- Pure A0 baseline commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- The model equations and A2 architecture are unchanged from `a91f45d`.

## 2. Exact commands

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/train_lw_shr_full25.py --pretrained /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/LW_SHR_FULL25_A2_8f6da26/A2 --seed 42 --epochs 25 --num-workers 4
python tools/evaluate_lw_shr_full25_baseline.py --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --environment-tsv /home/duyanhong/sshr-official-25ep-final-retry2-20260815/environment.tsv --status-tsv /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/status.tsv --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --output-dir /home/duyanhong/experiments/LW_SHR_FULL25_A2_8f6da26/C0_reference
python tools/analyze_lw_shr_full25.py --baseline-reference /home/duyanhong/experiments/LW_SHR_FULL25_A2_8f6da26/C0_reference/baseline_reference.json --a2-completion /home/duyanhong/experiments/LW_SHR_FULL25_A2_8f6da26/A2/completion.json --output-dir /home/duyanhong/experiments/LW_SHR_FULL25_A2_8f6da26/report
```

The A2 command is preserved verbatim in `/home/duyanhong/experiments/LW_SHR_FULL25_A2_8f6da26/A2/configs/training_config.json`. No test or LUAD evaluation was run.

## 3. Checkpoint SHA256

- Existing C0-Full25 FINAL: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- A2-Full25 FINAL: `c3109c2c0a355f88c3d35ffa6b6553d4cdf5dc980bd73e104c9589527d4a3f43`

## 4. C0/A2 training-equivalence audit

- Common initialization exact: `True`
- Common-key max absolute difference: `0.0`
- Dataset, seed, batch size, optimizer, LR schedule, loss weights, augmentation, BF16 precision, pretrained weights and FINAL-checkpoint rule match the prior official baseline.
- The only model difference is the frozen A2 Learnable Wavelet Gate at HFRM28_1.
- C0 was not retrained, per user instruction; it is the SHA-locked BCSS seed42 member of the earlier six-run official reproduction.

## 5. Epoch-wise overall metrics

The reused C0 run retained only Epoch25 FINAL. C0 intermediate validation metrics are therefore unavailable and are shown as `N/A`; A2 values are measured at all required epochs.

The archived A0 validation readout was 67.3102 mIoU. Re-evaluating its SHA-locked FINAL checkpoint with this run's exact BF16/TTA diagnostic harness gives 67.3360 (+0.0258 pp). The paired comparison below uses the latter because C0 and A2 must pass through the same evaluator.

| Epoch | C0 mIoU | A2 mIoU | A2 mDice |
|---:|---:|---:|---:|
| 1 | N/A | 49.8508 | 64.2663 |
| 5 | N/A | 66.2169 | 79.4393 |
| 10 | N/A | 66.0394 | 79.2939 |
| 15 | N/A | 66.3557 | 79.5179 |
| 20 | N/A | 67.3488 | 80.3062 |
| 25 | 67.3360 | 67.2508 | 80.2121 |

## 6. Epoch-wise CAM hierarchy

| Epoch | CAM56 | CAM28_1 | CAM28_2 | CAMdeep | Final |
|---:|---:|---:|---:|---:|---:|
| 1 | 30.8618 | 31.3758 | 53.0254 | 53.5693 | 49.8508 |
| 5 | 46.4907 | 65.4829 | 66.1868 | 65.3202 | 66.2169 |
| 10 | 56.8967 | 65.6134 | 65.7294 | 64.6460 | 66.0394 |
| 15 | 61.2687 | 66.3074 | 65.9430 | 64.6715 | 66.3557 |
| 20 | 61.7440 | 67.1196 | 66.6042 | 65.2859 | 67.3488 |
| 25 | 61.4191 | 66.9442 | 66.4556 | 64.9514 | 67.2508 |

## 7. Boundary/interior

| Model | Boundary accuracy | Boundary restricted mIoU | Interior accuracy | Interior restricted mIoU |
|---|---:|---:|---:|---:|
| C0 | 51.5617 | 31.7292 | 85.6834 | 71.5073 |
| A2 | 51.4514 | 31.6711 | 85.6126 | 71.4138 |

## 8. Object size

The historical size statistic is pixel-weighted component recall; size-restricted mIoU is diagnostic only.

| Model | Small recall/mIoU | Medium recall/mIoU | Large recall/mIoU |
|---|---:|---:|---:|
| C0 | 35.0796/17.9441 | 68.4060/45.9489 | 89.4314/78.5776 |
| A2 | 34.8532/17.7540 | 68.4373/46.0681 | 89.3130/78.3758 |

## 9. Per-class IoU

| Model | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.4481 | 70.5762 | 57.8549 | 64.4648 |
| A2 | 76.3759 | 70.4496 | 57.6910 | 64.4868 |

## 10. Filter evolution

| Epoch | dec_lo | dec_hi | Low drift | High drift | Low cosine | High cosine |
|---:|---|---|---:|---:|---:|---:|
| 1 | `[0.6675779819488525, 0.6675779819488525]` | `[0.6675780415534973, -0.6675781011581421]` | 0.05590215 | 0.05590202 | 1.00000000 | 1.00000000 |
| 5 | `[0.5415608286857605, 0.5415633916854858]` | `[0.5416466593742371, -0.5418666005134583]` | 0.23411551 | 0.23384047 | 1.00000000 | 1.00000000 |
| 10 | `[0.43780484795570374, 0.4378085136413574]` | `[0.4379521608352661, -0.43820154666900635]` | 0.38084784 | 0.38046578 | 1.00000012 | 1.00000000 |
| 15 | `[0.3741985857486725, 0.3742024004459381]` | `[0.3743574619293213, -0.374589204788208]` | 0.47080055 | 0.47041473 | 1.00000000 | 1.00000000 |
| 20 | `[0.3389236629009247, 0.33892831206321716]` | `[0.3390837013721466, -0.33930233120918274]` | 0.52068627 | 0.52030861 | 1.00000000 | 1.00000000 |
| 25 | `[0.32686084508895874, 0.3268657922744751]` | `[0.32702088356018066, -0.32723307609558105]` | 0.53774542 | 0.53737259 | 1.00000012 | 1.00000000 |

## 11. Gate evolution

| Epoch | Mean | Std | Spatial std | Channel std | Boundary mean | Interior mean | B-I |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000363 | 0.001265 | 0.00001179 | 0.00126431 | 1.000361 | 1.000363 | -0.00000140 |
| 5 | 1.010985 | 0.044482 | 0.00018072 | 0.04447700 | 1.010953 | 1.010983 | -0.00002958 |
| 10 | 1.012784 | 0.055000 | 0.00015525 | 0.05499639 | 1.012755 | 1.012781 | -0.00002613 |
| 15 | 1.013280 | 0.058882 | 0.00012632 | 0.05887964 | 1.013256 | 1.013278 | -0.00002141 |
| 20 | 1.013508 | 0.061157 | 0.00011394 | 0.06115519 | 1.013486 | 1.013506 | -0.00001936 |
| 25 | 1.013533 | 0.061774 | 0.00010765 | 0.06177278 | 1.013512 | 1.013531 | -0.00001843 |

## 12. Context residual evolution

| Epoch | Raw RMS | Gated RMS | Gated/raw | Boundary ratio | Interior ratio |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.051686 | 0.051926 | 1.004630 | 1.004718 | 1.004620 |
| 5 | 0.209575 | 0.258775 | 1.231362 | 1.230301 | 1.231636 |
| 10 | 0.235637 | 0.311358 | 1.316195 | 1.312057 | 1.316732 |
| 15 | 0.263072 | 0.356814 | 1.350324 | 1.344484 | 1.351391 |
| 20 | 0.294781 | 0.405497 | 1.371503 | 1.364355 | 1.372809 |
| 25 | 0.327112 | 0.451124 | 1.375254 | 1.366786 | 1.376783 |

## 13. Gradient evolution

Recorded `15` preregistered snapshots (steps 1–10 and ends of epochs 1/5/10/20/25). Full values are in `gradient_diagnostics.csv`; all recorded gradients were finite.

## 14. Epoch25 bootstrap CI

- Observed Delta mIoU: `-0.0852 pp`
- Bootstrap mean Delta: `-0.0853 pp`
- 95% CI: `[-0.1879, +0.0097] pp`
- 10,000 paired image-level resamples, seed42; image confusion matrices are summed before recomputing official global mIoU.

## 15. Scientific interpretation

- Final mIoU changed from `67.3360` to `67.2508` (-0.0852 pp).
- Final mDice changed from `80.2743` to `80.2121` (-0.0622 pp).
- CAM28_1 delta: `-0.0890 pp`; boundary accuracy delta: `-0.1103 pp`; interior accuracy delta: `-0.0708 pp`.
- Because the reused baseline retained only FINAL, a relative C0-vs-A2 epoch-wise curve cannot be claimed. This is a documented consequence of not rerunning baseline, not missing A2 data.
- No architecture, optimizer, loss, inference threshold or metric was changed based on validation results.

## 16. Final decision

Decision: `FULL25_NEGATIVE`.

DECISION = FULL25_NEGATIVE
