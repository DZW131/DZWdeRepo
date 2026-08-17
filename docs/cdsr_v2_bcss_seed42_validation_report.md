# CDSR-v2 BCSS Seed42 Validation Report

## 1. Scope and decision

This was the frozen formal development experiment for Full CDSR-v2 on BCSS:
seed 42, 25 epochs, BF16, and FINAL-checkpoint evaluation. Training completed
all 29,275 optimizer steps. Evaluation during training was disabled and the
BCSS test set was not evaluated.

| Model | Checkpoint rule | Validation mIoU (%) | Validation mDice (%) | Delta mIoU (pp) | Delta mDice (pp) |
|---|---|---:|---:|---:|---:|
| Frozen A0 | Epoch 25 FINAL | 67.3102 | 80.2563 | — | — |
| Full CDSR-v2 | Epoch 25 FINAL | **67.3412** | **80.2791** | **+0.0310** | **+0.0228** |

**Decision: Neutral.** The observed +0.0310 mIoU is inside the preregistered
Neutral interval (`67.1102 < mIoU < 67.5102`) and is far below the +0.4 pp Go
threshold. BCSS seed42 test evaluation therefore remains **locked**. No test,
LUAD, seed 11/17, or ablation run was started.

## 2. Exact training command

The model was trained from repository commit
`08286a8da7fd7d62311ba35a1c88b2d24f3a017a` using:

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python -u train_sshr.py \
  --dataset bcss \
  --seed 42 \
  --max_epoches 25 \
  --batch_size 20 \
  --num_workers 4 \
  --n_class 4 \
  --img_size 224 \
  --network network.resnet38_cls \
  --rectifier hfrm \
  --context-mode ch \
  --rectification-mode cdsr \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --valroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --testroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/test \
  --lr 0.01 \
  --wt_dec 0.0005 \
  --loss_w_56 0.10 \
  --loss_w_28_1 0.15 \
  --loss_w_28_2 0.25 \
  --loss_w_deep 0.50 \
  --train_cls_thr 0.20 \
  --cam_w_28_1 0.60 \
  --cam_w_28_2 0.20 \
  --cam_w_deep 0.20 \
  --amp_dtype bf16 \
  --eval_every 0 \
  --save_checkpoints \
  --save_last_k_checkpoints 0 \
  --save_folder /home/duyanhong/experiments/EXP_CDSR_V2_BCSS_SEED42_FINAL25 \
  --checkpoint_name stage1_last.pth
```

The `testroot` argument is present only because it is part of the frozen
training interface. `eval_every=0` produced exactly zero evaluation calls
during training. The later validation program accepts a validation root only
and has no test-root argument.

Frozen protocol evidence:

- Parsed training samples: 23,422; validation images/masks: 3,418/3,418.
- Max steps: 29,275; completed steps: 29,275.
- Official loss weights: 0.10/0.15/0.25/0.50.
- Official inference fusion: CAM56/CAM28_1/CAM28_2/CAMdeep =
  0/0.6/0.2/0.2, with BCSS presence thresholds 0.8/0.9/0.8/0.6.
- Pretrained SHA256:
  `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`.
- Environment: Python 3.10.20, PyTorch 2.11.0+cu128, cuDNN 9.19.0,
  NVIDIA GeForce RTX 5090 D v2.

## 3. FINAL checkpoint

- Path:
  `/home/duyanhong/experiments/EXP_CDSR_V2_BCSS_SEED42_FINAL25/stage1_last.pth`
- Size: 451,130,915 bytes.
- SHA256:
  `f0483a22e3332a630b92b7bf8fc31920521e7fb0ed1dc49ffdfa54084fb52395`.
- Checkpoint files retained by this experiment: one (`stage1_last.pth`).
- Selection: none; this is the Epoch 25 FINAL checkpoint.

## 4. Validation metrics

The released `tool.infer_fun.infer()` and `tool.iouutils.scores()` were used
unchanged for the official fused result. An independent observation pass
reproduced the complete official TTA/fusion pipeline with absolute mIoU and
mDice differences of exactly 0.0.

| BCSS class ID | IoU (%) | Dice (%) |
|---:|---:|---:|
| 0 | 76.5356 | 86.7084 |
| 1 | 70.5959 | 82.7639 |
| 2 | 58.0761 | 73.4787 |
| 3 | 64.1573 | 78.1656 |
| **Mean** | **67.3412** | **80.2791** |

## 5. Stage CAM quality

Each single-scale score uses the same three-way TTA, deep-logit class-presence
decision, per-CAM spatial normalization, background construction, and official
metric as the frozen fused inference. No threshold or fusion search was run.

| Prediction | Validation mIoU (%) | Validation mDice (%) |
|---|---:|---:|
| CAM56 | 61.5078 | 75.7503 |
| CAM28_1 | 67.0447 | 80.0604 |
| CAM28_2 | 66.5476 | 79.7368 |
| CAMdeep | 64.9354 | 78.5319 |
| Official fused | **67.3412** | **80.2791** |

## 6. Alpha and gamma trajectories

Values below are epoch-end parameter values. Stage 1/2/3 correspond to
F56/F28_1/F28_2. The two alpha values are the same shared parameter objects at
all three stages.

| Epoch | alpha_sem | alpha_ctx | S1 gamma_sem | S1 gamma_ctx | S2 gamma_sem | S2 gamma_ctx | S3 gamma_sem | S3 gamma_ctx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.100000 | 0.100000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | 0.111239 | 0.110029 | 0.058071 | 0.098864 | 0.158717 | 0.238314 | 0.390936 | 0.794500 |
| 2 | 0.122320 | 0.119547 | 0.132430 | 0.227366 | 0.393002 | 0.716016 | 0.448036 | 0.957876 |
| 3 | 0.132937 | 0.127575 | 0.249811 | 0.465042 | 0.528140 | 1.113742 | 0.472296 | 1.015100 |
| 4 | 0.143239 | 0.135046 | 0.372361 | 0.788424 | 0.584495 | 1.340441 | 0.492707 | 1.061575 |
| 5 | 0.153225 | 0.142573 | 0.446035 | 1.010877 | 0.601377 | 1.447284 | 0.508842 | 1.100348 |
| 6 | 0.162839 | 0.150345 | 0.492193 | 1.140595 | 0.604618 | 1.497542 | 0.520516 | 1.128716 |
| 7 | 0.171979 | 0.157804 | 0.527914 | 1.241208 | 0.603157 | 1.526209 | 0.532285 | 1.162082 |
| 8 | 0.180648 | 0.165056 | 0.557528 | 1.325272 | 0.599907 | 1.545574 | 0.540467 | 1.186150 |
| 9 | 0.188793 | 0.172065 | 0.581552 | 1.395466 | 0.597098 | 1.563914 | 0.545338 | 1.200174 |
| 10 | 0.196401 | 0.178746 | 0.601019 | 1.445180 | 0.593361 | 1.576001 | 0.550195 | 1.214707 |
| 11 | 0.203492 | 0.185295 | 0.616844 | 1.482275 | 0.588697 | 1.578188 | 0.553320 | 1.224159 |
| 12 | 0.210045 | 0.191463 | 0.629681 | 1.506398 | 0.584295 | 1.582968 | 0.556081 | 1.234261 |
| 13 | 0.216060 | 0.197051 | 0.641207 | 1.522020 | 0.579670 | 1.585950 | 0.558819 | 1.246155 |
| 14 | 0.221560 | 0.202377 | 0.652341 | 1.536743 | 0.576266 | 1.588383 | 0.559239 | 1.249719 |
| 15 | 0.226537 | 0.207368 | 0.662996 | 1.546296 | 0.572085 | 1.587643 | 0.558912 | 1.250767 |
| 16 | 0.230994 | 0.211724 | 0.672086 | 1.554038 | 0.568539 | 1.586811 | 0.559825 | 1.255947 |
| 17 | 0.234943 | 0.215584 | 0.681398 | 1.558947 | 0.566027 | 1.588746 | 0.561087 | 1.262744 |
| 18 | 0.238413 | 0.218930 | 0.690231 | 1.564217 | 0.563668 | 1.591732 | 0.561762 | 1.267844 |
| 19 | 0.241410 | 0.221891 | 0.698146 | 1.569071 | 0.561491 | 1.591483 | 0.562038 | 1.271657 |
| 20 | 0.243965 | 0.224513 | 0.705544 | 1.572054 | 0.559059 | 1.589987 | 0.562263 | 1.274572 |
| 21 | 0.246054 | 0.226628 | 0.712121 | 1.573386 | 0.557067 | 1.588807 | 0.562368 | 1.278014 |
| 22 | 0.247693 | 0.228273 | 0.717723 | 1.575542 | 0.555586 | 1.587625 | 0.562635 | 1.280700 |
| 23 | 0.248892 | 0.229489 | 0.722260 | 1.576874 | 0.554401 | 1.586533 | 0.562620 | 1.282390 |
| 24 | 0.249641 | 0.230247 | 0.725174 | 1.577746 | 0.553612 | 1.585193 | 0.562857 | 1.283936 |
| 25 | **0.249915** | **0.230530** | **0.726249** | **1.578166** | **0.553193** | **1.584430** | **0.562855** | **1.284530** |

Both shared selectivity parameters moved smoothly away from their 0.10
initialization. All first-batch epoch diagnostics and the full-validation
mechanism audit were finite.

## 7. Final Need, gates, and effective residuals

These statistics aggregate all 3,418 validation images at the FINAL
checkpoint. `Gated feature RMS` is `RMS(G * F_branch)`; `effective residual
RMS` is the actual injected term `RMS(gamma * G * F_branch)`.

| Stage | Need mean/std | Need p10/p50/p90 | G_sem mean/std | G_ctx mean/std | gamma_sem | gamma_ctx |
|---|---|---|---|---|---:|---:|
| F56 | 0.579361 / 0.262664 | 0.203837 / 0.606203 / 0.903291 | 0.894876 / 0.065644 | 0.903030 / 0.060552 | 0.726249 | 1.578166 |
| F28_1 | 0.451883 / 0.295342 | 0.050393 / 0.436638 / 0.865010 | 0.863017 / 0.073810 | 0.873643 / 0.068085 | 0.553193 | 1.584430 |
| F28_2 | 0.298766 / 0.241581 | 0.007468 / 0.274071 / 0.649977 | 0.824751 / 0.060375 | 0.838345 / 0.055692 | 0.562855 | 1.284530 |

| Stage | Gated semantic feature RMS | Gated context feature RMS | Effective semantic residual RMS | Effective context residual RMS |
|---|---:|---:|---:|---:|
| F56 | 0.213493 | 0.398534 | 0.155049 | 0.628952 |
| F28_1 | 0.123195 | 0.360601 | 0.068151 | 0.571346 |
| F28_2 | 0.139259 | 0.336445 | 0.078383 | 0.432173 |

Final shared values were `alpha_sem=0.249915` and `alpha_ctx=0.230530`.
Every Need map, gate, and effective branch tensor was finite.

## 8. Runtime and memory

| Phase | Wall time | Peak CUDA allocated | Peak CUDA reserved |
|---|---:|---:|---:|
| 25-epoch training | 2,746.72 s (45 min 46.72 s) | 4.66 GiB | 5.01 GiB |
| Official fused validation | 52.51 s | — | — |
| Four-scale CAM audit | 72.58 s | — | — |
| Full-validation mechanism audit | 8.48 s | — | — |
| Entire validation process | — | 2.07 GiB | 2.51 GiB |

Runtime is observational: another user had a concurrent process on the same
GPU, although available memory remained sufficient and no OOM or numerical
error occurred.

## 9. Reproducibility artifacts

- `audit/results/cdsr_v2_bcss_seed42_experiment_config.json`
- `audit/results/cdsr_v2_bcss_seed42_training_summary.json`
- `audit/results/cdsr_v2_bcss_seed42_diagnostics.jsonl`
- `audit/results/cdsr_v2_bcss_seed42_validation_result.json`

The validation result explicitly records `test_evaluated=false`. The FINAL
checkpoint remains on the 5090 server; it is not committed because it is
451 MB, but its exact path, size, and SHA256 are fixed above.
