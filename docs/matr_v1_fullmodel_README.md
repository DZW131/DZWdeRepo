# MATR-v1 Full-Model Experiment

## Purpose and frozen scope

MATR-v1 (Multi-Prototype Adaptive Tissue Rectification) modifies only the
official SSHR HFRM28_1/CAM28_1 path with two frozen innovations:

- OT-MTR: two centered morphology modes per class with detached balanced
  Sinkhorn assignment from image-label-gated CAM seeds.
- SACR: original trainable CH15 plus a zero-anchored pure-PyTorch dynamic sparse
  context correction.

All other SSHR branches, loss weights and official inference settings remain
unchanged. The formal experiment is BCSS seed42, 25 epochs and epoch25 FINAL
only; validation never selects a checkpoint.

## Environment

```bash
source /home/duyanhong/miniconda3/etc/profile.d/conda.sh
conda activate sshr5090
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## Relevant repository files

| Path | Role |
|---|---|
| `network/matr_multiprototype_head.py` | Centered two-mode log-mean-exp CAM28_1 head |
| `tool/sinkhorn.py` | Detached FP32 balanced 20-iteration Sinkhorn plan |
| `tools/matr_objectives.py` | Present-class seed selection and OT loss |
| `network/matr_sparse_context.py` | Pure-PyTorch FP32 sparse dynamic/reference sampling |
| `network/matr_hfrm28.py` | Original GSR/CH15 plus scaled zero-anchored SACR |
| `network/resnet38_cls_matr.py` | MATR training and inference models |
| `tools/preflight_matr.py` | Real batch20 BF16 engineering preflight; no optimizer step |
| `tools/train_matr_25ep.py` | Frozen fresh 25-epoch training/checkpoint logging |
| `tool/infer_matr.py` | Official fusion and standalone CAM28_1 evaluation |
| `tools/eval_matr.py` | A0/MATR FINAL-only validation comparison and report |
| `tests/test_matr_*.py` | Head, OT, SACR and protocol-contract tests |

## Data organization

```text
/home/duyanhong/reseg-data/raw/BCSS-WSSS/
├── training/                  # 23,422 image-level weak-label patches
└── val/
    ├── img/                   # 3,418 images
    └── mask/                  # 3,418 masks; final evaluation only
```

ImageNet initialization:

```text
/home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
```

No trained SSHR/MATR/other innovation checkpoint may initialize training.

## Local tests

```bash
python -m pytest \
  tests/test_matr_head.py \
  tests/test_matr_ot.py \
  tests/test_matr_sparse_context.py \
  tests/test_matr_protocol.py -q
```

## Minimal server preflight

```bash
python tools/preflight_matr.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output /home/duyanhong/experiments/MATR_V1_FULL_25EP_SEED42_<commit>/provenance/preflight.json \
  --num-workers 4
```

The preflight verifies CAM shapes, pretrained/optimizer coverage, parameter
budget, Sinkhorn marginals, exact zero-mode/base-head identity, exact initial
SACR zero residual, OT `D_raw` gradient, SACR predictor gradient, finite
`beta_adapt` gradient and absence of dense segmentation GT. It performs no
optimizer step.

## Formal training

```bash
python tools/train_matr_25ep.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output-dir /home/duyanhong/experiments/MATR_V1_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

The script saves epochs 5/10/15/20/25; `epoch25_final.pth` is the only primary
checkpoint. No validation/test inference occurs during training.

## Epoch25 validation evaluation

```bash
python tools/eval_matr.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --a0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --matr-checkpoint /home/duyanhong/experiments/MATR_V1_FULL_25EP_SEED42_<commit>/checkpoints/epoch25_final.pth \
  --experiment-dir /home/duyanhong/experiments/MATR_V1_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

The evaluation uses the exact official deep thresholds, hard class gate,
3-way TTA, per-class min-max normalization, 0/0.6/0.2/0.2 fusion and released
metric. It additionally reports standalone CAM28_1 with the same class gate.

## Visualization

Formal evaluation generates:

```text
figures/training_losses.png
figures/prototype_dynamics.png
figures/sacr_dynamics.png
```

## Parameter budget

| Model | Parameters | Added | Overhead |
|---|---:|---:|---:|
| SSHR A0 | 112,709,714 | — | — |
| MATR-v1 | 112,766,830 | 57,116 | 0.050675% |

## Result record

| Model | Dataset / split | Epoch | mIoU | mDice | Decision |
|---|---|---:|---:|---:|---|
| SSHR A0 | BCSS validation | 25 FINAL | 67.3283 | 80.2683 | reference |
| MATR-v1 | BCSS validation | 25 FINAL | 66.9280 | 80.0081 | `MATR_V1_NO_CLEAR_GAIN` |

MATR-v1 changed mIoU by -0.4003 pp and mDice by -0.2602 pp. Its standalone
CAM28_1 regressed by 2.5286 pp mIoU. At epoch25, mode-pair cosine similarity
remained 0.99997556–0.99999976 and the SACR residual was only 0.4218% of the
original CH15 RMS, so neither intended mechanism became functionally strong.

The completed report is available at `docs/matr_v1_full_25ep_report.md`, with
machine-readable evidence under `results/matr_v1_full_25ep/`. The primary
checkpoint remains on the server at
`/home/duyanhong/experiments/MATR_V1_FULL_25EP_SEED42_7aefd5a/checkpoints/epoch25_final.pth`
(SHA256 `50edaff87955f991a9ff0c1ada0cc4fd012f964c6914b2c4ba61f5886ca046fe`).

## Stop boundary

This run does not authorize BCSS test, LUAD, seeds 11/17, ablations, mode/
lambda/epsilon/offset sweeps, diversity loss, alternate operators or MATR-v2.
Any follow-up requires the frozen success gate and separate approval.
