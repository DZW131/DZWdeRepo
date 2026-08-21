# S²HR-v1 Full Model Reproduction Guide

S²HR-v1 reconstructs only SSHR's HFRM28_1 with two frozen mechanisms:

- BPS-CH reduces the original trainable CH15 response near detached deep-semantic transitions.
- SPSR backprojects the detached deep-vs-raw-CAM28_1 present-class discrepancy through the detached CAM28_1 classifier directions.

HFRM56, HFRM28_2, ResNet38, all CAM heads, the four classification losses, optimizer, schedule, augmentation, official inference and released metric remain unchanged.

## Environment

Use the existing SSHR environment. The formal run records Python, PyTorch, CUDA and GPU versions in `provenance/manifest.json`.

```bash
conda activate sshr5090
pip install -r requirements.txt
pip install mxnet==1.9.1 numpy==1.23.5
```

The formal server uses:

```text
/home/duyanhong/miniconda3/envs/sshr5090/bin/python
```

## Repository structure

```text
network/s2hfrm28_1.py          # frozen BPS-CH + SPSR equations
network/resnet38_cls_s2hr.py   # official SSHR with only HFRM28_1 replaced
tool/infer_s2hr.py             # shared A0/S²HR official BCSS postprocessing
tools/preflight_s2hr.py        # minimal two-batch no-step preflight
tools/train_s2hr_25ep.py       # fresh 25-epoch seed42 training
tools/eval_s2hr.py             # final-only A0 vs S²HR validation/report
tests/test_s2hr_preflight.py   # identity, presence, boundary and isolation tests
```

## Data organization

```text
BCSS-WSSS/
├── training/   # 23,422 weak-label tiles encoded in filenames
└── val/
    ├── img/   # 3,418 images from 22 slides
    └── mask/  # 3,418 masks; evaluation only
```

No validation mask enters model forward or training. BCSS test and LUAD are outside this execution.

## Local tests

```bash
python -m pytest -q tests/test_s2hr_preflight.py
```

## Minimal server preflight

```bash
python tools/preflight_s2hr.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output /path/to/experiment/provenance/preflight.json \
  --num-workers 4
```

This builds the released optimizer but performs no optimizer step. It checks two real batch20 BF16 forwards, finite losses, exact zero-gamma identity, training/inference presence logic, fixed boundary-band logic, pretrained compatibility and dense-GT exclusion.

## Fresh 25-epoch training

```bash
python -u tools/train_s2hr_25ep.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output-dir /home/duyanhong/experiments/S2HR_V1_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

The primary checkpoint is `checkpoints/epoch25_final.pth`. Epoch 5/10/15/20 checkpoints are archival only; no best checkpoint is created or selected.

## Final validation and metric evaluation

```bash
python -u tools/eval_s2hr.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --a0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --s2hr-checkpoint /home/duyanhong/experiments/S2HR_V1_FULL_25EP_SEED42_<commit>/checkpoints/epoch25_final.pth \
  --experiment-dir /home/duyanhong/experiments/S2HR_V1_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

Both checkpoints use identical three-way TTA, `[0.8, 0.9, 0.8, 0.6]` class thresholds, hard class gate, per-class min-max, `0/0.6/0.2/0.2` fusion and released `iouutils.scores()`.

## Visualizations

The evaluation command automatically generates:

```text
figures/training_losses.png
figures/mechanism_trajectory.png
```

## Results

The formal run writes the populated comparison to:

```text
validation/final_comparison.json
docs/s2hr_v1_fullmodel_25ep_report.md
```

| Model | Epoch | mIoU | mDice | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SSHR A0 seed42 | 25 | 67.3283 | 80.2683 | 76.4494 | 70.5721 | 57.8272 | 64.4646 |
| S²HR-v1 Full | 25 | 67.0500 | 80.0680 | 76.3745 | 70.1144 | 57.7191 | 63.9919 |

The frozen validation delta is **-0.2784 pp mIoU** and **-0.2003 pp
mDice**, giving `S2HR_FULLMODEL_NO_CLEAR_GAIN`. See the
[complete report](s2hr_v1_fullmodel_25ep_report.md) and the machine-readable
[`final_comparison.json`](../results/s2hr_v1_bcss_seed42/final_comparison.json).

## Resume and failure boundaries

- A failed preflight must be fixed before training.
- A failed/interrupted formal output directory is preserved and never overwritten.
- This runner intentionally has no automatic best-model selection or hyperparameter fallback.
- After the epoch25 validation decision, stop. Ablations, other seeds, LUAD and test require a new approved protocol.
