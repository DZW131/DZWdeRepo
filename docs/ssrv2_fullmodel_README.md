# SSR-v2 Full-Model Reproduction Guide

SSR-v2 adds Distillation-Aligned Spatial Semantic Rectification to the clean
official SSHR A0 baseline. Only HFRM28_1 changes: PCSD aligns the raw CAM28_1
present-class spatial distribution to a detached deep teacher, while PTCR
executes the detached discrepancy along shared CAM28_1 classifier directions
with a positive-only learned scale. Original GSR and CH15 remain unchanged.

The frozen formal experiment is BCSS seed42, fresh ImageNet-pretrained training,
25 epochs and epoch25 FINAL evaluation. It contains no BPS, boundary module,
validation checkpoint selection or hyperparameter sweep.

## Environment

The formal server interpreter is:

```text
/home/duyanhong/miniconda3/envs/sshr5090/bin/python
```

Install the baseline requirements and MXNet conversion dependency when building
a new environment:

```bash
conda create -n sshr5090 python=3.10 -y
conda activate sshr5090
pip install -r requirements.txt
pip install mxnet==1.9.1 numpy==1.23.5
```

## Repository structure

```text
network/hfrm28_1_ssrv2.py      # PCSD/PTCR and fixed epoch ramp
network/resnet38_cls_ssrv2.py  # clean SSHR-derived SSR-v2 network
tool/infer_ssrv2.py            # identical official postprocess for A0/SSR-v2
tools/preflight_ssrv2.py       # mandatory minimal BF16/build/gradient audit
tools/train_ssrv2_25ep.py      # frozen fresh 25-epoch trainer
tools/eval_ssrv2.py            # epoch25 FINAL A0/SSR-v2 validation comparison
tests/test_ssrv2_preflight.py  # CPU design and safety contracts
```

## Data organization

```text
BCSS-WSSS/
├── training/                  # 23,422 parsed weak-label training images
├── val/
│   ├── img/                   # 3,418 validation images
│   └── mask/                  # validation-only evaluation/diagnosis
└── test/                      # forbidden in this formal development run
```

Training filenames provide image-level labels. Segmentation masks never enter
the training model or loss.

## Local checks

```bash
python -m pytest -q tests/test_ssrv2_preflight.py
python -m compileall -q \
  network/hfrm28_1_ssrv2.py network/resnet38_cls_ssrv2.py \
  tool/infer_ssrv2.py tools/preflight_ssrv2.py \
  tools/train_ssrv2_25ep.py tools/eval_ssrv2.py
```

## Mandatory server preflight

```bash
python -u tools/preflight_ssrv2.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/reseg-data/raw/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output /home/duyanhong/experiments/SSRV2_FULL_25EP_SEED42_<commit>/provenance/preflight.json \
  --num-workers 4
```

This builds the released optimizer but performs no optimizer step. It verifies
two batch20 BF16 forwards, exact epoch1 A0 identity, detached teacher/PTCR paths,
single-present zero behavior, positive gamma, pretrained compatibility and
beta optimizer coverage.

## Formal training

```bash
python -u tools/train_ssrv2_25ep.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/reseg-data/raw/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output-dir /home/duyanhong/experiments/SSRV2_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

The trainer saves only epoch05/10/15/20/25, with `epoch25_final.pth` as the
primary checkpoint. It never evaluates validation during training.

## Epoch25 FINAL inference and evaluation

```bash
python -u tools/eval_ssrv2.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --a0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --ssrv2-checkpoint /home/duyanhong/experiments/SSRV2_FULL_25EP_SEED42_<commit>/checkpoints/epoch25_final.pth \
  --experiment-dir /home/duyanhong/experiments/SSRV2_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

Both models use BF16, the same file ordering, official 3-way TTA, class-presence
thresholds, hard gate, per-class min-max normalization, 0/0.6/0.2/0.2 fusion and
released `iouutils.scores()`.

## Visualization

Evaluation automatically generates:

```text
figures/training_losses.png
figures/mechanism_trajectory.png
```

## Results

| Model | Dataset / split | Epoch | mIoU | mDice | Decision |
|---|---|---:|---:|---:|---|
| SSHR A0 | BCSS validation | 25 FINAL | 67.3283 | 80.2683 | reference |
| SSR-v2 Full | BCSS validation | 25 FINAL | 66.8575 | 79.9354 | `SSRV2_FULLMODEL_NO_CLEAR_GAIN` |

SSR-v2 changed mIoU by −0.4708 pp and mDice by −0.3328 pp. C3 IoU
regressed by −1.3625 pp, triggering `SSRV2_CLASS_REGRESSION_REVIEW`.
At epoch25, gamma_spatial reached 0.141493. The validation-only GT-present
diagnostic found deep spatial accuracy 84.4096% and raw CAM28_1 accuracy
85.0509%, so the trained student exceeded the detached teacher by 0.6414 pp.

The completed report is written to `docs/ssrv2_full_25ep_report.md`, with
machine-readable metrics under `results/ssrv2_full_25ep/` and the primary
checkpoint retained on the server at
`/home/duyanhong/experiments/SSRV2_FULL_25EP_SEED42_04e4631/checkpoints/epoch25_final.pth`
(SHA256 `34265e42164f85dc5a59dcadaf56685bd1c34a89ab71300005dbc6d51c4ea6c3`).

## Stop boundary

After epoch25 validation, stop. Test, LUAD, seeds 11/17, ablations, SSR-v3 and
lambda/gamma/ramp sweeps require a separate reviewed protocol.
