# HALR-v1 Full-Model Experiment

## Purpose and frozen scope

HALR-v1 (Hierarchy-Adaptive Localization Rectification) is a training-only
extension of clean official SSHR. It adds Cross-View Localization Equivariance
(CVLE) and Reliability-Adaptive Hierarchical Distillation (RAHD) between
CAM28_1 and CAMdeep. The SSHR model, HFRM, CAM heads and official inference are
unchanged, and HALR adds zero trainable parameters.

The formal experiment is frozen to BCSS seed42, 25 epochs, effective base batch
20, BF16 and epoch25 FINAL evaluation. Validation is not used for checkpoint
selection.

## Environment

Use the existing server environment:

```bash
source /home/duyanhong/miniconda3/etc/profile.d/conda.sh
conda activate sshr5090
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## Relevant repository files

| Path | Role |
|---|---|
| `network/resnet38_cls.py` | Unmodified official SSHR training/inference model |
| `tools/halr_objectives.py` | Exact paired flips, present-class probabilities, CVLE and RAHD |
| `tools/preflight_halr_v1.py` | Real-batch BF16 engineering preflight; no optimizer step |
| `tools/train_halr_v1_25ep.py` | Frozen fresh 25-epoch training and checkpoint logging |
| `tool/infer_halr.py` | Official BCSS inference plus post-forward epoch25 diagnosis |
| `tools/eval_halr_v1.py` | A0/HALR FINAL-only validation comparison and report generation |
| `tests/test_halr_v1.py` | CPU objective and protocol-contract tests |

## Data organization

The server paths used by the formal run are:

```text
/home/duyanhong/reseg-data/raw/BCSS-WSSS/
├── training/                  # 23,422 weak-label image patches
└── val/
    ├── img/                   # 3,418 images
    └── mask/                  # 3,418 masks; evaluation/diagnosis only
```

ImageNet initialization:

```text
/home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
```

## Local tests

```bash
python -m pytest tests/test_halr_v1.py -q
```

## Minimal server preflight

```bash
python tools/preflight_halr_v1.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output /home/duyanhong/experiments/HALR_V1_FULL_25EP_SEED42_<commit>/provenance/preflight.json \
  --num-workers 4
```

The preflight checks paired CAM shapes, exact flip inversion, single-class zero
losses, detached reliability/teacher tensors, both hierarchy gradients, zero
new model parameters, pretrained compatibility and BF16 finiteness. It never
calls `optimizer.step()`.

## Formal training

```bash
python tools/train_halr_v1_25ep.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --output-dir /home/duyanhong/experiments/HALR_V1_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

The script saves epochs 5/10/15/20/25. `epoch25_final.pth` is the only primary
checkpoint. Training never evaluates validation or test.

## Epoch25 validation evaluation

```bash
python tools/eval_halr_v1.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --a0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --halr-checkpoint /home/duyanhong/experiments/HALR_V1_FULL_25EP_SEED42_<commit>/checkpoints/epoch25_final.pth \
  --experiment-dir /home/duyanhong/experiments/HALR_V1_FULL_25EP_SEED42_<commit> \
  --num-workers 4
```

HALR evaluation is the exact official single-network inference protocol: 3-way
TTA, BCSS class thresholds, hard class gate, per-class min-max normalization,
0/0.6/0.2/0.2 fusion and released `scores()`.

## Visualization

Formal evaluation writes:

```text
figures/training_losses.png
figures/teacher_dynamics.png
```

These are generated only from the frozen training history and do not affect
checkpoint selection or the decision.

## Result record

| Model | Dataset / split | Epoch | mIoU | mDice | Decision |
|---|---|---:|---:|---:|---|
| SSHR A0 | BCSS validation | 25 FINAL | pending | pending | reference |
| HALR-v1 | BCSS validation | 25 FINAL | pending | pending | pending |

The completed report will be written to
`docs/halr_v1_full_25ep_report.md`, with machine-readable metrics under the
experiment directory's `validation/` folder.

## Stop boundary

This run does not authorize BCSS test, LUAD, seeds 11/17, ablations, coefficient
or ramp sweeps, new teacher rules, or HALR-v2. If validation ΔmIoU is below
+0.05 pp, the protocol stops for manual review.
