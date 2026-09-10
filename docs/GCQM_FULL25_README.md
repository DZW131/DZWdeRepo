# GCQM Final Validation — BCSS Seed42 Full25

## Purpose

This workflow performs the single frozen performance validation authorized after `GCQM_PHASE0_STRONG_GO`. It starts from the official MXNet ResNet38 initialization, trains for exactly 25 epochs/29,275 optimizer steps, seals E25 before reading segmentation ground truth, and compares it with the hardware-matched frozen SSHR B0 endpoint.

The scientific architecture remains the exact `f327b91bba88bdf45391d991a13b214e9ab101ff` GCQM model. Full25 tooling must not modify the architecture, loss, optimizer, augmentation, inference thresholds, or checkpoint policy.

## Environment

- Python environment: `/home/duyanhong/miniconda3/envs/sshr5090`
- GPU: NVIDIA RTX 4090D with BF16
- Official initialization: `/home/duyanhong/sshr-resnet38-init.params`
- Initialization SHA-256: `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`

Install repository requirements if a new environment is used:

```bash
python -m pip install -r requirements.txt
```

## Relevant files

- `network/gcqm.py`: zero-parameter global class-conditioned query mixture.
- `network/gcqm_net.py`: frozen Stage1/CCRA/A/w/B/F model and losses.
- `tools/run_gcqm_full25_bcss_seed42.py`: training-only Full25 runner and E5/E10/E15/E20/E25 mechanism snapshots.
- `tools/eval_gcqm_full25_bcss_seed42.py`: sealed-E25 GCQM/SSHR evaluation, paired bootstrap, complexity, qualitative cases, and report.
- `tests/test_gcqm_full25.py`: frozen Full25 and decision-boundary regression tests.

## Data organization

```text
/home/duyanhong/reseg-data/raw/BCSS-WSSS/
  training/          # Full25 training; flat weak-label filenames
  val/img/           # opened only after E25 seal
  val/mask/          # opened only after E25 seal
```

Training rejects validation, test, and LUAD access. Evaluation rejects every split except BCSS validation and requires exactly 3,418 image/mask pairs.

## Regression tests

```bash
python -m pytest -q
```

## Two-step smoke

Use a new disposable output directory, then delete it before the formal run:

```bash
python tools/run_gcqm_full25_bcss_seed42.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-resnet38-init.params \
  --output-dir /home/duyanhong/experiments/GCQM_Final_Full25_SMOKE \
  --cohort-json /home/duyanhong/experiments/CQRFNet_Phase0_CCRA_Responsibility_BCSS_Seed42/cqrf_phase0_monitor_cohort.json \
  --phase0-gate-json /home/duyanhong/experiments/GCQM_Phase0_Global_Class_Conditioned_Query_Mixture_BCSS_Seed42/gcqm_gate_result.json \
  --smoke-steps 2
```

## Formal training

```bash
python tools/run_gcqm_full25_bcss_seed42.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-resnet38-init.params \
  --output-dir /home/duyanhong/experiments/GCQM_Final_Full25_BCSS_Seed42 \
  --cohort-json /home/duyanhong/experiments/CQRFNet_Phase0_CCRA_Responsibility_BCSS_Seed42/cqrf_phase0_monitor_cohort.json \
  --phase0-gate-json /home/duyanhong/experiments/GCQM_Phase0_Global_Class_Conditioned_Query_Mixture_BCSS_Seed42/gcqm_gate_result.json
```

Only `checkpoints/gcqm_full25_epoch25_final.pth` is a scientific endpoint. E5/E10/E15/E20 files are recovery/audit milestones, never candidates for metric-based selection.

## Final evaluation

Run only after the E25 metadata states `sealed_before_segmentation_evaluation=true`:

```bash
python tools/eval_gcqm_full25_bcss_seed42.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --gcqm-checkpoint /home/duyanhong/experiments/GCQM_Final_Full25_BCSS_Seed42/checkpoints/gcqm_full25_epoch25_final.pth \
  --gcqm-experiment /home/duyanhong/experiments/GCQM_Final_Full25_BCSS_Seed42 \
  --sshr-checkpoint /home/duyanhong/experiments/SSHR_B0_RTX4090D_BCSS_SEED42_FULL25_A0/checkpoints/stage1_last.pth \
  --sshr-experiment /home/duyanhong/experiments/SSHR_B0_RTX4090D_BCSS_SEED42_FULL25_A0
```

The evaluator uses the released `[0.8, 0.9, 0.8, 0.6]` class-presence thresholds, three-view TTA, fixed SSHR `0.6/0.2/0.2` fusion, spatial normalization, identical class mapping, and the same foreground confusion convention. It verifies the archived SSHR metrics and per-image confusions exactly before evaluating GCQM.

## Outputs

```text
checkpoints/   E5/E10/E15/E20 and sealed E25
metrics/       training loss and epoch summaries
mechanism/     fixed train-only mechanism snapshots
evaluation/    aggregate, per-class, per-image, bootstrap, protocol, complexity
visualizations/automatically selected wins/similar/failures
provenance/    source/config/init/dataset/environment/runtime evidence
report/        final Markdown report
```

## Result table

| Model | Seed | Epoch | fg mIoU | mDice | Decision |
|---|---:|---:|---:|---:|---|
| SSHR B0 | 42 | 25 | pending | pending | frozen baseline |
| GCQM | 42 | 25 | pending | pending | pending E25 evaluation |

## Visualization

Qualitative outputs are generated automatically during evaluation. The five largest paired improvements, five closest-to-zero cases, and five largest regressions are selected from all 3,418 images without manual intervention.

## Recovery

Only a pure engineering interruption may use `recovery/latest.pth` with `--resume-recovery`. The recovery payload restores the model, optimizer/global step, Python/NumPy/PyTorch/CUDA RNG, and DataLoader-generator state. Scientific tuning or checkpoint selection remains prohibited.
