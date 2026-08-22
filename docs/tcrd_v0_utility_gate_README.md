# TCRD-v0 Matched 5-Epoch Utility Gate

This directory documents the preregistered BCSS validation-only utility gate for
Tissue-Competitive Reaction-Diffusion Rectification (TCRD-v0). It is a mature
checkpoint continuation screen, not a fresh 25-epoch formal experiment.

## Frozen controls

- Source A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- A0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- Dataset: BCSS training/validation only (`23,422 / 3,418`)
- Branches: `C0`, `D`, `R`, `DR`
- Schedule: one shared `5 x 1171 x 20` manifest with per-sample augmentation
  seeds and per-step model seeds
- Optimization: fresh official `PolyOptimizer`; starting group learning rates
  are derived from epoch `20/25` of the official schedule and replayed over five
  epochs with power `0.9`
- Training: batch 20, image size 224, BF16, seed 42, official weighted
  classification loss only
- Evaluation: all 3,418 validation images at step 0 and epochs 1--5 using the
  official three-view TTA, predicted presence thresholds, hard gate, min-max
  normalization, `0/0.6/0.2/0.2` fusion, and released metric

Test, LUAD, other seeds, checkpoint selection, hyperparameter search, and fresh
25-epoch training are explicitly out of scope.

## Implementation layout

- `network/tcrd_dynamics.py`: frozen SPED and TCER equations
- `network/resnet38_cls_tcrd_gate.py`: official SSHR with dynamics only on the
  main CAM28_1 path; `C0` remains exactly the official path
- `tools/build_utility_schedule.py`: immutable matched schedule
- `tools/preflight_tcrd_gate.py`: real batch20 BF16 protocol and mechanism audit
- `tools/train_tcrd_utility.py`: one branch, five epochs, six validation points
- `tool/infer_tcrd.py`: frozen BCSS validation inference and diagnostics
- `tools/eval_tcrd_utility.py`: paired error analysis, gates, plots, and route
- `tools/run_tcrd_utility_gate.sh`: sequential end-to-end runner

## Verification

Local tests:

```bash
python -m py_compile \
  network/tcrd_dynamics.py network/resnet38_cls_tcrd_gate.py \
  tools/tcrd_common.py tools/build_utility_schedule.py \
  tools/preflight_tcrd_gate.py tools/train_tcrd_utility.py \
  tools/eval_tcrd_utility.py tool/infer_tcrd.py
python -m pytest tests/test_tcrd_*.py -q
```

The server preflight performs no optimizer step. It SHA-locks the A0 checkpoint,
checks exact common parameters for all branches, verifies the schedule and data
counts, runs a real batch20 BF16 official-loss backward pass, audits candidate
optimizer coverage, and checks all numerical mechanism contracts.

## Exact run command

From the repository root in the SSHR environment:

```bash
bash tools/run_tcrd_utility_gate.sh \
  /path/to/BCSS-WSSS/training \
  /path/to/BCSS-WSSS/val \
  /path/to/stage1_last.pth \
  /path/to/TCRD_V0_UTILITY_GATE_<commit> \
  4
```

The branches run sequentially in the fixed order `C0 -> D -> R -> DR`. Each
branch stores only its epoch-5 final checkpoint; validation histories are saved
at all six fixed evaluation points. The runner refuses to overwrite an existing
branch directory or schedule.

## Outputs

The experiment directory contains immutable provenance, the schedule and
preflight audit, four branch directories, machine-readable comparison tables,
five required figures, and:

```text
docs/tcrd_v0_utility_gate_report.md
comparison/route_decision.json
```

The finalizer emits exactly one of the preregistered routes A--E, after which the
experiment stops for human review.
