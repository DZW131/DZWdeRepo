# SSHR: Single-Stage Hierarchical Rectification for Weakly Supervised Histopathology Segmentation (MICCAI 2026)

## RDDR Phase-2A experimental branch

This branch adds **receiver-side context suppression at HFRM28_1 only** to the
pure official A0 base `4e9a288`. It contains no Phase-1 DDA or older innovations.
The frozen implementation is `6f45ac7`; subsequent commits add evaluation and
documentation only. The model adds zero parameters and keeps the original
semantic branch and raw feature unchanged.

- [Architecture contract](docs/rddr_phase2a_architecture_contract.md)
- [Complete validation report](docs/rddr_phase2a_dross_aware_context_suppression_report.md)
- [Verified metric artifacts](audit/results/rddr_phase2a/)

This experiment is BCSS seed42, 25 epochs from the official pretrained weights,
batch20 BF16, **FINAL checkpoint only, validation only**. Do not use the upstream
test-evaluation example below for this experiment. Milestone checkpoints are
diagnostics only, not candidates for model selection.

| Final checkpoint | Validation mIoU (%) | Validation mDice (%) |
|---|---:|---:|
| C0 (reused A0, same-evaluator reference) | 67.3363 | 80.2746 |
| GS (global context scaling) | 67.0918 | 80.0875 |
| RCS (receiver context suppression) | 67.0703 | 80.0749 |

These metrics were cross-checked against the unchanged official inference
function over all 3,418 validation images for each checkpoint. The historical
audit C0 score is not mixed into these deltas; see the
[evaluation provenance notes](docs/rddr_phase2a_evaluation_provenance.md).

### Environment and file map

The executed environment is `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`
(Python 3.10.20, PyTorch 2.11.0+cu128, NumPy 1.23.5), on RTX 5090 D v2. Reuse it;
do not upgrade a live experiment environment. Upstream fresh-environment
instructions are retained below, but are not a claim of an identical runtime.

```text
network/rddr_context.py                 detached normalized JSD + GS/RCS gates
network/resnet38_cls.py                 context-only mode; none remains A0
tools/run_rddr_phase2a_server.sh         GS Full25 then RCS Full25
tools/smoke_rddr_phase2a.py              no-new-parameter / BF16 readiness
tools/replay_rddr_phase0_frozen_populations.py  historical population replay
tools/evaluate_rddr_phase2a_server.sh    validation-only evaluation wrapper
tools/analyze_rddr_phase2a.py            metrics, diagnostics, bootstrap, report
tools/validate_rddr_phase2a_artifacts.py  CPU-only independent cross-check
tests/test_rddr_phase2a*.py              model and evaluation regression tests
```

Data on the server: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/training`
(23,422 parsed training images) and `val/img`, `val/mask` (3,418 matched pairs).
Raw data, pretrained weights, model checkpoints, and large population caches
are not committed to Git.

### Exact server workflow

Run from a clean checkout of this branch. Existing output directories are
immutable; use a new path when explicitly repeating an evaluation.

```bash
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
EXP=/home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7
DATA=/home/duyanhong/reseg-data/raw/BCSS-WSSS
PRETRAINED=/home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
C0=/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth
P0=/home/duyanhong/experiments/RDDR_PHASE0_586f402/formal
P1=/home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/report

# Training already completed: provenance only, do not rerun over existing files.
# bash tools/run_rddr_phase2a_server.sh "$EXP" "$PRETRAINED" "$DATA/training" "$PY"

# Reconstruct historical fixed populations in a fresh process, once.
# Skip if the verified cache and manifest already exist.
"$PY" tools/replay_rddr_phase0_frozen_populations.py \
  --phase0-repo /home/duyanhong/DZWdeRepo-rddr-phase0-586f402 \
  --phase0-dir "$P0" --checkpoint "$C0" --val-root "$DATA/val" \
  --output "$EXP/diagnostics/frozen_phase0_populations"

# Official TTA inference, official metric, mechanism diagnostics and bootstrap.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 bash tools/evaluate_rddr_phase2a_server.sh \
  "$EXP" "$P0" "$P1" "$C0" "$DATA/val" "$PRETRAINED" "$DATA/training" \
  "$PY" "$EXP/report_final" 0 10000

# Independent CPU checks; no model forward, no optimizer, no writes by default.
"$PY" tools/validate_rddr_phase2a_artifacts.py "$EXP/report_final" \
  --phase0-summary "$P0/rddr_phase0_summary.json"

OMP_NUM_THREADS=4 "$PY" -m pytest -q \
  tests/test_rddr_phase2a.py tests/test_rddr_phase2a_analysis.py
```

The evaluator checks unchanged official inference on eight images per model
(zero pixel disagreement), validates source/checkpoint hashes, and uses 10,000
paired image-level bootstrap samples. The original Phase-0 group counts are
checked **for every image**, not just in aggregate. The report includes explicit
notes on native-dtype TTA averaging, historical population backend settings,
and diagnostic metric definitions. No hyperparameter search is performed.

### Inspect results and visualizations

Open the Markdown report for the complete comparison, or inspect the CSV files
for epoch curves and q/gamma trajectories. This audit produces tables, not
additional prediction heatmaps; no visualization-based model selection is used.

```bash
less "$EXP/report_final/rddr_phase2a_dross_aware_context_suppression_report.md"
less "$EXP/report_final/rddr_phase2a_training_curves.csv"
less "$EXP/report_final/rddr_phase2a_q_dynamics.csv"
```

## Upstream documentation (retained)

## Abstract
  <details>
  <summary>Click to expand</summary>

Existing weakly supervised semantic segmentation (WSSS) methods in computational pathology rely on a multi-stage paradigm: class activation map (CAM) generation, offline pseudo-mask refinement, and fully supervised retraining. While established, this decoupled approach presents fundamental limitations. The multi-stage process not only incurs high computational training costs but also suffers from error propagation: local texture biases in shallow CNN layers generate false-positive artifacts that subsequent refinement steps often fail to correct.

To address these persistent challenges through a simple yet highly effective approach, we propose the Single-Stage Hierarchical Rectification (SSHR) framework. Rather than passively refining CAMs post-hoc, our method proactively purifies intermediate feature representations during the forward pass. We introduce a Hierarchical Feature Rectification Module (HFRM) that utilizes deep global semantic context to filter out local anomalies in shallow layers. This mechanism generates high-fidelity activation maps directly within a single training loop.

Experiments on the LUAD-HistoSeg and BCSS datasets demonstrate that SSHR outperforms state-of-the-art multi-stage methods. Furthermore, SSHR reduces training duration by 2 to 5 times. This efficiency minimizes computational overhead and accelerates clinical translation for large-scale histopathology workflows.

**Keywords:** Weakly supervised learning, semantic segmentation, computational pathology, single-stage learning.

  </details>

## Framework


<p align="center">
  <img src="assets/main_flow.png" width="700" alt="WaveDiT architecture">
</p>

## Directory Structure

```text
SSHR/
├── datasets/
│   ├── BCSS-WSSS/
│   │   ├── training/          # training images with image-level labels in filenames
│   │   ├── val/
│   │   │   ├── img/
│   │   │   └── mask/
│   │   └── test/
│   │       ├── img/
│   │       └── mask/
│   └── LUAD-HistoSeg/
│       ├── training/          # training images with image-level labels in filenames
│       ├── val/
│       │   ├── img/
│       │   └── mask/
│       └── test/
│           ├── img/
│           └── mask/
├── init_weights/              # pretrained initialization weights, ignored by git
└── checkpoints/               # training checkpoints, ignored by git
```


## Usage

### Step 1: Download Data and Weights

Download the pretrained classification initialization weight:

- [ImageNet initialization weight](https://drive.google.com/file/d/1Rka2SzqAwxUEFb28tbmiy2anhkkFOnTg/view?usp=drive_link)

Download the datasets:

- [LUAD-HistoSeg dataset](https://drive.google.com/file/d/1lWAeCp6UN30VRVmqv97kA2sJ1Pp2frhC/view?usp=drive_link)
- [BCSS-WSSS dataset](https://drive.google.com/file/d/178eSM9xs5jITt5P2kjaswDlJzwlU5gps/view?usp=drive_link)

Place the initialization weight at:

```text
init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
```

Place the datasets under `datasets/` following the structure above.

### Step 2: Setup Environment

```bash
conda create -n sshr python=3.10 -y
conda activate sshr
pip install -r requirements.txt
pip install mxnet==1.9.1
pip install numpy==1.23.5
```


### Step 3: Train and Evaluate


Run training and evaluation from a Slurm allocation. Set `DATA_ROOT` to the directory containing `LUAD` and `BCSS`.

### Training

```bash
DATA_ROOT=/path/to/weakly_seg_data
DATASET=luad      # use bcss for BCSS
DATASET_DIR=LUAD  # use BCSS for BCSS
SEED=11

python train_sshr.py \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --max_epoches 25 \
  --weights init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --trainroot "${DATA_ROOT}/${DATASET_DIR}/train" \
  --save_folder "checkpoints_${DATASET}_seed${SEED}" \
  --eval_every 0 \
  --save-last-k-checkpoints 0 \
  --n_class 4 \
  --img_size 224 \
  --num_workers 4 \
  --amp-dtype bf16
```

The final checkpoint is saved as `stage1_last.pth`.

### Evaluation

```bash
DATA_ROOT=/path/to/weakly_seg_data
DATASET=luad      # use bcss for BCSS
DATASET_DIR=LUAD  # use BCSS for BCSS
SEED=11

python train_sshr.py \
  --evaluate-only \
  --dataset "${DATASET}" \
  --seed "${SEED}" \
  --weights "checkpoints_${DATASET}_seed${SEED}/stage1_last.pth" \
  --testroot "${DATA_ROOT}/${DATASET_DIR}/test" \
  --n_class 4 \
  --img_size 224 \
  --num_workers 4 \
  --amp-dtype bf16
```

## Acknowledgement

We thank the authors of [ESFAN](https://github.com/OceanPetal/ESFAN), whose codebase provided a valuable foundation for this repository.

