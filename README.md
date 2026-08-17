# SSHR: Single-Stage Hierarchical Rectification for Weakly Supervised Histopathology Segmentation (MICCAI 2026)

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

## Research Development

The official SSHR implementation remains the exact default A0 baseline. The
current Innovation 1 candidate is **Cross-Hierarchy Disagreement-Guided
Selective Rectification (CDSR)**. It reuses the existing raw hierarchy CAMs to
compute a detached analytical Need map and only attenuates the original GSR
and CH15 residuals where rectification is less necessary.

- `--rectifier hfrm --context-mode ch`: exact A0 (default);
- `--rectifier hfrm --context-mode ch --rectification-mode cdsr`: Full CDSR;
- `--rectifier hfrm --context-mode fampr`: Full FA-MPR;
- `--rectifier hst`: archived negative-result implementation, isolated from
  current Innovation 1 and retained only for reproducibility.

CDSR adds only six learnable alpha logits and no new classifier, learned
uncertainty head, or loss. Its Phase-0 frozen signal audit passed as a regular
Go, but its 20-real-step implementation readiness audit failed because both
F28_2 alpha logits were indistinguishable from matched weight-decay-only
shadows. Consequently, the 25-epoch CDSR experiment is intentionally blocked
pending review.

The earlier FA-MPR and HST candidates are archived negative results. HST keeps
three selectable stages:

- A1: progressive-only correction-state propagation;
- A2: target-conditioned stage-specific transitions;
- A3: lightweight hierarchy-token interaction.

Their BCSS seed-42 final-checkpoint results did not improve over A0, so HST is
not part of the active Innovation 1 path. The source and tests remain available
to preserve the complete experimental record.

See
[`docs/cdsr_need_signal_feasibility.md`](docs/cdsr_need_signal_feasibility.md)
for the frozen Phase-0 evidence,
[`docs/cdsr_implementation_readiness_report.md`](docs/cdsr_implementation_readiness_report.md)
for the implementation stop decision, and
[`docs/innovation1_hst_migration.md`](docs/innovation1_hst_migration.md) for the
archived HST record.

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
├── network/cdsr/              # frozen analytical Need and six-scalar gates
├── tools/                     # CDSR audits, readiness smoke, and profiling
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

The formal CDSR command would add the following mode selection:

```bash
--rectifier hfrm --context-mode ch --rectification-mode cdsr
```

Do **not** start that 25-epoch run while the current readiness report is FAIL.
CDSR cannot be combined with FA-MPR or archived HST; the parser/model rejects
those combinations.

### CDSR readiness reproduction

```bash
python -m pytest -q

python tools/check_cdsr_a0_compatibility.py \
  --checkpoint /path/to/A0/stage1_last.pth \
  --output-json audit/results/cdsr_a0_compatibility.json

python tools/smoke_cdsr.py \
  --train-root /path/to/BCSS-WSSS/training \
  --weights init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --dataset bcss --batch-size 20 --steps 20 --formal-epochs 25 \
  --image-size 224 --seed 42 \
  --output-json audit/results/cdsr_readiness_smoke.json

python tools/profile_cdsr.py \
  --batch-size 20 --image-size 224 --warmup 3 --iterations 10 \
  --output-json audit/results/cdsr_resource_profile.json
```

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

