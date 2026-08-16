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
new Innovation 1 candidate is **Semantic-Conditioned Morphology-Preserving
Rectification (SC-MPR)**. It keeps original CH15 and adds bounded, spatially
selective fine/morphology residual proposals verified by read-only deep
semantic conditions.

- `--rectifier hfrm --context-mode ch`: exact A0 (default);
- `--rectifier hfrm --context-mode sc-mpr`: SC-MPR engineering candidate;
- `--rectifier hfrm --context-mode fampr`: archived Full FA-MPR;
- `--rectifier hst`: archived negative-result implementation, isolated from
  both active and archived HFRM-context candidates.

SC-MPR does not change the backbone, GSR branch, CH15, CAM heads, loss,
optimizer, training schedule, inference, or metrics. Its v1.0 forward design
is frozen in code. Engineering review found that exact per-channel spatial
de-meaning makes the SC-MPR residual invisible to SSHR's linear CAM plus global
average-pooling classification objective. Therefore **formal training is not
authorized yet**; see the implementation report for the proof and decision
options.

The earlier HST candidate is archived with three selectable stages:

- A1: progressive-only correction-state propagation;
- A2: target-conditioned stage-specific transitions;
- A3: lightweight hierarchy-token interaction.

Their BCSS seed-42 final-checkpoint results did not improve over A0. Full
FA-MPR likewise failed its validation go/no-go criterion. Both candidates are
archived without deleting their source, tests, or experiment record.

See [`docs/scmpr_implementation_report.md`](docs/scmpr_implementation_report.md)
for the current engineering audit,
[`docs/archive/innovation1_fampr.md`](docs/archive/innovation1_fampr.md) for
the FA-MPR archival record, and
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

SC-MPR is selectable for engineering smoke tests with:

```bash
--rectifier hfrm --context-mode sc-mpr
```

Do not start a formal SC-MPR run until the optimization blocker documented in
`docs/scmpr_implementation_report.md` is resolved and reviewed. Do not combine
`--rectifier hst` with an alternative HFRM context; the parser/model rejects
that combination to keep archived candidates isolated.

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

