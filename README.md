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

## OSMF-v1.0 research candidate

`research/osmf-v1` starts directly from the frozen official A0 commit and is
isolated from the archived Phase-0/Phase-0B routing experiments. OSMF inserts a
semantic/morphology factorizer only at the 512-channel post-HFRM H28_1 feature:

```text
H28_1 -> P_sem/P_morph -> U_sem/U_morph -> original CAM28_1 head
```

Phase -1 initialization parity is complete. On the frozen BCSS seed-42 final
checkpoint, random input, real validation input, every CAM output, and all 3418
official validation predictions were bit-exact with A0. Both models obtained
67.32791670% mIoU and 80.26795301% mDice, with zero differing pixels. Parameter
overhead is 0.4661%.

Current decision: `OSMF_PHASE_MINUS1_PASS`.

- Phase -1 report: `audit/results/osmf_phase_minus1/osmf_phase_minus1_readiness_report.md`
- Delivery and resume notes: `docs/osmf_phase_minus1_delivery.md`
- Frozen technical specification: `docs/specs/osmf_v1_technical_spec.md`
- Machine-readable evidence: `audit/results/osmf_phase_minus1/`

No optimizer step, SSHR training, test evaluation, or LUAD evaluation was run.
The next permitted action is the separately reviewed 128-batch Phase 0
structural sanity audit; the 3-epoch and 25-epoch experiments remain locked.

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

