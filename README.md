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

## RDDR Phase-1: Spatial-Semantic Dross Disposal

RDDR Phase-1 adds an identity-initialized Dross Disposal Adapter only before
`HFRM28_1`. The `UC` parameter-matched control removes an unconditioned
learned component; `DD` scales the same component by the detached normalized
Jensen–Shannon disagreement between the raw shallow and deep CAM probes.
Default mode `none` remains numerically identical to official A0.

The frozen architecture and optimizer contract is documented in
[`docs/rddr_phase1_architecture_contract.md`](docs/rddr_phase1_architecture_contract.md).

### Training UC and DD

The server runner executes UC Full25 followed by DD Full25 using the released
training loop. It saves Epoch 1/5/10/15/20 plus `stage1_last.pth`; the latter is
the Epoch-25 FINAL checkpoint. No validation or test metric participates in
training or checkpoint selection.

```bash
bash tools/run_rddr_phase1_server.sh \
  /path/to/RDDR_PHASE1_RUN \
  /path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  /path/to/BCSS-WSSS/training \
  /path/to/python
```

### Validation analysis

```bash
python tools/analyze_rddr_phase1.py \
  --c0-checkpoint /path/to/C0/stage1_last.pth \
  --uc-dir /path/to/RDDR_PHASE1_RUN/UC \
  --dd-dir /path/to/RDDR_PHASE1_RUN/DD \
  --phase0-dir /path/to/RDDR_PHASE0/formal \
  --val-root /path/to/BCSS-WSSS/val \
  --smoke-json /path/to/rddr_phase1_smoke.json \
  --pretrained /path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --train-root /path/to/BCSS-WSSS/training \
  --python-executable /path/to/python \
  --output-dir /path/to/RDDR_PHASE1_RUN/report \
  --bootstrap-resamples 10000
```

The analysis reproduces official three-view TTA and reports the full CAM
hierarchy, boundary/interior and object-size diagnostics, fixed Phase-0 dross
strata, fixed C0 CH-transition groups, q dynamics, disposal magnitude,
feature preservation, and paired image-level bootstrap intervals.

### Phase-1 validation result

| Variant | Dataset / split | Checkpoint | mIoU | mDice | Decision |
|---|---|---|---:|---:|---|
| C0 | BCSS validation | Epoch-25 FINAL | 67.3104 | 80.2564 | reference |
| UC | BCSS validation | Epoch-25 FINAL | 67.1569 | 80.1397 | control |
| DD | BCSS validation | Epoch-25 FINAL | 67.2081 | 80.1769 | `DROSS_DISPOSAL_SEMANTIC_DAMAGE` |

DD recovered `+0.0513` mIoU points over the parameter-matched UC control but
remained `-0.1022` points below C0 (paired 95% bootstrap CI
`[-0.3971, +0.1700]` points). The preregistered overall-improvement and
CAM28_1/interior-safety gates failed, so Phase-1 stops without a follow-on
model experiment. See
[`docs/rddr_phase1_spatial_semantic_dross_disposal_report.md`](docs/rddr_phase1_spatial_semantic_dross_disposal_report.md)
for the complete validation-only analysis.

## Acknowledgement

We thank the authors of [ESFAN](https://github.com/OceanPetal/ESFAN), whose codebase provided a valuable foundation for this repository.

