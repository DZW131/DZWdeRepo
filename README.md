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

## Phase-0 Decision Bottleneck Audit

The validation-only frozen-model audit is implemented in
`tools/audit_decision_bottleneck.py`. It starts from the official A0 baseline,
reuses one fixed final checkpoint, performs no SSHR retraining, and exposes no
test-set or inference-tuning options.

The completed BCSS seed-42 audit reproduced the released inference masks
exactly (zero differing pixels). Its frozen decision was
`NONLINEAR_ROUTING_REVIEW`: the five-fold out-of-fold 16-scalar class probe
changed mIoU by -0.1603 percentage points, while the diagnostic image-class and
pixel oracles showed ceilings of +1.6013 and +6.7134 points, respectively.

- Full report: `audit/results/decision_bottleneck_phase0/docs/phase0_decision_bottleneck_audit.md`
- Raw CSV tables: `audit/results/decision_bottleneck_phase0/tables/`
- Deterministic figures: `audit/results/decision_bottleneck_phase0/figures/`
- Reproduction config and environment: `audit/results/decision_bottleneck_phase0/config.json`

This result does not authorize test evaluation or a new routing architecture;
it stops at the preregistered human-review decision.

## Phase-0B Routing Signal Learnability Audit

Phase-0B tested whether GT-free evidence available from the frozen A0 model can
actually learn the branch-routing opportunity identified in Phase-0. The audit
used the same final BCSS seed-42 checkpoint and exact 5-fold slide grouping. It
performed no SSHR training and did not evaluate test or LUAD.

Released inference was reproduced exactly (zero differing pixels; validation
mIoU 67.3279). Diagnostic ceilings remained substantial: the safe image oracle
improved mIoU by +2.0936 points, the image-fusion oracle by +2.6299, and the
bounded local image-class oracle by +4.3354. This supports the diagnostic
phenotypes `SOFT_MIXTURE_FAVORED` and `CLASS_CONDITIONAL_SIGNAL`.

However, the preregistered primary MLP-C OOF router reduced validation mIoU by
0.7158 points, had 0/5 positive folds, and produced a slide-bootstrap 95% CI of
[-1.0178, -0.4312] points. The frozen decision is therefore
`ROUTING_SIGNAL_NOGO`: complementary branch information exists, but these
GT-free image-level routing signals do not identify it reliably enough.

- Full report: `audit/results/routing_signal_phase0b/docs/phase0b_routing_signal_learnability_audit.md`
- Raw CSV tables: `audit/results/routing_signal_phase0b/tables/`
- Deterministic figures: `audit/results/routing_signal_phase0b/figures/`
- Frozen contract and environment: `audit/results/routing_signal_phase0b/config/`
- Archive manifest: `audit/results/routing_signal_phase0b/ARTIFACTS.md`

This result stops the current image-level routing direction; diagnostic oracle
flags are not authorization to tune inference or implement another router.

## Acknowledgement

We thank the authors of [ESFAN](https://github.com/OceanPetal/ESFAN), whose codebase provided a valuable foundation for this repository.

