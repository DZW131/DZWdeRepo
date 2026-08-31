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

## Acknowledgement

We thank the authors of [ESFAN](https://github.com/OceanPetal/ESFAN), whose codebase provided a valuable foundation for this repository.

# Phase2B1.12 short-horizon ADT audit

This branch starts from immutable official A0 `4e9a288`. Original network,
preprocessing, training entrypoint, inference and metric sources are unchanged.
See [approved execution contract](docs/rddr_phase2b112_execution_contract.md)
and [full specification](docs/rddr_phase2b112_specification.md).

The independent entrypoint runs B/A/R from the same C0 model-only checkpoint,
with fresh matched official SGD states at the exactly reconstructed final-step
learning rates. It performs a single32-batch training-only calibration followed
by exactly500 steps, with final step500 as the only primary endpoint. BN affine
in the approved39-tensor auxiliary scope may receive ADT/RG gradients, while the
main-loss BN freeze and all BN running statistics remain unchanged. No test,
LUAD, seed sweep, parameter search or Full25 run is exposed by this entrypoint.

## Server environment and data

- Python: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python` (existing5090 environment).
- Checkout: `/home/duyanhong/DZWdeRepo-rddr-phase2b112`.
- BCSS train/val: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/{training,val}`.
- C0: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`.
- Immutable native reference: `/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz`.
- New outputs only: `/home/duyanhong/experiments/RDDR_PHASE2B112/<new-run-id>`.

No package upgrades are required. This branch uses existing torch/numpy/Pillow/
torchvision dependencies. CPU fixture tests are not CUDA/full-model readiness.

## Commands

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b112
/home/duyanhong/miniconda3/envs/sshr5090/bin/python -m unittest discover -s tests -p test_rddr_phase2b112.py -v
# CPU provenance only; requires a new output directory:
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b112.py --preflight-only --output /home/duyanhong/experiments/RDDR_PHASE2B112/preflight_new
# Complete finite job: tests, provenance, calibration,500 steps,validation,verification,report:
bash tools/execute_rddr_phase2b112.sh /home/duyanhong/experiments/RDDR_PHASE2B112/formal_new
```

Resource admission requires18GiB free for one active arm; other arms and optimizer
buffers are offloaded between turns. Exit75 is RESOURCE_BLOCKED, **not a
scientific NOGO**. The launcher does not poll, stop other jobs, change batch20,
retry failed runs, or overwrite output directories. A new directory must be used
for a later manually authorized launch; partial artifacts are retained.

## Files and result interpretation

`tools/rddr_phase2b112_common.py` contains the detached local auxiliary graph;
`run_rddr_phase2b112.py` implements matched batches/RNG and training;
`rddr_phase2b112_evaluation.py` instruments the unmodified canonical evaluator;
`verify_rddr_phase2b112.py` checks real artifacts;
`analyze_rddr_phase2b112.py` writes the requiredCSV/JSON and29-section report.

The final report is `docs/rddr_phase2b112_short_horizon_optimization_report.md`.
It must not be represented as complete until the real run and independent
verification finish. Tables in Markdown and the machine-readableCSV files are
the audit visualization; there is no separate dashboard or tracking service.

## 4090 migration (2026-08-31; training remains paused)

The current audit assets have been copied directly from the5090 server to
`duyanhong@10.15.20.149:54268`. Absolute paths and the `sshr5090` environment
prefix are unchanged; this is a copied Python environment, not a Conda base
installation. Invoke its Python executable directly. No package, driver,
architecture, objective, inference or training-protocol changes were made.

See the [migration report](docs/rddr_phase2b112_4090_migration_report.md) for
checksum verification,53 CPU tests, real batch20 BF16 zero-step smoke, package
inventory, filesystem ordering caveat, exact paths and future launch command.
The migration-only tool is `tools/verify_rddr_phase2b112_migration.py`; it never
calls optimizer.step or evaluates validation/test metrics. This engineering
PASS does not mean the500-step experiment has run or passed its scientific gates.

## Step0 record fix and authorized restart

The first4090 launch stopped after step0 B validation due to duplicate `arm`
keyword arguments while recording the result; all three optimizer step counts
were zero. The [minimal fix and restart note](docs/rddr_phase2b112_evaluation_record_fix.md)
documents57 passing tests, including a replay of the actual failed payload.
The old `formal_4090_r1` directory is preserved. The user-authorized restart uses
`/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2` and the same frozen
launcher/protocol. Only the result-recording callback changed, not model,
optimization, inference or metrics. Actual progress is in `formal_4090_r2.log`.

---
