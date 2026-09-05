# HQRF-Net Phase-0 Query-Mask Health Gate

This branch implements the preregistered, train-only HQRF mechanism gate on BCSS. It does **not** evaluate validation mIoU and is not a paper Full25 run. Its only endpoint asks whether restored PCRE-style patch queries, DETR query decoding, mask embeddings, PCA and PMEC produce selective and diverse tissue masks instead of HCRF-v1.1 all-window collapse.

## Environment

- Ubuntu 22.04 server with NVIDIA RTX 4090D
- Python: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`
- Native CUDA BF16 support
- Official MXNet ResNet38 initialization: `/home/duyanhong/sshr-resnet38-init.params`
- Initialization SHA256: `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`
- PCRE reference source: `xxf011/WSSS-PCRE`, commit `d89029439c2676ed98035a77bc94157afd0c777c`

The existing SSHR conda environment already contains PyTorch, torchvision, NumPy, Pillow and matplotlib. No validation/test/LUAD path is needed or accepted.

## Repository Structure

- `network/hqrf_backbone.py`: plain official ResNet38 and audited F3/F4/F5/FD taps.
- `network/hqrf_query.py`: patch embedding, two post-norm DETR decoder layers, CHPF, pixel decoder, mask MLP and PCA.
- `network/hqrf_targets.py`: normalized CAM tri-state targets, Full25-denominator locality and raw-logit masked BCE.
- `network/hqrf_pmec.py`: detached PCRE ROR/PMEC diagnostic.
- `network/hqrf_net.py`: frozen Phase-0 forward and loss.
- `tools/hqrf_diagnostics.py`: fixed-cohort health statistics and exact gates.
- `tools/hqrf_report.py`: fixed 28-section report.
- `train_hqrf_phase0.py`: smoke and formal max-three-epoch runner.
- `tests/test_hqrf_phase0.py`, `tests/test_hqrf_protocol.py`: architecture, semantics and protocol regressions.

## Data Organization

Only this directory is allowed:

```text
/home/duyanhong/reseg-data/raw/BCSS-WSSS/training
```

The runner expects 23,422 training images and 1,171 batches at batch size 20. Before the first update it freezes a Seed42 cohort of 32 training images: at least eight single-class and sixteen multi-class images when available, then random unused images. Cohort inference is deterministic, has no flips, and reads no segmentation masks.

## Local Validation

```bash
python -m pytest -q tests/test_hqrf_phase0.py tests/test_hqrf_protocol.py
python -m compileall -q network/hqrf_*.py tools/hqrf_*.py train_hqrf_phase0.py
```

## Engineering Smoke

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python train_hqrf_phase0.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-resnet38-init.params \
  --output /home/duyanhong/experiments/HQRF_Phase0_QueryMask_Health_SMOKE \
  --smoke-steps 2
```

Delete that exact smoke directory after checking both batches, finite losses and nonzero gradients. Start formal training in a fresh process and output directory.

## Formal Training

```bash
PYTHONUNBUFFERED=1 /home/duyanhong/miniconda3/envs/sshr5090/bin/python train_hqrf_phase0.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-resnet38-init.params \
  --output /home/duyanhong/experiments/HQRFNet_Phase0_QueryMask_Health_BCSS_Seed42
```

The formal run exports diagnostics at steps 250/500/1000 and every epoch end. Epoch 2 applies the catastrophic screen. If any fixed-cohort catastrophic condition affects more than 80% of images, training saves the Epoch2 endpoint, emits `HQRF_QUERY_REGION_NOGO`, and does not run Epoch3. Otherwise Epoch3 applies the complete GO gate.

## Inference and Evaluation

There is intentionally no validation inference or metric-evaluation command in Phase-0. The only evaluation is:

```text
32 frozen BCSS training images
+ image-level labels
+ online tri-state CAM pseudo labels
+ query/mask/PCA/PMEC statistics
```

If the decision is GO, a separate plan must restart from the fresh official initialization for Full25. The Phase-0 checkpoint must not be continued.

## Visualization

Eight fixed monitoring images are exported automatically at step500, step1000, Epoch2 and Epoch3 (if reached):

```text
visualizations/<snapshot>/<image-id>.png
```

Each panel contains the input, deep CAM, tri-state target, five top query masks, PCA assignment and PMEC regions.

## Artifact Index

- Configuration/provenance: `hqrf_phase0_config.json`, `hqrf_phase0_source_commit.txt`, `hqrf_phase0_init_identity.json`, `hqrf_phase0_provenance.json`, `hqrf_phase0_monitor_cohort.json`
- Training: `hqrf_phase0_train.log`, `hqrf_phase0_losses.csv`, `hqrf_phase0_runtime.json`
- Mechanism CSVs: mask area/logits, query diversity, semantic selectivity, PCA, PMEC, CHPF and catastrophic screen
- Endpoint: `hqrf_phase0_epoch2.pth` on early NOGO or `hqrf_phase0_epoch3.pth` after the final gate, plus SHA256
- Report: `HQRF_Phase0_QueryMask_Health_Report.md`

## Result Record

| Run | Split | Maximum duration | Validation metrics | Decision |
|---|---|---:|---|---|
| HQRF Phase-0 Seed42 | BCSS train only | 3 epochs | Prohibited | Filled by the preregistered mechanism gate |

## Important Frozen Interpretations

- FD/F5 CNN tensors are detached before their trainable projections. This blocks memory-path gradients into ResNet38 without leaving the projections randomly frozen.
- A locality radius is defined in 14×14 query-grid units and mapped by the exact 4× spatial scale to 56×56 masks.
- PCRE ROR is intersection(candidate, reference) divided by reference-mask area.
- “Absent confidence dominance” means more than half of the top-20 max-joint-confidence mass is assigned to absent classes; the GO gate requires this in fewer than 5% of monitoring images.
- Epoch2 “persistent” catastrophic collapse is operationalized as a frozen criterion being true for more than 80% of the 32 monitoring images at the Epoch2 endpoint.
