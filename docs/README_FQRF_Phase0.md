# FQRF-Net Phase-0 Focus Preservation Gate

This branch implements the BCSS/Seed42/RTX4090D train-only mechanism gate for Focus-Preserved Query Region Formation. It asks whether three hierarchical cross-attention-first stages, dynamic focus-aware positions, and previous-mask-guided attention reduce HQRF query redundancy while preserving semantic selectivity. It is not a segmentation-performance experiment and never reads validation masks or reports mIoU/mDice.

## Environment

The registered server command uses:

```bash
/home/duyanhong/miniconda3/envs/sshr5090/bin/python
```

Required runtime: Linux, CUDA-capable PyTorch with native BF16, an RTX 4090-class GPU, torchvision, Pillow, NumPy, matplotlib, pytest, and the repository's original ResNet38 dependencies. The launcher rejects non-4090 GPUs and non-BF16 environments.

## Repository Structure

- `network/fqrf_query.py`: patch content/base position, parameter-free memory positions, cross-first decoder, dynamic focus update, and previous-mask visibility/fallback.
- `network/fqrf_net.py`: three-stage FQRF model, stage-wise mask/PCA losses, and Stage3-only PMEC.
- `network/hqrf_backbone.py`: hash-locked plain ResNet38 feature taps inherited unchanged.
- `network/hqrf_targets.py`: unchanged tri-state CAM targets and Full25-denominator locality.
- `network/hqrf_pmec.py`: unchanged detached PCRE-style PMEC.
- `tools/fqrf_diagnostics.py`: fixed-cohort stage-wise diagnostics and exact epoch2/final gates.
- `tools/fqrf_report.py`: required 34-section report renderer.
- `train_fqrf_phase0.py`: deterministic Phase-0 trainer, monitor, artifact writer, and visualizer.
- `tests/test_fqrf_phase0.py`: architecture and mechanism tests.
- `tests/test_fqrf_protocol.py`: duration, thresholds, gate boundary, and report tests.

## Frozen Implementation Interpretations

- `Kp^D`, `Kp^5`, and `Kp^4` are parameter-free normalized 2-D sine/cosine encodings on the actual memory grid (28×28 for 224×224 input).
- DFPQ uses head-averaged attention returned by PyTorch, re-normalizes it in FP32, detaches `A`, and evaluates `h(A Kp + B)`. The learned base `B` remains in the differentiable path.
- CNN-side FD/F5/F4-context tensors are detached before trainable memory projections. This blocks memory-path gradients to the backbone while keeping the projections trainable.
- Stage2/3 visibility is `sigmoid(previous_mask)>=0.15`, bilinearly resized to the memory grid and detached. A row with zero visible keys is replaced by an all-visible/global row before attention.
- Attention-focus IoU thresholds each query map at its own mean attention; centroid distances use normalized `[0,1]×[0,1]` memory coordinates.
- No diversity, repulsion, orthogonality, contrastive, area, sparsity, Dice, focal, teacher/student, PTC, or PAR loss is present.

## Data Organization

Formal training accepts only:

```text
/home/duyanhong/reseg-data/raw/BCSS-WSSS/training
```

The loader must discover exactly 23,422 train images and produce 1,171 drop-last batches of 20 per epoch. The frozen monitoring cohort contains at least 8 single-class and 16 multi-class train images, totaling 32. Only images, image-level labels, and online CAM pseudo-labels are used.

Official initialization:

```text
/home/duyanhong/sshr-resnet38-init.params
SHA256 f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16
```

## Validation Commands

Run unit and protocol tests:

```bash
python -m pytest -q
```

Run the strict two-minibatch smoke test in a disposable output directory:

```bash
python train_fqrf_phase0.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-resnet38-init.params \
  --output /home/duyanhong/experiments/FQRF_Phase0_SMOKE \
  --smoke-steps 2
```

The smoke run must show finite loss, finite gradients in backbone/deep/query/focus/pixel/mask/PCA groups, and nonzero gradients in the new focus modules. Delete only that exact disposable directory before formal training.

## Formal Training Command

```bash
nohup env PYTHONUNBUFFERED=1 /usr/bin/time -v \
  /home/duyanhong/miniconda3/envs/sshr5090/bin/python train_fqrf_phase0.py \
  --trainroot /home/duyanhong/reseg-data/raw/BCSS-WSSS/training \
  --weights /home/duyanhong/sshr-resnet38-init.params \
  --output /home/duyanhong/experiments/FQRFNet_Phase0_Focus_Preservation_BCSS_Seed42 \
  > /home/duyanhong/FQRFNet_Phase0_Focus_Preservation_BCSS_Seed42_launcher.log 2>&1 &
```

Monitor without changing the run:

```bash
tail -f /home/duyanhong/FQRFNet_Phase0_Focus_Preservation_BCSS_Seed42_launcher.log
nvidia-smi
```

Snapshots occur at step 250, 500, 1000, epoch1, epoch2, and epoch3. Epoch2 applies the frozen catastrophic screen. The run ends at epoch2 on catastrophic NOGO or otherwise at epoch3.

## Inference and Evaluation

There is intentionally no inference command or segmentation evaluator in Phase-0. Do not run BCSS validation, segmentation GT evaluation, mIoU, mDice, or validation-based checkpoint selection. Only a separate fresh-initialization Full25 plan may introduce final evaluation after a Phase-0 GO.

## Visualization

The formal trainer automatically writes eight fixed-cohort panels at step500, step1000, epoch2, and epoch3 under:

```text
<output>/visualizations/<snapshot>/
```

Each panel contains the input, deep CAM, five masks for each of Stages1–3, top-query cross-attention per stage, Qp2/Qp3 focus-centroid overlays, Stage3 PCA assignments, and Stage3 PMEC. It contains no segmentation ground truth.

## Required Artifacts

The output directory contains config/source/init/provenance/runtime JSON, `fqrf_losses.csv`, all ten required diagnostic CSV files, the endpoint and SHA256, visualizations, epoch2 screen, final gate JSON, and `FQRF_Phase0_Focus_Preservation_Report.md`.

| Decision | Meaning | Next action |
|---|---|---|
| `FQRF_PHASE0_STRONG_GO` | Strong specialization and all guardrails pass | Freeze architecture; fresh-init Full25 |
| `FQRF_PHASE0_GO` | Primary specialization and all guardrails pass | Freeze architecture; fresh-init Full25 |
| `FQRF_FOCUS_NOGO` | Architecture-level focus preservation is insufficient | Archive route; no rescue tuning |
| `FQRF_ENGINEERING_BLOCKED` | Scientific decision invalidated by implementation/runtime failure | Fix engineering only and restart Phase-0 from step 0 |

## Reproducibility and Resume Policy

Phase-0 checkpoints are terminal evidence, never resume seeds. Any engineering restart must use a new empty output directory or archive the failed one, reload the official MXNet initialization, reset Seed42, and begin from optimizer step 0. If Phase-0 passes, Full25 must likewise start fresh rather than continue the Phase-0 endpoint.

