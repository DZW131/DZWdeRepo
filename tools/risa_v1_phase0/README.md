# RISA-v1 Phase-0 reproducibility guide

RISA is a 296,704-parameter adapter trained on image-level BCSS labels while the HQMR checkpoint, Backbone, CCRA, and HQMR remain frozen. The fixed run is one baseline replay, nine unit tests, a five-epoch train, E5-only B0/R1/R2/R3 evaluation, mechanism audits, figures, and a final report.

## Environment

- Python 3.10
- PyTorch 2.11 + CUDA 12.8
- NVIDIA RTX 4090 D
- NumPy, pandas/pyarrow, SciPy, Pillow, matplotlib, torchvision, PyYAML, pytest

## Repository structure

- `network/risa_v1.py`: adapter, frozen-HQMR wrapper, and preregistered losses.
- `configs/risa_v1_phase0.yaml`: immutable Phase-0 configuration.
- `tools/risa_v1_phase0/prepare_baseline.py`: seals the pre-modification replay.
- `tools/risa_v1_phase0/train.py`: fixed five-epoch image-label-only training.
- `tools/risa_v1_phase0/evaluate.py`: fixed-E5 evaluation, Hard-M1/M1 audits, counterfactual, compute audit, and figures.
- `tools/risa_v1_phase0/generate_report.py`: final 21-section Markdown report.
- `tests/test_risa_v1.py`: shape, isolation, no-GT, pooling, top-K, probability, and loss tests.

## Required data and sealed inputs

The BCSS training directory contains 23,422 images whose names encode four image-level labels. The validation directory contains 3,418 `img/` and `mask/` pairs. Training never opens validation images or segmentation masks. Evaluation requires the exact frozen UMRF directory containing `image_ids.npy`, `gt_free_prediction_maps.uint8.npy`, and `component_evidence_with_gt.parquet`; it refuses counts other than Hard-M1=5,037 and M1=4,402. The HQMR checkpoint must have SHA256 `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`.

## Exact commands

```bash
python -m pytest tests/test_risa_v1.py -q

python tools/risa_v1_phase0/prepare_baseline.py \
  --replay "$REPLAY" --hqmr "$HQMR" --umrf "$UMRF" --output "$RUN"

python -u tools/risa_v1_phase0/train.py \
  --checkpoint "$HQMR" --train-root "$BCSS_TRAIN" --output "$RUN" --num-workers 8

python -u tools/risa_v1_phase0/evaluate.py \
  --checkpoint "$HQMR" --val-root "$BCSS_VAL" --umrf "$UMRF" \
  --output "$RUN" --batch-size 8 --num-workers 8

python tools/risa_v1_phase0/generate_report.py --output "$RUN"
```

## Outputs and interpretation

The canonical directory is `$RUN/artifacts/risa_v1_phase0/`. It includes the sealed baseline, E5 adapter, all variant metrics and predictions, frozen-component identity evidence, causal isolation, region-dependency audit, compute audit, four figure groups, the research-state JSON, and the final report. `reports/RISA_v1_Phase0_Final_Report.md` is the human-readable decision artifact. GO/STRONG_GO/NOGO is computed only by the preregistered Python rule; the optional Jev sidecar is secondary and may not modify it.
