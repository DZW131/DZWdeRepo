# GCQM Full25 Failure Anatomy — BCSS Seed42

## Purpose

This is a zero-training post-hoc anatomy of the sealed GCQM E25 Full25 NOGO result. It identifies whether the decoder loses accuracy through FP/FN bias, interior or boundary errors, tissue-contact confusion, contaminated bases, diffuse global weights, or lost spatial coherence. It never updates model parameters or tunes inference.

## Environment and data

- Python: `/home/duyanhong/miniconda3/envs/sshr5090/bin/python`
- GPU: RTX 4090D
- Validation: `/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/{img,mask}`
- GCQM E25 SHA256: `6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f`
- SSHR B0 SHA256: `b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70`

Install dependencies in a fresh environment with `python -m pip install -r requirements.txt`. The audit additionally requires SciPy, pandas, scikit-learn, Pillow, and matplotlib.

## Files

- `tools/audit_gcqm_full25_failure_anatomy.py`: sealed inference, H1–H7 diagnostics, fixed counterfactuals, clustering, visualizations, and report.
- `tests/test_gcqm_failure_anatomy.py`: protocol and metric regression tests.
- `docs/GCQM_Final_Full25_BCSS_Seed42_artifacts/`: frozen source evaluation evidence.

## Run tests

```bash
python -m pytest tests/test_gcqm_failure_anatomy.py tests/test_gcqm_full25.py -q
```

## Run the audit

```bash
python tools/audit_gcqm_full25_failure_anatomy.py \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --gcqm-checkpoint /home/duyanhong/experiments/GCQM_Final_Full25_BCSS_Seed42/checkpoints/gcqm_full25_epoch25_final.pth \
  --gcqm-experiment /home/duyanhong/experiments/GCQM_Final_Full25_BCSS_Seed42 \
  --sshr-checkpoint /home/duyanhong/experiments/SSHR_B0_RTX4090D_BCSS_SEED42_FULL25_A0/checkpoints/stage1_last.pth \
  --sshr-experiment /home/duyanhong/experiments/SSHR_B0_RTX4090D_BCSS_SEED42_FULL25_A0 \
  --full25-delivery-summary /home/duyanhong/GCQM_FULL25_DELIVERY_SUMMARY.md \
  --output-dir /home/duyanhong/experiments/GCQM_Full25_Failure_Anatomy_BCSS_Seed42
```

## Fixed protocol

- Sealed E25 only; 3-view TTA and fixed `[0.8, 0.9, 0.8, 0.6]` class-presence thresholds.
- Boundary/contact radii `1/3/5`, with `3` primary.
- Top-k diagnostic curve `1/3/5/10/20/50/100/196`; no best-k selection.
- Paired bootstrap: 10,000 samples, seed 20260911.
- KMeans is descriptive only: fixed primary k=3, robustness k=2/4.
- GT is used only for post-hoc diagnosis and oracle ceilings.

## Outputs

The output directory contains provenance, reproduction gate, paired image/class metrics, FP/FN decomposition, boundary/interior and contact tables, weight and basis audits, correlations, morphology, fixed counterfactuals, automatically selected visualizations, and the final Markdown report.

## Stop rule

The audit must stop after reporting one registered bottleneck decision and confidence. It must not train or propose a complete GCQM-v2 architecture.
