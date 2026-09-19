# M1 Target Representation Shift Anatomy Audit

Zero-training BCSS/Seed42 audit of the frozen CCRA+HQMR-v1 checkpoint. It
reproduces the original validation prediction, builds M1 and matched TP
cohorts, extracts observable spatial features, and reports stage-wise tissue
geometry. It never calls an optimizer or changes model parameters.

## Environment and inputs

Use the existing `sshr5090` conda environment with PyTorch, NumPy, SciPy,
Pandas, PyArrow, scikit-learn, Pillow, and Matplotlib. Run scripts from the
repository checkout; they add the repository root to `sys.path` where needed.

Required immutable inputs:

- HQMR epoch-25 checkpoint, SHA256
  `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`.
- BCSS-WSSS validation `img/` and `mask/` folders, 3,418 images.
- BCSS-WSSS training images and reconstructed training GT masks, 23,422
  training images. GT is used only for this oracle mechanism audit.
- Frozen PSCR-v1 `compatibility/embedding_statistics.json` for historical
  C/D/M1 distance anchors. These distances are **not** freshly recomputed.
- Frozen morphology-audit prediction cache for the 100 case figures.

The parameter and cohort criteria are frozen in
`configs/audits/m1_target_shift_v1.yaml`.

## Order of execution

Replace the environment-specific paths in these examples. `OUT` must be a
new experiment directory; do not point it at a previous experiment.

```bash
PY=/path/to/sshr5090/bin/python
CHECKPOINT=/path/to/hqmr_epoch25_final.pth
VAL=/path/to/BCSS-WSSS/val
TRAIN=/path/to/BCSS-WSSS/training
TRAIN_GT=/path/to/BCSS-WSSS-training-oracle-masks
PSCR=/path/to/PSCR-v1/compatibility/embedding_statistics.json
CACHE=/path/to/HQMR-v1-Morphology-Audit
OUT=/path/to/M1-Target-Shift-Audit

$PY audits/m1_target_shift_v1/runtime_gate.py --checkpoint "$CHECKPOINT" --val-root "$VAL" --pscr-compatibility "$PSCR" --output "$OUT"
$PY audits/m1_target_shift_v1/extract_multistage_features.py target --checkpoint "$CHECKPOINT" --val-root "$VAL" --output "$OUT"
$PY audits/m1_target_shift_v1/extract_multistage_features.py reference --checkpoint "$CHECKPOINT" --train-root "$TRAIN" --train-gt-root "$TRAIN_GT" --reference-images 1500 --output "$OUT"
$PY audits/m1_target_shift_v1/compute_shift_metrics.py --output "$OUT"
$PY audits/m1_target_shift_v1/sensitivity.py --output "$OUT"
$PY audits/m1_target_shift_v1/stratify_true_match.py --output "$OUT"
$PY audits/m1_target_shift_v1/generate_report.py --output "$OUT" --val-root "$VAL" --cache-root "$CACHE"
```

`preflight.py` can inspect archived PSCR JSON and create a static hook manifest,
but its `ARCHIVAL_PASS_RUNTIME_PENDING` is **not** a substitute for the fresh
`runtime_gate.py` validation. The latter stops downstream work if mIoU, M1
count/area, checkpoint hash, or hook-safety checks fail.

## Actual tensor map and interpretation

| Plan position | Frozen code tensor | Is a spatial tissue feature? |
| --- | --- | --- |
| S0 H5_pre | raw backbone `F5` | Yes |
| S1 H5_cond | `context_feature` = CHPF(projected F5) | Yes, but **not CCRA-conditioned** |
| S2 H4_recon | `logits4` = upsampled `logits5` + `direct4` | No: query-mask logits |
| S3 H4_query | `query4` after region update | No: per-query vectors |
| S4 K4 | Stage3 HQMR `key4` = scale4.key(`F4_context`) | Yes, identical to CIRV/PSCR |

`H4_input` (`pixel_detail.F4_context`) is included as an additional observable
pre-reconstruction spatial stage. No spatial `H4_recon` or `H4_query` feature
exists in the frozen model. A cross-stage raw cosine comparison or a claim
that K4 is produced by `query4` would be invalid.

The primary TP cohort matches **predicted** class and log-area decile as the
supplied plan specifies. Because the M1 true class differs from its predicted
class, this makes raw true-class-distance comparisons class-confounded. The
`sensitivity.py` audit therefore also matches TP by M1 **GT true** class and
area. Any decision driven by that corrective analysis is explicitly reported
with reduced confidence. Same-image TP matching is rare, so residual tissue
context confounding remains.

## Outputs

The experiment directory contains the reproduction/hook gates, target and
reference cohort Parquet files, pooled feature NPZs, packed masks, stage
manifolds, CSV/Parquet metrics, 2,000-resample bootstrap intervals, machine
decision JSON, 100 representative case PNGs, and a 25-section Chinese report.
The report is observational: it neither evaluates a new architecture nor
changes the frozen HQMR mIoU.
