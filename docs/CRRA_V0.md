# CRRA-v0 execution guide

CRRA-v0 is a BCSS validation-only representation audit built directly from the
official A0 commit `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. It does not train SSHR.

## Scope

- Frozen checkpoint and frozen post-HFRM H28_1 features.
- Exact RGR-v0 coarse proposal, 8-connected components, and minimum area 2.
- Exactly WholeToken, CoreToken, and Core+Rim DualToken.
- One shared 5-fold GroupKFold assignment by the 22 BCSS source slides.
- Fixed StandardScaler plus multinomial logistic regression probe.
- Validation GT is used only for offline region labels and diagnostics.
- No test, LUAD, CRSR, GNN, prototype, attention, or segmentation training.

## Data and checkpoint

The command expects the released directory structure:

```text
datasets/BCSS-WSSS/val/
  img/*.png
  mask/*.png
```

The frozen checkpoint must have SHA256
`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.

## Environment and audit command

Use the same Python/PyTorch environment that runs official A0 inference, plus
NumPy, pandas, SciPy, scikit-learn, matplotlib, Pillow, and OpenCV.

```bash
python tools/audit_crra_v0.py \
  --val-root /path/to/datasets/BCSS-WSSS/val \
  --checkpoint /path/to/stage1_last.pth \
  --output-dir /path/to/experiments/CRRA_V0_<commit> \
  --audit-commit <commit> \
  --batch-size 20 \
  --num-workers 8 \
  --amp-dtype bf16
```

## Output index

- `provenance/run.json`: exact command, versions, source/checkpoint hashes.
- `regions/metadata.csv`: common region IDs, taxonomy, coverage, and diagnostics.
- `regions/features.npz`: aligned Whole/Core/Rim feature arrays.
- `folds/`: fixed slide-held-out assignments and fold manifest.
- `probes/{whole,core,core_rim}/`: OOF predictions and complete metrics.
- `diagnostics/`: exclusion, class/fold, bootstrap, dispersion, and rank-test tables.
- `figures/`: representation and fold-delta plots.
- `docs/crra_v0_region_representation_audit.md`: complete final report.
- `summary.json`: machine-readable decision record.

The program applies the preregistered decision thresholds and then stops.
