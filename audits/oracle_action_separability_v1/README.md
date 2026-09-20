# Oracle Action GT-Free Separability Audit v1

This is a frozen-model BCSS Seed42 audit, not a segmentation training run. It tests whether the Oracle alpha action on **all** baseline predicted 8-connected components and a beneficial single-class gate force-on action on **all** deep-gate-OFF image/class pairs are identifiable from inference-time signals. No segmentation parameters are updated.

## Environment and inputs

- The repository's `sshr5090` Python environment with PyTorch, NumPy, SciPy, pandas/pyarrow, scikit-learn, Matplotlib, PIL and pytest.
- BCSS validation root containing `img/` and `mask/` (3418 images).
- Sealed HQMR E25 checkpoint with SHA256 `84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb`.
- Frozen DLAG alpha prediction bank and Oracle metrics, UCRF component manifest, RACC-v1 final metrics.

## Required execution order

The feature freeze deliberately reads `img/` only; `mask/`, DLAG prediction bank and UCRF labels are opened only after `feature_manifest.json` and feature SHA256s exist. Do not merge or reorder these steps. Each command assumes execution from the repository root.

```bash
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
OUT=/home/duyanhong/experiments/Oracle_Action_Separability_v1_BCSS_Seed42
CKPT=/home/duyanhong/experiments/CCRA_HQMR_Full25_BCSS_Seed42/checkpoints/hqmr_epoch25_final.pth
VAL=/home/duyanhong/reseg-data/raw/BCSS-WSSS/val
DLAG=/home/duyanhong/experiments/DLAG_Oracle_v1_BCSS_Seed42
RACC=/home/duyanhong/experiments/RACC_v1_Phase0_BCSS_Seed42
UCRF=/home/duyanhong/experiments/UCRF_v1_Upstream_Class_Responsibility_Formation_Audit_BCSS_Seed42

$PY audits/oracle_action_separability_v1/reproduction_gate.py --checkpoint "$CKPT" --dlag "$DLAG" --racc "$RACC" --output "$OUT/00_reproduction_gate.json"
$PY audits/oracle_action_separability_v1/freeze_observables.py --checkpoint "$CKPT" --val-root "$VAL" --output "$OUT" --num-workers 2
$PY audits/oracle_action_separability_v1/build_labels.py --checkpoint "$CKPT" --val-root "$VAL" --dlag "$DLAG" --ucrf "$UCRF" --output "$OUT" --num-workers 2
$PY audits/oracle_action_separability_v1/analyze.py --output "$OUT"
$PY audits/oracle_action_separability_v1/visualize.py --checkpoint "$CKPT" --val-root "$VAL" --output "$OUT"
$PY audits/oracle_action_separability_v1/generate_report.py --output "$OUT"
$PY -m pytest -q tests/test_oracle_action_separability.py
```

The output directory is write-once. `freeze_observables.py` will refuse to overwrite an existing feature manifest. A rerun should use a fresh output directory and be compared via hashes, never edit sealed feature tables in place.

## Files and roles

| File | Role |
|---|---|
| `reproduction_gate.py` | Exact six-anchor and checkpoint SHA gate |
| `freeze_observables.py` | HQMR inference, GT-free feature extraction, feature SHA seal |
| `build_labels.py` | DLAG alpha component action and single-class force-on gate Oracle labels |
| `analyze.py` | Fixed univariate metrics, 5-fold patient-grouped linear/tree probes, 10 within-patient permutations, ablations and decision rules |
| `visualize.py` | 20 examples in each of eight error/success categories |
| `generate_report.py` | 25-section final Markdown report |

## Key outputs

`feature_manifest.json` and `feature_table_sha256.txt` document the anti-leakage boundary. `arbitration/observable_features.parquet` and `gate/gate_off_pairs.parquet` contain only model-observable features plus identifier/metadata columns that are explicitly excluded from probes. The two `*_labels.parquet` files, CV and ablation CSVs, subgroup/weighted metrics, leakage audit, decision JSONs, `visualizations/manifest.json`, and `Oracle_Action_GTFree_Separability_Audit_Report.md` complete the handoff.

The Oracle-supervised probes are *not deployable models*. A GO says information is separable under the specified grouped-validation diagnostic, not that a weakly supervised controller has already been built.
