# UMRF-v1 reproducible execution

This audit performs no training and must be run in the order below. Ground-truth masks are opened only by `evaluate.py`, after the evidence manifest and hashes have been written.

```bash
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
REPO=/home/duyanhong/DZWdeRepo-gcqm-phase0
OUT=/home/duyanhong/experiments/UMRF_v1_Upstream_MultiLevel_Responsibility_Formation_Audit_BCSS_Seed42
CKPT=/home/duyanhong/experiments/CCRA_HQMR_Full25_BCSS_Seed42/checkpoints/hqmr_epoch25_final.pth
VAL=/home/duyanhong/reseg-data/raw/BCSS-WSSS/val
ORACLE=/home/duyanhong/experiments/Oracle_Action_Separability_v1_BCSS_Seed42
cd "$REPO"
$PY -m audits.umrf_v1.reproduction_gate --checkpoint "$CKPT" --source-gate "$ORACLE/00_reproduction_gate.json" --output "$OUT"
$PY -m audits.umrf_v1.freeze_evidence --checkpoint "$CKPT" --val-root "$VAL" --output "$OUT" --reference-baseline "$ORACLE/baseline_predictions_fp32.npz" --num-workers 2
$PY -m audits.umrf_v1.evaluate --val-root "$VAL" --output "$OUT" --historical-labels "$ORACLE/arbitration/oracle_action_labels.parquet"
$PY -m audits.umrf_v1.visualize --val-root "$VAL" --output "$OUT" --per-category 20
$PY -m audits.umrf_v1.generate_report --output "$OUT"
```

Stop immediately if the reproduction gate, formula identities, common-query dimensional safety check, baseline bitwise comparison, or frozen-artifact hash verification fails.

