#!/usr/bin/env bash
set -euo pipefail
cd /home/duyanhong/DZWdeRepo-gcqm-phase0
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
OUT=/home/duyanhong/experiments/Oracle_Action_Separability_v1_BCSS_Seed42
CKPT=/home/duyanhong/experiments/CCRA_HQMR_Full25_BCSS_Seed42/checkpoints/hqmr_epoch25_final.pth
VAL=/home/duyanhong/reseg-data/raw/BCSS-WSSS/val
DLAG=/home/duyanhong/experiments/DLAG_Oracle_v1_BCSS_Seed42
UCRF=/home/duyanhong/experiments/UCRF_v1_Upstream_Class_Responsibility_Formation_Audit_BCSS_Seed42
while [[ ! -f "$OUT/feature_manifest_action.json" ]]; do sleep 15; done
echo '{"event":"RECONCILED_FREEZE_DETECTED"}'
$PY audits/oracle_action_separability_v1/build_labels.py --checkpoint "$CKPT" --val-root "$VAL" --dlag "$DLAG" --ucrf "$UCRF" --output "$OUT" --num-workers 2
$PY audits/oracle_action_separability_v1/analyze.py --output "$OUT"
$PY audits/oracle_action_separability_v1/visualize.py --checkpoint "$CKPT" --val-root "$VAL" --output "$OUT"
$PY audits/oracle_action_separability_v1/generate_report.py --output "$OUT"
$PY -m pytest -q tests/test_oracle_action_separability.py
echo '{"event":"AUDIT_PIPELINE_COMPLETE"}'
