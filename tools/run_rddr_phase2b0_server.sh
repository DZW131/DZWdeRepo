#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
RUN_ROOT=/home/duyanhong/experiments/RDDR_PHASE2B0
mkdir -p "$RUN_ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
"$PY" -m unittest discover -s tests -p test_rddr_phase2b0.py -v > "$RUN_ROOT/tests_$STAMP.log" 2>&1
"$PY" tools/run_rddr_phase2b0_relation_audit.py \
  --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
  --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
  --population-cache /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/frozen_phase0_populations \
  --phase0-results /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal \
  --output "$RUN_ROOT/${1:?provide unique output name}" ${2:+--smoke-images "$2"}
