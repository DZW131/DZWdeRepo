#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
ROOT_OUT=/home/duyanhong/experiments/RDDR_PHASE2B1
mkdir -p "$ROOT_OUT"
"$PY" -m unittest discover -s tests -p test_rddr_phase2b1.py -v
"$PY" tools/run_rddr_phase2b1_dual_hypothesis_audit.py \
 --checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth \
 --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val \
 --population-cache /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/frozen_phase0_populations \
 --phase0-results /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal \
 --output "$ROOT_OUT/${1:?provide unique run name}" ${2:+--smoke-images "$2"}
