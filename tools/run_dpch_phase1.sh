#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 VAL_ROOT C0_DIR BCCH_DIR CBCCH_A3_DIR OUTPUT_DIR NUM_WORKERS" >&2
  exit 2
fi

python tools/run_dpch_phase1.py \
  --val-root "$1" \
  --c0-dir "$2" \
  --bcch-dir "$3" \
  --cbcch-dir "$4" \
  --output-dir "$5" \
  --num-workers "$6" \
  --bootstrap-resamples 10000
