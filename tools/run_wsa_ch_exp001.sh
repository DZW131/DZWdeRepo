#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 VAL_ROOT C0_DIR CBCCH_A3_DIR OUTPUT_DIR NUM_WORKERS" >&2
  exit 2
fi

python tools/run_wsa_ch_exp001.py \
  --val-root "$1" \
  --c0-dir "$2" \
  --cbcch-dir "$3" \
  --output-dir "$4" \
  --num-workers "$5" \
  --bootstrap-resamples 10000
