#!/usr/bin/env bash
set -euo pipefail

: "${HQMR:?set HQMR checkpoint path}"
: "${BCSS_TRAIN:?set BCSS training image directory}"
: "${BCSS_VAL:?set BCSS validation root}"
: "${UMRF:?set frozen UMRF artifact directory}"
: "${REPLAY:?set completed baseline replay directory}"
: "${RUN:?set RISA experiment output directory}"

python -m pytest tests/test_risa_v1.py -q
python tools/risa_v1_phase0/prepare_baseline.py --replay "$REPLAY" --hqmr "$HQMR" --umrf "$UMRF" --output "$RUN"
python -u tools/risa_v1_phase0/train.py --checkpoint "$HQMR" --train-root "$BCSS_TRAIN" --output "$RUN" --num-workers 8
python -u tools/risa_v1_phase0/evaluate.py --checkpoint "$HQMR" --val-root "$BCSS_VAL" --umrf "$UMRF" --output "$RUN" --batch-size 8 --num-workers 8
python tools/risa_v1_phase0/generate_report.py --output "$RUN"
