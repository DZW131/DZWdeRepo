#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 9 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT COMMON_EPOCH20 SCHEDULE PHASE0_SUMMARY C0_DIR W1_DIR EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
COMMON_EPOCH20="$3"
SCHEDULE="$4"
PHASE0_SUMMARY="$5"
C0_DIR="$6"
W1_DIR="$7"
EXPERIMENT_DIR="$8"
NUM_WORKERS="$9"
REPORTS="${EXPERIMENT_DIR}/reports"
W2_DIR="${EXPERIMENT_DIR}/matched/W2"
CALIBRATION="${REPORTS}/wdch_strength_calibration.json"

test -f "${COMMON_EPOCH20}"
test -f "${SCHEDULE}"
test -f "${PHASE0_SUMMARY}"
test -f "${C0_DIR}/complete.json"
test -f "${W1_DIR}/complete.json"
test ! -e "${CALIBRATION}"
test ! -e "${W2_DIR}"

mkdir -p "${REPORTS}" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

python tools/calibrate_scwdch_strength.py \
  --train-root "${TRAIN_ROOT}" \
  --common-checkpoint "${COMMON_EPOCH20}" \
  --phase0-summary "${PHASE0_SUMMARY}" \
  --output "${CALIBRATION}" \
  --num-workers "${NUM_WORKERS}"

python tools/preflight_scwdch.py \
  --train-root "${TRAIN_ROOT}" \
  --schedule "${SCHEDULE}" \
  --common-checkpoint "${COMMON_EPOCH20}" \
  --calibration "${CALIBRATION}" \
  --output "${REPORTS}/scwdch_preflight.json"

python tools/train_scwdch_matched.py \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
  --schedule "${SCHEDULE}" \
  --common-checkpoint "${COMMON_EPOCH20}" \
  --calibration "${CALIBRATION}" \
  --output-dir "${W2_DIR}" \
  --num-workers "${NUM_WORKERS}"

python tools/analyze_scwdch_v2.py \
  --val-root "${VAL_ROOT}" \
  --common-checkpoint "${COMMON_EPOCH20}" \
  --schedule "${SCHEDULE}" \
  --calibration "${CALIBRATION}" \
  --c0-dir "${C0_DIR}" \
  --w1-dir "${W1_DIR}" \
  --w2-dir "${W2_DIR}" \
  --output-dir "${REPORTS}" \
  --num-workers "${NUM_WORKERS}"
