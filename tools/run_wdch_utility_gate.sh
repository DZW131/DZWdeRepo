#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT A0_FINAL PRETRAINED EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
A0_FINAL="$3"
PRETRAINED="$4"
EXPERIMENT_DIR="$5"
NUM_WORKERS="$6"
REPORTS="${EXPERIMENT_DIR}/reports"
SCHEDULE="${EXPERIMENT_DIR}/schedule/wdch_25epoch_schedule.npz"
COMMON_DIR="${EXPERIMENT_DIR}/matched/common"
C0_DIR="${EXPERIMENT_DIR}/matched/C0"
W1_DIR="${EXPERIMENT_DIR}/matched/W1"

mkdir -p "${REPORTS}" "${EXPERIMENT_DIR}/schedule" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

python tools/run_wdch_phase0.py \
  --val-root "${VAL_ROOT}" \
  --checkpoint "${A0_FINAL}" \
  --output-dir "${REPORTS}" \
  --sample-count 64

python tools/preflight_wdch.py \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
  --checkpoint "${A0_FINAL}" \
  --phase0-summary "${REPORTS}/wdch_phase0_summary.json" \
  --output "${REPORTS}/wdch_preflight.json"

python tools/eval_wdch_frozen.py \
  --val-root "${VAL_ROOT}" \
  --checkpoint "${A0_FINAL}" \
  --phase0-summary "${REPORTS}/wdch_phase0_summary.json" \
  --output-dir "${REPORTS}" \
  --num-workers "${NUM_WORKERS}"

python tools/build_wdch_schedule.py \
  --train-root "${TRAIN_ROOT}" \
  --output "${SCHEDULE}"

python tools/train_wdch_matched.py \
  --mode common \
  --train-root "${TRAIN_ROOT}" \
  --pretrained "${PRETRAINED}" \
  --schedule "${SCHEDULE}" \
  --output-dir "${COMMON_DIR}" \
  --num-workers "${NUM_WORKERS}"

for BRANCH in C0 W1; do
  BRANCH_DIR="${C0_DIR}"
  if [[ "${BRANCH}" == "W1" ]]; then
    BRANCH_DIR="${W1_DIR}"
  fi
  python tools/train_wdch_matched.py \
    --mode branch \
    --branch "${BRANCH}" \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON_DIR}/common_epoch20.pth" \
    --phase0-summary "${REPORTS}/wdch_phase0_summary.json" \
    --output-dir "${BRANCH_DIR}" \
    --num-workers "${NUM_WORKERS}"
done

python tools/analyze_wdch_gate.py \
  --val-root "${VAL_ROOT}" \
  --phase0-summary "${REPORTS}/wdch_phase0_summary.json" \
  --phase1-summary "${REPORTS}/wdch_frozen_intervention_metrics.json" \
  --c0-dir "${C0_DIR}" \
  --w1-dir "${W1_DIR}" \
  --output-dir "${REPORTS}" \
  --num-workers "${NUM_WORKERS}"
