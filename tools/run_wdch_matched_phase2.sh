#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT PRETRAINED EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
PRETRAINED="$3"
EXPERIMENT_DIR="$4"
NUM_WORKERS="$5"
REPORTS="${EXPERIMENT_DIR}/reports"
SCHEDULE="${EXPERIMENT_DIR}/schedule/wdch_25epoch_schedule.npz"
COMMON_DIR="${EXPERIMENT_DIR}/matched/common"
C0_DIR="${EXPERIMENT_DIR}/matched/C0"
W1_DIR="${EXPERIMENT_DIR}/matched/W1"

python - "${REPORTS}/wdch_phase0_summary.json" \
  "${REPORTS}/wdch_frozen_intervention_metrics.json" <<'PY'
import json
import sys

phase0 = json.load(open(sys.argv[1], encoding="utf-8"))
phase1 = json.load(open(sys.argv[2], encoding="utf-8"))
if phase0.get("phase0_status") != "PASS":
    raise SystemExit("Phase 0 has not passed")
if phase1.get("phase1_status") != "PASS":
    raise SystemExit("Phase 1 has not passed")
if phase1.get("catastrophic_failure"):
    raise SystemExit("Catastrophic frozen failure forbids matched training")
print("WDCH_PHASE2_PREREQUISITES_PASS", flush=True)
PY

mkdir -p "${EXPERIMENT_DIR}/schedule" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

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
