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

mkdir -p "${REPORTS}" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

if [[ -e "${CALIBRATION}" ]]; then
  python - "${CALIBRATION}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("experiment_id") != "EXP-WDCH-002":
    raise SystemExit("Existing calibration has the wrong experiment ID")
if not value.get("initial_gate_a_pass"):
    raise SystemExit("Existing calibration did not pass the initial strength gate")
if value.get("validation_used") or value.get("test_used"):
    raise SystemExit("Existing calibration is contaminated")
print("SCWDCH_REUSE_CALIBRATION", flush=True)
PY
else
  python tools/calibrate_scwdch_strength.py \
    --train-root "${TRAIN_ROOT}" \
    --common-checkpoint "${COMMON_EPOCH20}" \
    --phase0-summary "${PHASE0_SUMMARY}" \
    --output "${CALIBRATION}" \
    --num-workers "${NUM_WORKERS}"
fi

if [[ -e "${REPORTS}/scwdch_preflight.json" ]]; then
  python - "${REPORTS}/scwdch_preflight.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "PASS" or value.get("optimizer_step_performed"):
    raise SystemExit("Existing W2 preflight is not reusable")
print("SCWDCH_REUSE_PREFLIGHT", flush=True)
PY
else
  python tools/preflight_scwdch.py \
    --train-root "${TRAIN_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON_EPOCH20}" \
    --calibration "${CALIBRATION}" \
    --output "${REPORTS}/scwdch_preflight.json"
fi

if [[ -e "${W2_DIR}/complete.json" ]]; then
  python - "${W2_DIR}/complete.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "SCWDCH_MATCHED_BRANCH_COMPLETE":
    raise SystemExit("Existing W2 branch is incomplete")
if value.get("epochs") != [21, 22, 23, 24, 25]:
    raise SystemExit("Existing W2 branch has the wrong epochs")
print("SCWDCH_REUSE_COMPLETE_W2", flush=True)
PY
else
  python tools/train_scwdch_matched.py \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON_EPOCH20}" \
    --calibration "${CALIBRATION}" \
    --output-dir "${W2_DIR}" \
    --num-workers "${NUM_WORKERS}"
fi

if [[ -e "${REPORTS}/scwdch_v2_strength_calibration_final_report.md" ]]; then
  echo "SCWDCH_FINAL_REPORT_ALREADY_EXISTS"
else
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
fi
