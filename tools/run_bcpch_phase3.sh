#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT LOCKED_WDCH LOCKED_BCCH LOCKED_CBCCH BCPCH_EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
LOCKED_WDCH="$3"
LOCKED_BCCH="$4"
LOCKED_CBCCH="$5"
EXPERIMENT_DIR="$6"
NUM_WORKERS="$7"
COMMON="${LOCKED_WDCH}/matched/common/common_epoch20.pth"
SCHEDULE="${LOCKED_WDCH}/schedule/wdch_25epoch_schedule.npz"
C0_DIR="${LOCKED_WDCH}/matched/C0"
BCCH_DIR="${LOCKED_BCCH}/matched/BCCH"
CBCCH_DIR="${LOCKED_CBCCH}/matched/A3"
BCPCH_DIR="${EXPERIMENT_DIR}/matched/BCPCH"
REPORTS="${EXPERIMENT_DIR}/reports"

mkdir -p "${REPORTS}" "${EXPERIMENT_DIR}/matched" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

python - "${COMMON}" "${SCHEDULE}" "${C0_DIR}/complete.json" "${BCCH_DIR}/complete.json" "${CBCCH_DIR}/complete.json" <<'PY'
import hashlib
import json
import sys

expected_files = {
    sys.argv[1]: "2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb",
    sys.argv[2]: "fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea",
}
for path, expected in expected_files.items():
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if actual != expected:
        raise SystemExit(f"Locked artifact changed: {path}: {actual}")
for path, status, checkpoint in (
    (sys.argv[3], "WDCH_MATCHED_BRANCH_COMPLETE", "44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8"),
    (sys.argv[4], "BCCH_MATCHED_COMPLETE", "959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad"),
    (sys.argv[5], "CBCCH_MATCHED_COMPLETE", "2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e"),
):
    value = json.load(open(path, encoding="utf-8"))
    if value.get("status") != status or value.get("checkpoint_sha256") != checkpoint:
        raise SystemExit(f"Locked completion changed: {path}")
    if value.get("epochs") != [21, 22, 23, 24, 25] or value.get("test_used"):
        raise SystemExit(f"Locked protocol changed: {path}")
print("BCPCH_LOCKED_ARTIFACTS_PASS", flush=True)
PY

if [[ ! -e "${REPORTS}/bcpch_preflight.json" ]]; then
  python tools/preflight_bcpch.py \
    --train-root "${TRAIN_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON}" \
    --output "${REPORTS}/bcpch_preflight.json" \
    --num-workers "${NUM_WORKERS}"
fi

if [[ -e "${BCPCH_DIR}/complete.json" ]]; then
  python - "${BCPCH_DIR}/complete.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "BCPCH_MATCHED_COMPLETE":
    raise SystemExit("Existing BCP-CH run is invalid")
if value.get("epochs") != [21, 22, 23, 24, 25] or value.get("test_used"):
    raise SystemExit("Existing BCP-CH protocol differs")
print("BCPCH_REUSE_COMPLETE", flush=True)
PY
else
  python tools/train_bcpch_matched.py \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON}" \
    --output-dir "${BCPCH_DIR}" \
    --num-workers "${NUM_WORKERS}"
fi

python tools/analyze_bcpch_phase3.py \
  --common-checkpoint "${COMMON}" \
  --schedule "${SCHEDULE}" \
  --preflight "${REPORTS}/bcpch_preflight.json" \
  --c0-dir "${C0_DIR}" \
  --bcch-dir "${BCCH_DIR}" \
  --cbcch-dir "${CBCCH_DIR}" \
  --bcpch-dir "${BCPCH_DIR}" \
  --output-dir "${REPORTS}"

echo "BCPCH_PHASE3_COMPLETE"
