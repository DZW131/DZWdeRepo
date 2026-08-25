#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT LOCKED_WDCH_EXPERIMENT BCCH_EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
LOCKED="$3"
EXPERIMENT_DIR="$4"
NUM_WORKERS="$5"
COMMON="${LOCKED}/matched/common/common_epoch20.pth"
SCHEDULE="${LOCKED}/schedule/wdch_25epoch_schedule.npz"
C0_DIR="${LOCKED}/matched/C0"
W1_DIR="${LOCKED}/matched/W1"
BCCH_DIR="${EXPERIMENT_DIR}/matched/BCCH"
REPORTS="${EXPERIMENT_DIR}/reports"

mkdir -p "${REPORTS}" "${EXPERIMENT_DIR}/matched" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

python - "${COMMON}" "${SCHEDULE}" "${C0_DIR}/complete.json" "${W1_DIR}/complete.json" <<'PY'
import hashlib
import json
import sys

expected = {
    sys.argv[1]: "2aae7e7c83373a4bb8865084ede86ba91a79ae3788b732b19fa478ee6c4311fb",
    sys.argv[2]: "fa648405f40852e98f3d73776b7feee904bd59309ea1df2a97255650b0d00eea",
}
for path, digest in expected.items():
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if actual != digest:
        raise SystemExit(f"Locked artifact changed: {path}: {actual}")
for path, branch, checkpoint_sha in (
    (sys.argv[3], "C0", "44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8"),
    (sys.argv[4], "W1", "31976d27e5670256bd08565bb8b34efb510442e4afc0651a7797fee68d88b7fc"),
):
    value = json.load(open(path, encoding="utf-8"))
    if value.get("status") != "WDCH_MATCHED_BRANCH_COMPLETE" or value.get("branch") != branch:
        raise SystemExit(f"Invalid locked {branch} completion")
    if value.get("checkpoint_sha256") != checkpoint_sha:
        raise SystemExit(f"Locked {branch} checkpoint changed")
    if value.get("epochs") != [21, 22, 23, 24, 25] or value.get("test_used"):
        raise SystemExit(f"Invalid locked {branch} protocol")
print("BCCH_LOCKED_ARTIFACTS_PASS", flush=True)
PY

if [[ ! -e "${REPORTS}/bcch_preflight.json" ]]; then
  python tools/preflight_bcch.py \
    --train-root "${TRAIN_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON}" \
    --output "${REPORTS}/bcch_preflight.json" \
    --num-workers "${NUM_WORKERS}"
fi

if [[ -e "${BCCH_DIR}/complete.json" ]]; then
  python - "${BCCH_DIR}/complete.json" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "BCCH_MATCHED_COMPLETE":
    raise SystemExit("Existing BCCH run is incomplete")
if value.get("epochs") != [21, 22, 23, 24, 25] or value.get("test_used"):
    raise SystemExit("Existing BCCH protocol differs")
print("BCCH_REUSE_COMPLETE", flush=True)
PY
else
  python tools/train_bcch_matched.py \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON}" \
    --output-dir "${BCCH_DIR}" \
    --num-workers "${NUM_WORKERS}"
fi

python tools/analyze_bcch_phase1.py \
  --val-root "${VAL_ROOT}" \
  --common-checkpoint "${COMMON}" \
  --schedule "${SCHEDULE}" \
  --preflight "${REPORTS}/bcch_preflight.json" \
  --c0-dir "${C0_DIR}" \
  --w1-dir "${W1_DIR}" \
  --bcch-dir "${BCCH_DIR}" \
  --output-dir "${REPORTS}"

echo "BCCH_PHASE1_COMPLETE"
