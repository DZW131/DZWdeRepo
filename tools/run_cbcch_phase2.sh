#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT LOCKED_WDCH LOCKED_BCCH CBCCH_EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
LOCKED_WDCH="$3"
LOCKED_BCCH="$4"
EXPERIMENT_DIR="$5"
NUM_WORKERS="$6"
COMMON="${LOCKED_WDCH}/matched/common/common_epoch20.pth"
SCHEDULE="${LOCKED_WDCH}/schedule/wdch_25epoch_schedule.npz"
C0_DIR="${LOCKED_WDCH}/matched/C0"
W1_DIR="${LOCKED_WDCH}/matched/W1"
BCCH_DIR="${LOCKED_BCCH}/matched/BCCH"
A2_DIR="${EXPERIMENT_DIR}/matched/A2"
A3_DIR="${EXPERIMENT_DIR}/matched/A3"
REPORTS="${EXPERIMENT_DIR}/reports"

mkdir -p "${REPORTS}" "${EXPERIMENT_DIR}/matched" "${EXPERIMENT_DIR}/provenance"
git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"

python - "${COMMON}" "${SCHEDULE}" "${C0_DIR}/complete.json" "${W1_DIR}/complete.json" "${BCCH_DIR}/complete.json" <<'PY'
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
bcch = json.load(open(sys.argv[5], encoding="utf-8"))
if bcch.get("status") != "BCCH_MATCHED_COMPLETE":
    raise SystemExit("Invalid locked BC-CH completion")
if bcch.get("checkpoint_sha256") != "959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad":
    raise SystemExit("Locked BC-CH checkpoint changed")
for value in (
    json.load(open(sys.argv[3], encoding="utf-8")),
    json.load(open(sys.argv[4], encoding="utf-8")),
    bcch,
):
    if value.get("epochs") != [21, 22, 23, 24, 25] or value.get("test_used"):
        raise SystemExit("Invalid locked continuation protocol")
print("CBCCH_LOCKED_ARTIFACTS_PASS", flush=True)
PY

if [[ ! -e "${REPORTS}/cbcch_preflight.json" ]]; then
  python tools/preflight_cbcch.py \
    --train-root "${TRAIN_ROOT}" \
    --schedule "${SCHEDULE}" \
    --common-checkpoint "${COMMON}" \
    --output "${REPORTS}/cbcch_preflight.json" \
    --num-workers "${NUM_WORKERS}"
fi

run_variant() {
  local variant="$1"
  local output="$2"
  if [[ -e "${output}/complete.json" ]]; then
    python - "${output}/complete.json" "${variant}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "CBCCH_MATCHED_COMPLETE" or value.get("variant") != sys.argv[2]:
    raise SystemExit("Existing CBCCH result is invalid")
if value.get("epochs") != [21, 22, 23, 24, 25] or value.get("test_used"):
    raise SystemExit("Existing CBCCH protocol differs")
print(f"CBCCH_REUSE_COMPLETE variant={sys.argv[2]}", flush=True)
PY
  else
    python tools/train_cbcch_matched.py \
      --variant "${variant}" \
      --train-root "${TRAIN_ROOT}" \
      --val-root "${VAL_ROOT}" \
      --schedule "${SCHEDULE}" \
      --common-checkpoint "${COMMON}" \
      --output-dir "${output}" \
      --num-workers "${NUM_WORKERS}"
  fi
}

run_variant A2 "${A2_DIR}"
run_variant A3 "${A3_DIR}"

python tools/analyze_cbcch_phase2.py \
  --val-root "${VAL_ROOT}" \
  --common-checkpoint "${COMMON}" \
  --schedule "${SCHEDULE}" \
  --preflight "${REPORTS}/cbcch_preflight.json" \
  --c0-dir "${C0_DIR}" \
  --w1-dir "${W1_DIR}" \
  --bcch-dir "${BCCH_DIR}" \
  --a2-dir "${A2_DIR}" \
  --a3-dir "${A3_DIR}" \
  --output-dir "${REPORTS}"

echo "CBCCH_PHASE2_COMPLETE"
