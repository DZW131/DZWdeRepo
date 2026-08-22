#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 TRAIN_ROOT VAL_ROOT A0_CHECKPOINT EXPERIMENT_DIR NUM_WORKERS" >&2
  exit 2
fi

TRAIN_ROOT="$1"
VAL_ROOT="$2"
A0_CHECKPOINT="$3"
EXPERIMENT_DIR="$4"
NUM_WORKERS="$5"
SCHEDULE="${EXPERIMENT_DIR}/schedule/utility_schedule.npz"

mkdir -p "${EXPERIMENT_DIR}/provenance" \
  "${EXPERIMENT_DIR}/schedule" \
  "${EXPERIMENT_DIR}/preflight"

git rev-parse HEAD > "${EXPERIMENT_DIR}/provenance/implementation_commit.txt"
python tools/build_utility_schedule.py \
  --train-root "${TRAIN_ROOT}" \
  --output "${SCHEDULE}"

python tools/preflight_tcrd_gate.py \
  --train-root "${TRAIN_ROOT}" \
  --val-root "${VAL_ROOT}" \
  --a0-checkpoint "${A0_CHECKPOINT}" \
  --schedule "${SCHEDULE}" \
  --output "${EXPERIMENT_DIR}/preflight/preflight.json"

for BRANCH in C0 D R DR; do
  python tools/train_tcrd_utility.py \
    --branch "${BRANCH}" \
    --train-root "${TRAIN_ROOT}" \
    --val-root "${VAL_ROOT}" \
    --a0-checkpoint "${A0_CHECKPOINT}" \
    --schedule "${SCHEDULE}" \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --num-workers "${NUM_WORKERS}"
done

python tools/eval_tcrd_utility.py --experiment-dir "${EXPERIMENT_DIR}"
