#!/usr/bin/env bash
# Evaluation only. This wrapper never calls train_sshr.py or optimizer.step().
set -euo pipefail
if [[ $# -lt 9 || $# -gt 11 ]]; then
  echo "usage: $0 EXP_ROOT PHASE0_DIR PHASE1_DIR C0_CHECKPOINT VAL_ROOT PRETRAINED TRAIN_ROOT PYTHON OUTPUT_DIR [MAX_IMAGES] [RESAMPLES]" >&2
  exit 2
fi
exp_root=$1
phase0=$2
phase1=$3
c0=$4
val_root=$5
pretrained=$6
train_root=$7
python_bin=$8
output=$9
max_images=${10:-0}
resamples=${11:-10000}
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite existing output: $output" >&2
  exit 2
fi
mkdir -p "$output"
set +e
"$python_bin" -u tools/analyze_rddr_phase2a.py \
  --c0-checkpoint "$c0" --gs-dir "$exp_root/GS" --rcs-dir "$exp_root/RCS" \
  --phase0-dir "$phase0" --phase1-dir "$phase1" --val-root "$val_root" \
  --smoke-json "$exp_root/diagnostics/rddr_phase2a_smoke.json" \
  --pretrained "$pretrained" --train-root "$train_root" \
  --python-executable "$python_bin" --output-dir "$output" \
  --num-workers 4 --max-images "$max_images" --bootstrap-resamples "$resamples" \
  > "$output/evaluation.log" 2>&1
status=$?
set -e
printf '%s\n' "$status" > "$output/exit_code"
exit "$status"
