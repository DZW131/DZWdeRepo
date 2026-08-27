#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "usage: $0 COMMON SCHEDULE BCSS_TRAIN BCSS_VAL C0_DIR OUTPUT_DIR" >&2
  exit 2
fi

common="$1"
schedule="$2"
train_root="$3"
val_root="$4"
c0_dir="$5"
output="$6"

mkdir -p "$output/logs" "$output/matched" "$output/report"

python tools/run_lw_shr_phase0.py \
  --common-checkpoint "$common" \
  --schedule "$schedule" \
  --train-root "$train_root" \
  --output-dir "$output/phase0" \
  2>&1 | tee "$output/logs/phase0.log"

for variant in A1 A2; do
  if [[ ! -f "$output/matched/$variant/completion.json" ]]; then
    python tools/train_lw_shr_matched.py \
      --variant "$variant" \
      --common-checkpoint "$common" \
      --schedule "$schedule" \
      --phase0-summary "$output/phase0/lw_shr_phase0_summary.json" \
      --train-root "$train_root" \
      --val-root "$val_root" \
      --output-dir "$output/matched" \
      2>&1 | tee "$output/logs/${variant}.log"
  fi
done

decision_log="$output/logs/a3_decision.log"
python tools/analyze_lw_shr_phase1.py \
  --mode decision \
  --c0-dir "$c0_dir" \
  --experiment-root "$output/matched" \
  --val-root "$val_root" \
  --output-dir "$output/report" \
  2>&1 | tee "$decision_log"

if grep -q '^LW_SHR_A3_UNLOCK=YES$' "$decision_log"; then
  if [[ ! -f "$output/matched/A3/completion.json" ]]; then
    python tools/train_lw_shr_matched.py \
      --variant A3 \
      --common-checkpoint "$common" \
      --schedule "$schedule" \
      --phase0-summary "$output/phase0/lw_shr_phase0_summary.json" \
      --train-root "$train_root" \
      --val-root "$val_root" \
      --output-dir "$output/matched" \
      2>&1 | tee "$output/logs/A3.log"
  fi
fi

python tools/analyze_lw_shr_phase1.py \
  --mode final \
  --c0-dir "$c0_dir" \
  --experiment-root "$output/matched" \
  --val-root "$val_root" \
  --output-dir "$output/report" \
  2>&1 | tee "$output/logs/final_analysis.log"

echo LW_SHR_PHASE1_COMPLETE
