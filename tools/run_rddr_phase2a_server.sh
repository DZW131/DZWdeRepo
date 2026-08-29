#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 OUTPUT_ROOT PRETRAINED TRAIN_ROOT PYTHON" >&2
  exit 2
fi

output_root=$1
pretrained=$2
train_root=$3
python_bin=$4

mkdir -p "$output_root"

run_variant() {
  local label=$1
  local mode=$2
  local run_dir="$output_root/$label"
  mkdir -p "$run_dir"
  set +e
  "$python_bin" train_sshr.py \
    --dataset bcss \
    --seed 42 \
    --batch_size 20 \
    --max_epoches 25 \
    --network network.resnet38_cls \
    --lr 0.01 \
    --wt_dec 0.0005 \
    --weights "$pretrained" \
    --trainroot "$train_root" \
    --save_folder "$run_dir" \
    --checkpoint_name stage1_last.pth \
    --eval_every 0 \
    --save-last-k-checkpoints 0 \
    --save-epoch-milestones 1,5,10,15,20 \
    --training-curve-path "$run_dir/training_curve.csv" \
    --optimizer-audit-path "$run_dir/optimizer_audit.json" \
    --n_class 4 \
    --img_size 224 \
    --num_workers 4 \
    --amp-dtype bf16 \
    --rddr-context-mode "$mode" \
    > "$run_dir/train.log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" > "$run_dir/exit_code"
  if [[ $status -ne 0 ]]; then
    return "$status"
  fi
}

run_variant GS global
run_variant RCS receiver
