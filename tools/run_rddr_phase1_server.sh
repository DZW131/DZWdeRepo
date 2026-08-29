#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 OUTPUT_ROOT PRETRAINED TRAIN_ROOT PYTHON" >&2
  exit 2
fi

rddr_output_root="$1"
rddr_pretrained="$2"
rddr_train_root="$3"
rddr_python="$4"

mkdir -p "${rddr_output_root}/UC" "${rddr_output_root}/DD"

run_variant() {
  local variant_name="$1"
  local variant_mode="$2"
  local variant_dir="${rddr_output_root}/${variant_name}"

  "${rddr_python}" train_sshr.py \
    --dataset bcss \
    --seed 42 \
    --batch_size 20 \
    --max_epoches 25 \
    --network network.resnet38_cls \
    --lr 0.01 \
    --wt_dec 0.0005 \
    --weights "${rddr_pretrained}" \
    --trainroot "${rddr_train_root}" \
    --save_folder "${variant_dir}" \
    --checkpoint_name stage1_last.pth \
    --eval_every 0 \
    --save-last-k-checkpoints 0 \
    --save-epoch-milestones 1,5,10,15,20 \
    --training-curve-path "${variant_dir}/training_curve.csv" \
    --optimizer-audit-path "${variant_dir}/optimizer_audit.json" \
    --n_class 4 \
    --img_size 224 \
    --num_workers 4 \
    --amp-dtype bf16 \
    --rddr-phase1-mode "${variant_mode}" \
    > "${variant_dir}/train.log" 2>&1
  printf '0\n' > "${variant_dir}/exit_code"
}

run_variant UC uc
run_variant DD dd
printf 'RDDR_PHASE1_TRAINING_COMPLETE\n' > "${rddr_output_root}/training_complete"
