# SSHR-HMA-v0 audit harness

This directory implements the frozen-checkpoint HFRM mechanism autopsy. It is deliberately isolated from the released model and training source.

Safety properties:

- exact source contract anchored to A0 commit `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`;
- exact checkpoint SHA256 enforcement;
- 32-image, three-TTA, same-process hard instrumentation parity before downstream analysis;
- BCSS validation-only causal inference;
- exactly 32 fixed training batches for gradient observation, using `torch.autograd.grad` only;
- no optimizer construction, optimizer step, test/LUAD evaluation, or model update;
- parameter and buffer SHA256 equality before/after the gradient audit.

Formal command:

```bash
python tools/audit_hma_v0.py \
  --val-root /path/to/BCSS-WSSS/val \
  --train-root /path/to/BCSS-WSSS/training \
  --checkpoint /path/to/stage1_last.pth \
  --output-dir /path/to/SSHR_HMA_V0_<audit_commit> \
  --amp-dtype bf16 \
  --num-workers 4
```

The command refuses a non-empty output directory and emits `HFRM_MECHANISM_MAP_COMPLETE` only after all required artifacts and the final report are generated.
