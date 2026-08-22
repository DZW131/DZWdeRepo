# TCER R-only BCSS Seed42 Fresh-25 Exploratory Run

This is a deliberately post-gate exploratory experiment. It tests whether the
R-only TCER mechanism needs full co-adaptation from the released ImageNet
ResNet38 initialization. It does not replace or reinterpret the preregistered
TCRD-v0 utility result `ROUTE_E_CLOSE`.

## Frozen protocol

- BCSS training/validation only (`23,422 / 3,418`)
- seed 42, fresh 25 epochs, batch 20, image size 224, BF16
- released SSHR augmentation, four classification losses, PolyOptimizer,
  learning-rate schedule and actual momentum `0.0005`
- released pretrained ResNet38 SHA256:
  `f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16`
- frozen TCER R-only equations, `T=3`, initial `eta_R=0.10`, symmetric
  competition matrix with off-diagonal value 1
- no validation during training and no checkpoint selection
- evaluate the epoch-25 FINAL checkpoint on BCSS validation only
- no test, LUAD, other seed, tuning, diffusion or auxiliary loss

## Files

- `tools/preflight_tcer_r_full25.py`: real batch20 BF16, no-step readiness audit
- `tools/train_tcer_r_full25.py`: fresh training plus one overwritten recovery state
- `tools/eval_tcer_r_full25.py`: final-checkpoint validation, paired A0 mechanism
  comparison, decision and report
- `tools/tcer_r_full25_common.py`: frozen constants and success criteria

## Commands

```bash
PYTHON=/path/to/sshr/python
TRAIN=/path/to/BCSS-WSSS/training
VAL=/path/to/BCSS-WSSS/val
PRETRAINED=/path/to/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
A0=/path/to/a0_seed42_epoch25_final.pth
RUN=/path/to/TCER_R_V0_BCSS_SEED42_25EP_EXPLORATORY_<commit>

$PYTHON tools/preflight_tcer_r_full25.py \
  --train-root "$TRAIN" --pretrained "$PRETRAINED" \
  --output "$RUN/preflight.json"

$PYTHON -u tools/train_tcer_r_full25.py \
  --train-root "$TRAIN" --pretrained "$PRETRAINED" \
  --output-dir "$RUN" --num-workers 4

$PYTHON -u tools/eval_tcer_r_full25.py \
  --run-dir "$RUN" --val-root "$VAL" --a0-checkpoint "$A0" \
  --num-workers 4
```

If interrupted at an epoch boundary, repeat the training command with
`--resume`. `resume_latest.pth` is overwritten each epoch and never participates
in evaluation or model selection.

## Exploratory decision

All three must pass:

- final fused mIoU delta versus A0 at least `+0.15 pp`
- standalone CAM28_1 mIoU delta at least `+0.20 pp`
- present-class confusion relative reduction at least `0.5%`

Otherwise the decision is `TCER_R25_EXPLORATORY_CLOSE`. The evaluator then
stops without unlocking test or another run.
