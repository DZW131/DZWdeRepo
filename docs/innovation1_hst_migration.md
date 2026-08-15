# Innovation 1 HST Migration Report

## 1. Delivery scope

This branch migrates the complete, previously reviewed Innovation 1 design onto
the current official SSHR baseline in one pull request. A1, A2, and A3 remain
selectable controls, but they are delivered as one coherent HST implementation.

- New repository baseline: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- Official SSHR snapshot: `8a0d7e271070ad4588d0fa6cfcd904505bee1189`
- Legacy A1 implementation: `8a38e81`, `d34f55d`, `154bac6`
- Legacy A2 implementation/fix: `fc6f8c3`, `a0d0020`
- Legacy A3 implementation: `c84ee44` plus the reviewed A3 control-test fixes

Legacy checkpoints, training curves, and validation-selected results are not
migrated. They were produced under the earlier protocol and are not evidence
for the updated official baseline.

## 2. Experimental variants

| Variant | Progressive correction state | Target-conditioned transition | Hierarchy interaction |
|---|---:|---:|---:|
| A0 `hfrm` | no | no | no |
| A1 `hst/a1` | yes | no | identity |
| A2 `hst/a2` | yes | yes | identity |
| A3 `hst/a3` | yes | yes | MLP token mixer |

The official model remains the default:

```text
--rectifier hfrm
```

HST is opt-in:

```text
--rectifier hst --hst-variant a1|a2|a3
```

## 3. Architecture mapping

### 3.1 Shared semantic space

Each backbone hierarchy is projected without spatial feature fusion:

```text
z_i = LayerNorm(Linear(GAP(F_i)))
```

The latent dimension defaults to 256. No BatchNorm, decoder, raw-feature
pyramid, auxiliary supervision, or new loss is introduced.

### 3.2 A1: progressive-only control

```text
C_D = phi_D(z_D)
C_3 = C_D
C_2 = C_3
C_1 = C_2
```

The same correction intent is decoded by stage-specific gates. This isolates
the value of correction-state propagation from transition learning.

### 3.3 A2: stage-specific semantic transition

For each target hierarchy:

```text
u_i = concat(C_parent, z_i, C_parent - z_i, C_parent * z_i)
delta_i = MLP_i(u_i)
C_i = C_parent + rho_i * delta_i
```

The reviewed optimization fix is retained: `rho_3 = rho_2 = rho_1 = 0.01`.
Semantic residual scales remain zero-initialized. Manually setting every `rho`
to zero is an exact A1 degeneration control.

### 3.4 A3: hierarchy latent interaction

A3 stacks the descriptors in deep-to-shallow order:

```text
Z = [z_D, z_3, z_2, z_1]
Z_hat = Z + ChannelMixer(TokenMixer(LayerNorm(Z)))
```

Only four tokens are mixed. Stage positions are retained, and the interacted
descriptors feed the unchanged A2 transitions. Mamba is not used.

### 3.5 Rectification and CH

All HST variants use:

```text
w_i = sigmoid(W_i(C_i))
S_i = F_i * w_i
F_i^R = F_i + gamma_sem_i * S_i + gamma_ctx_i * CH_i(F_i)
```

`CH_i` is factored into a shared helper but keeps the original depthwise
convolution, kernel size 15, averaging-kernel initialization, state-dict key
layout, and numerical output.

## 4. Compatibility boundary

The migration changes only architecture selection and HST modules. It preserves
the latest official behavior for:

- ResNet38 backbone and frozen layers;
- CAM heads and ten-tensor public forward contract;
- classification loss weights `0.10 / 0.15 / 0.25 / 0.50`;
- released PolyOptimizer, including momentum `0.0005`;
- poly learning-rate schedule;
- augmentation and preprocessing;
- bf16 execution, cuDNN benchmark, and TF32 policy;
- 25-epoch final-checkpoint training command;
- official class thresholds, CAM fusion, three-view TTA, and metric.

Training and evaluation both call the same architecture resolver, so an HST
checkpoint cannot silently be evaluated with HFRM.

## 5. Files and interfaces

| File | Role |
|---|---|
| `network/hst/context.py` | Original Contextual Homogenization primitive |
| `network/hst/semantic_projector.py` | Hierarchy semantic projectors |
| `network/hst/transition_block.py` | A2/A3 target-conditioned residual transition |
| `network/hst/latent_interaction.py` | Identity and lightweight MLP interaction modes |
| `network/hst/hst_rectifier.py` | A1/A2/A3 correction-state execution and diagnostics |
| `network/resnet38_cls.py` | A0/HST switch while retaining public outputs |
| `train_sshr.py` | Shared training/evaluation architecture CLI |
| `tests/test_hst_a*.py` | Architecture, control, gradient, and compatibility tests |
| `tools/smoke_hst_a*.py` | Dataset-free optimization-readiness checks |
| `tools/profile_hst.py` | Parameter/FLOP/latency comparison |
| `tools/check_pretrained_load.py` | Pretrained backbone loading audit |

`forward_with_diagnostics()` exposes analysis-only tensors without changing
`forward()`:

- raw and interacted latent tokens;
- correction states and transition deltas;
- learnable transition scales;
- semantic gates and semantic features;
- context and rectified features;
- per-stage CAM logits.

## 6. Validation evidence

### 6.1 Unit and control tests

```text
python -m unittest discover -s tests -v
Ran 30 tests: PASS
```

The suite covers baseline tensor equivalence, state-key compatibility, CH
equivalence, A1 propagation, A2 `rho=0` degeneration, A3 identity degeneration,
configuration guards, optimizer-group uniqueness, frozen parameters, shape,
gradient connectivity, and finite outputs.

### 6.2 Optimization readiness

CPU structural smokes passed for all variants. In both A2 and A3, transition
MLPs and target projectors had finite nonzero gradients by step 2. A3's HLI path
also had finite nonzero gradients by step 2. The current branch still requires
a fresh CUDA smoke before any full HST training.

### 6.3 Complexity

Parameter counts from the migrated code:

| Variant | Total parameters | Rectifier parameters | Delta vs A0 |
|---|---:|---:|---:|
| A0 HFRM | 112,709,714 | 7,612,166 | - |
| A1 | 107,537,234 | 2,439,686 | -4.59% |
| A2 | 109,505,621 | 4,408,073 | -2.84% |
| A3 | 109,637,757 | 4,540,209 | -2.73% |

The 64 x 64 structural FLOP counter reports all HST variants within 0.07% of
A0. Current-branch CUDA latency and peak memory are reported separately after
server validation; old measurements are not reused.

## 7. Official-protocol command index

Choose `VARIANT=a1`, `a2`, or `a3`. Apart from the rectifier flags, this is the
updated official 25-epoch/final-checkpoint command.

```bash
DATA_ROOT=/path/to/weakly_seg_data
DATASET=bcss
DATASET_DIR=BCSS
SEED=42
VARIANT=a3

python train_sshr.py \
  --dataset "${DATASET}" \
  --rectifier hst \
  --hst-variant "${VARIANT}" \
  --hst-latent-dim 256 \
  --hst-context-kernel 15 \
  --seed "${SEED}" \
  --max_epoches 25 \
  --weights init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --trainroot "${DATA_ROOT}/${DATASET_DIR}/train" \
  --save_folder "checkpoints_hst_${VARIANT}_${DATASET}_seed${SEED}" \
  --eval_every 0 \
  --save-last-k-checkpoints 0 \
  --n_class 4 \
  --img_size 224 \
  --num_workers 4 \
  --amp-dtype bf16
```

Evaluation must repeat the HST variant:

```bash
python train_sshr.py \
  --evaluate-only \
  --dataset "${DATASET}" \
  --rectifier hst \
  --hst-variant "${VARIANT}" \
  --hst-latent-dim 256 \
  --hst-context-kernel 15 \
  --seed "${SEED}" \
  --weights "checkpoints_hst_${VARIANT}_${DATASET}_seed${SEED}/stage1_last.pth" \
  --testroot "${DATA_ROOT}/${DATASET_DIR}/test" \
  --n_class 4 \
  --img_size 224 \
  --num_workers 4 \
  --amp-dtype bf16
```

## 8. Server validation commands

```bash
python tools/smoke_hst_a1.py --device cuda --batch_size 2 --image_size 224
python tools/smoke_hst_a2.py --device cuda --batch_size 2 --image_size 224 --steps 10
python tools/smoke_hst_a3.py --device cuda --batch_size 2 --image_size 224 --steps 10
python tools/profile_hst.py --device cuda --batch_size 1 --image_size 224 \
  --hst_variant a3
python tools/check_pretrained_load.py \
  init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
  --rectifier hst --hst_variant a3
```

## 9. Handoff and deferred work

- This PR migrates and validates architecture code; it does not start HST full
  training.
- The currently running six-run A0 reproduction queue is independent and uses
  the merged `main` commit, not this branch.
- Future architecture selection must use validation data only if variants are
  compared. The official A0 reproduction remains a final-checkpoint reference.
- Raw-feature cascade, semantic-gap statistics, and visualization diagnostics
  remain analysis controls rather than prerequisites for this migration.
