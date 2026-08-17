# CLRR-v2 Phase -1 Structure Audit

## Scope

This audit is based only on the official SSHR A0 import at
`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9` (PR #1). It contains no HST,
FA-MPR, SC-MPR, or CDSR code or history after that baseline commit.

## Classifier structure

| State | Classifier | Weight shape for BCSS | Bias | Mapping |
|---|---|---|---|---|
| `H_56 = feat_56_rectified` | `ic_56` | `[4, 256, 1, 1]` | `[4]` | `A_56 = ic_56(H_56)` |
| `H_28_1 = feat_28_1_rectified` | `ic1` | `[4, 512, 1, 1]` | `[4]` | `A_28_1 = ic1(H_28_1)` |
| `H_28_2 = feat_28_2_rectified` | `ic2` | `[4, 1024, 1, 1]` | `[4]` | `A_28_2 = ic2(H_28_2)` |
| `F_D = feat_deep` | `fc8` | `[4, 4096, 1, 1]` | none | `A_D = fc8(F_D)` |

The three recurrent HFRM states are direct inputs to their final 1x1 Conv2d
classifiers. There is no dropout, activation, normalization, or other
trainable nonlinear block between a recurrent state and its raw stage logits.
Therefore each stage satisfies exactly:

```text
A_i = W_i H_i + b_i
```

The `ReLU` in `Net_CAM.forward_cam()` is applied after the classifier only to
form the released nonnegative inference CAM. CLRR feedback must probe the raw
pre-ReLU logits. The deep training branch contains dropout, but the released
CAM inference computes deterministic `fc8(feat_deep)`; Phase 0 will use that
same deterministic deep anchor.

## Backprojection eligibility

For every target stage, the actual classifier weight can be detached and used
directly in:

```text
B_i = W_i^T (Pbar_i - P_i)
```

No learned projection or missing nonlinear Jacobian is required.

## Decision

**PHASE_MINUS1_PASS — proceed to the frozen A0 Phase-0 virtual-feedback
audit.** This decision authorizes audit tooling only, not CLRR training-model
implementation.
