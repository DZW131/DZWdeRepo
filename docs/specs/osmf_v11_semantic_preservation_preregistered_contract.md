# OSMF-v1.1 Semantic Preservation — Preregistered Contract

This repository copy records the executable contract derived from the
author-supplied specification
`OSMF_v1.1_Semantic_Preservation_Factorization_Technical_Spec_v1.0.md`.

- Source SHA256: `66340fdbef15d0092870a7ad06081134670a094008783523cde28519c6c2ec23`
- Source length: 624 lines
- Prior result: `OSMF_PHASE0_NOGO`
- This is an independently preregistered version, not a continuation or
  hyperparameter repair of OSMF-v1.0.

## Scientific change

OSMF-v1.0 used a randomly initialized semantic auxiliary classifier and
stopped at batch 2 because `r_sem=4.106567, 2.480527`. OSMF-v1.1 changes only
the semantic objective formulation.

For post-HFRM `H28_1`:

```text
S     = P_sem(H)
M     = P_morph(H)
H_S   = U_sem(S)
H_M   = U_morph(M)
H_hat = H_S + H_M
```

The v1.0 `GAP(S) -> random Linear -> classification loss` path is deleted.
The pretrained SSHR CAM28_1 classifier geometry is reused without an auxiliary
gradient to it:

```text
Z_H = ic1(H)                                      # detached teacher target
Z_S = functional_ic1(H_S, weight=ic1.weight.detach())
```

Bias, when present, is also detached so the complete auxiliary path is unable
to update `ic1`; the original SSHR loss continues to update the live `ic1`.

Class-channel normalized spatial cosine preservation is:

```text
L_sem_pres = mean(1 - cosine(normalize(Z_S, dim=1),
                             normalize(Z_H.detach(), dim=1)))
```

No KL, temperature, classifier, MLP, prototype, EMA teacher, confidence mask,
spatial mask, or learned semantic policy is permitted.

## Frozen graph and objectives

- Factorization point: post-HFRM `H28_1`, 512 channels.
- Split: semantic/morphology `256/256`.
- Reconstruction form and complementary exact-identity initialization: frozen.
- `L_eq`, horizontal/vertical flips, interval 4: frozen.
- `L_orth`, `L_rec` with detached H target: frozen.
- Original SSHR task loss: frozen.
- Objective weights: semantic `0.20`, morphology `0.20`, orthogonality `0.05`, reconstruction `0.10`.
- Optimizer, schedule, augmentation, split, CAMs, fusion, threshold, TTA, and metric: frozen.
- New trainable parameters: exactly `p_sem`, `p_morph`, `u_sem`, `u_morph` weights.

## Gate 0 — Phase -1.1 parity

Before any optimizer step:

- `max|H_hat-H| < 1e-6`
- CAM56/CAM28_1/CAM28_2/CAMdeep differences exactly zero
- all 3,418 BCSS validation predictions identical
- absolute mIoU and mDice differences `<1e-7`

Failure returns `OSMF_V11_PARITY_NOGO` and stops.

## Gate 1 — 8 real BCSS batches

- Seed `20260817`, batch 20, image 224, official BF16 and optimizer.
- Audit points: `0,1,2,4,8`.
- Never process batch 9.
- Record all five losses, weighted ratios/cosines at H28_1, branch and response
  RMS, reconstruction, CrossCov, EqErr(M/S), semantic agreement, and all four
  parameter gradients/updates.

PASS requires:

- no two consecutive `r_sem_pres > 0.50`;
- mean `r_sem_pres <= 0.30`, or a step-4/8 value `<=0.30`;
- finite paths; active semantic and morphology paths;
- reconstruction cosine `>=0.95`; no branch/response collapse.

Outcomes are exactly:

- `OSMF_V11_SEMANTIC_READINESS_PASS`
- `OSMF_V11_SEMANTIC_READINESS_REVIEW`
- `OSMF_V11_SEMANTIC_READINESS_NOGO`

Only PASS authorizes Gate 2. REVIEW/NOGO stops without adjusting lambda.

## Gate 2 — 128 real BCSS batches

Gate 2 starts afresh from the same A0 checkpoint and fresh optimizer; it does
not continue the eight readiness updates.

- Seed `20260817`; same frozen protocol.
- Audit points: `0,1,2,4,8,16,32,64,96,128`.
- Continue all Gate-1 diagnostics and compute profiling.

Semantic response collapse is sustained `RMS(Z_S)/RMS(Z_H)<0.05`. Branch review
is outside `0.10–10`; branch NOGO is outside `0.05–20`. Reconstruction is GO at
`>=0.95`, REVIEW at `0.90–0.95`, and NOGO below `0.90`. Persistent auxiliary
ratio above `0.50`, dead/inactive paths, nonfinite state, loss explosion, false
decorrelation by collapse, or semantic response collapse is NOGO.

Outcomes are exactly:

- `OSMF_V11_PHASE0_GO`
- `OSMF_V11_PHASE0_REVIEW`
- `OSMF_V11_PHASE0_NOGO`

## Boundary

All outcomes stop after their authorized gate. Even Phase-0 GO does not permit
the 3-epoch pilot, 25-epoch training, test evaluation, LUAD, or other seeds.
No checkpoint from these audit updates is retained for continuation.

