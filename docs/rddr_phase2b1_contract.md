# RDDR Phase-2B1 frozen execution contract

User approved 2026-08-30, before implementation/results. Source specification:
`RDDR_Phase2B1_Dual_Hypothesis_Context_Adjudication_Audit_v1.0.md`.

## Frozen inputs and boundaries

- Pure A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- C0 Full25 seed42 SHA256
  `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- All 3418 BCSS validation images; no training, optimizer, gradients, learnable
  module, checkpoint save, test, LUAD, other seeds, search, or automatic Phase2B2.
- Model eval, requires_grad=False, no_grad, batch1 BF16 forward, FP32 probabilities.
  Exact original forward plus read-only hook. No official source edits.
- Original Phase0 backend: cudnn.benchmark=False, matmul precision=none,
  cudnn convolution precision=tf32. Input224 bilinear/ImageNet normalization.
- Native28-grid logits: ic1(F28_raw), fc8(Ddeep). No ReLU on logits, CAM norm,
  presence threshold or TTA. Internal baseline feature activations are unchanged.

## Equations (no changes permitted)

Full four-class distributions ps=softmax(Ls.float()), pd=softmax(Ld.float()).
Natural logs, epsilon1e-8, temperature1; exact Phase0 JSD epsilon placement.
15x15 in-image neighbors, radius7, exclude self (index112), no distance weighting.

```
cS_ij = clip(1 - JS(ps_i,ps_j)/ln2,0,1)
cD_ij = clip(1 - JS(pd_i,ps_j)/ln2,0,1)
SS_i = mean_j cS_ij; SD_i = mean_j cD_ij
Delta_i = SD_i - SS_i
choose deep iff Delta_i > 0; zero tie chooses shallow
wD_i = SD_i/(SS_i+SD_i+eps); wS_i=1-wD_i
p_anchor = wS_i*ps_i+wD_i*pd_i
fixed_average = .5*ps_i+.5*pd_i
p_ctx = mean_j ps_j  # diagnostic only, not part of anchor
```

No source/target reliability, one-hot hypotheses, threshold search, softmax over
neighbors, temperature, top-k, power, structural term or posthoc replacement.
All legal spatial neighbors are included, including GT-background positions.
GT must not alter support, weights, anchor or consensus.

## Populations and statistics

- GT foreground0–3, background4/ignore255 excluded only from metrics. Nearest
  224->28 projection. All foreground is the primary anchor-utility population.
- Adjudication label only on hard disagreement AND exactly-one-correct:
  Deep-Win=1, Shallow-Win=0. Other pixels are not silently relabeled negatives.
- Exact tied-score AUROC/AP sorting, pooled and image-balanced; primary AUROC
  is mean per-image AUROC over images with both labels. Undefined images=NA,
  with counts. No artificial0.5. AP is noninterpolated average precision.
- Gate B: pooled 2x2 confusion matrix, balanced accuracy=(recall0+recall1)/2;
  if either GT class absent, balanced accuracy is undefined. Also report per-image.
- 10,000 paired image bootstrap, seed42, percentile95% CIs. AUROC image mean;
  sign metrics from resampled summed 2x2 matrices; segmentation accuracy/mIoU
  from summed 4x4 matrices. No target/pair-naive bootstrap.
- Four-class mIoU/macroDice omit zero-union/denominator classes (NA). No GT
  background prediction overwrite. NLL=-log(pGT+eps); Brier=sum four squared
  class errors, averaged over eligible targets.
- Reuse frozen Phase0 populations from SHA-verified cache; verify all per-image
  fullres counts, report nearest-projected counts. Top20 is never reselected.
- Frozen native q quintile boundaries reuse Phase2B0 values:
  0.020935675129294395 / 0.072734534740448 / 0.163648784160614 /
  0.3369627296924591. Verify exact higher-quantile replay. Ties go to lower bin.
- Strength=abs(Delta) quintiles use all foreground targets, fixed once with
  NumPy quantile(method=higher), before computing outcome metrics. No selection.
- Boundary=fullres 8-neighbor FG-FG transition Euclidean distance<=7px;
  interior>7px; nearest-project existing definition, never feed it into support.
- Calibration uses [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1] on winner population.
- Historical by-CH means raw->full HFRM including semantic veto/context, not a
  CH-only intervention. All conclusions retain this limitation.

## Frozen gates and approved decision completion

A: image AUROC>=.65 and CI lower>.50.
B: pooled sign BA>=.60, Deep-Win recall>=.55, Shallow-Win recall>=.55.
C: anchor-minus-fixed accuracy>0 and mIoU>0; at least one CI lower>0.
D: Deep-Wrong accuracy delta>=-.02 and Top20 Deep-Wrong delta>=-.03;
also no hard-line failure described below. All deltas relative to FixedAvg.

Hard line: any **aggregate preregistered** Deep-Wrong subset delta<=-.10 fails
safety: all, Top20, Bottom80, hard disagreement, Q1–Q5, boundary, interior,
class0–3. No per-image/pixel-triggered rule, posthoc subset, or minimum-count
search. Empty subset=NA, report coverage; no fabricated passing score.

Decision precedence (approved to fill the spec's missing B-failure case):
1. D fail/hard-line fail: ADJUDICATION_DEEP_WRONG_UNSAFE.
2. Otherwise A or B fail: RDDR_PHASE2B1_NOGO.
3. Otherwise C fail: ADJUDICATION_EXISTS_FUSION_UTILITY_FAIL.
4. All pass: RDDR_PHASE2B1_GO.

STRONG_SIGNAL is an independent descriptive flag: primary AUROC>=.70 AND
anchor-fixed mIoU>=.01 AND global Deep-Wrong delta>=0. It cannot override gates.
All reports, exact commands, code, independent verification and PR are delivered;
then STOP, including GO. Do not modify formulas after observing outcomes.
