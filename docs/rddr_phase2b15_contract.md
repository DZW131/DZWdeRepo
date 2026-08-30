# Phase-2B1.5 frozen execution contract

Approved by the user before any new result was calculated (2026-08-30).
Source: `RDDR_Phase2B1_5_Adjudication_Bias_Decomposition_Audit_v1.0.md`.

## Scope and immutable assets

- Independent pure A0 base `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
- C0 Full25 seed42 checkpoint SHA256:
  `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Phase-2B1 native cache SHA256:
  `767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`.
- All 3418 validation images; native 28-grid cached FP32 probabilities from
  original BF16 forward. No new network forward needed.
- No training, optimizer, backward, learned parameters, model checkpoint writes,
  network edits, test, LUAD, other seeds, search, or automatic next phase.
- Prior Phase-2B1 remains NOGO irrespective of this diagnostic result.

## Equations

Use the unchanged Phase-2B1 JSD helper: natural logs, epsilon1e-8, temperature1,
15x15/r7 in-image neighbors, exclude self. All neighbors participate without GT.
T_ab is mean clipped (1-JS(target_a,source_b)/ln2).

```
T_SS = support(S <- S); T_SD = support(S <- D)
T_DS = support(D <- S); T_DD = support(D <- D)
B_S = T_SS - T_SD; B_D = T_DD - T_DS
B_family = .5*(B_S+B_D)
Delta_old = Delta_Ssrc = T_DS-T_SS
Delta_Dsrc = T_DD-T_SD
S_S_sym = .5*(T_SS+T_SD); S_D_sym = .5*(T_DS+T_DD)
Delta_sym = S_D_sym-S_S_sym
wD_sym = S_D_sym/(S_S_sym+S_D_sym+eps)
anchor_sym = (1-wD_sym)*ps+wD_sym*pd
ctx_S = mean_neighbors(ps); ctx_D = mean_neighbors(pd)
ctx_sym = .5*(ctx_S+ctx_D)
Delta_ctx = JS(ctx,ps)-JS(ctx,pd)
```

Zero sign chooses deep only when score>0, else shallow (ties included).
Frozen old SS/SD/Delta must reproduce exactly or max absolute error<=1e-7;
otherwise STOP, do not relax tolerance. No offset/calibration/class sign flip.

## Statistical interpretation

- GT target classes0-3; 4/255 excluded only from metric targets.
- All means/quantiles are pooled valid-target summaries. AUROC primary is
  image-balanced, excluding undefined images (no artificial0.5); also report
  exact pooled AUROC and AP with ties. BA/recalls use pooled 2x2 confusion.
- Four-class mIoU/Dice from summed confusion; absent zero-union classes=NA.
  NLL=-log(pGT+eps); Brier=sum of four squared errors. No GT overwrite.
- 10,000 paired image bootstrap, seed42, percentile95% CI. All 3418 images
  resampled as clusters. Bias mean CIs use resampled sums/counts, AUROC uses
  finite image means, classification/segmentation use resampled confusions.
- Reuse exact old winner, Top20, q Q1-Q5, boundary/interior populations.
- Ordered pairs: all12, min winner count<100 => LOW_SUPPORT; no pair conclusions.
- Class2/class3 min winner count<500 => UNDERPOWERED, not PASS or FAIL.
  Class3 Shallow-Win count418 is known before this audit and remains underpowered.
- Report fixed source variants, never choose a best source/pair/subset.
- Candidate mass margin is deep-candidate minus shallow-candidate mass.
- GT neighbor denominator is all legal non-self neighbors. Report separate
  foreground-other (not either candidate), background, ignore fractions.
  Candidate-S + candidate-D + other + background + ignore sums to1 on
  disagreement targets. GT-same overlaps these categories and is not added.
  GT composition is computed in an isolated diagnostic helper, never fed back.
- Both-Wrong accuracy equals ThirdClassRescueRate by definition; on one-correct
  targets intrusion equals harm. Verify both identities and do not claim
  independent evidence from these duplicated endpoints.

## Gates and approved completion of decision logic

A: all-FG mean B_S>0 and B_D>0, both95%CI lower>0.
B: symmetric imageAUC>=.70, pooledBA>=.62, bothrecalls>=.55,
   abs(all-FG mean sym)<.5*abs(all-FG mean old).
C class-level: UNDERPOWERED first; otherwise imageAUC>=.45 => PASS, else FAIL.
C aggregate: any powered FAIL => FAIL; else any UNDERPOWERED => UNDERPOWERED;
   otherwise PASS.
D: ctx_sym Both-Wrong accuracy>=.25, its CI lower>.20, rescue>=.15, harm<=.10.

Decision precedence:
1. A FAIL => SAME_FAMILY_BIAS_HYPOTHESIS_NOT_SUPPORTED.
2. A PASS/B FAIL => THIRD_EVIDENCE_REQUIRED_FOR_NEXT_DESIGN if D PASS,
   otherwise ADJUDICATION_BIAS_UNRESOLVED.
3. A/B PASS/C FAIL => ADJUDICATION_BIAS_UNRESOLVED (approved missing branch).
4. A/B PASS/C UNDERPOWERED => SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED.
5. A/B/C PASS => SYMMETRIC_ADJUDICATION_BIAS_RESOLVED.

Strong symmetry: imageAUC>=.75, BA>=.65, bothrecalls>=.60.
Strong third evidence: accuracy>=.30, rescue>=.20, harm<=.08.
Flags never override gates or authorize training. Report D as third-evidence
support independently of the final decision; not as a deployed third branch.

## Delivery / STOP

All specified CSV/JSON and the complete Markdown report, 10k bootstrap
replicates, independent verification, tests, paths/hashes/commands, and an
unmerged PR. No model-design step or training follows automatically.
