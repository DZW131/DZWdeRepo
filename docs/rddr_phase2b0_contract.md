# Frozen execution contract — RDDR Phase-2B0

Approved by the user on 2026-08-30 before execution. This branch starts at
`4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. Only diagnostic tools/tests/docs
are added. No model, optimizer, training, inference or metric source is changed.

## Inputs and exclusions

- BCSS validation only, all 3418 images, sorted filenames, batch 1, 224 input.
- Frozen C0 Full25 seed42 SHA256:
  `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Actual A0 `Net.forward`, eval/no_grad, BF16 CUDA feature extraction; FP32
  softmax/JSD/relations. No TTA or CAM normalization for native-grid logits.
- Match frozen Phase-0 backend: benchmark false, matmul precision `none`,
  convolution precision `tf32`. Bootstrap seed42 is not a training seed change.
- No training, optimizer, gradient, new checkpoint, test, LUAD, other seeds,
  score/window/temperature search, model selection or automatic Phase-2B launch.

## Relations, GT and frozen populations

Use native 28x28 `ic1(F28_raw)` and `fc8(Ddeep)` probabilities. Natural log,
epsilon 1e-8, temperature 1. Exact Phase-0 epsilon placement is preserved.
`q=clip(JS(ps,pd)/ln2,0,1)`, `r=1-q`,
`c(i,j)=clip(1-JS(pd_i,ps_j)/ln2,0,1)`.
U=1, SR=r_j, SC=c_ij, primary SRSC=r_j*c_ij. Legal 15x15 spatial neighbors,
radius7, exclude center index112. No receiver reliability or extra term.

Nonoracle scores AND propagation use all in-image neighbors including positions
whose GT is background. GT must never filter a nonoracle propagation graph.
Pair metrics and purity use only foreground-target/foreground-source edges.
Mass/N_eff use the actual, GT-independent propagation neighborhood; foreground
source mass is separately reported. Source/target GT classes are 0–3;
background4 and ignore255 are excluded only from metric eligibility.

Verify every existing frozen population npz against its SHA256 manifest. Reuse
its raw/rect/Top20 maps; nearest-project full-resolution populations and GT to28.
Report full-resolution and projected counts. Do not reselect Top20. Historical
Corrected/Harmed-by-CH names compare raw to **full HFRM**, not an isolated CH
intervention. Original population masks were recreated in Phase-2A using exact
per-image count parity; original historical pixel-mask hashes did not exist.

Conflict quintiles use exact, precomputed quantiles (NumPy `method=higher`) of
cached native-grid q_feature on projected foreground, before any relation run;
ties go to the lower quintile. They do not redefine historical Top20.
Boundary is the historical full-resolution foreground/foreground 8-neighbor
transition distance <=7 pixels, then nearest-project; interior >7 pixels.

## Statistics fixed before results

- Pair AUROC: 4096 equal-width bins on [0,1], endpoint1 in last bin; tied-bin
  rank probability 0.5. AUPRC is noninterpolated average precision. Validate
  against exact tied-score sort on 16 evenly spaced sorted image indices.
- Target quantiles: streaming 4096-bin histograms, range[0,1] for purity/gain
  shifted to[0,1] as applicable; range[0,224] for mass/N_eff. Report bin error.
- Image-balanced means exclude undefined images, with eligibility counts.
  No artificial AUROC=0.5 when a class is absent. Pooled metrics are secondary.
- Corrected-positive target diagnostic scores have fixed orientation: purity,
  SRSC-U gain, and **negative** wrong-class weighted mass. No posthoc sign flip.
- Corrected–Harmed primary purity contrast is paired per-image on images with
  both groups; Harmed gain uses all images containing Harmed targets.
- All six primary intervals: 10,000 paired image bootstrap, seed42, percentile
  95% CI. AUROC/purity means are image-balanced; neighbor accuracy and mIoU
  recompute from summed 4x4 confusion matrices; Top20 NetRepair is pooled
  (repair-harm)/eligible Top20 count, recomputed per image resample.
- Target discrimination additionally reports pooled AUC and bootstrap of the
  image-balanced AUC. Absent pair-class images are excluded, counts reported.
- Four-class mIoU/Dice omit zero-union/zero-denominator classes (NaN), never
  assign perfect absent-class score. NLL uses log(p+1e-8); Brier is sum of four
  squared class errors, then mean over foreground targets. No background fix.
- Oracle uses same-GT foreground neighbors only as a diagnostic; no eligible
  same-class neighbor => undefined, excluded with explicit coverage, no fallback.

## Gates (all metrics on 0–1 scale)

A: image-balanced SRSC pair AUROC >=.65 and CI lower>.50.

B: image-balanced SRSC-U purity >=.03 and >0, CI lower>0,
image-balanced actual-neighborhood SRSC mean N_eff >=5.

C: paired image-balanced SRSC Corrected-Harmed purity >0, CI lower>0,
and image-balanced Harmed SRSC-U gain >0.

D: SRSC-U neighbor accuracy>0 and mIoU>0; at least one CI lower>0;
pooled Top20 NetRepair SRSC-U>0.

Decision precedence: A/B failure => RDDR_PHASE2B0_NOGO; A/B pass, C failure =>
RELATION_SIGNAL_NOT_CH_OUTCOME_SPECIFIC; A/B/C pass,D failure =>
RELATION_EXISTS_NO_PROPAGATION_UTILITY; all pass => RDDR_PHASE2B0_GO.
Secondary controls/subgroups cannot replace SRSC. STOP after report and PR.
