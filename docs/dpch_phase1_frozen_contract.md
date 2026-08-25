# EXP-DPCH-001: CH-as-Semantic-Anchor Validation Contract

This contract was frozen before inspecting EXP-DPCH-001 outputs. Phase 1 is
validation-only and performs no training.

## Scope

- Dataset: BCSS validation split only.
- View: canonical, unflipped 224 x 224 input.
- Artifacts: locked C0, BC-CH, and CBCCH-A3 Epoch-25 final checkpoints and
  validation predictions from their matched Epoch-20-to-25 experiments.
- No test evaluation, optimization, threshold search, or model update.

## Primary same-space representation test

At HFRM28_1, use the CBCCH backbone feature `F` as the shared coordinate
space. Define:

- `F_s = CH_C0(F)`, applying the locked C0 CH15 kernel to that same `F`;
- `F_b = P_affinity(F)`, the CBCCH local contrastive-affinity representation;
- the routed CBCCH context is reported only as a secondary control.

A literal independently forwarded C0/CBCCH comparison is reported as a
sensitivity analysis and is not used for the primary decision.

## Spatial regions

Regions reuse the frozen HMA/BC-CH definition and are computed from
ground-truth foreground masks:

- boundary: foreground pixels whose distance to a foreground-class transition
  is at most 7 pixels;
- interior: foreground pixels whose distance to such a transition is greater
  than 7 pixels (reported as `>=8 px`).

Feature-space scalar maps are resized to mask resolution with bilinear
interpolation before aggregation.

## Frozen metrics

1. Semantic concentration: per-image, per-class pixel-to-class-centroid cosine
   for raw `F` and semantic consensus `F_s`, with interior and boundary strata.
   Inter-class centroid cosine is a guardrail against representational collapse.
2. Boundary compatibility: cosine between `F_b` and `F_s` on GT boundary
   pixels, compared with interior cosine between `F` and `F_s`.
3. Guidance-feasibility proxy: boundary similarity on pixels where CBCCH
   corrects C0 versus pixels where CBCCH harms C0, summarized by AUROC and
   Cohen's d. This is observational and is not a causal training claim.
4. Uncertainty: image-level paired bootstrap with seed 42 and 10,000
   resamples for the primary semantic-concentration delta.

## Frozen decision rule

`DPCH_PHASE1_GO` requires all three gates:

1. interior semantic-concentration delta (`F_s - F`) is positive and its
   paired-bootstrap 95% confidence-interval lower bound is above zero;
2. mean boundary `cos(F_b, F_s)` is at least 0.50 and is at least 80% of mean
   interior `cos(F, F_s)`;
3. corrected-versus-harmed AUROC is above 0.55 and Cohen's d is above 0.20.

Otherwise the decision is `DPCH_PHASE1_NOGO`. A no-go does not trigger an
alternative architecture or a new training run.
