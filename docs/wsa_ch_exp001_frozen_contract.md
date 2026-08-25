# WSA-CH EXP001 Semantic Assignment Feasibility Contract

This validation-only contract was frozen before inspecting EXP001 outputs.

## Scope

- BCSS validation only; canonical unflipped 224 x 224 images; BF16 inference.
- Locked C0 and CBCCH-A3 Epoch-25 final checkpoints from the matched
  Epoch-20-to-25 experiments.
- No training, test/LUAD access, checkpoint selection, parameter tuning, or
  architecture modification.

## Same-space representations

At HFRM28_1, the CBCCH backbone feature is the shared coordinate system:

- `F`: CBCCH HFRM28_1 input;
- `F_CH = CH_C0(F)`: locked C0 CH15 applied to that same `F`;
- `F_b = P_affinity(F)`: CBCCH local semantic-affinity output before routing.

## Primary semantic groups

For every foreground class present in the image-level weak label, the locked
C0 `ic1` probe is applied to `F_CH`. Spatial weights are
`softmax(ReLU(CAM_c))` with implicit temperature 1.0 and no threshold. The
class group `G_c` is the normalized weighted mean of `F_CH`.

Segmentation ground truth is not used to build primary groups. It is used only
to identify GT boundary pixels and to score same/wrong assignment.

## Oracle ceiling

A secondary oracle forms each `G_c` from GT interior (`>7 px`) pixels of
`F_CH`. It is an upper-bound diagnostic only and cannot affect the decision.

## Assignment task

- GT boundary: foreground pixels within 7 px of a foreground-class
  transition, reusing the frozen HMA/BC-CH definition.
- Candidate groups: foreground classes present in the image weak label.
- Images require at least two candidate classes and at least one eligible GT
  boundary pixel; exclusions are reported.
- `G_wrong` is the highest-similarity wrong candidate group.
- `margin = sim(query, G_same) - max_wrong sim(query, G_wrong)`.
- Queries are compared for raw `F`, `F_CH`, and CBCCH `F_b` against the same
  primary groups.
- Easy boundary pixels are correctly assigned by raw `F`; hard pixels are
  incorrectly assigned by raw `F`.

## Aggregation and uncertainty

- Report pixel-pooled, image-balanced, and per-class margin/accuracy.
- Report `F_b` correction rate on raw-hard pixels and harm rate on raw-easy
  pixels.
- Use image-level paired bootstrap with seed 42 and 10,000 resamples.

## Frozen decision

`WSA_CH_EXP001_GO` requires all three gates:

1. image-balanced `F_b` margin is positive and its bootstrap 95% CI lower
   bound is above zero;
2. image-balanced `F_b` assignment accuracy minus per-image `1/K` chance is
   positive and its CI lower bound is above zero;
3. paired image-balanced `F_b - F` assignment-accuracy improvement is positive
   and its CI lower bound is above zero.

If gates 1 and 2 pass but gate 3 fails, the frozen result is
`WSA_CH_ASSIGNMENT_EXISTS_NO_REFINEMENT_GAIN`. Otherwise it is
`WSA_CH_EXP001_NOGO`. Only `WSA_CH_EXP001_GO` may justify a separately
preregistered full WSA-CH implementation.
