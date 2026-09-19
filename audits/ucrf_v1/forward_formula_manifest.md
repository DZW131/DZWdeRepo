# Frozen HQMR-v1 forward formula (UCRF-v1)

Source: `network/hqmr.py`, `network/hqmr_net.py`, `network/gcqm.py`, and
`tools/eval_gcqm_full25_bcss_seed42.py` on `audit/ucrf-v1`.

For Stage3, query index `q` is **not** tissue-class index `c`.

1. `q0 = LayerNorm(stage3_query)`.
2. `k5,v5 = independent Conv1x1+GroupNorm(context_feature)` and likewise
   `k4,v4` from `F4_context`, `k3,v3` from raw F3. The key and value
   projections are learned **frozen** checkpoint parameters.
3. `logits5[q,x] = dot(q0[q],k5[x])/sqrt(256)`.
4. `q5 = update5(q0,logits5,v5)`, where update uses FP32 normalized
   `sigmoid(logits5)` region pooling, a frozen linear projection, residual
   LayerNorm, frozen FFN, residual LayerNorm.
5. `direct4[q,x] = dot(q5[q],k4[x])/sqrt(256)`.
6. `logits4 = bilinear_resize(logits5, H4, align_corners=False) + direct4`.
   No extra activation, coefficient, or normalization is inserted here.
7. `q4 = update4(q5,logits4,v4)` with the same update structure as step 4.
8. `direct3[q,x] = dot(q4[q],k3[x])/sqrt(256)`.
9. `logits3 = bilinear_resize(logits4, H3, align_corners=False) + direct3`.
10. Frozen detached GCQM weights `W[q,c]` have `sum_q W[q,c]=1`.
    The actual tissue-class map at each stage is the **nonlinear** mixture
    `C_l[c,x] = clamp(sum_q W[q,c]*sigmoid(logits_l[q,x]),0,1)`.
    `direct4` alone is **not** a tissue-class logit.
11. Final inference bilinearly resizes each TTA view's Stage3 class mixture
    to the input image, unflips it, and averages three views. It min-max
    normalizes each class map separately, applies frozen deep-gate presence
    thresholds `[0.8,0.9,0.8,0.6]` (argmax fallback if none present), then
    chooses the maximum of the remaining class maps per pixel.

The audit distinguishes three quantities:

- **Actual class competition:** compare true and final-rival scores in `C_l`
  for `logits5`, upsampled `logits5`, `logits4`, upsampled `logits4`, and
  `logits3`. The `softmax` over region-level scores is only a diagnostic
  normalization; it is not HQMR's inference operation.
- **Exact query-logit arithmetic:** `logits4=U(logits5)+direct4` and
  `logits3=U(logits4)+direct3`, checked at runtime. Since sigmoid is
  nonlinear, the class mixtures do **not** add linearly.
- **Frozen arithmetic counterfactuals:** the actual direct4 class-margin
  effect is `margin(C4)-margin(C_U5)` with W held fixed. The isolated q4
  update effect compares `U(logits4)+direct_affinity(q4,k3)` to the same
  expression with `q5` in place of q4, using the same frozen k3 and W.
  Neither counterfactual changes the model's prediction.

Any use of the words "query effect" in outputs refers to this q4-vs-q5
arithmetic counterfactual. The observed `logits4→logits3` effect additionally
contains the direct3 feature/key term and must not be attributed to q4 alone.
