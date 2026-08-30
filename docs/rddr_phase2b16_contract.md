# Phase-2B1.6 approved execution contract

User-approved clarification of `RDDR_Phase2B1_6_Trainability_Integration_Audit_v1.0.md`.
Written before any new gradient result. Base: pure A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
Independent branch `feature/rddr-phase2b16-trainability`. No model, training, inference or metric source edits.

## Frozen inputs and population

- BCSS validation only, all 3418 cache-sorted images; checkpoint SHA256 `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Native cache SHA256 `767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`.
- Symmetric cache SHA256 `237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514`.
- Frozen native28 q, Top20 and Q1–Q5 edges; no reselection, thresholds or search.
- Unit-coefficient FP32 KL: `sum_k t_k*(log(t_k+1e-8)-log(softmax(L)_k+1e-8))`.
- U = mean KL(sym||rect); FA = sum(q*KL(fixedavg||rect))/(sum(q)+eps);
  CCA = sum(q*KL(sym||rect))/(sum(q)+eps).
- Loss denominator includes ALL native pixels, including GT background/ignore. GT only enters diagnostics.
- Main gradients use batch1/per-image denominator. Batch20 uses whole-batch denominator. BF16 forward, FP32 loss.
- Teacher and q are detached. No third/context teacher, no new model parameters, no classifier, no optimizer object or step.
- Only seven tensors under HFRM28_1 and ic1 require grad; eval mode throughout, all BN buffers frozen.

## Statistics and gates

- Metrics pool GT foreground 0–3 at native28, excluding background/ignore; absent union classes are NA and excluded from macro mean. No official background overwrite in this diagnostic metric. Official inference remains unmodified.
- dM uses actual infinitesimal directional derivative of max: among tied maximal non-GT logits use max(-g). Report ties. Strict >0/<0 signs, no tuned tolerance.
- Explicitly disclose positive-scalar CCA/U identity when q>0 and NetRepair rate = teacher accuracy - rect accuracy.
- Class sufficient support = at least 500 valid foreground pixels and 30 contributing images. Do not inherit old Shallow-Win counts.
- 10,000 paired image-bootstrap resamples, seed42, pooled estimands. No pixel bootstrap.
- Gate precedence D fail > A fail > B fail > C fail > C underpowered > all pass.
- B failure label: `CONFLICT_WEIGHTING_LOCALIZATION_NOT_SUPPORTED`.
- C: all and Top20 BenefitRate > HarmRate, at least 5/6 positive mean_dM across all/Top20/class0–3. Insufficient class support cannot be PASS.
- Preference flag: all-population CCA BenefitRate >= FA and Top20 CCA mean_dM >= FA.
- D branch nonzero means nonzero aggregated gradient in each context/semantic/head group; also require nonzero feature gradient and all finite throughout. Report individual zero-image counts, not silently replace them.

## Fixed engineering checks

- Deterministic 32 indices: `linspace(0,3417,32,dtype=int)`.
- Random 128: NumPy default_rng(42), without replacement from remaining sorted indices.
- Batch20 = first 20 of deterministic32. Save selected names before new forwards.
- Inference identity on fixed160: original `tool.infer_fun.infer`, unchanged TTA, thresholds, CAM fusion and decoding; restrict Dataset indices with audit-only patch and intercept scores solely to hash raw predictions before background overwrite. No inference source edits. Compare before vs after full backward audit.
- Full validation main logits are also replayed on selected160; compare zero-step before/after tensors. State_dict tensors/buffers and checkpoint SHA must remain identical.
- Peak CUDA reserved memory <=22 GiB. Selected-HFRM backward memory is NOT full-unfrozen Full25 memory evidence.
- No checkpoint writes; new output directories only; no deletion. Runtime/static access checks exclude test/LUAD. No Full25, lambda selection, or further experiments, even on PASS.

## Delivery

All required CSV/JSON plus independent NumPy verification, full Markdown report, README commands and PR against baseline/official-a0. Stop for review.
