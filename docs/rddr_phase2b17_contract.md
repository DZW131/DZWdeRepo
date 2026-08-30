# Phase-2B1.7 approved, pre-outcome execution contract

User approved the recommended clarification of `RDDR_Phase2B1_7_Contextual_Correction_Acceptance_Audit_v1.0.md`.
Pure A0 base: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.
New independent branch: `feature/rddr-phase2b17-acceptance`.
Only standalone audit/tools/tests/docs/results additions. No original model/train/inference/metric changes.

## Immutable assets

- C0 Full25 seed42 checkpoint SHA256 `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`.
- Phase2B1 cache SHA256 `767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a`.
- Phase2B1.5 cache SHA256 `237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514`.
- Phase2B1.6 gradients/logits SHA256 `5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5`.
- All 3418 BCSS validation images. Frozen foreground/Top20/quintile/boundary/HFRM transition masks reused.
- `tools/rddr_phase2b16_common.py` is the frozen prior audit mathematics utility, not an innovation model.

## Equations and numerics

- Rebuild p_rect by FP32 CUDA softmax of saved FP32 logits from the original BF16 forward; do not treat prior FP64 report statistics as a probability cache.
- First replay teacher/support/q and original U/CCA loss-gradient. Teacher/support <=1e-7, q <=1e-7; logit equality exact. Record all differences before new outcomes.
- New R_S/R_D/T_S/T_D use exactly `mean(1-JS/ln2)`, inherited epsilon-inside-log JS (eps=1e-8), 15x15, self excluded, in-image neighbors only. No extra clipping, offset or normalization of the new support/Delta.
- `Delta = .5*(T_S+T_D)-.5*(R_S+R_D)`; `m=Delta>0`; `a=relu(Delta)` only.
- U/CCA/HA/SA use unit coefficient and the prior epsilon-inside-log KL. Teacher, q, Delta, m, a all detached.
- Main loss denominator includes ALL 784 pixels per image (also background/ignore); batch1. Empty accepted image yields zero loss and zero gradient using eps. GT does not choose the loss population.
- Diagnostic Benefit/Harm/ActiveGradientFraction denominator = all valid GT foreground pixels in the specified stratum, including rejected zero-gradient pixels. Conditional accepted-only statistics are additional and cannot replace primary gates.
- dM labels reuse prior CCA gradients and the exact tied-max directional derivative. HA/SA can scale or zero, not reverse, accepted-pixel directions. Strict >0/<0; report ties and numerical exceptions.

## Statistics and flags

- Image-balanced AUROC = equal mean of image AUROCs with both labels present. Report dropped/eligible image counts. Pooled AUROC tie-ranks exact; AUPRC=average precision, positive label Teacher-Win or dM>0. Fixed score direction; no post-hoc flips.
- Sign BA/recalls/F1 use pooled counts at Delta>0. Confidence controls keep their specified directions; entropy uses natural logs and eps=1e-8; JS control is the inherited unnormalized JS.
- 10000 paired image-bootstrap resamples, seed42, all 3418 indices; recompute pooled estimands, average eligible per-image AUROCs within each resample. No pixel bootstrap.
- Per-class power: >=500 Teacher-Win, >=500 Rect-Win, >=30 images containing BOTH. Gate C cannot pass an underpowered class assessment. Existing source counts meet the rule in all four classes.
- Gate D ActiveGradientFraction refers to HA/all valid foreground; 0.10 threshold unchanged. No rates conditional on accepted targets in this gate.
- Soft flag compares SA vs HA ALL Mean_dM, then Rect_Correct HarmRate; strong AUROCs are image-balanced for both winner and gradient benefit.
- NetRepair rate equals teacher-minus-rect accuracy on the same region, not independent evidence.

## Complete decision precedence

1. Engineering/parity/detach/identity failure: stop with `ACCEPTANCE_AUDIT_ENGINEERING_NOGO`.
2. A fail: `CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED`.
3. B fail: `WINNER_ACCEPTANCE_NOT_GRADIENT_ACCEPTANCE`.
4. C fail: `ACCEPTANCE_SIGNAL_EXISTS_CONSUMPTION_UNSAFE`.
5. D fail: `ACCEPTANCE_PROTECTION_CAPACITY_FAIL`.
6. C underpowered with A/B/D passing: `ACCEPTANCE_SIGNAL_CLASS_SAFETY_UNDERPOWERED`.
7. All pass: `RDDR_PHASE2B17_ACCEPTANCE_GO`.

## Engineering and delivery

- No optimizer construction/step, checkpoint write, BN update, test/LUAD/other seeds, threshold/window/lambda/search, feature replacement, or Full25.
- Main logit gradients cover all 3418; real backward only enables the seven HFRM28_1/ic1 tensors approved in Phase2B16. All modules eval.
- Reuse the preselected fixed160 inference-identity images and first20 deterministic images for a supplemental BF16 smoke; reserved CUDA <=22 GiB. This is selected-path memory, NOT full-unfrozen training memory proof.
- Preserve all existing assets and create unique outputs. Complete all specified CSV/JSON, independent verification, full Markdown and PR against baseline/official-a0. Even if GO, stop for review.
