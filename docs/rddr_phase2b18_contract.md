# Phase-2B1.8 approved execution contract (pre-outcome)

User approved the recommended clarification of RDDR_Phase2B1_8_PreRectification_Teacher_Guidance_Audit_v1.0.md.
Pure A0: 4e9a2887b220d17e27649d72a3d13f32b7ebe8f9.
Independent branch: feature/rddr-phase2b18-prerect-guidance; PR base baseline/official-a0.
Only standalone audit tools/tests/docs/results. No original source changes or new model parameters.

## Frozen inputs

- C0 Full25 seed42 checkpoint: 509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579.
- Phase2B1 native: 767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a.
- Phase2B1.5 symmetric: 237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514.
- Phase2B1.6 rect: 5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5.
- All 3418 BCSS validation images in frozen cache order. Reuse frozen native28 ps/pd/q, symmetric teacher, Top20/quintiles/boundary masks.
- Need=q, NOT garbage/correctness/acceptance. Prior rect-guidance/acceptance failures remain No-Go.

## Raw student and gradient path

- F28_raw is ReLU(bn45(b4_5(...b4(feat56)))) immediately BEFORE HFRM28_1.
- Primary student = conv2d(F28_raw, ic1.weight.detach(), ic1.bias.detach()) within original BF16 autocast.
- Whole b4 stage (b4,b4_1,...,b4_5) and bn45 may temporarily require grad, including their affine BN parameters.
- All modules remain eval, all BN running buffers unchanged. This diagnostic permission does NOT change original training freeze rules.
- Require nonzero aggregate b4 convolution gradient, not merely bn45 gradient. Record individual/zero-image gradients.
- HFRMs, ic1, b3 and earlier, b5 and deeper remain outside primary gradient path.
- Shared-head diagnostic alone permits ic1 gradient; same loss/teacher/feature, no update.
- Deep source gradient must be None/0 through teacher construction. Shared upstream receiving legitimate student gradients is not a detach failure.

## Equations and numerical contract

- Symmetric teacher unchanged; q/teacher/fixed-average/deep source detached from training probes.
- Exactly Uraw, FAraw, PRG with coefficient1; plus one shared-head PRG diagnostic only.
- Inherited KL = sum t*(log(t+1e-8)-log(softmax(L)+1e-8)); no extra normalization/clipping.
- Main gradients batch1 with all784 positions per-image loss denominator, including background/ignore; batch20 whole-batch denominator.
- Frozen ps/pd/teacher replay tolerance <=1e-7; q <=1e-7 with cached q retained. Exact original-vs-frozen-head raw logits and original rect logits.
- Native cache contains probabilities, NOT old raw logits. Never reconstruct raw logits by log(ps). Save newly observed raw logits, verify two real paths match exactly and softmax reproduces frozen ps.
- g_q = derivative of sum_i JS(softmax(L_s)_i,p_d_i)/ln2 w.r.t. L_s, separate graph, with p_d detached; no gradient of q weighting enters PRG.
- BF16 real forward/backward; probability/logit-loss/q derivative FP32. Store FP32 gradients. Diagnostic inner products/norms/aggregates use FP64 accumulation.
- v=-g (unit step, NOT unit-normalized gradient). Tied-max dM uses max(v) among CURRENT tied maximal non-GT logits.
- dQ=<g_q,v>; CosCollapse=dQ/(norm(g_q)*norm(v)+1e-8), no changing epsilon for small norms.
- Strict signs >0/<0; report zero/ties. Rates divide by all foreground positions of the stratum, not nonzero subset.
- BRR denominator all Deep-Win, HHCR denominator all Shallow-Win; boundary/interior intersection if nonempty, otherwise NA.

## Metrics, bootstrap, shared-head denominator and decision

- Native28 4-class pooled confusion matrix on GT0-3 only; bg4/ignore255 excluded, absent union class NA excluded from macro. No official background overwrite in diagnostic metrics.
- NLL=-log(p_GT+eps); Brier=sum of four squared class errors (not /4). Raw/teacher metrics must reproduce historical full-precision cache outputs.
- 10000 paired image bootstrap seed42, all3418 images; recompute pooled estimands/confusion per replicate; no pixel bootstrap.
- Shared-head energy fraction = sum_images ||g_ic1||^2 / sum_images(||g_ic1||^2+||g_approved_upstream||^2).
- Never include feature-gradient energy in that denominator. Report feature and parameter norms separately; risk flag is parameterization-dependent, NOT measured future optimization allocation.
- All A/B/C/D/E and strong/localization/absorption flags exactly as supplied. E failure first; then A fail; then B or C fail; then D fail; otherwise GO.
- Shallow-Win risk operationalized by frozen HHCR<=.30 and teacher accuracy>=.60, not a post-hoc subjective cutoff.

## Engineering and stop

- Full3418 finite loss/logit/feature/upstream gradients and q derivative; real PRG plus shared-head parameter backwards.
- Fixed32 linspace indices and128 seed42 remaining random images; fixed20 first20 deterministic for BF16 smoke.
- Original infer fixed160 before/after with hashes BEFORE official background overwrite; full raw logit replay on fixed160.
- State_dict/BN/checkpoint identical, no optimizer object/step/save; no test/LUAD/train split/other seed/search.
- Reserved batch20 memory <=22GiB; selected b4-path memory does NOT prove full-unfrozen Full25 memory.
- Stream feature/parameter gradient statistics; no multi-GB feature-gradient cache needed.
- New output directories only. All old experiments and assets preserved.
- Independent FP32 replay and FP64 analytic/autograd derivative proof, independent bootstrap/decision checks.
- All specified CSV/JSON, full38-section Markdown, README, PR, then STOP even if GO. No Full25/lambda selection.
