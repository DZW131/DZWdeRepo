# Phase-2B1.10 approved pre-outcome contract

User confirmed the recommended contract. Pure A0 4e9a2887b220d17e27649d72a3d13f32b7ebe8f9; branch feature/rddr-phase2b110-residual-coverage, PR base baseline/official-a0. Original sources untouched.

## Scope / inputs

- All3418 frozen BCSS validation observations, C0 Full25 seed42. Native Phase2B1, symmetric Phase2B15, gradients Phase2B19 and Phase2B19 runtime/summary/identity are pinned by SHA256 before and after.
- No training, model instantiation, optimizer/step, weight/BN update, checkpoint write, new inference, test/LUAD/train split access or additional seed. Cache-only statistics; probability support/context replay is allowed, not network forward.
- Prior Phase2B19 decision remains ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE. Lower HHCR did not prove complete safety: negative Shallow-Win dM magnitude increased and class3 safety was underpowered.

## Equations / labels / controls

- S_S=.5*(T_SS+T_SD), S_D=.5*(T_DS+T_DD), Delta=S_D-S_S. Existing mD=(Delta>0) exact replay, no new recovery gate.
- Primary score S_D only. Diagnostic controls Delta (not negated/absolute), frozen q, max(pd)-max(ps), H(ps)-H(pd). Entropy nats=-sum p*log(p+1e-8), FP64 from frozen FP32 probabilities. No substitution/fusion/tuning.
- R=mD==0; foreground diagnostics GT0-3, background4/ignore255 excluded. R_RW=R & raw-wrong; exactly one-correct residual has positives Deep-Win, negatives Shallow-Win.
- UDT/ADT dM from frozen FP32 logit gradients, FP64 accumulation, v=-g and exact max-competitor tie directional derivative. Beneficial>0, harmful<0; zero separately reported and excluded only from binary ranking. No GT in score construction.
- RequiredAdditionalBenefit=ceil(2*N_RW/5)-B_ADT using integer counts, not rounded printed rates. Also verify equivalence with full-precision stated formula.
- Primary headroom rate=ResidualBeneficial/N_RW. Each bootstrap draw re-pools numerator/Raw-Wrong denominator; count-equivalent=that ratio times FIXED original N_RW. Compare its lower95%CI with FIXED required additional count. Residual prevalence is a separate denominator, not Gate A.
- Both Delta and S_D quintiles on R_RW INCLUDING zero derivatives; linear20/40/60/80 percentiles, ties stay together using searchsorted(side=left); actual counts disclosed. Diagnostic bins only, no deployment/training masks.
- Image-balanced AUC mean over dual-label images. Pooled AUPRC uses inherited non-interpolated Average Precision with score ties grouped. 10000 paired image bootstrap seed42, same draws for all endpoints; no pixel bootstrap.
- Context uses15x15 valid in-image neighbors excluding self, ctx_sym=.5*(ctx_S+ctx_D), original FP32 order. Scores/support/context must replay caches; inherited q recomputation tolerance1e-7 with stored q retained, no alteration near zero.
- Third-class means context pred differs from BOTH raw and deep. Rescue=third & correct; intrusion=third; precision=rescue/third; harm=third & wrong. On Both-Wrong rescue equals accuracy; these are not independent endpoints. No context loss.
- Context metrics: foreground4-class pooled confusion, absent union class NA excluded; NLL=-log(p_GT+1e-8), Brier=sum four squared class errors (not /4), no official background overwrite.

## Gates / completion of supplied decision logic

- A: residual beneficial count>=fixed gap AND lower95%CI count-equivalent>=fixed gap.
- B: S_D utility imageAUC>=.65 AND lowerCI>.50.
- C: S_D rejected winner imageAUC>=.65 AND lowerCI>.50.
- Class power: >=500 beneficial, >=500 harmful, >=30 dual-label images. D needs interior imageAUC>.60 AND >=3 powered classes with imageAUC>.55. Underpowered classes are not automatically failed or passed. If interior passes and missing evidence could bring the tally to3, D=UNDERPOWERED; otherwise unmet observed conditions D=FAIL. Neither is PASS.
- Third flag: rejected Both-Wrong ctx accuracy>=.25, lowerCI>.20, rescue>=.20. Strong flag: A-D PASS and both primary imageAUCs>=.75.
- Decision precedence: A notPASS -> RESIDUAL_COVERAGE_HEADROOM_INSUFFICIENT; A-D PASS -> DUAL_RESIDUAL_RECOVERY_SIGNAL_SUPPORTED if third flag else RESIDUAL_DEEP_RECOVERY_SIGNAL_SUPPORTED; A PASS with ANY B/C/D notPASS -> RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED if third flag else RESIDUAL_COVERAGE_NOT_RECOVERABLE_WITH_FROZEN_EVIDENCE. Includes the missing D-only-failure case.
- Failed asset/engineering verification stops delivery of scientific gate conclusions until resolved. Supported ranking is not a validated safe selector; not-supported refers to this preregistered score/criteria, not proof that all frozen evidence is useless.

## Delivery / stop

- Required24 tests, independent arithmetic/ranking/context/bootstrap/gate verifier; all required CSV/JSON, full29-section report, runnable README and PR. Existing outputs preserved, no overwrites.
- Identity records distinguish new file SHA/original-source checks from inherited Phase2B19 state/BN/prediction tests. No claim of a new model identity/inference test when none ran.
- STOP regardless of outcome. No recovery threshold/gate, lambda selection, training or Full25.
