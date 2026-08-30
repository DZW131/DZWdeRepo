# Phase-2B1.11 approved pre-outcome contract

User approved the recommended contract before this phase's candidate results were computed. Source specification SHA256 b5da257d95e0cf13c9b041eede6db26ff527234ad6e270a7c96b5194601b6a19.

## Scope and provenance

- Pure A0 4e9a2887b220d17e27649d72a3d13f32b7ebe8f9. Independent feature/rddr-phase2b111-third-evidence; PR base baseline/official-a0. No network/tool/train_sshr.py changes, no architecture imports.
- All3418 BCSS validation native28 observations. Frozen C0 seed42 Full25 checkpoint SHA509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579. Pin native/derived/Phase19 observations and Phase110 summary/runtime/identity/verification; check hashes before and after.
- No training/model loading/network forward/backward/autograd/optimizer/step/BN update/checkpoint write/new seed/test/LUAD/train split access. Frozen probability-only GPU replay is allowed. Existing files retained, new output directories only.
- Phase110 A/B/C/D = PASS/FAIL/FAIL/FAIL, decision RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED unchanged; neither its GT-defined context rescue nor earlier hierarchy diagnostics establish safe deployment.

## GT-blind construction and GT-only evaluation

- Compute U_R=(mD==0) on ALL native784 positions with no GT. First-index argmax for cs/cd/cc; a_alt=(cc!=cs)&(cc!=cd); M_alt=ctx(cc)-max(ctx(cs),ctx(cd)); A_alt=U_R&a_alt&(M_alt>0). Strict zero rejected, no epsilon relabel. Report argmax ties and old a_alt-vs-strict differences.
- Report all-native candidate/universe counts, then separate foreground/background4/ignore255 counts. Scores/candidate functions accept no labels. All scientific precision/ranking/GT utility evaluation uses foreground0-3 AFTER construction; primary precision denominator is ALL foreground candidates, never only Both-Wrong.
- Primary rescue/gradient score M_alt. Neither-hierarchy score where(A_alt,M_alt,0) over all foreground U_R. Controls C_ctx=max(ctx), E_ctx=sum(ctx*log(ctx+eps)), frozen q, frozen Delta, D_hier=1-max(max(ps),max(pd)), fixed directions, no fusion/substitution. Controls retain their raw values in U_R (only primary detection score is zero outside candidates).
- Complete hard-state partition: Both-Wrong+ctx-correct; Both-Wrong+ctx-wrong; Deep-Win; Shallow-Win; Stable-Correct; other count must reconcile. Candidate-only raw->ctx hard events and full-universe diagnostic predictions are BOTH reported. Noncandidates retain raw, not context. No prediction files or model state written.

## Context gradient, denominators, numerics

- Inherit KL=sum t*(log(t+eps)-log(p+eps)), t=detached ctx, eps1e-8. Per candidate, unweighted/unreduced: r=t*p/(p+eps); g=p*sum(r)-r, v=-g. No q multiplication, candidate normalization or training loss aggregation. Evaluate analytic formula using frozen FP32 p/t promoted to FP64; retain FP32 probability replay as source. Independent FP64 softmax from frozen logits and finite differences check numerical accuracy, do NOT replace main labels.
- Noncandidate gradients/dM are exactly zero. GT margin derivative uses all max-tied nonGT competitors, no epsilon-based >0/<0 relabel. Benefit>0/harm<0/zero==0; zeros only excluded from binary gradient ranking, never rates.
- Candidate all denominator is primary for Gate D. Protection uses full U_R Raw-Correct/Deep-Win/Shallow-Win/Stable-Correct denominators, inactive positions contribute zero; active-only measures disclosed. Unconditional ctx accuracy is labeled separately from actually activated hard effects.
- Probability/context/support/M_alt from frozen FP32; statistics and analytic accumulation FP64. Full exact FP32 support/context replay, q≤1e-7 with stored q retained, raw-probability replay≤1e-7. Independent context FP64≤1e-6; analytic vs independent real-logit numerical checks≤1e-6; formula synthetic finite difference checks. No backward even in tests.
- Reuse old boundary<=7px/interior>7px, Top20 mask and Q_EDGES [0.020935675129294395,0.072734534740448,0.163648784160614,0.3369627296924591], side=left. No residual-specific quantile recomputation.
- Undefined empty/single-label summaries are explicitly NA/JSON null, not numerical path failures. All tensor values must be finite.

## Bootstrap and gates

- 10000 paired image resamples of3418, default_rng42, same indices across all endpoints. Mean image AUC over dual-label images; pooled noninterpolated Average Precision with tied scores grouped. Percentile95%CI. No pixel bootstrap.
- ThirdRescue count-equivalent = bootstrap pooled rescue / bootstrap ALL Raw-Wrong * FIXED original708407. Compare its lower CI with FIXED31266 gap. Hard NetRepair count-equivalent uses bootstrap pooled net repair / bootstrap foreground U_R * fixed original foreground U_R; also report normalized U_R and all-foreground accuracy deltas. These hard metrics do NOT prove old gradient Gate E passes or future training gain.
- A: rescue>=31266 and count-equivalent lower>=31266 and CandidatePrecision>=.65 with lower>.55.
- B: candidate M_alt rescue imageAUC>=.65 with lower>.50.
- C: U_R M_alt_score Both-Wrong imageAUC>=.65 with lower>.50 AND BW prevalence among candidates > BW prevalence in U_R.
- D: all-candidate BenefitRate>HarmRate, Mean_dM>0, candidate M_alt gradient imageAUC>=.60.
- E: full U_R Raw-Correct hard harm<=.08 and dM harm<=.15; full rejected Deep-Win intrusion<=.12; full rejected Shallow-Win intrusion<=.10.
- F: interior candidate rescue imageAUC>.60 and at least3 powered classes imageAUC>.55. Class power>=500rescue,>=500failure,>=30dual images (gradient power separately reported). If observed conditions fail irrevocably F=FAIL; if only missing power could change result F=UNDERPOWERED, not auto-FAIL or PASS.
- Secondary confidence flag compares C_ctx vs M_alt candidate rescue imageAUC (same population). Strong requires all A-F PASS plus precision>=.75, rescueAUC>=.75, gradientAUC>=.70, rawcorrect hardharm<=.05, rescue>=2gap.
- Decision precedence: engineering/identity failure stops scientific adjudication; A FAIL -> OPERATIONAL_HEADROOM_INSUFFICIENT; A PASS and either B/C FAIL -> EXISTS_BUT_NOT_SELECTABLE; A/B/C PASS and D FAIL -> HARD_RESCUE_BUT_GRADIENT_UNSAFE; A-D PASS and E FAIL -> SIGNAL_WITH_PROTECTION_FAILURE; A-E PASS F FAIL -> SIGNAL_NOT_ROBUST; all PASS -> GTBLIND_FEASIBILITY_SUPPORTED (each prefixed THIRD_EVIDENCE_). If only F underpowered prevents a verdict, report evidence insufficient and pause rather than force one of six scientific decisions.

## Delivery and stop

- Required tests, independent replay/formula/ranking/count/bootstrap/decision verification; all22 required JSON/CSV; complete33-section Markdown; runnable README; PR without merge.
- New file/source identity explicitly separated from inherited model/BN/prediction identity. No claim of new prediction identity.
- STOP regardless of result. No new threshold/recovery mechanism/context loss/lambda/Full25 or follow-on phase without user approval.
