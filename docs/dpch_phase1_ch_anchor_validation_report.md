# DPCH Phase-1 CH-as-Semantic-Anchor Validation

## 1. Executive conclusion

**Decision: `DPCH_PHASE1_NOGO`.**

The frozen CH-anchor hypothesis fails: guidance_utility.

This is a validation-only representation audit. It does not train or evaluate a dual-path model, so the guidance result is a feasibility signal rather than causal evidence of benefit.

## 2. Frozen contract and provenance

- Experiment: `EXP-DPCH-001`; implementation commit: `7f938172cde37b4bcb094a2e3a87af8148e81e5a`.
- BCSS validation only; 3,418 canonical, unflipped 224×224 views; BF16 inference.
- Primary coordinate system: CBCCH HFRM28_1 input `F`; `F_s=CH_C0(F)`; `F_b=P_affinity(F)`.
- Routed CBCCH context and independently forwarded C0 features are secondary controls.
- GT boundary/interior: foreground-class transition distance `≤7 px` / `>7 px`.
- Paired image bootstrap: 10,000 resamples, seed 42.
- No training, test, LUAD, alternate seed, checkpoint selection, or tuning.
- Exact command: `tools/run_dpch_phase1.py --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --c0-dir /home/duyanhong/experiments/WDCH_UTILITY_GATE_a00fb90/matched/C0 --bcch-dir /home/duyanhong/experiments/EXP-BCCH-001-f2a4c14/matched/BCCH --cbcch-dir /home/duyanhong/experiments/EXP-CBCCH-002-8057faa/matched/A3 --output-dir /home/duyanhong/experiments/EXP-DPCH-001-7f93817/full --num-workers 4 --bootstrap-resamples 10000`

| Artifact | SHA256 | Final val mIoU | Final val mDice |
|---|---|---:|---:|
| C0 | `44b8678f7d043c39488fc2d777d7b137ef8e379c6aa2c1859efedd35dd4a95b8` | 66.8555 | 79.9194 |
| BC-CH | `959ba77c16e440a8e69ece8740238f03f2711f4ea0faca737c33c4b92131f2ad` | 66.8429 | 79.9135 |
| CBCCH-A3 | `2a128636fba2417342e130787c88cd9d30410702f0797ad93b908173bc70cc4e` | 66.7300 | 79.8305 |

## 3. Semantic concentration

Values are image-balanced pixel-to-own-GT-class-centroid cosine similarities.

| Region | Raw F | CH anchor F_s | Δ(F_s−F) |
|---|---:|---:|---:|
| interior | 0.728095 | 0.972889 | 0.244794 |
| boundary | 0.690571 | 0.951557 | 0.260986 |

- Interior Δ bootstrap 95% CI: **0.244794 [0.243679, 0.245898]** across 3418 images.
- Inter-class centroid cosine changes from 0.802982 to 0.906985 (Δ=0.104003). Lower is more separated; this is a collapse guardrail, not a gate.

### Per-class concentration and compatibility

| Class | Interior raw | Interior F_s | Δ | Boundary raw | Boundary F_s | Δ | Boundary cos(F_b,F_s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.731187 | 0.971466 | 0.240279 | 0.677081 | 0.940998 | 0.263917 | 0.750035 |
| 1 | 0.742282 | 0.974147 | 0.231865 | 0.709949 | 0.957444 | 0.247495 | 0.745927 |
| 2 | 0.763868 | 0.981837 | 0.217969 | 0.723267 | 0.970448 | 0.247181 | 0.748081 |
| 3 | 0.734452 | 0.979986 | 0.245535 | 0.703593 | 0.967717 | 0.264123 | 0.733029 |

## 4. Boundary-to-CH compatibility

- Boundary mean `cos(F_b,F_s)`: **0.748601**.
- Interior mean `cos(F,F_s)`: **0.576327**.
- Boundary/interior ratio: **1.298917**.
- Routed CBCCH-context-to-CH boundary cosine: 0.712069 (secondary control).

## 5. Guidance-feasibility proxy

Positive samples are GT-boundary pixels where CBCCH-A3 corrects C0; negative samples are those where it harms C0.

| Aggregation | Corrected n | Harmed n | Corrected mean | Harmed mean | AUROC | Cohen's d |
|---|---:|---:|---:|---:|---:|---:|
| Pixel-level (gate) | 363933 | 291261 | 0.742975 | 0.743907 | 0.490210 | -0.032290 |
| Image-balanced control | 1106 | 1099 | 0.740982 | 0.741890 | 0.485974 | -0.037914 |

Against BC-CH rather than C0 (secondary): AUROC=0.494170, Cohen's d=-0.017633.

## 6. Cross-checkpoint sensitivity

The following literal comparison forwards C0 and CBCCH independently. It is not used for GO/NO-GO because independent continuation can rotate or rescale channels.

- Boundary `cos(F_b^CBCCH,F_s^C0)`: 0.782078.
- Interior `cos(F^C0,F_s^C0)`: 0.558846.
- Ratio: 1.399451.

## 7. Preregistered gates

| Gate | Criterion | Observed | Result |
|---|---|---:|:---:|
| Semantic concentration | Δ>0 and bootstrap CI low>0 | Δ=0.244794; low=0.243679 | PASS |
| Boundary compatibility | cosine≥0.50 and ratio≥0.80 | cosine=0.748601; ratio=1.298917 | PASS |
| Guidance utility | AUROC>0.55 and d>0.20 | AUROC=0.490210; d=-0.032290 | FAIL |

## 8. Scientific interpretation

CH strongly raises interior within-class concentration (delta=0.244794), and F_b is directionally compatible with F_s (cosine=0.748601). However, CH also raises inter-class centroid cosine by 0.104003, while the same F_b-to-F_s similarity is non-discriminative for corrections versus harms (AUROC=0.490210, Cohen d=-0.032290). Thus semantic concentration and directional alignment do not establish a useful teacher signal. The existing checkpoints do not justify Dual-path CH training, and this no-go must not be repaired by post-hoc threshold or representation changes.

## 9. Runtime and outputs

- Runtime: 0.57 min (0.0100 s/image).
- Peak CUDA allocated memory: 0.902 GiB.
- Machine-readable outputs: `dpch_phase1_summary.json`, `dpch_per_image.csv`, `dpch_per_class_image.csv`, and `dpch_per_class_summary.csv`.

STOP. No dual-path training was started.

