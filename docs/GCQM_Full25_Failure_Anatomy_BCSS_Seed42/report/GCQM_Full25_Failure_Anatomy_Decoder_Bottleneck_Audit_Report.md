# GCQM Full25 Failure Anatomy & Decoder Bottleneck Audit Report

## 1 Executive Diagnosis

The sealed E25 audit identifies **MISSING_SPATIAL_COHERENCE** with **HIGH** confidence. Because Full25 failure is dominated by interior coherence loss, FN-biased missing tissue, and consistently elevated components/holes/compactness, the next decoder should restore region homogeneity while preserving CCRA.

## 2 Frozen Evidence

GCQM E25 SHA256 `6e1b909fc86a870e652213831521e8ff552371a083f85faad7dac3a21d969d0f`; SSHR B0 SHA256 `b71e2c10c597b295e38775f44adf5c2674f2f956d6a74e9bee190ee45c27fa70`. No training, optimizer, parameter update, threshold tuning, or checkpoint selection occurred.

## 3 Reproduction Gate

`REPRODUCTION_GATE_PASS` on 3418 paired images. GCQM mIoU=64.3543, SSHR mIoU=66.6967, delta=-2.3424 pp.

## 4 Why This Is Not Mechanism Collapse

CCRA allocation remained internally valid at E25 (Stage3 JS=0.3637, top1 class difference=1.0000, D_perm=0.1026). This audit concerns the decoder translation from allocation to dense masks.

## 5 H1 FP-vs-FN

mean normalized delta FP=-2.8940, CI=[-7.811661122058156, 0.24882769187139683]; delta FN=+0.0227, CI=[0.01787270903331121, 0.027661577361096625] Decision: `FN_DOMINANT`.

## 6 H2 Interior-vs-Boundary

r=3 interior loss=+0.0241; boundary loss=+0.0114; direction is class-dependent (loss for classes 0/1, gain for 2/3) Decision: `INTERIOR_DOMINANT`. Radius 3 is primary; radii 1 and 5 are robustness checks.

## 7 H3 Contact-Region Analysis

5 powered d=3 pairs; maximum excess contact loss=-0.0042 (<0.03) Decision: `FALSE`. Distance 3 is primary; distances 1 and 5 are robustness checks.

## 8 H4 Top-Weight Basis Purity

weighted purity=0.6455, rival=0.2955, BG=0.0590, top-tail=+0.2733, oracle-k5 purity=0.7387 Decision: `MIXED`.

## 9 Contribution Purity

Contribution-mass target/rival/background fractions are recorded per image/class in `basis_purity/weighted_basis_purity.csv`.

## 10 H5 Weight Diffuseness

mean Neff=132.78 Decision: `VERY_DIFFUSE`.

## 11 H6 Diffuseness-vs-DeltaIoU

rho(Neff,deltaIoU)=-0.2497; rho(top5,deltaIoU)=+0.2207; k=10/20/50 deltas=-0.1063/-0.0828/-0.0600 pp, so truncation does not rescue it Decision: `STRONG`.

## 12 Neff Quartile Analysis

Fixed image-level mean-Neff quartiles and their delta IoU, FP/FN, boundary, and interior statistics are in `weight_diffuseness/neff_quartiles.csv`.

## 13 H7 Spatial Property Loss vs SSHR

Ranked labels: INTERIOR_COHERENCE_LOSS, UNDERSEGMENTATION, FRAGMENTATION. component delta=+0.2912, hole-count delta=+0.2448, compactness delta=+0.4788; all three worsen in 4/4 classes

## 14 Fragmentation / Holes / Compactness

Per-image/class morphology and paired summaries are in `spatial_property/`; calculations use the fixed valid-pixel domain.

## 15 Pixel-wise-vs-Global Diagnostic

GCQM-global mIoU=64.3543; FOMD-style pixel diagnostic mIoU=64.4510. This is diagnostic-only and not a new model result.

## 16 Top-k Diagnostic Curve

Fixed k=1,3,5,10,20,50,100,196 results are in `paired/topk_counterfactual_curve.csv`; no k was selected as a performance result.

## 17 Oracle Basis Ceiling

GT-ranked equal and purity-weighted k=1,3,5,10 basis ceilings are in `basis_purity/oracle_basis_ceiling.csv`; these are diagnostic-only.

## 18 Phase-0 Proxy-vs-GT Validity

PHASE0_PROXY_MISMATCH is UNDETERMINED because only aggregate train-cohort proxy snapshots exist; no invalid train-to-validation image pairing was fabricated.

## 19 Per-Class Failure Matrix

| Class | Delta IoU pp | FP/FN | Interior loss | Boundary loss | Weighted purity | Neff | Fragmentation delta |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | -1.6617 | FN | 0.0166 | 0.0117 | 0.7480 | 143.09 | 127.3307 |
| 1 | -3.0460 | FN | 0.0546 | 0.0325 | 0.6062 | 128.65 | 0.0070 |
| 2 | -1.8365 | MIXED | -0.0257 | -0.0329 | 0.5775 | 122.19 | -0.0677 |
| 3 | -2.8254 | MIXED | -0.0265 | -0.0292 | 0.5490 | 133.47 | -0.0042 |

## 20 Failure Phenotype Clusters

Descriptive z-scored KMeans uses k=3; k=2/4 assignments are retained only for robustness. Cluster representatives are nearest-to-centroid, never manually selected.

## 21 Representative Cases

Automatic cluster representatives, five wins, five similar images, and five failures are under `visualizations/` with predictions, error maps, bands, weights, bases, contributions, and top-k panels.

## 22 Hypothesis Decision Matrix

| Hypothesis | Result | Main Evidence | Confidence |
|---|---|---|---|
| H1 FP/FN bias | FN_DOMINANT | mean normalized delta FP=-2.8940, CI=[-7.811661122058156, 0.24882769187139683]; delta FN=+0.0227, CI=[0.01787270903331121, 0.027661577361096625] | High |
| H2 Interior/Boundary | INTERIOR_DOMINANT | r=3 interior loss=+0.0241; boundary loss=+0.0114; direction is class-dependent (loss for classes 0/1, gain for 2/3) | Med |
| H3 Contact failure | FALSE | 5 powered d=3 pairs; maximum excess contact loss=-0.0042 (<0.03) | High |
| H4 Basis purity | MIXED | weighted purity=0.6455, rival=0.2955, BG=0.0590, top-tail=+0.2733, oracle-k5 purity=0.7387 | High |
| H5 Weight diffuseness | VERY_DIFFUSE | mean Neff=132.78 | High |
| H6 Diffuseness-performance link | STRONG | rho(Neff,deltaIoU)=-0.2497; rho(top5,deltaIoU)=+0.2207; k=10/20/50 deltas=-0.1063/-0.0828/-0.0600 pp, so truncation does not rescue it | High |
| H7 Lost spatial property | INTERIOR_COHERENCE_LOSS | component delta=+0.2912, hole-count delta=+0.2448, compactness delta=+0.4788; all three worsen in 4/4 classes | High |

## 23 Decoder Bottleneck Decision

`DECISION = MISSING_SPATIAL_COHERENCE`

## 24 What Is Preserved

CCRA allocation is internally valid; class-conditioned allocation survives Full25; query identity remains meaningful; only the current global mixture decoder is performance-invalid on BCSS Seed42 Full25.

## 25 What Is Falsified

Healthy weak-semantic diagnostics are not sufficient to imply improved dense segmentation, and the current global mixture decoder is not an acceptable final model.

## 26 Exact Next Research Target

Restore region homogeneity/coherence while preserving CCRA.

## 27 What Must NOT Be Done

Do not call CCRA failed, tune E25 inference thresholds/top-k, select another checkpoint, treat oracle/pixel counterfactuals as model performance, or train a new model from this audit.

## 28 Exact Final Decision

DECISION = MISSING_SPATIAL_COHERENCE

CONFIDENCE = HIGH
