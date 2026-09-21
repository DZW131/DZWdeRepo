# SSQA-v1 External Semantic Source Qualification — Final Report

> BCSS seed42 · frozen HQMR/UMRF components · no training · BBOX15 primary · patient-group DEV/HOLDOUT

## Executive decision

FINAL_DECISION = **CLOSE_ZERO_SHOT_VLM_SEMANTIC_ROUTE**<br>
READY_SOURCES = BiomedCLIP, PLIP, QuiltNet<br>
SKIPPED_SOURCES = {'CONCH': 'SKIP_ACCESS', 'CPLIP': 'SKIP_INCOMPATIBLE', 'CPathCLIP': 'SKIP_ACCESS'}<br>
BEST_DEV_SOURCE = BiomedCLIP<br>
BEST_HOLDOUT_SOURCE = BiomedCLIP<br>
SEMANTIC_SOURCE_GO = False<br>
ALL_TESTED_CANDIDATES_NOGO = True<br>
STRONG_SEMANTIC_SOURCE_GO = False<br>
DUAL_SOURCE_FOLLOWUP_JUSTIFIED = False<br>
ZERO_SHOT_VLM_ROUTE = CLOSED_BY_THIS_AUDIT<br>
NEXT_ROUTE = pathology visual foundation representation + image-level learned semantic head<br>
JEV_SIDECAR = SKIPPED_NO_API

BiomedCLIP: DEV_HTRP=61.24%, DEV_HTop1=24.62%, DEV_NRR=43.63%, HOLDOUT=55.03%; STATUS=READY<br>
CONCH: DEV_HTRP=—, DEV_HTop1=—, DEV_NRR=—, HOLDOUT=—; STATUS=SKIP_ACCESS<br>
CPLIP: DEV_HTRP=—, DEV_HTop1=—, DEV_NRR=—, HOLDOUT=—; STATUS=SKIP_INCOMPATIBLE<br>
CPathCLIP: DEV_HTRP=—, DEV_HTop1=—, DEV_NRR=—, HOLDOUT=—; STATUS=SKIP_ACCESS<br>
PLIP: DEV_HTRP=56.98%, DEV_HTop1=33.47%, DEV_NRR=0.00%, HOLDOUT=57.71%; STATUS=READY<br>
QuiltNet: DEV_HTRP=59.14%, DEV_HTop1=25.67%, DEV_NRR=33.87%, HOLDOUT=49.11%; STATUS=READY<br>

## Main results

| Source | Dev HTRP area | Holdout HTRP area | Holdout Top1 | NRR vs PLIP (DEV) | Control Top1 (DEV) | Class breadth ≥55% | Collapse | Decision |
|---|---:|---:|---:|---:|---:|---:|---|---|
| BiomedCLIP | 61.24% | 55.03% | 20.22% | 43.63% | 34.93% | 2/4 | False | SOURCE_NOGO |
| CONCH | — | — | — | — | — | — | — | SKIP_ACCESS |
| CPLIP | — | — | — | — | — | — | — | SKIP_INCOMPATIBLE |
| CPathCLIP | — | — | — | — | — | — | — | SKIP_ACCESS |
| PLIP | 56.98% | 57.71% | 33.68% | 0.00% | 41.91% | 2/4 | False | SOURCE_NOGO |
| QuiltNet | 59.14% | 49.11% | 23.90% | 33.87% | 37.68% | 3/4 | False | SOURCE_NOGO |

### Per-class DEV Hard-M1 HTRP (area-weighted)

| Source | Tumor | Stroma | Inflammation | Necrosis |
|---|---:|---:|---:|---:|
| BiomedCLIP | 83.92% | 17.73% | 88.98% | 45.03% |
| PLIP | 81.37% | 23.03% | 77.87% | 28.33% |
| QuiltNet | 94.16% | 17.09% | 59.69% | 63.89% |

### DEV predicted-class distribution on Hard-M1

| Source | Tumor | Stroma | Inflammation | Necrosis |
|---|---:|---:|---:|---:|
| BiomedCLIP | 32.93% | 5.29% | 20.94% | 40.84% |
| PLIP | 42.80% | 29.25% | 13.55% | 14.40% |
| QuiltNet | 68.78% | 2.62% | 6.49% | 22.11% |

Prediction/true frequencies, bias ratios and confusion matrices are saved both on Hard-M1 (`dev/class_bias.csv`, `dev/confusion_matrices.json`) and on the entire evaluable DEV cohort (`dev/class_bias_full_evaluable.csv`, `dev/confusion_full_evaluable.json`), with holdout equivalents only for selected sources and the PLIP reference.


## Protocol and leakage controls

Frozen UMRF components: 11,778 over 3,418 tiles; exact Hard-M1=5037, exact M1=4402. No components were selected or recropped using GT. Rival is the frozen sequential L5 class, not final segmentation.

Each predicted 8-connected component was cropped from its own bounding box with a fixed 15% expansion per dimension, made square within the original 224×224 tile, preserving surrounding H&E tissue. MASKED uses the same crop with external pixels filled with crop-mean RGB. It never sets the primary gate.

GT-free crop-size caveat: 1529 / 11778 crops are ≤4 px before native resizing, while 4352 span the full 224 px tile. These are fixed consequences of the frozen component geometry, not a tuned view. Component-weighted Top1 and whole-tile/context sensitivity must be interpreted accordingly.

All sources used the same four class phrases and five prompt templates. Each prompt embedding was normalized, averaged within class, and normalized again; image/text cosine and margins were computed in FP32 without logit scale. Official native tokenizers and image transforms were retained. No source was trained, tuned, or injected into CCRA.

Prompt SHA256: `97296837be55d0c79f85a1161e14ff59ea0e0ba5cf63d2603c0377cfcfbec09c`. GT table SHA256 (opened after all READY source caches were frozen): `60031c46e7a80f0949df4d4dcd859f5f7e970e1d7d6cbff8c058155932b16894`.

Patient split seed=20260921: 13 DEV and 9 HOLDOUT patients. Because true classes and Hard-M1 membership were sealed until embedding freeze, split stratification used only frozen predicted-class area, component area and counts as proxy; actual true-class balance was disclosed after unseal.

DEV true-class Hard-M1 counts: {'0': 805, '1': 1836, '2': 618, '3': 255}; HOLDOUT: {'0': 318, '1': 893, '2': 277, '3': 35}. Controls were greedily matched within each split by GT class, component-area quartile and baseline-confidence quartile, without replacement; achieved counts appear below.

Only the two highest DEV-ranked non-PLIP sources plus any additional STRONG_DEV source entered holdout. PLIP was evaluated there only as a fixed negative-control reference needed for NRR.

## Source availability and reproducibility

| Source | Official ID | Checkpoint SHA256 | Status / reason |
|---|---|---|---|
| BiomedCLIP | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | `52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be` | READY: HF commit 9f341de24bfb00180f1b847274256e9b65a3a32e |
| CONCH | `MahmoodLab/CONCH` | `—` | SKIP_ACCESS: gated; HTTP 401 without approved access |
| CPLIP | `iyyakuttiiyappan/CPLIP` | `—` | SKIP_INCOMPATIBLE: official Google Drive artifact fails ZIP CRC; missing model_configs in official code |
| CPathCLIP | `PathFoundation/CPath-Omni` | `—` | SKIP_ACCESS: Virchow2 base and delta absent on server |
| PLIP | `vinid/plip` | `98a7f8d2a1f4a8fc8f6dedb3a16ff7efbe02a7ef67c93904c80bca9767c69630` | READY: local frozen model |
| QuiltNet | `wisdomik/QuiltNet-B-16-PMB` | `642a4702d0fc9fd0ab0388d0340cb19ff8d234fd79155679d88f78e4ac0880e1` | READY: HF commit deb13a5596f2ff72a501fa121b26b3a4ed81705d |

PLIP uses frozen `vinid/plip` revision `67ade53ddd32195868f422585f72698ef5d15094`; the SSQA runtime used `transformers==4.35.2` and `open_clip_torch==2.23.0`. CONCH's official model endpoint denied unauthenticated access (HTTP 401); no mirror was substituted. Two independent downloads of the official CPLIP Google Drive file had identical SHA256 `ae30f04b869c3cefe8fff73335f42fdcbcbcaff45280d761be691789976190dc` and failed ZIP CRC at `archive/data/340`; official repository revision `82ee44972c7e967f97b40c53f18d7ea8e5e94ae9` also lacks the model configuration needed for reproducible loading. CPLIP was not evaluated. CPath-CLIP was skipped because the authorized Virchow2 base plus official delta were not present.

## Margins, controls, stability

- BiomedCLIP: DEV true−rival mean=-0.00105, median=-0.00137, Q25/Q75=-0.02720/+0.02282; matched controls=1632, ControlTop1=34.93%; HOLDOUT HTRP patient-bootstrap 95% CI=[44.81%, 65.44%], stable=True.
- PLIP: DEV true−rival mean=+0.00502, median=+0.00437, Q25/Q75=-0.00501/+0.01646; matched controls=1632, ControlTop1=41.91%; HOLDOUT HTRP patient-bootstrap 95% CI=[42.56%, 72.41%], stable=True.
- QuiltNet: DEV true−rival mean=+0.00297, median=+0.00250, Q25/Q75=-0.00494/+0.01113; matched controls=1632, ControlTop1=37.68%; HOLDOUT HTRP patient-bootstrap 95% CI=[35.79%, 67.17%], stable=False.

Bootstrap used 2,000 resamples with TCGA patient as cluster, seed 42. Complete CIs for HTRP, Top1, NRR, controls and margin are in `bootstrap/patient_cluster_ci.json`.

### MASKED view and component strata (secondary Q-DEV)

| Source | BBOX15 HTRP | MASKED HTRP | Context-dependent >15 pp | Area quartiles Q1–Q4 | Purity <0.5 / 0.5–0.7 / 0.7–0.9 / ≥0.9 |
|---|---:|---:|---|---|---|
| BiomedCLIP | 61.24% | 49.39% | False | 44.93% / 52.20% / 64.00% / 60.01% | 42.48% / 56.46% / 75.60% / 60.15% |
| PLIP | 56.98% | 43.08% | False | 65.73% / 62.99% / 62.66% / 49.89% | 30.94% / 52.03% / 66.02% / 58.80% |
| QuiltNet | 59.14% | 48.33% | False | 67.24% / 54.57% / 60.10% / 58.87% | 54.08% / 56.20% / 70.99% / 56.68% |

## Source independence and oracle ceiling

Q-DEV multi-source theoretical positive-rate oracle: 83.04%. This GT oracle is not deployable and is not a GO result. Pair agreement, error Jaccard/phi and rescue overlap are in `independence/`. Dual-source follow-up justified: False.

## Qualitative cases and secondary analyses

Fixed Q-DEV panels per READY source use groups A (PLIP fails, source rescues), B (both fail), C (baseline-correct control harmed). Counts available: {'BiomedCLIP': {'A_novel_rescue': 20, 'B_shared_failure': 20, 'C_control_harm': 20}, 'PLIP': {'A_novel_rescue': 0, 'B_shared_failure': 20, 'C_control_harm': 20}, 'QuiltNet': {'A_novel_rescue': 20, 'B_shared_failure': 20, 'C_control_harm': 20}}. All generated cases are in `visualizations/`; missing examples indicate the group had fewer than 20 cases. No holdout case was used to change prompt, model variant or crop. Whole-tile classification was not used for qualification.

## Final interpretation

No tested zero-shot source met both DEV and independent HOLDOUT qualification. Do not claim an HQMR segmentation gain or continue prompt/crop searches on these data. The preregistered next route is a pathology visual foundation representation with an image-level learned semantic head in a separate experiment; PDSR-v1, VPCA-v1 and CCRA remain unchanged. This route decision applies to the tested READY sources; it does not assign NOGO to unavailable CONCH, CPLIP or CPath-CLIP, which were never scored.

## Artifact index

- `manifests/`: official availability, frozen prompt/split/crop protocol and GT-unseal evidence.
- `source_cache/`: normalized prototypes, per-view embeddings/scores and integrity hashes.
- `dev/` and `holdout/`: component-level scores, margins, controls, per-class summaries.
- `independence/`: source agreement, error phi, rescue overlap and theoretical pair oracle.
- `bootstrap/`: patient-cluster 95% confidence intervals.
- `visualizations/`: fixed representative Q-DEV case panels.
- `decision.json`: machine-readable preregistered GO/NOGO outcomes.
