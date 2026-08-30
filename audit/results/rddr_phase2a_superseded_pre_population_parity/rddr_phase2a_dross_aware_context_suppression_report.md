# RDDR Phase-2A Dross-Aware Context Suppression Report

## 1. Frozen provenance and commands

- Implementation commit: `6f45ac7676b2e7bd7ae21c23db3303de95e02c6c`
- Evaluation commit: `123a19ad7385a1084f7248a6149ec8f9a58026f0`
- Pure A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- C0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- GS checkpoint SHA256: `f748b4290e6cb6eedc6e3372b313bd1f977d41ce48c8fc048949d250f82a7031`
- RCS checkpoint SHA256: `2f9960bfa5bdd61a560e8a60ecd1af139ebbe7594dd1c7b6954bdd034fd333ac`
- Locked JSD helper SHA256: `1142ff8e8f95d3447012af9c4eb8f91eb923a48d5e8f840ea42098cc2f1de58b`
- Locked model source SHA256: `a6f6cf3a82c23d5a7a99c41c6f1348c118428aa6a508ee0dc71d7f44ac9f1f3d`
- Dataset/split: 3418 BCSS validation images; no test or LUAD access.

```bash
bash tools/run_rddr_phase2a_server.sh /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7 /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params /home/duyanhong/reseg-data/raw/BCSS-WSSS/training /home/duyanhong/miniconda3/envs/sshr5090/bin/python
/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/analyze_rddr_phase2a.py --c0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --gs-dir /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/GS --rcs-dir /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/RCS --phase0-dir /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal --phase1-dir /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/report --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --smoke-json /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/diagnostics/rddr_phase2a_smoke.json --pretrained /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --python-executable /home/duyanhong/miniconda3/envs/sshr5090/bin/python --output-dir /home/duyanhong/experiments/RDDR_PHASE2A_6f45ac7/report --num-workers 4 --max-images 0 --bootstrap-resamples 10000
```

## 2. Architecture, capacity, and semantic-preservation contract

Only the HFRM28_1 context residual is scaled. The original feature F and semantic veto residual remain untouched.

```text
C0: F' = F + gamma_sem*R_sem + gamma_ctx*R_ctx
GS: F' = F + gamma_sem*R_sem + gamma_ctx*mean(1-q)*R_ctx
RCS: F' = F + gamma_sem*R_sem + gamma_ctx*(1-q_i)*R_ctx
```

- Total parameters (all variants): 112709714
- Additional trainable parameters: 0
- Initial zero-gamma max absolute difference: 0
- Same-checkpoint pre-HFRM feature max difference: 0
- Same-checkpoint pre-HFRM feature cosine: 1.000000000

## 3. Training equivalence

GS and RCS use seed42, batch20, BF16, epoch0→25, official pretrained weights, released augmentation, loss 0.10/0.15/0.25/0.50, released PolyOptimizer/LR schedule, and Epoch-25 FINAL checkpoints. Training never evaluated validation or test.

## 4. Overall metrics and CAM hierarchy

Official three-view TTA is averaged in the native output dtype before FP32 normalization. BCSS presence thresholds are [0.8,0.9,0.8,0.6], with argmax fallback when none pass; final fusion is 0.6/0.2/0.2 (CAM56 diagnostic only). The initial copied audit helper averaged after FP32 conversion; this was corrected before this evaluation. Original infer()/metric/model files were not changed.

Direct pixel parity against unchanged official infer(): {'C0': {'images': 8, 'mismatched_prediction_pixels': 0}, 'GS': {'images': 8, 'mismatched_prediction_pixels': 0}, 'RCS': {'images': 8, 'mismatched_prediction_pixels': 0}}

Metric retains official GT-background overwrite; foreground classes 0–3 enter the mean. Absent-class IoU is NaN/excluded; absent-class Dice is 0. Boundary masks include foreground-to-foreground transitions only. Size bins use per-class 8-connected GT-component area q25/q75; recall is pixel-weighted and size mIoU is mask-restricted, not instance IoU.

| Variant | CAM56 mIoU | CAM28_1 mIoU | CAM28_2 mIoU | CAMdeep mIoU | Final mIoU | Final mDice |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.4553 | 67.0172 | 66.4908 | 64.9535 | 67.3173 | 80.2610 |
| GS | 61.3792 | 66.7806 | 66.2841 | 64.7484 | 67.0915 | 80.0873 |
| RCS | 61.3915 | 66.5150 | 66.3321 | 64.8004 | 67.0663 | 80.0723 |

## 5. Boundary, interior, and object size

| Variant | Boundary acc | Boundary mIoU | Interior acc | Interior mIoU | Small recall/mIoU | Medium recall/mIoU | Large recall/mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.5589 | 31.7224 | 85.6669 | 71.4869 | 35.0712/17.9405 | 68.3625/45.9002 | 89.4281/78.5756 |
| GS | 51.6517 | 31.5524 | 85.6207 | 71.2416 | 35.3613/18.1635 | 68.2575/45.5505 | 89.4213/78.4090 |
| RCS | 51.7163 | 31.5963 | 85.5577 | 71.2058 | 35.7642/18.3942 | 68.2046/45.5533 | 89.3663/78.3904 |

## 6. Per-class IoU

| Variant | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.4212 | 70.5567 | 57.8264 | 64.4650 |
| GS | 76.4236 | 70.6037 | 57.4814 | 63.8574 |
| RCS | 76.3107 | 70.5144 | 57.4570 | 63.9832 |

## 7. q dynamics

q is JS/ln(2), computed at 28x28; these dynamics include all grid positions. Phase1-DD rows are imported observations, not re-trained models.

| Source | Epoch | Mean | Std | Min | p05 | p25 | p50 | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reconstructed-init | 0 | 0.068370 | 0.053279 | 0.000001 | 0.010736 | 0.031040 | 0.055002 | 0.090555 | 0.172149 | 0.601937 |
| Phase0-C0 | 25 | 0.192099 | 0.206927 | 0.000000 | 0.002367 | 0.033056 | 0.115840 | 0.283482 | 0.651628 | 0.999801 |
| GS | 1 | 0.324066 | 0.150833 | 0.000008 | 0.091033 | 0.252536 | 0.311371 | 0.365049 | 0.685675 | 0.971978 |
| GS | 5 | 0.178140 | 0.182068 | 0.000000 | 0.004911 | 0.038916 | 0.114280 | 0.259966 | 0.572782 | 0.998001 |
| GS | 10 | 0.182054 | 0.182386 | 0.000000 | 0.004227 | 0.040675 | 0.121549 | 0.268204 | 0.570506 | 0.998671 |
| GS | 15 | 0.179674 | 0.179882 | 0.000000 | 0.003575 | 0.038459 | 0.120371 | 0.268552 | 0.555338 | 0.997815 |
| GS | 20 | 0.180423 | 0.186824 | 0.000000 | 0.002270 | 0.033520 | 0.116639 | 0.271370 | 0.573719 | 0.998571 |
| GS | 25 | 0.181434 | 0.196241 | 0.000000 | 0.001417 | 0.027441 | 0.110043 | 0.274935 | 0.600073 | 0.999372 |
| RCS | 1 | 0.323979 | 0.150920 | 0.000015 | 0.090637 | 0.252431 | 0.311277 | 0.365047 | 0.686228 | 0.971887 |
| RCS | 5 | 0.178860 | 0.182069 | 0.000000 | 0.005067 | 0.039490 | 0.115335 | 0.260765 | 0.573389 | 0.997839 |
| RCS | 10 | 0.182147 | 0.182038 | 0.000000 | 0.004299 | 0.040948 | 0.122052 | 0.268169 | 0.569482 | 0.998559 |
| RCS | 15 | 0.179122 | 0.180201 | 0.000000 | 0.003528 | 0.038097 | 0.119193 | 0.267394 | 0.556267 | 0.997984 |
| RCS | 20 | 0.179906 | 0.187583 | 0.000000 | 0.002214 | 0.032868 | 0.115215 | 0.270251 | 0.575939 | 0.998543 |
| RCS | 25 | 0.180756 | 0.197188 | 0.000000 | 0.001367 | 0.026680 | 0.108230 | 0.273330 | 0.603208 | 0.999382 |
| Phase1-DD | 1 | 0.328794 | 0.141606 | 0.000064 | 0.097272 | 0.264186 | 0.322489 | 0.370905 | 0.651642 | 0.957629 |
| Phase1-DD | 5 | 0.201243 | 0.219697 | 0.000000 | 0.007639 | 0.046728 | 0.120846 | 0.264012 | 0.757411 | 0.999290 |
| Phase1-DD | 10 | 0.204167 | 0.217057 | 0.000000 | 0.006921 | 0.048268 | 0.126847 | 0.274284 | 0.740190 | 0.998972 |
| Phase1-DD | 15 | 0.209221 | 0.234235 | 0.000000 | 0.005429 | 0.043500 | 0.121273 | 0.276480 | 0.798546 | 0.999740 |
| Phase1-DD | 20 | 0.206919 | 0.233196 | 0.000000 | 0.003955 | 0.039354 | 0.118877 | 0.280349 | 0.789096 | 0.999911 |
| Phase1-DD | 25 | 0.206797 | 0.240753 | 0.000000 | 0.002583 | 0.032111 | 0.111486 | 0.285042 | 0.799433 | 0.999941 |

## 8. Effective context strength

| Variant | Mean reliability | Mean suppression | r p05/p25/p50/p75/p95 | Context RMS before | after | ratio |
|---|---:|---:|---|---:|---:|---:|
| GS | 0.818565 | 0.181435 | 0.721395/0.778159/0.817139/0.861931/0.916662 | 0.447175 | 0.372074 | 0.832055 |
| RCS | 0.819245 | 0.180755 | 0.396745/0.726679/0.891771/0.973317/0.998632 | 0.440834 | 0.379384 | 0.860607 |

## 9. gamma dynamics and compensation

| Variant | Epoch | gamma_context | gamma_veto | Mean r | EffectiveContextScale |
|---|---:|---:|---:|---:|---:|
| Phase0-C0 | 25 | +1.572197 | +0.567998 | 1.000000 | 1.572197 |
| GS | 1 | +0.203889 | +0.165168 | 0.675934 | 0.137815 |
| GS | 5 | +1.354367 | +0.708207 | 0.821860 | 1.113100 |
| GS | 10 | +1.547089 | +0.746523 | 0.817945 | 1.265434 |
| GS | 15 | +1.561313 | +0.733230 | 0.820326 | 1.280786 |
| GS | 20 | +1.561190 | +0.724867 | 0.819577 | 1.279516 |
| GS | 25 | +1.553347 | +0.720936 | 0.818566 | 1.271517 |
| RCS | 1 | +0.203044 | +0.164931 | 0.676021 | 0.137262 |
| RCS | 5 | +1.339128 | +0.711007 | 0.821140 | 1.099612 |
| RCS | 10 | +1.529943 | +0.761328 | 0.817853 | 1.251269 |
| RCS | 15 | +1.541970 | +0.753959 | 0.820878 | 1.265769 |
| RCS | 20 | +1.540607 | +0.748476 | 0.820094 | 1.263443 |
| RCS | 25 | +1.532349 | +0.745409 | 0.819245 | 1.255369 |

## 10. Frozen Phase-0 Top20 / Bottom80

| Variant | Top20 repair/harm/net | Bottom80 repair/harm/net |
|---|---:|---:|
| GS | 0.6672/0.7953/-0.1281 pp | 0.2888/0.3005/-0.0117 pp |
| RCS | 0.9816/1.2430/-0.2614 pp | 0.3344/0.3784/-0.0440 pp |

## 11. Frozen C0 CH-transition groups

| Variant/group | Repair | Harm | Net change |
|---|---:|---:|---:|
| GS/Corrected_by_CH | 0.2769 pp | 1.0909 pp | -0.8139 pp |
| GS/Still_Wrong | 0.5387 pp | 1.2033 pp | -0.6647 pp |
| GS/Harmed_by_CH | 2.7324 pp | 0.4531 pp | +2.2793 pp |
| GS/Stable_Correct | 0.2463 pp | 0.0947 pp | +0.1517 pp |
| RCS/Corrected_by_CH | 0.2981 pp | 1.7197 pp | -1.4216 pp |
| RCS/Still_Wrong | 0.8275 pp | 1.3084 pp | -0.4809 pp |
| RCS/Harmed_by_CH | 4.2102 pp | 0.6198 pp | +3.5904 pp |
| RCS/Stable_Correct | 0.2623 pp | 0.1698 pp | +0.0925 pp |

## 12. Frozen-C0 q-quintile selectivity

All bins are defined from the locked C0, never from candidate q. Prediction bins use full-resolution foreground q; context bins use 28x28 C0 q with nearest-resized foreground masks and separately computed quintiles. They are resolution-specific populations, not identical pixels. Exact thresholds and counts are in the JSON/CSV.

| Variant | Quintile | Mean r | Context RMS before | after | ratio | Accuracy delta vs C0 |
|---|---|---:|---:|---:|---:|---:|
| GS | Q1 | 0.849485 | 0.475054 | 0.408845 | 0.860629 | -0.0036 pp |
| GS | Q2 | 0.835342 | 0.466183 | 0.394991 | 0.847286 | -0.0055 pp |
| GS | Q3 | 0.823243 | 0.452759 | 0.377832 | 0.834510 | -0.0066 pp |
| GS | Q4 | 0.808772 | 0.437219 | 0.358194 | 0.819256 | -0.0311 pp |
| GS | Q5 | 0.783502 | 0.414879 | 0.330222 | 0.795947 | -0.1281 pp |
| RCS | Q1 | 0.992710 | 0.471437 | 0.468181 | 0.993094 | -0.0179 pp |
| RCS | Q2 | 0.956284 | 0.459450 | 0.441311 | 0.960519 | -0.0259 pp |
| RCS | Q3 | 0.884898 | 0.445233 | 0.397704 | 0.893251 | -0.0343 pp |
| RCS | Q4 | 0.758130 | 0.429619 | 0.332015 | 0.772812 | -0.0980 pp |
| RCS | Q5 | 0.517577 | 0.407849 | 0.224868 | 0.551350 | -0.2613 pp |

## 13. Paired image-level bootstrap

| Comparison | Observed delta mIoU | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| RCS-C0 | -0.2510 pp | -0.2515 pp | [-0.4138, -0.1161] pp |
| RCS-GS | -0.0252 pp | -0.0256 pp | [-0.1131, +0.0789] pp |
| GS-C0 | -0.2258 pp | -0.2259 pp | [-0.4079, -0.0699] pp |

## 14. Preregistered gates

| Gate | Requirement | Result | Pass |
|---|---|---|:---:|
| A | RCS mIoU > C0 and RCS-C0 CI low >= 0 | delta=-0.002510, low=-0.004138 | False |
| B | RCS > GS with nonnegative CI low or CAM28_1+Top20 fallback | delta=-0.000252, low=-0.001131, fallback=False | False |
| C | RCS CAM28_1 >= C0, interior >= -0.10 pp, large mIoU >= -0.20 pp | CAM=-0.005022, interior=-0.001092, large=-0.001852 | False |
| D | RCS Harmed-by-CH > 0 and > GS; Stable-Correct >= -0.10 pp | RCS_harmed=+0.035904, GS_harmed=+0.022793, stable=+0.000925 | True |

## 15. Scientific interpretation

The semantic-safety gate fails: even receiver-only context suppression damages CAM28_1, interior, or large-region behavior under the frozen thresholds. Failed gates: A, B, C. No post-hoc transformation or tuning is permitted.

## 16. Engineering and artifact record

- Main final-checkpoint evaluation: 2.88 min; complete evaluation including dynamics/bootstrap: 4.81 min; peak CUDA memory 3.155 GiB.
- GS/RCS training runtime: 45.21 / 45.28 min.
- All required curves, q/context/gamma, fixed-strata, CH, quintile, per-class, bootstrap, optimizer, per-image, and summary artifacts were generated.
- No BCSS test, LUAD, best-epoch selection, or post-hoc tuning was used.

## 17. Epoch0 initialization observation

Retrospective seed42/pretrained reconstruction, eval mode, validation images, batch20 BF16; zero training steps

This reconstructs initialization after training has finished; it is not a contemporaneous training log. Shared raw features and q are computed once, then the frozen GS/RCS context scaling is applied. Initial gammas are zero, so attenuated context does not yet contribute to the output.

| Variant | Mean r | Mean suppression | Context RMS before | after | ratio |
|---|---:|---:|---:|---:|---:|
| GS | 0.931630 | 0.068370 | 0.030606 | 0.028517 | 0.931764 |
| RCS | 0.931630 | 0.068370 | 0.030606 | 0.028479 | 0.930509 |

DECISION = CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE
