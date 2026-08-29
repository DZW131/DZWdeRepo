# RDDR Phase-1 Spatial-Semantic Dross Disposal Report

## 1. Frozen provenance and commands

- Implementation commit: `bacd3dc11797271acc2173964b0e6af846f92929`
- Pure A0 commit: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`
- C0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- UC checkpoint SHA256: `57b19ff24090184cb5e8a194220fe3b1678647568ca189fff2a34d06902b6ceb`
- DD checkpoint SHA256: `fde43993c653c34ad676747c6d4cd993f5f54f0a2d7c77cdebf9c3ba63671c38`
- Locked JSD/DDA source SHA256: `32f5b1152010837359ab8fc0ced1fba7327ef0f9da8b3eeb2f07bd2c48b38aaa`
- Dataset/split: BCSS validation only; no test or LUAD access.
- Provenance note: UC started from the immediately preceding candidate; the only later correction was inside the DD-only JSD helper, so the executed UC forward/backward graph is identical to the locked implementation. DD was started only after the exact Phase-0 JSD expression was synchronized and re-smoked.

```bash
bash tools/run_rddr_phase1_server.sh /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params /home/duyanhong/reseg-data/raw/BCSS-WSSS/training /home/duyanhong/miniconda3/envs/sshr5090/bin/python
tools/analyze_rddr_phase1.py --c0-checkpoint /home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth --uc-dir /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/UC --dd-dir /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/DD --phase0-dir /home/duyanhong/experiments/RDDR_PHASE0_586f402/formal --val-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/val --smoke-json /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/diagnostics/rddr_phase1_smoke_bacd3dc.json --pretrained /home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params --train-root /home/duyanhong/reseg-data/raw/BCSS-WSSS/training --python-executable /home/duyanhong/miniconda3/envs/sshr5090/bin/python --output-dir /home/duyanhong/experiments/RDDR_PHASE1_4e08c9d/report --num-workers 4 --bootstrap-resamples 10000
```

## 2. Architecture and identity contract

Only the HFRM28_1 input changes. UC uses `F_clean=F-DDA(F)`; DD uses `F_clean=F-q*DDA(F)` with detached normalized JSD q. HFRM56, HFRM28_2, heads, loss, optimizer, inference, and metric are unchanged.
Initial C0/DD FP32 maximum absolute difference: `0`.

| Variant | Total parameters | Added parameters | Added MACs@28×28 | Added conv FLOPs@28×28 |
|---|---:|---:|---:|---:|
| C0 | 112709714 | 0 | 0 | 0 |
| UC | 112842706 | 132992 | 103663616 | 207327232 |
| DD | 112842706 | 132992 | 103663616 + analytical JSD | 207327232 + analytical JSD |

## 3. Training protocol equivalence

UC and DD use seed42, batch20, BF16, epoch0→25, released augmentation, loss 0.10/0.15/0.25/0.50, released PolyOptimizer and FINAL checkpoint selection. Both use the same DDA initialization and scratch-LR groups. No validation or test metric influenced training or checkpoint selection.

## 4. Overall metrics and CAM hierarchy

| Variant | CAM56 mIoU | CAM28_1 mIoU | CAM28_2 mIoU | CAMdeep mIoU | Final mIoU | Final mDice |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 61.4515 | 67.0106 | 66.4836 | 64.9483 | 67.3104 | 80.2564 |
| UC | 61.4494 | 66.8625 | 66.2826 | 64.8823 | 67.1569 | 80.1397 |
| DD | 61.3251 | 66.9777 | 66.3270 | 64.8427 | 67.2081 | 80.1769 |

## 5. Boundary, interior, and object size

| Variant | Boundary acc | Boundary mIoU | Interior acc | Interior mIoU | Small recall/mIoU | Medium recall/mIoU | Large recall/mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 51.5534 | 31.7186 | 85.6590 | 71.4795 | 35.0715/17.9615 | 68.3358/45.8841 | 89.4285/78.5755 |
| UC | 51.4695 | 31.5766 | 85.6474 | 71.3155 | 34.7128/17.9975 | 68.5542/45.8130 | 89.3115/78.3581 |
| DD | 51.5990 | 31.9842 | 85.6375 | 71.3254 | 35.7580/18.3243 | 68.6614/46.5467 | 89.2612/77.9687 |

## 6. Per-class IoU

| Variant | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---:|---:|---:|---:|
| C0 | 76.4034 | 70.5463 | 57.8268 | 64.4652 |
| UC | 76.5732 | 70.4510 | 57.8415 | 63.7617 |
| DD | 76.6003 | 70.4386 | 57.8111 | 63.9826 |

## 7. q dynamics, disposal, and feature preservation

| Source | Epoch | Mean | Std | Min | p05 | p25 | p50 | p75 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase0-C0 | 0 | 0.192099 | 0.206927 | 0.000000 | 0.002367 | 0.033056 | 0.115840 | 0.283482 | 0.651628 | 0.999801 |
| DD | 1 | 0.328794 | 0.141606 | 0.000064 | 0.097272 | 0.264186 | 0.322489 | 0.370905 | 0.651642 | 0.957629 |
| DD | 5 | 0.201243 | 0.219697 | 0.000000 | 0.007639 | 0.046728 | 0.120846 | 0.264012 | 0.757411 | 0.999290 |
| DD | 10 | 0.204167 | 0.217057 | 0.000000 | 0.006921 | 0.048268 | 0.126847 | 0.274284 | 0.740190 | 0.998972 |
| DD | 15 | 0.209221 | 0.234235 | 0.000000 | 0.005429 | 0.043500 | 0.121273 | 0.276480 | 0.798546 | 0.999740 |
| DD | 20 | 0.206919 | 0.233196 | 0.000000 | 0.003955 | 0.039354 | 0.118877 | 0.280349 | 0.789096 | 0.999911 |
| DD | 25 | 0.206797 | 0.240753 | 0.000000 | 0.002583 | 0.032111 | 0.111486 | 0.285042 | 0.799433 | 0.999941 |

At Epoch25 DD: q=0.206796±0.240754; RMS(ΔF)/RMS(F)=0.134886.

| Variant | RMS(F) | RMS(D(F)) | RMS(ΔF) | RMS(ΔF)/RMS(F) |
|---|---:|---:|---:|---:|
| UC | 0.275827 | 0.043544 | 0.043544 | 0.157865 |
| DD | 0.256930 | 0.109200 | 0.034656 | 0.134886 |

| Variant/bin | q mean | ΔF pixel RMS | cos(Fclean,F) | norm ratio |
|---|---:|---:|---:|---:|
| UC/AllPixels | 1.000000 | 0.043544 | 0.979266 | 0.997054 |
| DD/Bottom20 | 0.008552 | 0.001178 | 0.999989 | 0.999294 |
| DD/20-40 | 0.045094 | 0.005178 | 0.999690 | 0.994692 |
| DD/40-60 | 0.113966 | 0.012762 | 0.997556 | 0.986986 |
| DD/60-80 | 0.243056 | 0.027167 | 0.987621 | 0.978698 |
| DD/Top20 | 0.625570 | 0.071380 | 0.933740 | 1.003146 |

## 8. Frozen Phase-0 strata and CH transition re-audit

| Variant | Top20 net repair | Bottom80 net repair |
|---|---:|---:|
| UC | -0.0634 pp | -0.0060 pp |
| DD | +0.0586 pp | -0.0347 pp |

C0-defined Corrected-by-CH / Still-Wrong / Harmed-by-CH / Stable-Correct groups are never redefined using UC or DD.

| Variant/group | Repair | Harm | Net accuracy change |
|---|---:|---:|---:|
| UC/Corrected_by_CH | 1.0073 pp | 0.7837 pp | +0.2236 pp |
| UC/Still_Wrong | 1.1883 pp | 1.5305 pp | -0.3422 pp |
| UC/Harmed_by_CH | 1.8245 pp | 2.6848 pp | -0.8603 pp |
| UC/Stable_Correct | 0.3455 pp | 0.3009 pp | +0.0445 pp |
| DD/Corrected_by_CH | 0.9159 pp | 0.6362 pp | +0.2797 pp |
| DD/Still_Wrong | 1.3956 pp | 1.3750 pp | +0.0207 pp |
| DD/Harmed_by_CH | 1.6024 pp | 2.2236 pp | -0.6213 pp |
| DD/Stable_Correct | 0.2253 pp | 0.2786 pp | -0.0533 pp |

## 9. UC versus DD and paired bootstrap

| Variant | Correction RMS ratio | Top20 repair | Bottom80 harm | CAM28_1 mIoU | Final mIoU |
|---|---:|---:|---:|---:|---:|
| UC | 0.157865 | 1.0331 pp | 0.4964 pp | 66.8625 | 67.1569 |
| DD | 0.134886 | 1.1177 pp | 0.4173 pp | 66.9777 | 67.2081 |

| Comparison | Observed ΔmIoU | Bootstrap mean | 95% CI |
|---|---:|---:|---:|
| DD-C0 | -0.1022 pp | -0.1028 pp | [-0.3971, +0.1700] pp |
| DD-UC | +0.0513 pp | +0.0496 pp | [-0.2788, +0.3785] pp |
| UC-C0 | -0.1535 pp | -0.1524 pp | [-0.4022, +0.0826] pp |

## 10. Preregistered gates

| Gate | Requirement | Result | Pass |
|---|---|---|:---:|
| A | DD mIoU > C0 and DD-C0 CI low >= 0 | delta=-0.001022, low=-0.003971 | False |
| B | DD > UC with nonnegative CI low or CAM28_1+Top20 fallback | delta=+0.000513, low=-0.002788, fallback=True | True |
| C | DD CAM28_1 >= C0 and interior accuracy delta >= -0.10 pp | CAMdelta=-0.000329, interior=-0.000215 | False |
| D | DD Top20 net > 0 and > DD Bottom80 net | top=+0.000586, bottom=-0.000347 | True |

## 11. Scientific interpretation

The frozen subtractive-disposal hypothesis does not establish the full causal utility chain. Failed gates: A, C. No post-hoc change to q, temperature, adapter depth, kernel, reduction, or checkpoint selection is permitted.

## 12. Engineering and artifact record

- Analysis runtime: 3.20 min; peak CUDA memory 3.141 GiB.
- UC/DD training runtime: 45.45 / 45.80 min.
- Required optimizer, training-curve, q, disposal, fixed-strata, CH, per-class, bootstrap, and summary artifacts were generated.
- No BCSS test, LUAD, best-epoch selection, or post-hoc parameter tuning was used.

DECISION = DROSS_DISPOSAL_SEMANTIC_DAMAGE
