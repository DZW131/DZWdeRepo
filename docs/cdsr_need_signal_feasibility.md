# CDSR Need-Signal Feasibility Audit

## 1. Frozen protocol

This is a zero-training Phase-0 audit. It uses BCSS validation only,
the A0 seed-42 final checkpoint, no test data, no retraining, and no
formula or weight search. Raw and post-HFRM logits reuse the same
trained stage CAM heads. Risk math is detached FP32.

- validation samples: 3418
- checkpoint: `/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth`
- checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`
- checkpoint loading: strict state-dict match (no missing or unexpected keys)
- model path: official A0 HFRM with CH context (`context_mode=ch`)
- PyTorch: `2.11.0+cu128`
- device: `NVIDIA GeForce RTX 5090 D v2`
- network forward precision: `bf16`
- analysis pixels: native-stage pixels whose GT is foreground 0-3;
  background 4 is excluded because the CAMs have four foreground
  channels and the official SSHR metric excludes/overwrites background
- GT is resized to each native stage by nearest-neighbor interpolation

## 2. Preregistered need signal

```text
D = JSD(P_stage, P_deep) / ln(2)
U = entropy(P_stage) / ln(C)
R = 1 - entropy(P_deep) / ln(C)
N = R * (1 - (1-D) * (1-U))
```

No class-presence labels, GT class IDs, thresholds, temperatures, or
manual D/U weights are used in the signal.

## 3. Overall signal ranking

| Stage | pixels | raw-error AUROC | raw-error AUPR | corrected/harmed AUROC | corrected/harmed AUPR | corrected-harmed Cohen d |
|---|---:|---:|---:|---:|---:|---:|
| F56 | 9,915,516 | 0.6811 | 0.6239 | 0.6414 | 0.8407 | 0.4732 |
| F28_1 | 2,479,143 | 0.7077 | 0.5136 | 0.5684 | 0.8658 | 0.1634 |
| F28_2 | 2,479,143 | 0.6571 | 0.3307 | 0.4921 | 0.7200 | -0.0183 |

## 4. Need distributions

| Stage | group | count | mean | median | std |
|---|---|---:|---:|---:|---:|
| F56 | raw wrong | 4,069,438 | 0.7024 | 0.7988 | 0.2405 |
| F56 | raw correct | 5,846,078 | 0.5611 | 0.5798 | 0.2337 |
| F56 | corrected by HFRM | 2,107,395 | 0.7587 | 0.8397 | 0.2107 |
| F56 | harmed by HFRM | 616,729 | 0.6565 | 0.7145 | 0.2329 |
| F28_1 | raw wrong | 708,422 | 0.6239 | 0.6898 | 0.2700 |
| F28_1 | raw correct | 1,770,721 | 0.4148 | 0.3934 | 0.2793 |
| F28_1 | corrected by HFRM | 325,670 | 0.7146 | 0.8015 | 0.2349 |
| F28_1 | harmed by HFRM | 68,931 | 0.6766 | 0.7445 | 0.2219 |
| F28_2 | raw wrong | 567,865 | 0.4279 | 0.4230 | 0.2296 |
| F28_2 | raw correct | 1,911,278 | 0.2956 | 0.2595 | 0.2443 |
| F28_2 | corrected by HFRM | 161,624 | 0.4937 | 0.4780 | 0.2001 |
| F28_2 | harmed by HFRM | 63,921 | 0.4973 | 0.4903 | 0.1926 |

## 5. Quartile analysis

The preregistered interpretation of 'clearly higher' is an
image-paired bootstrap 95% CI whose lower bound is above zero.

| Stage | bottom raw error | top raw error | bottom net correction | top net correction | top-bottom net gap | paired-image 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| F56 | 0.2830 | 0.6878 | 0.0458 | 0.3674 | 0.3216 | [0.2354, 0.2505] |
| F28_1 | 0.1142 | 0.5101 | 0.0169 | 0.2569 | 0.2400 | [0.1931, 0.2056] |
| F28_2 | 0.0863 | 0.3282 | 0.0016 | 0.0669 | 0.0653 | [0.0568, 0.0655] |

## 6. Per-class analysis

Per-class results are analysis-only and do not alter the signal.

| Stage | GT class | pixels | raw AUROC | corrected/harmed AUROC | corrected-harmed Cohen d |
|---|---|---:|---:|---:|---:|
| F56 | 0 Tumor | 3,860,914 | 0.6597 | 0.5800 | 0.2375 |
| F56 | 1 Stroma | 4,178,186 | 0.6530 | 0.6054 | 0.4030 |
| F56 | 2 Inflammatory | 1,294,215 | 0.6857 | 0.6420 | 0.4561 |
| F56 | 3 Necrosis | 582,201 | 0.4604 | 0.5784 | 0.5664 |
| F28_1 | 0 Tumor | 965,133 | 0.7347 | 0.4915 | -0.0569 |
| F28_1 | 1 Stroma | 1,044,692 | 0.6600 | 0.4984 | 0.0270 |
| F28_1 | 2 Inflammatory | 323,515 | 0.6763 | 0.5601 | 0.0874 |
| F28_1 | 3 Necrosis | 145,803 | 0.5181 | 0.7334 | 0.7362 |
| F28_2 | 0 Tumor | 965,133 | 0.6612 | 0.3449 | -0.5370 |
| F28_2 | 1 Stroma | 1,044,692 | 0.6906 | 0.6290 | 0.4563 |
| F28_2 | 2 Inflammatory | 323,515 | 0.5355 | 0.3894 | -0.3652 |
| F28_2 | 3 Necrosis | 145,803 | 0.5217 | 0.2925 | -0.7194 |

## 7. Go / No-Go evaluation

| Stage | passed conditions | stage go | strong go | major reverse |
|---|---:|---|---|---|
| F56 | 4/4 | True | True | False |
| F28_1 | 3/4 | True | False | False |
| F28_2 | 2/4 | True | False | False |

- Go stages: 3/3
- Strong-Go stages: 1/3
- Major reverse stages: 0/3
- Final decision: **CDSR_SIGNAL_GO**

## 8. Interpretation

This is a Go, not a Strong Go. F56 provides the clearest
corrected-versus-harmed signal; F28_1 is weaker, and F28_2
passes through raw-error ranking plus quartile net correction
rather than direct corrected-versus-harmed discrimination.
The per-class reversals shown above are a material limitation
and must not be hidden or used to retune the frozen formula.

## 9. Development consequence

The frozen analytical signal passes the preregistered Phase-0
gate. CDSR model engineering is permitted, but no model code or
training is included in this Phase-0 branch.

CDSR_SIGNAL_GO
