# Phase2B1.12 Short-Horizon ADT Optimization Dynamics Audit

All metric values are fractions unless a column explicitly says pp (percentage points). Validation supports only the preregistered step500 endpoint; no checkpoint selection is performed.

## 1. Provenance

Status: FINAL. Approved pure A0 base: `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`. Locked C0 checkpoint SHA256: `509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`. These identifiers are the preregistered references, not a substitute for the supplied provenance/verification evidence. No training or model selection is performed by this analysis.

```json
{
  "provenance_status": "PASS",
  "provenance_errors": [],
  "source_hashes": {
    "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/environment.tsv": "66d701e2d55dfb9aafd7dc8929605a1a6be13ca5697a58d2295a9724f5798482",
    "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/status.tsv": "904b605bc5613015631ae1baab704c40b481fb394469df5321c973d1b7a705d4",
    "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/train.log": "21ecfb1b9d7dfb2d4828c83d1dea3eb96fe042a9e5ba7dddbc8ef7356aead44e",
    "/home/duyanhong/run_official_25ep_retry2.sh": "0dbf5707f623c50565a08bda01ba46d9123ab81d304d848c5f6fbee691cb3e7b",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/train_sshr.py": "178bf0fcb6185aa0ee07708ad79a1568226d3b116d597437aea871d272e5b03a",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/torchutils.py": "cce24c7096d6b02688dc3b3e0c60edbca65ad9cc8bc643e8624e784fbc67ac1b",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/network/resnet38d.py": "77b699d2db5321393320de5a59382ac07e87ecb999797a914dda1f38eebb73fa",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/network/resnet38_cls.py": "570c9cffdeb0beb22eadd05a748c3917e48a26d48fdf1b6b4563b529d774b0ec",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/GenDataset.py": "073c9b4ffe10aab71bd8fbcb63e0703579c587d8e60cef7518e9d1c9a6dde969",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/infer_fun.py": "0d84fa70507199faaf10eaf55e838a7a4bc2a85ae44d6183436dc9d1e64cd520",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/iouutils.py": "7614fd9b75f13c65cfe6121efe1e44fa2b3a0d9a0d441f6985e8c307dc460490"
  },
  "verification_passed": true
}
```

[rddr_phase2b112_verification.json](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_verification.json)

## 2. Optimizer provenance

```json
{
  "resolved": true,
  "state_policy": "fresh_optimizer_state",
  "checkpoint": "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/checkpoints/stage1_last.pth",
  "checkpoint_sha256": "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579",
  "checkpoint_bytes": 451130207,
  "model_state_sha256": "c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5",
  "source_hashes": {
    "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/environment.tsv": "66d701e2d55dfb9aafd7dc8929605a1a6be13ca5697a58d2295a9724f5798482",
    "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/status.tsv": "904b605bc5613015631ae1baab704c40b481fb394469df5321c973d1b7a705d4",
    "/home/duyanhong/sshr-official-25ep-final-retry2-20260815/runs/bcss_seed42/train.log": "21ecfb1b9d7dfb2d4828c83d1dea3eb96fe042a9e5ba7dddbc8ef7356aead44e",
    "/home/duyanhong/run_official_25ep_retry2.sh": "0dbf5707f623c50565a08bda01ba46d9123ab81d304d848c5f6fbee691cb3e7b",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/train_sshr.py": "178bf0fcb6185aa0ee07708ad79a1568226d3b116d597437aea871d272e5b03a",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/torchutils.py": "cce24c7096d6b02688dc3b3e0c60edbca65ad9cc8bc643e8624e784fbc67ac1b",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/network/resnet38d.py": "77b699d2db5321393320de5a59382ac07e87ecb999797a914dda1f38eebb73fa",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/network/resnet38_cls.py": "570c9cffdeb0beb22eadd05a748c3917e48a26d48fdf1b6b4563b529d774b0ec",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/GenDataset.py": "073c9b4ffe10aab71bd8fbcb63e0703579c587d8e60cef7518e9d1c9a6dde969",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/infer_fun.py": "0d84fa70507199faaf10eaf55e838a7a4bc2a85ae44d6183436dc9d1e64cd520",
    "/home/duyanhong/DZWdeRepo-rddr-phase2b112/tool/iouutils.py": "7614fd9b75f13c65cfe6121efe1e44fa2b3a0d9a0d441f6985e8c307dc460490"
  },
  "original_environment": {
    "started_at": "2026-08-15T21:11:43+08:00",
    "repo_commit": "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9",
    "python": "/home/duyanhong/miniconda3/envs/sshr5090/bin/python",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "gpu": "NVIDIA GeForce RTX 5090 D v2, 595.71.05",
    "weights": "/home/duyanhong/sshr-reproduction/SSHR/init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params",
    "weights_sha256": "f668a2add80e33dfa8f1a0695df91f6d8cfad5ffbb26d1dc7bcd35903a1f6e16",
    "queue_finished_at": "2026-08-16T01:07:58+08:00"
  },
  "global_step": 29275,
  "max_step": 29275,
  "last_applied_step": 29274,
  "lr_power": 0.9,
  "groups": [
    {
      "index": 0,
      "lr": 9.55328615544644e-07,
      "momentum": 0.0005,
      "weight_decay": 0.0005,
      "dampening": 0,
      "nesterov": false,
      "names": [
        "conv1a.weight",
        "b2.bn_branch2a.weight",
        "b2.conv_branch2a.weight",
        "b2.bn_branch2b1.weight",
        "b2.conv_branch2b1.weight",
        "b2.conv_branch1.weight",
        "b2_1.bn_branch2a.weight",
        "b2_1.conv_branch2a.weight",
        "b2_1.bn_branch2b1.weight",
        "b2_1.conv_branch2b1.weight",
        "b2_2.bn_branch2a.weight",
        "b2_2.conv_branch2a.weight",
        "b2_2.bn_branch2b1.weight",
        "b2_2.conv_branch2b1.weight",
        "b3.bn_branch2a.weight",
        "b3.conv_branch2a.weight",
        "b3.bn_branch2b1.weight",
        "b3.conv_branch2b1.weight",
        "b3.conv_branch1.weight",
        "b3_1.bn_branch2a.weight",
        "b3_1.conv_branch2a.weight",
        "b3_1.bn_branch2b1.weight",
        "b3_1.conv_branch2b1.weight",
        "b3_2.bn_branch2a.weight",
        "b3_2.conv_branch2a.weight",
        "b3_2.bn_branch2b1.weight",
        "b3_2.conv_branch2b1.weight",
        "b4.bn_branch2a.weight",
        "b4.conv_branch2a.weight",
        "b4.bn_branch2b1.weight",
        "b4.conv_branch2b1.weight",
        "b4.conv_branch1.weight",
        "b4_1.bn_branch2a.weight",
        "b4_1.conv_branch2a.weight",
        "b4_1.bn_branch2b1.weight",
        "b4_1.conv_branch2b1.weight",
        "b4_2.bn_branch2a.weight",
        "b4_2.conv_branch2a.weight",
        "b4_2.bn_branch2b1.weight",
        "b4_2.conv_branch2b1.weight",
        "b4_3.bn_branch2a.weight",
        "b4_3.conv_branch2a.weight",
        "b4_3.bn_branch2b1.weight",
        "b4_3.conv_branch2b1.weight",
        "b4_4.bn_branch2a.weight",
        "b4_4.conv_branch2a.weight",
        "b4_4.bn_branch2b1.weight",
        "b4_4.conv_branch2b1.weight",
        "b4_5.bn_branch2a.weight",
        "b4_5.conv_branch2a.weight",
        "b4_5.bn_branch2b1.weight",
        "b4_5.conv_branch2b1.weight",
        "bn45.weight",
        "b5.bn_branch2a.weight",
        "b5.conv_branch2a.weight",
        "b5.bn_branch2b1.weight",
        "b5.conv_branch2b1.weight",
        "b5.conv_branch1.weight",
        "b5_1.bn_branch2a.weight",
        "b5_1.conv_branch2a.weight",
        "b5_1.bn_branch2b1.weight",
        "b5_1.conv_branch2b1.weight",
        "b5_2.bn_branch2a.weight",
        "b5_2.conv_branch2a.weight",
        "b5_2.bn_branch2b1.weight",
        "b5_2.conv_branch2b1.weight",
        "bn52.weight",
        "b6.bn_branch2a.weight",
        "b6.conv_branch2a.weight",
        "b6.bn_branch2b1.weight",
        "b6.conv_branch2b1.weight",
        "b6.bn_branch2b2.weight",
        "b6.conv_branch2b2.weight",
        "b6.conv_branch1.weight",
        "b7.bn_branch2a.weight",
        "b7.conv_branch2a.weight",
        "b7.bn_branch2b1.weight",
        "b7.conv_branch2b1.weight",
        "b7.bn_branch2b2.weight",
        "b7.conv_branch2b2.weight",
        "b7.conv_branch1.weight",
        "bn7.weight"
      ],
      "numel": 105048576
    },
    {
      "index": 1,
      "lr": 1.910657231089288e-06,
      "momentum": 0.0005,
      "weight_decay": 0.0,
      "dampening": 0,
      "nesterov": false,
      "names": [
        "b2.bn_branch2a.bias",
        "b2.bn_branch2b1.bias",
        "b2_1.bn_branch2a.bias",
        "b2_1.bn_branch2b1.bias",
        "b2_2.bn_branch2a.bias",
        "b2_2.bn_branch2b1.bias",
        "b3.bn_branch2a.bias",
        "b3.bn_branch2b1.bias",
        "b3_1.bn_branch2a.bias",
        "b3_1.bn_branch2b1.bias",
        "b3_2.bn_branch2a.bias",
        "b3_2.bn_branch2b1.bias",
        "b4.bn_branch2a.bias",
        "b4.bn_branch2b1.bias",
        "b4_1.bn_branch2a.bias",
        "b4_1.bn_branch2b1.bias",
        "b4_2.bn_branch2a.bias",
        "b4_2.bn_branch2b1.bias",
        "b4_3.bn_branch2a.bias",
        "b4_3.bn_branch2b1.bias",
        "b4_4.bn_branch2a.bias",
        "b4_4.bn_branch2b1.bias",
        "b4_5.bn_branch2a.bias",
        "b4_5.bn_branch2b1.bias",
        "bn45.bias",
        "b5.bn_branch2a.bias",
        "b5.bn_branch2b1.bias",
        "b5_1.bn_branch2a.bias",
        "b5_1.bn_branch2b1.bias",
        "b5_2.bn_branch2a.bias",
        "b5_2.bn_branch2b1.bias",
        "bn52.bias",
        "b6.bn_branch2a.bias",
        "b6.bn_branch2b1.bias",
        "b6.bn_branch2b2.bias",
        "b7.bn_branch2a.bias",
        "b7.bn_branch2b1.bias",
        "b7.bn_branch2b2.bias",
        "bn7.bias"
      ],
      "numel": 25408
    },
    {
      "index": 2,
      "lr": 9.55328615544644e-06,
      "momentum": 0.0005,
      "weight_decay": 0.0005,
      "dampening": 0,
      "nesterov": false,
      "names": [
        "hfrm_56.gamma_veto",
        "hfrm_56.gamma_context",
        "hfrm_56.veto_mlp.0.weight",
        "hfrm_56.veto_mlp.2.weight",
        "hfrm_56.context_conv.weight",
        "hfrm_28_1.gamma_veto",
        "hfrm_28_1.gamma_context",
        "hfrm_28_1.veto_mlp.0.weight",
        "hfrm_28_1.veto_mlp.2.weight",
        "hfrm_28_1.context_conv.weight",
        "hfrm_28_2.gamma_veto",
        "hfrm_28_2.gamma_context",
        "hfrm_28_2.veto_mlp.0.weight",
        "hfrm_28_2.veto_mlp.2.weight",
        "hfrm_28_2.context_conv.weight",
        "ic_56.weight",
        "ic1.weight",
        "ic2.weight",
        "fc8.weight"
      ],
      "numel": 7635718
    },
    {
      "index": 3,
      "lr": 1.910657231089288e-05,
      "momentum": 0.0005,
      "weight_decay": 0.0,
      "dampening": 0,
      "nesterov": false,
      "names": [
        "ic_56.bias",
        "ic1.bias",
        "ic2.bias"
      ],
      "numel": 12
    }
  ],
  "scheduler_rule": "Original PolyOptimizer: after max_step retain last applied LR; no restart",
  "optimizer_buffers_recovered": false,
  "initial_optimizer_states_empty": true,
  "training_samples": 23422,
  "batch_size": 20,
  "num_workers": 4,
  "seed": 42,
  "specification_sha256": "52fdab8bc97c0303ff43f7ce237c9a34af07c856c1e6eb99011f65009b77e260",
  "contract_sha256": "348e812ded965cce9921eff547a7f33c37b6838faeb2282b663186b9e389e404"
}
```

[rddr_phase2b112_optimizer_provenance.json](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_optimizer_provenance.json)

## 3. Exact starting state

```json
{
  "three_arms_bitwise_equal": true,
  "strict_load": true,
  "initial_state_sha256": "c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5",
  "state_hashes": {
    "B": "c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5",
    "A": "c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5",
    "R": "c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5"
  },
  "original_sources_unchanged": true,
  "initial_prediction_sha256": "c18e46ad87f99541402725db30a1dfbe30a4b6f269579990932c05a99da5ab12",
  "main_forward_parity": true,
  "status": "weights_and_forward_verified",
  "prediction_hashes": {
    "B": "c18e46ad87f99541402725db30a1dfbe30a4b6f269579990932c05a99da5ab12",
    "A": "c18e46ad87f99541402725db30a1dfbe30a4b6f269579990932c05a99da5ab12",
    "R": "c18e46ad87f99541402725db30a1dfbe30a4b6f269579990932c05a99da5ab12"
  },
  "full_validation_snapshot_bitwise_equal": true
}
```

Snapshot-array bitwise step0 equality: True. Checkpoint identity is separately attested by the supplied identity artifact.

## 4. Three-arm design

Registered arms: B = official SSHR continuation; A = official loss + calibrated ADT; R = official loss + rate-matched random gate. All start at C0, target exactly 500 steps, and use BCSS seed42 / batch20 / BF16. Auxiliary updates include the approved b4..b4_5/bn45 affine parameters; all BN running statistics remain frozen. BN-affine and the corresponding original optimizer weight-decay effects are part of the A/R treatment. Actual execution checks:

```json
{
  "all_finite": true,
  "no_amp_skipped_step": true,
  "no_unexpected_gradient_path": true,
  "no_state_corruption": true,
  "bn_statistics_frozen": true,
  "no_test_access": true,
  "no_luad_access": true
}
```

## 5. ADT implementation

Registered formula: `sum(q*m_D*KL(stopgrad(p_d)||p_s_aux))/(sum(q*m_D)+eps)`, with `q=JS(p_s,p_d)/ln(2)` and `m_D=(Delta_sym>0)`. q indicates need, not correctness or direction. The 15x15 exclude-self support is frozen. feat56, ic1, deep target, q, Delta and the gate are detached. No third evidence or threshold/loss redesign is authorized. Path correctness is an engineering verification claim, not something snapshot-only statistics can prove.

[rddr_phase2b112_verification.json](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_verification.json)

## 6. Random control

The registered random gate matches the current A-arm per-image active count, uses independent seed42 randomness, and weights R's own predictions/q. This is a three-arm comparison, not a gate seed search. The following are actual paired snapshot differences; implementation/rate matching requires the independent checks.

| comparison | metric | delta_pp | ci_low_pp | ci_high_pp |
| --- | --- | --- | --- | --- |
| A-R | miou | -0.00075765818 | -0.0018808506 | 0.00027511071 |
| A-R | mdice | -0.00057332866 | -0.0014078329 | 0.00020362257 |
| R-B | miou | 0.0016452181 | 0.00030043364 | 0.0030355577 |
| R-B | mdice | 0.001290789 | 0.00029354519 | 0.0023192561 |

[rddr_phase2b112_random_control.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_random_control.csv)

## 7. Lambda calibration

```json
{
  "batches": 32,
  "seed": 42,
  "lambda_value": 0.027074256246554088,
  "ratios": [
    0.5041619439423308,
    0.34363420546084117,
    0.1458683099301866,
    0.23050200153352465,
    0.2690932170649772,
    0.2868644355834325,
    0.2756091385338726,
    0.25192790010550464,
    0.16952115601073228,
    0.2724570855470248,
    0.1781402673308618,
    0.4002179617368389,
    0.34489319994307405,
    0.3053973707861081,
    0.28374051893650043,
    0.2891602017677385,
    0.3038757199475611,
    0.15906437174275248,
    0.2270091748626174,
    0.22013526681416995,
    0.27101269629491664,
    0.41737736348636684,
    0.15674048128213894,
    0.2316765924985148,
    0.2704724286361651,
    0.2641999794257152,
    0.13312012202161647,
    0.3915811829726143,
    0.14879255455958826,
    0.2974649251749936,
    0.14398142173674014,
    0.33384727694668653
  ],
  "r_median": 0.27074256246554085,
  "rows": [
    {
      "batch": 1,
      "main_norm": 3.9352542408959104,
      "aux_norm": 7.805536064626945,
      "ratio": 0.5041619439423308
    },
    {
      "batch": 2,
      "main_norm": 3.2344984587961805,
      "aux_norm": 9.412620757651629,
      "ratio": 0.34363420546084117
    },
    {
      "batch": 3,
      "main_norm": 2.912494788689131,
      "aux_norm": 19.96660404596711,
      "ratio": 0.1458683099301866
    },
    {
      "batch": 4,
      "main_norm": 3.0700546081335545,
      "aux_norm": 13.318993264282001,
      "ratio": 0.23050200153352465
    },
    {
      "batch": 5,
      "main_norm": 2.736846850145686,
      "aux_norm": 10.170627403045597,
      "ratio": 0.2690932170649772
    },
    {
      "batch": 6,
      "main_norm": 2.865643830173781,
      "aux_norm": 9.989540256103599,
      "ratio": 0.2868644355834325
    },
    {
      "batch": 7,
      "main_norm": 2.4581032128077074,
      "aux_norm": 8.918801543111796,
      "ratio": 0.2756091385338726
    },
    {
      "batch": 8,
      "main_norm": 2.289492483474514,
      "aux_norm": 9.087887764699424,
      "ratio": 0.25192790010550464
    },
    {
      "batch": 9,
      "main_norm": 2.6272409794765683,
      "aux_norm": 15.498012399201828,
      "ratio": 0.16952115601073228
    },
    {
      "batch": 10,
      "main_norm": 3.7745816492261004,
      "aux_norm": 13.853857531079163,
      "ratio": 0.2724570855470248
    },
    {
      "batch": 11,
      "main_norm": 2.2299933795654527,
      "aux_norm": 12.518188117694129,
      "ratio": 0.1781402673308618
    },
    {
      "batch": 12,
      "main_norm": 4.407567302042801,
      "aux_norm": 11.012917258668146,
      "ratio": 0.4002179617368389
    },
    {
      "batch": 13,
      "main_norm": 4.242141536624198,
      "aux_norm": 12.299870028969684,
      "ratio": 0.34489319994307405
    },
    {
      "batch": 14,
      "main_norm": 2.7775644639140116,
      "aux_norm": 9.09491936263383,
      "ratio": 0.3053973707861081
    },
    {
      "batch": 15,
      "main_norm": 2.1718537203843886,
      "aux_norm": 7.654365776475626,
      "ratio": 0.28374051893650043
    },
    {
      "batch": 16,
      "main_norm": 2.680694825731405,
      "aux_norm": 9.27062163621331,
      "ratio": 0.2891602017677385
    },
    {
      "batch": 17,
      "main_norm": 3.1958554228360057,
      "aux_norm": 10.516981812001129,
      "ratio": 0.3038757199475611
    },
    {
      "batch": 18,
      "main_norm": 1.9546459466590629,
      "aux_norm": 12.288395720881972,
      "ratio": 0.15906437174275248
    },
    {
      "batch": 19,
      "main_norm": 3.0993454770154396,
      "aux_norm": 13.65295247040577,
      "ratio": 0.2270091748626174
    },
    {
      "batch": 20,
      "main_norm": 1.9835231555094481,
      "aux_norm": 9.010474250736536,
      "ratio": 0.22013526681416995
    },
    {
      "batch": 21,
      "main_norm": 2.860644329663965,
      "aux_norm": 10.555388607480138,
      "ratio": 0.27101269629491664
    },
    {
      "batch": 22,
      "main_norm": 3.5683623283341195,
      "aux_norm": 8.5494869543324,
      "ratio": 0.41737736348636684
    },
    {
      "batch": 23,
      "main_norm": 1.6167186288641953,
      "aux_norm": 10.314620792739777,
      "ratio": 0.15674048128213894
    },
    {
      "batch": 24,
      "main_norm": 2.24057467500599,
      "aux_norm": 9.671130987061575,
      "ratio": 0.2316765924985148
    },
    {
      "batch": 25,
      "main_norm": 2.47054788377996,
      "aux_norm": 9.134194910485958,
      "ratio": 0.2704724286361651
    },
    {
      "batch": 26,
      "main_norm": 3.2747904249561133,
      "aux_norm": 12.395119899072066,
      "ratio": 0.2641999794257152
    },
    {
      "batch": 27,
      "main_norm": 1.7558548339306905,
      "aux_norm": 13.190003178590596,
      "ratio": 0.13312012202161647
    },
    {
      "batch": 28,
      "main_norm": 3.224976235711125,
      "aux_norm": 8.235779378655321,
      "ratio": 0.3915811829726143
    },
    {
      "batch": 29,
      "main_norm": 1.7139950341907544,
      "aux_norm": 11.519360211107943,
      "ratio": 0.14879255455958826
    },
    {
      "batch": 30,
      "main_norm": 3.4569596813592263,
      "aux_norm": 11.62140267915925,
      "ratio": 0.2974649251749936
    },
    {
      "batch": 31,
      "main_norm": 2.2722202173645685,
      "aux_norm": 15.78134309632911,
      "ratio": 0.14398142173674014
    },
    {
      "batch": 32,
      "main_norm": 2.7082744188772,
      "aux_norm": 8.112315428504221,
      "ratio": 0.33384727694668653
    }
  ],
  "no_optimizer_step": true,
  "state_unchanged": true,
  "budget": 0.1,
  "recalibrations": 0,
  "use_validation": false
}
```

All-step weighted-loss consistency audit:

```json
{
  "B": {
    "rows": 500,
    "step_sequence_exact": true,
    "weighted_gradient_ratio_median_all500": 0.0,
    "weighted_gradient_ratio_max": 0.0,
    "lambda_weight_consistent_all500": true,
    "lambda_max_absolute_weight_error": 0.0
  },
  "A": {
    "rows": 500,
    "step_sequence_exact": true,
    "weighted_gradient_ratio_median_all500": 0.11301148506817371,
    "weighted_gradient_ratio_max": 0.3025172659321994,
    "lambda_weight_consistent_all500": true,
    "lambda_max_absolute_weight_error": 0.0
  },
  "R": {
    "rows": 500,
    "step_sequence_exact": true,
    "weighted_gradient_ratio_median_all500": 0.09869657580985505,
    "weighted_gradient_ratio_max": 0.2765523775260766,
    "lambda_weight_consistent_all500": true,
    "lambda_max_absolute_weight_error": 0.0
  }
}
```

This is one train-only no-step 32-batch calibration, not a validation-selected strength. Logged loss-weight equality checks consistency; detachment and absence of validation use require implementation verification.

## 8. Train stream synchronization

Same transformed batches and main-network RNG across arms are preregistered. Batch manifest evidence (large entry arrays remain in the linked artifact):

```json
{
  "seed": 42,
  "batch_size": 20,
  "num_workers": 4,
  "same_batches_all_arms": true,
  "policy": "Official seeded4-worker DataLoader read once per step; tensor reused across all arms",
  "calibration_count": 32,
  "training_count": 500,
  "artifact_sha256": "e9ff00a5b1f03b7f3734517e8454f4fa5b1fbc86b14ad7a8cc4340f48ee5e2f9"
}
```

[rddr_phase2b112_batch_manifest.json](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_batch_manifest.json)

Matching names alone does not prove augmentation/RNG equality; those require verification.

## 9. Snapshot schedule

```json
{
  "evaluated_steps": [
    0,
    50,
    100,
    250,
    500
  ],
  "primary_step": 500,
  "registered_checkpoint_steps": [
    0,
    250,
    500
  ],
  "actual_training_steps": {
    "B": 500,
    "A": 500,
    "R": 500
  },
  "validation_images": 3418
}
```

No best-checkpoint selection, horizon extension, test/LUAD evaluation or other seed is authorized.

## 10. Official validation curves

Same FINAL-style canonical evaluator/resolution/background handling must be supplied by the runner and verified independently. The analysis pools per-image 5x5 matrices, zeros [4,4], computes foreground nanmean IoU, and assigns absent-class Dice zero, matching `tool/iouutils.py`. Values below are fractions.

| arm | step | miou | mdice | accuracy | n_images |
| --- | --- | --- | --- | --- | --- |
| B | 0 | 0.67327917 | 0.80267975 | 0.82903273 | 3418 |
| A | 0 | 0.67327917 | 0.80267975 | 0.82903273 | 3418 |
| R | 0 | 0.67327917 | 0.80267975 | 0.82903273 | 3418 |
| B | 50 | 0.67330748 | 0.80270781 | 0.82902096 | 3418 |
| A | 50 | 0.67323561 | 0.80265048 | 0.82898886 | 3418 |
| R | 50 | 0.6733181 | 0.80271503 | 0.82902673 | 3418 |
| B | 100 | 0.6732861 | 0.80268423 | 0.82904424 | 3418 |
| A | 100 | 0.67328832 | 0.80268594 | 0.82904182 | 3418 |
| R | 100 | 0.67332904 | 0.80271314 | 0.82908448 | 3418 |
| B | 250 | 0.67337862 | 0.80275529 | 0.82909568 | 3418 |
| A | 250 | 0.67333256 | 0.80272639 | 0.82903003 | 3418 |
| R | 250 | 0.67333057 | 0.80272432 | 0.82903385 | 3418 |
| B | 500 | 0.67336433 | 0.80273687 | 0.8291172 | 3418 |
| A | 500 | 0.6733732 | 0.80274405 | 0.82911566 | 3418 |
| R | 500 | 0.67338078 | 0.80274978 | 0.82911751 | 3418 |

[rddr_phase2b112_official_metrics.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_official_metrics.csv)

## 11. Primary mIoU and mDice

Only step500 is the primary endpoint. Native28 metrics cannot replace official metrics.

| arm | miou | mdice |
| --- | --- | --- |
| B | 0.67336433 | 0.80273687 |
| A | 0.6733732 | 0.80274405 |
| R | 0.67338078 | 0.80274978 |

| comparison | metric | delta_pp | ci_low_pp | ci_high_pp |
| --- | --- | --- | --- | --- |
| A-B | miou | 0.00088755993 | -0.00041603015 | 0.0022416198 |
| A-B | mdice | 0.00071746031 | -0.0002449749 | 0.0017334262 |
| A-R | miou | -0.00075765818 | -0.0018808506 | 0.00027511071 |
| A-R | mdice | -0.00057332866 | -0.0014078329 | 0.00020362257 |
| R-B | miou | 0.0016452181 | 0.00030043364 | 0.0030355577 |
| R-B | mdice | 0.001290789 | 0.00029354519 | 0.0023192561 |

## 12. ADT-Baseline bootstrap

10,000 paired image-level percentile bootstrap replicates, seed42. Each draw recomputes the metric from pooled image confusion matrices. It does not average image IoU. CI differences are percentage points.

| step | metric | delta_pp | ci_low_pp | ci_high_pp | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| 0 | miou | 0 | 0 | 0 | 10000 |
| 0 | mdice | 0 | 0 | 0 | 10000 |
| 50 | miou | -0.0071875792 | -0.024459213 | 0.0020432723 | 10000 |
| 50 | mdice | -0.0057336765 | -0.019444598 | 0.0013666913 | 10000 |
| 100 | miou | 0.00022208136 | -0.0010068648 | 0.0014073405 | 10000 |
| 100 | mdice | 0.00017108568 | -0.0007293712 | 0.001048866 | 10000 |
| 250 | miou | -0.0046067081 | -0.017201889 | 0.0021685718 | 10000 |
| 250 | mdice | -0.0028899619 | -0.011282225 | 0.0016334482 | 10000 |
| 500 | miou | 0.00088755993 | -0.00041603015 | 0.0022416198 | 10000 |
| 500 | mdice | 0.00071746031 | -0.0002449749 | 0.0017334262 | 10000 |

[rddr_phase2b112_bootstrap.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_bootstrap.csv)

## 13. ADT-Random bootstrap

| step | metric | delta_pp | ci_low_pp | ci_high_pp | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| 0 | miou | 0 | 0 | 0 | 10000 |
| 0 | mdice | 0 | 0 | 0 | 10000 |
| 50 | miou | -0.0082490811 | -0.025355674 | 0.00019022052 | 10000 |
| 50 | mdice | -0.0064550796 | -0.020025619 | 0.00016537204 | 10000 |
| 100 | miou | -0.0040716623 | -0.012037887 | 0.00031136458 | 10000 |
| 100 | mdice | -0.0027198186 | -0.0080219609 | 0.00022454729 | 10000 |
| 250 | miou | 0.00019839174 | -0.0017553108 | 0.0016785109 | 10000 |
| 250 | mdice | 0.00020781945 | -0.0010841255 | 0.0012383838 | 10000 |
| 500 | miou | -0.00075765818 | -0.0018808506 | 0.00027511071 | 10000 |
| 500 | mdice | -0.00057332866 | -0.0014078329 | 0.00020362257 | 10000 |

Random-Baseline uses the same paired resamples and is retained in [rddr_phase2b112_random_control.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_random_control.csv).

## 14. Native28 curves

Mechanism diagnostic only. All metrics mask foreground truth 0..3, ignoring background/255. NLL retains the prior diagnostic `-mean(log(p_GT+1e-8))` (additive EPS, not a probability floor); Brier is the sum of squared errors over four classes.

| arm | step | head | accuracy | miou | mdice | nll | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | raw | 0.71425206 | 0.43634813 | 0.58173036 | 0.7848377 | 0.40625041 |
| B | 0 | deep | 0.76743576 | 0.5662991 | 0.71734402 | 1.7849545 | 0.40958757 |
| B | 0 | rect | 0.81779389 | 0.63790907 | 0.77532364 | 0.84068625 | 0.30371035 |
| A | 0 | raw | 0.71425206 | 0.43634813 | 0.58173036 | 0.7848377 | 0.40625041 |
| A | 0 | deep | 0.76743576 | 0.5662991 | 0.71734402 | 1.7849545 | 0.40958757 |
| A | 0 | rect | 0.81779389 | 0.63790907 | 0.77532364 | 0.84068625 | 0.30371035 |
| R | 0 | raw | 0.71425206 | 0.43634813 | 0.58173036 | 0.7848377 | 0.40625041 |
| R | 0 | deep | 0.76743576 | 0.5662991 | 0.71734402 | 1.7849545 | 0.40958757 |
| R | 0 | rect | 0.81779389 | 0.63790907 | 0.77532364 | 0.84068625 | 0.30371035 |
| B | 50 | raw | 0.71426537 | 0.43637239 | 0.58176511 | 0.7848378 | 0.4062509 |
| B | 50 | deep | 0.76742407 | 0.56626838 | 0.71731616 | 1.7849337 | 0.40960367 |
| B | 50 | rect | 0.81780317 | 0.63791211 | 0.77532484 | 0.84064701 | 0.30370643 |
| A | 50 | raw | 0.7142448 | 0.43634304 | 0.58173093 | 0.78482117 | 0.40624601 |
| A | 50 | deep | 0.76741358 | 0.56626218 | 0.71731259 | 1.7849031 | 0.40960307 |
| A | 50 | rect | 0.81779349 | 0.63792195 | 0.775335 | 0.84058804 | 0.30369872 |
| R | 50 | raw | 0.71425085 | 0.4363504 | 0.58173959 | 0.7848248 | 0.40624745 |
| R | 50 | deep | 0.767416 | 0.56626765 | 0.71731683 | 1.7849265 | 0.40961007 |
| R | 50 | rect | 0.81779873 | 0.63793259 | 0.77534333 | 0.84060875 | 0.30370434 |
| B | 100 | raw | 0.7142331 | 0.43632463 | 0.58170994 | 0.78482986 | 0.40624779 |
| B | 100 | deep | 0.7674281 | 0.56628632 | 0.71733513 | 1.7848316 | 0.40957506 |
| B | 100 | rect | 0.81777534 | 0.6378748 | 0.77529702 | 0.84063483 | 0.30371031 |
| A | 100 | raw | 0.71424359 | 0.43633809 | 0.58172239 | 0.78481082 | 0.40624491 |
| A | 100 | deep | 0.7674398 | 0.56629949 | 0.71734438 | 1.7847871 | 0.40959004 |
| A | 100 | rect | 0.81779671 | 0.63791572 | 0.77532867 | 0.84052431 | 0.30369633 |
| R | 100 | raw | 0.71425368 | 0.4363534 | 0.58174043 | 0.78480825 | 0.40624252 |
| R | 100 | deep | 0.76743133 | 0.56629255 | 0.71734022 | 1.7847932 | 0.40959062 |
| R | 100 | rect | 0.81779268 | 0.63792335 | 0.77533599 | 0.84054917 | 0.30369252 |
| B | 250 | raw | 0.71426336 | 0.43636934 | 0.58175837 | 0.78482656 | 0.40623437 |
| B | 250 | deep | 0.76747408 | 0.5663396 | 0.7173771 | 1.7849932 | 0.40953776 |
| B | 250 | rect | 0.81780196 | 0.63791185 | 0.77532472 | 0.84077943 | 0.30370416 |
| A | 250 | raw | 0.71426618 | 0.43638891 | 0.58178612 | 0.78477059 | 0.40622259 |
| A | 250 | deep | 0.76744504 | 0.56630837 | 0.71735062 | 1.7848446 | 0.40956251 |
| A | 250 | rect | 0.81780882 | 0.63795438 | 0.77536064 | 0.84054151 | 0.30367734 |
| R | 250 | raw | 0.71427909 | 0.43641993 | 0.58182085 | 0.78476852 | 0.40621869 |
| R | 250 | deep | 0.76744504 | 0.56631233 | 0.71735489 | 1.7849156 | 0.40957009 |
| R | 250 | rect | 0.81781003 | 0.63795406 | 0.77535978 | 0.84055209 | 0.30366828 |
| B | 500 | raw | 0.71428554 | 0.43643758 | 0.58184791 | 0.78481314 | 0.40622651 |
| B | 500 | deep | 0.76745875 | 0.566286 | 0.71732122 | 1.7850991 | 0.40954107 |
| B | 500 | rect | 0.81780882 | 0.6379427 | 0.77535048 | 0.84075402 | 0.30367821 |
| A | 500 | raw | 0.71428352 | 0.43648693 | 0.58192302 | 0.78470832 | 0.40620609 |
| A | 500 | deep | 0.76742165 | 0.56624427 | 0.71728407 | 1.7848521 | 0.40960034 |
| A | 500 | rect | 0.81782414 | 0.63803692 | 0.77542953 | 0.84024195 | 0.30361671 |
| R | 500 | raw | 0.71429522 | 0.43652597 | 0.58197189 | 0.78470896 | 0.40620224 |
| R | 500 | deep | 0.76740833 | 0.56623121 | 0.71727314 | 1.7849456 | 0.4096044 |
| R | 500 | rect | 0.81783342 | 0.63807562 | 0.77546163 | 0.84029531 | 0.30360998 |

[rddr_phase2b112_native28_metrics.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_native28_metrics.csv)

## 15. Frozen Deep-Win

Frozen B-step0 membership: 314754 foreground pixels in 3397 images. All means use pooled foreground pixels. GT margin is a logit difference; probability is reported separately.

| arm | step | raw_accuracy | rect_accuracy | raw_gt_probability | raw_gt_margin | raw_accuracy_gain_vs_step0 | rect_accuracy_gain_vs_step0 | raw_gt_probability_gain_vs_step0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 0 | 0.75743597 | 0.22679912 | -1.0778063 | 0 | 0 | 0 |
| A | 0 | 0 | 0.75743597 | 0.22679912 | -1.0778063 | 0 | 0 | 0 |
| R | 0 | 0 | 0.75743597 | 0.22679912 | -1.0778063 | 0 | 0 | 0 |
| B | 50 | 0.00147099 | 0.75747727 | 0.22679778 | -1.0778043 | 0.00147099 | 4.1302096e-05 | -1.3397827e-06 |
| A | 50 | 0.0014551046 | 0.75752175 | 0.22681023 | -1.0776811 | 0.0014551046 | 8.5781277e-05 | 1.1111196e-05 |
| R | 50 | 0.00147099 | 0.75746138 | 0.22680904 | -1.077696 | 0.00147099 | 2.5416675e-05 | 9.9276783e-06 |
| B | 100 | 0.0013280212 | 0.75735654 | 0.22679882 | -1.0777939 | 0.0013280212 | -7.9427108e-05 | -2.9396197e-07 |
| A | 100 | 0.0014392192 | 0.75748045 | 0.22681635 | -1.0775896 | 0.0014392192 | 4.4479181e-05 | 1.7238924e-05 |
| R | 100 | 0.001467813 | 0.75747409 | 0.22681905 | -1.0775734 | 0.001467813 | 3.8125012e-05 | 1.9931229e-05 |
| B | 250 | 0.0015631255 | 0.75738196 | 0.22679206 | -1.0779597 | 0.0015631255 | -5.4010434e-05 | -7.0535922e-06 |
| A | 250 | 0.0016234901 | 0.75753128 | 0.22683758 | -1.0774601 | 0.0016234901 | 9.531253e-05 | 3.8464811e-05 |
| R | 250 | 0.0016076047 | 0.75760435 | 0.22684538 | -1.077412 | 0.0016076047 | 0.00016838547 | 4.6261528e-05 |
| B | 500 | 0.0016330213 | 0.75737242 | 0.22677996 | -1.0780102 | 0.0016330213 | -6.3541687e-05 | -1.915172e-05 |
| A | 500 | 0.0016616151 | 0.75762024 | 0.22687758 | -1.0769051 | 0.0016616151 | 0.00018427089 | 7.8467465e-05 |
| R | 500 | 0.0017410422 | 0.7577092 | 0.22689091 | -1.076867 | 0.0017410422 | 0.00027322925 | 9.1795083e-05 |

Step500 paired A-B image bootstrap:

| metric | delta | ci_low | ci_high | unit | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| raw_accuracy | 2.8593759e-05 | -0.00010927351 | 0.00016469113 | fraction | 10000 |
| deep_accuracy | 3.1770843e-06 | -0.00012650737 | 0.00013170706 | fraction | 10000 |
| rect_accuracy | 0.00024781258 | 0.00012469723 | 0.00037400577 | fraction | 10000 |
| raw_gt_probability | 9.7619185e-05 | 9.0202908e-05 | 0.00010483385 | fraction | 10000 |
| deep_gt_probability | 1.6016944e-05 | -5.7520376e-06 | 3.7405067e-05 | fraction | 10000 |
| rect_gt_probability | 0.00023656796 | 0.00020472761 | 0.00026906114 | fraction | 10000 |
| raw_gt_margin | 0.0011051367 | 0.0010467578 | 0.0011611248 | logit | 10000 |
| deep_gt_margin | -0.0013672895 | -0.0019017504 | -0.00082730595 | logit | 10000 |

[rddr_phase2b112_deepwin.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_deepwin.csv)

## 16. Frozen Shallow-Win

Frozen B-step0 membership: 182904 foreground pixels in 3192 images. All means use pooled foreground pixels. GT margin is a logit difference; probability is reported separately.

| arm | step | raw_accuracy | rect_accuracy | raw_gt_probability | raw_gt_margin | raw_accuracy_gain_vs_step0 | rect_accuracy_gain_vs_step0 | raw_gt_probability_gain_vs_step0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 1 | 0.75935463 | 0.61872384 | 1.0926992 | 0 | 0 | 0 |
| A | 0 | 1 | 0.75935463 | 0.61872384 | 1.0926992 | 0 | 0 | 0 |
| R | 0 | 1 | 0.75935463 | 0.61872384 | 1.0926992 | 0 | 0 | 0 |
| B | 50 | 0.99884639 | 0.75931636 | 0.61871953 | 1.0926858 | -0.0011536106 | -3.8271443e-05 | -4.3106603e-06 |
| A | 50 | 0.99867143 | 0.75929449 | 0.61870739 | 1.0926132 | -0.0013285658 | -6.0140839e-05 | -1.6456687e-05 |
| R | 50 | 0.99875891 | 0.75931636 | 0.61870066 | 1.0925866 | -0.0012410882 | -3.8271443e-05 | -2.3183318e-05 |
| B | 100 | 0.99880265 | 0.75925622 | 0.61872378 | 1.0926851 | -0.0011973494 | -9.8412282e-05 | -5.988653e-08 |
| A | 100 | 0.99873704 | 0.75929996 | 0.61868138 | 1.092476 | -0.0012629576 | -5.467349e-05 | -4.2459697e-05 |
| R | 100 | 0.99880812 | 0.75923435 | 0.61868043 | 1.0924831 | -0.0011918821 | -0.00012028168 | -4.3412383e-05 |
| B | 250 | 0.99877531 | 0.75939837 | 0.61880662 | 1.0931153 | -0.0012246862 | 4.3738792e-05 | 8.2774805e-05 |
| A | 250 | 0.99875344 | 0.75928903 | 0.61871061 | 1.0926479 | -0.0012465556 | -6.5608188e-05 | -1.3230827e-05 |
| R | 250 | 0.99878625 | 0.75925075 | 0.61871459 | 1.0926767 | -0.0012137515 | -0.00010387963 | -9.25606e-06 |
| B | 500 | 0.99872611 | 0.75942571 | 0.61885886 | 1.0934535 | -0.0012738923 | 7.1075537e-05 | 0.00013501861 |
| A | 500 | 0.99875344 | 0.75917968 | 0.61863729 | 1.092355 | -0.0012465556 | -0.00017495517 | -8.6555628e-05 |
| R | 500 | 0.99866597 | 0.75926716 | 0.61864708 | 1.0924219 | -0.0013340332 | -8.7477584e-05 | -7.6761091e-05 |

Step500 paired A-B image bootstrap:

| metric | delta | ci_low | ci_high | unit | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| raw_accuracy | 2.7336745e-05 | -0.00013768622 | 0.00019210117 | fraction | 10000 |
| deep_accuracy | -0.00018588987 | -0.00043291942 | 6.5407403e-05 | fraction | 10000 |
| rect_accuracy | -0.0002460307 | -0.00040071927 | -9.6612368e-05 | fraction | 10000 |
| raw_gt_probability | -0.00022157424 | -0.00023061559 | -0.00021234024 | fraction | 10000 |
| deep_gt_probability | -0.00023904394 | -0.00027174915 | -0.00020638953 | fraction | 10000 |
| rect_gt_probability | -0.0002470772 | -0.00027008214 | -0.00022423479 | fraction | 10000 |
| raw_gt_margin | -0.0010985554 | -0.0011515065 | -0.0010452705 | logit | 10000 |
| deep_gt_margin | -0.0029746501 | -0.0035896892 | -0.0023601734 | logit | 10000 |

[rddr_phase2b112_shallowwin.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_shallowwin.csv)

## 17. Stable-Correct

Frozen B-step0 membership: 1587829 foreground pixels in 3412 images. All means use pooled foreground pixels. GT margin is a logit difference; probability is reported separately.

| arm | step | raw_accuracy | rect_accuracy | raw_gt_probability | raw_gt_margin | raw_accuracy_gain_vs_step0 | rect_accuracy_gain_vs_step0 | raw_gt_probability_gain_vs_step0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 1 | 0.98430247 | 0.80842316 | 2.7088818 | 0 | 0 | 0 |
| A | 0 | 1 | 0.98430247 | 0.80842316 | 2.7088818 | 0 | 0 | 0 |
| R | 0 | 1 | 0.98430247 | 0.80842316 | 2.7088818 | 0 | 0 | 0 |
| B | 50 | 0.99972037 | 0.98430876 | 0.80841407 | 2.7087732 | -0.00027962709 | 6.2979074e-06 | -9.0869287e-06 |
| A | 50 | 0.99972982 | 0.98429365 | 0.80840318 | 2.7086273 | -0.00027018023 | -8.8170704e-06 | -1.9975475e-05 |
| R | 50 | 0.99972226 | 0.98429491 | 0.80840463 | 2.7086254 | -0.00027773772 | -7.5574889e-06 | -1.8528176e-05 |
| B | 100 | 0.9997147 | 0.98430813 | 0.80841386 | 2.7087276 | -0.0002852952 | 5.6681167e-06 | -9.2965e-06 |
| A | 100 | 0.99972793 | 0.98430624 | 0.80838529 | 2.7083791 | -0.0002720696 | 3.7787444e-06 | -3.786732e-05 |
| R | 100 | 0.99972478 | 0.98430121 | 0.80839364 | 2.7084831 | -0.00027521855 | -1.2595815e-06 | -2.9521149e-05 |
| B | 250 | 0.99971219 | 0.9843075 | 0.80848594 | 2.7096262 | -0.00028781437 | 5.0383259e-06 | 6.2783415e-05 |
| A | 250 | 0.9997103 | 0.98429743 | 0.80842528 | 2.7088706 | -0.00028970374 | -5.0383259e-06 | 2.1165752e-06 |
| R | 250 | 0.99972352 | 0.98429428 | 0.80843537 | 2.7090032 | -0.00027647813 | -8.1872796e-06 | 1.2209418e-05 |
| B | 500 | 0.99970589 | 0.98430498 | 0.8085126 | 2.7100876 | -0.00029411228 | 2.519163e-06 | 8.9439318e-05 |
| A | 500 | 0.99971093 | 0.98430121 | 0.80837281 | 2.7083612 | -0.00028907395 | -1.2595815e-06 | -5.0353017e-05 |
| R | 500 | 0.99972226 | 0.9842905 | 0.80839796 | 2.7086994 | -0.00027773772 | -1.1966024e-05 | -2.5199603e-05 |

Step500 paired A-B image bootstrap:

| metric | delta | ci_low | ci_high | unit | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| raw_accuracy | 5.0383259e-06 | -1.9422159e-05 | 3.0437412e-05 | fraction | 10000 |
| deep_accuracy | -5.920033e-05 | -8.6534513e-05 | -3.1422893e-05 | fraction | 10000 |
| rect_accuracy | -3.7787444e-06 | -2.0125972e-05 | 1.2478916e-05 | fraction | 10000 |
| raw_gt_probability | -0.00013979233 | -0.00014332187 | -0.00013624327 | fraction | 10000 |
| deep_gt_probability | -5.7446512e-05 | -6.3447078e-05 | -5.1480097e-05 | fraction | 10000 |
| rect_gt_probability | -2.2342532e-05 | -2.5042497e-05 | -1.9731527e-05 | fraction | 10000 |
| raw_gt_margin | -0.0017264505 | -0.0017633283 | -0.0016891761 | logit | 10000 |
| deep_gt_margin | -0.0083636309 | -0.0086917574 | -0.0080363674 | logit | 10000 |

[rddr_phase2b112_stablecorrect.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_stablecorrect.csv)

## 18. Raw-Wrong

Frozen B-step0 membership: 708410 foreground pixels in 3407 images. All means use pooled foreground pixels. GT margin is a logit difference; probability is reported separately.

| arm | step | raw_accuracy | rect_accuracy | raw_gt_probability | raw_gt_margin | raw_accuracy_gain_vs_step0 | rect_accuracy_gain_vs_step0 | raw_gt_probability_gain_vs_step0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 0 | 0.45967025 | 0.18735402 | -1.5538263 | 0 | 0 | 0 |
| A | 0 | 0 | 0.45967025 | 0.18735402 | -1.5538263 | 0 | 0 | 0 |
| R | 0 | 0 | 0.45967025 | 0.18735402 | -1.5538263 | 0 | 0 | 0 |
| B | 50 | 0.000971189 | 0.45969848 | 0.18735695 | -1.5537803 | 0.000971189 | 2.8232238e-05 | 2.9296173e-06 |
| A | 50 | 0.0009231942 | 0.45970413 | 0.18736529 | -1.5536625 | 0.0009231942 | 3.3878686e-05 | 1.1268511e-05 |
| R | 50 | 0.00093872193 | 0.45971401 | 0.18736351 | -1.5536805 | 0.00093872193 | 4.375997e-05 | 9.490286e-06 |
| B | 100 | 0.00088225745 | 0.45961802 | 0.18735987 | -1.5537532 | 0.00088225745 | -5.2229641e-05 | 5.8497827e-06 |
| A | 100 | 0.00090625485 | 0.45968578 | 0.18737215 | -1.5535334 | 0.00090625485 | 1.5527731e-05 | 1.8123441e-05 |
| R | 100 | 0.00093025226 | 0.45969989 | 0.18737193 | -1.5535531 | 0.00093025226 | 2.964385e-05 | 1.7905261e-05 |
| B | 250 | 0.0010008329 | 0.45967589 | 0.18734816 | -1.5540595 | 0.0010008329 | 5.6464477e-06 | -5.8622366e-06 |
| A | 250 | 0.0010205954 | 0.45975071 | 0.1873799 | -1.5535196 | 0.0010205954 | 8.0461879e-05 | 2.5877267e-05 |
| R | 250 | 0.0010276535 | 0.45977188 | 0.18738076 | -1.5535492 | 0.0010276535 | 0.00010163606 | 2.6732681e-05 |
| B | 500 | 0.0011052921 | 0.45969848 | 0.18734245 | -1.5541109 | 0.0011052921 | 2.8232238e-05 | -1.1572406e-05 |
| A | 500 | 0.0010798831 | 0.45982411 | 0.18740927 | -1.5529631 | 0.0010798831 | 0.0001538657 | 5.5248624e-05 |
| R | 500 | 0.0011179966 | 0.45985799 | 0.18740868 | -1.5530715 | 0.0011179966 | 0.00018774439 | 5.4659024e-05 |

Step500 paired A-B image bootstrap:

| metric | delta | ci_low | ci_high | unit | valid_replicates |
| --- | --- | --- | --- | --- | --- |
| raw_accuracy | -2.5409015e-05 | -9.9493283e-05 | 4.6081776e-05 | fraction | 10000 |
| deep_accuracy | 5.0818029e-05 | -3.3040622e-05 | 0.00013571612 | fraction | 10000 |
| rect_accuracy | 0.00012563346 | 4.9004206e-05 | 0.00020338507 | fraction | 10000 |
| raw_gt_probability | 6.682103e-05 | 6.2310767e-05 | 7.1163977e-05 | fraction | 10000 |
| deep_gt_probability | 2.472761e-05 | 8.2278714e-06 | 4.0698125e-05 | fraction | 10000 |
| rect_gt_probability | 0.00012356679 | 0.00010206351 | 0.00014517318 | fraction | 10000 |
| raw_gt_margin | 0.0011478404 | 0.0011027754 | 0.0011905633 | logit | 10000 |
| deep_gt_margin | 0.0030537463 | 0.0025777881 | 0.003521833 | logit | 10000 |

[rddr_phase2b112_rawwrong.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_rawwrong.csv)

The old Phase2B1.9 >=40% local BenefitRate gate is not re-used or reinterpreted. Its `ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE` decision remains frozen.

## 19. Per-class

All four official foreground classes are reported and conservatively checked; no post-hoc powered-class exclusion. The full horizon is retained in the CSV.

| arm | class_id | iou | dice | gt_pixels | iou_delta_vs_B_pp | iou_delta_vs_R_pp |
| --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 0.76466647 | 0.86664135 | 61781842 | 0 | 0.00089560458 |
| B | 1 | 0.70582554 | 0.82754716 | 66845621 | 0 | 0.00032906004 |
| B | 2 | 0.57826994 | 0.73278965 | 20708801 | 0 | -0.003045655 |
| B | 3 | 0.64469537 | 0.78396934 | 9303081 | 0 | -0.004759882 |
| A | 0 | 0.76465508 | 0.86663404 | 61781842 | -0.0011392736 | -0.00024366906 |
| A | 1 | 0.70582204 | 0.82754476 | 66845621 | -0.00034990053 | -2.0840496e-05 |
| A | 2 | 0.57828821 | 0.73280432 | 20708801 | 0.001827601 | -0.001218054 |
| A | 3 | 0.64472749 | 0.78399308 | 9303081 | 0.0032118129 | -0.0015480691 |
| R | 0 | 0.76465751 | 0.8666356 | 61781842 | -0.00089560458 | 0 |
| R | 1 | 0.70582225 | 0.8275449 | 66845621 | -0.00032906004 | 0 |
| R | 2 | 0.57830039 | 0.7328141 | 20708801 | 0.003045655 | 0 |
| R | 3 | 0.64474297 | 0.78400453 | 9303081 | 0.004759882 | 0 |

[rddr_phase2b112_per_class.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_per_class.csv)

## 20. Gate dynamics

These are contextual diagnostic gates Delta>0 for each checkpoint. A uses this gate for training; R instead trains with random per-image counts matched to A. R's Delta>0 here is a counterfactual diagnostic, not its actual random training gate. Actual R training activation is recorded in training_curve.csv. Active fraction includes an all-grid denominator (used by Gate H for A) and a foreground-only diagnostic. DeepCapture/ShallowProtection/SelectionPrecision use frozen exactly-one-correct labels; current populations are separately recorded as drift diagnostics. All q quantiles, Delta summaries, current counts and fractions are in the linked CSV.

| arm | step | mean_q_all | active_fraction_all | active_fraction_foreground | deep_capture_frozen | shallow_protection_frozen | deep_selection_precision_frozen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B | 0 | 0.19209938 | 0.28130486 | 0.28118991 | 0.64032228 | 0.79095591 | 0.84054066 |
| A | 0 | 0.19209938 | 0.28130486 | 0.28118991 | 0.64032228 | 0.79095591 | 0.84054066 |
| R | 0 | 0.19209938 | 0.28130486 | 0.28118991 | 0.64032228 | 0.79095591 | 0.84054066 |
| B | 50 | 0.19209834 | 0.28127202 | 0.28115442 | 0.64034135 | 0.79111446 | 0.84064632 |
| A | 50 | 0.19210219 | 0.28127799 | 0.2811649 | 0.64031593 | 0.79108166 | 0.84061997 |
| R | 50 | 0.19210213 | 0.28129515 | 0.28117902 | 0.64027145 | 0.79104886 | 0.84058962 |
| B | 100 | 0.19208913 | 0.28130112 | 0.28118346 | 0.64024286 | 0.791109 | 0.84062221 |
| A | 100 | 0.19210171 | 0.28128321 | 0.28116732 | 0.64028098 | 0.79105432 | 0.84059512 |
| R | 100 | 0.19209741 | 0.28129777 | 0.28118225 | 0.64027463 | 0.79106526 | 0.84060081 |
| B | 250 | 0.19207132 | 0.28133882 | 0.28122258 | 0.64038265 | 0.79092311 | 0.84053227 |
| A | 250 | 0.19209308 | 0.28128582 | 0.28116571 | 0.640389 | 0.79099418 | 0.84057917 |
| R | 250 | 0.19209548 | 0.28128097 | 0.28115159 | 0.64028416 | 0.791109 | 0.84063085 |
| B | 500 | 0.19208353 | 0.28130971 | 0.28120282 | 0.64040171 | 0.79093404 | 0.84054327 |
| A | 500 | 0.19212914 | 0.2812556 | 0.28114554 | 0.64040489 | 0.79109806 | 0.8406491 |
| R | 500 | 0.19212851 | 0.2812541 | 0.2811407 | 0.64039218 | 0.79098872 | 0.84057633 |

[rddr_phase2b112_gate_dynamics.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_gate_dynamics.csv)

Frozen population definitions/counts:

```json
{
  "All_FG": {
    "n_pixels": 2479143,
    "n_images": 3416
  },
  "Deep-Win_0": {
    "n_pixels": 314754,
    "n_images": 3397
  },
  "Shallow-Win_0": {
    "n_pixels": 182904,
    "n_images": 3192
  },
  "Both-Wrong_0": {
    "n_pixels": 393656,
    "n_images": 3218
  },
  "Stable-Correct_0": {
    "n_pixels": 1587829,
    "n_images": 3412
  },
  "Raw-Correct_0": {
    "n_pixels": 1770733,
    "n_images": 3416
  },
  "Raw-Wrong_0": {
    "n_pixels": 708410,
    "n_images": 3407
  },
  "Exactly-One-Correct_0": {
    "n_pixels": 497658,
    "n_images": 3410
  },
  "Top20_q0": {
    "n_pixels": 485451,
    "n_images": 3405
  },
  "boundary": {
    "n_pixels": 201144,
    "n_images": 2151
  },
  "interior": {
    "n_pixels": 2277999,
    "n_images": 3416
  },
  "Q1_q0": {
    "n_pixels": 495827,
    "n_images": 3404
  },
  "Q2_q0": {
    "n_pixels": 495934,
    "n_images": 3411
  },
  "Q3_q0": {
    "n_pixels": 495727,
    "n_images": 3415
  },
  "Q4_q0": {
    "n_pixels": 495875,
    "n_images": 3413
  },
  "Q5_q0": {
    "n_pixels": 495780,
    "n_images": 3410
  }
}
```

Q-bin edges are inherited, with exact ties assigned to the lower bin. Top20 and boundary labels are inherited unchanged. All frozen-population metrics: [rddr_phase2b112_population_metrics.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_population_metrics.csv)

## 21. Gate drift

Flip rates compare each current arm's Delta>0 with the identical B-step0 gate; frozen masks are never reselected.

| arm | population | n_pixels | gate_flip_rate |
| --- | --- | --- | --- |
| B | all_grid | 2679712 | 0.0019558072 |
| B | All_FG | 2479143 | 0.0019385731 |
| B | Top20_q0 | 485451 | 0.0014913967 |
| B | Deep-Win_0 | 314754 | 0.001769636 |
| B | Shallow-Win_0 | 182904 | 0.0019245068 |
| A | all_grid | 2679712 | 0.0019882734 |
| A | All_FG | 2479143 | 0.0019700356 |
| A | Top20_q0 | 485451 | 0.0015634946 |
| A | Deep-Win_0 | 314754 | 0.0017918756 |
| A | Shallow-Win_0 | 182904 | 0.0019573109 |
| R | all_grid | 2679712 | 0.0020002149 |
| R | All_FG | 2479143 | 0.0019756827 |
| R | Top20_q0 | 485451 | 0.0015717343 |
| R | Deep-Win_0 | 314754 | 0.0018617714 |
| R | Shallow-Win_0 | 182904 | 0.0021213314 |

[rddr_phase2b112_gate_drift.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_gate_drift.csv)

## 22. Representation drift

Runner-supplied diagnostic on the registered fixed 160 validation images. Feature drift cannot be reconstructed from probability-only snapshots. The table below reproduces supplied rows, without inventing missing features.

| arm | step | name | feature_cosine | feature_norm_ratio | raw_logits_cosine | deep_logits_cosine |
| --- | --- | --- | --- | --- | --- | --- |
| B | 0 | TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+0 | 1.0000000000000002 | 1.0 | 1.0 | 1.0000000000000002 |
| B | 0 | TCGA-EW-A1PB-DX1_xmin57214_ymin25940_MPP-0.2500+31 | 0.9999999999999999 | 1.0 | 0.9999999999999999 | 1.0 |
| B | 0 | TCGA-EW-A1PH-DX1_xmin3455_ymin52049_MPP-0.2500+13 | 1.0 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-EW-A1PH-DX1_xmin3455_ymin52049_MPP-0.2500+229 | 0.9999999999999998 | 1.0 | 1.0 | 0.9999999999999999 |
| B | 0 | TCGA-GM-A2DB-DX1_xmin50586_ymin43110_MPP-0.2500+120 | 1.0 | 1.0 | 0.9999999999999998 | 0.9999999999999999 |
| B | 0 | TCGA-GM-A2DB-DX1_xmin50586_ymin43110_MPP-0.2500+220 | 1.0000000000000002 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-GM-A2DB-DX1_xmin50586_ymin43110_MPP-0.2500+32 | 0.9999999999999999 | 1.0 | 0.9999999999999998 | 1.0 |
| B | 0 | TCGA-GM-A2DB-DX1_xmin50586_ymin43110_MPP-0.2500+419 | 0.9999999999999999 | 1.0 | 1.0000000000000002 | 0.9999999999999998 |
| B | 0 | TCGA-GM-A2DB-DX1_xmin50586_ymin43110_MPP-0.2500+518 | 1.0 | 1.0 | 1.0 | 0.9999999999999999 |
| B | 0 | TCGA-GM-A2DD-DX1_xmin47260_ymin22408_MPP-0.2500+125 | 1.0 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-GM-A2DD-DX1_xmin47260_ymin22408_MPP-0.2500+224 | 1.0000000000000002 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-GM-A2DD-DX1_xmin47260_ymin22408_MPP-0.2500+323 | 1.0 | 1.0 | 1.0 | 0.9999999999999998 |
| B | 0 | TCGA-GM-A2DD-DX1_xmin47260_ymin22408_MPP-0.2500+422 | 1.0000000000000002 | 1.0 | 0.9999999999999999 | 1.0 |
| B | 0 | TCGA-GM-A2DF-DX1_xmin50637_ymin43774_MPP-0.2500+13 | 1.0 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-GM-A2DF-DX1_xmin50637_ymin43774_MPP-0.2500+6 | 1.0 | 1.0 | 1.0000000000000002 | 1.0 |
| B | 0 | TCGA-GM-A2DH-DX1_xmin50963_ymin56303_MPP-0.2500+39 | 0.9999999999999999 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-GM-A3XL-DX1_xmin29910_ymin15820_MPP-0.2500+112 | 1.0000000000000002 | 1.0 | 1.0 | 1.0000000000000002 |
| B | 0 | TCGA-GM-A3XL-DX1_xmin29910_ymin15820_MPP-0.2500+211 | 0.9999999999999999 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-GM-A3XL-DX1_xmin29910_ymin15820_MPP-0.2500+98 | 0.9999999999999999 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-LL-A441-DX1_xmin82006_ymin43121_MPP-0.2500+63 | 1.0 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-LL-A5YO-DX1_xmin36631_ymin44396_MPP-0.2500+86 | 1.0000000000000002 | 1.0 | 0.9999999999999999 | 1.0 |
| B | 0 | TCGA-LL-A740-DX1_xmin39436_ymin24080_MPP-0.2500+36 | 0.9999999999999998 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-OL-A5D6-DX1_xmin115108_ymin40554_MPP-0.2500+45 | 1.0 | 1.0 | 0.9999999999999998 | 1.0 |
| B | 0 | TCGA-OL-A5D7-DX1_xmin114443_ymin22490_MPP-0.2500+153 | 1.0 | 1.0 | 0.9999999999999999 | 1.0 |
| B | 0 | TCGA-OL-A5D7-DX1_xmin114443_ymin22490_MPP-0.2500+252 | 1.0000000000000002 | 1.0 | 0.9999999999999999 | 1.0000000000000002 |
| B | 0 | TCGA-OL-A66I-DX1_xmin38433_ymin22957_MPP-0.2500+21 | 1.0 | 1.0 | 1.0 | 1.0000000000000002 |
| B | 0 | TCGA-OL-A66P-DX1_xmin30143_ymin20310_MPP-0.2500+153 | 0.9999999999999999 | 1.0 | 1.0000000000000002 | 1.0 |
| B | 0 | TCGA-OL-A66P-DX1_xmin30143_ymin20310_MPP-0.2500+5 | 0.9999999999999999 | 1.0 | 0.9999999999999999 | 1.0 |
| B | 0 | TCGA-OL-A97C-DX1_xmin68058_ymin32495_MPP-0.2500+0 | 1.0000000000000002 | 1.0 | 1.0 | 1.0 |
| B | 0 | TCGA-OL-A97C-DX1_xmin68058_ymin32495_MPP-0.2500+39 | 1.0 | 1.0 | 1.0 | 1.0 |

Showing first 30 of 2400 rows; full per-image evidence is linked.

[rddr_phase2b112_representation_drift.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_representation_drift.csv)

## 23. Gradient interaction

Runner-supplied no-step diagnostics on the same fixed training minibatch at 0/50/250/500. These diagnostic rows are distinct from Gate H's median over all 500 A training steps.

| arm | step | main_loss | aux_loss | weighted_aux_loss | total_loss | main_grad_norm | aux_grad_norm | weighted_gradient_ratio | gradient_cosine | total_grad_norm | active_fraction | finite | adjudicated_active_fraction | main_bn_gradients_none | aux_parameter_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0 | 0.23276787996292114 | 1.2157292366027832 | 0.032914964878211335 | 0.26568284484113247 | 3.9352542408959104 | 7.805536064626945 | 0.05370150696294084 | 0.02679604971625028 | 6.265587546959025 | 0.31613520408163265 | True | 0.31613520408163265 | True | 39 |
| R | 0 | 0.23276787996292114 | 1.049800157546997 | 0.02842255847310025 | 0.2611904384360214 | 3.9352542408959104 | 6.444795670785304 | 0.04433971436732438 | 0.05742560382589289 | 6.267189764886808 | 0.31613520408163265 | True | 0.31613520408163265 | True | 39 |
| B | 0 | 0.23276787996292114 | 0.0 | 0.0 | 0.23276787996292114 | 3.9352542408959104 | 0.0 | 0.0 | 0.0 | 6.258462936742635 | 0.31613520408163265 | True | 0.31613520408163265 | True | 0 |
| A | 50 | 0.23284384608268738 | 1.2160117626190186 | 0.03292261405997121 | 0.2657664601426586 | 3.9384710058158934 | 7.807988554687707 | 0.05367450516986572 | 0.024920406004265955 | 6.277628025302261 | 0.31588010204081635 | True | 0.31588010204081635 | True | 39 |
| R | 50 | 0.23284737765789032 | 1.0487456321716309 | 0.02839400798286909 | 0.2612413856407594 | 3.935636440503079 | 6.451433012328662 | 0.044381068406594656 | 0.07156098146752674 | 6.272523142127168 | 0.31588010204081635 | True | 0.31588010204081635 | True | 39 |
| B | 50 | 0.23285526037216187 | 0.0 | 0.0 | 0.23285526037216187 | 3.9421748820177593 | 0.0 | 0.0 | 0.0 | 6.27535767596205 | 0.31607142857142856 | True | 0.31607142857142856 | True | 0 |
| A | 250 | 0.2327982485294342 | 1.2165015935897827 | 0.032935875869191174 | 0.26573412439862537 | 3.9373599038873834 | 7.805126566619019 | 0.053669972093761294 | 0.027601748318838472 | 6.2756198530453045 | 0.31600765306122447 | True | 0.31600765306122447 | True | 39 |
| R | 250 | 0.23281708359718323 | 1.0453717708587646 | 0.028302663197144216 | 0.2611197467943274 | 3.9362840037871387 | 6.651844680589097 | 0.045752223864882376 | 0.05478088033953043 | 6.273929417972536 | 0.31600765306122447 | True | 0.31607142857142856 | True | 39 |
| B | 250 | 0.23277781903743744 | 0.0 | 0.0 | 0.23277781903743744 | 3.9368311794362083 | 0.0 | 0.0 | 0.0 | 6.267063810734685 | 0.31594387755102044 | True | 0.31594387755102044 | True | 0 |
| A | 500 | 0.23253542184829712 | 1.2179666757583618 | 0.03297554187924554 | 0.2655109637275427 | 3.9364090120002846 | 7.808008689918355 | 0.05370275976523137 | 0.02760308820698084 | 6.278298873860035 | 0.31600765306122447 | True | 0.31600765306122447 | True | 39 |
| R | 500 | 0.2327917218208313 | 1.0367177724838257 | 0.028068362627583856 | 0.26086008444841513 | 3.9290159285691626 | 6.786160564355838 | 0.04676240893961338 | 0.05841146751594883 | 6.269023781953491 | 0.31600765306122447 | True | 0.31594387755102044 | True | 39 |
| B | 500 | 0.23274126648902893 | 0.0 | 0.0 | 0.23274126648902893 | 3.931152702816537 | 0.0 | 0.0 | 0.0 | 6.25842655208761 | 0.31613520408163265 | True | 0.31613520408163265 | True | 0 |

[rddr_phase2b112_gradient_interaction.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_gradient_interaction.csv)

## 24. Loss/gradient stability

Every-step loss, norms, LR, active fraction, finite flag and timing are retained. Each 50-step block is summarized by mean/population-std/median/min/max. No clipping is introduced by this analysis.

```json
{
  "present": true,
  "errors": [],
  "arms": {
    "B": {
      "rows": 500,
      "step_sequence_exact": true,
      "weighted_gradient_ratio_median_all500": 0.0,
      "weighted_gradient_ratio_max": 0.0,
      "lambda_weight_consistent_all500": true,
      "lambda_max_absolute_weight_error": 0.0
    },
    "A": {
      "rows": 500,
      "step_sequence_exact": true,
      "weighted_gradient_ratio_median_all500": 0.11301148506817371,
      "weighted_gradient_ratio_max": 0.3025172659321994,
      "lambda_weight_consistent_all500": true,
      "lambda_max_absolute_weight_error": 0.0
    },
    "R": {
      "rows": 500,
      "step_sequence_exact": true,
      "weighted_gradient_ratio_median_all500": 0.09869657580985505,
      "weighted_gradient_ratio_max": 0.2765523775260766,
      "lambda_weight_consistent_all500": true,
      "lambda_max_absolute_weight_error": 0.0
    }
  }
}
```

[rddr_phase2b112_training_curve.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_training_curve.csv)

[rddr_phase2b112_loss_gradient_dynamics.csv](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_loss_gradient_dynamics.csv)

## 25. Runtime/memory

```json
{
  "completed": true,
  "steps_per_arm": {
    "B": 500,
    "A": 500,
    "R": 500
  },
  "checks": {
    "all_finite": true,
    "no_amp_skipped_step": true,
    "no_unexpected_gradient_path": true,
    "no_state_corruption": true,
    "bn_statistics_frozen": true,
    "no_test_access": true,
    "no_luad_access": true
  },
  "torch": "2.11.0+cu128",
  "gpu": "NVIDIA GeForce RTX 4090 D",
  "command": "/home/duyanhong/miniconda3/envs/sshr5090/bin/python tools/run_rddr_phase2b112.py --output /home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2",
  "seconds": 1786.089118734002,
  "peak_allocated_bytes": 3909639168,
  "peak_reserved_bytes": 4555014144,
  "clipping_events": 0,
  "amp_overflow_events": 0,
  "amp_skipped_steps": 0,
  "gradient_scaler_enabled": false,
  "dataset_paths_accessed": 6836,
  "code_commit": "4cd44f5e4ae66a80bfb40e25f5ccc9ba2f088360",
  "lambda_value": 0.027074256246554088,
  "precision": "Official BF16 network; FP32 support/loss; FP64 norm/statistics",
  "test_access": false,
  "luad_access": false,
  "original_sources_changed": false
}
```

[rddr_phase2b112_runtime.json](/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2/rddr_phase2b112_runtime.json)

## 26. Gate A-H

| gate | status | rule |
| --- | --- | --- |
| A | WEAK_POSITIVE | step500 A-B official mIoU >= +0.10 pp and paired 95% CI lower > 0; weak-positive is not PASS |
| B | FAIL | A-B mIoU > 0 at 250 and 500; delta500 >= delta100 - 0.10 pp |
| C | FAIL | step500 A-R official mIoU >= +0.05 pp and paired 95% CI lower > 0 |
| D | FAIL | Deep-Win_0 raw accuracy A-B > 0 with CI lower > 0 and raw logit GT margin A-B > 0 |
| E | PASS | Shallow-Win_0 accuracy A-B >= -0.20 pp; CI lower > -0.50 pp; margin_A-margin_B >= -0.05*abs(step0 mean margin) |
| F | PASS | Stable-Correct_0 raw and rect accuracy A-B each >= -0.10 pp; either < -0.30 pp is automatic FAIL |
| G | PASS | Conservative all4-class check: each IoU A-B >= -0.50 pp and macro mean class delta > 0 |
| H | PASS | All numerical/path/state/paired-run evidence passes; step500 A all-grid active fraction in [0.05,0.60]; median weighted gradient ratio over all500 A steps <= 0.30 |

Exact facts (fraction-to-pp conversion only, no rounded-threshold decision):

```json
{
  "A": {
    "delta_pp": 0.0008875599278757029,
    "ci_low_pp": -0.0004160301499214758
  },
  "B": {
    "delta100_pp": 0.00022208135576073929,
    "delta250_pp": -0.004606708079901711,
    "delta500_pp": 0.0008875599278757029
  },
  "C": {
    "delta_pp": -0.0007576581818602435,
    "ci_low_pp": -0.0018808505896944494
  },
  "D": {
    "raw_accuracy_delta_pp": 0.002859375893554967,
    "ci_low_pp": -0.010927351462874943,
    "raw_logit_gt_margin_delta": 0.0011051366826354868
  },
  "E": {
    "raw_accuracy_delta_pp": 0.0027336744959104227,
    "ci_low_pp": -0.013768622010011417,
    "raw_logit_gt_margin_delta": -0.0010985553939840025,
    "step0_mean_logit_margin": 1.0926991557687025,
    "fixed_margin_tolerance": 0.05463495778843513
  },
  "F": {
    "raw_accuracy_delta_pp": 0.000503832591547404,
    "rect_accuracy_delta_pp": -0.00037787444365777745,
    "automatic_fail_below_minus030pp": false
  },
  "G": {
    "per_class_iou_delta_pp": [
      -0.0011392736486093291,
      -0.00034990053202621496,
      0.0018276009626783107,
      0.0032118129294378406
    ],
    "macro_delta_pp": 0.0008875599278701518,
    "classes_checked": [
      0,
      1,
      2,
      3
    ]
  },
  "H": {
    "checks": {
      "runtime.all_finite": true,
      "runtime.no_amp_skipped_step": true,
      "runtime.no_unexpected_gradient_path": true,
      "runtime.no_state_corruption": true,
      "runtime.bn_statistics_frozen": true,
      "runtime.no_test_access": true,
      "runtime.no_luad_access": true,
      "runtime.completed": true,
      "runtime.steps_per_arm": true,
      "identity.three_arms_bitwise_equal": true,
      "identity.strict_load": true,
      "identity.main_forward_parity": true,
      "identity.original_sources_unchanged": true,
      "snapshot.step0_bitwise_equal": true,
      "verification.passed": true,
      "calibration.valid": true,
      "training.complete_finite_consistent": true,
      "step500.active_fraction_all_grid": true,
      "training.A_median_weighted_gradient_ratio_all500": true
    },
    "active_fraction_all_grid_step500_A": 0.2812555976164603,
    "weighted_gradient_ratio_median_all500_A": 0.11301148506817371,
    "weighted_gradient_ratio_median_all500_R": 0.09869657580985505,
    "training_errors": []
  }
}
```

## 27. STRONG_SHORT_HORIZON_ADT_SIGNAL

```json
{
  "STRONG_SHORT_HORIZON_ADT_SIGNAL": false
}
```

Requires all A-H PASS, A-B >=+0.30 pp, A-R >=+0.15 pp, positive Deep-Win accuracy CI, Shallow-Win protection, and every class IoU delta >=-0.25 pp. Missing evidence is not a false/true scientific result.

## 28. Scientific interpretation

The official translation or persistence criterion is not met; a positive sub-threshold point estimate is insufficient. Frozen-point local mechanism evidence has not met this run's preregistered standard for reliable short-horizon optimization translation.

Phase2B1.9 remains `ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE`; Phase2B1.11 remains `THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT`. Neither historical decision is changed. No hyperparameter rescue, threshold search, third-evidence route, or automatic next phase is authorized.

## 29. Exact final decision

Approved priority: provenance blocked; H fail; A/B nonpass; D/E/F/G fail; C fail; all pass GO.

DECISION = ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION
