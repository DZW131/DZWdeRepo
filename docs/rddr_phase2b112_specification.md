# RDDR-Net Phase-2B1.12
# Short-Horizon ADT Optimization Dynamics Audit
## Codex 完整执行技术规格 v1.0

> 性质：短程真实训练 / paired controlled optimization / 不做 Full25 / 不做超参数搜索
>
> 目标：验证 Phase-2B1.9 已证明具有局部机制价值的 ADT，能否在真实参数更新下把 zero-update 局部梯度优势转化为可测的 validation 改善，同时不重新引入 hierarchy-safety 回退。
>
> Phase-2B1.11 已经说明：
>
> ```text
> third evidence can enrich Both-Wrong               ✓
> GT-blind third alternative precision               ✗
> safe third-evidence selector                       ✗
> ```
>
> 因此第三证据路线在本轮冻结、归档，不进入模型。
>
> 本轮只验证：
>
> ```text
> Innovation 1:
> Hierarchical Semantic Conflict Awareness
>
> Innovation 2:
> Symmetric Contextual Adjudication
> + Adjudication-Guided Directional Transfer (ADT)
> ```
>
> 唯一核心问题：
>
> **Do ADT's frozen-point local advantages survive real short-horizon optimization?**

---

# 0. Frozen Scientific Status

## 0.1 Innovation 1 — Frozen

```text
q_i = JS(p_s(i), p_d(i))/ln(2)
```

角色固定为：

```text
Need for Rectification
```

不表示 deep correctness / direction / teacher reliability / third-evidence activation。

Phase-2B1.10：
```text
q residual utility image AUROC = 0.9067
q rejected winner image AUROC  = 0.5036
```

因此：
```text
need != direction
```

## 0.2 Innovation 2 — Current Stable Core

Phase-2B1.9：
```text
Delta_sym = S_D_sym - S_S_sym
m_D = 1[Delta_sym > 0]
```

只在 `m_D=1` 执行 `deep -> shallow`。

冻结证据：
```text
Image AUROC               = 0.784842
DeepSelectionPrecision    = 84.0522%
DeepCapture               = 64.0314%
ShallowProtection         = 79.0939%
BRR_ADT                   = 64.0311%
HHCR_ADT                  = 20.1598%
ADT significantly > rate-matched random gating
```

唯一旧失败：
```text
Raw-Wrong all-denominator BenefitRate
= 35.5865% < 40%
```

旧 decision 永久保持：
```text
ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE
```

不得改判。

## 0.3 Phase-2B1.10 / 1.11 Routes Frozen

以下路线本轮全部禁止：
```text
absolute S_D residual recovery
Delta relaxation
third evidence
M_alt
ctx alternative class
confidence/entropy rescue
Top20-only recovery
class-specific rule
```

Phase-2B1.11 final：
```text
DECISION =
THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT
```

---

# 1. Why a Short-Horizon Real Update Is Now Needed

目前已有大量 zero-update / logit-gradient / feature-gradient / directional-derivative / bootstrap 证据，但这些不能回答：

```text
Will shared-parameter optimization
actually improve the model?
```

因此本轮第一次允许：
```text
real optimizer.step()
```

但仅允许：
```text
short horizon
paired arms
single pre-registered auxiliary strength
no validation tuning
no Full25
```

---

# 2. Experimental Arms

必须从完全相同的 C0 权重 clone 三份。

## Arm B — Baseline Continuation
```text
L_B = original SSHR training objective
```

## Arm A — ADT
```text
L_A = L_SSHR + lambda_ADT * L_ADT
```

其中：
```text
L_ADT
=
sum_i q_i*m_D(i)*KL(
    stopgrad(p_d(i))
    ||
    p_s_aux(i)
)
/
(sum_i q_i*m_D(i)+eps)
```

## Arm R — Rate-Matched Random-Gate Control
```text
L_R = L_SSHR + lambda_ADT * L_RG
```

```text
L_RG
=
sum_i q_i*m_rand(i)*KL(
    stopgrad(p_d(i))
    ||
    p_s_aux(i)
)
/
(sum_i q_i*m_rand(i)+eps)
```

`m_rand` 与当前 ADT gate 在每张图上的 active count 完全匹配。

目的：
```text
验证真实训练收益来自 contextual adjudication，
而不是“只在较少位置施加辅助梯度”。
```

---

# 3. Starting Point

三臂统一：
```text
C0 Full25 BCSS seed42 final checkpoint
SHA256:
509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579
```

必须 strict load / same state hash / same initial predictions。

step0 前三臂必须 bitwise identical。

---

# 4. Optimizer Provenance Gate

本轮禁止凭空发明 optimizer/scheduler。

Codex 必须依次尝试：

1. 从原 C0 run artifact 恢复 optimizer state；
2. 若无 optimizer state，则从官方 SSHR training code + frozen run config 精确重建 optimizer type / LR / weight decay / momentum / betas / scheduler；
3. 若 final-step LR 可从 scheduler 精确重建，则使用该 LR；
4. 若无法确定唯一 optimizer + LR，则立即停止：

```text
DECISION =
SHORT_HORIZON_OPTIMIZER_PROVENANCE_BLOCKED
```

禁止：
```text
手工试 LR
试多个 optimizer
经验指定新 LR
```

---

# 5. Optimizer State Policy

若原 optimizer state 可恢复：
```text
resume exact state
```

三臂复制同一 optimizer state。

若只有配置可恢复：
```text
fresh optimizer state
```

但三臂必须：
```text
same initialization
same optimizer config
same LR
same scheduler state reconstruction
```

最终报告必须明确是哪一种。

---

# 6. Auxiliary Gradient Scope

Main SSHR loss：保持官方原始梯度路径。

ADT / RG auxiliary branch 只允许对：
```text
b4
b4_1
b4_2
b4_3
b4_4
b4_5
bn45 affine
```
产生额外梯度。

辅助分支必须：
```text
detach feat56 input
detach ic1 weight/bias
detach deep target
detach q
detach Delta_sym
detach m_D
```

因此：
```text
main SSHR loss = original full-network learning
ADT loss       = local pre-HFRM semantic guidance only
```

禁止 auxiliary gradient 进入：
```text
b3及更早
HFRM28_1
ic1
b5及更深
deep branch
```

---

# 7. Online Dynamic Gate

训练中每个 step 使用当前网络重新计算：
```text
p_s
p_d
q
T_SS/T_SD/T_DS/T_DD
S_S_sym
S_D_sym
Delta_sym
m_D
```

随后全部 detach。

即：
```text
dynamic evidence
but non-differentiable gate
```

禁止使用 validation GT。

---

# 8. Frozen Support Geometry

保持 Phase-2B1.5：
```text
15×15
radius7
exclude self
same support equations
```

禁止 window search / temperature / support normalization modification。

---

# 9. Auxiliary Strength — No Lambda Search

禁止 lambda sweep。

定义一次性 gradient-budget calibration。

在 step0：
```text
32 fixed training mini-batches
seed42
no optimizer.step()
```

计算：
```text
G_main
=
L2 norm of main-loss gradient
over approved ADT-upstream parameters

G_ADT
=
L2 norm of ADT gradient
over same parameters
```

每批：
```text
r_b = G_main/(G_ADT+eps)
```

取：
```text
r_med = median(r_b)
```

固定：
```text
lambda_ADT = 0.10 * r_med
```

含义：
```text
起始点 auxiliary gradient
约占 main gradient norm 的10%预算
```

只校准一次。

禁止：
```text
validation调lambda
训练中重算lambda
试5%/10%/20%
```

Arm R 使用同一个 lambda_ADT。

---

# 10. Training Data / Mini-Batch Stream

只使用 BCSS training split。

与原 baseline 完全相同：
```text
preprocessing
augmentation
sampler
batch size
```

三臂必须复用完全相同：
```text
image order
augmentation RNG
batch boundaries
```

保存：
```text
batch manifest
augmentation seeds
```

---

# 11. Horizon

固定：
```text
500 optimizer steps
```

快照：
```text
step0
step50
step100
step250
step500
```

Primary endpoint：
```text
step500
```

禁止 best checkpoint / early stopping / horizon extension。

---

# 12. Three-Arm Synchronization

每个 global step 三臂读取相同 mini-batch 与 augmentation。

不得不同数据顺序、不同 augmentation、不同 sampler seed。

---

# 13. Validation Policy

只使用 BCSS validation。

不访问 test / LUAD / other seeds。

在：
```text
0/50/100/250/500
```
统一评估。

validation 不用于调lambda、调gate、提前停止、选择checkpoint。

---

# 14. Official Primary Validation Metric

必须运行：
```text
same canonical official evaluator
same resolution
same background handling
same FINAL-style pipeline
```

报告：
```text
mIoU
mDice
per-class IoU
per-class Dice
```

Primary：
```text
dataset-level official mIoU
at step500
```

---

# 15. Paired Image Bootstrap

每个快照：
```text
10,000 image-level paired bootstrap
seed42
```

比较：
```text
ADT - Baseline
ADT - Random
Random - Baseline
```

至少 mIoU / mDice。

---

# 16. Native28 Mechanism Evaluation

在 validation native28 继续记录 raw shallow / deep / rect。

指标：
```text
accuracy
mIoU
Dice
NLL
Brier
```

仅 mechanism diagnostic，不能替代 official metric。

---

# 17. Frozen Step0 Populations

在 step0 冻结：
```text
Deep-Win_0
Shallow-Win_0
Both-Wrong_0
Stable-Correct_0
Raw-Correct_0
Raw-Wrong_0
Top20_q0
Q1-Q5_q0
boundary/interior
```

后续 safety primary diagnostics 都在这些 frozen populations 上评估。

---

# 18. Dynamic Population Diagnostics

每个 snapshot 重新计算 current Deep-Win / Shallow-Win / Both-Wrong / q / m_D，仅用于 mechanism drift。

---

# 19. Deep-Win Training Effect

在 `Deep-Win_0` 比较 step500：
```text
raw accuracy
raw GT probability
raw GT margin
```

主要比较 ADT vs Baseline，并报告 paired CI。

---

# 20. Shallow-Win Protection

在 `Shallow-Win_0`：
```text
raw accuracy
GT probability
GT margin
```

重点比较 ADT vs Baseline。

---

# 21. Stable-Correct Protection

在 `Stable-Correct_0`：
```text
raw accuracy
rect accuracy
GT probability
```

避免 auxiliary loss 损坏原本稳定语义。

---

# 22. Raw-Wrong Correction

在 `Raw-Wrong_0`：
```text
raw accuracy gain
rect accuracy gain
GT probability gain
```

本轮不再设置旧的 `>=40% local BenefitRate`。

因为那是 Phase-2B1.9 frozen-point readiness gate；旧结果永久保留 NOGO。

本轮测试的是 actual parameter-update effect。

---

# 23. Gate Dynamics

每个 snapshot 记录：
```text
mean q
q quintiles
m_D active fraction
Delta mean/median
DeepCapture on frozen exactly-one-correct population
ShallowProtection
DeepSelectionPrecision
```

---

# 24. Gate Drift

定义：
```text
GateFlipRate_t
=
P(m_D_t != m_D_0)
```

并分：
```text
all
Top20
Deep-Win_0
Shallow-Win_0
```

---

# 25. Representation Drift

固定160张 validation images：

记录：
```text
cos(F28_raw_t, F28_raw_0)
norm ratio
raw logits cosine
deep logits cosine
```

三臂比较。

---

# 26. Main-vs-Aux Gradient Interaction

在：
```text
step0
step50
step250
step500
```

固定同一 diagnostic training mini-batch：

分别计算：
```text
g_main
g_aux
```

approved parameters 上报告：
```text
cos(g_main,g_aux)
||g_aux||/||g_main||
```

不 step。

---

# 27. Loss Dynamics

每 step 记录：
```text
L_SSHR
L_ADT or L_RG
lambda*L_aux
total loss
grad norm
LR
```

每50 steps 汇总 mean/std/median/max。

---

# 28. Numerical / Optimization Stability

每 step 检查：
```text
loss finite
gradient finite
parameter finite
```

记录 gradient clipping event / AMP overflow / NaN/Inf。

若官方训练无 clipping，不得新增 clipping。

---

# 29. Gate A — Real Validation Translation

Primary：
```text
step500 official mIoU:
ADT > Baseline
```

且：
```text
paired bootstrap 95% CI lower > 0
```

点估计：
```text
ADT - Baseline >= +0.10 pp
```

若仅 >0 但 <0.10pp：
```text
WEAK_POSITIVE
```
不算 PASS。

---

# 30. Gate B — Persistence Across Horizon

要求：
```text
ADT - Baseline official mIoU > 0
```
在 step250 与 step500 同时成立。

且：
```text
step500 delta >= step100 delta - 0.10 pp
```

避免瞬时早期脉冲。

---

# 31. Gate C — Context-Specific Attribution

step500：
```text
ADT mIoU > RandomGate mIoU
```

且：
```text
paired bootstrap 95% CI lower > 0
```

点差至少：
```text
+0.05 pp
```

---

# 32. Gate D — Deep-Win Mechanism Gain

在 `Deep-Win_0`：

要求 step500：
```text
raw accuracy_ADT > raw accuracy_Baseline
```

且 paired image bootstrap CI lower > 0。

同时：
```text
mean GT margin_ADT > Baseline
```

---

# 33. Gate E — Shallow-Win Protection

在 `Shallow-Win_0`：

要求：
```text
raw accuracy_ADT - raw accuracy_Baseline >= -0.20 pp
```

且：
```text
95% CI lower > -0.50 pp
```

同时：
```text
mean GT margin delta >= Baseline - tolerance
```

固定：
```text
tolerance = 5% of |step0 mean margin|
```

---

# 34. Gate F — Stable-Correct Protection

在 `Stable-Correct_0`：

要求：
```text
raw accuracy_ADT - Baseline >= -0.10 pp
rect accuracy_ADT - Baseline >= -0.10 pp
```

任一低于：
```text
-0.30 pp
```
自动 FAIL。

---

# 35. Gate G — No Major Class Collapse

step500 official per-class IoU 相对 Baseline：

```text
no powered class < -0.50 pp
macro mean class delta > 0
```

4类全部报告。

---

# 36. Gate H — Optimization Stability

要求：
```text
no NaN/Inf
no AMP overflow causing skipped step
no unexpected gradient path
no state corruption
```

且 step500：
```text
0.05 <= mean active transfer fraction <= 0.60
```

以及：
```text
median ||lambda*g_aux||/||g_main|| <= 0.30
```

---

# 37. Strong Short-Horizon Signal

若：
```text
all Gates A-H PASS
ADT-Baseline mIoU >= +0.30 pp
ADT-Random mIoU   >= +0.15 pp
Deep-Win gain CI positive
Shallow-Win protection PASS
all class IoU delta >= -0.25 pp
```

则：
```text
STRONG_SHORT_HORIZON_ADT_SIGNAL = TRUE
```

---

# 38. Decision Logic

A-H 全 PASS：
```text
DECISION =
RDDR_ADT_SHORT_HORIZON_DYNAMICS_GO
```

下一阶段才允许设计：
```text
Phase-2B2 Full25 ADT Training Protocol
```

A PASS, C FAIL：
```text
DECISION =
SHORT_HORIZON_GAIN_NOT_CONTEXT_SPECIFIC
```

A FAIL：
```text
DECISION =
ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION
```

A PASS，但 D/E/F/G 任一 FAIL：
```text
DECISION =
ADT_OPTIMIZATION_GAIN_WITH_SEMANTIC_SAFETY_REGRESSION
```

H FAIL：
```text
DECISION =
ADT_SHORT_HORIZON_ENGINEERING_NOGO
```

optimizer provenance 不能唯一确定：
```text
DECISION =
SHORT_HORIZON_OPTIMIZER_PROVENANCE_BLOCKED
```

---

# 39. No Hyperparameter Rescue

结果出来后禁止：
```text
lambda sweep
horizon extension
LR change
gate threshold
q threshold
Top20-only loss
class mask
boundary mask
third evidence
loss normalization search
random gate seed sweep
```

任何变化必须新 phase。

---

# 40. Checkpoint Policy

允许保存：
```text
step0
step250
step500
```

仅用于 audit。

禁止选择 step50/100 作为 best。

Primary checkpoint：
```text
step500 only
```

---

# 41. Required Artifacts

```text
rddr_phase2b112_optimizer_provenance.json
rddr_phase2b112_lambda_calibration.json
rddr_phase2b112_batch_manifest.json
rddr_phase2b112_training_curve.csv
rddr_phase2b112_loss_gradient_dynamics.csv
rddr_phase2b112_gate_dynamics.csv
rddr_phase2b112_gate_drift.csv
rddr_phase2b112_representation_drift.csv
rddr_phase2b112_official_metrics.csv
rddr_phase2b112_native28_metrics.csv
rddr_phase2b112_deepwin.csv
rddr_phase2b112_shallowwin.csv
rddr_phase2b112_stablecorrect.csv
rddr_phase2b112_rawwrong.csv
rddr_phase2b112_per_class.csv
rddr_phase2b112_random_control.csv
rddr_phase2b112_gradient_interaction.csv
rddr_phase2b112_bootstrap.csv
rddr_phase2b112_runtime.json
rddr_phase2b112_identity_step0.json
rddr_phase2b112_verification.json
rddr_phase2b112_summary.json
```

---

# 42. Required Tests

至少：
```text
test_three_arms_bitwise_equal_step0
test_optimizer_provenance_exact
test_same_batch_manifest
test_same_augmentation_seed
test_adt_formula
test_random_gate_rate_match
test_random_gate_seed42
test_aux_feat56_detached
test_aux_ic1_detached
test_aux_deep_detached
test_aux_q_detached
test_aux_delta_detached
test_aux_gate_detached
test_aux_only_b4_bn45_extra_grad
test_main_loss_gradient_path_unchanged
test_lambda_calibration_32_batches
test_lambda_no_validation_use
test_lambda_frozen_after_step0
test_snapshot_schedule_exact
test_no_early_stop
test_no_test_luad
test_no_other_seed
test_step500_primary
test_official_evaluator_same
test_bootstrap_reproducible
test_no_third_evidence
test_no_threshold_search
test_no_lambda_sweep
```

---

# 43. Required Final Report

输出：
```text
rddr_phase2b112_short_horizon_optimization_report.md
```

必须包含：
1. provenance
2. optimizer provenance
3. exact starting state
4. three-arm design
5. ADT implementation
6. random control
7. lambda calibration
8. train stream synchronization
9. snapshot schedule
10. official val curves
11. mIoU/mDice
12. ADT-Baseline bootstrap
13. ADT-Random bootstrap
14. native28 curves
15. frozen Deep-Win
16. frozen Shallow-Win
17. Stable-Correct
18. Raw-Wrong
19. per-class
20. gate dynamics
21. gate drift
22. representation drift
23. gradient interaction
24. loss/grad stability
25. runtime/memory
26. Gate A-H
27. STRONG_SHORT_HORIZON_ADT_SIGNAL
28. scientific interpretation
29. exact final decision

最后一行只能是：
```text
DECISION = RDDR_ADT_SHORT_HORIZON_DYNAMICS_GO
```
或：
```text
DECISION = SHORT_HORIZON_GAIN_NOT_CONTEXT_SPECIFIC
```
或：
```text
DECISION = ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION
```
或：
```text
DECISION = ADT_OPTIMIZATION_GAIN_WITH_SEMANTIC_SAFETY_REGRESSION
```
或：
```text
DECISION = ADT_SHORT_HORIZON_ENGINEERING_NOGO
```
或：
```text
DECISION = SHORT_HORIZON_OPTIMIZER_PROVENANCE_BLOCKED
```

---

# 44. Codex Strict Execution Order

```text
1. Audit immutable Phase2B1.9/1.10/1.11 evidence.
2. Lock A0/C0/checkpoint SHAs.
3. Recover exact optimizer/scheduler provenance.
4. If provenance unresolved -> STOP.
5. Clone three identical arms.
6. Verify bitwise-equal step0.
7. Freeze train batch/augmentation manifest.
8. Implement dynamic ADT gate.
9. Implement rate-matched random gate.
10. Verify auxiliary gradient scope.
11. Run 32-batch no-step lambda calibration.
12. Freeze lambda.
13. Evaluate step0.
14. Train/evaluate step50.
15. Train/evaluate step100.
16. Train/evaluate step250 + save.
17. Train/evaluate step500 + save.
18. Run frozen-population diagnostics.
19. Run gate/representation drift.
20. Run gradient-interaction diagnostics.
21. 10k image bootstrap.
22. Independent verification.
23. Generate CSV/JSON/report.
24. Apply Gates A-H.
25. Output exact Decision.
26. STOP.
```

---

# 45. Critical Scientific Interpretation

本轮不是：
```text
“试试看加个loss会不会涨点”
```

而是验证：
```text
hierarchical conflict
      ↓
contextual adjudication
      ↓
selective deep-to-shallow transfer
      ↓
local gradient utility
      ↓
REAL parameter update
      ↓
validation improvement ?
```

如果失败：
```text
说明 frozen-point directional evidence
不能可靠转化为 shared-parameter optimization gain
```

如果成功：
```text
说明第二创新点第一次跨过了
mechanism evidence -> optimization evidence
这道门槛
```

届时才有资格进入 Full25。
