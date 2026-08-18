# OSMF-v1.3-R1 执行方案摘要

## 核心修正

模型、loss、λ、optimizer、augmentation 全部保持 OSMF-v1.3 不变，只修正错误的 gradient-connectivity gate。

正确计算图要求：

\[
\nabla_{P_{morph}}L_{struct}\neq0
\]

\[
\nabla_{U_{morph}}L_{struct}=0
\]

其中第二项是 **EXPECTED BY GRAPH**，不再判失败。

但必须满足：

\[
\nabla_{U_{morph}}L_{total}\neq0
\]

且 `u_morph` 在训练过程中产生可测参数更新。

## Stage A — Graph + Parity

先验证：

- `grad(L_struct,p_morph)>0`
- `grad(L_struct,u_morph)=0`
- `grad(L_total,p_morph)>0`
- `grad(L_total,u_morph)>0`
- A0 与 v1.3-R1 初始化 prediction 完全一致。

任何一项不符合计算图预期即 STOP。

## Stage B — Fresh 8-Batch Readiness

从原 A0 checkpoint 重新开始，固定：

- `λ_sem=0.05`
- `λ_struct=0.05`
- `λ_orth=0.05`
- `λ_rec=0.10`
- `structural_interval=4`
- seed `20260817`

重点检查：

\[
Mean(r_{sem})\le0.20,\qquad Max(r_{sem})\le0.30
\]

\[
Mean(r_{struct})\le0.20,\qquad Max(r_{struct})\le0.30.
\]

step 4 / 8 继续执行 exact same-pair causal before/after。

至少一项改善，并要求：

\[
Mean(\Delta StructErr)<0.
\]

通过后输出：

`OSMF_V13R1_READINESS_PASS`

否则 REVIEW/NOGO 并停止。

## Stage C — Fresh 128-Batch Phase-0S

只有 8B PASS 后才允许，而且必须再次从 A0 fresh restart。

32 个 structural-active steps：

\[
4,8,\ldots,128
\]

全部执行 same-pair causal test。

核心 causal gate：

\[
ImprovedFraction\ge75\%
\]

且：

\[
Mean(\Delta StructErr)<0.
\]

同时建立固定、GT-free 的 64-image morphology probe，检查：

\[
AffinityEqErr_M(128)<AffinityEqErr_M(0)
\]

且相对改善：

\[
\ge5\%.
\]

继续要求：

\[
SemAgree_{128}\ge0.90
\]

\[
Cos(H,\hat H)_{128}\ge0.95
\]

以及无 branch collapse、无梯度支配、SSHR 主任务稳定。

## 最终决策

只能输出：

- `OSMF_V13R1_PHASE0S_GO`
- `OSMF_V13R1_PHASE0S_REVIEW`
- `OSMF_V13R1_PHASE0S_NOGO`

即使 GO 也必须 STOP，不自动进入 3-epoch pilot。

如果 corrected 128B 最终仍然 NOGO，则正式结束 OSMF local-structural morphology specialization 路线，不继续 v1.4/v1.5 无限修补。

