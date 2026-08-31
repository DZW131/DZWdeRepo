# Phase2B1.12 执行准备与资源阻塞记录

日期：2026-08-31。状态：**RESOURCE_BLOCKED，尚未开始真实训练。**

这不是短程优化实验结果报告，也不是科学 NO-GO。完整29节结果报告必须在实际500步、validation评估和独立核验完成后生成。当前没有校准lambda、validation收益、A–H判定或可用于论文的结果。

## 已完成

- 从纯官方A0 `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9` 创建独立分支 `feature/rddr-phase2b112-short-horizon`。
- 实现B/A/R三臂，固定500步、32batch一次性校准、匹配批次/增强/主网络RNG，R逐图匹配A激活数量。
- 实现辅助39张量局部分支；主损失保留官方冻结，辅助允许指定BN affine梯度；所有BN运行统计不更新。
- 实现原样调用官方评估器的3TTA快照、native28和固定160图representation诊断、逐图confusion matrix缓存。
- 实现10000次配对image bootstrap、全部CSV/JSON、A–H判定、29节Markdown生成及独立artifact verifier。
- 原 `network/`、`tool/`、`train_sshr.py` 与A0零差异；旧实验/权重/数据未修改、未删除。

## 已实际验证的证据

| 检查 | 结果 | 证据边界 |
|---|---|---|
| 独立CPU测试 | 53/53 PASS | 公式、原optimizer、薄通道图、控制逻辑、原metric及10k bootstrap oracle；不是完整网络CUDA smoke |
| 5090服务器现有环境重跑CPU测试 | 53/53 PASS，正式启动链中14.370s | PyTorch2.11.0+cu128；没有运行GPU forward |
| C0 checkpoint文件 | SHA256完全匹配 | 只读加载260个tensor，文件451130207bytes |
| 三臂初始权重 | strict load，state hash完全一致 | 尚未验证GPU初始预测 |
| 训练样本解析 | 23422 | 通过真实Dataset实例化计数 |
| 优化器来源 | PASS | 原launcher、environment、日志、源码及checkpoint共同锁定 |
| CUDA batch20、32batch校准、500步 | **未执行** | 显存准入前停止 |
| validation/test | **均未执行本轮评估** | test不会进入本轮 |

Checkpoint SHA256：

`509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579`

三臂model-state SHA256：

`c56943fe7749a4ca8a9b66d08e5dfd1e83b95af04ae5a58d5009343c7b7090e5`

模型文件只保存权重，没有optimizer state。因此本轮明确使用fresh optimizer states，不称为完整断点恢复。原SGD momentum=.0005，原poly exponent=.9，global_step=max_step=29275；四组LR分别为：

`9.55328615544644e-7 / 1.910657231089288e-6 / 9.55328615544644e-6 / 1.910657231089288e-5`。

这来自原代码最后一次已应用LR，原PolyOptimizer达到max_step后保持该值；没有重启或人为放大学习率。

## 资源阻塞

正式启动尝试目录：

`/home/duyanhong/experiments/RDDR_PHASE2B112/formal_attempt_r1`

进程建立CUDA上下文后的可用显存：289210368bytes，约275.8MiB。运行器的单臂batch20保守准入余量为18GiB，因此准入失败，并安全退出。

退出后再次查询：GPU总显存24455MiB，已占23189MiB；两个现有GPU进程属于 `zhangjiaqing`，PID221626与1849152。本轮未停止、修改或干扰这些进程；未缩小batch、启用梯度累积或更换设备。

没有后台等待/轮询任务仍在运行，也没有安排自动重试。真实optimizer steps=0，没有新训练checkpoint。

## 恢复执行

协调出足够显存后，使用新输出目录运行同一冻结合同，不需要重新设计实验：

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b112
bash tools/execute_rddr_phase2b112.sh /home/duyanhong/experiments/RDDR_PHASE2B112/formal_r1
```

该有限任务会自动完成：CPU测试 → 来源校验 → 显存准入 → 32batch校准 → 三臂500步及固定validation快照 → 独立核验 → 统计 → 报告。不会覆盖本次阻塞记录，或在失败后自动调整训练方案。

完成后报告位置：`docs/rddr_phase2b112_short_horizon_optimization_report.md`。

## 本次可交付记录

- 批准合同：`docs/rddr_phase2b112_execution_contract.md`
- 独立审查：`docs/rddr_phase2b112_review.md`
- 本次真实启动日志：`audit/results/rddr_phase2b112_preflight/execution_log.txt`
- 来源、初始权重一致性、资源准入、未运行状态：同目录4份JSON。
- 代码审查PR：[DZWdeRepo #50](https://github.com/DZW131/DZWdeRepo/pull/50)，目标为纯A0分支 `baseline/official-a0`，未合并。

**当前科学判定：PENDING，未产生GO/NO-GO结果。**
