# Phase2B1.12 初始验证记录错误修复

日期：2026-08-31。范围：修复工程记录错误并按用户授权重新启动；不是协议或模型修改。

## 原因与失败证据

`formal_4090_r1` 已完成32-batch校准和B臂step0 validation计算，但在返回记录时退出：

```text
TypeError: dict() got multiple values for keyword argument 'arm'
```

`evaluate_snapshot()` 返回的字典已含 `arm` 和 `step`，runner再次使用 `dict(arm=arm, step=step, **result)` 构造记录，造成重复关键字。

实际 `steps_per_arm` 为 B=0、A=0、R=0；这是工程中断，不是显存不足、训练发散或科学NO-GO。

## 最小修复

1. 先断言返回结果中的 `arm/step` 与请求一致。
2. 使用 `dict(result)` 保存既有结果，不重复添加字段，不重算指标。

只有runner的 `evaluate` 回调内这两处行为改变。去掉该回调后，修复前后runner的AST完全一致；原network、tool、train_sshr以及common、evaluator、analyzer、launcher均未改变。没有改变seed、batch、BF16、lambda规则、SGD、LR、数据增强、推理或metric。

## 回归验证

- 新测试在旧代码上真实复现同一个TypeError，不仅检查源码字符串。
- 原53项测试之外新增4项：
  - 实际回调覆盖全部五个snapshot step和B/A/R三臂，保留返回元数据及日志；
  - 回放本次真实 `snapshot_0000_B.json`，验证可正确记录；
  - 错误arm被拒绝，不写入结果；
  - 错误step被拒绝，不写入结果。
- 本地：**57/57 PASS，9.290秒**。
- 4090环境：**57/57 PASS，5.465秒**。

回调测试mock GPU与I/O边界，但实际执行生产回调AST；它不是完整GPU验证的替代。因此重启后仍需确认三个step0快照通过并真正执行optimizer.step。

## 旧记录保留与重新执行

旧目录完整保留：

`/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r1`

其9个产物的SHA256逐项验证通过。失败日志、异常记录、真实回调返回结果及SHA清单保存在仓库 `audit/results/rddr_phase2b112_record_fix/`。

新目录：

`/home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2`

使用同一C0模型权重重新执行原流程：来源预检 → 32-batch校准 → 三臂step0 validation → 各500步及固定validation快照 → verifier → 统计和报告。不会覆盖或拼接r1结果，也不把r1当作已训练checkpoint续跑。不运行test或full25，不进行自动失败重试。

```bash
cd /home/duyanhong/DZWdeRepo-rddr-phase2b112
bash tools/execute_rddr_phase2b112.sh \
  /home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2
```

后台启动日志：

```bash
tail -f /home/duyanhong/experiments/RDDR_PHASE2B112/formal_4090_r2.log
```

最终科学报告仍为 `docs/rddr_phase2b112_short_horizon_optimization_report.md`；此修复说明不代表正式实验已经完成或通过科学gate。

独立修复分支：`fix/rddr-phase2b112-evaluation-record`，以迁移分支为base提交PR，不自动merge。
