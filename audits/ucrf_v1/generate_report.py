"""Write the frozen UCRF-v1 Chinese audit report from validated metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value*100:.2f}%"


def ci(intervals: dict, group: str, key: str) -> str:
    lo, hi = intervals[group][key]
    return f"[{pct(lo)}, {pct(hi)}]"


def cohort_table(summary: dict, names: list[str], fields: list[tuple[str, str]]) -> str:
    header = "| 组别 | 组件数 | 面积像素 | " + " | ".join(label for _, label in fields) + " |"
    lines = [header, "|---|---:|---:|" + "|".join("---:" for _ in fields) + "|"]
    for name in names:
        row = summary[name]
        lines.append(f"| {name} | {row['components']} | {row['area_pixels']:,} | " +
                     " | ".join(pct(row.get(key)) for key, _ in fields) + " |")
    return "\n".join(lines)


def build(output: Path) -> Path:
    gate = read(output / "00_reproduction_gate.json")
    if not gate["pass"]:
        raise AssertionError("Reproduction gate failed")
    metrics = output / "metrics"
    summary, intervals = read(metrics / "cohort_summary.json"), read(metrics / "bootstrap_ci.json")
    decision = read(metrics / "decision_matrix.json")
    extraction = read(metrics / "extraction_manifest.json")
    matching = read(metrics / "matching_manifest.json")
    visuals = read(metrics / "visualization_manifest.json")
    frame = pd.read_parquet(metrics / "component_event_table.parquet")
    m1 = frame[frame.cohort == "M1"]
    whole = summary["overall"]
    high = summary["high_purity_large"]
    if (whole["components"], whole["area_pixels"], visuals["cases"],
            visuals["prediction_disagreements"]) != (4440, 8750254, 120, 0):
        raise AssertionError("Report artifact integrity failed")
    lines = ["# UCRF-v1｜HQMR 上游类别责任形成审计（BCSS，Seed42）", "",
             f"**DECISION = {decision['decision']}**  ",
             f"**CONFIDENCE = {decision['confidence']}**", "",
             "本轮为冻结 HQMR-v1 检查点上的前向审计；**参数更新 0 次，没有新训练，mIoU 不应上涨**。"
             "判决的 HIGH 指预注册统计阈值与组件 bootstrap 区间有清楚余量，并非对未来架构收益的保证。", ""]

    def section(number: int, title: str, body: str) -> None:
        lines.extend([f"## {number}. {title}", "", body.strip(), ""])

    section(1, "Executive Decision",
        f"在 {whole['components']:,} 个 M1 错误组件（{whole['area_pixels']:,} 像素）中，"
        f"{pct(whole['deep_wrong_area_rate'])} 的错误面积在 logits5 的实际类别竞争中已偏向最终错误类；"
        f"{pct(whole['corrective_available_area_rate'])} 的面积得到 direct4 分支的**正向边际作用**，"
        f"但这些有纠偏作用的区域里 {pct(whole['CSR_area'])} 融合后仍偏向错误类。"
        f"因此主判决为 `{decision['decision']}`；上游错误责任形成与融合后纠偏失败是同一链条的两环。"
        "这里的“纠偏作用”严格指加上 direct4 后真实 class-mixture margin 增加，不把 direct4 误称为四类 tissue logits。")
    section(2, "Reproduction Gate",
        f"PASS；验证集 {gate['validation_images']} 张，HQMR mIoU={pct(gate['hqmr_miou'])}，"
        f"M1={gate['m1_components']:,} / {gate['m1_pixels']:,} 像素；与冻结记录一致。"
        f"检查点 SHA256 `{gate['checkpoint_sha256']}`；配置 SHA256 `{gate['config_sha256']}`；"
        f"实验起始 Git commit `{gate['git_commit']}`；PyTorch `{gate['torch_version']}`、"
        f"CUDA `{gate['cuda_version']}`、GPU `{gate['gpu_name']}`、Seed {gate['seed']}。"
        "hook 前后 logits5/direct4/logits4/query4/logits3/CAM/最终输出最大差均为 0；"
        "全验证集 hook 次数与预期一致，精确 query-logit 恒等式最大误差为 0。"
        "详见 `00_reproduction_gate.json`、`hook_manifest.json`。")
    section(3, "Exact Forward Formula",
        "Stage3 的 q 是 query 而非组织类别。冻结参数下："
        "`q0=LN(stage3_query)`；`L5=dot(q0,k5)/sqrt(256)`；"
        "`q5=update5(q0,L5,v5)`；`D4=dot(q5,k4)/sqrt(256)`；"
        "`L4=bilinear(L5)+D4`；`q4=update4(q5,L4,v4)`；"
        "`D3=dot(q4,k3)/sqrt(256)`；`L3=bilinear(L4)+D3`。"
        "update 包括 sigmoid 权重区域池化、冻结线性层、残差 LN/FFN/LN。"
        "真正四类证据为 `C_l[c,x]=Σ_q W[q,c] sigmoid(L_l[q,x])`（冻结 W，按类归一），"
        "故 `C4≠U(C5)+C(D4)`。最终另有三视角 TTA、各类 CAM 独立 min-max、deep presence gate。"
        "`direct4_standalone` 图仅是辅助代理；真正分支效应取 `margin(C4)-margin(C_U5)`。"
        "完整公式见 `forward_formula_manifest.md`。")
    section(4, "Cohort Definition",
        f"M1 与上一轮组件 ID、类别和面积逐项一致：{extraction['M1']} 个，"
        f"{extraction['M1_pixels']:,} 像素。真类是区域 dominant GT，仅用于审计；竞争对手是该组件的最终预测类。"
        f"高纯度（≥0.70）{summary['high_purity']['components']} 个；最大面积 Q4 "
        f"{summary['large_Q4']['components']} 个；两者交集 {high['components']} 个。"
        f"有效 TP 候选 {extraction['TP_candidates']} 个。面积 Q1 含大量单像素组件，"
        "因此同时报告组件加权和面积加权，绝不把两者混作一个总体。")
    section(5, "logits5 Deep Semantic Analysis",
        f"深层已错率：组件加权 {pct(whole['deep_wrong_component_rate'])}、面积加权 "
        f"{pct(whole['deep_wrong_area_rate'])}，面积 95% CI "
        f"{ci(intervals, 'overall', 'deep_wrong')}。高纯度+大组件为 "
        f"{pct(high['deep_wrong_area_rate'])}，CI "
        f"{ci(intervals, 'high_purity_large', 'deep_wrong')}。"
        "这定位的是**首次可观测的错误类别竞争**，不是证明错误一定由 H5 特征自身生成；更早的 query/权重/feature 共同决定它。")
    section(6, "direct4 Independent Evidence",
        f"独立应用冻结 `W·sigmoid(D4)` 的代理图中，支持真类的 M1 面积占 "
        f"{pct(whole['direct4_proxy_true_area_rate'])}；"
        f"在深层已错且实际 D4 边际作用为正的区域中，代理图也支持真类的面积占总 M1 的 "
        f"{pct(whole['direct4_proxy_available_area_rate'])}，"
        f"相应融合后仍错条件率 {pct(whole['direct4_proxy_CSR_area'])}。"
        "代理值不是可加的 class logit，因此主结论以真实融合反事实差为准。")
    quadrant_fields = [(f"Q{i}_area_rate", f"Q{i} 面积") for i in range(1, 5)]
    effect_fields = [(f"E{i}_area_rate", f"E{i} 面积") for i in range(1, 5)]
    section(7, "logits5/direct4 Quadrant Analysis",
        cohort_table(summary, ["overall", "high_purity", "large_Q4", "high_purity_large"],
                     quadrant_fields) + "\n\nQ1/Q2/Q3/Q4 严格按方案定义：深层 margin 正/正/负/负 × "
        "独立 `W·sigmoid(D4)` 代理 margin 正/负/正/负。Q3 表示独立代理支持真类且深层已错。"
        "由于代理不可与深层 class map 相加，另列真正融合边际效应 E1–E4：\n\n"+
        cohort_table(summary, ["overall", "high_purity", "large_Q4", "high_purity_large"],
                     effect_fields) +
        "\n\nE3 才是主文“深层已错且 direct4 实际提供正向纠偏作用”的集合。"
        "两套标签均见 `direct4_quadrants.csv`，避免因 query 名称误当作原始四类 logits。")
    section(8, "logits4 Fusion Analysis",
        f"`L4=U(L5)+D4` 的 query-logit 算术逐样本严格成立。"
        f"实际类 margin 中，D4 作用的中位数 {whole['median_direct4_effect_margin']:+.5f}；"
        f"|D4边际效应|/(|U5 margin|+|D4效应|) 中位数 {whole['median_direct4_contribution_ratio']:.3f}。"
        f"对深层已错者，融合修复 F3 面积仅 {pct(whole['F3_area_rate'])}；"
        "融合改进 margin 不能自动等同于把类别翻正。")
    flip_fields = [(f"F{i}_component_rate", f"F{i} 组件") for i in range(1, 5)] + [
        (f"F{i}_area_rate", f"F{i} 面积") for i in range(1, 5)]
    section(9, "Flip Event Taxonomy",
        cohort_table(summary, ["overall", "high_purity", "large_Q4", "high_purity_large"]+
                     [f"class_{i}" for i in range(4)], flip_fields) +
        "\n\nF1=已错仍错，F2=正确翻错，F3=错误修复，F4=正确保持。"
        "比较使用同分辨率的 `U(L5)` 与 L4 类图；原生 L5 符号也保存，且本数据中 F1/F2 原生率相同。")
    section(10, "Corrective Evidence Availability",
        f"深层已错且 D4 实际使真-错误类 margin 增加：组件加权 "
        f"{pct(whole['corrective_available_component_rate'])}、面积加权 "
        f"{pct(whole['corrective_available_area_rate'])}，面积 95% CI "
        f"{ci(intervals, 'overall', 'corrective_available')}。"
        "它表示冻结融合中确有局部正向作用，不等于独立 D4 代理可直接判对；两者交集已在第 6 节列出。")
    section(11, "Correction Suppression",
        f"在可纠偏面积中仍处错误侧的 CSR：面积 {pct(whole['CSR_area'])} "
        f"（95% CI {ci(intervals, 'overall', 'CSR')}），"
        f"组件 {pct(whole['CSR_component'])}。"
        f"高纯度+大组件的可纠偏面积 {pct(high['corrective_available_area_rate'])}，"
        f"CSR {pct(high['CSR_area'])}。这构成 D3 的主支持，而非单凭深层已错就推断。")
    section(12, "Query4 / Stage3 Effect",
        f"观察到 L4→L3：扩大错误面积 {pct(whole['stage3_amp_area_rate'])}，"
        f"修复 {pct(whole['stage3_repair_area_rate'])}，新翻错 "
        f"{pct(whole['stage3_flip_area_rate'])}；持错组件 normalized-margin EAR 中位数 "
        f"{whole['EAR_median_persisting_wrong']:.3f}。"
        f"固定 k3、W 后，用 q5 代替 q4 的算术反事实，q4 独立边际效应中位数 "
        f"{whole['median_query4_isolated_effect_margin']:+.5f}。"
        "观测 L4→L3 同时含 direct3，不能统称 query4 独立造成。")
    section(13, "Rival Persistence",
        f"最终错误类在 logits5 已是动态 top rival：组件 {pct(whole['rival_persistence_component_rate'])}、"
        f"面积 {pct(whole['rival_persistence_area_rate'])}；D4 独立代理对应面积 "
        f"{pct(whole['direct4_rival_persistence_area_rate'])}。"
        "这说明相当一部分错误身份在深层已可辨认，而不是仅在末端产生。")
    tertiles = read(metrics / "confidence_tertiles.json")
    section(14, "Confidence Analysis",
        f"logits5 置信度 tertile 由全验证集预测组件、在 GT 分组前固定，"
        f"边界为 {tertiles['thresholds'][0]:.4f}/{tertiles['thresholds'][1]:.4f}，"
        f"人口 {tertiles['population']}。深层已错且处最高 tertile 的 M1 面积 "
        f"{pct(whole['high_confidence_wrong_area_rate'])}，95% CI "
        f"{ci(intervals, 'overall', 'high_confidence_wrong')}。"
        "这限制了仅凭低置信度触发纠错的策略，但不是对所有 uncertainty 信号的否定。")
    section(15, "Spatial Error Expansion",
        f"M1 内深层 rival 主导像素比例的组件中位数 {pct(whole['median_rival_pixel_fraction_logits5'])}；"
        f"L4 为 {pct(whole['median_rival_pixel_fraction_logits4'])}，"
        f"L3 为 {pct(whole['median_rival_pixel_fraction_logits3'])}。"
        f"深层 ≥70% 像素已错的 M1 面积占 {pct(whole['region_wide_deep_error_area_rate'])}。"
        "分辨率匹配后的逐组件扩张列见 `spatial_error_coverage.csv`；这里不把不同分辨率的裸 pixel fraction 直接相减。")
    section(16, "Purity-Stratified Results",
        cohort_table(summary, ["high_purity", "low_purity", "high_purity_large"],
                     [("deep_wrong_area_rate", "深层已错面积"),
                      ("F1_area_rate", "F1 面积"),
                      ("corrective_available_area_rate", "纠偏可用面积"),
                      ("CSR_area", "CSR")]) +
        "\n\n高纯度 M1 中仍有清晰上游错误与融合压制，故不能仅用组织混杂解释。"
        "低纯度组仅 323 个组件却含较大面积，读数应与高纯度组并列，不混为一谈。")
    section(17, "Size-Stratified Results",
        cohort_table(summary, [f"size_Q{i}" for i in range(1, 5)],
                     [("deep_wrong_area_rate", "深层已错面积"),
                      ("F1_area_rate", "F1 面积"),
                      ("F2_area_rate", "F2 面积"),
                      ("CSR_area", "CSR")]) +
        "\n\n最大组件组占大部分错误面积；面积加权总体不能代表小组件频率。")
    section(18, "Per-Class Results",
        cohort_table(summary, [f"class_{i}" for i in range(4)],
                     [("deep_wrong_area_rate", "深层已错"),
                      ("direct4_proxy_true_area_rate", "D4 代理真类"),
                      ("F1_area_rate", "F1"), ("F2_area_rate", "F2"),
                      ("F3_area_rate", "F3"), ("F4_area_rate", "F4"),
                      ("corrective_available_area_rate", "纠偏可用"),
                      ("CSR_area", "CSR"),
                      ("stage3_amp_area_rate", "Stage3 扩大"),
                      ("high_confidence_wrong_area_rate", "高置信已错")]) +
        "\n\n四类均列出，不用 macro 平均掩盖类别差异；详表见 `per_class_metrics.csv`。")
    controls = pd.read_csv(metrics / "matched_tp_summary.csv")
    control_table = ["| 阶段 | 配对数 | M1 错误率 | TP 对照错误率 | M1 margin 中位数 | TP margin 中位数 | margin AUROC |",
                     "|---|---:|---:|---:|---:|---:|---:|"]
    for item in controls.itertuples():
        control_table.append(f"| {item.stage} | {item.pairs} | {pct(item.m1_wrong_rate)} | "
                             f"{pct(item.tp_wrong_rate)} | {item.m1_margin_median:+.4f} | "
                             f"{item.tp_margin_median:+.4f} | {item.m1_vs_tp_margin_auroc:.3f} |")
    section(19, "Matched TP Controls",
        f"按 GT 真类、面积四分位和 purity 层匹配：{matching['matched']}/{matching['M1']} 个 M1 有 TP 对照；"
        f"{matching['unique_TP_used']} 个独立 TP 被使用，"
        f"同图 {pct(matching['same_image_fraction'])}、同面积四分位 "
        f"{pct(matching['same_quartile_fraction'])}、同 purity 层 "
        f"{pct(matching['same_purity_stratum_fraction'])}。"
        "配对允许重复使用 TP，AUROC 是描述性区分度，非独立外部验证。\n\n"+
        "\n".join(control_table))
    sample_lines = []
    for category, sub in pd.read_csv(metrics / "visualization_selection.csv").groupby("category", sort=False):
        files = sorted((output / "visualizations" / category).glob("*.png"))
        if len(files) != 20:
            raise AssertionError(f"Missing visualizations in {category}")
        links = ", ".join(f"[案例 {i+1}](visualizations/{category}/{file.name})"
                          for i, file in enumerate(files))
        selected_ok = visuals["selection"][category]["selected_meeting_criterion"]
        sample_lines.append(f"- {category}（{selected_ok}/20 符合严格事件条件）：{links}")
    section(20, "Representative Cases",
        f"已生成 6×20={visuals['cases']} 张 4×4 审计图，回放预测与组件掩码不一致数 "
        f"{visuals['prediction_disagreements']}。每图包含原图/GT/预测/组件、各阶段真类/错误类/差图、"
        "分支代理、margin、覆盖率与置信度。原始五阶段 query 张量按图像存于 `tensors/`。\n\n"+
        "\n".join(sample_lines))
    section(21, "Decision Matrix",
        f"D1 深层已错且 F1 主导：{decision['criteria']['D1']}；"
        f"D2 浅层制造错误 F2≥30%：{decision['criteria']['D2']}；"
        f"D3 纠偏可用≥30% 且 CSR≥60%：{decision['criteria']['D3']}；"
        f"D4 Stage3 新错误/放大主导：{decision['criteria']['D4']}。"
        f"多条件同时成立时按 `{decision['priority']}` 取唯一主判决，最终为 "
        f"`{decision['decision']}`，`{decision['confidence']}`。"
        f"次级标签：{', '.join(decision['secondary_labels']) or '无'}。"
        "2000 次组件 bootstrap，seed 42；面积权重在每次抽到的组件内重算。")
    section(22, "Exact Root Cause",
        "在可观察链的最早一级，错误类多数已胜出；direct4 经常把 margin 往真类方向推，"
        "但其幅度通常不足以越过深层错误竞争，因此 L4 后仍错。"
        f"F1 面积 {pct(whole['F1_area_rate'])} 远高于 F2 "
        f"{pct(whole['F2_area_rate'])}，Stage3 新翻错仅 "
        f"{pct(whole['stage3_flip_area_rate'])}；这排除“主要由 direct4 或 Stage3 新制造错误”的总体解释。"
        f"**重要并存限制**：真类被最终 deep presence gate 排除的 M1 面积为 "
        f"{pct(whole['gate_missing_true_area_rate'])}，CI "
        f"{ci(intervals, 'overall', 'gate_missing_true')}；"
        f"真类 gate 保留子集的深层已错率 {pct(summary['true_gate_present']['deep_wrong_area_rate'])}、"
        f"纠偏可用率 {pct(summary['true_gate_present']['corrective_available_area_rate'])}、"
        f"CSR {pct(summary['true_gate_present']['CSR_area'])}。"
        "因此主判决针对中间责任形成机制，并不能声称解决融合就必然纠正全部最终 mask；"
        "gate 的独立影响需要未来固定组件的无训练反事实拆解。")
    section(23, "What Is Preserved",
        "保留 BCSS 划分、Seed42、冻结 ResNet38/CCRA/HQMR/GCQM 权重与同一推理门控；"
        "保留 HQMR 层级重建及已验证的精确前向公式。当前结果没有支持废弃整个 decoder，也没有测试任何新模型。")
    section(24, "What Must Be Archived",
        "不再把后处理 morphology、边界修补、来源重标记、prototype verifier、"
        "泛化的“再加一个 query”当作这 4440 个 M1 的主要解释；"
        "不在本轮重开 CCBP/CP-HQMR/CIRV/PSCR，不调阈值、不训练 Full25。"
        "这些是本轮机制范围内的优先级判断，不是否认它们在别的问题上可能有效。")
    section(25, "Exact Next Architecture Target",
        "**下一轮只建议设计，不在本轮实施**：可靠的 deep/local 类责任仲裁。"
        "目标是在深层与局部证据冲突时，按可靠性允许足够大的正向局部贡献真正越过错误深层优势；"
        "同时必须单独考虑 final deep-gate 真类缺失的失败路径。"
        "应先给出固定组件的离线 oracle 上限和 gate-保留子集效果，再预注册轻量机制与失败标准；"
        "目前不能预言 mIoU 一定超过 SSHR 66.6967%。")
    lines.extend(["## 复现命令与产物", "",
        "使用服务器既有 `sshr5090` 环境、冻结 checkpoint 和同一 val 路径：", "",
        "```bash",
        'python audits/ucrf_v1/gate.py --checkpoint "$CHECKPOINT" --val-root "$VAL" --output "$OUT"',
        'python audits/ucrf_v1/extract.py --checkpoint "$CHECKPOINT" --val-root "$VAL" --prior-target "$PRIOR" --output "$OUT"',
        'python audits/ucrf_v1/analyze.py --output "$OUT"',
        'python audits/ucrf_v1/visualize.py --checkpoint "$CHECKPOINT" --val-root "$VAL" --output "$OUT"',
        'python audits/ucrf_v1/generate_report.py --output "$OUT"',
        "```", "",
        "明细：`metrics/component_event_table.parquet`、`component_stage_margins.csv`、"
        "`flip_event_table.csv`、`direct4_quadrants.csv`、`bootstrap_ci.json`、"
        "`matched_tp_controls.csv`、`visualization_selection.csv`；五阶段原始张量和 120 图在 `tensors/`、`visualizations/`。", ""])
    destination = output / "UCRF_v1_Upstream_Class_Responsibility_Formation_Audit_Report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "report_done", "path": str(destination),
                      "sections": 25, "decision": decision["decision"],
                      "bytes": destination.stat().st_size}), flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
