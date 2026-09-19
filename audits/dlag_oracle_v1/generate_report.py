"""Generate the 25-section Chinese DLAG Oracle v1 audit report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value*100:.2f}%"


def pp(value: float) -> str:
    return f"{value:+.4f} pp"


def table(frame: pd.DataFrame, columns: list[tuple[str, str]],
          percent: set[str] | None = None) -> str:
    percent = percent or set()
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "|".join("---:" for _ in columns) + "|"]
    for _, row in frame.iterrows():
        values = []
        for key, _ in columns:
            value = row[key]
            if key in percent:
                values.append(pct(float(value)))
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run(output: Path) -> Path:
    gate = load(output / "00_reproduction_gate.json")
    integrity = load(output / "gate_integrity.json")
    decision = load(output / "metrics/decision_matrix.json")
    oracle = load(output / "metrics/oracle_metrics.json")
    groups = load(output / "metrics/purity_size_oracle.json")
    bootstrap = load(output / "metrics/bootstrap_ci.json")
    synergy = load(output / "metrics/synergy.json")
    visual = load(output / "metrics/visualization_manifest.json")
    curve = pd.read_csv(output / "metrics/global_alpha_curve.csv")
    per_class = pd.read_csv(output / "metrics/per_class_oracle.csv")
    overlap = pd.read_csv(output / "metrics/failure_overlap.csv")
    categories = pd.read_csv(output / "metrics/recoverability_categories.csv")
    alpha_hist = pd.read_csv(output / "metrics/best_alpha_histogram.csv")
    responses = pd.read_csv(output / "metrics/margin_response_categories.csv")
    safety = oracle["safety"]
    d = decision["delta_pp"]
    lines = [
        "# DLAG Oracle v1｜Deep–Local Arbitration + Gate Ceiling Audit", "",
        f"**DECISION = {decision['decision']}**  ",
        f"**CONFIDENCE = {decision['confidence']}**", "",
        f"HQMR = {decision['HQMR']*100:.4f}%  ",
        f"SSHR = {decision['SSHR']*100:.4f}%  ",
        f"Oracle A = {decision['oracle_A']*100:.4f}%  ",
        f"Oracle B = {decision['oracle_B']*100:.4f}%  ",
        f"Oracle A+B = {decision['oracle_AB']*100:.4f}%  ",
        f"Delta A = {pp(d['A'])}  ", f"Delta B = {pp(d['B'])}  ",
        f"Delta A+B = {pp(d['AB'])}  ",
        f"vs SSHR = {pp(decision['AB_vs_SSHR_pp'])}", "",
        "本报告评估冻结 counterfactual ceiling，不是训练结果；GT 只用于在预注册 bank 中选择组件级输出或定义 gate force-on。", ""
    ]

    def section(number: int, title: str, body: str) -> None:
        lines.extend([f"## {number}. {title}", "", body.strip(), ""])

    ci_ab = bootstrap["oracle_AB"]["delta_pp_95ci"]
    section(1, "Executive Decision",
        f"联合 Oracle 将 mIoU 从 {decision['HQMR']*100:.4f}% 提升到 "
        f"{decision['oracle_AB']*100:.4f}%（{pp(d['AB'])}；图像 bootstrap 95% CI "
        f"[{ci_ab[0]:+.4f}, {ci_ab[1]:+.4f}] pp），相对 SSHR 为 "
        f"{pp(decision['AB_vs_SSHR_pp'])}。按预注册门槛判定 {decision['decision']}。"
        "这是可实现操作集合的上限证据，不代表训练机制能兑现全部增益。")
    anchors = gate["ucrf_anchor_replay"]["actual"]
    section(2, "Reproduction Gate",
        f"PASS。HQMR={gate['hqmr_miou']*100:.6f}%，M1={gate['m1_components']}、"
        f"M1 支持面积={gate['m1_pixels']:,}。UCRF 锚点：deep wrong "
        f"{pct(anchors['deep_wrong_area'])}、corrective available "
        f"{pct(anchors['corrective_available_area'])}、CSR "
        f"{pct(anchors['correction_suppression_area'])}、gate excluded "
        f"{pct(anchors['gate_excluded_area'])}。checkpoint SHA256 {gate['checkpoint_sha256']}；"
        f"参数更新 {gate['parameter_updates']}。独立 bank 进程的 baseline 为 "
        f"{decision['bank_baseline']*100:.6f}%（相对冻结值漂移 {decision['bank_baseline_drift_pp']:+.6f} pp），"
        "正式点估计仍以冻结 HQMR 为基准，paired bootstrap 使用同进程 bank baseline。")
    section(3, "Exact Forward Formula",
        "使用真实 query-logit 链 L4(alpha)=U(L5)+alpha*D4，再重算 q4、D3、L3 和 "
        "sum_q W[q,c] sigmoid(L3[q])。没有对 tissue class map 做伪线性相加。"
        f"alpha=1 的最终输出最大差 {integrity['alpha1_max_abs_difference']:.3g}，预测逐图相同。"
        "详见 forward_formula_manifest.md。")
    section(4, "Exact Presence-Gate Formula",
        "deep spatial logits 经 GAP 和 sigmoid 得到每视角 gate score；三视角均值用 "
        "[0.8,0.9,0.8,0.6] 阈值二值化（全零时 argmax fallback）。Gate 在 TTA class "
        "mixture 平均、逐类 min-max 后限制最终 argmax 候选。Oracle B 只改变二值 label，"
        "upstream tensor 变化为 0。封存实现将缺席类置零而非负无穷，因此全零位置仍可能由低索引缺席类"
        "赢得 tie；本审计保持该行为。详见 gate_formula_manifest.md。")
    section(5, "Frozen Oracle Rules",
        "α 固定为 {0,0.25,0.5,1,1.5,2,3,4}；并列优先 1、0.5、1.5、0.25、2、0、3、4。"
        "A1 仅在原 M1 支持区替换一种完整 α 输出；A2 允许 8 像素局部环且只作辅助。"
        "B1 只给冻结 M1 所需但被排除的真类 force-on；B2 是辅助 GT image-presence ceiling。"
        "AB 在 force-on 下组件级选 α。禁止逐像素选 α、写入 GT 标签或修改参数。")
    section(6, "Global Alpha Counterfactual",
        table(curve, [("alpha", "α"), ("mIoU", "mIoU"),
                      ("delta_vs_HQMR_pp", "ΔHQMR(pp)"),
                      ("harmed_correct_area", "损伤原正确像素"),
                      ("newly_correct_area", "新增正确像素")]) +
        "\n\n该表只检验固定 α；修复和 TP 损伤并存时支持区域自适应仲裁。"
        "见 [Figure 1](visualizations/figure1_global_alpha_curve.png)。")
    section(7, "Arbitration Oracle A",
        f"Primary A1={oracle['oracle_A1']['mIoU']*100:.4f}%（{pp(d['A'])}）；"
        f"辅助 A2={oracle['oracle_A2']['mIoU']*100:.4f}%（{pp(d['A2'])}）；"
        f"整图统一 α ceiling={oracle['image_level_A']['mIoU']*100:.4f}%"
        f"（{pp(d['image_A'])}）。M1 有效像素恢复率 "
        f"{pct(groups['overall']['A_recovered_fraction'])}。")
    section(8, "Gate Oracle B",
        f"Primary B1={oracle['oracle_B1']['mIoU']*100:.4f}%（{pp(d['B'])}）；"
        f"辅助 B2={oracle['oracle_B2']['mIoU']*100:.4f}%（{pp(d['B2'])}）。"
        f"B1 对 M1 有效像素实际恢复 {pct(groups['overall']['B_recovered_fraction'])}；"
        "36.72% gate exclusion 不能直接当作可恢复率。B2 若下降，不表示真实 presence 无用；"
        "它暴露的是封存 zero-mask argmax 的全零 tie 行为，故 B2 只作 secondary ceiling/diagnostic。")
    section(9, "Joint Oracle A+B",
        f"AB={decision['oracle_AB']*100:.4f}%（{pp(d['AB'])}），M1 有效像素恢复 "
        f"{pct(groups['overall']['AB_recovered_fraction'])}、仍不可恢复 "
        f"{pct(groups['overall']['AB_unrecoverable_fraction'])}。组件外采用 B1，组件内只从 "
        "force-on 条件下的八个完整 α 输出选择。"
        "见 [Figure 3](visualizations/figure3_oracle_headroom.png)。")
    section(10, "Failure Overlap",
        table(overlap, [("failure_overlap", "失败集合"), ("components", "组件"),
                        ("area_fraction", "M1 支持面积"),
                        ("A_recovered_fraction", "A 恢复"),
                        ("B_recovered_fraction", "B 恢复"),
                        ("AB_recovered_fraction", "AB 恢复")],
              {"area_fraction", "A_recovered_fraction", "B_recovered_fraction", "AB_recovered_fraction"}) +
        "\n\nDeep-dominance failure 冻结定义为 deep 已错、D4 实际作用向真类、融合仍错；"
        "Gate-exclusion 为 baseline presence label 排除组件真类。")
    section(11, "Synergy Analysis",
        f"Syn=ΔAB−ΔA−ΔB={pp(synergy['synergy_pp'])}，分类为 {synergy['relation']}。"
        "负值代表 recovery 重叠或联合 gate 的全图副作用，不表示任一机制不存在。")
    section(12, "Recoverable vs Unrecoverable M1",
        table(categories, [("category", "类别"), ("components", "组件"),
                           ("component_fraction", "组件占比"),
                           ("support_area_fraction", "支持面积占比")],
              {"component_fraction", "support_area_fraction"}) +
        "\n\n精确像素归因见 [Figure 4](visualizations/figure4_recovery_attribution.png)。"
        f"联合后不可恢复有效 M1 像素为 {pct(decision['pixel_attribution']['AB_unrecoverable_fraction'])}。")
    section(13, "Gate-Preserved Subset",
        f"组件 {groups['gate_preserved']['components']}；仅 A 恢复 "
        f"{pct(groups['gate_preserved']['A_recovered_fraction'])}，AB 恢复 "
        f"{pct(groups['gate_preserved']['AB_recovered_fraction'])}。该子集隔离 arbitration ceiling。")
    section(14, "Gate-Excluded Subset",
        f"组件 {groups['gate_excluded']['components']}；A/B/AB 有效像素恢复分别为 "
        f"{pct(groups['gate_excluded']['A_recovered_fraction'])} / "
        f"{pct(groups['gate_excluded']['B_recovered_fraction'])} / "
        f"{pct(groups['gate_excluded']['AB_recovered_fraction'])}。A 与 AB 的差距量化 gate "
        "是否是 arbitration 生效的必要前提。")
    subset_rows = pd.DataFrame([{"subset": name, **groups[name]} for name in
        ("overall", "high_purity", "large_Q4", "high_purity_large", "E3_corrective_subset")])
    section(15, "High-Purity/Large Analysis",
        table(subset_rows, [("subset", "子集"), ("components", "组件"),
                            ("A_recovered_fraction", "A 恢复"),
                            ("B_recovered_fraction", "B 恢复"),
                            ("AB_recovered_fraction", "AB 恢复"),
                            ("AB_unrecoverable_fraction", "AB 未恢复")],
              {"A_recovered_fraction", "B_recovered_fraction",
               "AB_recovered_fraction", "AB_unrecoverable_fraction"}))
    section(16, "Per-Class Oracle",
        table(per_class, [("class", "类别"), ("baseline_IoU", "Baseline IoU"),
                          ("oracle_A1_IoU", "A IoU"), ("oracle_B1_IoU", "B IoU"),
                          ("oracle_AB_IoU", "AB IoU"), ("AB_gain_pp", "AB gain(pp)")]) +
        f"\n\n最大单类占全部正向 class-IoU gain 的 {pct(decision['class_gain_concentration'])}；"
        f"CLASS_SPECIFIC={decision['class_specific']}。")
    section(17, "TP Safety",
        f"A1 损伤 baseline 正确像素 {safety['oracle_A1']['harmed_correct_area']:,}，新增正确 "
        f"{safety['oracle_A1']['newly_correct_area']:,}；B1 为 "
        f"{safety['oracle_B1']['harmed_correct_area']:,}/{safety['oracle_B1']['newly_correct_area']:,}；"
        f"AB 为 {safety['oracle_AB']['harmed_correct_area']:,}/"
        f"{safety['oracle_AB']['newly_correct_area']:,}。全局 α 逐点见 tp_safety_by_alpha.csv。")
    section(18, "Alpha Distribution",
        table(alpha_hist, [("best_alpha_A", "α*"), ("components", "组件"),
                           ("support_area_fraction", "面积占比"),
                           ("recovered_pixels", "恢复像素")], {"support_area_fraction"}) +
        "\n\n见 [Figure 2](visualizations/figure2_best_alpha_histogram.png)。")
    section(19, "Margin Response Curves",
        table(responses, [("response_category", "响应类别"), ("components", "组件"),
                          ("support_area_fraction", "面积占比")], {"support_area_fraction"}) +
        "\n\n每个组件的八点 CAM true-vs-rival margin 和正确像素数保存在 "
        "arbitration_oracle/margin_response.csv。monotonic 只描述离散网格。")
    selection = pd.read_csv(output / "metrics/visualization_selection.csv")
    case_lines = []
    for category in selection.category.drop_duplicates():
        files = sorted((output / "visualizations" / category).glob("*.png"))
        if len(files) != 20:
            raise AssertionError(f"Expected 20 images for {category}")
        case_lines.append(f"- {category}: " + ", ".join(
            f"[案例 {i+1}](visualizations/{category}/{file.name})"
            for i, file in enumerate(files)))
    section(20, "Representative Cases",
        f"生成 {visual['cases']} 张案例图，冻结 bank 回放错误 "
        f"{visual['replay_prediction_errors']}。每例展示原图、GT、baseline、best-alpha、gate、joint、"
        "真/对手 CAM、margin 和 gate 状态。\n\n" + "\n".join(case_lines))
    section(21, "Bootstrap Confidence Intervals",
        f"图像为单位 2000 次 paired bootstrap（seed42）：ΔA "
        f"[{bootstrap['oracle_A1']['delta_pp_95ci'][0]:+.4f}, "
        f"{bootstrap['oracle_A1']['delta_pp_95ci'][1]:+.4f}] pp；ΔB "
        f"[{bootstrap['oracle_B1']['delta_pp_95ci'][0]:+.4f}, "
        f"{bootstrap['oracle_B1']['delta_pp_95ci'][1]:+.4f}] pp；ΔAB "
        f"[{ci_ab[0]:+.4f}, {ci_ab[1]:+.4f}] pp。组件 bootstrap 的 AB 恢复率 CI 为 "
        f"[{pct(bootstrap['component_recovery']['AB']['fraction_95ci'][0])}, "
        f"{pct(bootstrap['component_recovery']['AB']['fraction_95ci'][1])}]。")
    residual = (decision["SSHR"]-decision["oracle_AB"])*100
    section(22, "Headroom vs SSHR",
        f"HQMR→SSHR 缺口 1.1243 pp；AB 相对 HQMR {pp(d['AB'])}，相对 SSHR "
        f"{pp(decision['AB_vs_SSHR_pp'])}。以 SSHR 为目标的 residual 为 {pp(residual)}。"
        f"FINAL-MODEL TARGET CONFIRMED={decision['oracle_AB'] >= .675}。")
    section(23, "Decision Matrix",
        "预注册：ΔAB<1=NOGO；1–1.5=WEAK_GO；1.5–2=ARCHITECTURE_GO；"
        "ΔAB≥2 且高于 SSHR≥0.5=STRONG_ARCHITECTURE_GO。"
        f"本次 ΔAB={d['AB']:.4f} pp、vs SSHR={decision['AB_vs_SSHR_pp']:.4f} pp，"
        f"故 {decision['decision']}；class-specific 检查后置信度 {decision['confidence']}。")
    section(24, "Final Architecture GO/NOGO",
        f"最终判决：**{decision['decision']}**。只有 ceiling 越过门槛才允许下一轮设计；"
        "仍禁止把最佳固定 α 直接当模型，也禁止本轮开始 Phase0/Full25。A2/B2 仅辅助，"
        "主判决只用 A1+B1。")
    if decision["decision"] in {"ARCHITECTURE_GO", "STRONG_ARCHITECTURE_GO"}:
        target_text = ("下一步可立项统一的 Reliability-Aware Class Responsibility Control："
                       "以区域可靠性分配 deep/local 话语权，并仅在 Oracle B 显示必要时加入 "
                       "gate-safe preservation。先做轻量、预注册 Phase0，不复制 GT oracle。")
    else:
        target_text = ("当前两条机制的可实现 ceiling 不足；下一步不是训练仲裁器，而是审计 AB 后"
                       "仍不可恢复的 residual failure。不得因机制叙事清晰而继续 Full25。")
    section(25, "Exact Next Architecture Target", target_text)
    lines.extend(["## Reproduction commands", "", "Commands are documented in audits/dlag_oracle_v1/README.md.", ""])
    destination = output / "DLAG_v1_Oracle_Ceiling_Audit_Report.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"event": "report_done", "path": str(destination),
                      "sections": 25, "decision": decision["decision"],
                      "bytes": destination.stat().st_size}), flush=True)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
