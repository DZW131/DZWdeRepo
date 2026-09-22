"""Generate the fixed 21-section RISA-v1 Phase-0 final report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.risa_v1_phase0.common import CLASSES


def percent(value) -> str:
    return "—" if value is None else f"{100 * value:.3f}%"


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    frame = frame[columns]
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---" if frame[column].dtype == object else "---:" for column in columns) + "|"
    rows = []
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join((header, divider, *rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.output / "artifacts/risa_v1_phase0"
    result = json.loads((artifact / "final_result.json").read_text())
    baseline = json.loads((artifact / "baseline_replay.json").read_text())
    training = json.loads((artifact / "training_runtime.json").read_text())
    state = json.loads((artifact / "risa_v1_research_state.json").read_text())
    jev = json.loads((artifact / "jev_risa_v1_sidecar.json").read_text())
    segmentation = pd.read_csv(artifact / "all_variant_metrics.csv")
    hard = pd.read_csv(artifact / "hardm1_identity_audit.csv")
    preservation = pd.read_csv(artifact / "correct_region_preservation.csv")
    per_class = pd.read_csv(artifact / "per_class_identity_metrics.csv")
    epochs = pd.read_csv(artifact / "training_epoch_metrics.csv")
    decision = result["RISA_V1_PHASE0_DECISION"]
    primary = result["PRIMARY_VARIANT"]
    evidence = result["decision_evidence"]
    sections = []
    def add(title: str, body: str) -> None:
        sections.append((title, body))
    add("Executive Summary", f"**RISA_V1_PHASE0_DECISION = {decision}**\n\n**PRIMARY_MECHANISM_CONCLUSION = {result['PRIMARY_MECHANISM_CONCLUSION']}**\n\n固定 E5 主候选为 `{primary}`：Hard-M1 HIRR={percent(evidence['HIRR'])}，CRP={percent(evidence['CRP'])}，mIoU 相对 B0={evidence['mIoU_gain_pp']:+.4f} pp，true-rival margin Δ={evidence['margin_delta']:+.6f}。")
    add("Research Question", "在 Backbone、CCRA、HQMR 及其空间 responsibility 完全冻结时，仅用图像级类别标签训练的动态 region identity assignment，能否恢复 spatially coherent but semantically wrong regions，并保持原正确区域？")
    add("Frozen Baseline Replay", f"重放状态 `{baseline['BASELINE_REPLAY']}`；3418 张验证图；mIoU={100*baseline['observed']['mIoU']:.4f}，mDice={100*baseline['observed']['mDice']:.4f}，相对冻结记录偏差={baseline['delta_mIoU_pp']:+.6f} pp。冻结 Hard-M1={baseline['hard_m1_components']}，M1={baseline['m1_components']}；checkpoint SHA256 `{baseline['checkpoint_sha256']}`。")
    add("Implementation Audit", f"新增参数 {training['risa_parameters']:,}（基线的 {training['additional_percent']:.4f}%）；RISA 内部只接收 detached F3/F4/F5 与 detached HQMR L3 responsibility，不访问 GT、pseudo-mask、VLM 或跨图 prototype。checkpoint replay max-abs={training['checkpoint_replay_max_abs']:.3e}。")
    isolation = result["isolation"]
    max_delta = max(value["max_abs"] for value in isolation["tensor_audit"].values())
    add("Gradient / Causal Isolation Audit", f"`ISOLATION_PASS={isolation['ISOLATION_PASS']}`。逐 tensor 检查 CCRA K/V、final responsibility、HQMR query0/L5/q5/L4/q4/L3，最大训练前后漂移={max_delta:.3e}；冻结参数改变数={len(isolation['changed_frozen_parameters'])}。")
    add("RISA Architecture", "F3/F4/F5 分别经 1×1 Conv + GroupNorm + GELU 投影至 128 维，在 F3 尺度融合。RISA-Mean 以 HQMR responsibility 加权池化；Full RISA 以 top-1/top-2 tissue embedding 构造 rival direction，在固定 Top-K=20% token 上读出 hard descriptor，并以固定 δ=0.15、τg=0.05 连续门控融合。最终 πq,c 是逐图、逐 region、逐 query 动态 softmax。")
    add("Training Configuration", f"Seed42，AdamW(lr=3e-4, weight decay=1e-4)，clip=1.0，无 scheduler，BF16，effective batch=20。Epoch1 mean-only；Epoch2–5 full RISA；固定 E5，无 early stopping、无验证集选择。总计 {training['optimizer_steps']} optimizer steps，耗时 {training['train_seconds']/60:.2f} 分钟，峰值显存 {training['peak_vram_gib']:.3f} GiB。")
    add("Identity Learning Behaviour", table(epochs, ["epoch", "hard_refinement", "L_MIL", "L_rank", "L_hard", "mean_identity_entropy", "mean_top1_top2_margin", "mean_hard_gate", "presence_AUROC_macro"]) + f"\n\nE5 class histogram={result['class_collapse']['class_histogram']}；`CLASS_COLLAPSE={result['class_collapse']['CLASS_COLLAPSE']}`。")
    add("Hard-M1 Identity Recovery", table(hard, ["variant", "components", "HIRR", "HIRR_area_weighted", "margin_mean", "margin_delta_mean"]) + f"\n\n冻结 Hard-M1 总数={result['hard_m1_components']}；主候选不由验证指标选 epoch，只在固定 E5 的预注册 variants 中按最终 mIoU 描述。")
    margin = result["identity_audit"][primary]
    add("True-vs-Rival Margin", f"`{primary}` margin：{margin['margin']}。相对 frozen sequential-L5 baseline 的 Δmargin：{margin['margin_delta']}。预注册核心方向要求 Δmargin>0。")
    add("Correct-region Preservation", table(preservation, ["variant", "components", "component_rate", "pixel_area_weighted_rate"]) + "\n\nCRP 直接在冻结 baseline-correct components 上计算；没有按 RISA 输出重新定义区域。")
    add("Per-class Analysis", table(per_class, ["variant", "class_name", "HIRR", "margin_delta_mean", "CRP", "IoU", "IoU_delta_pp"]) + f"\n\n主候选 margin 改善覆盖 {result['class_breadth_positive_margin'][primary]}/4 类。")
    add("R1 vs R2 Hard-Rival Ablation", f"同一 E5 checkpoint 的结果：{result['hard_rival_ablation']}。若 R2≈R1，则 hard-rival refiner 不应作为后续结构保留。")
    add("R2 vs R3 Presence Ablation", f"R3 在 class-wise CAM normalization 后以 soft P_c 替代原 hard-zero gate，避免标量在 normalization 中被抵消。结果：{result['soft_presence_ablation']}。")
    dependency = result["region_dependency"]
    add("Region Dependency Counterfactual", f"`REGION_DEPENDENCY_PASS={dependency['REGION_DEPENDENCY_PASS']}`；seed={dependency['seed']}，固定 {dependency['images']} 张图。真实/循环平移/空间置换 true-rival margin={dependency['true_rival_margin']}；query top-1 change={dependency['query_top1_change_fraction']}。")
    add("Segmentation Performance", table(segmentation, ["variant", "mIoU", "mDice", "IoU_C0", "IoU_C1", "IoU_C2", "IoU_C3"]) + "\n\nB0/R1/R2/R3 均完整报告；R1/R2 保留原 hard presence，R3 使用 soft presence。")
    cost = result["parameter_runtime"]
    add("Parameter / FLOPs / Runtime", f"Baseline params={cost['baseline_parameters']:,}；RISA params={cost['risa_parameters']:,}；additional={cost['additional_percent']:.4f}%；lightweight pass={cost['lightweight_target_pass']}。Baseline FLOPs/view={cost['baseline_flops_per_view']:,}；RISA extra={cost['risa_extra_flops_per_view']:,}（{cost['extra_flops_percent']:.4f}%）。单 view baseline/RISA 时间={cost['baseline_seconds_per_image_per_view']:.5f}/{cost['risa_seconds_per_image_per_view']:.5f}s；峰值显存={cost['baseline_peak_vram_gib']:.3f}/{cost['risa_peak_vram_gib']:.3f} GiB。")
    add("Failure Visualizations", f"四组可视化已生成：{result['visualizations']}。目录分别覆盖 recovered Hard-M1、harmed correct regions、hard-rival token maps 与 query identity probabilities。若某组不足 20，表示固定数据中满足该语义条件的去重图像不足 20。")
    add("Mechanism Conclusion", f"主候选 `{primary}` 的决策证据为 {evidence}。Class-collapse、causal isolation 与 region-dependency 三项共同决定该现象能否解释为 region-specific identity learning，而不是 global classifier 或空间路径漂移。")
    add("Final GO / STRONG_GO / NOGO", f"预注册 Python rule 的最终输出是 **RISA_V1_PHASE0 = {decision}**。Failure attribution={result['failure_attribution']}。Jev sidecar 状态 `{jev['JEV_STATUS']}`，没有改写 Python 决策。")
    if decision == "STRONG_GO":
        recommendation = "进入 RISA-v1 Full25 Final Validation，不修改结构或超参数。"
    elif decision == "GO":
        recommendation = "进入一次有限 Full25 验证，不做结构修改。"
    else:
        recommendation = "先依据 failure attribution 审查 identity feature 或 region granularity；不延长 epoch、不 sweep temperature/Top-K/loss，不回退到 PDSR/VPCA/CIRV/RACC。"
    add("Recommended Next Step", recommendation)
    report = "# RISA-v1 Phase-0 Final Report\n\n" + "\n\n".join(
        f"## {index}. {title}\n\n{body}" for index, (title, body) in enumerate(sections, 1)
    ) + f"\n\n---\n\nRISA_V1_PHASE0_DECISION = {decision}\n\nPRIMARY_MECHANISM_CONCLUSION = {result['PRIMARY_MECHANISM_CONCLUSION']}\n"
    reports = args.output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "RISA_v1_Phase0_Final_Report.md").write_text(report, encoding="utf-8")
    (artifact / "RISA_v1_Phase0_Final_Report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"event": "RISA_REPORT_COMPLETE", "decision": decision,
                      "report": str((reports / 'RISA_v1_Phase0_Final_Report.md').resolve())}))


if __name__ == "__main__":
    main()
