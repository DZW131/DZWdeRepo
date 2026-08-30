"""Offline sufficient-statistic analysis; never forwards or changes a model."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.rddr_phase2b0_common import (
    GROUPS, PAIR_GROUPS, VARIANTS, ESTIMATORS, FIELDS, BINS,
    nanmean, binary_metrics, cm_metrics, bootstrap_means, ci_row,
    sha256, write_csv, write_json,
)


def ratio(n, d):
    n, d = np.asarray(n, dtype=float), np.asarray(d, dtype=float)
    return np.divide(n, d, out=np.full(np.broadcast_shapes(n.shape, d.shape), np.nan), where=d > 0)


def quantiles(hist, upper):
    total = hist.sum()
    if not total:
        return {f"p{x:02}": np.nan for x in (5, 25, 50, 75, 95)}
    cum = np.cumsum(hist)
    return {f"p{x:02}": (min(np.searchsorted(cum, x/100*(total-1), side="right"), BINS-1)+.5)*upper/BINS
            for x in (5, 25, 50, 75, 95)}


def table(rows, columns):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.6f}" if np.isfinite(x) else "NA"
        return str(x)
    return "| " + " | ".join(label for _, label in columns) + " |\n|" + "---|"*len(columns) + "\n" + "\n".join(
        "| " + " | ".join(fmt(r.get(key, "")) for key, _ in columns) + " |" for r in rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tick = time.perf_counter()
    source, out = Path(args.input), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    runtime = json.loads((source / "rddr_phase2b0_runtime.json").read_text(encoding="utf-8"))
    manifest = json.loads((source / "rddr_phase2b0_population_manifest.json").read_text(encoding="utf-8"))
    assert runtime["images"] == 3418 and not runtime["smoke"]
    assert runtime["frozen_q_feature_max_abs_difference"] == 0 and runtime["unchanged_model_state"]
    data = np.load(source / "rddr_phase2b0_sufficient_statistics.npz")
    names, sums, counts = data["names"], data["sums"], data["counts"]
    means = ratio(sums, counts)
    cm, repair = data["cm"], data["repair"]
    n = len(names)
    gi = {g: i for i, g in enumerate(GROUPS)}
    group_rows = []
    for gidx, group in enumerate(GROUPS):
        for vi, variant in enumerate(VARIANTS):
            row = dict(group=group, variant=variant, targets=int(data["population_counts"][:, gidx].sum()))
            for fi, field in enumerate(FIELDS):
                row[field+"_pixel_mean"] = float(ratio(sums[:, gidx, vi, fi].sum(), counts[:, gidx, vi, fi].sum()))
                row[field+"_image_mean"] = float(nanmean(means[:, gidx, vi, fi]))
                row[field+"_eligible_images"] = int(np.isfinite(means[:, gidx, vi, fi]).sum())
                row.update({field+"_"+k: v for k, v in quantiles(data["value_hist"][gidx, vi, fi], 1 if fi == 0 else 224).items()})
            row["purity_gain_image_mean"] = float(nanmean(means[:, gidx, vi, 0]-means[:, gidx, 0, 0]))
            row["purity_gain_pixel_mean"] = row["purity_pixel_mean"] - float(ratio(sums[:, gidx, 0, 0].sum(), counts[:, gidx, 0, 0].sum()))
            group_rows.append(row)
    pair_rows = []
    for pgi, group in enumerate(PAIR_GROUPS):
        for vi, variant in enumerate(VARIANTS):
            vals = data["pair_values"][:, pgi, vi]
            row = dict(group=group, variant=variant, **binary_metrics(data["pair_hist"][pgi, vi]),
                       image_balanced_auroc=float(nanmean(vals[:, 0])),
                       image_balanced_auprc=float(nanmean(vals[:, 1])),
                       auroc_eligible_images=int(np.isfinite(vals[:, 0]).sum()),
                       auroc_excluded_images=int((~np.isfinite(vals[:, 0])).sum()))
            pair_rows.append(row)
    estimator_rows, repair_rows, echo_rows = [], [], []
    for gidx, group in enumerate(GROUPS):
        count = int(data["population_counts"][:, gidx].sum())
        for ei, estimator in enumerate(ESTIMATORS):
            total_cm = cm[:, gidx, ei].sum(0)
            m = cm_metrics(total_cm)
            proper = data["proper"][:, gidx, ei].sum(0)
            estimator_rows.append(dict(group=group, estimator=estimator, accuracy=float(m["accuracy"]),
                                       miou=float(m["miou"]), dice=float(m["dice"]),
                                       **{f"iou_class{k}": v for k, v in enumerate(m["class_iou"])},
                                       nll=float(ratio(proper[0], proper[2])), brier=float(ratio(proper[1], proper[2])),
                                       targets=int(total_cm.sum()), coverage=float(ratio(total_cm.sum(), count))))
            r = repair[:, gidx, ei].sum(0)
            repair_rows.append(dict(group=group, estimator=estimator, repair_count=int(r[0]), harm_count=int(r[1]),
                                    targets=int(r[2]), repair=float(ratio(r[0], r[2])), harm=float(ratio(r[1], r[2])),
                                    net_repair=float(ratio(r[0]-r[1], r[2]))))
        e = data["echo"][:, gidx].sum(0)
        echo_rows.append(dict(group=group, echo_count=int(e[0]), targets=int(e[1]), echo_fraction=float(ratio(e[0], e[1])),
                              non_echo_count=int(e[2]), srsc_correct_deep_wrong_count=int(e[3]),
                              srsc_wrong_deep_correct_count=int(e[4]),
                              srsc_correct_deep_wrong_given_non_echo=float(ratio(e[3], e[2])),
                              srsc_wrong_deep_correct_given_non_echo=float(ratio(e[4], e[2]))))
    # The same seed and draw order is used for ALL primary paired bootstrap contrasts.
    corrected = means[:, gi["Corrected_by_CH"], 3, 0]
    harmed = means[:, gi["Harmed_by_CH"], 3, 0]
    mean_columns = np.column_stack([
        data["pair_values"][:, 0, 3, 0], means[:, 0, 3, 0]-means[:, 0, 0, 0],
        corrected-harmed,
        means[:, gi["Harmed_by_CH"], 3, 0]-means[:, gi["Harmed_by_CH"], 0, 0],
        means[:, 0, 3, 2], data["target_auc"],
    ])
    mean_keys = ["SRSC_image_balanced_pair_AUROC", "SRSC_minus_U_purity",
                 "SRSC_Corrected_minus_Harmed_purity", "Harmed_SRSC_minus_U_purity",
                 "SRSC_mean_N_eff", "target_AUROC_purity", "target_AUROC_purity_gain", "target_AUROC_negative_wrong_mass"]
    boot_means = bootstrap_means(mean_columns)
    boot_rows = [dict(ci_row(k, nanmean(mean_columns[:, i]), boot_means[:, i]),
                      eligible_images=int(np.isfinite(mean_columns[:, i]).sum()), aggregation="image_balanced")
                 for i, k in enumerate(mean_keys)]
    boot_curves = {k: boot_means[:, i] for i, k in enumerate(mean_keys)}
    rng = np.random.default_rng(42)
    acc_boot, iou_boot, top_boot = [], [], []
    top = repair[:, gi["Top20"], :4]
    use_cm = cm[:, 0, [0, 3]]
    for start in range(0, 10000, 50):
        idx = rng.integers(0, n, (min(50, 10000-start), n), dtype=np.int32)
        sample = use_cm[idx].sum(1)
        metrics = cm_metrics(sample)
        acc_boot.extend(metrics["accuracy"][:, 1]-metrics["accuracy"][:, 0])
        iou_boot.extend(metrics["miou"][:, 1]-metrics["miou"][:, 0])
        sampled_top = top[idx].sum(1)
        net = ratio(sampled_top[:, :, 0]-sampled_top[:, :, 1], sampled_top[:, :, 2])
        top_boot.extend(net[:, 3]-net[:, 0])
    total_metrics = cm_metrics(use_cm.sum(0))
    total_top = top.sum(0)
    total_net = ratio(total_top[:, 0]-total_top[:, 1], total_top[:, 2])
    for key, observed, values in (
        ("SRSC_minus_U_neighbor_accuracy", total_metrics["accuracy"][1]-total_metrics["accuracy"][0], acc_boot),
        ("SRSC_minus_U_neighbor_mIoU", total_metrics["miou"][1]-total_metrics["miou"][0], iou_boot),
        ("Top20_SRSC_minus_U_NetRepair", total_net[3]-total_net[0], top_boot),
    ):
        boot_rows.append(dict(ci_row(key, observed, values), eligible_images=n, aggregation="pooled_recomputed_per_image_resample"))
        boot_curves[key] = np.asarray(values)
    ci = {r["metric"]: r for r in boot_rows}
    a = ci["SRSC_image_balanced_pair_AUROC"]
    b = ci["SRSC_minus_U_purity"]
    c = ci["SRSC_Corrected_minus_Harmed_purity"]
    da, di = ci["SRSC_minus_U_neighbor_accuracy"], ci["SRSC_minus_U_neighbor_mIoU"]
    gates = {
        "A": a["observed"] >= .65 and a["ci95_low"] > .50,
        "B": b["observed"] >= .03 and b["observed"] > 0 and b["ci95_low"] > 0 and ci["SRSC_mean_N_eff"]["observed"] >= 5,
        "C": c["observed"] > 0 and c["ci95_low"] > 0 and ci["Harmed_SRSC_minus_U_purity"]["observed"] > 0,
        "D": da["observed"] > 0 and di["observed"] > 0 and (da["ci95_low"] > 0 or di["ci95_low"] > 0)
             and ci["Top20_SRSC_minus_U_NetRepair"]["observed"] > 0,
    }
    decision = ("RDDR_PHASE2B0_NOGO" if not gates["A"] or not gates["B"] else
                "RELATION_SIGNAL_NOT_CH_OUTCOME_SPECIFIC" if not gates["C"] else
                "RELATION_EXISTS_NO_PROPAGATION_UTILITY" if not gates["D"] else "RDDR_PHASE2B0_GO")
    target_rows = []
    for k, name in enumerate(("purity", "purity_gain", "negative_wrong_mass")):
        target_rows.append(dict(score=name, positive="Corrected_by_CH", **binary_metrics(data["target_hist"][k]),
                               **{x: y for x, y in ci["target_AUROC_"+name].items() if x != "metric"}))
    summary = dict(decision=decision, gates=gates, primary_comparison="SRSC_vs_U", ci=ci, images=n,
                   checkpoint_sha256=runtime["checkpoint_sha256"], extraction_commit=runtime["commit"],
                   analysis_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   sufficient_statistics_sha256=sha256(source / "rddr_phase2b0_sufficient_statistics.npz"),
                   paired_corrected_harmed_images=int(np.isfinite(corrected-harmed).sum()),
                   target_discrimination=target_rows, analysis_seconds=time.perf_counter()-tick,
                   oracle_purity=float(ratio(data["oracle_diag"][:, 0].sum(), data["oracle_diag"][:, 1].sum())),
                   oracle_valid_targets=int(data["oracle_diag"][:, 1].sum()),
                   oracle_mean_same_class_neighbors=float(ratio(data["oracle_diag"][:, 2].sum(), data["oracle_diag"][:, 1].sum())),
                   gate_metric_scale="fractions; multiply differences by100 for percentage points",
                   no_training=True, no_test=True, stop_after_report=True)
    out.mkdir(parents=True)
    write_json(out / "rddr_phase2b0_summary.json", summary)
    write_json(out / "rddr_phase2b0_runtime.json", dict(runtime, analysis_seconds=summary["analysis_seconds"]))
    write_json(out / "rddr_phase2b0_population_manifest.json", manifest)
    write_csv(out / "rddr_phase2b0_pair_metrics.csv", pair_rows)
    write_csv(out / "rddr_phase2b0_purity.csv", [r for r in group_rows if r["group"] == "all"])
    write_csv(out / "rddr_phase2b0_group_analysis.csv", group_rows)
    write_csv(out / "rddr_phase2b0_conflict_quintiles.csv", [r for r in group_rows if r["group"].startswith("Q")])
    write_csv(out / "rddr_phase2b0_effective_neighbors.csv", group_rows)
    write_csv(out / "rddr_phase2b0_neighbor_estimator.csv", estimator_rows)
    write_csv(out / "rddr_phase2b0_top20_repair.csv", [r for r in repair_rows if r["group"] == "Top20"])
    write_csv(out / "rddr_phase2b0_all_group_repair.csv", repair_rows)
    write_csv(out / "rddr_phase2b0_deep_echo.csv", echo_rows)
    write_csv(out / "rddr_phase2b0_target_discrimination.csv", target_rows)
    write_csv(out / "rddr_phase2b0_bootstrap.csv", boot_rows)
    write_csv(out / "rddr_phase2b0_bootstrap_replicates.csv", [dict(replicate=i, **{k: v[i] for k, v in boot_curves.items()}) for i in range(10000)])
    # Join subgroup relation, estimator, repair and echo evidence without ambiguity.
    joined = []
    for g in GROUPS:
        for vi, v in enumerate(VARIANTS):
            row = dict(next(r for r in group_rows if r["group"] == g and r["variant"] == v))
            row["uniform_purity_image_mean"] = float(nanmean(means[:, gi[g], 0, 0]))
            pair = next((r for r in pair_rows if r["group"] == g and r["variant"] == v), {})
            row.update({"pair_"+k: x for k, x in pair.items() if k not in ("group", "variant")})
            est = next(r for r in estimator_rows if r["group"] == g and r["estimator"] == v)
            row.update({"neighbor_"+k: x for k, x in est.items() if k not in ("group", "estimator")})
            rep = next(r for r in repair_rows if r["group"] == g and r["estimator"] == v)
            row["net_repair_vs_raw"] = rep["net_repair"]
            if v == "SRSC":
                row.update(next(r for r in echo_rows if r["group"] == g))
            joined.append(row)
    write_csv(out / "rddr_phase2b0_boundary_interior.csv", [r for r in joined if r["group"] in ("boundary", "interior")])
    write_csv(out / "rddr_phase2b0_per_class.csv", [r for r in joined if r["group"].startswith("class")])
    deep_rows = [r for r in joined if r["group"] in ("Deep_Correct", "Deep_Wrong", "Top20_Deep_Correct", "Top20_Deep_Wrong")]
    for row in deep_rows:
        stratum = row["group"].removeprefix("Top20_")
        rep = next(r for r in repair_rows if r["group"] == "Top20_"+stratum and r["estimator"] == row["variant"])
        row["top20_net_repair"] = rep["net_repair"]
        row["top20_count"] = rep["targets"]
    write_csv(out / "rddr_phase2b0_deep_strata.csv", deep_rows)
    per_image = []
    for i, name in enumerate(names):
        row = dict(image_id=name)
        for gidx, group in enumerate(GROUPS):
            row[group+"_targets"] = int(data["population_counts"][i, gidx])
            row[group+"_U_purity"] = means[i, gidx, 0, 0]
            row[group+"_SRSC_purity"] = means[i, gidx, 3, 0]
            row[group+"_SRSC_N_eff"] = means[i, gidx, 3, 2]
        for vi, v in enumerate(VARIANTS):
            row[v+"_pair_auroc"] = data["pair_values"][i, 0, vi, 0]
            row[v+"_pair_auprc"] = data["pair_values"][i, 0, vi, 1]
            row[v+"_pair_positive"] = int(data["pair_values"][i, 0, vi, 2])
            row[v+"_pair_negative"] = int(data["pair_values"][i, 0, vi, 3])
        for ei, e in enumerate(ESTIMATORS):
            row[e+"_confusion"] = json.dumps(cm[i, 0, ei].tolist(), separators=(",", ":"))
            row[e+"_top20_repair"] = int(repair[i, gi["Top20"], ei, 0])
            row[e+"_top20_harm"] = int(repair[i, gi["Top20"], ei, 1])
        for k, key in enumerate(mean_keys):
            row[key] = mean_columns[i, k]
        per_image.append(row)
    write_csv(out / "rddr_phase2b0_per_image.csv", per_image)
    # Explicit reporting, no score modifications or conditional experiment launches.
    pcols = [("variant", "Relation"), ("image_balanced_auroc", "Image AUROC"), ("auroc", "Pooled AUROC"),
             ("auprc", "AP"), ("prevalence", "Prevalence"), ("auprc_over_prevalence", "AP/prevalence"),
             ("auroc_eligible_images", "Eligible images")]
    gcols = [("group", "Group"), ("variant", "Relation"), ("purity_image_mean", "Image purity"),
             ("purity_pixel_mean", "Pixel purity"), ("purity_gain_image_mean", "Gain vs U"),
             ("neff_image_mean", "Image N_eff"), ("same_mass_pixel_mean", "Same mass"), ("wrong_mass_pixel_mean", "Wrong mass")]
    ecols = [("estimator", "Estimator"), ("accuracy", "Accuracy"), ("miou", "mIoU"), ("dice", "Dice"),
             ("nll", "NLL"), ("brier", "Brier"), ("coverage", "Coverage")]
    all_group = [r for r in group_rows if r["group"] == "all"]
    with (source / "rddr_phase2b0_histogram_validation.csv").open(newline="") as f:
        exact_rows = list(csv.DictReader(f))
    max_auc_error = max(float(r["abs_auroc_error"]) for r in exact_rows if r["abs_auroc_error"] != "nan")
    max_ap_error = max(float(r["abs_auprc_error"]) for r in exact_rows if r["abs_auprc_error"] != "nan")
    sections = ["# RDDR Phase-2B0 Reliable Relation Feasibility Audit\n",
        f"**最终判定：`{decision}`。** 本轮是 C0 冻结权重、BCSS validation-only、零训练、零新增参数的关系审计。\n"
        f"Gate A/B/C/D：{gates}。下文除明确注明 pp 外均用 0–1 比例；差值乘100才是百分点。\n",
        "## 1. Provenance / frozen assets\n\n"
        f"- Pure A0: `{runtime['a0_commit']}`\n- Extraction commit: `{runtime['commit']}`\n"
        f"- Analysis commit: `{summary['analysis_commit']}`\n- Checkpoint: `{runtime['checkpoint']}`\n"
        f"- Checkpoint SHA256: `{runtime['checkpoint_sha256']}`\n"
        f"- Statistics SHA256: `{summary['sufficient_statistics_sha256']}`\n"
        "- 只新增 tools/tests/docs/audit；官方网络、预处理、训练和推理源文件保持 A0 原样。\n",
        "## 2. Exact commands / environment\n\n```bash\n" + runtime["command"] + "\n"
        + " ".join(sys.argv) + "\n```\n\n"
        f"Python {runtime['python']}; torch {runtime['torch']}; NumPy {runtime['numpy']}; {runtime['gpu']}。"
        "batch1, BF16 forward, FP32 probability/relation；无TTA。"
        f"benchmark={runtime['benchmark']}, matmul={runtime['matmul_precision']}, conv={runtime['conv_precision']}。\n",
        "## 3. Tensor / forward contract\n\n"
        "实际运行未经修改的 `Net.forward`，用只读 hook 取得 HFRM28_1 输入/输出。"
        "F28_raw/F28_rect=[1,512,28,28]，Ddeep=[1,4096,28,28]。"
        "Ls=ic1(F28_raw)，Ld=fc8(Ddeep)，未做 ReLU/CAM normalize。dropout 在 eval 中关闭。"
        "使用 softmax(logits.float())，温度1。所有参数 requires_grad=False，无 optimizer，无 backward。\n",
        "## 4. GT / frozen population projection\n\n"
        "224×224 GT/历史人口 mask nearest 到28×28。类别0–3计入指标，4/255只从指标中排除。"
        "预测/关系权重不读取GT，也不依GT删除背景来源。历史 mask 已逐文件 SHA 校验并逐图对照 Phase0 CSV 人数。"
        "历史原始逐像素文件未保留；本次复用的是 Phase2A 在原代码/权重/backend 下重建且逐图人数 exact 的 immutable cache，"
        "不能声称曾与不存在的历史像素哈希比较。\n\n"
        + table([dict(group=k, full=v, projected=manifest["projected_counts"][k]) for k,v in manifest["full_resolution_counts"].items()],
                [("group","Group"),("full","224-grid count"),("projected","28-grid count")])
        + f"\n本轮重新提取 q 与缓存原始算术最大误差={runtime['frozen_q_feature_max_abs_difference']}；"
        f"Torch/NumPy ln2 除法舍入差异最大={runtime['torch_vs_numpy_q_division_max_abs_difference']:.9g}，单独记录不用于重分组。\n",
        "## 5. Neighborhood / eligibility contract\n\n"
        "15×15、radius7，只保留图内邻居，排除self。角落63、中部224个邻居。"
        "Propagation使用全部合法来源；pair/purity只评价FG–FG。实际 Mass/N_eff 分母绝不GT过滤。"
        "无前景来源时 purity 未定义，而非设0或完美；无同类来源时oracle未定义且不回退。\n",
        "## 6. Frozen U / SR / SC / SRSC formulas\n\n"
        "```text\nq_j=clip(JS(ps_j,pd_j)/ln2,0,1); r_j=1-q_j\n"
        "c_ij=clip(1-JS(pd_i,ps_j)/ln2,0,1)\nU=1; SR=r_j; SC=c_ij; SRSC=r_j*c_ij\n"
        "p_tilde_i=sum_j(A_ij*ps_j)/(sum_j A_ij+1e-8)\n```\n\n"
        "自然对数、epsilon=1e-8，epsilon位置与Phase0一致。无receiver reliability、learned层、其他score或搜索。"
        "Primary始终SRSC vs U，SR/SC只解释。U是均匀语义概率聚合对照，**不等于训练后CH卷积核的特征输出**。\n",
        "## 7. Counts / streaming / resources\n\n"
        f"3418图；FG target={runtime['target_count']:,}；FG–FG directed pair={runtime['foreground_pair_count']:,}；"
        f"FG target的实际空间边={runtime['actual_relation_count']:,}；无FG邻居target={runtime['foreground_targets_without_foreground_source']}。\n\n"
        f"提取全程 {runtime['total_seconds']:.1f}s；forward+relations {runtime['forward_relation_seconds']:.1f}s；"
        f"统计 {runtime['statistics_seconds']:.1f}s；bootstrap/汇总 {summary['analysis_seconds']:.1f}s。"
        f"CUDA peak allocated={runtime['peak_cuda_allocated_bytes']/2**30:.3f}GiB，reserved={runtime['peak_cuda_reserved_bytes']/2**30:.3f}GiB。"
        "仅逐图pair临时张量、累计直方图和逐图充分统计量，无全数据集pair缓存、无新模型checkpoint。\n",
        "## 8. Pair AUROC / AUPRC\n\n" + table([r for r in pair_rows if r["group"] == "all"], pcols)
        + "\nAUPRC按非插值AP；同分数计tie。4096固定bin，16张等间隔确定性图像全部四配置对照exact排序："
        f"最大AUROC误差={max_auc_error:.8f}，最大AP误差={max_ap_error:.8f}。详见 histogram_validation.csv。\n",
        "## 9. Image-balanced AUROC + CI\n\n" + table([a], [("observed","AUROC"),("ci95_low","95% low"),("ci95_high","95% high"),("eligible_images","Eligible images")])
        + "\n缺少正/负pair的图像AUROC为NA，未人为赋0.5；bootstrap对图像采样后忽略该指标NA。\n",
        "## 10. Weighted neighbor purity\n\n" + table(all_group, gcols)
        + "\nFG–FG purity完整p25/median/p75见 purity.csv；其百分位采用4096bin，绝对量化误差<=0.5/4096。"
        + f"主image-balanced增益={100*b['observed']:+.4f}pp，95%CI=[{100*b['ci95_low']:+.4f},{100*b['ci95_high']:+.4f}]pp。\n",
        "## 11. Effective neighbors / mass\n\n" + table([r for r in group_rows if r["variant"] == "SRSC" and r["group"] in ("all","Top20","Bottom80","Corrected_by_CH","Harmed_by_CH")],
        [("group","Group"),("mass_pixel_mean","Mass mean"),("mass_p05","Mass p05"),("mass_p50","Mass p50"),("mass_p95","Mass p95"),
         ("neff_pixel_mean","N_eff mean"),("neff_p05","N_eff p05"),("neff_p50","N_eff p50"),("neff_p95","N_eff p95")])
        + "\n此表包括GT背景来源的实际传播图；FG same/wrong mass另报，不可混用分母。分位数量化误差<=224/(2×4096)=0.027344。\n",
        "## 12. Corrected / Harmed mechanism\n\n" + table([r for r in group_rows if r["variant"] in ("U","SRSC") and r["group"] in ("Corrected_by_CH","Harmed_by_CH")], gcols)
        + f"\n主paired差只用同时包含两组的 {summary['paired_corrected_harmed_images']} 张图。"
        f"Corrected-Harmed={c['observed']:.6f}，95%CI=[{c['ci95_low']:.6f},{c['ci95_high']:.6f}]。"
        "上表各组全体image mean与这个配对样本总体不同，不可直接相减替代主检验。"
        "历史by-CH命名实际为 raw→完整HFRM（含GSR）；这里是关联证据，不能证明CH单独因果。\n\n"
        + table(target_rows, [("score","Corrected-positive score"),("auroc","Pooled AUC"),("observed","Image AUC"),("ci95_low","95% low"),("ci95_high","95% high"),("eligible_images","Images")])
        + "\nwrong mass在预注册时即取负号；gain线性映射到[0,1]只为固定hist，并未改变排序。\n",
        "## 13. Frozen Top20 / Bottom80\n\n" + table([r for r in group_rows if r["group"] in ("Top20","Bottom80")], gcols)
        + "\nTop20来自历史224-grid阈值mask，nearest后不强制正好20%，没有按本轮score重新选择。\n",
        "## 14. Conflict quintiles\n\n"
        + f"固定native q_feature边界：{manifest['q_quintile_thresholds']}。method=higher，tie归较低分位。\n\n"
        + table([r for r in group_rows if r["group"].startswith("Q")], gcols),
        "## 15. Boundary / interior\n\n" + table([r for r in joined if r["variant"] == "SRSC" and r["group"] in ("boundary","interior")],
        [("group","Group"),("pair_image_balanced_auroc","Image AUROC"),("pair_auroc","Pooled AUROC"),("purity_image_mean","Purity"),("purity_gain_image_mean","Gain"),("neff_image_mean","N_eff")])
        + "\n边界仅由fullres FG–FG transition的欧氏距离<=7px构建，再投影；不进入score。\n",
        "## 16. Per-class relation audit\n\n" + table([r for r in joined if r["variant"] == "SRSC" and r["group"].startswith("class")],
        [("group","Class"),("pair_prevalence","Same prevalence"),("pair_auroc","AUROC"),("pair_auprc","AP"),("uniform_purity_image_mean","U purity"),("purity_image_mean","SRSC purity"),("purity_gain_image_mean","Gain vs U"),("neff_image_mean","N_eff")])
        + "\n四类完整U/SR/SC/SRSC数据均保留，任何单类收益不替代主门槛。\n",
        "## 17. Training-free neighbor estimator\n\n" + table([r for r in estimator_rows if r["group"] == "all"], ecols)
        + "\n原生28-grid四类指标，不能与224-grid/TTA final CAM mIoU直接比较。没有background prediction overwrite；"
        "zero-union类别从macro mean排除并保留NA，不设1。Brier是四类平方误差求和再target均值，NLL=-log(p_GT+eps)。"
        "raw/deep只是参照，没有参加protocol选择。\n",
        "## 18. Frozen Top20 repair / harm\n\n" + table([r for r in repair_rows if r["group"] == "Top20"],
        [("estimator","Estimator"),("repair_count","Repair n"),("harm_count","Harm n"),("targets","Targets"),("repair","Repair"),("harm","Harm"),("net_repair","NetRepair")])
        + "\n基准raw=原生28-grid argmax(ps)，不是历史upsampled raw。repair/harm均除以全部eligible Top20。\n",
        "## 19. Deep-Correct / Deep-Wrong\n\n" + table([r for r in deep_rows if r["variant"] in ("U","SRSC")],
        [("group","Stratum"),("variant","Relation"),("purity_image_mean","Purity"),("purity_gain_image_mean","Gain"),("neighbor_accuracy","Neighbor acc"),("net_repair_vs_raw","Stratum NetRepair"),("top20_net_repair","Top20 NetRepair")])
        + "\nDeep-anchored relation的安全性必须条件化解释：deep错误时兼容性可以偏向错误假设，不能称为普遍安全。\n",
        "## 20. Deep-hypothesis echo\n\n" + table([r for r in echo_rows if r["group"] in ("all","Top20","Deep_Correct","Deep_Wrong")],
        [("group","Group"),("echo_fraction","Echo fraction"),("non_echo_count","Non-echo n"),("srsc_correct_deep_wrong_count","SRSC right / deep wrong"),("srsc_wrong_deep_correct_count","SRSC wrong / deep right")])
        + "\n非echo总量还包括双方均错但预测类别不同的情况，所以末两列不一定加和为非echo总量。\n",
        "## 21. Oracle diagnostic upper bound\n\n" + table([r for r in estimator_rows if r["group"] == "all" and r["estimator"] in ("U","SRSC","oracle")], ecols)
        + f"\nOracle有效targets={summary['oracle_valid_targets']:,}；实测purity={summary['oracle_purity']:.9f}；平均同类邻居数={summary['oracle_mean_same_class_neighbors']:.6f}。"
        + "\nOracle仅同类GT邻居；有邻居时purity约1，零同类邻居为未定义，coverage显式报告；没有raw/deep fallback。"
        "它是GT关系选择的诊断参照，并非所有预测指标的数学上界（同类source的shallow语义仍可能错）。无oracle调参。\n",
        "## 22. Paired image bootstrap\n\n" + table(boot_rows, [("metric","Metric"),("observed","Observed"),("ci95_low","95% low"),("ci95_high","95% high"),("eligible_images","Images")])
        + "\n10000次、seed42、相同图像索引，同一image内所有target/pair保持一起。mIoU重加4×4confusion后重算，"
        "非逐图mIoU平均；Top20按sample总repair-harm/sample总Top20。未做pair-level naive bootstrap。"
        "完整replicates CSV和充分统计NPZ保留，可独立重算。\n",
        "## 23. Preregistered gates\n\n" + table([dict(gate=k, result="PASS" if v else "FAIL") for k,v in gates.items()], [("gate","Gate"),("result","Result")])
        + "\nA: image AUROC>=.65且CI下界>.5；B: image purity增益>=.03且CI下界>0、meanN_eff>=5；"
        "C: paired Corrected-Harmed>0且CI下界>0、Harmed gain>0；D: neighbor accuracy/mIoU均增、至少一项CI下界>0、Top20净修复增。\n",
        "## 24. Scientific interpretation / delivery\n\n"
        + ({"RDDR_PHASE2B0_NOGO": "至少一个关系判别/纯度主门槛失败，当前r×c formulation未达到预注册可行性条件，应停止当前formulation。",
            "RELATION_SIGNAL_NOT_CH_OUTCOME_SPECIFIC": "一般同类关系信号成立，但不能按预注册标准解释Corrected/Harmed机制差异。",
            "RELATION_EXISTS_NO_PROPAGATION_UTILITY": "关系/机制信号成立，但直接one-step邻居聚合没有满足utility门槛；只能把消费方式作为另一个独立问题。",
            "RDDR_PHASE2B0_GO": "关系判别、纯度、CH-outcome关联及无训练聚合utility均过门槛；仅支持进入下一独立训练假设，不保证最终CAM会提升。"}[decision])
        + "\n\nSR/SC超过primary、某子集改善、oracle headroom均不触发替换primary或posthoc调参。"
        "与先前feature/context suppression失败形成的证据链限于：冲突存在≠必须少用context；本轮检验source selection，而非新模型有效性。"
        "本轮遵循研究实施交付流程，保留冻结合同、可运行命令、测试、逐图/汇总CSV、JSON、原始充分统计和独立PR；"
        "未训练、未访问test/LUAD、未新增权重、未自动merge。此前各实验不删除不覆盖。\n",
        "## 25. Exact decision / stop\n\n完成报告后停止。即使GO，也不自动启动Phase-2B训练。\n\n"
        + "DECISION = " + decision,
    ]
    (out / "rddr_phase2b0_reliable_relation_feasibility_report.md").write_text("\n".join(sections)+"\n", encoding="utf-8")
    print(json.dumps(dict(decision=decision, gates=gates, ci=ci), indent=2))


if __name__ == "__main__":
    main()
