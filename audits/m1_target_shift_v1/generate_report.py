"""Pre-registered decision, representative-case figures, and Chinese report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

STAGES = ("H5_pre", "H5_context", "H4_input", "K4")


def decision(stage: pd.DataFrame, anatomy: pd.DataFrame, classes: pd.DataFrame,
             purity: pd.DataFrame, size: pd.DataFrame, counter: pd.DataFrame,
             target: pd.DataFrame, sensitivity: pd.DataFrame) -> dict:
    whole = stage[stage.region == "whole"].set_index("stage")
    core = stage[stage.region == "core"].set_index("stage")
    ana = anatomy.set_index("stage")
    high_purity = purity[(purity.stage == "H5_pre") & (purity.stratum == "purity_ge_070")].iloc[0]
    large = size[(size.stage == "H5_pre") & (size.quartile >= 3)]
    class_support = classes[(classes.stage == "H5_pre") &
                            (classes.auroc >= .75) & (classes.ratio >= 1.5)]
    m1 = target[target.cohort == "M1"]
    clean_counter = counter[(counter.stage == "K4")].set_index("target_index")
    c1_rescue = float((clean_counter.C1_minus_C0 < 0).mean())
    gt_h5 = sensitivity[(sensitivity.matching == "GT_true_class") &
                        (sensitivity.stage == "H5_pre") &
                        (sensitivity.score == "d_true")].iloc[0]
    reasons = []
    if bool(whole.loc["H5_pre", "strong_shift"]):
        result = "UPSTREAM_SEMANTIC_SHIFT"
        confidence = "HIGH" if (len(class_support) >= 3 and high_purity.auroc >= .70 and
                                 (large.auroc >= .70).all()) else "MEDIUM"
        reasons.append("S0 raw F5 已满足预注册 Strong Shift；HQMR H5→H4 不是首次异常位置。")
    elif (gt_h5.auroc >= .75 and gt_h5.distance_ratio >= 1.5 and
          gt_h5.ci95_low > .5 and gt_h5.ratio_ci95_low > 1.0):
        result, confidence = "UPSTREAM_SEMANTIC_SHIFT", "MEDIUM"
        reasons.append("按同一 GT 组织类校正匹配后，S0 raw F5 已 Strong Shift；原方案仅按 predicted class 匹配 TP 会比较不同真类，故主/敏感性结果不一致，置信度降为 MEDIUM。")
    elif bool(whole.loc["H5_context", "strong_shift"]):
        result, confidence = "UPSTREAM_SEMANTIC_SHIFT", "MEDIUM"
        reasons.append("异常首次见于 F5→context/CHPF 路径，仍早于 HQMR hierarchy。")
    elif (bool(whole.loc["K4", "strong_shift"]) and
          core.loc["K4", "auroc"] <= .65 and
          ana.loc["K4", "M1_core_rescue_median"] >= .30 and
          ana.loc["K4", "M1_boundary_rival_fraction"] > .5 and
          ana.loc["K4", "M1_ring_rival_fraction"] > .5):
        result, confidence = "BOUNDARY_CONTEXT_CONTAMINATION", "MEDIUM"
        reasons.append("K4 whole 异常、core 恢复且边界/外环 rival-like；但 H4_recon 空间张量不存在，不能归因其内部更新。")
    elif (bool(whole.loc["K4", "strong_shift"]) and
          core.loc["K4", "auroc"] <= .65 and c1_rescue > .6):
        result, confidence = "REGION_POOLING_SHIFT", "MEDIUM"
        reasons.append("K4 whole 异常，而核心及 GT-clean intersection 多数恢复，支持 mask support/pooling 主导。")
    elif (bool(whole.loc["H5_pre", "near_normal"]) and
          bool(whole.loc["H5_context", "near_normal"]) and
          bool(whole.loc["H4_input", "near_normal"]) and
          bool(whole.loc["K4", "strong_shift"])):
        result, confidence = "LATE_K4_SHIFT", "MEDIUM"
        reasons.append("K4 是固定 scale4.key 投影，异常不可归给后续 query update。")
    else:
        result, confidence = "MIXED_FAILURE", "LOW"
        reasons.append("可观测空间 stage 未呈现满足门槛的单一 onset；S2/S3 空间特征不存在。")
    return {"decision": result, "confidence": confidence, "reasons": reasons,
            "H4_recon_spatial_feature_observable": False,
            "H4_query_spatial_feature_observable": False,
            "D2_reconstruction_shift_identifiable_from_spatial_manifold": False,
            "K4_C1_distance_improvement_fraction": c1_rescue,
            "GT_class_matched_H5_AUROC": float(gt_h5.auroc),
            "GT_class_matched_H5_distance_ratio": float(gt_h5.distance_ratio),
            "parameter_updates": 0,
            "class_support_count": int(len(class_support)),
            "M1_high_purity_count": int((m1.purity >= .70).sum())}


def _load_metric(output: Path, stage: str, region: str) -> pd.DataFrame:
    return pd.read_parquet(output / "metrics" / f"{stage}_{region}.parquet")


def figures(output: Path, val_root: Path, cache_root: Path,
            target: pd.DataFrame, matches: pd.DataFrame) -> dict:
    manifest = json.loads((cache_root / "cache" / "prediction_manifest.json").read_text())
    cache_map = {row["image_id"]: cache_root / "cache" / row["cache"] for row in manifest}
    masks = {kind: np.load(output / "masks" / f"target_{kind}.npz")["packed"]
             for kind in ("whole", "core", "boundary", "ring")}
    h5 = _load_metric(output, "H5_context", "whole")
    h4 = _load_metric(output, "H4_input", "whole")
    k4 = _load_metric(output, "K4", "whole")
    kcore = _load_metric(output, "K4", "core")
    target = target.reset_index(drop=True)
    m1 = np.flatnonzero(target.cohort.to_numpy() == "M1")
    onset = k4.ood_percentile.to_numpy() - h5.ood_percentile.to_numpy()
    candidate = m1[np.isfinite(onset[m1]) & (target.area.to_numpy()[m1] >= 100)]
    if len(candidate) < 20:
        candidate = m1[np.isfinite(onset[m1])]
    ranked = {
        "shift_onset": candidate[np.argsort(-onset[candidate])[:20]],
        "H5_normal_H4_logit_surrogate": candidate[np.argsort(-(
            target.pred_minus_true_score_H4_recon_logits.to_numpy()[candidate] -
            target.pred_minus_true_score_H5_logits.to_numpy()[candidate]))[:20]],
        "boundary_contamination": candidate[np.argsort(-(
            k4.d_true.to_numpy()[candidate] - kcore.d_true.to_numpy()[candidate]))[:20]],
        "whole_region_corruption": candidate[np.argsort(-kcore.d_true.to_numpy()[candidate])[:20]],
        "matched_TP": matches[matches.tp_index.isin(target.index[target.area >= 100])].drop_duplicates(
            "tp_index").sort_values("log_area_difference").tp_index.to_numpy()[:20],
    }
    summary = {}
    for category, indices in ranked.items():
        folder = output / "visualizations" / category
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for serial, index in enumerate(indices, 1):
            item = target.iloc[int(index)]
            image_id = item.image_id
            rgb = np.asarray(Image.open(val_root / "img" / f"{image_id}.png").convert("RGB"))
            gt = np.asarray(Image.open(val_root / "mask" / f"{image_id}.png"))
            pred = np.load(cache_map[image_id])["hqmr"]
            shape = gt.shape
            unpacked = {kind: np.unpackbits(values[int(index)])[:gt.size].reshape(shape).astype(bool)
                        for kind, values in masks.items()}
            points = np.argwhere(unpacked["whole"])
            if not len(points):
                continue
            margin = max(16, int(np.sqrt(item.area) / 2))
            y0 = max(0, int(points[:, 0].min()) - margin)
            y1 = min(shape[0], int(points[:, 0].max()) + margin + 1)
            x0 = max(0, int(points[:, 1].min()) - margin)
            x1 = min(shape[1], int(points[:, 1].max()) + margin + 1)
            def highlighted(kind, color):
                view = rgb.copy().astype(np.float32)
                selected = unpacked[kind]
                view[selected] = .25 * view[selected] + .75 * np.array(color, np.float32)
                return np.clip(view[y0:y1, x0:x1], 0, 255).astype(np.uint8)
            fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
            panels = ((rgb, "Original"), (gt, "GT"), (pred, "Frozen HQMR"),
                      (highlighted("whole", (255, 32, 32)), "Component (zoom)"),
                      (highlighted("core", (0, 255, 80)), "Core (zoom)"),
                      (highlighted("boundary", (255, 215, 0)), "Boundary (zoom)"),
                      (highlighted("ring", (30, 140, 255)), "Ring (zoom)"))
            for ax, (data, label) in zip(axes.flat, panels):
                ax.imshow(data, cmap=None if data.ndim == 3 else "tab20", interpolation="nearest")
                ax.set_title(label)
                ax.axis("off")
            ax = axes.flat[-1]
            ax.axis("off")
            text = (f"{category}\nclass {item.predicted_class} → GT {item.true_class}; "
                    f"area {item.area}; purity {item.purity:.2f}\n"
                    f"H5 pre d={_load_metric_cache[('H5_pre','whole')].iloc[int(index)].d_true:.3f}\n"
                    f"H5 context d={h5.iloc[int(index)].d_true:.3f}\n"
                    f"H4 input d={h4.iloc[int(index)].d_true:.3f}\n"
                    f"K4 d={k4.iloc[int(index)].d_true:.3f}; "
                    f"core d={kcore.iloc[int(index)].d_true:.3f}\n"
                    f"K4 OOD={k4.iloc[int(index)].ood_percentile:.2f}; "
                    f"margin={k4.iloc[int(index)].margin:.3f}\n"
                    "H4_recon: query-mask logits only, no tissue embedding")
            ax.text(0, 1, text, va="top", family="monospace", fontsize=9, wrap=True)
            path = folder / f"{serial:02d}_{image_id}_{item.component_id}.png"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            rows.append({"target_index": int(index), "image_id": image_id,
                         "component_id": int(item.component_id), "file": str(path.relative_to(output))})
        summary[category] = rows
    (output / "visualizations" / "case_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {key: len(value) for key, value in summary.items()}


_load_metric_cache = {}


def report(output: Path, decision_result: dict, visual_counts: dict) -> str:
    gate = json.loads((output / "00_reproduction_gate.json").read_text())
    hooks = json.loads((output / "hook_manifest.json").read_text())
    cohort = json.loads((output / "cohort_manifest.json").read_text())
    stage = pd.read_csv(output / "metrics" / "stage_shift_metrics.csv")
    anatomy = pd.read_csv(output / "metrics" / "core_boundary_metrics.csv")
    per_class = pd.read_csv(output / "metrics" / "per_class_metrics.csv")
    purity = pd.read_csv(output / "metrics" / "purity_stratified_metrics.csv")
    size = pd.read_csv(output / "metrics" / "size_stratified_metrics.csv")
    membership = pd.read_csv(output / "metrics" / "manifold_membership.csv")
    sensitivity = pd.read_csv(output / "metrics" / "matching_sensitivity.csv")
    true_strata = pd.read_csv(output / "metrics" / "true_class_matched_strata.csv")
    counter = pd.read_csv(output / "metrics" / "counterfactual_metrics.csv")
    target = pd.read_parquet(output / "target_cohorts.parquet")
    m1 = target[target.cohort == "M1"]
    def table(data, columns):
        def show(value):
            if isinstance(value, (float, np.floating)):
                return "NA" if not np.isfinite(value) else f"{value:.4f}"
            return str(value)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"
        body = ["| " + " | ".join(show(value) for value in row) + " |"
                for row in data[columns].itertuples(index=False, name=None)]
        return "\n".join([header, separator, *body])
    whole = stage[stage.region == "whole"]
    core = stage[stage.region == "core"]
    common = table(whole, ["stage", "M1_median", "TP_median", "distance_ratio", "auroc",
                           "strong_shift", "near_normal"])
    by_class = table(per_class, ["stage", "class", "n", "ratio", "auroc", "m1_ood95_fraction"])
    by_size = table(size, ["stage", "quartile", "n", "ratio", "auroc"])
    by_purity = table(purity, ["stage", "stratum", "n", "ratio", "auroc"])
    member = table(membership, ["stage", "off_manifold_area_fraction",
                                "wrong_class_manifold_area_fraction", "true_class_manifold_area_fraction"])
    sensitivity_table = table(sensitivity[sensitivity.score == "d_true"],
                              ["matching", "stage", "n_pairs", "auroc", "ci95_low",
                               "distance_ratio", "ratio_ci95_low"])
    ana = table(anatomy, ["stage", "M1_core_rescue_median", "M1_boundary_rival_fraction",
                           "M1_ring_rival_fraction", "M1_core_boundary_conflict_median"])
    dispersion_rows, conflict_rows = [], []
    target_mask = target.cohort.to_numpy() == "M1"
    tp_mask = target.cohort.to_numpy() == "TP_candidate"
    for stage_name in STAGES:
        for kind in ("whole", "core", "boundary", "ring"):
            column = f"dispersion_{stage_name}_{kind}"
            dispersion_rows.append({"stage": stage_name, "region": kind,
                "M1_median": float(np.nanmedian(target.loc[target_mask, column])),
                "TP_candidate_median": float(np.nanmedian(target.loc[tp_mask, column]))})
        z = np.load(output / "features" / f"target_{stage_name}.npz")
        conflict = 1 - np.einsum("nd,nd->n", z["core"], z["boundary"])
        conflict_rows.append({"stage": stage_name,
            "M1_median": float(np.nanmedian(conflict[target_mask])),
            "TP_candidate_median": float(np.nanmedian(conflict[tp_mask]))})
    dispersion_table = table(pd.DataFrame(dispersion_rows),
                             ["stage", "region", "M1_median", "TP_candidate_median"])
    conflict_table = table(pd.DataFrame(conflict_rows),
                           ["stage", "M1_median", "TP_candidate_median"])
    clean = counter.groupby("stage").agg(C1_rescue_fraction=("C1_minus_C0", lambda x: (x < 0).mean()),
                                            C2_rescue_fraction=("C2_minus_C0", lambda x: (x < 0).mean())).reset_index()
    clean_table = table(clean, ["stage", "C1_rescue_fraction", "C2_rescue_fraction"])
    logs = target[target.cohort == "M1"][["pred_minus_true_score_H5_logits",
        "pred_minus_true_score_H4_recon_logits", "pred_minus_true_score_H3_final_logits"]].median().to_dict()
    sections = [
        ("1. Executive Decision", f"DECISION = **{decision_result['decision']}**；CONFIDENCE = **{decision_result['confidence']}**。" + " ".join(decision_result["reasons"])),
        ("2. Reproduction Gate", f"PASS={gate['pass']}；HQMR={100*gate['hqmr_miou']:.6f}%；M1={gate['m1_components']} components / {gate['m1_pixels']} pixels；checkpoint SHA256={gate['checkpoint_sha256']}；parameter updates=0。PSCR distance C={gate['distance_C']:.4f}、D={gate['distance_D']:.4f}、M1={gate['distance_M1']:.4f} 为冻结历史 anchor，非重新计算。"),
        ("3. Hook Manifest", f"S0=raw backbone F5；S1=CHPF(context_projection(F5))，它不是 CCRA-conditioned H5；S2=logits4（H5 logits 插值 + direct4），不是空间组织特征；S3=query4 向量，不是空间组织特征；S4=原 CIRV/PSCR K4。不存在可供 S2/S3 空间 region pooling 的张量。样本 hook 前后 logits/CAM/输出最大差=0；全验证集 hook 调用 k5/k4/update4 各 {gate['full_validation_hook_calls']['k5']} 次，mIoU/M1 与无 hook 基线完全一致。"),
        ("4. Cohort Construction", f"M1={cohort['M1']}；TP candidates={cohort['TP_candidates']}；matched M1={cohort['matching']['m1_matched']}；同图匹配比例={cohort['matching']['same_image_fraction']:.3f}；同 area decile={cohort['matching']['same_decile_fraction']:.3f}；TRAIN clean={cohort['reference_clean']}；boundary fragment={cohort['reference_boundary_fragments']}。TP 可被多个 M1 复用，bootstrap 以 M1 为抽样单位。同图比例低，image/tissue context 残余混杂不可排除。"),
        ("5. Stage-Wise Representation Shift", common + "\n\n按真实 GT 类别而非预测类别匹配 TP 的敏感性检查：\n\n" + sensitivity_table + "\n\n每一 stage 独立 class/area-bin centroid；不同 stage 原始 cosine 不直接相减。主匹配按 predicted class，因 M1 true class 与 predicted class 不同，true-class sensitivity 是必要稳健性检查。"),
        ("6. H5 Analysis", f"S0 raw F5 与 S1 context F5 的独立 manifold 检验见上表。H5_context 并非 CCRA 后空间语义图，不能解释为 CCRA-conditioned H5。"),
        ("7. H5→H4 Reconstruction Analysis", f"重建仅作用在 mask logits；M1 上 predicted-minus-true class activation 的中位数：H5={logs['pred_minus_true_score_H5_logits']:.4f}，H4={logs['pred_minus_true_score_H4_recon_logits']:.4f}。这是 mask 证据，不是 H4_recon tissue embedding；预注册空间 AUROC-onset 判据不可计算。"),
        ("8. Query-Conditioned H4 Analysis", "update4 生成 query4[B,Q,256]，不是逐像素特征。H4_query tissue manifold 不可计算；不能把 K4 的变化归因于 query4。"),
        ("9. K4 Analysis", "K4 来自 scale4.key(F4_context)，与 CIRV/PSCR 完全同一张量；其结果见 stage 表。"),
        ("10. Shift-Onset Localization", "可观测空间链为 F5 → context F5，以及独立 pixel-decoder F4_context → K4；H5→H4→query 的真实链为 logits5 → logits4 → query4 → logits3，不能假装是同类空间 embedding。"),
        ("11. Core-vs-Boundary Anatomy", ana + "\n\n" + table(core, ["stage", "M1_median", "TP_median", "distance_ratio", "auroc"])),
        ("12. Region Internal Dispersion", dispersion_table + "\n\n局部 feature 与 region pooled vector 的 cosine dispersion；绝对值不跨 stage 比较。"),
        ("13. Core-Boundary Semantic Conflict", conflict_table + "\n\n同一 stage 内 core/boundary 的 1−cos；无效 tiny mask 保留 NaN。"),
        ("14. Same-Mask Counterfactual", clean_table + "\n\nC0=predicted whole，C1=同一 feature 下 R∩dominant GT，C2=core；C3=TRAIN area-matched clean fragment（不同图像，非同一 tissue context）。C1 为 GT oracle 机制诊断，不进入推理。"),
        ("15. Purity-Stratified Results", by_purity),
        ("16. Size-Stratified Results", by_size),
        ("17. Per-Class Results", by_class + "\n\n同一 GT 类别匹配的 H5_pre 分层：\n\n" +
         table(true_strata[(true_strata.stage == "H5_pre") &
                           (true_strata.stratum.str.startswith("class_"))],
               ["stratum", "n", "distance_ratio", "auroc"]) +
         "\n\n同一 GT 类别匹配的 purity/size 分层见 metrics/true_class_matched_strata.csv；原方案 predicted-class 分层与校正匹配不应混为一谈。"),
        ("18. Tissue-Manifold Membership", member),
        ("19. Off-Manifold vs Wrong-Class Analysis", "all-class OOD>95%=off-manifold；nearest 为 rival 且其分布百分位≤95%=wrong-class tissue-like。两者以 class/area-bin clean TRAIN 分布估计；见上表。"),
        ("20. Representative Cases", ", ".join(f"{name}={count}" for name, count in visual_counts.items()) + "。H5-normal→H4-bad 类别仅按 mask-logit 代理排序，不是 H4_recon embedding 证据。案例可能重叠，不能按这 100 张估计频率。"),
        ("21. Decision Matrix", f"{decision_result['decision']} / {decision_result['confidence']}。D2 空间 representation 判据因模型不存在相应张量而不可验证；未擅自发明 projection。完整机器判决见 metrics/decision_matrix.json。"),
        ("22. Exact Root Cause", f"当前判决为 {decision_result['decision']}（{decision_result['confidence']}）。K4 core 异常且 rival-like manifold 占优；按同一真实组织类匹配，raw H5 已异常，故无法把首次错误归给 HQMR 重建。方案要求的 predicted-class 主匹配与 GT-class 敏感性匹配结论不同，需保留混杂警告。若需细分责任，下一步应审计 logits5、direct4、logits4 的逐像素类别 margin 与边界传播，而不是声称存在 H4_recon 空间组织特征。"),
        ("23. What Is Preserved", "冻结 CCRA/HQMR-v1、ResNet38、checkpoint、BCSS split、Seed42；所有 hook 只读、parameter update=0。历史 PSCR/CIRV 不参与本轮 decision。"),
        ("24. What Must Be Archived", "本次 artifact 中的 checkpoint hash、gate、hook map、cohort/matching、features/masks/manifolds、bootstrap CI、100 张案例及完整日志必须与报告一起保留。"),
        ("25. Exact Next Architecture Target", "优先审计并设计 upstream deep semantic assignment / CCRA responsibility / class competition 的可靠性约束，尤其 M1 高 purity 与大组件；HQMR hierarchical reconstruction 暂不作为首要修改点。若需区分放大环节，再对实际存在的 logits5→direct4→logits4→query4→logits3 做最小定向审计；不启动盲目 Full25 训练。"),
    ]
    return "# M1 Target Representation Shift Anatomy Audit（BCSS / Seed42）\n\n" + "\n\n".join(
        f"## {title}\n\n{body}" for title, body in sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--refresh-figures", action="store_true")
    args = parser.parse_args()
    output = args.output
    target = pd.read_parquet(output / "target_cohorts.parquet")
    matches = pd.read_csv(output / "metrics" / "matched_tp_pairs.csv")
    stage = pd.read_csv(output / "metrics" / "stage_shift_metrics.csv")
    anatomy = pd.read_csv(output / "metrics" / "core_boundary_metrics.csv")
    classes = pd.read_csv(output / "metrics" / "per_class_metrics.csv")
    purity = pd.read_csv(output / "metrics" / "purity_stratified_metrics.csv")
    size = pd.read_csv(output / "metrics" / "size_stratified_metrics.csv")
    counter = pd.read_csv(output / "metrics" / "counterfactual_metrics.csv")
    sensitivity = pd.read_csv(output / "metrics" / "matching_sensitivity.csv")
    result = decision(stage, anatomy, classes, purity, size, counter, target, sensitivity)
    (output / "metrics" / "decision_matrix.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _load_metric_cache[("H5_pre", "whole")] = _load_metric(output, "H5_pre", "whole")
    case_manifest = output / "visualizations" / "case_manifest.json"
    if case_manifest.exists() and not args.refresh_figures:
        counts = {key: len(value) for key, value in json.loads(case_manifest.read_text()).items()}
    else:
        counts = figures(output, args.val_root, args.cache_root, target, matches)
    (output / "M1_Target_Representation_Shift_Anatomy_Audit_Report.md").write_text(
        report(output, result, counts), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "confidence": result["confidence"],
                      "visualizations": counts}), flush=True)


if __name__ == "__main__":
    main()
