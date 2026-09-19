"""Matched-control, class/area-conditioned geometry for frozen HQMR M1 regions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.m1_target_shift_v1.extract_multistage_features import REGIONS, STAGES
from audits.m1_target_shift_v1.preflight import EXPECTED


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def percentile_of(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, values, side="right") / max(len(sorted_values), 1)


def bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:-1], values, side="right")


def match_tp(target: pd.DataFrame, edges: np.ndarray) -> tuple[pd.DataFrame, dict]:
    frame = target.copy()
    frame["area_bin"] = bins(np.log1p(frame.area.to_numpy()), edges)
    m1 = frame[frame.cohort == "M1"]
    tp = frame[frame.cohort == "TP_candidate"]
    matches = []
    for index, row in m1.iterrows():
        candidates = tp[(tp.predicted_class == row.predicted_class) &
                        (tp.area_bin == row.area_bin)]
        if candidates.empty:
            candidates = tp[tp.predicted_class == row.predicted_class]
        if candidates.empty:
            continue
        same = candidates[candidates.image_id == row.image_id]
        if not same.empty:
            candidates = same
        score = np.abs(np.log1p(candidates.area.to_numpy()) - np.log1p(row.area))
        selected = candidates.iloc[np.argsort(score, kind="stable")[:3]]
        for rank, (tp_index, control) in enumerate(selected.iterrows(), 1):
            matches.append({"m1_index": int(index), "tp_index": int(tp_index), "rank": rank,
                            "same_image": bool(control.image_id == row.image_id),
                            "same_decile": bool(control.area_bin == row.area_bin),
                            "log_area_difference": float(abs(np.log1p(control.area) - np.log1p(row.area)))})
    matched = pd.DataFrame(matches)
    coverage = {"m1_total": int(len(m1)), "m1_matched": int(matched.m1_index.nunique()),
                "tp_candidates": int(len(tp)), "tp_unique_matched": int(matched.tp_index.nunique()),
                "same_image_fraction": float(matched.same_image.mean()),
                "same_decile_fraction": float(matched.same_decile.mean()),
                "median_log_area_difference": float(matched.log_area_difference.median())}
    return matched, coverage


def geometry(target: pd.DataFrame, reference: pd.DataFrame, target_z: dict,
             reference_z: dict, stage: str, kind: str, edges: np.ndarray) -> tuple[pd.DataFrame, dict]:
    clean = reference.cohort.to_numpy() == "clean_GT"
    target_bin = bins(np.log1p(target.area.to_numpy()), edges)
    reference_bin = bins(np.log1p(reference.area.to_numpy()), edges)
    centroids = np.zeros((len(edges)-1, 4, target_z.shape[1]), np.float32)
    counts = np.zeros((len(edges)-1, 4), np.int32)
    clean_whole = reference_z["whole"]
    for b in range(len(edges)-1):
        for cls in range(4):
            use = clean & (reference.true_class.to_numpy() == cls) & (reference_bin == b)
            if use.sum() < 20:
                use = clean & (reference.true_class.to_numpy() == cls)
            valid = use & np.isfinite(clean_whole).all(1)
            if not valid.any():
                raise AssertionError(f"No clean reference for {stage}, class {cls}, bin {b}")
            centroids[b, cls] = unit(clean_whole[valid].mean(0))
            counts[b, cls] = int(valid.sum())
    embedding = target_z
    distances = 1 - np.einsum("nd,ncd->nc", embedding, centroids[target_bin])
    distances[~np.isfinite(embedding).all(1)] = np.nan
    true = target.true_class.to_numpy(np.int64)
    true_dist = distances[np.arange(len(target)), true]
    rival = distances.copy()
    rival[np.arange(len(target)), true] = np.inf
    nearest = np.nanargmin(np.where(np.isfinite(distances), distances, np.inf), axis=1)
    output = pd.DataFrame({"d_true": true_dist, "d_near": np.nanmin(distances, axis=1),
                           "d_rival": np.nanmin(rival, axis=1),
                           "margin": np.nanmin(rival, axis=1) - true_dist,
                           "nearest_class": nearest, "area_bin": target_bin})
    reference_d = 1 - np.einsum("nd,nd->n", reference_z[kind],
                               centroids[reference_bin, reference.true_class.to_numpy(np.int64)])
    ood = np.full(len(target), np.nan, np.float32)
    all_ood = np.full((len(target), 4), np.nan, np.float32)
    for b in range(len(edges)-1):
        for cls in range(4):
            use = clean & (reference.true_class.to_numpy() == cls) & (reference_bin == b) & np.isfinite(reference_d)
            if use.sum() < 20:
                use = clean & (reference.true_class.to_numpy() == cls) & np.isfinite(reference_d)
            distribution = np.sort(reference_d[use])
            mask = target_bin == b
            all_ood[mask, cls] = percentile_of(distribution, distances[mask, cls])
    ood = all_ood[np.arange(len(target)), true]
    output["ood_percentile"] = ood
    output["nearest_ood_percentile"] = all_ood[np.arange(len(target)), nearest]
    output["all_class_ood"] = np.all(all_ood > .95, axis=1)
    output["wrong_class_manifold"] = (nearest != true) & (output.nearest_ood_percentile <= .95)
    output["true_class_manifold"] = (nearest == true) & (ood <= .95)
    return output, {"centroids": centroids, "reference_counts": counts,
                    "reference_d_true": reference_d}


def auroc_ci(m1: np.ndarray, tp: np.ndarray, rng: np.random.Generator,
             n_boot: int = 2000) -> dict:
    valid = np.isfinite(m1) & np.isfinite(tp)
    m1, tp = m1[valid], tp[valid]
    if len(m1) < 20:
        return {"n_pairs": int(len(m1)), "auroc": None, "distance_ratio": None,
                "auroc_ci95": [None, None], "distance_ratio_ci95": [None, None],
                "distance_difference_ci95": [None, None]}
    def stats(left, right):
        auc = roc_auc_score(np.r_[np.ones(len(left)), np.zeros(len(right))], np.r_[left, right])
        ratio = float(np.median(left) / max(np.median(right), 1e-8))
        difference = float(np.median(left) - np.median(right))
        return auc, ratio, difference
    point = stats(m1, tp)
    boot = np.empty((n_boot, 3), np.float64)
    for iteration in range(n_boot):
        index = rng.integers(0, len(m1), len(m1))
        boot[iteration] = stats(m1[index], tp[index])
    low, high = np.quantile(boot, [.025, .975], axis=0)
    return {"n_pairs": int(len(m1)), "auroc": float(point[0]), "distance_ratio": float(point[1]),
            "median_difference": float(point[2]),
            "auroc_ci95": [float(low[0]), float(high[0])],
            "distance_ratio_ci95": [float(low[1]), float(high[1])],
            "distance_difference_ci95": [float(low[2]), float(high[2])]}


def paired_scores(metric: pd.DataFrame, matches: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = matches.groupby("m1_index").tp_index.apply(list)
    ids = grouped.index.to_numpy(np.int64)
    m1 = metric.iloc[ids].d_true.to_numpy(np.float64)
    tp = np.array([np.nanmean(metric.iloc[controls].d_true.to_numpy())
                   for controls in grouped], np.float64)
    return ids, m1, tp


def summarize_subset(metric: pd.DataFrame, target: pd.DataFrame,
                     matches: pd.DataFrame, selected: np.ndarray) -> dict:
    sub = matches[matches.m1_index.isin(selected)]
    if sub.empty:
        return {"n": 0}
    ids, m1, tp = paired_scores(metric, sub)
    valid = np.isfinite(m1) & np.isfinite(tp)
    if valid.sum() < 2:
        return {"n": int(valid.sum())}
    auc = roc_auc_score(np.r_[np.ones(valid.sum()), np.zeros(valid.sum())],
                        np.r_[m1[valid], tp[valid]])
    return {"n": int(valid.sum()), "m1_median": float(np.median(m1[valid])),
            "tp_median": float(np.median(tp[valid])),
            "ratio": float(np.median(m1[valid]) / max(np.median(tp[valid]), 1e-8)),
            "auroc": float(auc),
            "m1_ood95_fraction": float((metric.iloc[ids].ood_percentile.to_numpy()[valid] > .95).mean()),
            "m1_margin_median": float(np.nanmedian(metric.iloc[ids].margin.to_numpy()[valid])),
            "m1_nearest_accuracy": float((metric.iloc[ids].nearest_class.to_numpy()[valid] ==
                                           target.iloc[ids].true_class.to_numpy()[valid]).mean())}


def run(output: Path) -> None:
    gate = json.loads((output / "00_reproduction_gate.json").read_text())
    if not gate.get("pass"):
        raise AssertionError("Fresh reproduction gate must pass")
    target = pd.read_parquet(output / "target_cohorts.parquet")
    reference = pd.read_parquet(output / "reference_cohorts.parquet")
    m1_mask = target.cohort.to_numpy() == "M1"
    if m1_mask.sum() != EXPECTED["m1_components"]:
        raise AssertionError("M1 cohort count mismatch")
    log_area = np.log1p(target.loc[m1_mask, "area"].to_numpy())
    edges = np.r_[-np.inf, np.unique(np.quantile(log_area, np.linspace(.1, .9, 9))), np.inf]
    matches, coverage = match_tp(target, edges)
    if coverage["m1_matched"] < .9 * EXPECTED["m1_components"]:
        raise AssertionError(f"Insufficient TP matching: {coverage}")
    (output / "metrics").mkdir(exist_ok=True)
    (output / "manifolds").mkdir(exist_ok=True)
    matches.to_csv(output / "metrics" / "matched_tp_pairs.csv", index=False)
    (output / "cohort_manifest.json").write_text(json.dumps({"target_regions": int(len(target)),
        "M1": int(m1_mask.sum()), "M1_pixels": int(target.loc[m1_mask, "area"].sum()),
        "TP_candidates": int((target.cohort == "TP_candidate").sum()),
        "reference_clean": int((reference.cohort == "clean_GT").sum()),
        "reference_boundary_fragments": int((reference.cohort == "boundary_fragment").sum()),
        "matching": coverage, "log_area_edges": edges.tolist()}, indent=2), encoding="utf-8")
    stage_rows, anatomy_rows, per_class, purity_rows, size_rows, membership_rows = [], [], [], [], [], []
    bootstrap = {}
    metrics = {}
    m1_indices = np.flatnonzero(m1_mask)
    quartile = np.searchsorted(np.quantile(target.loc[m1_mask, "area"], [.25, .5, .75]),
                               target.area.to_numpy(), side="right") + 1
    for stage in STAGES:
        tz = dict(np.load(output / "features" / f"target_{stage}.npz"))
        rz = dict(np.load(output / "features" / f"reference_{stage}.npz"))
        center_payload = None
        for kind in REGIONS:
            metric, manifold = geometry(target, reference, tz[kind], rz,
                                        stage, kind, edges)
            metrics[(stage, kind)] = metric
            if center_payload is None:
                center_payload = manifold
            if kind in ("whole", "core", "boundary", "ring"):
                ids, m1, tp = paired_scores(metric, matches)
                ci = auroc_ci(m1, tp, np.random.default_rng(42), 2000)
                row = {"stage": stage, "region": kind,
                       "M1_median": float(np.nanmedian(m1)), "TP_median": float(np.nanmedian(tp)),
                       "M1_ood95_fraction": float((metric.iloc[ids].ood_percentile > .95).mean()),
                       "M1_margin_median": float(np.nanmedian(metric.iloc[ids].margin)),
                       "M1_nearest_accuracy": float((metric.iloc[ids].nearest_class.to_numpy() ==
                                                     target.iloc[ids].true_class.to_numpy()).mean()), **ci}
                row["strong_shift"] = bool(ci["auroc"] is not None and ci["auroc"] >= .75 and
                                           ci["distance_ratio"] >= 1.5 and
                                           ci["auroc_ci95"][0] > .5 and
                                           ci["distance_ratio_ci95"][0] > 1.0)
                row["near_normal"] = bool(ci["auroc"] is not None and ci["auroc"] <= .60 and
                                          ci["distance_ratio"] < 1.20)
                stage_rows.append(row)
                bootstrap[f"{stage}_{kind}"] = ci
            metric.assign(target_index=np.arange(len(target)), stage=stage, region=kind).to_parquet(
                output / "metrics" / f"{stage}_{kind}.parquet", index=False, compression="zstd")
        np.savez_compressed(output / "manifolds" / f"{stage}_reference.npz",
                            centroids=center_payload["centroids"],
                            reference_counts=center_payload["reference_counts"],
                            area_edges=edges)
        whole, core, boundary, ring = (metrics[(stage, kind)] for kind in ("whole", "core", "boundary", "ring"))
        rescue = (whole.d_true.to_numpy() - core.d_true.to_numpy()) / np.maximum(whole.d_true.to_numpy(), 1e-8)
        anatomy_rows.append({"stage": stage, "M1_core_rescue_median": float(np.nanmedian(rescue[m1_mask])),
                             "M1_whole_core_delta_median": float(np.nanmedian((whole.d_true-core.d_true)[m1_mask])),
                             "M1_boundary_rival_fraction": float((boundary.margin[m1_mask] < 0).mean()),
                             "M1_ring_rival_fraction": float((ring.margin[m1_mask] < 0).mean()),
                             "M1_core_boundary_conflict_median": float(np.nanmedian(
                                 1-np.einsum("nd,nd->n", tz["core"][m1_mask], tz["boundary"][m1_mask]))),
                             "M1_core_ring_conflict_median": float(np.nanmedian(
                                 1-np.einsum("nd,nd->n", tz["core"][m1_mask], tz["ring"][m1_mask])))})
        for cls in range(4):
            selected = m1_indices[target.iloc[m1_indices].true_class.to_numpy() == cls]
            per_class.append({"stage": stage, "class": cls, **summarize_subset(whole, target, matches, selected)})
        for label, select in (("purity_lt_070", target.purity.to_numpy() < .7),
                              ("purity_ge_070", target.purity.to_numpy() >= .7)):
            purity_rows.append({"stage": stage, "stratum": label,
                                **summarize_subset(whole, target, matches, m1_indices[select[m1_indices]])})
        for q in range(1, 5):
            size_rows.append({"stage": stage, "quartile": q,
                              **summarize_subset(whole, target, matches,
                                                 m1_indices[quartile[m1_indices] == q])})
        sub = whole.iloc[m1_indices]
        area = target.iloc[m1_indices].area.to_numpy()
        membership_rows.append({"stage": stage, "M1_components": int(len(sub)),
            "off_manifold_fraction": float(sub.all_class_ood.mean()),
            "off_manifold_area_fraction": float(np.average(sub.all_class_ood, weights=area)),
            "wrong_class_manifold_fraction": float(sub.wrong_class_manifold.mean()),
            "wrong_class_manifold_area_fraction": float(np.average(sub.wrong_class_manifold, weights=area)),
            "true_class_manifold_fraction": float(sub.true_class_manifold.mean()),
            "true_class_manifold_area_fraction": float(np.average(sub.true_class_manifold, weights=area))})
    pd.DataFrame(stage_rows).to_csv(output / "metrics" / "stage_shift_metrics.csv", index=False)
    pd.DataFrame(anatomy_rows).to_csv(output / "metrics" / "core_boundary_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "metrics" / "per_class_metrics.csv", index=False)
    pd.DataFrame(purity_rows).to_csv(output / "metrics" / "purity_stratified_metrics.csv", index=False)
    pd.DataFrame(size_rows).to_csv(output / "metrics" / "size_stratified_metrics.csv", index=False)
    pd.DataFrame(membership_rows).to_csv(output / "metrics" / "manifold_membership.csv", index=False)
    counterfactual = []
    for stage in STAGES:
        whole = metrics[(stage, "whole")].iloc[m1_indices].d_true.to_numpy()
        clean = metrics[(stage, "intersection")].iloc[m1_indices].d_true.to_numpy()
        core = metrics[(stage, "core")].iloc[m1_indices].d_true.to_numpy()
        for index, w, c, k in zip(m1_indices, whole, clean, core):
            counterfactual.append({"target_index": int(index), "stage": stage,
                "C0_whole_distance": w, "C1_GT_clean_intersection_distance": c,
                "C2_core_distance": k, "C1_minus_C0": c-w, "C2_minus_C0": k-w})
    pd.DataFrame(counterfactual).to_csv(output / "metrics" / "counterfactual_metrics.csv", index=False)
    (output / "metrics" / "bootstrap_ci.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    print(json.dumps({"event": "metrics_done", "coverage": coverage,
                      "stages": len(stage_rows)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
