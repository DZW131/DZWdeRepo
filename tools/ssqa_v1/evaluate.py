"""GT-unseal, preregistered SSQA gates, patient bootstrap, and final report."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from network.cirv import extract_regions
from .prepare import json_once, sha256


NAMES = ("Tumor", "Stroma", "Inflammation", "Necrosis")


def fraction(numerator: np.ndarray, denominator: np.ndarray | None = None, weights: np.ndarray | None = None) -> float:
    if denominator is None:
        denominator = np.ones(len(numerator), bool)
    selected = np.asarray(denominator, bool)
    if not selected.any(): return float("nan")
    if weights is None: return float(np.mean(np.asarray(numerator)[selected]))
    weight = np.asarray(weights, np.float64)[selected]
    return float(np.average(np.asarray(numerator, np.float64)[selected], weights=weight)) if weight.sum() else float("nan")


def metric_row(df: pd.DataFrame, scores: np.ndarray, plip: np.ndarray, controls: pd.DataFrame) -> dict:
    idx = df.row_index.to_numpy(np.int64)
    truth = df.true_class.to_numpy(np.int64); rival = df.sequential5_pred.to_numpy(np.int64)
    values = scores[idx]; baseline = plip[idx]
    margin = values[np.arange(len(df)), truth] - values[np.arange(len(df)), rival]
    plip_margin = baseline[np.arange(len(df)), truth] - baseline[np.arange(len(df)), rival]
    positive = margin > 0; top1 = values.argmax(1) == truth
    area = df.area.to_numpy(np.float64)
    cidx = controls.row_index.to_numpy(np.int64); ct = controls.true_class.to_numpy(np.int64)
    cs = scores[cidx]; cpred = cs.argmax(1)
    control_margin = cs[np.arange(len(controls)), ct] - np.where(np.eye(4, dtype=bool)[ct], -np.inf, cs).max(1) if len(controls) else np.array([])
    per_class = {str(c): fraction(positive, truth == c, area) for c in range(4)}
    prediction_frequency = {str(c): float(np.mean(values.argmax(1) == c)) for c in range(4)} if len(df) else {}
    true_frequency = {str(c): float(np.mean(truth == c)) for c in range(4)} if len(df) else {}
    confusion = [[int(np.sum((truth == t) & (values.argmax(1) == p))) for p in range(4)] for t in range(4)]
    result = {
        "components": len(df), "area": float(area.sum()),
        "HTRP_area": fraction(positive, weights=area), "HTRP_component": fraction(positive),
        "HTop1": fraction(top1), "NRR_over_PLIP": fraction(positive, plip_margin <= 0),
        "mean_margin": float(np.mean(margin)), "median_margin": float(np.median(margin)),
        "margin_q25": float(np.quantile(margin, .25)), "margin_q75": float(np.quantile(margin, .75)),
        "true_rank_mean": float(np.mean(1 + np.sum(values > values[np.arange(len(df)), truth, None], axis=1))),
        "rival_rank_mean": float(np.mean(1 + np.sum(values > values[np.arange(len(df)), rival, None], axis=1))),
        "rival_top1_rate": float(np.mean(values.argmax(1) == rival)),
        "per_class_HTRP_area": per_class, "class_breadth_ge55": int(sum(x >= .55 for x in per_class.values() if np.isfinite(x))),
        "class_breadth_gt50": int(sum(x > .50 for x in per_class.values() if np.isfinite(x))),
        "ControlTop1": float(np.mean(cpred == ct)) if len(controls) else float("nan"),
        "ControlMargin_mean": float(np.mean(control_margin)) if len(controls) else float("nan"),
        "matched_control_count": len(controls),
        "prediction_frequency": prediction_frequency, "true_frequency": true_frequency,
        "bias_ratio": {str(c): float(prediction_frequency[str(c)] / true_frequency[str(c)]) if true_frequency[str(c)] else float("nan") for c in range(4)},
        "confusion_matrix": confusion,
        "SOURCE_CLASS_COLLAPSE": bool(max(prediction_frequency.values()) > .70),
    }
    result["SOURCE_NOGO"] = bool(result["HTRP_area"] <= .525 or result["mean_margin"] <= 0 or
        sum(x < .5 for x in per_class.values() if np.isfinite(x)) >= 2 or result["SOURCE_CLASS_COLLAPSE"])
    result["QUALIFIED_DEV"] = bool(result["HTRP_area"] >= .60 and result["HTop1"] >= .35 and
        result["NRR_over_PLIP"] >= .15 and result["class_breadth_ge55"] >= 3 and
        result["ControlTop1"] >= .55 and not result["SOURCE_CLASS_COLLAPSE"])
    result["STRONG_DEV"] = bool(result["QUALIFIED_DEV"] and result["HTRP_area"] >= .65 and
        result["HTop1"] >= .45 and result["NRR_over_PLIP"] >= .20 and
        result["ControlTop1"] >= .60 and result["class_breadth_gt50"] == 4)
    return result


def match_controls(frame: pd.DataFrame, hard: pd.DataFrame, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    available = frame[frame.baseline_correct & frame.evaluable].copy()
    chosen = []
    groups = ("true_class", "area_quartile", "confidence_quartile")
    for key, target in hard.groupby(list(groups), sort=True):
        options = available
        for column, value in zip(groups, key): options = options[options[column] == value]
        if len(options):
            ids = options.index.to_numpy(); rng.shuffle(ids)
            chosen.extend(ids[:min(len(ids), len(target))].tolist())
    return available.loc[chosen].sort_values("row_index").reset_index(drop=True)


def bootstrap_patient(hard: pd.DataFrame, controls: pd.DataFrame, scores: np.ndarray, plip: np.ndarray, seed: int = 42, n: int = 2000) -> dict:
    patients = sorted(set(hard.patient_id) | set(controls.patient_id))
    pindex = {p: i for i, p in enumerate(patients)}
    hidx = hard.row_index.to_numpy(np.int64); t = hard.true_class.to_numpy(np.int64); r = hard.sequential5_pred.to_numpy(np.int64)
    s = scores[hidx]; b = plip[hidx]; a = hard.area.to_numpy(np.float64)
    margin = s[np.arange(len(hard)), t] - s[np.arange(len(hard)), r]
    plip_fail = b[np.arange(len(hard)), t] <= b[np.arange(len(hard)), r]
    positive = margin > 0; top1 = s.argmax(1) == t
    cidx = controls.row_index.to_numpy(np.int64); ct = controls.true_class.to_numpy(np.int64)
    ctop1 = scores[cidx].argmax(1) == ct if len(controls) else np.array([], bool)
    sums = np.zeros((len(patients), 10), np.float64)
    for i, patient in enumerate(hard.patient_id):
        row = sums[pindex[patient]]
        row[0] += a[i] * positive[i]; row[1] += a[i]; row[2] += top1[i]; row[3] += 1
        row[4] += positive[i] and plip_fail[i]; row[5] += plip_fail[i]
        row[6] += margin[i]; row[7] += 1
    for i, patient in enumerate(controls.patient_id):
        row = sums[pindex[patient]]; row[8] += ctop1[i]; row[9] += 1
    rng = np.random.default_rng(seed); draws = rng.integers(0, len(patients), size=(n, len(patients)))
    sample = sums[draws].sum(1)
    metrics = (("HTRP_area", 0, 1), ("HTop1", 2, 3), ("NRR_over_PLIP", 4, 5),
               ("mean_margin", 6, 7), ("ControlTop1", 8, 9))
    result = {"unit": "TCGA patient", "patients": len(patients), "resamples": n, "seed": seed}
    for name, numerator, denominator in metrics:
        values = np.divide(sample[:, numerator], sample[:, denominator], out=np.full(n, np.nan), where=sample[:, denominator] > 0)
        good = values[np.isfinite(values)]
        result[name] = [float(np.quantile(good, .025)), float(np.quantile(good, .975))] if len(good) else [float("nan"), float("nan")]
    return result


def rows_for_csv(hard: pd.DataFrame, scores: np.ndarray, plip: np.ndarray, name: str) -> pd.DataFrame:
    idx = hard.row_index.to_numpy(np.int64); t = hard.true_class.to_numpy(np.int64); r = hard.sequential5_pred.to_numpy(np.int64)
    s = scores[idx]; b = plip[idx]
    value = hard[["row_index", "image_id", "patient_id", "baseline_class", "component_id", "area", "true_class", "purity", "sequential5_pred"]].copy()
    value["source"] = name
    for c in range(4): value[f"score_c{c}"] = s[:, c]
    value["true_rival_margin"] = s[np.arange(len(hard)), t] - s[np.arange(len(hard)), r]
    value["true_rank"] = 1 + np.sum(s > s[np.arange(len(hard)), t, None], axis=1)
    value["rival_rank"] = 1 + np.sum(s > s[np.arange(len(hard)), r, None], axis=1)
    value["top1"] = s.argmax(1) == t
    value["plip_fail"] = b[np.arange(len(hard)), t] <= b[np.arange(len(hard)), r]
    return value


def phi(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, np.float64); y = np.asarray(right, np.float64)
    sx = x.std(); sy = y.std()
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy)) if sx and sy else float("nan")


def independence(hard: pd.DataFrame, scores: dict[str, np.ndarray], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    idx = hard.row_index.to_numpy(np.int64); t = hard.true_class.to_numpy(np.int64); r = hard.sequential5_pred.to_numpy(np.int64)
    names = sorted(scores)
    pred = {name: scores[name][idx].argmax(1) for name in names}
    correct = {name: pred[name] == t for name in names}
    positive = {name: scores[name][idx, t] > scores[name][idx, r] for name in names}
    rows = []
    for a, b in itertools.combinations(names, 2):
        err_a, err_b = ~correct[a], ~correct[b]
        rescue_a, rescue_b = positive[a], positive[b]
        pair = float(np.mean(rescue_a | rescue_b))
        rows.append({"source_a": a, "source_b": b, "prediction_agreement": float(np.mean(pred[a] == pred[b])),
                     "error_jaccard": float(np.sum(err_a & err_b) / max(np.sum(err_a | err_b), 1)),
                     "error_phi": phi(err_a, err_b), "true_rival_agreement": float(np.mean(rescue_a == rescue_b)),
                     "rescue_jaccard": float(np.sum(rescue_a & rescue_b) / max(np.sum(rescue_a | rescue_b), 1)),
                     "pair_oracle_positive_rate": pair,
                     "pair_oracle_gain_over_best": pair - max(float(np.mean(rescue_a)), float(np.mean(rescue_b)))})
    table = pd.DataFrame(rows)
    table[["source_a", "source_b", "prediction_agreement"]].to_csv(output / "agreement.csv", index=False)
    table[["source_a", "source_b", "error_jaccard", "error_phi"]].to_csv(output / "error_phi.csv", index=False)
    table[["source_a", "source_b", "true_rival_agreement", "rescue_jaccard"]].to_csv(output / "rescue_overlap.csv", index=False)
    table[["source_a", "source_b", "pair_oracle_positive_rate", "pair_oracle_gain_over_best"]].to_csv(output / "pair_oracle.csv", index=False)
    heatmaps = {
        "prediction_agreement": lambda a, b: float(np.mean(pred[a] == pred[b])),
        "error_phi": lambda a, b: phi(~correct[a], ~correct[b]),
        "rescue_overlap": lambda a, b: float(np.sum(positive[a] & positive[b]) / max(np.sum(positive[a] | positive[b]), 1)),
    }
    for title, compute in heatmaps.items():
        matrix = np.asarray([[compute(a, b) for b in names] for a in names])
        fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(matrix, vmin=-1 if title == "error_phi" else 0,
                                                                     vmax=1, cmap="coolwarm" if title == "error_phi" else "Blues")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names)
        fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(output / f"{title}_heatmap.png", dpi=160); plt.close(fig)
    return {"pairs": rows, "multi_source_oracle_positive_rate": float(np.mean(np.logical_or.reduce(list(positive.values())))),
            "unit": "Q_DEV Hard-M1", "oracle_deployable": False}


def render_cases(hard: pd.DataFrame, controls: pd.DataFrame, scores: np.ndarray, plip: np.ndarray,
                 name: str, views: pd.DataFrame, bank: np.ndarray, bank_indices: dict[str, int], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    hidx = hard.row_index.to_numpy(np.int64); t = hard.true_class.to_numpy(np.int64); r = hard.sequential5_pred.to_numpy(np.int64)
    sm = scores[hidx, t] - scores[hidx, r]; pm = plip[hidx, t] - plip[hidx, r]
    control_idx = controls.row_index.to_numpy(np.int64)
    conditions = {
        "A_novel_rescue": hard.loc[(pm <= 0) & (sm > 0)],
        "B_shared_failure": hard.loc[(pm <= 0) & (sm <= 0)],
        "C_control_harm": controls.loc[scores[control_idx].argmax(1) != controls.true_class.to_numpy(np.int64)],
    }
    counts = {}
    for group, frame in conditions.items():
        selected = frame.sort_values(["patient_id", "image_id", "component_id"]).head(20)
        counts[group] = len(selected)
        for rank, row in enumerate(selected.itertuples(), 1):
            view = views.iloc[int(row.row_index)]
            image = Image.open(view.bbox_path).convert("RGB")
            value = scores[int(row.row_index)]
            truth = int(row.true_class); rival = int(row.sequential5_pred)
            fig, ax = plt.subplots(figsize=(5, 5)); ax.imshow(image)
            prediction = bank[0, bank_indices[str(row.image_id)]]
            matches = [region["mask"] for region in extract_regions(prediction)
                       if int(region["class_id"]) == int(row.baseline_class) and int(region["component_id"]) == int(row.component_id)]
            if len(matches) != 1: raise AssertionError("Panel component identity changed")
            patch = matches[0][int(view.y0):int(view.y1), int(view.x0):int(view.x1)]
            outline = patch & ~ndimage.binary_erosion(patch)
            rgba = np.zeros((*patch.shape, 4), np.float32); rgba[outline] = (1, 1, 0, .9)
            ax.imshow(rgba)
            ax.set_title(f"{name} | {group} {rank:02d}\nGT={NAMES[truth]} rival={NAMES[rival]}\n"
                + " ".join(f"{NAMES[c][:3]}:{value[c]:+.3f}" for c in range(4))
                + f"\ntrue rank={1+np.sum(value>value[truth])} rival rank={1+np.sum(value>value[rival])} margin={value[truth]-value[rival]:+.3f}", fontsize=8)
            ax.axis("off"); fig.tight_layout(); fig.savefig(output / f"{group}_{rank:02d}.png", dpi=120); plt.close(fig)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umrf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output; manifests = out / "manifests"
    availability = json.loads((manifests / "source_availability_manifest.json").read_text(encoding="utf-8"))["sources"]
    ready = [name for name, spec in availability.items() if spec["status"] == "READY"]
    if "PLIP" not in ready or len(ready) < 3: raise RuntimeError("SSQA_INSUFFICIENT_SOURCES")
    freeze = json.loads((manifests / "view_freeze_manifest.json").read_text(encoding="utf-8"))
    views_path = out / "component_views.parquet"
    if sha256(views_path) != freeze["component_views_sha256"]: raise AssertionError("View index altered")
    scores = {}
    for name in ready:
        cache = out / "source_cache" / name
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "FROZEN_BEFORE_GT" or manifest["component_ids_sha256"] != freeze["component_ids_sha256"]:
            raise AssertionError(f"Source cache not frozen: {name}")
        for filename, digest in manifest["sha256"].items():
            if sha256(cache / filename) != digest: raise AssertionError(f"Source cache changed: {name}/{filename}")
        scores[name] = np.load(cache / "BBOX15_scores.npy")
    views = pd.read_parquet(views_path)
    # No GT source is opened above this line.
    gt_path = args.umrf / "component_evidence_with_gt.parquet"
    gt_sha = sha256(gt_path)
    gt = pd.read_parquet(gt_path)
    keys = ["image_id", "baseline_class", "component_id"]
    frame = views.merge(gt[keys + ["true_class", "purity", "evaluable", "m1", "baseline_correct", "sequential5_pred", "sequential4_pred", "sequential3_pred", "sequential5_p0", "sequential5_p1", "sequential5_p2", "sequential5_p3"]], on=keys, validate="one_to_one", sort=False)
    frame = frame.sort_values("row_index").reset_index(drop=True)
    frame["hard_m1"] = frame.evaluable & (frame.sequential5_pred != frame.true_class) & (frame.sequential4_pred != frame.true_class) & (frame.sequential3_pred != frame.true_class)
    if int(frame.hard_m1.sum()) != 5037 or int(frame.m1.sum()) != 4402:
        raise AssertionError(f"Frozen challenge mismatch: Hard-M1={frame.hard_m1.sum()}, M1={frame.m1.sum()}")
    frame["area_quartile"] = pd.qcut(frame.area.rank(method="first"), 4, labels=False).astype(int)
    confidence = frame[[f"sequential5_p{c}" for c in range(4)]].max(axis=1)
    frame["confidence_quartile"] = pd.qcut(confidence.rank(method="first"), 4, labels=False).astype(int)
    split = json.loads((manifests / "ssqa_patient_split_v1.json").read_text(encoding="utf-8"))
    dev_set = set(split["dev_patient_ids"]); hold_set = set(split["holdout_patient_ids"])
    if set(frame.patient_id) != dev_set | hold_set or dev_set & hold_set: raise AssertionError("Patient split invalid")
    json_once(manifests / "gt_unseal_manifest.json", {"gt_table_sha256": gt_sha,
        "hard_m1": int(frame.hard_m1.sum()), "m1": int(frame.m1.sum()),
        "dev_true_class_counts": frame.loc[frame.hard_m1 & frame.patient_id.isin(dev_set), "true_class"].value_counts().sort_index().to_dict(),
        "holdout_true_class_counts": frame.loc[frame.hard_m1 & frame.patient_id.isin(hold_set), "true_class"].value_counts().sort_index().to_dict(),
        "source_caches_verified_before_gt": ready})
    cohorts = {"dev": frame[frame.patient_id.isin(dev_set)].copy(), "holdout": frame[frame.patient_id.isin(hold_set)].copy()}
    controls = {}; hard = {}
    for split_name, part in cohorts.items():
        hard[split_name] = part[part.hard_m1].copy()
        controls[split_name] = match_controls(part, hard[split_name], 42)
    results, bootstraps = {"dev": {}, "holdout": {}}, {"dev": {}, "holdout": {}}
    for name in ready:
        row = metric_row(hard["dev"], scores[name], scores["PLIP"], controls["dev"])
        results["dev"][name] = row
        bootstraps["dev"][name] = bootstrap_patient(hard["dev"], controls["dev"], scores[name], scores["PLIP"])
    candidates = [name for name in ready if name != "PLIP"]
    ranked = sorted(candidates, key=lambda name: (results["dev"][name]["HTRP_area"], results["dev"][name]["HTop1"]), reverse=True)
    selected = set(ranked[:2]) | {name for name in candidates if results["dev"][name]["STRONG_DEV"]}
    # PLIP is evaluated as a fixed holdout reference, never selected as a candidate.
    for name in ["PLIP", *[n for n in ranked if n in selected]]:
        row = metric_row(hard["holdout"], scores[name], scores["PLIP"], controls["holdout"])
        boot = bootstrap_patient(hard["holdout"], controls["holdout"], scores[name], scores["PLIP"])
        row["patient_CI95"] = boot
        row["HOLDOUT_CONFIRMED"] = bool(row["HTRP_area"] >= .58 and boot["HTRP_area"][0] > .50 and
            row["HTop1"] >= .33 and row["class_breadth_gt50"] >= 3 and
            not row["SOURCE_CLASS_COLLAPSE"] and results["dev"][name]["HTRP_area"] - row["HTRP_area"] <= .10)
        row["SEMANTIC_SOURCE_GO"] = bool(name != "PLIP" and results["dev"][name]["QUALIFIED_DEV"] and row["HOLDOUT_CONFIRMED"])
        row["STRONG_SEMANTIC_SOURCE_GO"] = bool(row["SEMANTIC_SOURCE_GO"] and row["HTRP_area"] >= .65 and
            row["HTop1"] >= .40 and row["NRR_over_PLIP"] >= .20 and row["ControlTop1"] >= .60)
        results["holdout"][name] = row; bootstraps["holdout"][name] = boot
    for split_name in ("dev", "holdout"):
        directory = out / split_name; directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"source": name, **{k: v for k, v in result.items() if np.isscalar(v)}} for name, result in results[split_name].items()]).to_csv(directory / "source_summary.csv", index=False)
        pd.concat([rows_for_csv(hard[split_name], scores[name], scores["PLIP"], name) for name in results[split_name]]).to_csv(directory / "hard_m1.csv", index=False)
        pd.DataFrame([{"source": name, **{f"c{c}": value["per_class_HTRP_area"][str(c)] for c in range(4)}} for name, value in results[split_name].items()]).to_csv(directory / "per_class.csv", index=False)
        pd.DataFrame([{"source": name, "mean": value["mean_margin"], "median": value["median_margin"], "q25": value["margin_q25"], "q75": value["margin_q75"]} for name, value in results[split_name].items()]).to_csv(directory / "margins.csv", index=False)
        pd.DataFrame([{"source": name, "row_index": int(row.row_index), "true_class": int(row.true_class),
                       "predicted_class": int(scores[name][int(row.row_index)].argmax())} for name in results[split_name] for row in controls[split_name].itertuples()]).to_csv(directory / "controls.csv", index=False)
        pd.DataFrame([{"source": name, "class": c, "prediction_frequency": result["prediction_frequency"][str(c)],
                       "true_frequency": result["true_frequency"][str(c)], "bias_ratio": result["bias_ratio"][str(c)]}
                      for name, result in results[split_name].items() for c in range(4)]).to_csv(directory / "class_bias.csv", index=False)
        (directory / "confusion_matrices.json").write_text(json.dumps({name: result["confusion_matrix"]
            for name, result in results[split_name].items()}, indent=2), encoding="utf-8")
        evaluable = cohorts[split_name][cohorts[split_name].evaluable]
        eidx = evaluable.row_index.to_numpy(np.int64); etrue = evaluable.true_class.to_numpy(np.int64)
        full_bias, full_confusion = [], {}
        for name in results[split_name]:
            epred = scores[name][eidx].argmax(1)
            full_confusion[name] = [[int(np.sum((etrue == t) & (epred == p))) for p in range(4)] for t in range(4)]
            for c in range(4):
                predicted = float(np.mean(epred == c)); true = float(np.mean(etrue == c))
                full_bias.append({"source": name, "class": c, "prediction_frequency": predicted,
                                  "true_frequency": true, "bias_ratio": predicted / true if true else float("nan")})
        pd.DataFrame(full_bias).to_csv(directory / "class_bias_full_evaluable.csv", index=False)
        (directory / "confusion_full_evaluable.json").write_text(json.dumps(full_confusion, indent=2), encoding="utf-8")
    indep = independence(hard["dev"], scores, out / "independence")
    secondary = {}
    for name in ready:
        cache = out / "source_cache" / name
        masked = np.load(cache / "MASKED_scores.npy")
        row = metric_row(hard["dev"], masked, scores["PLIP"], controls["dev"])
        area_groups = {}
        purity_groups = {}
        for quartile in range(4):
            part = hard["dev"][hard["dev"].area_quartile == quartile]
            area_groups[f"Q{quartile+1}"] = metric_row(part, scores[name], scores["PLIP"], controls["dev"])["HTRP_area"] if len(part) else float("nan")
        for label, lo, hi in (("lt_0.5", -1, .5), ("0.5_to_0.7", .5, .7), ("0.7_to_0.9", .7, .9), ("ge_0.9", .9, 2)):
            part = hard["dev"][(hard["dev"].purity >= lo) & (hard["dev"].purity < hi)]
            purity_groups[label] = metric_row(part, scores[name], scores["PLIP"], controls["dev"])["HTRP_area"] if len(part) else float("nan")
        secondary[name] = {"MASKED_HTRP_area": row["HTRP_area"], "BBOX15_HTRP_area": results["dev"][name]["HTRP_area"],
                           "CONTEXT_DEPENDENT_SEMANTICS": abs(row["HTRP_area"] - results["dev"][name]["HTRP_area"]) > .15,
                           "area_quartile_HTRP": area_groups, "purity_HTRP": purity_groups}
    (out / "bootstrap").mkdir(exist_ok=True)
    (out / "bootstrap" / "patient_cluster_ci.json").write_text(json.dumps(bootstraps, indent=2, allow_nan=True), encoding="utf-8")
    panels = {}
    bank = np.load(args.umrf / "gt_free_prediction_maps.uint8.npy", mmap_mode="r")
    bank_ids = np.load(args.umrf / "image_ids.npy", allow_pickle=False).astype(str)
    bank_indices = {image_id: index for index, image_id in enumerate(bank_ids)}
    for name in ready:
        panels[name] = render_cases(hard["dev"], controls["dev"], scores[name], scores["PLIP"],
                                   name, views, bank, bank_indices, out / "visualizations" / name)
    go = [name for name in selected if results["holdout"].get(name, {}).get("SEMANTIC_SOURCE_GO")]
    best_holdout = max(selected, key=lambda name: (results["holdout"][name]["HTRP_area"],
                        results["holdout"][name]["HTop1"])) if selected else None
    dual = False
    if len(go) >= 2:
        for pair in indep["pairs"]:
            if {pair["source_a"], pair["source_b"]}.issubset(go) and pair["error_phi"] < .5 and pair["pair_oracle_gain_over_best"] >= .10:
                dual = True
    all_candidate_nogo = all(results["dev"][name]["SOURCE_NOGO"] or
        results["holdout"].get(name, {}).get("SOURCE_NOGO", False) for name in candidates)
    if go:
        route = "QUALIFIED_VLM_TO_PRE_CCRA"
    elif dual:
        route = "SEMANTIC_COMPLEMENTARITY_AUDIT"
    elif all_candidate_nogo:
        route = "CLOSE_ZERO_SHOT_VLM_SEMANTIC_ROUTE"
    else:
        route = "INCONCLUSIVE_NO_QUALIFIED_SOURCE"
    decision = {"FINAL_DECISION": route, "READY_SOURCES": ready,
        "SKIPPED_SOURCES": {name: spec["status"] for name, spec in availability.items() if spec["status"] != "READY"},
        "BEST_DEV_SOURCE": ranked[0] if ranked else None, "BEST_HOLDOUT_SOURCE": best_holdout,
        "HOLDOUT_SELECTED": sorted(selected), "SEMANTIC_SOURCE_GO": bool(go),
        "ALL_TESTED_CANDIDATES_NOGO": all_candidate_nogo,
        "STRONG_SEMANTIC_SOURCE_GO": any(results["holdout"][name]["STRONG_SEMANTIC_SOURCE_GO"] for name in go),
        "DUAL_SOURCE_FOLLOWUP_JUSTIFIED": dual,
        "ZERO_SHOT_VLM_ROUTE": "OPEN" if go else ("CLOSED_BY_THIS_AUDIT" if all_candidate_nogo else "INCONCLUSIVE"),
        "NEXT_ROUTE": "best qualified source to I1_PRE_CONTEXT in a separately preregistered experiment" if go else
            ("pathology visual foundation representation + image-level learned semantic head" if all_candidate_nogo else "new preregistration required; do not integrate an unqualified source"),
        "JEV_SIDECAR": "SKIPPED_NO_API", "DEV": results["dev"], "HOLDOUT": results["holdout"],
        "INDEPENDENCE": indep, "SECONDARY": secondary, "PANELS": panels,
        "protocol_notes": ["No model training or CCRA injection", "Q-DEV all READY; Q-HOLDOUT top-2 non-PLIP plus strong DEV; PLIP holdout reference",
                           "Patient split frozen on GT-free predicted-class proxy; true-class balance disclosed only after source caches frozen",
                           "Dual-source follow-up conservatively requires two single-source GO decisions; theoretical oracle is not deployable"]}
    (out / "jev").mkdir(exist_ok=True)
    (out / "jev" / "jev_sidecar.json").write_text(json.dumps({"status": "SKIPPED_NO_API"}, indent=2), encoding="utf-8")
    (out / "decision.json").write_text(json.dumps(decision, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({"event": "SSQA_EVALUATED", "decision": route, "go_sources": go}), flush=True)


if __name__ == "__main__": main()
