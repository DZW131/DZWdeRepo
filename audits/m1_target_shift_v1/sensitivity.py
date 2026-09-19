"""Sensitivity to the class identity used for M1/TP area matching."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

STAGES = ("H5_pre", "H5_context", "H4_input", "K4")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    target = pd.read_parquet(output / "target_cohorts.parquet").reset_index(drop=True)
    matches = pd.read_csv(output / "metrics" / "matched_tp_pairs.csv")
    m1 = np.flatnonzero(target.cohort.to_numpy() == "M1")
    tp = np.flatnonzero(target.cohort.to_numpy() == "TP_candidate")
    log_area = np.log1p(target.area.to_numpy())
    true_matched = []
    for index in m1:
        valid = tp[target.true_class.to_numpy()[tp] == target.true_class.iloc[index]]
        if not len(valid):
            continue
        same_image = valid[target.image_id.to_numpy()[valid] == target.image_id.iloc[index]]
        if len(same_image):
            valid = same_image
        closest = valid[np.argmin(abs(log_area[valid] - log_area[index]))]
        true_matched.append((index, int(closest)))
    true_match = pd.DataFrame(true_matched, columns=("m1_index", "tp_index"))
    true_match.to_csv(output / "metrics" / "true_class_matched_tp_pairs.csv", index=False)
    rows = []
    rng = np.random.default_rng(42)
    for matching, pair in (("predicted_class", matches.drop_duplicates("m1_index")),
                           ("GT_true_class", true_match)):
        left_idx = pair.m1_index.to_numpy(np.int64)
        right_idx = pair.tp_index.to_numpy(np.int64)
        for stage in STAGES:
            metric = pd.read_parquet(output / "metrics" / f"{stage}_whole.parquet")
            for field in ("d_true", "ood_percentile"):
                left = metric[field].to_numpy()[left_idx]
                right = metric[field].to_numpy()[right_idx]
                valid = np.isfinite(left) & np.isfinite(right)
                left, right = left[valid], right[valid]
                auc = roc_auc_score(np.r_[np.ones(len(left)), np.zeros(len(right))],
                                    np.r_[left, right])
                boot = np.empty(2000, np.float64)
                ratio_boot = np.empty(2000, np.float64)
                for iteration in range(2000):
                    chosen = rng.integers(0, len(left), len(left))
                    boot[iteration] = roc_auc_score(np.r_[np.ones(len(left)), np.zeros(len(right))],
                                                    np.r_[left[chosen], right[chosen]])
                    ratio_boot[iteration] = np.median(left[chosen]) / max(np.median(right[chosen]), 1e-8)
                rows.append({"matching": matching, "stage": stage, "score": field,
                             "n_pairs": len(left), "auroc": float(auc),
                             "ci95_low": float(np.quantile(boot, .025)),
                             "ci95_high": float(np.quantile(boot, .975)),
                             "m1_median": float(np.median(left)),
                             "tp_median": float(np.median(right)),
                             "distance_ratio": float(np.median(left) / max(np.median(right), 1e-8)),
                             "ratio_ci95_low": float(np.quantile(ratio_boot, .025)),
                             "ratio_ci95_high": float(np.quantile(ratio_boot, .975))})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics" / "matching_sensitivity.csv", index=False)
    print(json.dumps({"event": "sensitivity_done", "matched_true_class": len(true_match),
                      "K4": frame[(frame.stage == "K4") & (frame.score == "d_true")][
                          ["matching", "auroc", "distance_ratio"]].to_dict("records")}), flush=True)


if __name__ == "__main__":
    main()
