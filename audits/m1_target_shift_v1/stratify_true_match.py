"""Stratify the true-class-matched sensitivity that informs the final decision."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    target = pd.read_parquet(output / "target_cohorts.parquet").reset_index(drop=True)
    pairs = pd.read_csv(output / "metrics" / "true_class_matched_tp_pairs.csv")
    m1_indices = target.index[target.cohort == "M1"].to_numpy()
    quartile = np.searchsorted(np.quantile(target.iloc[m1_indices].area, [.25, .5, .75]),
                               target.area.to_numpy(), side="right") + 1
    rows = []
    for stage in ("H5_pre", "H5_context", "H4_input", "K4"):
        metric = pd.read_parquet(output / "metrics" / f"{stage}_whole.parquet")
        for label, chosen in (
            *( (f"class_{cls}", pairs.m1_index[target.iloc[pairs.m1_index].true_class.to_numpy() == cls])
               for cls in range(4)),
            ("purity_lt_070", pairs.m1_index[target.iloc[pairs.m1_index].purity.to_numpy() < .7]),
            ("purity_ge_070", pairs.m1_index[target.iloc[pairs.m1_index].purity.to_numpy() >= .7]),
            *( (f"area_Q{q}", pairs.m1_index[quartile[pairs.m1_index.to_numpy()] == q])
               for q in range(1, 5)),
        ):
            subset = pairs[pairs.m1_index.isin(chosen)]
            left = metric.iloc[subset.m1_index.to_numpy()].d_true.to_numpy()
            right = metric.iloc[subset.tp_index.to_numpy()].d_true.to_numpy()
            valid = np.isfinite(left) & np.isfinite(right)
            left, right = left[valid], right[valid]
            if len(left) < 10:
                continue
            rows.append({"stage": stage, "stratum": label, "n": len(left),
                         "M1_median": float(np.median(left)), "TP_median": float(np.median(right)),
                         "distance_ratio": float(np.median(left) / max(np.median(right), 1e-8)),
                         "auroc": float(roc_auc_score(np.r_[np.ones(len(left)), np.zeros(len(right))],
                                                       np.r_[left, right]))})
    pd.DataFrame(rows).to_csv(output / "metrics" / "true_class_matched_strata.csv", index=False)
    print(pd.DataFrame(rows).query("stage == 'H5_pre'").to_string(index=False))


if __name__ == "__main__":
    main()
