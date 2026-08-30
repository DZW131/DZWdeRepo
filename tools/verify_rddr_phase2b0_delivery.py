"""Independent NumPy/CSV cross-checks; does not import the audit implementation."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cm_values(matrix):
    matrix = np.asarray(matrix, dtype=float)
    diag = matrix.diagonal(axis1=-2, axis2=-1)
    union = matrix.sum(-1)+matrix.sum(-2)-diag
    iou = np.divide(diag, union, out=np.full_like(diag, np.nan), where=union>0)
    return diag.sum(-1)/matrix.sum(axis=(-2, -1)), np.nanmean(iou, axis=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    args = p.parse_args()
    root = Path(args.report)
    names = ["summary.json", "per_image.csv", "pair_metrics.csv", "purity.csv", "group_analysis.csv",
             "conflict_quintiles.csv", "boundary_interior.csv", "per_class.csv", "neighbor_estimator.csv",
             "top20_repair.csv", "deep_strata.csv", "effective_neighbors.csv", "bootstrap.csv",
             "runtime.json", "population_manifest.json"]
    assert all((root/("rddr_phase2b0_"+n)).is_file() for n in names)
    summary = json.loads((root/"rddr_phase2b0_summary.json").read_text(encoding="utf-8"))
    items = rows(root/"rddr_phase2b0_per_image.csv")
    assert len(items) == len({r["image_id"] for r in items}) == 3418
    vec = lambda k: np.array([float(r[k]) for r in items])
    fg = vec("all_targets")
    for partition in (("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct"),
                      ("Top20", "Bottom80"), ("Q1", "Q2", "Q3", "Q4", "Q5"),
                      ("boundary", "interior"), ("Deep_Correct", "Deep_Wrong"), ("class0", "class1", "class2", "class3")):
        assert np.array_equal(sum(vec(g+"_targets") for g in partition), fg), partition
    assert np.array_equal(vec("Top20_Deep_Correct_targets")+vec("Top20_Deep_Wrong_targets"), vec("Top20_targets"))
    conf = {k: np.array([json.loads(r[k+"_confusion"]) for r in items], dtype=np.int64)
            for k in ("U", "SR", "SC", "SRSC", "raw", "deep", "oracle")}
    for k, value in conf.items():
        if k != "oracle":
            assert np.array_equal(value.sum((1, 2)), fg)
    estimator = rows(root/"rddr_phase2b0_neighbor_estimator.csv")
    for k, value in conf.items():
        a, i = cm_values(value.sum(0))
        row = next(r for r in estimator if r["group"] == "all" and r["estimator"] == k)
        assert abs(a-float(row["accuracy"])) < 1e-12
        assert abs(i-float(row["miou"])) < 1e-12
    for r in rows(root/"rddr_phase2b0_pair_metrics.csv"):
        if r["variant"] == "U":
            assert abs(float(r["auroc"])-.5) < 1e-12
            assert abs(float(r["auprc"])-float(r["prevalence"])) < 1e-12
    # Same bootstrap seed, independent code path and CSV roundtrip.
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(items), (32, len(items)), dtype=np.int32)
    measured = {}
    for k in ("SRSC_image_balanced_pair_AUROC", "SRSC_minus_U_purity", "SRSC_Corrected_minus_Harmed_purity",
              "Harmed_SRSC_minus_U_purity", "SRSC_mean_N_eff", "target_AUROC_purity", "target_AUROC_purity_gain", "target_AUROC_negative_wrong_mass"):
        measured[k] = np.nanmean(vec(k)[idx], axis=1)
    a_u, i_u = cm_values(conf["U"][idx].sum(1))
    a_s, i_s = cm_values(conf["SRSC"][idx].sum(1))
    measured["SRSC_minus_U_neighbor_accuracy"] = a_s-a_u
    measured["SRSC_minus_U_neighbor_mIoU"] = i_s-i_u
    nr = vec("SRSC_top20_repair")-vec("SRSC_top20_harm")-vec("U_top20_repair")+vec("U_top20_harm")
    measured["Top20_SRSC_minus_U_NetRepair"] = nr[idx].sum(1)/vec("Top20_targets")[idx].sum(1)
    reps = rows(root/"rddr_phase2b0_bootstrap_replicates.csv")
    assert len(reps) == 10000
    errors = {}
    for key, actual in measured.items():
        expected = np.array([float(r[key]) for r in reps[:32]])
        errors[key] = float(np.nanmax(np.abs(actual-expected)))
        assert errors[key] < 1e-12, (key, errors[key])
    for row in rows(root/"rddr_phase2b0_bootstrap.csv"):
        key = row["metric"]
        sample = np.array([float(r[key]) for r in reps])
        lo, hi = np.nanquantile(sample, [.025, .975])
        assert abs(lo-float(row["ci95_low"])) < 1e-12
        assert abs(hi-float(row["ci95_high"])) < 1e-12
    ci = summary["ci"]
    a, b, c = [ci[k] for k in ("SRSC_image_balanced_pair_AUROC", "SRSC_minus_U_purity", "SRSC_Corrected_minus_Harmed_purity")]
    da, di = [ci[k] for k in ("SRSC_minus_U_neighbor_accuracy", "SRSC_minus_U_neighbor_mIoU")]
    gates = dict(A=a["observed"]>=.65 and a["ci95_low"]>.5,
                 B=b["observed"]>=.03 and b["ci95_low"]>0 and ci["SRSC_mean_N_eff"]["observed"]>=5,
                 C=c["observed"]>0 and c["ci95_low"]>0 and ci["Harmed_SRSC_minus_U_purity"]["observed"]>0,
                 D=da["observed"]>0 and di["observed"]>0 and max(da["ci95_low"],di["ci95_low"])>0
                   and ci["Top20_SRSC_minus_U_NetRepair"]["observed"]>0)
    assert gates == summary["gates"]
    decision = ("RDDR_PHASE2B0_NOGO" if not gates["A"] or not gates["B"] else
                "RELATION_SIGNAL_NOT_CH_OUTCOME_SPECIFIC" if not gates["C"] else
                "RELATION_EXISTS_NO_PROPAGATION_UTILITY" if not gates["D"] else "RDDR_PHASE2B0_GO")
    assert summary["decision"] == decision
    report = (root/"rddr_phase2b0_reliable_relation_feasibility_report.md").read_text(encoding="utf-8")
    assert report.rstrip().endswith("DECISION = "+decision)
    result = dict(status="PASS", verified_images=3418, required_artifacts=len(names),
                  all_population_partitions_exact=True, all_estimator_confusions_exact=True,
                  independent_bootstrap_replicates=32, bootstrap_max_errors=errors,
                  all_10000_replicate_CI_quantiles_exact=True, independently_recomputed_gates=gates, decision=decision)
    (root/"rddr_phase2b0_independent_verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
