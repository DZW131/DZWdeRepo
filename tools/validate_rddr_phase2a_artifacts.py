"""Read-only, CPU-only cross-check of complete Phase-2A result artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric(hist):
    hist = np.asarray(hist, dtype=np.float64).copy()
    hist[4, 4] = 0
    diagonal = np.diag(hist)[:4]
    support = (hist.sum(0) + hist.sum(1))[:4]
    return float(np.mean(diagonal / (support - diagonal))), float(np.mean(2 * diagonal / support))


def validate(root, phase0=None):
    root = Path(root)
    summary = json.loads((root / "rddr_phase2a_summary.json").read_text())
    assert summary["images"] == 3418
    assert summary["engineering"]["test_used"] is False
    assert summary["engineering"]["luad_used"] is False
    assert summary["engineering"]["checkpoint_selection"] == "FINAL epoch25 only"
    assert summary["engineering"]["additional_trainable_parameters"] == 0
    assert summary["engineering"]["semantic_preservation"]["max_abs_diff"] == 0
    for row in summary["engineering"]["official_inference_parity"].values():
        assert row["images"] == 8 and row["mismatched_prediction_pixels"] == 0
    curves = read_csv(root / "rddr_phase2a_training_curves.csv")
    for variant in ("GS", "RCS"):
        rows = [row for row in curves if row["variant"] == variant]
        assert [int(row["epoch"]) for row in rows] == list(range(1, 26))
        assert int(rows[-1]["optimizer_global_step"]) == 29275
        assert all(float(row["optimizer_momentum"]) == 0.0005 for row in rows)
        lr = np.array([float(row["lr_backbone_weight"]) for row in rows])
        assert np.isfinite(lr).all() and np.all(np.diff(lr) < 0)
        assert np.isfinite([float(row["loss_cls"]) for row in rows]).all()
    audits = summary["engineering"]["optimizer_audits"]
    assert audits["GS"]["optimizer_groups"] == audits["RCS"]["optimizer_groups"]
    assert all(row["all_parameters_grouped_exactly_once"] for row in audits.values())
    per_image = read_csv(root / "rddr_phase2a_per_image.csv")
    assert len(per_image) == 3418 and len({row["image_id"] for row in per_image}) == 3418
    metrics = {}
    for variant in ("C0", "GS", "RCS"):
        hist = sum(np.array([
            [int(row[f"{variant}_hist_{i}_{j}"]) for j in range(5)]
            for i in range(5)
        ], dtype=np.int64) for row in per_image)
        miou, dice = metric(hist)
        np.testing.assert_allclose(
            [miou, dice],
            [summary["metrics"][variant]["Final"]["mIoU"], summary["metrics"][variant]["Final"]["mDice"]],
            rtol=0, atol=1e-14,
        )
        metrics[variant] = {"mIoU": miou, "mDice": dice}
    bootstrap = read_csv(root / "rddr_phase2a_bootstrap.csv")
    assert len(bootstrap) == 10000
    for label, key in (("RCS-C0", "RCS_minus_C0"), ("RCS-GS", "RCS_minus_GS"), ("GS-C0", "GS_minus_C0")):
        values = np.array([float(row[key]) for row in bootstrap])
        evidence = summary["bootstrap"][label]
        assert evidence["seed"] == 42 and evidence["resamples"] == 10000
        np.testing.assert_allclose(
            [values.mean(), *np.quantile(values, [.025, .975])],
            [evidence["bootstrap_mean"], evidence["ci95_low"], evidence["ci95_high"]],
            rtol=0, atol=1e-14,
        )
    for row in bootstrap:
        assert abs(float(row["RCS_minus_C0"]) - float(row["RCS_minus_GS"]) - float(row["GS_minus_C0"])) < 1e-14
    frozen_counts = {}
    if phase0:
        previous = json.loads(Path(phase0).read_text())
        expected_top20 = next(row["flagged_pixels"] for row in previous["score_bins"]
                              if row["score"] == "S_JS" and row["top_fraction"] == .2)
        assert summary["fixed_strata"]["RCS"]["Top20"]["pixels"] == expected_top20
        phase0_rows = {row["image_id"]: row for row in read_csv(Path(phase0).parent / "rddr_phase0_per_image.csv")}
        for row in per_image:
            old = phase0_rows[row["image_id"]]
            assert int(row["frozen_top20_pixels"]) == int(old["S_JS_top20_flagged"])
            for group in ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct"):
                assert int(row[f"frozen_{group}_pixels"]) == int(old[f"ch_{group}_count"])
        for variant in ("GS", "RCS"):
            for row in summary["ch_transition"][variant]:
                expected = previous["ch_transition"][row["group"]]["count"]
                actual = row["pixels"]
                frozen_counts[row["group"]] = {"phase0": expected, "phase2a": actual}
                assert actual == expected, f"Frozen C0 group drift: {row['group']}"
    g = {name: row["pass"] for name, row in summary["gates"].items()}
    m = summary["metrics"]
    b = summary["bootstrap"]
    ch = {variant: {row["group"]: row for row in rows}
          for variant, rows in summary["ch_transition"].items()}
    computed = {
        "A": m["RCS"]["Final"]["mIoU"] > m["C0"]["Final"]["mIoU"] and b["RCS-C0"]["ci95_low"] >= 0,
        "B": m["RCS"]["Final"]["mIoU"] > m["GS"]["Final"]["mIoU"] and (
            b["RCS-GS"]["ci95_low"] >= 0 or (
                m["RCS"]["CAM28_1"]["mIoU"] > m["GS"]["CAM28_1"]["mIoU"]
                and summary["fixed_strata"]["RCS"]["Top20"]["net_repair"] > summary["fixed_strata"]["GS"]["Top20"]["net_repair"]
            )),
        "C": m["RCS"]["CAM28_1"]["mIoU"] >= m["C0"]["CAM28_1"]["mIoU"]
            and summary["zones"]["RCS"]["interior_gt_7"]["accuracy"] - summary["zones"]["C0"]["interior_gt_7"]["accuracy"] >= -.001
            and summary["object_size"]["RCS"]["large"]["diagnostic_size_restricted_mIoU"] - summary["object_size"]["C0"]["large"]["diagnostic_size_restricted_mIoU"] >= -.002,
        "D": ch["RCS"]["Harmed_by_CH"]["net_repair"] > 0
            and ch["RCS"]["Harmed_by_CH"]["net_repair"] > ch["GS"]["Harmed_by_CH"]["net_repair"]
            and ch["RCS"]["Stable_Correct"]["net_repair"] >= -.001,
    }
    assert g == computed
    decision = (
        "RDDR_PHASE2A_GO" if all(g.values()) else
        "CONTEXT_REDUCTION_WORKS_SPATIAL_SPECIFICITY_FAIL" if g["A"] and g["C"] and g["D"] and not g["B"] else
        "CONTEXT_SUPPRESSION_SEMANTIC_DAMAGE" if not g["C"] else
        "LOCAL_CH_HARM_REDUCED_NO_GLOBAL_GAIN" if g["D"] and not g["A"] else
        "RDDR_PHASE2A_NOGO"
    )
    assert decision == summary["decision"]
    report = (root / "rddr_phase2a_dross_aware_context_suppression_report.md").read_text()
    assert report.rstrip().splitlines()[-1] == "DECISION = " + decision
    assert all(sha in report for sha in summary["checkpoint_sha256"].values())
    return {"status": "PASS", "images": 3418, "bootstrap_replicates": 10000,
            "metrics_recomputed_from_per_image_confusion": metrics,
            "frozen_ch_group_counts": frozen_counts, "decision": decision}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir")
    parser.add_argument("--phase0-summary")
    args = parser.parse_args()
    print(json.dumps(validate(args.result_dir, args.phase0_summary), indent=2))
