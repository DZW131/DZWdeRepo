"""Markdown report writer for the frozen Phase-0R audit."""

from __future__ import annotations

import json
from pathlib import Path


def _pct(value):
    return f"{100.0 * value:.4f}%"


def write_report(output_dir, summary):
    output_dir = Path(output_dir)
    baseline = summary["baseline"]
    oracle = summary["oracle"]
    primary = summary["primary"]
    decision = summary["decision"]
    lines = [
        "# SSHR Phase-0R Region-Centric Representation Feasibility Audit",
        "",
        "## 1. Executive decision",
        "",
        f"**{decision['label']}**",
        "",
        decision["interpretation"],
        "",
        "## 2. Frozen protocol and parity",
        "",
        f"- Audit source commit: `{summary['provenance']['audit_commit']}`",
        f"- Frozen A0 source commit: `{summary['provenance']['a0_commit']}`",
        f"- Checkpoint: `{summary['provenance']['checkpoint']}`",
        f"- Checkpoint SHA256: `{summary['provenance']['checkpoint_sha256']}`",
        f"- Validation images/slides: {summary['dataset']['images']} / {summary['dataset']['slides']}",
        f"- Differing pixels versus released inference: {summary['parity']['differing_pixels']}",
        f"- Maximum metric difference: {summary['parity']['maximum_metric_difference']:.3e}",
        "",
        "## 3. Baseline and region oracle",
        "",
        "| Protocol | mIoU | mDice | ΔmIoU |",
        "|---|---:|---:|---:|",
        f"| Official A0 final | {_pct(baseline['mean_iou'])} | {_pct(baseline['mean_dice'])} | — |",
        f"| Shape-preserving majority oracle | {_pct(oracle['mean_iou'])} | {_pct(oracle['mean_dice'])} | {oracle['gain_miou_pp']:+.4f} pp |",
        "",
        f"Recoverable error fraction: {oracle['recovery_fraction']:.4f}.",
        "",
        "## 4. Region purity and error taxonomy",
        "",
        f"Primary area ≥ 8 regions: {primary['n_regions']}. Pure-region fraction: {primary['pure_fraction']:.4f}.",
        "",
        "See `tables/region_purity_by_class.csv` and `tables/taxonomy_error_mass.csv` for the complete breakdown.",
        "",
        "## 5. Frozen feature probes",
        "",
        "| Representation | Accuracy | Macro-F1 | Balanced accuracy | Segmentation ΔmIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["probes"].items():
        lines.append(
            f"| {name} | {item['accuracy']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['balanced_accuracy']:.4f} | {item['segmentation_gain_miou_pp']:+.4f} pp |"
        )
    lines += [
        "",
        f"Region−BBox accuracy: {decision['region_bbox_gain']:+.4f}; "
        f"Region−Centroid accuracy: {decision['region_centroid_gain']:+.4f}.",
        "",
        f"Positive slide-held-out folds for Region relabeling: {decision['positive_folds']}/5.",
        "",
        "## 6. Area sensitivity and representation geometry",
        "",
        "The fixed area thresholds 1, 8, and 32 are reported in `tables/probe_results.csv`. "
        "Silhouette, Davies–Bouldin, and between/within scatter diagnostics are in "
        "`tables/representation_cluster_metrics.csv`.",
        "",
        "## 7. Qualitative audit",
        "",
        "The 32 automatically selected cases and four fixed-category panels are recorded in "
        "`tables/qualitative_selection.csv` and `figures/`. No examples were hand-picked.",
        "",
        "## 8. Decision evidence",
        "",
        f"- Oracle gain: {oracle['gain_miou_pp']:+.4f} pp",
        f"- Region probe gain over BBox: {decision['region_bbox_gain']:+.4f}",
        f"- Region probe gain over Centroid: {decision['region_centroid_gain']:+.4f}",
        f"- Region relabeling gain: {decision['region_seg_gain_pp']:+.4f} pp",
        f"- Oracle recovery fraction: {oracle['recovery_fraction']:.4f}",
        f"- Mixed-boundary share of error mass: {decision['mixed_error_fraction']:.4f}",
        f"- Region+Geo relabeling gain: {decision['region_geo_seg_gain_pp']:+.4f} pp",
        "",
        "## 9. Scope guard",
        "",
        "This was a validation-only diagnostic. It did not train a model, inspect test data, "
        "change inference, or alter the frozen SSHR A0 architecture.",
        "",
    ]
    (output_dir / "SSHR_Phase0R_Region_Centric_Feasibility_Report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
