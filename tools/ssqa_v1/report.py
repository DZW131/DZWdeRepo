"""Write the human-readable SSQA-v1 final experimental report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .prepare import sha256


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output
    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    availability = json.loads((out / "manifests" / "source_availability_manifest.json").read_text(encoding="utf-8"))["sources"]
    unseal = json.loads((out / "manifests" / "gt_unseal_manifest.json").read_text(encoding="utf-8"))
    split = json.loads((out / "manifests" / "ssqa_patient_split_v1.json").read_text(encoding="utf-8"))
    views = pd.read_parquet(out / "component_views.parquet")
    crop_sides = views.x1 - views.x0
    dev, hold = decision["DEV"], decision["HOLDOUT"]
    lines = ["# SSQA-v1 External Semantic Source Qualification — Final Report", "",
        "> BCSS seed42 · frozen HQMR/UMRF components · no training · BBOX15 primary · patient-group DEV/HOLDOUT", "",
        "## Executive decision", "",
        f"FINAL_DECISION = **{decision['FINAL_DECISION']}**<br>",
        f"READY_SOURCES = {', '.join(decision['READY_SOURCES'])}<br>",
        f"SKIPPED_SOURCES = {decision['SKIPPED_SOURCES']}<br>",
        f"BEST_DEV_SOURCE = {decision['BEST_DEV_SOURCE']}<br>",
        f"BEST_HOLDOUT_SOURCE = {decision['BEST_HOLDOUT_SOURCE']}<br>",
        f"SEMANTIC_SOURCE_GO = {decision['SEMANTIC_SOURCE_GO']}<br>",
        f"ALL_TESTED_CANDIDATES_NOGO = {decision['ALL_TESTED_CANDIDATES_NOGO']}<br>",
        f"STRONG_SEMANTIC_SOURCE_GO = {decision['STRONG_SEMANTIC_SOURCE_GO']}<br>",
        f"DUAL_SOURCE_FOLLOWUP_JUSTIFIED = {decision['DUAL_SOURCE_FOLLOWUP_JUSTIFIED']}<br>",
        f"ZERO_SHOT_VLM_ROUTE = {decision['ZERO_SHOT_VLM_ROUTE']}<br>",
        f"NEXT_ROUTE = {decision['NEXT_ROUTE']}<br>",
        f"JEV_SIDECAR = {decision['JEV_SIDECAR']}", ""]
    for name in availability:
        d = dev.get(name); h = hold.get(name)
        lines.extend([f"{name}: DEV_HTRP={pct(d['HTRP_area']) if d else '—'}, DEV_HTop1={pct(d['HTop1']) if d else '—'}, "
                      f"DEV_NRR={pct(d['NRR_over_PLIP']) if d else '—'}, "
                      f"HOLDOUT={pct(h['HTRP_area']) if h else '—'}; STATUS={availability[name]['status']}<br>"])
    lines.extend(["", "## Main results", "",
        "| Source | Dev HTRP area | Holdout HTRP area | Holdout Top1 | NRR vs PLIP (DEV) | Control Top1 (DEV) | Class breadth ≥55% | Collapse | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|"])
    for name, spec in availability.items():
        d = dev.get(name); h = hold.get(name)
        if not d:
            lines.append(f"| {name} | — | — | — | — | — | — | — | {spec['status']} |")
            continue
        label = ("SEMANTIC_SOURCE_GO" if h and h.get("SEMANTIC_SOURCE_GO") else
                 ("SOURCE_NOGO" if d["SOURCE_NOGO"] or (h and h["SOURCE_NOGO"]) else "NOT_QUALIFIED"))
        if d["QUALIFIED_DEV"] and h and not h.get("HOLDOUT_CONFIRMED"): label = "DEV_ONLY_NOT_CONFIRMED"
        lines.append(f"| {name} | {pct(d['HTRP_area'])} | {pct(h['HTRP_area']) if h else '—'} | "
                     f"{pct(h['HTop1']) if h else '—'} | {pct(d['NRR_over_PLIP'])} | {pct(d['ControlTop1'])} | "
                     f"{d['class_breadth_ge55']}/4 | {d['SOURCE_CLASS_COLLAPSE']} | {label} |")
    lines.extend(["", "### Per-class DEV Hard-M1 HTRP (area-weighted)", "",
                  "| Source | Tumor | Stroma | Inflammation | Necrosis |", "|---|---:|---:|---:|---:|"])
    for name, result in dev.items():
        p = result["per_class_HTRP_area"]
        lines.append(f"| {name} | {pct(p['0'])} | {pct(p['1'])} | {pct(p['2'])} | {pct(p['3'])} |")
    lines.extend(["", "### DEV predicted-class distribution on Hard-M1", "",
                  "| Source | Tumor | Stroma | Inflammation | Necrosis |", "|---|---:|---:|---:|---:|"])
    for name, result in dev.items():
        p = result["prediction_frequency"]
        lines.append(f"| {name} | {pct(p['0'])} | {pct(p['1'])} | {pct(p['2'])} | {pct(p['3'])} |")
    lines.extend(["", "Prediction/true frequencies, bias ratios and confusion matrices are saved both on Hard-M1 "
                  "(`dev/class_bias.csv`, `dev/confusion_matrices.json`) and on the entire evaluable DEV cohort "
                  "(`dev/class_bias_full_evaluable.csv`, `dev/confusion_full_evaluable.json`), with holdout equivalents "
                  "only for selected sources and the PLIP reference.", ""])
    lines.extend(["", "## Protocol and leakage controls", "",
        f"Frozen UMRF components: 11,778 over 3,418 tiles; exact Hard-M1={unseal['hard_m1']}, exact M1={unseal['m1']}. "
        "No components were selected or recropped using GT. Rival is the frozen sequential L5 class, not final segmentation.", "",
        "Each predicted 8-connected component was cropped from its own bounding box with a fixed 15% expansion "
        "per dimension, made square within the original 224×224 tile, preserving surrounding H&E tissue. "
        "MASKED uses the same crop with external pixels filled with crop-mean RGB. It never sets the primary gate.", "",
        f"GT-free crop-size caveat: {int((crop_sides <= 4).sum())} / {len(views)} crops are ≤4 px before native resizing, "
        f"while {int((crop_sides == 224).sum())} span the full 224 px tile. These are fixed consequences of the frozen "
        "component geometry, not a tuned view. Component-weighted Top1 and whole-tile/context sensitivity must be interpreted accordingly.", "",
        "All sources used the same four class phrases and five prompt templates. Each prompt embedding was normalized, "
        "averaged within class, and normalized again; image/text cosine and margins were computed in FP32 without logit scale. "
        "Official native tokenizers and image transforms were retained. No source was trained, tuned, or injected into CCRA.", "",
        f"Prompt SHA256: `{sha256(out / 'manifests' / 'bcss_source_qualification_prompts_v1.yaml')}`. "
        f"GT table SHA256 (opened after all READY source caches were frozen): `{unseal['gt_table_sha256']}`.", "",
        f"Patient split seed=20260921: {len(split['dev_patient_ids'])} DEV and {len(split['holdout_patient_ids'])} HOLDOUT patients. "
        "Because true classes and Hard-M1 membership were sealed until embedding freeze, split stratification used only frozen predicted-class area, "
        "component area and counts as proxy; actual true-class balance was disclosed after unseal.", "",
        f"DEV true-class Hard-M1 counts: {unseal['dev_true_class_counts']}; HOLDOUT: {unseal['holdout_true_class_counts']}. "
        "Controls were greedily matched within each split by GT class, component-area quartile and baseline-confidence quartile, "
        "without replacement; achieved counts appear below.", "",
        "Only the two highest DEV-ranked non-PLIP sources plus any additional STRONG_DEV source entered holdout. "
        "PLIP was evaluated there only as a fixed negative-control reference needed for NRR.", ""])
    lines.extend(["## Source availability and reproducibility", "",
        "| Source | Official ID | Checkpoint SHA256 | Status / reason |", "|---|---|---|---|"])
    for name, spec in availability.items():
        lines.append(f"| {name} | `{spec['official_repo_or_model_id']}` | `{spec['checkpoint_sha256'] or '—'}` | "
                     f"{spec['status']}: {spec['code_commit_or_access_note']} |")
    lines.extend(["", "PLIP uses frozen `vinid/plip` revision `67ade53ddd32195868f422585f72698ef5d15094`; "
        "the SSQA runtime used `transformers==4.35.2` and `open_clip_torch==2.23.0`. "
        "CONCH's official model endpoint denied unauthenticated access (HTTP 401); no mirror was substituted. "
        "Two independent downloads of the official CPLIP Google Drive file had identical SHA256 "
        "`ae30f04b869c3cefe8fff73335f42fdcbcbcaff45280d761be691789976190dc` and failed ZIP CRC at "
        "`archive/data/340`; official repository revision `82ee44972c7e967f97b40c53f18d7ea8e5e94ae9` also lacks the "
        "model configuration needed for reproducible loading. CPLIP was not evaluated. "
        "CPath-CLIP was skipped because the authorized Virchow2 base plus official delta were not present.", "",
        "## Margins, controls, stability", ""])
    for name, d in dev.items():
        h = hold.get(name)
        lines.append(f"- {name}: DEV true−rival mean={d['mean_margin']:+.5f}, median={d['median_margin']:+.5f}, "
                     f"Q25/Q75={d['margin_q25']:+.5f}/{d['margin_q75']:+.5f}; matched controls={d['matched_control_count']}, "
                     f"ControlTop1={pct(d['ControlTop1'])}; "
                     + (f"HOLDOUT HTRP patient-bootstrap 95% CI=[{pct(h['patient_CI95']['HTRP_area'][0])}, {pct(h['patient_CI95']['HTRP_area'][1])}], "
                        f"stable={d['HTRP_area']-h['HTRP_area'] <= .10}." if h else "holdout not opened for this source."))
    lines.extend(["", "Bootstrap used 2,000 resamples with TCGA patient as cluster, seed 42. "
        "Complete CIs for HTRP, Top1, NRR, controls and margin are in `bootstrap/patient_cluster_ci.json`.", "",
        "### MASKED view and component strata (secondary Q-DEV)", "",
        "| Source | BBOX15 HTRP | MASKED HTRP | Context-dependent >15 pp | Area quartiles Q1–Q4 | Purity <0.5 / 0.5–0.7 / 0.7–0.9 / ≥0.9 |",
        "|---|---:|---:|---|---|---|"])
    for name, value in decision["SECONDARY"].items():
        areas = value["area_quartile_HTRP"]; purity = value["purity_HTRP"]
        lines.append(f"| {name} | {pct(value['BBOX15_HTRP_area'])} | {pct(value['MASKED_HTRP_area'])} | "
            f"{value['CONTEXT_DEPENDENT_SEMANTICS']} | "
            + " / ".join(pct(areas[f"Q{i}"]) for i in range(1, 5)) + " | "
            + " / ".join(pct(purity[key]) for key in ("lt_0.5", "0.5_to_0.7", "0.7_to_0.9", "ge_0.9")) + " |")
    lines.extend(["",
        "## Source independence and oracle ceiling", "",
        f"Q-DEV multi-source theoretical positive-rate oracle: {pct(decision['INDEPENDENCE']['multi_source_oracle_positive_rate'])}. "
        "This GT oracle is not deployable and is not a GO result. Pair agreement, error Jaccard/phi and rescue overlap are in `independence/`. "
        f"Dual-source follow-up justified: {decision['DUAL_SOURCE_FOLLOWUP_JUSTIFIED']}.", "",
        "## Qualitative cases and secondary analyses", "",
        "Fixed Q-DEV panels per READY source use groups A (PLIP fails, source rescues), B (both fail), "
        "C (baseline-correct control harmed). Counts available: " + str(decision["PANELS"]) + ". "
        "All generated cases are in `visualizations/`; missing examples indicate the group had fewer than 20 cases. "
        "No holdout case was used to change prompt, model variant or crop. Whole-tile classification was not used for qualification.", "",
        "## Final interpretation", ""])
    if decision["SEMANTIC_SOURCE_GO"]:
        lines.append(f"{decision['BEST_HOLDOUT_SOURCE']} met the preregistered DEV and independent HOLDOUT gates. "
            "The next experiment may test a simple frozen-source injection at the already validated I1_PRE_CONTEXT interface. "
            "This audit itself did not implement or train that interface, and does not reopen PDSR-v1 or VPCA-v1.")
    elif decision["ALL_TESTED_CANDIDATES_NOGO"]:
        lines.append("No tested zero-shot source met both DEV and independent HOLDOUT qualification. "
            "Do not claim an HQMR segmentation gain or continue prompt/crop searches on these data. "
            "The preregistered next route is a pathology visual foundation representation with an image-level learned semantic head "
            "in a separate experiment; PDSR-v1, VPCA-v1 and CCRA remain unchanged. "
            "This route decision applies to the tested READY sources; it does not assign NOGO to unavailable CONCH, "
            "CPLIP or CPath-CLIP, which were never scored.")
    else:
        lines.append("No source is qualified, but not every candidate meets the prespecified SOURCE_NOGO rule. "
            "This is inconclusive for route closure; a new preregistration would be needed. No source should be integrated now.")
    lines.extend(["", "## Artifact index", "",
        "- `manifests/`: official availability, frozen prompt/split/crop protocol and GT-unseal evidence.",
        "- `source_cache/`: normalized prototypes, per-view embeddings/scores and integrity hashes.",
        "- `dev/` and `holdout/`: component-level scores, margins, controls, per-class summaries.",
        "- `independence/`: source agreement, error phi, rescue overlap and theoretical pair oracle.",
        "- `bootstrap/`: patient-cluster 95% confidence intervals.",
        "- `visualizations/`: fixed representative Q-DEV case panels.",
        "- `decision.json`: machine-readable preregistered GO/NOGO outcomes.", ""])
    path = out / "SSQA_v1_External_Semantic_Source_Qualification_Final_Report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(path), "sha256": sha256(path), "decision": decision["FINAL_DECISION"]}), flush=True)


if __name__ == "__main__": main()
