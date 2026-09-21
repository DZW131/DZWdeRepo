"""Independently replay serialized primary metrics and artifact counts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output
    decision = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    split = json.loads((out / "manifests" / "ssqa_patient_split_v1.json").read_text(encoding="utf-8"))
    dev, hold = set(split["dev_patient_ids"]), set(split["holdout_patient_ids"])
    if dev & hold or len(dev | hold) != 22: raise AssertionError("Patient split invalid")
    totals = {}
    for partition in ("dev", "holdout"):
        frame = pd.read_csv(out / partition / "hard_m1.csv")
        totals[partition] = {}
        for name, sub in frame.groupby("source"):
            expected = decision[partition.upper()][name]
            rate = float(np.average((sub.true_rival_margin > 0).astype(float), weights=sub.area))
            top1 = float(sub.top1.mean())
            if abs(rate - expected["HTRP_area"]) > 1e-6 or abs(top1 - expected["HTop1"]) > 1e-6:
                raise AssertionError(f"Serialized metrics changed: {partition}/{name}")
            if set(sub.patient_id) & (hold if partition == "dev" else dev):
                raise AssertionError(f"Patient leakage: {partition}/{name}")
            totals[partition][name] = {"components": len(sub), "HTRP_area": rate, "HTop1": top1}
    if totals["dev"]["PLIP"]["components"] + totals["holdout"]["PLIP"]["components"] != 5037:
        raise AssertionError("Hard-M1 count changed")
    panels = len(list((out / "visualizations").rglob("*.png")))
    if panels != sum(sum(groups.values()) for groups in decision["PANELS"].values()):
        raise AssertionError("Representative panel count changed")
    if not (out / "SSQA_v1_External_Semantic_Source_Qualification_Final_Report.md").is_file():
        raise FileNotFoundError("Final report missing")
    print(json.dumps({"decision": decision["FINAL_DECISION"], "metrics_verified": totals,
                      "patient_groups": len(dev | hold), "panels": panels}, sort_keys=True))


if __name__ == "__main__": main()
