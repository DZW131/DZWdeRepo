from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "CRRA_V0_be298c1"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_formal_artifact_integrity_and_decision():
    summary_path = ARTIFACT / "summary.json"
    metadata_path = ARTIFACT / "regions" / "metadata.csv"
    assert _sha256(summary_path) == "342a3c58ec0de03d50c923649f74250f6bfbb6e64c135b9267e8a2bf39281e99"
    assert _sha256(metadata_path) == "d410e588d16d33d1f4c3ede4f326ae5a50eb4a31f16ff4157ed71573a75e47fa"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["decision"]["decision"] == "CRRA_V0_NOGO"
    assert summary["decision"]["representation_flag"] == "REGION_REPRESENTATION_ROUTE_CLOSED"
    assert summary["stop_boundary"] == {
        "crsr": False,
        "extra_representation": False,
        "luad": False,
        "sshr_training": False,
        "test": False,
    }


def test_compact_artifact_counts_and_oof_metrics_recompute():
    summary = json.loads((ARTIFACT / "summary.json").read_text(encoding="utf-8"))
    metadata = pd.read_csv(ARTIFACT / "regions" / "metadata.csv")
    assert len(metadata) == 8480
    assert int(metadata.common_support.sum()) == 6954
    reference = None
    for name in ("whole", "core", "core_rim"):
        frame = pd.read_csv(ARTIFACT / "probes" / name / "oof_predictions.csv")
        assert len(frame) == 4738
        alignment = frame[["token_index", "fold", "gt_label"]]
        if reference is None:
            reference = alignment
        else:
            assert reference.equals(alignment)
        score = f1_score(
            frame.gt_label,
            frame.oof_prediction,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
        assert abs(score - summary["representations"][name]["macro_f1"]) < 1e-12
