"""Fresh HQMR reproduction plus frozen UCRF-anchor gate for DLAG Oracle v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.ucrf_v1.gate import run as run_ucrf_gate

EXPECTED = {
    "deep_wrong_area": 0.9167845870531302,
    "corrective_available_area": 0.5126954028991616,
    "correction_suppression_area": 0.9376396360852077,
    "gate_excluded_area": 0.367186826805256,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def run(args: argparse.Namespace) -> None:
    # Fresh full-validation checkpoint/M1/formula reproduction.
    run_ucrf_gate(args)
    gate_path = args.output / "00_reproduction_gate.json"
    gate = json.loads(gate_path.read_text())
    summary = json.loads((args.ucrf_output / "metrics/cohort_summary.json").read_text())
    actual = {
        "deep_wrong_area": summary["overall"]["deep_wrong_area_rate"],
        "corrective_available_area": summary["overall"]["corrective_available_area_rate"],
        "correction_suppression_area": summary["overall"]["CSR_area"],
        "gate_excluded_area": summary["overall"]["gate_missing_true_area_rate"],
    }
    differences = {key: abs(actual[key]-EXPECTED[key]) for key in EXPECTED}
    anchor_pass = all(value <= 1e-12 for value in differences.values())
    gate["ucrf_anchor_replay"] = {"pass": anchor_pass, "actual": actual,
                                   "expected": EXPECTED, "absolute_difference": differences,
                                   "source": str(args.ucrf_output)}
    gate["dlag_config_sha256"] = digest(ROOT / "configs/audits/dlag_oracle_v1.yaml")
    gate["pass"] = bool(gate["pass"] and anchor_pass)
    gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps({"event": "dlag_gate_done", "pass": gate["pass"],
                      "anchors": actual}), flush=True)
    if not gate["pass"]:
        raise AssertionError("DLAG reproduction/anchor gate failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--ucrf-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
