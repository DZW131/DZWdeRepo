#!/usr/bin/env python3
"""Re-apply the frozen confidence rule to completed coverage-audit artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_semantic_coverage_reachability import coverage_confidence, report_text
from tools.hqrf_phase0_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); output = Path(args.output_dir).resolve()
    result_path = output / "oracle/coverage_audit_final_result.json"
    result = json.loads(result_path.read_text())
    coverage = pd.read_csv(output / "basis_coverage/basis_max_coverage.csv").to_dict("records")
    rescue = pd.read_csv(output / "sshr_advantage/rescue_region_type_summary.csv").to_dict("records")
    confidence, evidence = coverage_confidence(coverage, rescue)
    if result["decision"]["decision"] == "QUERY_MASK_COVERAGE_LIMIT":
        result["decision"]["confidence"] = confidence
        result["decision"]["evidence"]["confidence_rule"] = evidence
    write_json(result_path, result)
    write_json(output / "provenance/coverage_audit_confidence_rule.json", evidence)
    report = output / "report/Semantic_Coverage_Propagation_Reachability_Audit_Report.md"
    report.write_text(report_text(result))
    print(json.dumps({"decision": result["decision"]["decision"],
                      "confidence": result["decision"]["confidence"],
                      "evidence": evidence, "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
