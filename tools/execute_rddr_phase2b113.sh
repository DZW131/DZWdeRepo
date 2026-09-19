#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/duyanhong/miniconda3/envs/sshr5090/bin/python}"
output_dir="${1:?usage: execute_rddr_phase2b113.sh OUTPUT_DIR}"
report_path="$repo_root/docs/rddr_phase2b113_parameter_gradient_attribution_report.md"

cd "$repo_root"
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite output: $output_dir" >&2
  exit 64
fi
if [[ -e "$report_path" ]]; then
  echo "Refusing to overwrite report: $report_path" >&2
  exit 64
fi

"$python_bin" -m unittest discover -s tests -p test_rddr_phase2b113.py -v
"$python_bin" tools/run_rddr_phase2b113.py --output "$output_dir"
"$python_bin" tools/verify_rddr_phase2b113.py --input "$output_dir"
"$python_bin" tools/analyze_rddr_phase2b113.py --input "$output_dir" --report "$report_path"
"$python_bin" tools/verify_rddr_phase2b113.py --input "$output_dir" --report "$report_path" --post-analysis
