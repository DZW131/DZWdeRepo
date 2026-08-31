#!/usr/bin/env bash
# A finite job, never a GPU polling service. Exit75 means resource admission failed.
set -Eeuo pipefail
cd /home/duyanhong/DZWdeRepo-rddr-phase2b112
PYTHON=/home/duyanhong/miniconda3/envs/sshr5090/bin/python
RUN_DIR="${1:?Provide a NEW absolute output directory under /home/duyanhong/experiments/RDDR_PHASE2B112}"
case "$RUN_DIR" in /home/duyanhong/experiments/RDDR_PHASE2B112/*) ;; *) exit 64 ;; esac
test ! -e "$RUN_DIR"
"$PYTHON" -m unittest discover -s tests -p test_rddr_phase2b112.py -v
"$PYTHON" tools/run_rddr_phase2b112.py --output "$RUN_DIR"
"$PYTHON" tools/verify_rddr_phase2b112.py --run-dir "$RUN_DIR"
"$PYTHON" tools/analyze_rddr_phase2b112.py --run-dir "$RUN_DIR" --report docs/rddr_phase2b112_short_horizon_optimization_report.md
"$PYTHON" tools/verify_rddr_phase2b112.py --run-dir "$RUN_DIR" --post-analysis
"$PYTHON" tools/analyze_rddr_phase2b112.py --run-dir "$RUN_DIR" --report docs/rddr_phase2b112_short_horizon_optimization_report.md --report-only
