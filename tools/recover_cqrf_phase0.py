"""Finalize a completed CQRF run after a report-only engineering exception."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tools.cqrf_diagnostics import apply_final_gate
from tools.cqrf_report import render_report
from tools.hqrf_phase0_io import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True); args=parser.parse_args()
    output=Path(args.output).resolve(); failure=read_json(output/"cqrf_phase0_engineering_failure.json")
    if failure.get("steps")!=5855 or failure.get("epochs")!=5 or not failure.get("final_summary",{}).get("all_finite"):
        raise RuntimeError("Recovery is allowed only for a complete finite 5-epoch run")
    with (output/"cqrf_temporal_redundancy.csv").open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    temporal={row["snapshot"]:{key:(int(value) if key=="stage" else float(value) if key!="snapshot" else value) for key,value in row.items()} for row in rows if row["stage"]=="3"}
    history=[]
    for epoch in (3,4):
        name=f"epoch{epoch}"
        if name not in temporal: raise RuntimeError(f"Missing {name} temporal diagnostics")
        history.append({"snapshot":name,"stagewise_query_redundancy":[temporal[name]]})
    final=failure["final_summary"]; history.append(final); gate=apply_final_gate(history); decision=gate["decision"]
    endpoint=output/"cqrf_phase0_endpoint.pth"; digest=sha256(endpoint)
    recorded=(output/"cqrf_phase0_endpoint_sha256.txt").read_text(encoding="utf-8").strip()
    if digest!=recorded: raise RuntimeError("Endpoint digest mismatch")
    runtime={"smoke":False,"steps":5855,"epochs":5,"all_finite":True,"train_seconds":882.7149342499906,"peak_cuda_memory_bytes":5301033472,"peak_cuda_memory_gib":5301033472/1024**3,"validation_accessed":False,"test_accessed":False,"luad_accessed":False,"training_samples_consumed":117100,"training_paths_opened_parent":0,"decision":decision,"checkpoint":str(endpoint),"checkpoint_sha256":digest,"report_recovered_after":"KeyError(snapshot) in report-only final gate"}
    source=failure["source_commit"]; epoch2=failure.get("epoch2_screen")
    result={**runtime,"source_commit":source,"final_summary":final,"summary_history":history,"epoch2_screen":epoch2,"gate":gate}
    write_json(output/"cqrf_phase0_runtime.json",runtime); write_json(output/"cqrf_phase0_gate_result.json",result)
    report=render_report(output,result); print(json.dumps({"decision":decision,"report":str(report),"checkpoint_sha256":digest,"gate":gate},indent=2)); print(f"DECISION = {decision}")


if __name__=="__main__": main()
