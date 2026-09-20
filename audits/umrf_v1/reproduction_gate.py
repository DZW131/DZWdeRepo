"""Seal the already exact HQMR replay as the UMRF reproduction gate."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from audits.umrf_v1.core import EXPECTED_CHECKPOINT_SHA256, sha256

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--source-gate",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    source=json.loads(a.source_gate.read_text()); checkpoint_ok=sha256(a.checkpoint)==EXPECTED_CHECKPOINT_SHA256
    passed=bool(source.get("pass") and checkpoint_ok and abs(float(source.get("hqmr_miou",-1))-0.6557244403737567)<=1e-12)
    result={"pass":passed,"checkpoint_sha256":sha256(a.checkpoint),"expected_checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,
            "hqmr_miou":source.get("hqmr_miou"),"source_m1_components":source.get("m1_components"),"source_m1_pixels":source.get("m1_pixels"),
            "source_gate":str(a.source_gate),"source_gate_sha256":sha256(a.source_gate),"parameter_updates":0,
            "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    a.output.mkdir(parents=True,exist_ok=True); (a.output/"00_reproduction_gate.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"event":"UMRF_REPRODUCTION_GATE","pass":passed,"mIoU":result["hqmr_miou"]}),flush=True)
    if not passed: raise SystemExit("Reproduction gate failed")
if __name__=="__main__": main()

