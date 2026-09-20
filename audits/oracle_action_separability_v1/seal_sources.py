"""Record all frozen Oracle-bank/UCRF source hashes before GT access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ("dlag","ucrf","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    args=p.parse_args();out=args.output
    if out.exists():raise FileExistsError(out)
    files=[args.dlag/f"counterfactuals/alpha_{k:03d}/predictions.npz" for k in (0,25,50,100,150,200,300,400)]
    files += [args.dlag/"gate_oracle/gate_exclusion_manifest.csv",
              args.ucrf/"metrics/component_event_table.parquet"]
    anchor=json.loads((out.parent/"00_reproduction_gate.json").read_text())
    if not anchor["pass"]:raise AssertionError("Anchor gate failed")
    hashes={str(path):sha(path) for path in files}
    for path,expected in anchor["source_sha256"].items():
        if sha(Path(path))!=expected:raise AssertionError(f"Frozen anchor source changed: {path}")
    out.write_text(json.dumps({"pass":True,"source_sha256":hashes,
                               "anchor_source_sha256":anchor["source_sha256"],
                               "gt_or_oracle_labels_read":False},indent=2),encoding="utf-8")
    print(json.dumps({"event":"ORACLE_SOURCES_SEALED","files":len(files)}),flush=True)


if __name__=="__main__":main()
