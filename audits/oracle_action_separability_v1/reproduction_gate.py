"""Check sealed HQMR/DLAG/RACC anchors before the GT-free feature pass."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ANCHORS = {"hqmr": .6557244403737567, "oracle_A1": .6638007242653466,
           "oracle_B1": .6798242332322914, "oracle_AB": .6916322205519831,
           "racc_A": .6562874967422022, "racc_G": .6091274974732586}
CHECKPOINT_SHA = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "dlag", "racc", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    dlag_gate = args.dlag/"00_reproduction_gate.json"
    dlag_metrics = args.dlag/"metrics/oracle_metrics.json"
    racc_metrics = args.racc/"metrics/final_result.json"
    source = {str(p): sha(p) for p in (args.checkpoint,dlag_gate,dlag_metrics,racc_metrics)}
    dg=json.loads(dlag_gate.read_text())
    dm=json.loads(dlag_metrics.read_text())
    rm=json.loads(racc_metrics.read_text())
    observed={"hqmr":float(dg["hqmr_miou"]),
              **{key:float(dm[key]["mIoU"]) for key in ("oracle_A1","oracle_B1","oracle_AB")},
              "racc_A":float(rm["metrics"]["P1_RACC_A"]["mIoU"]),
              "racc_G":float(rm["metrics"]["P2_RACC_G"]["mIoU"])}
    checks={key:abs(observed[key]-expected)<1e-10 for key,expected in ANCHORS.items()}
    checks["checkpoint_sha_match"] = source[str(args.checkpoint)]==CHECKPOINT_SHA
    checks["dlag_reproduction_pass"] = bool(dg["pass"])
    checks["dlag_checkpoint_match"] = dg["checkpoint_sha256"]==CHECKPOINT_SHA
    checks["racc_reproduction_pass"] = rm.get("baseline_reproduction")=="PASS"
    output={"pass":all(checks.values()),"checks":checks,"observed":observed,
            "expected":ANCHORS,"source_sha256":source,"parameter_updates":0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output),flush=True)
    if not output["pass"]:
        raise AssertionError("Frozen reproduction gate failed; STOP")


if __name__=="__main__":
    main()
