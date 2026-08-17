"""Verify exact A0 checkpoint compatibility and the six-key CDSR delta."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from network.resnet38_cls import Net


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    state = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    uniform = Net(n_class=4, rectification_mode="uniform")
    uniform.load_state_dict(state, strict=True)
    cdsr = Net(n_class=4, rectification_mode="cdsr")
    incompatible = cdsr.load_state_dict(state, strict=False)
    expected_missing = sorted(
        name
        for name, _ in cdsr.named_parameters()
        if ".selective_gate." in name
    )
    uniform_parameters = sum(
        parameter.numel() for parameter in uniform.parameters()
    )
    cdsr_parameters = sum(parameter.numel() for parameter in cdsr.parameters())
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size_bytes": args.checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "uniform_strict_load": True,
        "cdsr_missing_keys": sorted(incompatible.missing_keys),
        "expected_cdsr_missing_keys": expected_missing,
        "cdsr_unexpected_keys": sorted(incompatible.unexpected_keys),
        "uniform_parameters": uniform_parameters,
        "cdsr_parameters": cdsr_parameters,
        "additional_parameters": cdsr_parameters - uniform_parameters,
        "pass": (
            sorted(incompatible.missing_keys) == expected_missing
            and not incompatible.unexpected_keys
            and cdsr_parameters - uniform_parameters == 6
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
