"""Run offline load/finite/dimension/determinism smoke tests per source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .source import SemanticSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path)
    args = parser.parse_args()
    source = SemanticSource(args.name, args.model_dir, args.text_dir)
    texts = ["histopathology image of tumor tissue", "H&E stained tissue showing stromal tissue"]
    image = Image.new("RGB", (224, 224), (151, 102, 138))
    text = source.encode_text(texts)
    first = source.encode_image([image, image])
    second = source.encode_image([image, image])
    result = {
        "source": args.name,
        "text_shape": list(text.shape),
        "image_shape": list(first.shape),
        "finite": bool(torch.isfinite(text).all() and torch.isfinite(first).all()),
        "norm_error": float(max((text.norm(dim=-1)-1).abs().max(), (first.norm(dim=-1)-1).abs().max())),
        "same_batch_drift": float((first[0]-first[1]).abs().max()),
        "repeat_drift": float((first-second).abs().max()),
    }
    if not result["finite"] or text.shape[1] != first.shape[1] or result["norm_error"] > 1e-4 or result["repeat_drift"] >= 1e-5:
        raise AssertionError(result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
