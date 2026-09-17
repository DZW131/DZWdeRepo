#!/usr/bin/env python3
"""Create a pixel-level SHA256 manifest for BCSS-WSSS training patches."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainroot", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.trainroot).resolve()
    images = sorted(root.glob("*.png"))
    if len(images) != 23422:
        raise AssertionError(f"Expected 23422 BCSS training images, got {len(images)}")
    rows = []
    for index, path in enumerate(images, 1):
        pixels = np.asarray(Image.open(path).convert("RGB"))
        if pixels.shape != (224, 224, 3):
            raise AssertionError(f"Unexpected patch shape: {path.name} {pixels.shape}")
        rows.append({"filename": path.name,
                     "image_pixel_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()})
        if index % 2000 == 0:
            print(f"hashed={index}", flush=True)
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False, compression="zstd")
    print(f"manifest={output} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
