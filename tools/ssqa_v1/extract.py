"""Extract frozen zero-shot source embeddings before opening any GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from numpy.lib.format import open_memmap

from .prepare import json_once, sha256
from .source import SemanticSource


def source_paths(name: str, model_root: Path) -> tuple[Path, Path | None]:
    if name == "PLIP": return model_root.parent / "vinid-plip", None
    if name == "QuiltNet": return model_root / "QuiltNet-B-16-PMB", model_root / "PubMedBERT"
    if name == "BiomedCLIP": return model_root / "BiomedCLIP", model_root / "PubMedBERT"
    raise ValueError(f"Source is not READY: {name}")


def embed_view(source: SemanticSource, paths: list[str], output: Path, batch: int) -> None:
    tmp = output.with_suffix(".partial.npy")
    if tmp.exists():
        tmp.unlink()
    arr = open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(paths), 512))
    for start in range(0, len(paths), batch):
        images = []
        for path in paths[start:start + batch]:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        values = source.encode_image(images).numpy()
        if not np.isfinite(values).all() or np.max(np.abs(np.linalg.norm(values, axis=1) - 1)) > 1e-4:
            raise AssertionError(f"Invalid image embedding at {start}")
        arr[start:start + len(values)] = values
        if (start + batch) % 1600 < batch or start + batch >= len(paths):
            arr.flush()
            print(json.dumps({"event": "SSQA_EMBED_PROGRESS", "source": source.name,
                              "view": output.stem, "components": min(start + batch, len(paths))}), flush=True)
    del arr
    tmp.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=("PLIP", "QuiltNet", "BiomedCLIP"), required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=24)
    args = parser.parse_args()
    out = args.output; manifests = out / "manifests"
    freeze = json.loads((manifests / "view_freeze_manifest.json").read_text(encoding="utf-8"))
    for key, path in (("component_views_sha256", out / "component_views.parquet"),
                      ("crop_protocol_sha256", manifests / "crop_protocol_manifest.json"),
                      ("prompt_sha256", manifests / "bcss_source_qualification_prompts_v1.yaml"),
                      ("patient_split_sha256", manifests / "ssqa_patient_split_v1.json")):
        if sha256(path) != freeze[key]: raise AssertionError(f"Frozen input changed: {key}")
    availability = json.loads((manifests / "source_availability_manifest.json").read_text(encoding="utf-8"))["sources"]
    if availability[args.name]["status"] != "READY": raise AssertionError(f"Source {args.name} not READY")
    model_dir, text_dir = source_paths(args.name, args.model_root)
    weight = model_dir / ("pytorch_model.bin" if args.name == "PLIP" else "open_clip_pytorch_model.bin")
    if sha256(weight) != availability[args.name]["checkpoint_sha256"]: raise AssertionError("Checkpoint SHA mismatch")
    frame = pd.read_parquet(out / "component_views.parquet")
    if len(frame) != 11778 or not np.array_equal(frame.row_index.to_numpy(), np.arange(len(frame))):
        raise AssertionError("Frozen component order mismatch")
    prompts = yaml.safe_load((manifests / "bcss_source_qualification_prompts_v1.yaml").read_text(encoding="utf-8"))
    source = SemanticSource(args.name, model_dir, text_dir)
    prompt_texts = [template.replace("{CLASS}", phrase) for phrase in prompts["classes"] for template in prompts["templates"]]
    embeddings = source.encode_text(prompt_texts)
    if embeddings.shape != (20, 512) or not torch.isfinite(embeddings).all():
        raise AssertionError("Text embedding shape/finite test failed")
    prototypes = F.normalize(embeddings.reshape(4, 5, 512).mean(dim=1), dim=-1).numpy()
    cache = out / "source_cache" / args.name; cache.mkdir(parents=True, exist_ok=True)
    np.save(cache / "text_prototypes.npy", prototypes)
    for view, column in (("BBOX15", "bbox_path"), ("MASKED", "masked_path")):
        output = cache / f"{view}_embeddings.npy"
        if not output.exists(): embed_view(source, frame[column].tolist(), output, args.batch)
        arr = np.load(output, mmap_mode="r")
        if arr.shape != (len(frame), 512) or not np.isfinite(arr).all(): raise AssertionError(f"Invalid {view} cache")
    bbox = np.load(cache / "BBOX15_embeddings.npy", mmap_mode="r")
    masked = np.load(cache / "MASKED_embeddings.npy", mmap_mode="r")
    for view, values in (("BBOX15", bbox), ("MASKED", masked)):
        scores = np.asarray(values @ prototypes.T, dtype=np.float32)
        np.save(cache / f"{view}_scores.npy", scores)
    # Pre-registered deterministic 100-component replay, two independent forwards.
    subset = np.linspace(0, len(frame) - 1, 100, dtype=np.int32)
    drifts = {}
    for view, column in (("BBOX15", "bbox_path"), ("MASKED", "masked_path")):
        paths = frame.iloc[subset][column].tolist()
        repeats = []
        for _ in range(2):
            blocks = []
            for start in range(0, len(paths), args.batch):
                images = []
                for path in paths[start:start + args.batch]:
                    with Image.open(path) as opened: images.append(opened.convert("RGB"))
                blocks.append(source.encode_image(images).numpy())
            repeats.append(np.concatenate(blocks) @ prototypes.T)
        drifts[view] = float(np.max(np.abs(repeats[0] - repeats[1])))
        if drifts[view] >= 1e-5: raise AssertionError(f"NONDETERMINISTIC_SOURCE {args.name} {view}: {drifts[view]}")
    result = {"source": args.name, "status": "FROZEN_BEFORE_GT", "checkpoint_sha256": sha256(weight),
              "component_ids_sha256": freeze["component_ids_sha256"],
              "crop_protocol_sha256": freeze["crop_protocol_sha256"], "prompt_sha256": freeze["prompt_sha256"],
              "embedding_dimension": 512, "components": len(frame), "two_pass_100_score_max_drift": drifts,
              "sha256": {path.name: sha256(path) for path in sorted(cache.glob("*.npy"))}}
    json_once(cache / "manifest.json", result)
    print(json.dumps({"event": "SSQA_SOURCE_FROZEN", "source": args.name, "drift": drifts}), flush=True)


if __name__ == "__main__": main()
