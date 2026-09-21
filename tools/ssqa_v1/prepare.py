"""Freeze SSQA prompts, patient split, and GT-free component views."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from network.cirv import extract_regions


CLASSES = ("tumor tissue", "stromal tissue", "lymphocytic inflammatory infiltrate", "necrotic tissue")
TEMPLATES = (
    "histopathology image of {CLASS}",
    "H&E stained tissue showing {CLASS}",
    "microscopic H&E appearance of {CLASS}",
    "histology image containing {CLASS}",
    "{CLASS} tissue on H&E histopathology",
)
EXPECTED = {"images": 3418, "components": 11778}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise FileExistsError(f"Frozen file differs: {path}")
    else:
        path.write_text(value, encoding="utf-8")


def patient_id(image_id: str) -> str:
    match = re.match(r"^(TCGA-[^-]+-[^-]+)", image_id)
    if match is None:
        raise ValueError(f"Cannot recover TCGA patient ID from {image_id}")
    return match.group(1)


def source_manifest(model_root: Path) -> dict:
    specs = {
        "PLIP": ("vinid/plip", model_root.parent / "vinid-plip" / "pytorch_model.bin", "MIT", 224, "ViT-B/32", "CLIP text transformer", 512, 32, "official CLIPImageProcessor", "local frozen model"),
        "CONCH": ("MahmoodLab/CONCH", None, "CC-BY-NC-ND-4.0", 224, "ViT-B/16", "CONCH text encoder", 512, 16, "official CONCH preprocessing", "gated; HTTP 401 without approved access"),
        "CPLIP": ("iyyakuttiiyappan/CPLIP", None, "MIT", 224, "CTransPath/Swin", "PubMedBERT", 512, 4, "official CPLIP validation transform", "official Google Drive artifact fails ZIP CRC; missing model_configs in official code"),
        "QuiltNet": ("wisdomik/QuiltNet-B-16-PMB", model_root / "QuiltNet-B-16-PMB" / "open_clip_pytorch_model.bin", "MIT", 224, "ViT-B/16", "PubMedBERT", 512, 16, "OpenCLIP model preprocess_cfg", "HF commit deb13a5596f2ff72a501fa121b26b3a4ed81705d"),
        "BiomedCLIP": ("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", model_root / "BiomedCLIP" / "open_clip_pytorch_model.bin", "MIT", 224, "ViT-B/16", "BiomedBERT/PubMedBERT", 512, 16, "OpenCLIP model preprocess_cfg", "HF commit 9f341de24bfb00180f1b847274256e9b65a3a32e"),
        "CPathCLIP": ("PathFoundation/CPath-Omni", None, "requires Virchow2 license", None, "Virchow2 + official delta", "CPath-CLIP text encoder", None, None, "official reconstruction", "Virchow2 base and delta absent on server"),
    }
    entries = {}
    for name, (official, weight, license_name, size, vision, text, dim, patch, preprocessing, code_commit) in specs.items():
        status = "READY" if name in ("PLIP", "QuiltNet", "BiomedCLIP") else ("SKIP_INCOMPATIBLE" if name == "CPLIP" else "SKIP_ACCESS")
        if status == "READY" and (weight is None or not weight.is_file()):
            raise FileNotFoundError(f"READY weight missing: {name}: {weight}")
        entries[name] = {
            "name": name, "official_repo_or_model_id": official, "weight_available": bool(weight and weight.is_file()),
            "license": license_name, "input_size": size, "vision_encoder": vision, "text_encoder": text,
            "embedding_dimension": dim, "patch_size": patch, "native_preprocessing": preprocessing,
            "checkpoint_sha256": sha256(weight) if weight and weight.is_file() else None,
            "code_commit_or_access_note": code_commit, "status": status,
        }
    if sum(value["status"] == "READY" for key, value in entries.items() if key != "PLIP") < 2:
        raise RuntimeError("SSQA_INSUFFICIENT_SOURCES")
    return {"protocol": "SSQA-v1", "sources": entries, "minimum_source_gate": "PASS", "training": False}


def freeze_split(frame: pd.DataFrame) -> dict:
    # True class/Hard-M1 are not visible before GT. Use only predicted class and
    # GT-free component area/count as a balancing proxy; report true balance later.
    stats = []
    for patient, sub in frame.groupby("patient_id", sort=True):
        areas = np.asarray([sub.loc[sub.baseline_class == cls, "area"].sum() for cls in range(4)], dtype=np.float64)
        stats.append((patient, np.r_[len(sub), float(sub.area.sum()), areas]))
    rng = np.random.default_rng(20260921)
    rng.shuffle(stats)
    stats.sort(key=lambda item: -item[1][1])
    vectors = np.stack([item[1] for item in stats]); total = vectors.sum(0)
    n_dev = round(0.6 * len(stats)); dev, dev_sum = [], np.zeros(vectors.shape[1])
    for i, (patient, vector) in enumerate(stats):
        left = len(stats) - i
        places = n_dev - len(dev)
        if places == left:
            assign_dev = True
        elif places == 0:
            assign_dev = False
        else:
            scale = np.maximum(total, 1)
            with_dev = np.square((dev_sum + vector - 0.6 * total) / scale).sum()
            without_dev = np.square((dev_sum - 0.6 * total) / scale).sum()
            assign_dev = bool(with_dev <= without_dev)
        if assign_dev:
            dev.append(patient); dev_sum += vector
    dev_set = set(dev)
    holdout = sorted(patient for patient, _ in stats if patient not in dev_set)
    return {
        "seed": 20260921, "unit": "TCGA patient", "fraction_target": 0.6,
        "stratification": "GT-free predicted-class component area, total component area/count; no GT or Hard-M1 labels",
        "dev_patient_ids": sorted(dev), "holdout_patient_ids": holdout,
        "dev_gt_free_component_count": int(dev_sum[0]), "dev_gt_free_area": int(dev_sum[1]),
        "holdout_gt_free_component_count": int(total[0] - dev_sum[0]), "holdout_gt_free_area": int(total[1] - dev_sum[1]),
        "dev_predicted_class_area": [int(x) for x in dev_sum[2:]],
        "holdout_predicted_class_area": [int(x) for x in total[2:] - dev_sum[2:]],
        "true_class_counts": "sealed until embeddings frozen",
    }


def bbox15_square(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Empty frozen component")
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    side = min(224, max(1, math.ceil(max((x1 - x0) * 1.3, (y1 - y0) * 1.3))))
    x = min(max(int(round((x0 + x1 - side) / 2)), 0), 224 - side)
    y = min(max(int(round((y0 + y1 - side) / 2)), 0), 224 - side)
    if x > x0 or y > y0 or x + side < x1 or y + side < y1:
        raise AssertionError("Square crop lost component pixels")
    return x, y, x + side, y + side


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umrf", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output; manifests = out / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((args.umrf / "01_evidence_freeze_manifest.json").read_text(encoding="utf-8"))
    for filename, digest in freeze["sha256"].items():
        if sha256(args.umrf / filename) != digest:
            raise AssertionError(f"UMRF frozen input changed: {filename}")
    frame = pd.read_parquet(args.umrf / "component_evidence.parquet")
    ids = np.load(args.umrf / "image_ids.npy", allow_pickle=False).astype(str)
    if len(ids) != EXPECTED["images"] or len(frame) != EXPECTED["components"]:
        raise AssertionError(f"Frozen universe count mismatch: {len(ids)}, {len(frame)}")
    if frame[["image_id", "baseline_class", "component_id"]].duplicated().any():
        raise AssertionError("Duplicate frozen component IDs")
    frame["patient_id"] = frame.image_id.map(patient_id)
    frame["row_index"] = np.arange(len(frame), dtype=np.int32)
    component_keys = "\n".join(f"{r.image_id}|{r.baseline_class}|{r.component_id}" for r in frame.itertuples())
    component_sha = hashlib.sha256(component_keys.encode()).hexdigest()
    json_once(manifests / "source_availability_manifest.json", source_manifest(args.model_root))
    prompt = {"version": "v1", "classes": list(CLASSES), "templates": list(TEMPLATES)}
    prompt_path = manifests / "bcss_source_qualification_prompts_v1.yaml"
    yaml_text = yaml.safe_dump(prompt, allow_unicode=True, sort_keys=True)
    if prompt_path.exists():
        if prompt_path.read_text(encoding="utf-8") != yaml_text:
            raise FileExistsError("Frozen prompts differ")
    else:
        prompt_path.write_text(yaml_text, encoding="utf-8")
    json_once(manifests / "prompt_manifest.json", {"sha256": sha256(prompt_path), "class_count": 4, "templates_per_class": 5})
    split = freeze_split(frame)
    json_once(manifests / "ssqa_patient_split_v1.json", split)
    json_once(manifests / "challenge_set_manifest.json", {
        "frozen_universe_components": len(frame), "frozen_images": len(ids),
        "component_ids_sha256": component_sha,
        "umrf_evidence_sha256": freeze["sha256"],
        "hard_m1_expected_after_gt": 5037, "m1_expected_after_gt": 4402,
        "gt_opened": False,
    })
    protocol = {"primary": "BBOX15", "secondary": "MASKED", "image_size": [224, 224],
                "component_connectivity": 8, "source": "frozen UMRF baseline map",
                "expansion_per_bbox_dimension": 0.15, "square": True, "clip_to_image": True,
                "masked_outside_fill": "mean RGB of BBOX15 crop", "gt_used": False}
    json_once(manifests / "crop_protocol_manifest.json", protocol)
    bbox_root = out / "views" / "BBOX15"; masked_root = out / "views" / "MASKED"
    bbox_root.mkdir(parents=True, exist_ok=True); masked_root.mkdir(parents=True, exist_ok=True)
    bank = np.load(args.umrf / "gt_free_prediction_maps.uint8.npy", mmap_mode="r")
    records = []
    for image_index, image_id in enumerate(ids):
        sub = frame[frame.image_index == image_index]
        paths = list((args.val_root / "img").glob(f"{image_id}.*"))
        if len(paths) != 1:
            raise FileNotFoundError(f"Expected exactly one RGB tile for {image_id}: {paths}")
        image = np.asarray(Image.open(paths[0]).convert("RGB"))
        if image.shape != (224, 224, 3):
            raise AssertionError(f"Unexpected tile shape {image_id}: {image.shape}")
        regions = extract_regions(bank[0, image_index])
        if len(regions) != len(sub):
            raise AssertionError(f"Component replay count changed: {image_id}")
        for (_, row), region in zip(sub.iterrows(), regions):
            if int(row.baseline_class) != int(region["class_id"]) or int(row.component_id) != int(region["component_id"]) or int(row.area) != int(region["area"]):
                raise AssertionError(f"Component replay identity changed: {image_id}")
            box = bbox15_square(region["mask"])
            x0, y0, x1, y1 = box
            crop = image[y0:y1, x0:x1].copy()
            masked = crop.copy(); inside = region["mask"][y0:y1, x0:x1]
            fill = np.rint(crop.mean(axis=(0, 1))).astype(np.uint8)
            masked[~inside] = fill
            filename = f"{int(row.row_index):05d}.png"
            bbox_path = bbox_root / filename; masked_path = masked_root / filename
            if not bbox_path.exists(): Image.fromarray(crop).save(bbox_path)
            if not masked_path.exists(): Image.fromarray(masked).save(masked_path)
            records.append({"row_index": int(row.row_index), "image_id": image_id, "patient_id": row.patient_id,
                            "baseline_class": int(row.baseline_class), "component_id": int(row.component_id),
                            "area": int(row.area), "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                            "bbox_path": str(bbox_path), "masked_path": str(masked_path)})
        if (image_index + 1) % 400 == 0:
            print(json.dumps({"event": "SSQA_GT_FREE_CROPS", "images": image_index + 1, "components": len(records)}), flush=True)
    table = pd.DataFrame(records)
    if len(table) != len(frame): raise AssertionError("Incomplete view universe")
    table_path = out / "component_views.parquet"
    if table_path.exists():
        if not table.equals(pd.read_parquet(table_path)):
            raise FileExistsError("Frozen crop index differs")
    else:
        table.to_parquet(table_path, index=False)
    json_once(manifests / "view_freeze_manifest.json", {"component_ids_sha256": component_sha,
        "component_views_sha256": sha256(table_path), "crop_protocol_sha256": sha256(manifests / "crop_protocol_manifest.json"),
        "prompt_sha256": sha256(prompt_path), "patient_split_sha256": sha256(manifests / "ssqa_patient_split_v1.json"),
        "gt_opened": False, "views": len(table) * 2})
    print(json.dumps({"event": "SSQA_VIEWS_FROZEN", "components": len(table), "patients": len(split["dev_patient_ids"]) + len(split["holdout_patient_ids"])}), flush=True)


if __name__ == "__main__": main()
