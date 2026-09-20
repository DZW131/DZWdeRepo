"""Freeze all GT-free UMRF-v1 evidence before opening BCSS masks."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.umrf_v1.core import EXPECTED_CHECKPOINT_SHA256, rules_from_probabilities, sha256, softmax_class
from audits.ucrf_v1.gate import load_model
from network.cirv import extract_regions
from network.hqmr import direct_affinity
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import TTA, normalize_cam, prediction_from_cam, presence, resize_unflip

MAP_KEYS = ("baseline", "sequential5", "sequential4", "sequential3",
            "common5", "common4", "common3",
            "sequential_r1", "sequential_r2", "sequential_r3",
            "common_r1", "common_r2", "common_r3")


def class_evidence(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("qc,qhw->chw", weights.float(), logits.float().sigmoid()).clamp(0, 1)


@torch.inference_mode()
def infer(model, image: torch.Tensor) -> tuple[dict[str, np.ndarray], dict]:
    hw = tuple(int(v) for v in image.shape[-2:])
    names = ("sequential5", "sequential4", "sequential3", "common5", "common4", "common3", "direct4", "direct3")
    views = {name: [] for name in names}; cams = []; gates = []
    errors = {"sequential_l4": 0.0, "sequential_l3": 0.0, "common_l5_identity": 0.0}
    shapes = None
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, output_flip in TTA:
        captured = {"k5": [], "k4": [], "k3": []}
        handles = [model.hqmr.scale5.key.register_forward_hook(lambda _m, _i, x: captured["k5"].append(x.detach())),
                   model.hqmr.scale4.key.register_forward_hook(lambda _m, _i, x: captured["k4"].append(x.detach())),
                   model.hqmr.scale3.key.register_forward_hook(lambda _m, _i, x: captured["k3"].append(x.detach()))]
        try:
            value = torch.flip(image, dims=input_flip) if input_flip else image
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(value, dummy, step=29275, hqmr_mode="full")
                item = output["stages"][2]["hqmr"]
                if tuple(map(len, (captured["k5"], captured["k4"], captured["k3"]))) != (2, 2, 1):
                    raise AssertionError("Unexpected HQMR key hook call pattern")
                k5, k4, k3 = captured["k5"][-1], captured["k4"][-1], captured["k3"][0]
                q0 = item["query0"]
                l5, l4, l3 = item["logits5"], item["logits4"], item["logits3"]
                d4, d3 = item["direct4"], item["direct3"]
                c5, c4, c3 = direct_affinity(q0, k5), direct_affinity(q0, k4), direct_affinity(q0, k3)
                u5 = F.interpolate(l5, d4.shape[-2:], mode="bilinear", align_corners=False)
                u4 = F.interpolate(l4, d3.shape[-2:], mode="bilinear", align_corners=False)
                weights = item["weights"][0]
                tensors = {"sequential5": l5[0], "sequential4": l4[0], "sequential3": l3[0],
                           "common5": c5[0], "common4": c4[0], "common3": c3[0],
                           "direct4": d4[0], "direct3": d3[0]}
            errors["sequential_l4"] = max(errors["sequential_l4"], float(((u5+d4).float()-l4.float()).abs().max()))
            errors["sequential_l3"] = max(errors["sequential_l3"], float(((u4+d3).float()-l3.float()).abs().max()))
            errors["common_l5_identity"] = max(errors["common_l5_identity"], float((c5.float()-l5.float()).abs().max()))
            shapes = {"q0": list(q0.shape), "q5": list(item["query5"].shape), "q4": list(item["query4"].shape),
                      "k5": list(k5.shape), "k4": list(k4.shape), "k3": list(k3.shape), "weights": list(weights.shape)}
            for name, tensor in tensors.items():
                evidence = class_evidence(tensor, weights)
                views[name].append(resize_unflip(evidence[None].float(), hw, output_flip).cpu().numpy())
            cams.append(resize_unflip(output["primary_output"].float(), hw, output_flip).cpu())
            gates.append(output["deep_gate"].detach().float().cpu())
        finally:
            for handle in handles: handle.remove()
    if max(errors.values()) > 1e-6:
        raise AssertionError(f"Forward identity failure: {errors}")
    probabilities = {name: softmax_class(np.mean(values, axis=0)) for name, values in views.items()}
    cam = normalize_cam(torch.stack(cams).mean(0).numpy())
    label = presence(torch.stack(gates).mean(0).numpy()[0])
    baseline = prediction_from_cam(cam, label, np.empty(hw)).astype(np.uint8)
    maps = {"baseline": baseline}
    for chain in ("sequential", "common"):
        ruled = rules_from_probabilities(*(probabilities[f"{chain}{s}"] for s in ("5", "4", "3")))
        for key, value in ruled.items(): maps[f"{chain}{key if key in ('5','4','3') else '_'+key}"] = value
    return {"maps": maps, "probabilities": probabilities}, {"errors": errors, "shapes": shapes}


def component_rows(bundle: dict, image_id: str, image_index: int) -> list[dict]:
    rows = []
    for region in extract_regions(bundle["maps"]["baseline"]):
        mask = region["mask"]
        row = {"image_id": image_id, "image_index": image_index,
               "component_id": int(region["component_id"]), "baseline_class": int(region["class_id"]),
               "area": int(region["area"])}
        for chain in ("sequential", "common"):
            for stage in ("5", "4", "3"):
                p = bundle["probabilities"][f"{chain}{stage}"][:, mask].mean(1)
                row.update({f"{chain}{stage}_p{c}": float(p[c]) for c in range(4)})
                order = np.argsort(p)
                row[f"{chain}{stage}_pred"] = int(order[-1])
                row[f"{chain}{stage}_margin"] = float(p[order[-1]]-p[order[-2]])
        for stage in ("4", "3"):
            p = bundle["probabilities"][f"direct{stage}"][:, mask].mean(1)
            row.update({f"direct{stage}_p{c}": float(p[c]) for c in range(4)})
            row[f"direct{stage}_pred"] = int(np.argmax(p))
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--val-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--reference-baseline", type=Path)
    p.add_argument("--num-workers", type=int, default=2)
    args = p.parse_args()
    if sha256(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256: raise AssertionError("Checkpoint SHA mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "01_evidence_freeze_manifest.json"
    if manifest_path.exists(): raise FileExistsError("Frozen evidence already exists; refusing overwrite")
    dataset = Stage1_InferDataset(str(args.val_root/"img"), img_size=224)
    if len(dataset) != 3418: raise AssertionError(f"Expected 3418 images, got {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = load_model(args.checkpoint)
    maps_path = args.output/"gt_free_prediction_maps.uint8.npy"
    map_store = open_memmap(maps_path, mode="w+", dtype=np.uint8, shape=(len(MAP_KEYS), len(dataset), 224, 224))
    rows, ids, max_errors, shapes = [], [], {"sequential_l4": 0., "sequential_l3": 0., "common_l5_identity": 0.}, None
    started = time.perf_counter()
    for index, (names, image) in enumerate(loader):
        image_id = str(names[0]); ids.append(image_id)
        bundle, audit = infer(model, image.cuda(non_blocking=True))
        for j, key in enumerate(MAP_KEYS): map_store[j, index] = bundle["maps"][key]
        rows.extend(component_rows(bundle, image_id, index))
        for key, value in audit["errors"].items(): max_errors[key] = max(max_errors[key], value)
        shapes = audit["shapes"]
        if (index+1) % 100 == 0 or index+1 == len(dataset):
            map_store.flush()
            print(json.dumps({"event":"umrf_freeze_progress","images":index+1,"components":len(rows),
                              "elapsed_s":round(time.perf_counter()-started,1)}), flush=True)
    del map_store
    ids_path=args.output/"image_ids.npy"; np.save(ids_path, np.asarray(ids))
    table_path=args.output/"component_evidence.parquet"; pd.DataFrame(rows).to_parquet(table_path,index=False)
    reference = {"checked": False}
    if args.reference_baseline:
        ref=np.load(args.reference_baseline, allow_pickle=False)
        current=np.load(maps_path,mmap_mode="r")[MAP_KEYS.index("baseline")]
        reference={"checked":True,"path":str(args.reference_baseline),"equal":bool(np.array_equal(current,ref["predictions"])),
                   "pixel_differences":int(np.count_nonzero(current != ref["predictions"]))}
        if not reference["equal"]: raise AssertionError(f"Exact baseline replay mismatch: {reference}")
    artifacts={p.name:sha256(p) for p in (maps_path,ids_path,table_path)}
    manifest={"status":"EVIDENCE_FROZEN_BEFORE_GT","gt_opened":False,"checkpoint_sha256":EXPECTED_CHECKPOINT_SHA256,
              "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "parameter_updates":0,"images":len(ids),"components":len(rows),"map_keys":list(MAP_KEYS),
              "map_shape":[len(MAP_KEYS),len(ids),224,224],"class_probability":"softmax_c(C_l), temperature=1",
              "component_pooling":"uniform mean over exact baseline 8-connected component pixels",
              "common_query_safety":{"pass":bool(shapes and shapes["q0"][-1]==shapes["k5"][1]==shapes["k4"][1]==shapes["k3"][1]),"shapes":shapes},
              "formula_max_abs_error":max_errors,"baseline_reference":reference,"sha256":artifacts,
              "elapsed_seconds":time.perf_counter()-started}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"event":"UMRF_EVIDENCE_FROZEN","components":len(rows),"sha256":artifacts}),flush=True)


if __name__ == "__main__": main()
