"""Freeze GT-free HQMR observables before any Oracle/GT label is opened."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.ucrf_v1.gate import load_model
from network.cirv import extract_regions
from network.hqmr import class_mixture
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import TTA, THRESHOLDS, normalize_cam, prediction_from_cam, presence, resize_unflip

EXPECTED_SHA = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
SUPPORT_THRESHOLD = 0.8  # fixed without seeing GT


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def spatial(value: torch.Tensor, hw: tuple[int, int], flip: tuple[int, ...]) -> np.ndarray:
    return resize_unflip(value.float(), hw, flip).detach().cpu().numpy()


def safe_stats(x: np.ndarray) -> tuple[float, float, float]:
    return float(np.mean(x)), float(np.std(x)), float(np.quantile(x, .9))


def entropy(p: np.ndarray, axis: int = 0) -> np.ndarray:
    return -(p * np.log(np.maximum(p, 1e-8))).sum(axis=axis)


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def stage_features(maps: dict[str, np.ndarray], mask: np.ndarray, cls: int, prefix: str) -> dict:
    scores = maps[prefix][:, mask].mean(1)
    order = np.argsort(scores)
    p = softmax(scores)
    return {f"{prefix}_margin": float(scores[order[-1]] - scores[order[-2]]),
            f"{prefix}_pred_margin": float(scores[cls] - scores[np.argsort(scores)[-2 if order[-1] == cls else -1]]),
            f"{prefix}_entropy": float(entropy(p)),
            f"{prefix}_top_class": int(order[-1])}


@torch.inference_mode()
def infer(model, image: torch.Tensor) -> dict:
    hw = tuple(int(v) for v in image.shape[-2:])
    stacks = {k: [] for k in ("u5", "d4", "c5", "c4", "c3", "cam")}
    gates = []
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, hqmr_mode="full")
            item = output["stages"][2]["hqmr"]
            l5, d4, l4, l3 = (item[k] for k in ("logits5", "direct4", "logits4", "logits3"))
            u5 = F.interpolate(l5, size=d4.shape[-2:], mode="bilinear", align_corners=False)
            weights = item["weights"]
            query_maps = {"c5": class_mixture(l5.sigmoid(), weights),
                          "c4": class_mixture(l4.sigmoid(), weights),
                          "c3": item["mixture"]}
        stacks["u5"].append(spatial(u5, hw, cam_flip))
        stacks["d4"].append(spatial(d4, hw, cam_flip))
        for key, mapping in query_maps.items():
            stacks[key].append(spatial(mapping, hw, cam_flip))
        stacks["cam"].append(spatial(output["primary_output"], hw, cam_flip))
        gates.append(output["deep_gate"].detach().float().cpu().numpy()[0])
    maps = {key: np.mean(values, axis=0) for key, values in stacks.items() if key!="cam"}
    # Match sealed DLAG: FP32 interpolation *before* resize, then torch mean.
    # Resizing BF16 first changes weak-class CAM normalization and components.
    maps["cam"] = normalize_cam(torch.stack([torch.from_numpy(v) for v in stacks["cam"]]).mean(0).numpy())
    gate_views = np.stack(gates)
    gate = gate_views.mean(0)
    label = presence(gate)
    prediction = prediction_from_cam(maps["cam"], label, np.empty(hw))
    return {"maps": maps, "gate": gate, "gate_views": gate_views,
            "label": label, "prediction": prediction, "cam_views": np.stack(stacks["cam"])}


def component_features(bundle: dict, region: dict, image_id: str, image_index: int,
                       class_counts: dict[int, int]) -> dict:
    mask = region["mask"]
    cls = int(region["class_id"])
    u, d = bundle["maps"]["u5"][:, mask], bundle["maps"]["d4"][:, mask]
    pu, pd_ = softmax(u), softmax(d)
    mix = .5 * (pu + pd_)
    js = .5 * ((pu * (np.log(np.maximum(pu, 1e-8)) - np.log(np.maximum(mix, 1e-8)))).sum(0)
               + (pd_ * (np.log(np.maximum(pd_, 1e-8)) - np.log(np.maximum(mix, 1e-8)))).sum(0))
    cosine = (u*d).sum(0) / np.maximum(np.linalg.norm(u, axis=0)*np.linalg.norm(d, axis=0), 1e-8)
    delta = np.abs(1/(1+np.exp(-np.clip(u, -30, 30))) - 1/(1+np.exp(-np.clip(d, -30, 30))))
    area = int(region["area"])
    ys, xs = np.where(mask)
    edge = mask & ~ndimage.binary_erosion(mask)
    perimeter = int(edge.sum())
    stage = {}
    for key in ("c5", "c4", "c3"):
        stage.update(stage_features(bundle["maps"], mask, cls, key))
    tops = [stage[f"{key}_top_class"] for key in ("c5", "c4", "c3")]
    view_scores = bundle["cam_views"][:, cls][:, mask].mean(1)
    cam_scores = bundle["maps"]["cam"][:, mask]
    a = {"image_id": image_id, "image_index": image_index, "predicted_class": cls,
         "component_id": int(region["component_id"]), "area": area,
         "a_disagreement_mean": float(delta.mean()), "a_disagreement_p90": float(np.quantile(delta, .9)),
         "a_js_mean": float(js.mean()), "a_js_p90": float(np.quantile(js, .9)),
         "a_cosine_mean": float(cosine.mean()),
         "a_u5_mean": float(u.mean()), "a_d4_mean": float(d.mean()),
         "a_energy_ratio": float(np.mean(np.abs(d))/max(np.mean(np.abs(u)), 1e-8)),
         "a_local_positive": float((d>0).mean()), "a_local_negative": float((d<0).mean()),
         "a_deep_entropy": float(entropy(pu).mean()), "a_local_entropy": float(entropy(pd_).mean()),
         "a_entropy_delta": float((entropy(pd_)-entropy(pu)).mean()),
         "a_deep_top_prob": float(pu.max(0).mean()), "a_local_top_prob": float(pd_.max(0).mean()),
         "a_query_agree": float((pu.argmax(0)==pd_.argmax(0)).mean()),
         **{k: v for k, v in stage.items() if not k.endswith("_top_class")},
         "a_c5_c4_same": int(tops[0]==tops[1]), "a_c4_c3_same": int(tops[1]==tops[2]),
         "a_three_same": int(tops[0]==tops[1]==tops[2]),
         "a_margin_gain_53": stage["c3_pred_margin"]-stage["c5_pred_margin"],
         "a_tta_class_conf_std": float(view_scores.std()),
         "a_tta_class_vote": float(np.mean([np.argmax(v[:, mask].mean(1))==cls for v in bundle["cam_views"]])),
         "a_gate_view_std": float(bundle["gate_views"][:, cls].std()),
         "a_cam_margin": float((cam_scores[cls]-np.max(np.delete(cam_scores, cls, axis=0), axis=0)).mean()),
         "a_log_area": float(np.log1p(area)), "a_perimeter_area": perimeter/area,
         "a_compactness": float(4*np.pi*area/max(perimeter**2,1)),
         "a_bbox_aspect": float((xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)),
         "a_border_touch": int(ys.min()==0 or xs.min()==0 or ys.max()==mask.shape[0]-1 or xs.max()==mask.shape[1]-1),
         "a_same_class_count": int(class_counts[cls]),
         "gate_preserved": int(bundle["label"][cls]>0)}
    return a


def gate_features(bundle: dict, image_id: str, image_index: int, cls: int) -> dict:
    gate = bundle["gate"]
    views = bundle["gate_views"][:, cls]
    c4 = normalize_cam(bundle["maps"]["c4"])[cls]
    c3 = normalize_cam(bundle["maps"]["c3"])[cls]
    flat4, flat3 = c4.ravel(), c3.ravel()
    def topmean(v, fraction):
        n = max(1, int(np.ceil(len(v)*fraction)))
        return float(np.partition(v, -n)[-n:].mean())
    support = c3 >= SUPPORT_THRESHOLD
    labels, count = ndimage.label(support, structure=np.ones((3, 3), np.uint8))
    sizes = np.bincount(labels.ravel())[1:]
    mass = c3 / max(c3.sum(), 1e-8)
    q = float(np.quantile(flat3, .95))
    overlap = float(((c4>=np.quantile(flat4,.95)) & (c3>=q)).sum()/max((c3>=q).sum(),1))
    top4 = int(np.argmax(bundle["maps"]["c4"].mean((1,2))))
    top3 = int(np.argmax(bundle["maps"]["c3"].mean((1,2))))
    return {"image_id": image_id, "image_index": image_index, "candidate_class": cls,
            "g_deep_score": float(gate[cls]), "g_gate_threshold": float(THRESHOLDS[cls]),
            "g_gate_margin": float(gate[cls]-THRESHOLDS[cls]),
            "g_abs_margin": float(abs(gate[cls]-THRESHOLDS[cls])),
            "g_deep_rank": int(np.argsort(np.argsort(-gate))[cls]+1),
            "g_tta_std": float(views.std()), "g_tta_min": float(views.min()),
            "g_tta_max": float(views.max()),
            "g_tta_above": int((views>THRESHOLDS[cls]).sum()),
            "g_c4_mean": float(flat4.mean()), "g_c4_max": float(flat4.max()),
            "g_c4_q95": float(np.quantile(flat4,.95)), "g_c4_q99": float(np.quantile(flat4,.99)),
            "g_c4_top01": topmean(flat4,.01), "g_c4_top05": topmean(flat4,.05),
            "g_c3_mean": float(flat3.mean()), "g_c3_max": float(flat3.max()),
            "g_c3_q90": float(np.quantile(flat3,.90)), "g_c3_q95": q,
            "g_c3_q99": float(np.quantile(flat3,.99)),
            "g_c3_top01": topmean(flat3,.01), "g_c3_top05": topmean(flat3,.05),
            "g_c3_top10": topmean(flat3,.10),
            "g_support_area": float(support.mean()), "g_support_count": int(count),
            "g_largest_support": int(sizes.max()) if len(sizes) else 0,
            "g_spatial_entropy": float(-(mass*np.log(np.maximum(mass,1e-8))).sum()/np.log(mass.size)),
            "g_c4_c3_corr": float(np.corrcoef(flat4,flat3)[0,1]) if flat4.std()>0 and flat3.std()>0 else 0.,
            "g_c4_c3_top05_overlap": overlap,
            "g_c3_c4_top05_delta": topmean(flat3,.05)-topmean(flat4,.05),
            "g_c4_top_class_same": int(top4==cls), "g_c3_top_class_same": int(top3==cls),
            "g_contradiction": topmean(flat3,.05)-float(gate[cls]),
            "g_local_deep_ratio": topmean(flat3,.05)/max(float(gate[cls]),1e-6),
            "g_query_support": float(bundle["maps"]["c3"][cls].mean()),
            "g_query_persistence": int(top4==cls and top3==cls)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--val-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()
    if digest(args.checkpoint) != EXPECTED_SHA:
        raise AssertionError("HQMR checkpoint hash mismatch")
    if (args.output/"feature_manifest.json").exists():
        raise FileExistsError("Feature freeze exists; do not overwrite")
    if len(list((args.val_root/"img").glob("*.png"))) != 3418:
        raise AssertionError("Expected 3418 image-only validation files")
    model = load_model(args.checkpoint)
    loader = DataLoader(Stage1_InferDataset(str(args.val_root/"img"), img_size=224),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    components, gate_pairs, predictions, ids = [], [], [], []
    start = time.perf_counter()
    for image_index, (names, image) in enumerate(loader):
        image_id = str(names[0]); ids.append(image_id)
        bundle = infer(model, image.cuda(non_blocking=True))
        prediction = bundle["prediction"].astype(np.uint8)
        predictions.append(prediction)
        regions = extract_regions(prediction)
        counts = {c: sum(int(r["class_id"]==c) for r in regions) for c in range(4)}
        for region in regions:
            components.append(component_features(bundle, region, image_id, image_index, counts))
        for cls in range(4):
            if bundle["label"][cls] == 0:
                gate_pairs.append(gate_features(bundle, image_id, image_index, cls))
        if (image_index+1)%100 == 0 or image_index+1 == len(loader):
            print(json.dumps({"event":"feature_progress","images":image_index+1,
                              "components":len(components),"gate_pairs":len(gate_pairs),
                              "elapsed_s":round(time.perf_counter()-start,1)}), flush=True)
    a = pd.DataFrame(components); g = pd.DataFrame(gate_pairs)
    if not len(a) or not len(g) or len(predictions)!=3418:
        raise AssertionError("Incomplete GT-free universe")
    (args.output/"arbitration").mkdir(parents=True,exist_ok=True)
    (args.output/"gate").mkdir(parents=True,exist_ok=True)
    a_path=args.output/"arbitration/observable_features.parquet"
    g_path=args.output/"gate/gate_off_pairs.parquet"
    a.to_parquet(a_path,index=False); g.to_parquet(g_path,index=False)
    np.savez_compressed(args.output/"baseline_predictions.npz",predictions=np.stack(predictions),image_ids=np.asarray(ids))
    a_features=[c for c in a if c.startswith("a_")]
    g_features=[c for c in g if c.startswith("g_")]
    if len(a_features)>=50 or len(g_features)>=50:
        raise AssertionError("Feature count exceeds prespecified cap")
    hashes={str(p.relative_to(args.output)):digest(p) for p in (a_path,g_path,args.output/"baseline_predictions.npz")}
    manifest={"feature_freeze_before_gt":True,"checkpoint_sha256":EXPECTED_SHA,
              "features_A":a_features,"features_G":g_features,
              "excluded_from_predictor":["image_id","image_index","component_id","predicted_class","candidate_class","area","gate_preserved"],
              "support_threshold":SUPPORT_THRESHOLD,"component_connectivity":8,
              "validation_images":len(ids),"components":len(a),"gate_off_pairs":len(g),
              "parameter_updates":0,"sha256":hashes,"elapsed_seconds":time.perf_counter()-start}
    (args.output/"feature_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    (args.output/"feature_table_sha256.txt").write_text("\n".join(f"{v}  {k}" for k,v in hashes.items())+"\n",encoding="utf-8")
    print(json.dumps({"event":"FEATURES_FROZEN","components":len(a),"gate_pairs":len(g),"sha256":hashes}),flush=True)


if __name__=="__main__":
    main()
