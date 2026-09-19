"""Frozen 8-alpha HQMR bank plus component/gate/joint oracle construction."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audits.ucrf_v1.gate import load_model
from network.hqmr import class_mixture, direct_affinity, residual_logits
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import (
    TTA, foreground_confusion, normalize_cam, prediction_from_cam, presence,
    resize_unflip,
)

ALPHAS = np.asarray([0., .25, .5, 1., 1.5, 2., 3., 4.], np.float32)
TIE_ORDER = [3, 2, 4, 1, 5, 0, 6, 7]  # 1,.5,1.5,.25,2,0,3,4
EXPECTED_MIOU = 0.6557244403737567
EXPECTED_M1 = 4440


def unpack_mask(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return np.unpackbits(packed)[:np.prod(shape)].reshape(shape).astype(bool)


def choose(counts: np.ndarray) -> int:
    best = int(counts.max())
    return next(index for index in TIE_ORDER if int(counts[index]) == best)


@torch.inference_mode()
def infer_alpha_bank(model, image: torch.Tensor, original_hw: tuple[int, int]) -> dict:
    views = {index: [] for index in range(len(ALPHAS))}
    gates, baseline_views = [], []
    maximum_alpha1_error = 0.0
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        captured4, captured3 = [], []
        handles = [model.hqmr.scale4.register_forward_hook(
                       lambda _module, _inputs, value: captured4.append(value)),
                   model.hqmr.scale3.register_forward_hook(
                       lambda _module, _inputs, value: captured3.append(value))]
        try:
            value = torch.flip(image, dims=input_flip) if input_flip else image
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(value, dummy, step=29275, hqmr_mode="full")
                item = output["stages"][2]["hqmr"]
                if len(captured4) != 2 or len(captured3) != 1:
                    raise AssertionError("Unexpected HQMR projection call count")
                v4 = captured4[-1][1]
                k3 = captured3[0][0]
                u5 = F.interpolate(item["logits5"], size=item["direct4"].shape[-2:],
                                   mode="bilinear", align_corners=False)
                for alpha_index, alpha in enumerate(ALPHAS):
                    logits4 = u5 + float(alpha) * item["direct4"]
                    query4 = model.hqmr.update4(item["query5"], logits4, v4)
                    logits3 = residual_logits(logits4, direct_affinity(query4, k3))
                    mixture = class_mixture(logits3.sigmoid(), item["weights"])
                    if alpha == 1:
                        maximum_alpha1_error = max(maximum_alpha1_error,
                            float((mixture.float()-item["mixture"].float()).abs().max()))
                    views[alpha_index].append(resize_unflip(mixture, original_hw, cam_flip).float().cpu())
            baseline_views.append(resize_unflip(output["primary_output"], original_hw,
                                                cam_flip).float().cpu())
            gates.append(output["deep_gate"].detach().float().cpu())
        finally:
            for handle in handles:
                handle.remove()
    mean_gate = torch.stack(gates).mean(0).numpy()[0]
    baseline_mean = torch.stack(baseline_views).mean(0).numpy()
    cams = [normalize_cam(torch.stack(views[index]).mean(0).numpy())
            for index in range(len(ALPHAS))]
    baseline_cam = normalize_cam(baseline_mean)
    maximum_alpha1_error = max(maximum_alpha1_error,
        float(np.max(np.abs(cams[3]-baseline_cam))))
    return {"cams": cams, "gate_score": mean_gate,
            "baseline_cam": baseline_cam, "alpha1_max_abs_error": maximum_alpha1_error}


def save_prediction_stack(directory: Path, predictions: np.ndarray,
                          image_ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "predictions.npz",
                        predictions=predictions, image_ids=np.asarray(image_ids))


@torch.inference_mode()
def run(args: argparse.Namespace) -> None:
    gate = json.loads((args.output / "00_reproduction_gate.json").read_text())
    if not gate.get("pass"):
        raise AssertionError("Fresh DLAG reproduction gate required")
    np.random.seed(42); torch.manual_seed(42)
    model = load_model(args.checkpoint)
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)
    ucrf_frame = pd.read_parquet(args.ucrf_output / "metrics/component_event_table.parquet")
    mask_file = np.load(args.ucrf_output / "masks/whole.npz")
    packed = mask_file["packed"]
    mask_shape = tuple(int(v) for v in mask_file["shape"])
    if mask_shape != (224, 224):
        raise AssertionError(f"Expected 224x224 M1 masks, got {mask_shape}")
    m1_indices = np.flatnonzero(ucrf_frame.cohort.to_numpy() == "M1")
    if len(m1_indices) != EXPECTED_M1:
        raise AssertionError("Frozen UCRF M1 count mismatch")
    rows_by_image: dict[str, list[int]] = defaultdict(list)
    tp_by_image: dict[str, list[int]] = defaultdict(list)
    for row_index, row in ucrf_frame.iterrows():
        if row.cohort == "M1": rows_by_image[str(row.image_id)].append(int(row_index))
        elif row.cohort == "TP_candidate": tp_by_image[str(row.image_id)].append(int(row_index))

    n, h, w = len(dataset), *mask_shape
    alpha_predictions = np.empty((len(ALPHAS), n, h, w), np.uint8)
    method_names = ("baseline", "oracle_A1", "oracle_A2", "oracle_B1",
                    "oracle_B2", "oracle_AB")
    method_predictions = {name: np.empty((n, h, w), np.uint8) for name in method_names}
    confusions = {name: np.empty((n, 4, 4), np.int64) for name in method_names}
    alpha_confusions = np.empty((len(ALPHAS), n, 4, 4), np.int64)
    component_records, margin_records, gate_records, tp_records, method_safety = [], [], [], [], []
    image_ids: list[str] = []
    alpha1_max_abs_error = 0.0
    baseline_prediction_equal = True
    started = time.perf_counter()

    for image_index, (names, image) in enumerate(loader):
        image_id = str(names[0]); image_ids.append(image_id)
        truth = np.asarray(Image.open(args.val_root / "mask" / f"{image_id}.png"))
        if truth.shape != mask_shape:
            raise AssertionError(f"Unexpected truth size {truth.shape} for {image_id}")
        inferred = infer_alpha_bank(model, image.cuda(non_blocking=True), truth.shape)
        alpha1_max_abs_error = max(alpha1_max_abs_error, inferred["alpha1_max_abs_error"])
        base_label = presence(inferred["gate_score"])
        predictions = [prediction_from_cam(cam, base_label, truth) for cam in inferred["cams"]]
        baseline = prediction_from_cam(inferred["baseline_cam"], base_label, truth)
        baseline_prediction_equal &= np.array_equal(predictions[3], baseline)
        force_label = base_label.copy()
        for row_index in rows_by_image.get(image_id, []):
            true_class = int(ucrf_frame.iloc[row_index].true_class)
            force_label[true_class] = 1
        force_predictions = [prediction_from_cam(cam, force_label, truth)
                             for cam in inferred["cams"]]
        gate_prediction = prediction_from_cam(inferred["cams"][3], force_label, truth)
        gt_label = np.zeros(4, np.float32)
        gt_label[np.unique(truth[truth < 4]).astype(int)] = 1
        gt_gate_prediction = prediction_from_cam(inferred["cams"][3], gt_label, truth)

        oracle_a1, oracle_a2, oracle_ab = baseline.copy(), baseline.copy(), gate_prediction.copy()
        local = []
        for row_index in rows_by_image.get(image_id, []):
            row = ucrf_frame.iloc[row_index]
            mask = unpack_mask(packed[row_index], mask_shape)
            valid = mask & (truth < 4)
            counts_a = np.asarray([(prediction[valid] == truth[valid]).sum()
                                   for prediction in predictions], np.int64)
            counts_ab = np.asarray([(prediction[valid] == truth[valid]).sum()
                                    for prediction in force_predictions], np.int64)
            best_a, best_ab = choose(counts_a), choose(counts_ab)
            oracle_a1[mask] = predictions[best_a][mask]
            oracle_ab[mask] = force_predictions[best_ab][mask]
            baseline_correct = int((baseline[valid] == truth[valid]).sum())
            b_correct = int((gate_prediction[valid] == truth[valid]).sum())
            a_pixel = predictions[best_a][valid] == truth[valid]
            b_pixel = gate_prediction[valid] == truth[valid]
            ab_pixel = force_predictions[best_ab][valid] == truth[valid]
            fa = bool(row.margin_logits5 < 0 and row.direct4_effect_margin > 0 and
                      row.margin_logits4 < 0)
            fb = bool(base_label[int(row.true_class)] == 0)
            margins = []
            mass = max(int(valid.sum()), 1)
            for alpha_index, (alpha, cam) in enumerate(zip(ALPHAS, inferred["cams"])):
                margin = float((cam[int(row.true_class)][valid] -
                                cam[int(row.predicted_class)][valid]).mean()) if valid.any() else np.nan
                margins.append(margin)
                margin_records.append({"image_id": image_id, "component_id": int(row.component_id),
                    "frame_index": row_index, "alpha": float(alpha), "margin": margin,
                    "correct_pixels": int(counts_a[alpha_index]),
                    "correct_fraction": float(counts_a[alpha_index]/mass),
                    "forced_gate_correct_pixels": int(counts_ab[alpha_index]),
                    "forced_gate_correct_fraction": float(counts_ab[alpha_index]/mass)})
            differences = np.diff(np.asarray(margins))
            response = ("never_recoverable" if counts_a.max() == 0 else
                        "monotonic_correction" if np.all(differences >= -1e-6) else
                        "local_harmful" if ALPHAS[best_a] < 1 else "non_monotonic")
            record = {"image_id": image_id, "image_index": image_index,
                "component_id": int(row.component_id), "frame_index": row_index,
                "true_class": int(row.true_class), "predicted_class": int(row.predicted_class),
                "area": int(row.area), "valid_area": int(valid.sum()), "purity": float(row.purity),
                "best_alpha_A": float(ALPHAS[best_a]), "best_alpha_AB": float(ALPHAS[best_ab]),
                "baseline_correct_pixels": baseline_correct,
                "A_correct_pixels": int(counts_a[best_a]), "B_correct_pixels": b_correct,
                "AB_correct_pixels": int(counts_ab[best_ab]),
                "A_only_correct_pixels": int((a_pixel & ~b_pixel).sum()),
                "B_only_correct_pixels": int((b_pixel & ~a_pixel).sum()),
                "both_individual_correct_pixels": int((a_pixel & b_pixel).sum()),
                "joint_only_correct_pixels": int((ab_pixel & ~a_pixel & ~b_pixel).sum()),
                "AB_unrecoverable_pixels": int((~ab_pixel).sum()),
                "A_majority_recovered": bool(counts_a[best_a] >= .5*mass),
                "B_majority_recovered": bool(b_correct >= .5*mass),
                "AB_majority_recovered": bool(counts_ab[best_ab] >= .5*mass),
                "deep_local_failure": fa, "gate_excluded": fb,
                "gate_score_true": float(inferred["gate_score"][int(row.true_class)]),
                "gate_score_pred": float(inferred["gate_score"][int(row.predicted_class)]),
                "response_category": response,
                "A_gain_pixels": int(counts_a[best_a]-baseline_correct),
                "B_gain_pixels": int(b_correct-baseline_correct),
                "AB_gain_pixels": int(counts_ab[best_ab]-baseline_correct)}
            component_records.append(record)
            local.append((record, mask, best_a))
        claimed = np.zeros(mask_shape, bool)
        for record, mask, best_a in sorted(local,
                key=lambda item: (item[0]["A_gain_pixels"], item[0]["area"]), reverse=True):
            ring = ndimage.binary_dilation(mask, iterations=8)
            replace = ring & ~claimed
            oracle_a2[replace] = predictions[best_a][replace]
            claimed |= replace

        method = {"baseline": baseline, "oracle_A1": oracle_a1, "oracle_A2": oracle_a2,
                  "oracle_B1": gate_prediction, "oracle_B2": gt_gate_prediction,
                  "oracle_AB": oracle_ab}
        for name, prediction in method.items():
            method_predictions[name][image_index] = prediction.astype(np.uint8)
            confusions[name][image_index] = foreground_confusion(truth, prediction)
            valid = truth < 4
            baseline_correct_map = (baseline == truth) & valid
            method_safety.append({"image_id": image_id, "method": name,
                "baseline_correct_area": int(baseline_correct_map.sum()),
                "preserved_correct_area": int((baseline_correct_map & (prediction == truth)).sum()),
                "harmed_correct_area": int((baseline_correct_map & (prediction != truth)).sum()),
                "newly_correct_area": int(((baseline != truth) & (prediction == truth) & valid).sum())})
        for alpha_index, prediction in enumerate(predictions):
            alpha_predictions[alpha_index, image_index] = prediction.astype(np.uint8)
            alpha_confusions[alpha_index, image_index] = foreground_confusion(truth, prediction)
        gate_records.append({"image_id": image_id, "baseline_gate": base_label.astype(int).tolist(),
            "forced_gate": force_label.astype(int).tolist(), "gt_presence_gate": gt_label.astype(int).tolist(),
            "gate_scores": inferred["gate_score"].tolist(),
            "forced_class_count": int(np.maximum(force_label-base_label, 0).sum())})
        valid_image = truth < 4
        baseline_correct_map = (baseline == truth) & valid_image
        for alpha_index, prediction in enumerate(predictions):
            tp_records.append({"image_id": image_id, "alpha": float(ALPHAS[alpha_index]),
                "baseline_correct_area": int(baseline_correct_map.sum()),
                "preserved_correct_area": int((baseline_correct_map & (prediction == truth)).sum()),
                "harmed_correct_area": int((baseline_correct_map & (prediction != truth)).sum()),
                "newly_correct_area": int(((baseline != truth) & (prediction == truth) & valid_image).sum())})
        if (image_index+1) % 100 == 0:
            print(json.dumps({"event": "bank_progress", "images": image_index+1,
                "components": len(component_records), "elapsed_s": round(time.perf_counter()-started, 1)}), flush=True)

    if not baseline_prediction_equal or alpha1_max_abs_error > 1e-6:
        raise AssertionError(f"alpha=1 integrity failed: equal={baseline_prediction_equal}, error={alpha1_max_abs_error}")
    component_frame = pd.DataFrame(component_records)
    if len(component_frame) != EXPECTED_M1:
        raise AssertionError("Oracle component count mismatch")
    component_frame["failure_overlap"] = np.select([
        component_frame.deep_local_failure & ~component_frame.gate_excluded,
        ~component_frame.deep_local_failure & component_frame.gate_excluded,
        component_frame.deep_local_failure & component_frame.gate_excluded],
        ["deep_dominance_only", "gate_exclusion_only", "both"], default="neither")
    component_frame["recoverability"] = np.select([
        (component_frame.A_correct_pixels > 0) & (component_frame.B_correct_pixels == 0),
        (component_frame.B_correct_pixels > 0) & (component_frame.A_correct_pixels == 0),
        (component_frame.A_correct_pixels == 0) & (component_frame.B_correct_pixels == 0) &
            (component_frame.AB_correct_pixels > 0),
        component_frame.AB_correct_pixels == 0],
        ["R1_arbitration", "R2_gate", "R3_joint_only", "R4_unrecoverable"],
        default="both_individually")

    for alpha_index, alpha in enumerate(ALPHAS):
        tag = f"alpha_{int(round(alpha*100)):03d}"
        save_prediction_stack(args.output / "counterfactuals" / tag,
                              alpha_predictions[alpha_index], image_ids)
    for name, directory in (("oracle_A1", "arbitration_oracle/oracle_A_predictions"),
                            ("oracle_A2", "arbitration_oracle/oracle_A2_predictions"),
                            ("oracle_B1", "gate_oracle/oracle_B_predictions"),
                            ("oracle_B2", "gate_oracle/oracle_B2_predictions"),
                            ("oracle_AB", "joint_oracle/oracle_AB_predictions")):
        save_prediction_stack(args.output / directory, method_predictions[name], image_ids)
    (args.output / "arbitration_oracle").mkdir(exist_ok=True)
    (args.output / "gate_oracle").mkdir(exist_ok=True)
    (args.output / "joint_oracle").mkdir(exist_ok=True)
    (args.output / "metrics").mkdir(exist_ok=True)
    component_frame.to_csv(args.output / "arbitration_oracle/component_best_alpha.csv", index=False)
    pd.DataFrame(margin_records).to_csv(args.output / "arbitration_oracle/margin_response.csv", index=False)
    pd.DataFrame(gate_records).to_csv(args.output / "gate_oracle/gate_exclusion_manifest.csv", index=False)
    component_frame.to_csv(args.output / "joint_oracle/recoverability_table.csv", index=False)
    pd.DataFrame(tp_records).to_csv(args.output / "metrics/tp_safety_by_alpha.csv", index=False)
    pd.DataFrame(method_safety).to_csv(args.output / "metrics/oracle_safety.csv", index=False)
    np.savez_compressed(args.output / "metrics/confusions.npz", image_ids=np.asarray(image_ids),
                        alpha_confusions=alpha_confusions,
                        **{name: value for name, value in confusions.items()})
    integrity = {"pass": True, "alpha1_prediction_equal_baseline": baseline_prediction_equal,
        "alpha1_max_abs_difference": alpha1_max_abs_error,
        "presence_gate_application_changes_upstream_tensors": False,
        "gate_only_changes": ["binary_presence_label", "final_argmax_eligible_classes"],
        "parameter_updates": 0, "validation_images": n, "M1_components": len(component_frame),
        "runtime_seconds": time.perf_counter()-started}
    (args.output / "gate_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({"event": "oracle_bank_done", **integrity}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--ucrf-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
