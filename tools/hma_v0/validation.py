"""Exact parity and full BCSS validation same-forward mechanism audit."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from tool.GenDataset import Stage1_InferDataset
from tools.hma_v0 import (
    BCSS_THRESHOLDS,
    EXPECTED_VAL_IMAGES,
    FINAL_VARIANTS,
    IMAGE_SIZE,
    PARITY_IMAGES,
    STAGES,
    TTA_TRANSFORMS,
    VARIANTS,
)
from tools.hma_v0.instrumentation import presence_from_probability
from tools.hma_v0.metrics import (
    ComplementarityAccumulator,
    ErrorTaxonomyAccumulator,
    OfficialMetricAccumulator,
    SpatialTransitionAccumulator,
    foreground_boundary_distance,
    minmax_normalize,
    prediction_from_fusion,
    prediction_from_standalone,
)


def _dataset(val_root):
    dataset = Stage1_InferDataset(str(Path(val_root) / "img"), img_size=IMAGE_SIZE)
    dataset.object = sorted(dataset.object)
    if len(dataset) != EXPECTED_VAL_IMAGES:
        raise AssertionError(f"Expected {EXPECTED_VAL_IMAGES} validation images, got {len(dataset)}")
    return dataset


def _flip_back(tensor, dims):
    return torch.flip(tensor, dims=dims) if dims else tensor


def _resize(tensor, original_size):
    return F.interpolate(tensor, original_size, mode="bilinear", align_corners=False)[0]


def audit_tta_forward(model, image, original_size, amp_dtype="bf16", collect_gate=True):
    dtype = torch.bfloat16 if amp_dtype == "bf16" else None
    relu_lists = {
        variant: {stage: [] for stage in STAGES} for variant in VARIANTS
    }
    logit_lists = {
        variant: {stage: [] for stage in STAGES} for variant in VARIANTS
    }
    pooled_lists = {
        variant: {stage: [] for stage in STAGES} for variant in VARIANTS
    }
    deep_relu, deep_logits, probabilities = [], [], []
    canonical_gates = None
    with torch.no_grad():
        for tta_index, (input_dims, output_dims) in enumerate(TTA_TRANSFORMS):
            augmented = torch.flip(image, dims=input_dims) if input_dims else image
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=dtype is not None
            ):
                audit = model.forward_hfrm_audit(augmented, apply_deep_dropout=False)
            if collect_gate and tta_index == 0:
                canonical_gates = {
                    stage: audit["gates"][stage][0, :, 0, 0].detach().float().cpu().numpy()
                    for stage in STAGES
                }
            for variant in VARIANTS:
                for stage in STAGES:
                    relu_lists[variant][stage].append(_flip_back(
                        _resize(audit["cam_relu"][variant][stage], original_size), output_dims
                    ))
                    logit_lists[variant][stage].append(_flip_back(
                        _resize(audit["cam_logits"][variant][stage], original_size), output_dims
                    ))
                    pooled_lists[variant][stage].append(
                        audit["pooled_logits"][variant][stage][0]
                    )
            deep_relu.append(_flip_back(_resize(audit["deep_relu"], original_size), output_dims))
            deep_logits.append(_flip_back(_resize(audit["deep_logits"], original_size), output_dims))
            probabilities.append(audit["y_deep"][0])
    return {
        "cam_relu": {
            variant: {
                stage: torch.stack(values).mean(0).detach().float().cpu().numpy()
                for stage, values in stages.items()
            }
            for variant, stages in relu_lists.items()
        },
        "cam_logits": {
            variant: {
                stage: torch.stack(values).mean(0).detach().float().cpu().numpy()
                for stage, values in stages.items()
            }
            for variant, stages in logit_lists.items()
        },
        "pooled_logits": {
            variant: {
                stage: torch.stack(values).mean(0).detach().float().cpu().numpy()
                for stage, values in stages.items()
            }
            for variant, stages in pooled_lists.items()
        },
        "deep_relu": torch.stack(deep_relu).mean(0).detach().float().cpu().numpy(),
        "deep_logits": torch.stack(deep_logits).mean(0).detach().float().cpu().numpy(),
        "probability": torch.stack(probabilities).mean(0).detach().float().cpu().numpy(),
        "gates": canonical_gates,
    }


def run_instrumentation_parity(model, val_root, amp_dtype="bf16"):
    dataset = _dataset(val_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dtype = torch.bfloat16 if amp_dtype == "bf16" else None
    max_abs = 0.0
    exact_tensors = True
    differing_pixels = 0
    tensors_compared = 0
    model.eval()
    for image_index, (name_tuple, image) in enumerate(loader):
        if image_index >= PARITY_IMAGES:
            break
        image_id = name_tuple[0]
        original_size = np.asarray(
            Image.open(Path(val_root) / "img" / f"{image_id}.png")
        ).shape[:2]
        image = image.cuda(non_blocking=True)
        audit_cams = {stage: [] for stage in ("28_1", "28_2", "deep")}
        official_cams = {stage: [] for stage in ("28_1", "28_2", "deep")}
        audit_probabilities, official_probabilities = [], []
        for input_dims, output_dims in TTA_TRANSFORMS:
            augmented = torch.flip(image, dims=input_dims) if input_dims else image
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=dtype, enabled=dtype is not None
            ):
                audit = model.forward_hfrm_audit(augmented, apply_deep_dropout=False)
                official = model.forward_cam(augmented)
            pairs = (
                (audit["cam_relu"]["full"]["56"], official[0]),
                (audit["cam_relu"]["full"]["28_1"], official[1]),
                (audit["cam_relu"]["full"]["28_2"], official[2]),
                (audit["deep_relu"], official[3]),
                (audit["y_deep"], official[4]),
            )
            for left, right in pairs:
                exact_tensors &= bool(torch.equal(left, right))
                max_abs = max(max_abs, float((left.float() - right.float()).abs().max().item()))
                tensors_compared += 1
            for stage, audit_tensor, official_tensor in (
                ("28_1", audit["cam_relu"]["full"]["28_1"], official[1]),
                ("28_2", audit["cam_relu"]["full"]["28_2"], official[2]),
                ("deep", audit["deep_relu"], official[3]),
            ):
                audit_cams[stage].append(_flip_back(_resize(audit_tensor, original_size), output_dims))
                official_cams[stage].append(_flip_back(_resize(official_tensor, original_size), output_dims))
            audit_probabilities.append(audit["y_deep"][0])
            official_probabilities.append(official[4][0])
        audit_probability = torch.stack(audit_probabilities).mean(0)
        official_probability = torch.stack(official_probabilities).mean(0)
        audit_presence = presence_from_probability(audit_probability[None])[0].float().cpu().numpy()
        official_presence = presence_from_probability(official_probability[None])[0].float().cpu().numpy()
        audit_prediction, _ = prediction_from_fusion(
            *[
                torch.stack(audit_cams[stage]).mean(0).float().cpu().numpy()
                for stage in ("28_1", "28_2", "deep")
            ],
            presence=audit_presence,
        )
        official_prediction, _ = prediction_from_fusion(
            *[
                torch.stack(official_cams[stage]).mean(0).float().cpu().numpy()
                for stage in ("28_1", "28_2", "deep")
            ],
            presence=official_presence,
        )
        differing_pixels += int(np.count_nonzero(audit_prediction != official_prediction))
    decision = (
        "SSHR_HMA_PARITY_PASS"
        if exact_tensors and max_abs == 0.0 and differing_pixels == 0
        else "SSHR_HMA_INSTRUMENTATION_NOGO"
    )
    result = {
        "decision": decision,
        "images": PARITY_IMAGES,
        "tta_views_per_image": len(TTA_TRANSFORMS),
        "tensors_compared": tensors_compared,
        "same_process_exact_tensor_equality": bool(exact_tensors),
        "maximum_absolute_difference": float(max_abs),
        "final_prediction_differing_pixels": int(differing_pixels),
    }
    if decision != "SSHR_HMA_PARITY_PASS":
        raise RuntimeError(result)
    return result


def _gt_presence(ground_truth):
    return np.asarray([(ground_truth == index).any() for index in range(4)], dtype=bool)


def _present_confusion_update(store, stage, ground_truth, raw, gsr, presence):
    foreground = ground_truth < 4
    raw_correct = raw == ground_truth
    gsr_correct = gsr == ground_truth
    predicted_present_raw = presence[raw]
    predicted_present_gsr = presence[gsr]
    raw_error = foreground & ~raw_correct & predicted_present_raw
    gsr_error = foreground & ~gsr_correct & predicted_present_gsr
    target = store[stage]
    target["raw_present_confusion"] += int(raw_error.sum())
    target["gsr_present_confusion"] += int(gsr_error.sum())
    target["recovered"] += int((raw_error & gsr_correct).sum())
    target["harmed"] += int((foreground & raw_correct & gsr_error).sum())


def _gate_contribution_update(store, ground_truth, no_gate, gated, presence):
    foreground = ground_truth < 4
    no_gate_correct = no_gate == ground_truth
    gated_correct = gated == ground_truth
    absent_no_gate = foreground & ~no_gate_correct & ~presence[no_gate]
    absent_gated = foreground & ~gated_correct & ~presence[gated]
    present_no_gate = foreground & ~no_gate_correct & presence[no_gate]
    present_gated = foreground & ~gated_correct & presence[gated]
    store["absent_error_no_gate"] += int(absent_no_gate.sum())
    store["absent_error_gated"] += int(absent_gated.sum())
    store["absent_error_removed"] += int((absent_no_gate & gated_correct).sum())
    store["present_error_no_gate"] += int(present_no_gate.sum())
    store["present_error_gated"] += int(present_gated.sum())
    store["recovered_pixels"] += int((foreground & ~no_gate_correct & gated_correct).sum())
    store["harmed_pixels"] += int((foreground & no_gate_correct & ~gated_correct).sum())


def _response_rows(image_id, result, ground_truth, gt_presence):
    rows = []
    foreground = ground_truth < 4
    for stage in STAGES:
        for class_index in range(4):
            raw = result["cam_relu"]["raw"][stage][class_index]
            gsr = result["cam_relu"]["gsr"][stage][class_index]
            full = result["cam_relu"]["full"][stage][class_index]
            normalized_raw = minmax_normalize(result["cam_relu"]["raw"][stage])[class_index]
            normalized_gsr = minmax_normalize(result["cam_relu"]["gsr"][stage])[class_index]
            normalized_full = minmax_normalize(result["cam_relu"]["full"][stage])[class_index]
            target = ground_truth == class_index
            off_target = foreground & ~target
            raw_range = float(raw.max() - raw.min())
            full_range = float(full.max() - full.min())
            raw_delta = float(full.mean() - raw.mean())
            normalized_delta = float(normalized_full.mean() - normalized_raw.mean())
            scaled_raw_delta = raw_delta / ((raw_range + full_range) * 0.5 + 1e-8)
            rows.append({
                "image_id": image_id,
                "stage": stage,
                "class": class_index,
                "present": bool(gt_presence[class_index]),
                "raw_logit": float(result["pooled_logits"]["raw"][stage][class_index]),
                "gsr_logit": float(result["pooled_logits"]["gsr"][stage][class_index]),
                "delta_gsr_logit": float(
                    result["pooled_logits"]["gsr"][stage][class_index]
                    - result["pooled_logits"]["raw"][stage][class_index]
                ),
                "delta_gsr_raw_cam_mass": float(gsr.mean() - raw.mean()),
                "delta_gsr_normalized_cam_mass": float(normalized_gsr.mean() - normalized_raw.mean()),
                "delta_gsr_target_gt_mass": (
                    float((gsr[target] - raw[target]).mean()) if target.any() else np.nan
                ),
                "delta_gsr_other_foreground_mass": (
                    float((gsr[off_target] - raw[off_target]).mean()) if off_target.any() else np.nan
                ),
                "raw_cam_range": raw_range,
                "full_cam_range": full_range,
                "raw_cam_mean": float(raw.mean()),
                "raw_cam_std": float(raw.std()),
                "full_cam_mean": float(full.mean()),
                "full_cam_std": float(full.std()),
                "normalized_raw_mean": float(normalized_raw.mean()),
                "normalized_raw_std": float(normalized_raw.std()),
                "normalized_full_mean": float(normalized_full.mean()),
                "normalized_full_std": float(normalized_full.std()),
                "full_minus_raw_mass": raw_delta,
                "full_minus_raw_normalized_mass": normalized_delta,
                "range_scaled_raw_delta": float(scaled_raw_delta),
                "normalization_amplification_ratio": float(
                    abs(normalized_delta) / (abs(scaled_raw_delta) + 1e-8)
                ),
            })
    return rows


def summarize_response(frame):
    result = {"by_stage_class": {}, "absent_primary": {}, "normalization": {}}
    for stage in STAGES:
        result["by_stage_class"][stage] = {}
        for class_index in range(4):
            subset = frame[(frame.stage == stage) & (frame["class"] == class_index)]
            absent = subset[~subset.present]
            present = subset[subset.present]
            result["by_stage_class"][stage][str(class_index)] = {
                "absent_cases": int(len(absent)),
                "absent_delta_logit_mean": float(absent.delta_gsr_logit.mean()),
                "absent_delta_logit_median": float(absent.delta_gsr_logit.median()),
                "absent_delta_raw_cam_mass_mean": float(absent.delta_gsr_raw_cam_mass.mean()),
                "absent_delta_normalized_cam_mass_mean": float(absent.delta_gsr_normalized_cam_mass.mean()),
                "absent_fraction_logit_suppressed": float((absent.delta_gsr_logit < 0).mean()),
                "present_cases": int(len(present)),
                "present_delta_logit_mean": float(present.delta_gsr_logit.mean()),
                "present_delta_target_gt_mass_mean": float(present.delta_gsr_target_gt_mass.mean()),
                "present_delta_other_foreground_mass_mean": float(
                    present.delta_gsr_other_foreground_mass.mean()
                ),
                "normalization_amplification_ratio_median": float(
                    subset.normalization_amplification_ratio.median()
                ),
                "absolute_range_scaled_raw_delta_median": float(
                    subset.range_scaled_raw_delta.abs().median()
                ),
            }
        stage_absent = frame[(frame.stage == stage) & (~frame.present)]
        result["absent_primary"][stage] = {
            "cases": int(len(stage_absent)),
            "median_delta_logit": float(stage_absent.delta_gsr_logit.median()),
            "mean_delta_logit": float(stage_absent.delta_gsr_logit.mean()),
            "fraction_suppressed": float((stage_absent.delta_gsr_logit < 0).mean()),
        }
        stage_frame = frame[frame.stage == stage]
        result["normalization"][stage] = {
            "raw_full_minus_raw_mass_abs_median": float(
                stage_frame.full_minus_raw_mass.abs().median()
            ),
            "normalized_full_minus_raw_mass_abs_median": float(
                stage_frame.full_minus_raw_normalized_mass.abs().median()
            ),
            "range_scaled_raw_delta_abs_median": float(
                stage_frame.range_scaled_raw_delta.abs().median()
            ),
            "amplification_ratio_median": float(
                stage_frame.normalization_amplification_ratio.median()
            ),
            "fraction_amplified_over_2x": float(
                (stage_frame.normalization_amplification_ratio > 2.0).mean()
            ),
        }
    all_absent = frame[~frame.present]
    result["absent_primary"]["all_stages"] = {
        "cases": int(len(all_absent)),
        "median_delta_logit": float(all_absent.delta_gsr_logit.median()),
        "mean_delta_logit": float(all_absent.delta_gsr_logit.mean()),
        "fraction_suppressed": float((all_absent.delta_gsr_logit < 0).mean()),
    }
    result["normalization"]["all_stages"] = {
        "raw_full_minus_raw_mass_abs_median": float(frame.full_minus_raw_mass.abs().median()),
        "normalized_full_minus_raw_mass_abs_median": float(
            frame.full_minus_raw_normalized_mass.abs().median()
        ),
        "range_scaled_raw_delta_abs_median": float(
            frame.range_scaled_raw_delta.abs().median()
        ),
        "amplification_ratio_median": float(
            frame.normalization_amplification_ratio.median()
        ),
        "fraction_amplified_over_2x": float(
            (frame.normalization_amplification_ratio > 2.0).mean()
        ),
    }
    return result


def summarize_gates(gate_vectors, labels):
    labels = np.asarray(labels, dtype=bool)
    statistics, semantic_rows = {}, []
    for stage, vectors in gate_vectors.items():
        vectors = np.asarray(vectors, dtype=np.float32)
        flat = vectors.reshape(-1)
        statistics[stage] = {
            "images": int(vectors.shape[0]),
            "channels": int(vectors.shape[1]),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "p05": float(np.quantile(flat, 0.05)),
            "p50": float(np.quantile(flat, 0.50)),
            "p95": float(np.quantile(flat, 0.95)),
            "fraction_below_0.1": float((flat < 0.1).mean()),
            "fraction_above_0.9": float((flat > 0.9).mean()),
            "inter_image_variance": float(vectors.var(axis=0).mean()),
        }
        for class_index in range(4):
            present = vectors[labels[:, class_index]]
            absent = vectors[~labels[:, class_index]]
            mean_present, mean_absent = present.mean(0), absent.mean(0)
            cosine = float(
                np.dot(mean_present, mean_absent)
                / (np.linalg.norm(mean_present) * np.linalg.norm(mean_absent) + 1e-12)
            )
            semantic_rows.append({
                "stage": stage,
                "class": class_index,
                "present_images": int(len(present)),
                "absent_images": int(len(absent)),
                "mean_gate_present": float(present.mean()),
                "mean_gate_absent": float(absent.mean()),
                "mean_gate_difference": float(present.mean() - absent.mean()),
                "mean_vector_cosine": cosine,
                "mean_vector_cosine_distance": float(1.0 - cosine),
                "mean_vector_l2_distance": float(np.linalg.norm(mean_present - mean_absent)),
            })
    return statistics, semantic_rows


def run_validation_audit(model, val_root, num_workers=4, amp_dtype="bf16"):
    dataset = _dataset(val_root)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    model.eval()
    final_metrics = {
        name: OfficialMetricAccumulator() for name in FINAL_VARIANTS
    }
    standalone_metrics = {
        stage: {variant: OfficialMetricAccumulator() for variant in VARIANTS}
        for stage in STAGES
    }
    standalone_metrics["deep"] = {"raw": OfficialMetricAccumulator(), "full": OfficialMetricAccumulator()}
    pipeline_metrics = {
        name: OfficialMetricAccumulator()
        for name in ("raw_no_gate", "full_no_gate", "raw_official_gate", "full_official_gate")
    }
    different_pixels = defaultdict(int)
    response_rows = []
    gate_vectors = {stage: [] for stage in STAGES}
    image_labels = []
    present_confusion = {stage: defaultdict(int) for stage in STAGES}
    spatial = SpatialTransitionAccumulator(("raw_to_ch", "gsr_to_full"))
    taxonomy = ErrorTaxonomyAccumulator(("gsr_only", "ch_only", "official_full"))
    complementarity = ComplementarityAccumulator()
    class_gate = defaultdict(int)

    for image_index, (name_tuple, image) in enumerate(loader):
        image_id = name_tuple[0]
        image_path = Path(val_root) / "img" / f"{image_id}.png"
        mask_path = Path(val_root) / "mask" / f"{image_id}.png"
        original_size = np.asarray(Image.open(image_path)).shape[:2]
        ground_truth = np.asarray(Image.open(mask_path), dtype=np.uint8)
        if ground_truth.shape != tuple(original_size):
            raise AssertionError(f"Image/mask size mismatch for {image_id}")
        if not set(np.unique(ground_truth).tolist()).issubset(set(range(5))):
            raise ValueError(f"Unexpected BCSS label for {image_id}")
        result = audit_tta_forward(
            model, image.cuda(non_blocking=True), original_size, amp_dtype=amp_dtype
        )
        probability = result["probability"]
        presence = (probability > np.asarray(BCSS_THRESHOLDS, dtype=np.float32)).astype(np.float32)
        if presence.sum() == 0:
            presence[int(np.argmax(probability))] = 1.0
        gt_presence = _gt_presence(ground_truth)
        image_labels.append(gt_presence)
        for stage in STAGES:
            gate_vectors[stage].append(result["gates"][stage])
        response_rows.extend(_response_rows(image_id, result, ground_truth, gt_presence))

        predictions = {}
        for name, (variant_28_1, variant_28_2) in FINAL_VARIANTS.items():
            prediction, _ = prediction_from_fusion(
                result["cam_relu"][variant_28_1]["28_1"],
                result["cam_relu"][variant_28_2]["28_2"],
                result["deep_relu"],
                presence=presence,
            )
            predictions[name] = prediction
            final_metrics[name].update(ground_truth, prediction)
        full_prediction = predictions["official_full"]
        for name, prediction in predictions.items():
            different_pixels[name] += int(np.count_nonzero(prediction != full_prediction))

        raw_no_gate, _ = prediction_from_fusion(
            result["cam_relu"]["raw"]["28_1"],
            result["cam_relu"]["raw"]["28_2"], result["deep_relu"], presence=None
        )
        full_no_gate, _ = prediction_from_fusion(
            result["cam_relu"]["full"]["28_1"],
            result["cam_relu"]["full"]["28_2"], result["deep_relu"], presence=None
        )
        pipeline_predictions = {
            "raw_no_gate": raw_no_gate,
            "full_no_gate": full_no_gate,
            "raw_official_gate": predictions["all_hfrm_off"],
            "full_official_gate": full_prediction,
        }
        for name, prediction in pipeline_predictions.items():
            pipeline_metrics[name].update(ground_truth, prediction)
        _gate_contribution_update(class_gate, ground_truth, full_no_gate, full_prediction, gt_presence)

        for stage in STAGES:
            stage_predictions = {}
            for variant in VARIANTS:
                prediction = prediction_from_standalone(
                    result["cam_relu"][variant][stage], presence
                )
                stage_predictions[variant] = prediction
                standalone_metrics[stage][variant].update(ground_truth, prediction)
            _present_confusion_update(
                present_confusion, stage, ground_truth,
                stage_predictions["raw"], stage_predictions["gsr"], gt_presence,
            )
        deep_prediction = prediction_from_standalone(result["deep_relu"], presence)
        standalone_metrics["deep"]["raw"].update(ground_truth, deep_prediction)
        standalone_metrics["deep"]["full"].update(ground_truth, deep_prediction)

        bins = foreground_boundary_distance(ground_truth)
        spatial.update(
            "raw_to_ch", ground_truth,
            predictions["all_hfrm_off"], predictions["ch_only"], bins,
        )
        spatial.update(
            "gsr_to_full", ground_truth,
            predictions["gsr_only"], full_prediction, bins,
        )
        for candidate in ("gsr_only", "ch_only", "official_full"):
            taxonomy.update(
                candidate, ground_truth, predictions["all_hfrm_off"],
                predictions[candidate], gt_presence, bins,
            )
        complementarity.update(
            ground_truth, predictions["all_hfrm_off"], predictions["gsr_only"],
            predictions["ch_only"], full_prediction,
        )
        if (image_index + 1) % 200 == 0:
            print(f"VALIDATION_PROGRESS {image_index + 1}/{len(dataset)}", flush=True)

    response_frame = pd.DataFrame(response_rows)
    gate_arrays = {
        stage: np.stack(values).astype(np.float32) for stage, values in gate_vectors.items()
    }
    gate_statistics, gate_semantics = summarize_gates(gate_arrays, np.stack(image_labels))
    final_scores = {
        name: {
            **accumulator.scores(),
            "differing_pixels_vs_full": int(different_pixels[name]),
        }
        for name, accumulator in final_metrics.items()
    }
    standalone_scores = {
        stage: {
            variant: accumulator.scores()
            for variant, accumulator in variants.items()
        }
        for stage, variants in standalone_metrics.items()
    }
    pipeline_scores = {
        name: accumulator.scores() for name, accumulator in pipeline_metrics.items()
    }
    present_confusion_summary = {}
    for stage, values in present_confusion.items():
        raw_errors = values["raw_present_confusion"]
        present_confusion_summary[stage] = {
            **{key: int(value) for key, value in values.items()},
            "net": int(values["recovered"] - values["harmed"]),
            "recovery_rate": float(values["recovered"] / max(raw_errors, 1)),
        }
    class_gate_summary = {
        **{key: int(value) for key, value in class_gate.items()},
        "net_pixels": int(class_gate["recovered_pixels"] - class_gate["harmed_pixels"]),
    }
    return {
        "response_frame": response_frame,
        "gate_vectors": gate_arrays,
        "gate_semantic_rows": gate_semantics,
        "summary": {
            "final_variants": final_scores,
            "standalone_cam": standalone_scores,
            "pipeline_decomposition": pipeline_scores,
            "class_gate_contribution": class_gate_summary,
            "gate_statistics": gate_statistics,
            "gsr_response": summarize_response(response_frame),
            "present_confusion": present_confusion_summary,
            "ch_spatial_effect": spatial.summary(),
            "error_taxonomy": taxonomy.summary(),
            "complementarity": complementarity.summary(),
        },
    }
