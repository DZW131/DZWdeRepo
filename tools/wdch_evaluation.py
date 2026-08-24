"""Official BCSS evaluation plus observation-only WD-CH diagnostics."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from tool.GenDataset import Stage1_InferDataset
from tools.wdch_common import (
    CAM_WEIGHTS,
    OfficialMetricAccumulator,
    TTA_TRANSFORMS,
    minmax_normalize,
    presence_from_probability,
    verify_validation_root,
)


STAGES = ("56", "28_1", "28_2", "deep")


def forward_cam_compatible(model, x):
    """Run the released CAM path for either the training Net or Net_CAM.

    The public repository exposes ``forward_cam`` only on ``Net_CAM``, while
    matched continuation necessarily keeps the trainable ``Net`` instance.
    This is the same released forward_cam equation applied to that instance;
    no state is copied and no evaluation-only model is introduced.
    """
    if hasattr(model, "forward_cam"):
        return model.forward_cam(x)

    x = model.conv1a(x)
    x = model.b2(x); x = model.b2_1(x); x = model.b2_2(x)
    x = model.b3(x); x = model.b3_1(x); x = model.b3_2(x)
    feat_56 = x
    x = model.b4(x); x = model.b4_1(x); x = model.b4_2(x)
    x = model.b4_3(x); x = model.b4_4(x); x = model.b4_5(x)
    feat_28_1 = F.relu(model.bn45(x))
    x, _ = model.b5(x, get_x_bn_relu=True)
    x = model.b5_1(x); x = model.b5_2(x)
    feat_28_2 = F.relu(model.bn52(x))
    x, _ = model.b6(x, get_x_bn_relu=True)
    x = model.b7(x)
    feat_deep = F.relu(model.bn7(x))

    cam_56 = F.relu(model.ic_56(model.hfrm_56(feat_56, feat_deep)))
    cam_28_1 = F.relu(model.ic1(model.hfrm_28_1(feat_28_1, feat_deep)))
    cam_28_2 = F.relu(model.ic2(model.hfrm_28_2(feat_28_2, feat_deep)))
    raw_deep = model.fc8(feat_deep)
    cam_deep = F.relu(raw_deep)
    probability = torch.sigmoid(F.adaptive_avg_pool2d(raw_deep, 1).flatten(1))
    return cam_56, cam_28_1, cam_28_2, cam_deep, probability


def _resize_unflip(cam, size, output_dims):
    cam = F.interpolate(cam, size, mode="bilinear", align_corners=False)[0]
    return torch.flip(cam, dims=output_dims) if output_dims else cam


def _rms(tensor):
    return float(tensor.detach().float().square().mean().sqrt())


def _feature_diagnostics(module, feature):
    input_rms = _rms(feature)
    frequency = {}
    if hasattr(module, "wdch") and hasattr(module.wdch, "variant"):
        context, raw = module.wdch.forward_with_diagnostics(feature)
        operator = f"FDHR-{module.wdch.variant}-k{module.wdch.kernel_size}"
        ablated = []
        frequency = {key: float(value) for key, value in raw.items()}
    elif hasattr(module, "wdch"):
        context = module.wdch(feature)
        operator = f"WDCH{module.wdch.kernel_size}"
        ablated = list(module.wdch.ablated_bands)
    else:
        context = module.context_conv(feature)
        operator = "CH15"
        ablated = []
    return {
        "operator": operator,
        "ablated_bands": ablated,
        "input_rms": input_rms,
        "context_output_rms": _rms(context),
        "output_input_rms": _rms(context) / max(input_rms, 1.0e-12),
        "rectification_rms": _rms(context - feature) / max(input_rms, 1.0e-12),
        "effective_context_residual_rms": _rms(module.gamma_context * context)
        / max(input_rms, 1.0e-12),
        "gamma_context": float(module.gamma_context.detach().float()),
        "gamma_veto": float(module.gamma_veto.detach().float()),
        **frequency,
    }


def evaluate_bcss(
    model,
    val_root: str,
    num_workers: int = 4,
    prediction_output: str | None = None,
):
    verify_validation_root(val_root)
    model = model.cuda()
    model.eval()
    dataset = Stage1_InferDataset(os.path.join(val_root, "img"), img_size=224)
    if len(dataset) != 3418:
        raise AssertionError(f"Expected 3418 images, found {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    metrics = {stage: OfficialMetricAccumulator() for stage in (*STAGES, "final")}
    predictions, truths, image_ids = [], [], []
    feature_rows = []
    captured = {}

    def capture_input(_module, inputs):
        captured["feature"] = inputs[0].detach()

    hook = model.hfrm_28_1.register_forward_pre_hook(capture_input)
    torch.cuda.synchronize()
    started = time.time()
    try:
        with torch.no_grad():
            for index, (name_tuple, image) in enumerate(loader, start=1):
                image_id = name_tuple[0]
                truth = np.asarray(
                    Image.open(Path(val_root) / "mask" / f"{image_id}.png"),
                    dtype=np.uint8,
                )
                image = image.cuda(non_blocking=True)
                views = {stage: [] for stage in STAGES}
                probabilities = []
                feature_row = None
                for view_index, (input_dims, output_dims) in enumerate(TTA_TRANSFORMS):
                    augmented = torch.flip(image, dims=input_dims) if input_dims else image
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        cam56, cam28, cam28_2, camdeep, probability = (
                            forward_cam_compatible(model, augmented)
                        )
                        if view_index == 0:
                            feature_row = _feature_diagnostics(
                                model.hfrm_28_1, captured["feature"]
                            )
                    for stage, value in (
                        ("56", cam56),
                        ("28_1", cam28),
                        ("28_2", cam28_2),
                        ("deep", camdeep),
                    ):
                        views[stage].append(
                            _resize_unflip(value, truth.shape, output_dims)
                        )
                    probabilities.append(probability[0])
                feature_rows.append({"image": image_id, **feature_row})
                probability = (
                    torch.stack(probabilities).mean(0).detach().float().cpu().numpy()
                )
                presence = presence_from_probability(probability)
                normalized = {
                    stage: minmax_normalize(
                        torch.stack(values).mean(0).detach().float().cpu().numpy()
                    )
                    for stage, values in views.items()
                }
                for stage in STAGES:
                    response = normalized[stage] * presence.reshape(4, 1, 1)
                    metrics[stage].update(truth, response.argmax(0).astype(np.uint8))
                fused = sum(CAM_WEIGHTS[stage] * normalized[stage] for stage in STAGES)
                fused *= presence.reshape(4, 1, 1)
                prediction = fused.argmax(0).astype(np.uint8)
                metrics["final"].update(truth, prediction)
                if prediction_output is not None:
                    predictions.append(prediction)
                    truths.append(truth)
                    image_ids.append(image_id)
                if index % 200 == 0:
                    print(f"WDCH_EVAL_PROGRESS {index}/{len(dataset)}", flush=True)
    finally:
        hook.remove()
    torch.cuda.synchronize()
    elapsed = time.time() - started
    if prediction_output is not None:
        target = Path(prediction_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            image_ids=np.asarray(image_ids),
            predictions=np.stack(predictions),
            truths=np.stack(truths),
        )
    feature_summary = {}
    for key in (
        "input_rms",
        "context_output_rms",
        "output_input_rms",
        "rectification_rms",
        "effective_context_residual_rms",
    ):
        values = np.asarray([row[key] for row in feature_rows], dtype=np.float64)
        feature_summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    for key in ("E_LL", "E_HF", "interaction_rms", "interaction_input_rms"):
        if key in feature_rows[0]:
            values = np.asarray([row[key] for row in feature_rows], dtype=np.float64)
            feature_summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }
    feature_summary.update(
        {
            "operator": feature_rows[0]["operator"],
            "ablated_bands": feature_rows[0]["ablated_bands"],
            "gamma_context": feature_rows[0]["gamma_context"],
            "gamma_veto": feature_rows[0]["gamma_veto"],
        }
    )
    return {
        "scores": {stage: accumulator.result() for stage, accumulator in metrics.items()},
        "feature_diagnostics": feature_summary,
        "runtime": {
            "images": len(dataset),
            "seconds": elapsed,
            "seconds_per_image": elapsed / len(dataset),
            "precision": "bf16",
            "tta": TTA_TRANSFORMS,
            "fusion": [0.0, 0.6, 0.2, 0.2],
        },
        "test_used": False,
    }
