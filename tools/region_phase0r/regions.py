"""Connected-component metadata and frozen feature pooling."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.measure import perimeter as sk_perimeter

from tools.region_phase0r import BACKGROUND, GEOMETRY_COLUMNS, PURITY_THRESHOLD


CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


def slide_id_from_image(image_id: str) -> str:
    slide = image_id.split("_xmin", 1)[0]
    if not slide.startswith("TCGA-"):
        raise ValueError(f"Cannot parse BCSS slide id from {image_id}")
    return slide


def connected_components(prediction: np.ndarray):
    result = {}
    for predicted_class in range(4):
        labels, count = ndimage.label(
            prediction == predicted_class, structure=CONNECTIVITY_8
        )
        result[predicted_class] = (labels.astype(np.int32), int(count))
    return result


def _pool_tokens(feature: torch.Tensor, masks, bboxes, centroids, image_shape):
    if not masks:
        empty = np.empty((0, feature.shape[0]), dtype=np.float32)
        return empty, empty.copy(), empty.copy()
    height, width = image_shape
    mask_tensor = torch.from_numpy(np.stack(masks).astype(np.float32))[:, None]
    bbox_tensor = torch.zeros_like(mask_tensor)
    for index, (y0, x0, y1, x1) in enumerate(bboxes):
        bbox_tensor[index, 0, y0:y1, x0:x1] = 1.0
    target_size = tuple(feature.shape[-2:])
    region_weights = F.interpolate(mask_tensor, target_size, mode="area")[:, 0]
    bbox_weights = F.interpolate(bbox_tensor, target_size, mode="area")[:, 0]
    flat_feature = feature.float().reshape(feature.shape[0], -1).t()

    def weighted(weights):
        flat = weights.reshape(weights.shape[0], -1)
        return (flat @ flat_feature) / (flat.sum(1, keepdim=True) + 1e-8)

    region_tokens = weighted(region_weights)
    bbox_tokens = weighted(bbox_weights)
    grid = torch.tensor(
        [
            [2.0 * (x + 0.5) / width - 1.0, 2.0 * (y + 0.5) / height - 1.0]
            for y, x in centroids
        ],
        dtype=torch.float32,
    ).reshape(1, -1, 1, 2)
    centroid_tokens = F.grid_sample(
        feature[None].float(), grid, mode="bilinear", align_corners=False
    )[0, :, :, 0].t()
    return (
        region_tokens.numpy().astype(np.float32),
        bbox_tokens.numpy().astype(np.float32),
        centroid_tokens.numpy().astype(np.float32),
    )


def extract_image_regions(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    feature: torch.Tensor,
    image_id: str,
    image_index: int,
):
    height, width = prediction.shape
    slide_id = slide_id_from_image(image_id)
    components = connected_components(prediction)
    rows, masks, bboxes, centroids = [], [], [], []
    for predicted_class in range(4):
        label_map, count = components[predicted_class]
        areas = np.bincount(label_map.ravel(), minlength=count + 1)[1:]
        order = np.argsort(-areas, kind="stable")
        ranks = np.empty(count, dtype=np.int32)
        ranks[order] = np.arange(1, count + 1)
        for component_label in range(1, count + 1):
            mask = label_map == component_label
            ys, xs = np.nonzero(mask)
            area = int(len(ys))
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            bbox_area = (y1 - y0) * (x1 - x0)
            centroid_y, centroid_x = float(ys.mean()), float(xs.mean())
            perimeter = float(sk_perimeter(mask, neighborhood=8))
            counts = np.bincount(ground_truth[mask].astype(np.int64), minlength=5)
            majority_gt = int(np.argmax(counts))
            purity = float(counts[majority_gt] / area)
            if purity >= PURITY_THRESHOLD and majority_gt == predicted_class:
                taxonomy = "A_correct_pure"
            elif purity >= PURITY_THRESHOLD and majority_gt == BACKGROUND:
                taxonomy = "C_false_positive_pure"
            elif purity >= PURITY_THRESHOLD:
                taxonomy = "B_misclassified_pure"
            else:
                taxonomy = "D_mixed_boundary"
            wrong_pixels = int(np.count_nonzero(ground_truth[mask] != predicted_class))
            row = {
                "image_index": image_index,
                "image_id": image_id,
                "slide_id": slide_id,
                "region_id": f"{image_id}:{predicted_class}:{component_label}",
                "predicted_class": predicted_class,
                "component_label": component_label,
                "pixel_area": area,
                "area_fraction": area / float(height * width),
                "bbox_y0": y0, "bbox_x0": x0,
                "bbox_y1": y1, "bbox_x1": x1,
                "bbox_area": bbox_area,
                "bbox_fill_ratio": area / float(bbox_area),
                "centroid_y": centroid_y,
                "centroid_x": centroid_x,
                "perimeter": perimeter,
                "boundary_density": perimeter / float(area),
                "aspect_ratio": (x1 - x0) / float(y1 - y0),
                "compactness": 4.0 * math.pi * area / (perimeter * perimeter + 1e-8),
                "component_count_in_image": sum(item[1] for item in components.values()),
                "same_class_component_count": count,
                "component_rank": int(ranks[component_label - 1]),
                "majority_gt": majority_gt,
                "purity": purity,
                "taxonomy": taxonomy,
                "wrong_foreground_pixels": wrong_pixels,
            }
            for gt_class in range(5):
                row[f"gt_pixels_{gt_class}"] = int(counts[gt_class])
            rows.append(row)
            masks.append(mask)
            bboxes.append((y0, x0, y1, x1))
            centroids.append((centroid_y, centroid_x))

    region, bbox, centroid = _pool_tokens(
        feature, masks, bboxes, centroids, prediction.shape
    )
    geometry = np.asarray(
        [
            [
                math.log(row["pixel_area"] + 1.0), row["area_fraction"],
                row["bbox_fill_ratio"], row["aspect_ratio"],
                row["compactness"], row["boundary_density"],
                row["component_rank"], row["same_class_component_count"],
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    assert geometry.shape[1] == len(GEOMETRY_COLUMNS)
    return rows, region, bbox, centroid, geometry


def relabel_predictions(predictions, region_frame, labels):
    """Project region labels back to fixed component supports."""
    output = predictions.copy()
    mappings = {}
    for row, label in zip(region_frame.itertuples(index=False), labels):
        mappings[(row.image_index, row.predicted_class, row.component_label)] = int(label)
    for image_index in sorted(region_frame.image_index.unique()):
        components = connected_components(predictions[image_index])
        for predicted_class, (label_map, count) in components.items():
            for component_label in range(1, count + 1):
                key = (int(image_index), predicted_class, component_label)
                if key in mappings:
                    output[image_index][label_map == component_label] = mappings[key]
    return output
