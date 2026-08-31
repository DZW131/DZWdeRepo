"""Bounded Phase2B1.12 validation snapshots, without changing official inference.

Provenance: docs/rddr_phase2b112_execution_contract.md and the approved
RDDR_Phase2B1_12_Short_Horizon_ADT_Optimization_Dynamics_Audit_v1.0.md,
sections 13--25.  The canonical tool.infer_fun.infer is called literally:
BCSS, 224 input, BF16, its default class thresholds and CAM weights, all three
official TTAs, original-image output resolution, and original scores().

The only instrumentation is a sorted original dataset factory, a forward_cam
observer returning the *same* output object, and a scores/_fast_hist observer.
Native28 probes are first-TTA diagnostics, never inputs to official predictions.
This module neither constructs an optimizer nor steps, tunes, or saves a model.
The caller owns model placement and evaluation RNG save/restore.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools.rddr_phase2b112_common import (
    NATIVE_SHA, SNAPSHOTS, adjudicate, clean, digest, sha256, write_json,
)


N_IMAGES = 3418
N_TTA = 3
NATIVE_SHAPE = (4, 28, 28)
REFERENCE_NAME = "representation_reference_step0.npz"


def _require(condition, message):
    """Scientific checks must still run if Python is launched with -O."""
    if not condition:
        raise RuntimeError(message)


def _load_native(path):
    actual = sha256(path)
    _require(actual == NATIVE_SHA, f"Immutable native cache SHA mismatch: {actual}")
    with np.load(path, allow_pickle=False) as archive:
        required = ("names", "truth", "boundary", "top20")
        _require(set(required).issubset(archive.files), "Native cache fields missing")
        native = {key: archive[key] for key in required}
    names = native["names"].astype(str)
    _require(names.shape == (N_IMAGES,), "Validation must contain all 3418 images")
    _require(len(set(names.tolist())) == N_IMAGES, "Duplicate native image names")
    _require(names.tolist() == sorted(names.tolist()), "Native names are not sorted")
    for key in ("truth", "boundary", "top20"):
        _require(native[key].shape == (N_IMAGES, 784), f"Native {key} shape mismatch")
    _require(np.isin(native["truth"], (0, 1, 2, 3, 4, 255)).all(),
             "Native truth has unexpected labels")
    for key in ("boundary", "top20"):
        _require(np.isin(native[key], (0, 1)).all(), f"Native {key} is not binary")
        native[key] = native[key].astype(bool)
    native["truth"] = native["truth"].astype(np.uint8)
    native["names"] = names
    return native


def _reference_selection(n):
    """Exact fixed32 + independent seed42 random128 rule from Phase2B1.9."""
    fixed = np.linspace(0, n - 1, 32, dtype=int)
    remaining = np.setdiff1d(np.arange(n), fixed)
    random = np.random.default_rng(42).choice(remaining, 128, replace=False)
    selection = np.r_[fixed, random]
    _require(len(np.unique(selection)) == 160, "Representation selection is not 160 unique images")
    return selection


def _load_reference(path, selection, names, state_sha, step):
    if not path.is_file():
        raise FileNotFoundError(f"Evaluate step0 arm B first: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = ("indices", "names", "raw_features", "raw_logits", "deep_logits",
                "native_sha256", "model_state_sha256")
        _require(set(keys).issubset(archive.files), "Representation reference fields missing")
        reference = {key: archive[key] for key in keys}
    _require(np.array_equal(reference["indices"], selection), "Representation selection changed")
    _require(np.array_equal(reference["names"].astype(str), names[selection]),
             "Representation reference names changed")
    _require(str(reference["native_sha256"].item()) == NATIVE_SHA,
             "Representation reference native provenance changed")
    if step == 0:
        _require(str(reference["model_state_sha256"].item()) == state_sha,
                 "Step0 arm is not bitwise-identical to arm B model state")
    for key, shape in (("raw_features", (160, 512, 784)),
                       ("raw_logits", (160, 4, 784)), ("deep_logits", (160, 4, 784))):
        _require(reference[key].shape == shape and reference[key].dtype == np.float32,
                 f"Representation reference {key} shape/dtype mismatch")
        _require(np.isfinite(reference[key]).all(), f"Nonfinite representation reference {key}")
    return reference


def _cosine_and_ratio(current, reference):
    # Accumulate per-image representation statistics in FP64, although the
    # immutable feature/logit observations themselves are stored as numpy FP32.
    left = np.asarray(current, dtype=np.float64).ravel()
    right = np.asarray(reference, dtype=np.float64).ravel()
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    _require(left_norm > 0 and right_norm > 0, "Zero-norm representation/logit vector")
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    _require(np.isfinite(cosine), "Nonfinite representation cosine")
    return cosine, left_norm / right_norm


def _metrics_from_cm(per_image_cm):
    """Independent parity calculation; never substitutes for original scores()."""
    hist = per_image_cm.sum(axis=0, dtype=np.int64).astype(np.float64)
    # This is the original evaluator's dataset-level [4,4] exclusion, applied
    # AFTER summation. The stored per-image matrices deliberately retain [4,4].
    hist[4, 4] = 0
    diagonal = np.diag(hist)[:4]
    rows, columns = hist.sum(axis=1)[:4], hist.sum(axis=0)[:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = diagonal / (rows + columns - diagonal)
        dice = np.divide(2 * diagonal, rows + columns,
                         out=np.zeros(4, dtype=np.float64), where=(rows + columns) > 0)
    return {"miou": float(np.nanmean(iou)), "mdice": float(dice.mean()),
            "iou": iou, "dice": dice}


def _save_npz_exclusive(path, arrays):
    # Passing an exclusively-created file handle also prevents np.savez from
    # silently appending an extension or overwriting an existing audit result.
    # A failed write stays as a visible partial artifact and is NOT auto-reused.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def evaluate_snapshot(model, val_root, native_path, output_dir, arm, step):
    """Run one complete official validation and save its native28 observations.

    Returns a JSON-safe small summary with ``representation_rows`` (160 rows).
    ``official_miou``/``official_mdice`` and per-class arrays are fractions;
    ``*_pp`` scalar aliases are percentage points. The untouched canonical
    scores result is retained as ``official_result``. Snapshot files are never
    overwritten. Step0 B must be called first to create the feature reference.

    ``model`` must be the already-loaded four-class Net_CAM on CUDA. Module
    training flags and parameter requires_grad flags are restored even on error.
    RNG save/restore and model/optimizer offloading remain caller responsibilities.
    """
    from tool import infer_fun
    from network.resnet38_cls import Net_CAM

    _require(arm in ("B", "A", "R"), f"Unknown arm: {arm}")
    _require(isinstance(step, (int, np.integer)) and step in SNAPSHOTS,
             f"Unapproved snapshot step: {step}")
    step = int(step)
    _require(isinstance(model, Net_CAM), "Expected the official Net_CAM model")
    _require(model.ic1.out_channels == model.fc8.out_channels == 4,
             "Expected four foreground classes")
    _require(all(parameter.device.type == "cuda" for parameter in model.parameters()),
             "The caller must put this arm on CUDA before evaluation")
    val_root, native_path, output_dir = map(Path, (val_root, native_path, output_dir))
    _require(val_root.name.lower() == "val" and val_root.parent.name.lower() == "bcss-wsss",
             "Only the BCSS-WSSS validation split is authorized")
    _require((val_root / "img").is_dir() and (val_root / "mask").is_dir(),
             "Validation img/mask directories missing")
    snapshot_path = output_dir / f"snapshot_{step:04d}_{arm}.npz"
    summary_path = snapshot_path.with_suffix(".json")
    reference_path = output_dir / REFERENCE_NAME
    create_reference = step == 0 and arm == "B"
    for path in (snapshot_path, summary_path, *([reference_path] if create_reference else [])):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite snapshot/reference: {path}")
    started = time.perf_counter()
    native = _load_native(native_path)
    names = native["names"]
    selection = _reference_selection(len(names))
    selection_positions = {int(index): position for position, index in enumerate(selection)}
    state_before = digest(model.state_dict().items())
    reference = None if create_reference else _load_reference(
        reference_path, selection, names, state_before, step)
    if create_reference:
        reference = {"indices": selection, "names": names[selection],
                     "raw_features": np.empty((160, 512, 784), dtype=np.float32),
                     "raw_logits": np.empty((160, 4, 784), dtype=np.float32),
                     "deep_logits": np.empty((160, 4, 784), dtype=np.float32),
                     "native_sha256": np.asarray(NATIVE_SHA),
                     "model_state_sha256": np.asarray(state_before)}
    arrays = {key: np.empty((N_IMAGES, 4, 784), dtype=np.float32)
              for key in ("ps", "pd", "rect", "raw_logits", "deep_logits")}
    arrays.update({key: np.empty((N_IMAGES, 784), dtype=np.float32) for key in ("q", "delta")})
    capture = {}
    tracker = {"forward_calls": 0, "images": 0, "dataset_calls": 0, "scores_calls": 0,
               "hook_active": False, "hook_calls": 0, "failure": None}
    confusion = []
    representation_rows = {}
    original_dataset = infer_fun.Stage1_InferDataset
    original_scores = infer_fun.iouutils.scores
    original_hist = infer_fun.iouutils._fast_hist
    original_forward_cam = model.forward_cam

    def dataset_factory(*args, **kwargs):
        tracker["dataset_calls"] += 1
        dataset = original_dataset(*args, **kwargs)
        dataset.object = sorted(dataset.object)
        actual_names = np.asarray([Path(filename).stem for filename in dataset.object])
        _require(np.array_equal(actual_names, names),
                 "Sorted official validation dataset differs from immutable native names")
        return dataset

    def feature_hook(_module, inputs, output):
        if tracker["hook_active"]:
            tracker["hook_calls"] += 1
            _require(len(inputs) == 2, "Unexpected HFRM28_1 inputs")
            capture.update(raw=inputs[0], deep=inputs[1], rect=output)
        # Returning None leaves the original HFRM output untouched.

    def forward_cam_observer(image):
        call_index = tracker["forward_calls"]
        first_tta = call_index % N_TTA == 0
        tracker["hook_active"] = first_tta
        capture.clear()
        try:
            outputs = original_forward_cam(image)
            if first_tta:
                index = call_index // N_TTA
                _require(index < N_IMAGES, "More than 3418 validation images observed")
                _require(image.shape == (1, 3, 224, 224), "Official validation input shape changed")
                _require(set(capture) == {"raw", "deep", "rect"}, "Native feature hook did not fire")
                _require(capture["raw"].shape == capture["rect"].shape == (1, 512, 28, 28),
                         "Unexpected raw/rect native feature shape")
                _require(capture["deep"].shape == (1, 4096, 28, 28),
                         "Unexpected native deep feature shape")
                _require(all(not module.training for module in model.modules()),
                         "Native probes must run in eval mode without dropout")
                _require(not torch.is_grad_enabled(), "Canonical inference lost its no_grad scope")
                _require(torch.is_autocast_enabled("cuda") and
                         torch.get_autocast_dtype("cuda") == torch.bfloat16,
                         "Native heads must reuse the existing canonical BF16 autocast scope")
                # Net.forward applies ic1 to rectified F28 and fc8 to deep F28;
                # dropout7 is identity in eval. The raw counterfactual reuses
                # that SAME ic1 head before HFRM. Do not softmax ReLU CAMs.
                # These operations occur after original_forward_cam returns,
                # still within infer()'s existing BF16 autocast context.
                raw_logits = F.conv2d(capture["raw"], model.ic1.weight, model.ic1.bias)
                rect_logits = F.conv2d(capture["rect"], model.ic1.weight, model.ic1.bias)
                deep_logits = F.conv2d(capture["deep"], model.fc8.weight, model.fc8.bias)
                _require(tuple(raw_logits.shape[1:]) == NATIVE_SHAPE and
                         raw_logits.shape == rect_logits.shape == deep_logits.shape,
                         "Native head shape changed")
                ps, pd, rect = (value.float().softmax(1)
                                for value in (raw_logits, deep_logits, rect_logits))
                # This is the shared, exact Phase2B1.5 15x15/radius7/exclude-self
                # adjudicator; no alternate gate formula, window, or threshold.
                evidence = adjudicate(ps, pd)
                tensors = {"ps": ps, "pd": pd, "rect": rect,
                           "raw_logits": raw_logits, "deep_logits": deep_logits,
                           "q": evidence["q"], "delta": evidence["delta"]}
                for key, value in tensors.items():
                    array = value.detach().float().cpu().numpy().reshape(arrays[key][index].shape)
                    _require(np.isfinite(array).all(), f"Nonfinite native {key} at {names[index]}")
                    arrays[key][index] = array
                if index in selection_positions:
                    position = selection_positions[index]
                    raw_feature = capture["raw"].detach().float().cpu().numpy().reshape(512, 784)
                    _require(np.isfinite(raw_feature).all(), "Nonfinite raw representation")
                    current = {"raw_features": raw_feature,
                               "raw_logits": arrays["raw_logits"][index],
                               "deep_logits": arrays["deep_logits"][index]}
                    if create_reference:
                        for key, value in current.items():
                            reference[key][position] = value
                    if step == 0:
                        _require(all(np.array_equal(value, reference[key][position])
                                     for key, value in current.items()),
                                 "Step0 representation/logits differ from arm B")
                    feature_cosine, feature_ratio = _cosine_and_ratio(
                        raw_feature, reference["raw_features"][position])
                    raw_cosine, _ = _cosine_and_ratio(
                        current["raw_logits"], reference["raw_logits"][position])
                    deep_cosine, _ = _cosine_and_ratio(
                        current["deep_logits"], reference["deep_logits"][position])
                    representation_rows[index] = {"arm": arm, "step": step, "name": str(names[index]),
                        "feature_cosine": feature_cosine, "feature_norm_ratio": feature_ratio,
                        "raw_logits_cosine": raw_cosine, "deep_logits_cosine": deep_cosine}
                tracker["images"] += 1
                if tracker["images"] % 500 == 0:
                    print(f"snapshot step={step} arm={arm}: {tracker['images']}/{N_IMAGES} images", flush=True)
            tracker["forward_calls"] += 1
            # Identity, not a reconstructed tuple: official CAM/y values and
            # their normalizations, masks and ensemble downstream are untouched.
            return outputs
        except Exception as error:
            tracker["failure"] = error
            raise
        finally:
            tracker["hook_active"] = False
            capture.clear()

    def scores_observer(label_trues, label_preds, n_class):
        tracker["scores_calls"] += 1
        _require(n_class == 4 and len(label_trues) == len(label_preds) == N_IMAGES,
                 "Official scores did not receive all 3418 validation pairs")

        def histogram_observer(label_true, label_pred, classes):
            _require(classes == 5, "Official scores class count changed")
            # Original scores has ALREADY executed lp[lt == 4] = 4 here.
            # Observe its actual _fast_hist result before dataset aggregation
            # zeros hist[4,4]; do not duplicate/mutate prediction arrays.
            _require(np.all(label_pred[label_true == 4] == 4),
                     "Official GT-background overwrite missing")
            hist = original_hist(label_true, label_pred, classes)
            _require(hist.shape == (5, 5) and np.issubdtype(hist.dtype, np.integer),
                     "Unexpected per-image confusion matrix")
            confusion.append(hist.copy())
            return hist

        with patch.object(infer_fun.iouutils, "_fast_hist", histogram_observer):
            return original_scores(label_trues, label_preds, n_class)

    # Important original-source behavior: Net.train(False), called by eval(),
    # also freezes early/BN requires_grad flags. Restore these flags directly,
    # without invoking Net.train() again and silently changing the caller state.
    training_flags = [(module, module.training) for module in model.modules()]
    gradient_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    handle = model.hfrm_28_1.register_forward_hook(feature_hook)
    try:
        with patch.object(infer_fun, "Stage1_InferDataset", dataset_factory), \
             patch.object(infer_fun.iouutils, "scores", scores_observer), \
             patch.object(model, "forward_cam", forward_cam_observer):
            # No explicit thr/cam_weights: canonical [.8,.9,.8,.6], .6/.2/.2,
            # three TTAs and original-image interpolation are the source defaults.
            official_result = infer_fun.infer(
                model, str(val_root), 4,
                SimpleNamespace(dataset="bcss", img_size=224, amp_dtype="bf16", num_workers=0))
        # Original infer catches exceptions and returns None. Treat this as a
        # hard failed snapshot, never as a missing/zero metric or an empty cache.
        if official_result is None:
            raise RuntimeError("Canonical official inference returned None; snapshot aborted") from tracker["failure"]
    finally:
        handle.remove()
        capture.clear()
        for module, training in training_flags:
            module.training = training
        for parameter, requires_grad in gradient_flags:
            parameter.requires_grad_(requires_grad)
    state_after = digest(model.state_dict().items())
    _require(state_before == state_after, "Snapshot evaluation changed model weights or BN buffers")
    _require(tracker["dataset_calls"] == tracker["scores_calls"] == 1,
             "Canonical dataset/scores call count changed")
    _require(tracker["forward_calls"] == N_IMAGES * N_TTA and
             tracker["images"] == tracker["hook_calls"] == len(confusion) == N_IMAGES,
             "Incomplete first-TTA/native/official snapshot coverage")
    _require(set(representation_rows) == set(selection.tolist()), "Incomplete representation coverage")
    _require(sha256(native_path) == NATIVE_SHA, "Immutable native cache changed during inference")
    arrays.update(native)
    arrays["official_cm"] = np.stack(confusion).astype(np.int64, copy=False)
    parity = _metrics_from_cm(arrays["official_cm"])
    for key, actual in (("Mean IoU", parity["miou"]), ("Mean Dice", parity["mdice"])):
        _require(np.isclose(float(official_result[key]), actual, rtol=0, atol=1e-12),
                 f"Captured confusion matrices do not reproduce canonical {key}")
    for key, values in (("Class IoU", parity["iou"]), ("Dice Coefficients", parity["dice"])):
        _require(np.allclose([official_result[key][i] for i in range(4)], values,
                            rtol=0, atol=1e-12, equal_nan=True),
                 f"Captured confusion matrices do not reproduce canonical {key}")
    _require(np.isfinite(parity["miou"]) and np.isfinite(parity["mdice"]),
             "Official dataset metrics are nonfinite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if create_reference:
        _save_npz_exclusive(reference_path, reference)
    _save_npz_exclusive(snapshot_path, arrays)
    summary = clean({"arm": arm, "step": step, "images": N_IMAGES,
        "snapshot_path": snapshot_path, "snapshot_sha256": sha256(snapshot_path),
        "official_miou": official_result["Mean IoU"], "official_mdice": official_result["Mean Dice"],
        "official_miou_pp": 100 * float(official_result["Mean IoU"]),
        "official_mdice_pp": 100 * float(official_result["Mean Dice"]),
        "official_per_class_iou": [official_result["Class IoU"][i] for i in range(4)],
        "official_per_class_dice": [official_result["Dice Coefficients"][i] for i in range(4)],
        "official_result": official_result, "metric_units": "fraction (explicit *_pp aliases are percent)",
        "official_cm_parity": True, "official_cm_background_retained": True,
        "native_sha256": NATIVE_SHA, "state_before": state_before, "state_after": state_after,
        "state_unchanged": state_before == state_after, "training_and_requires_grad_flags_restored": True,
        "native_probe_tta": "first/unflipped only", "forward_cam_calls": tracker["forward_calls"],
        "threshold_policy": "canonical default [.8,.9,.8,.6]; no explicit thr",
        "cam_weights_policy": "canonical default [.6,.2,.2]", "official_tta_count": N_TTA,
        "official_output_resolution": "original image", "network_precision": "BF16 autocast",
        "native_array_precision": "FP32", "representation_statistics_precision": "FP64",
        "representation_reference_path": reference_path,
        "representation_reference_sha256": sha256(reference_path),
        "representation_rows": [representation_rows[int(index)] for index in selection],
        "canonical_infer_source_sha256": sha256(Path(infer_fun.__file__)),
        "canonical_scores_source_sha256": sha256(Path(infer_fun.iouutils.__file__)),
        "known_c0_official_miou_pp": 67.3102,
        "step0_minus_known_c0_miou_pp": 100 * float(official_result["Mean IoU"]) - 67.3102 if step == 0 else None,
        "known_c0_comparison_policy": "report only; never adjust predictions or metrics",
        "elapsed_seconds": time.perf_counter() - started})
    # Reserve exclusively, then let the shared JSON writer fill this explicitly
    # owned empty file. An existing result is rejected even in a concurrent call.
    with summary_path.open("x", encoding="utf-8"):
        pass
    write_json(summary_path, summary)
    return summary
