"""Frozen BCSS validation-only audit. No optimizer, backward or checkpoint save."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.rddr_phase2b0_common import (
    A0, CKPT_SHA, EPS, BINS, VARIANTS, GROUPS, PAIR_GROUPS, FIELDS, ESTIMATORS,
    sha256, write_json, binary_hist, binary_metrics, exact_binary_metrics,
    build_relations, relation_gt_metrics, project, boundary_masks, populations,
    confusion, nanmean,
)


def validate_root(path):
    path = Path(path).resolve()
    if path.name != "val" or path.parent.name != "BCSS-WSSS":
        raise ValueError("Only the BCSS-WSSS/val split is permitted")
    assert (path / "img").is_dir() and (path / "mask").is_dir()
    return path


def state_digest(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight(args, dataset):
    cache_root = Path(args.population_cache)
    manifest_path = cache_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["images"] == 3418
    assert manifest["checkpoint_sha256"] == CKPT_SHA
    assert manifest["source_commit"] == "586f402a30f446c409c625b55953e329cc041dcc"
    expected_hashes = {v["image_id"]: v["sha256"] for v in manifest["files"]}
    with (Path(args.phase0_results) / "rddr_phase0_per_image.csv").open(newline="") as f:
        expected = {r["image_id"]: r for r in csv.DictReader(f)}
    ids = [Path(p).stem for p in dataset.object]
    assert set(ids) == set(expected_hashes) == set(expected)
    assert set(ids) == {p.stem for p in (args.val_root / "mask").glob("*.png")}
    assert len(ids) == len(set(ids)) == 3418
    q_values, counts, full_counts, files = [], {}, {}, []
    for image_id in ids:
        path = cache_root / (image_id + ".npz")
        assert sha256(path) == expected_hashes[image_id], image_id
        mask_path = args.val_root / "mask" / (image_id + ".png")
        gt = np.asarray(Image.open(mask_path), dtype=np.uint8)
        assert gt.shape == (224, 224) and set(np.unique(gt)) <= {0, 1, 2, 3, 4, 255}
        with np.load(path) as cache:
            groups = populations(cache, gt)
            for name, mask in groups.items():
                full_counts[name] = full_counts.get(name, 0) + int(mask.sum())
                counts[name] = counts.get(name, 0) + int(project(mask).sum())
                if name in ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct"):
                    assert int(mask.sum()) == int(expected[image_id][f"ch_{name}_count"])
            assert int(groups["Top20"].sum()) == int(expected[image_id]["S_JS_top20_flagged"])
            assert cache["q_feature"].shape == (28, 28)
            q_values.append(cache["q_feature"][project(groups["all"]).astype(bool)])
        files.append(dict(image_id=image_id, cache_sha256=expected_hashes[image_id], mask_sha256=sha256(mask_path)))
    for name, count in manifest["counts"].items():
        assert full_counts[name] == count
    thresholds = np.quantile(np.concatenate(q_values), [.2, .4, .6, .8], method="higher")
    return dict(source_manifest_sha256=sha256(manifest_path), source_manifest=manifest_path,
                source_commit=manifest["source_commit"], checkpoint_sha256=CKPT_SHA,
                original_per_image_count_parity=True, all_cache_hashes_verified=True,
                historical_pixel_hash_comparison_available=False,
                full_resolution_counts=full_counts, projected_counts=counts,
                q_quintile_thresholds=thresholds, files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-root", required=True)
    parser.add_argument("--population-cache", required=True)
    parser.add_argument("--phase0-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-images", type=int, default=0, choices=(0, 2))
    args = parser.parse_args()
    start = time.perf_counter()
    args.val_root = validate_root(args.val_root)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Never overwrite a previous audit: " + str(output))
    assert sha256(args.checkpoint) == CKPT_SHA
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(["git", "diff", "--name-only", A0], cwd=ROOT, text=True).splitlines()
    assert all(p.startswith(("tools/", "tests/", "docs/", "audit/")) for p in changed), changed
    assert not subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
    assert torch.cuda.is_available()
    # Do not call training set_seed: original Phase-0 used these backend defaults.
    torch.backends.cudnn.benchmark = False
    assert torch.backends.cuda.matmul.fp32_precision == "none"
    assert torch.backends.cudnn.conv.fp32_precision == "tf32"
    torch.set_num_threads(4)
    from network.resnet38_cls import Net
    from tool.GenDataset import Stage1_InferDataset
    dataset = Stage1_InferDataset(str(args.val_root / "img"), img_size=224)
    dataset.object = sorted(dataset.object)
    manifest = preflight(args, dataset)
    print("PREFLIGHT_PASS all 3418 frozen cache SHA/count checks", flush=True)
    output.mkdir(parents=True)
    write_json(output / "rddr_phase2b0_population_manifest.json", manifest)
    selected = list(range(len(dataset))) if not args.smoke_images else [0, len(dataset)-1]
    exact_indices = set(np.linspace(0, 3417, 16, dtype=int).tolist())
    n = len(selected)
    loader = DataLoader(Subset(dataset, selected), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    model = Net(4).cuda()
    load_info = model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False), strict=True)
    # A0 overrides train() without returning self; do not chain eval().
    model.eval()
    model.requires_grad_(False)
    digest_before = state_digest(model)
    captured = {}
    def hook(module, inputs, result):
        captured.update(raw=inputs[0], deep=inputs[1], rect=result)
    hook_handle = model.hfrm_28_1.register_forward_hook(hook)
    shape = (n, len(GROUPS), len(VARIANTS), len(FIELDS))
    sums, counts = np.zeros(shape), np.zeros(shape, dtype=np.int64)
    value_hist = np.zeros((len(GROUPS), len(VARIANTS), len(FIELDS), BINS), dtype=np.int64)
    pair_hist = np.zeros((len(PAIR_GROUPS), 4, 2, BINS), dtype=np.int64)
    pair_values = np.full((n, len(PAIR_GROUPS), 4, 4), np.nan)  # auc,ap,pos,neg
    target_hist = np.zeros((3, 2, BINS), dtype=np.int64)
    target_auc = np.full((n, 3), np.nan)
    cm = np.zeros((n, len(GROUPS), len(ESTIMATORS), 4, 4), dtype=np.int64)
    proper = np.zeros((n, len(GROUPS), len(ESTIMATORS), 3))  # nll sum,brier sum,count
    repair = np.zeros((n, len(GROUPS), len(ESTIMATORS), 3), dtype=np.int64)  # repair,harm,count
    echo = np.zeros((n, len(GROUPS), 5), dtype=np.int64)  # echo,count,not echo,SCorrectDWrong,SWrongDCorrect
    population_counts = np.zeros((n, len(GROUPS)), dtype=np.int64)
    finite_zero_fg_neighbors = 0
    exact_rows, names = [], []
    phase0_q_maxdiff = 0.
    forward_seconds = stat_seconds = 0.
    torch.cuda.reset_peak_memory_stats()
    forward_start = time.perf_counter()
    with torch.no_grad():
        for index, (image_ids, image) in enumerate(loader):
            image_id = image_ids[0]
            names.append(image_id)
            tick = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(image.cuda(non_blocking=True))
                assert captured["raw"].shape == captured["rect"].shape == (1, 512, 28, 28)
                assert captured["deep"].shape == (1, 4096, 28, 28)
                shallow_logits = model.ic1(captured["raw"])
                deep_logits = outputs[8]
            ps, pd = shallow_logits.float().softmax(1), deep_logits.float().softmax(1)
            rel = build_relations(ps, pd)
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            torch.cuda.synchronize()
            forward_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            # No GT loaded until nonoracle relation weights and predictions exist.
            truth_full = np.asarray(Image.open(args.val_root / "mask" / (image_id + ".png")), dtype=np.uint8)
            truth = project(truth_full).astype(np.int64).ravel()
            with np.load(Path(args.population_cache) / (image_id + ".npz")) as cache:
                group = {k: project(v).astype(bool).ravel() for k, v in populations(cache, truth_full).items()}
                q_feature = cache["q_feature"].copy()
            q_live = rel["q"][0].cpu().numpy()
            diff = float(np.abs(q_live-q_feature).max())
            phase0_q_maxdiff = max(phase0_q_maxdiff, diff)
            assert diff == 0., ("Frozen feature q mismatch", image_id, diff)
            gt = relation_gt_metrics(rel, torch.as_tensor(truth, device="cuda"))
            for k, v in boundary_masks(truth_full).items():
                group[k] = project(v).astype(bool).ravel()
            quintile = np.searchsorted(manifest["q_quintile_thresholds"], q_feature.ravel(), side="left")
            for k in range(5):
                group[f"Q{k+1}"] = group["all"] & (quintile == k)
            for k in range(4):
                group[f"class{k}"] = truth == k
            ps_np, pd_np = ps[0].flatten(1).cpu().numpy(), pd[0].flatten(1).cpu().numpy()
            raw_pred, deep_pred = ps_np.argmax(0), pd_np.argmax(0)
            group["Deep_Correct"] = group["all"] & (deep_pred == truth)
            group["Deep_Wrong"] = group["all"] & (deep_pred != truth)
            distributions = np.concatenate([rel["distribution"][0].cpu().numpy(), ps_np[None], pd_np[None], gt["oracle"][0].cpu().numpy()[None]])
            predictions = distributions.argmax(1)
            oracle_valid = gt["oracle_valid"][0].cpu().numpy()
            weights = rel["weights"][0].cpu().numpy()
            eligible = gt["eligible"][0].cpu().numpy()
            same = gt["same"][0].cpu().numpy()
            vals = np.stack([gt["purity"][0].cpu().numpy(), rel["mass"][0].cpu().numpy(),
                             rel["neff"][0].cpu().numpy(), gt["same_mass"][0].cpu().numpy(),
                             gt["wrong_mass"][0].cpu().numpy(), gt["fg_mass"][0].cpu().numpy()], 1)
            finite_zero_fg_neighbors += int((group["all"] & ~np.isfinite(vals[0, 0])).sum())
            for gi, g in enumerate(GROUPS):
                mask = group[g]
                population_counts[index, gi] = int(mask.sum())
                for vi in range(4):
                    for fi in range(len(FIELDS)):
                        arr = vals[vi, fi, mask]
                        arr = arr[np.isfinite(arr)]
                        sums[index, gi, vi, fi] = arr.sum(dtype=np.float64)
                        counts[index, gi, vi, fi] = len(arr)
                        upper = 1 if fi == 0 else 224
                        bins = np.minimum((np.clip(arr/upper, 0, 1)*BINS).astype(int), BINS-1)
                        value_hist[gi, vi, fi] += np.bincount(bins, minlength=BINS)
                for ei in range(len(ESTIMATORS)):
                    emask = mask & (oracle_valid if ei == 6 else True)
                    cm[index, gi, ei] = confusion(truth, predictions[ei], emask)
                    y = truth[emask]
                    prob = distributions[ei][:, emask]
                    proper[index, gi, ei] = [-(np.log(prob[y, np.arange(len(y))]+EPS)).sum(dtype=np.float64),
                                             ((prob-np.eye(4, dtype=np.float32)[y].T)**2).sum(dtype=np.float64), len(y)]
                    repair[index, gi, ei] = [int((emask & (raw_pred != truth) & (predictions[ei] == truth)).sum()),
                                            int((emask & (raw_pred == truth) & (predictions[ei] != truth)).sum()), int(emask.sum())]
                different = predictions[3] != deep_pred
                echo[index, gi] = [int((mask & ~different).sum()), int(mask.sum()), int((mask & different).sum()),
                                  int((mask & different & (predictions[3] == truth) & (deep_pred != truth)).sum()),
                                  int((mask & different & (predictions[3] != truth) & (deep_pred == truth)).sum())]
            for pgi, g in enumerate(PAIR_GROUPS):
                edge_mask = eligible & group[g][None]
                labels = same[edge_mask]
                for vi, variant in enumerate(VARIANTS):
                    score = weights[vi][edge_mask]
                    hist = binary_hist(score, labels)
                    pair_hist[pgi, vi] += hist
                    m = binary_metrics(hist)
                    pair_values[index, pgi, vi] = [m["auroc"], m["auprc"], m["positive"], m["negative"]]
                    if g == "all" and selected[index] in exact_indices:
                        exact = exact_binary_metrics(score, labels)
                        exact_rows.append(dict(image_id=image_id, variant=variant, pairs=len(labels),
                                               exact_auroc=exact["auroc"], hist_auroc=m["auroc"],
                                               exact_auprc=exact["auprc"], hist_auprc=m["auprc"],
                                               abs_auroc_error=abs(exact["auroc"]-m["auroc"]),
                                               abs_auprc_error=abs(exact["auprc"]-m["auprc"])))
            target_mask = group["Corrected_by_CH"] | group["Harmed_by_CH"]
            score_list = [vals[3, 0], (vals[3, 0]-vals[0, 0]+1)/2, 1-vals[3, 4]/224]
            for k, score in enumerate(score_list):
                mask = target_mask & np.isfinite(score)
                hist = binary_hist(score[mask], group["Corrected_by_CH"][mask])
                target_hist[k] += hist
                target_auc[index, k] = binary_metrics(hist)["auroc"]
            stat_seconds += time.perf_counter() - tick
            if (index+1) % 100 == 0 or index+1 == n:
                print(f"AUDIT {index+1}/{n} elapsed={time.perf_counter()-forward_start:.1f}s q_exact={phase0_q_maxdiff == 0}", flush=True)
    hook_handle.remove()
    assert state_digest(model) == digest_before, "A0 state mutated"
    assert sha256(args.checkpoint) == CKPT_SHA
    np.savez_compressed(output / "rddr_phase2b0_sufficient_statistics.npz", names=np.array(names),
                        sums=sums, counts=counts, value_hist=value_hist, pair_hist=pair_hist, pair_values=pair_values,
                        target_hist=target_hist, target_auc=target_auc, cm=cm, proper=proper,
                        repair=repair, echo=echo, population_counts=population_counts)
    from tools.rddr_phase2b0_common import write_csv
    write_csv(output / "rddr_phase2b0_histogram_validation.csv", exact_rows)
    runtime = dict(commit=commit, a0_commit=A0, checkpoint=args.checkpoint, checkpoint_sha256=CKPT_SHA,
                   command=" ".join(sys.argv), argv=sys.argv, images=n, smoke=bool(args.smoke_images),
                   checkpoint_missing_keys=load_info.missing_keys, checkpoint_unexpected_keys=load_info.unexpected_keys,
                   model_state_digest_before_after=digest_before, unchanged_model_state=True,
                   frozen_q_feature_max_abs_difference=phase0_q_maxdiff,
                   target_count=int(population_counts[:, 0].sum()),
                   foreground_pair_count=int(pair_hist[0, 0].sum()),
                   foreground_targets_without_foreground_source=finite_zero_fg_neighbors,
                   actual_relation_count=int(sums[:, 0, 0, 1].sum()),
                   forward_relation_seconds=forward_seconds, statistics_seconds=stat_seconds,
                   total_seconds=time.perf_counter()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), gpu=torch.cuda.get_device_name(),
                   python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                   benchmark=torch.backends.cudnn.benchmark, matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                   conv_precision=torch.backends.cudnn.conv.fp32_precision, requires_grad=False, optimizer_created=False,
                   checkpoint_written=False, test_access=False, luad_access=False,
                   exact_subset_sorted_indices=sorted(exact_indices), bins=BINS)
    write_json(output / "rddr_phase2b0_runtime.json", runtime)
    print("EXTRACTION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
