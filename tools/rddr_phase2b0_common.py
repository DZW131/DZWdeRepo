"""GT-independent relation construction and diagnostic-only statistics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

A0 = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
CKPT_SHA = "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
EPS = 1e-8
BINS = 4096
VARIANTS = ("U", "SR", "SC", "SRSC")
GROUPS = ("all", "Corrected_by_CH", "Still_Wrong", "Harmed_by_CH",
          "Stable_Correct", "Top20", "Bottom80", "Q1", "Q2", "Q3", "Q4", "Q5",
          "boundary", "interior", "class0", "class1", "class2", "class3",
          "Deep_Correct", "Deep_Wrong")
PAIR_GROUPS = ("all", "boundary", "interior", "class0", "class1", "class2", "class3")
FIELDS = ("purity", "mass", "neff", "same_mass", "wrong_mass", "fg_mass")
ESTIMATORS = (*VARIANTS, "raw", "deep", "oracle")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean_json(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x) if isinstance(x, Path) else x


def write_json(path, x):
    Path(path).write_text(json.dumps(clean_json(x), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def phase0_js(p, q, dim=1):
    """Same FP32 operation order and epsilon placement as Phase-0."""
    m = .5 * (p + q)
    return .5 * ((p * ((p + EPS).log() - (m + EPS).log())).sum(dim)
                 + (q * ((q + EPS).log() - (m + EPS).log())).sum(dim))


def neighbors(x):
    b, c, h, w = x.shape
    return F.unfold(x, kernel_size=15, padding=7).reshape(b, c, 225, h * w)


@torch.no_grad()
def build_relations(ps, pd):
    """No GT argument or data access; all nonoracle propagation is GT-blind."""
    assert ps.shape == pd.shape and ps.shape[1:] == (4, 28, 28)
    ps, pd = ps.float(), pd.float()
    source = neighbors(ps)
    valid = neighbors(torch.ones_like(ps[:, :1]))[:, 0].bool()
    valid[:, 112] = False
    q = (phase0_js(ps, pd) / math.log(2)).clamp(0, 1)
    r = neighbors((1 - q)[:, None])[:, 0]
    c = (1 - phase0_js(pd.flatten(2)[:, :, None], source) / math.log(2)).clamp(0, 1)
    weights = torch.stack((torch.ones_like(r), r, c, r * c), 1) * valid[:, None]
    mass = weights.sum(2)
    assert (mass > 0).all(), "Undefined nonoracle neighborhood; no fallback allowed"
    neff = mass.square() / (weights.square().sum(2) + EPS)
    distribution = (weights[:, :, None] * source[:, None]).sum(3) / (mass[:, :, None] + EPS)
    for value in (q, weights, distribution, neff):
        assert torch.isfinite(value).all()
    return dict(q=q, weights=weights, source=source, valid=valid,
                mass=mass, neff=neff, distribution=distribution)


@torch.no_grad()
def relation_gt_metrics(rel, truth):
    """GT first enters HERE, after nonoracle scores and predictions exist."""
    y = truth.reshape(1, 1, 28, 28).float()
    yn = neighbors(y)[:, 0].long()
    target = y.flatten(2)[:, 0].long()
    fg = (target >= 0) & (target < 4)
    eligible = rel["valid"] & (yn >= 0) & (yn < 4) & fg[:, None]
    same = (yn == target[:, None]) & eligible
    weighted_fg = rel["weights"] * eligible[:, None]
    fg_mass = weighted_fg.sum(2)
    same_mass = (weighted_fg * same[:, None]).sum(2)
    purity = same_mass / (fg_mass + EPS)
    purity = purity.masked_fill(fg_mass <= 0, float("nan"))
    oracle_mass = same.sum(1)
    oracle = (rel["source"] * same[:, None]).sum(2) / (oracle_mass[:, None] + EPS)
    return dict(eligible=eligible, same=same, purity=purity, fg_mass=fg_mass,
                same_mass=same_mass, wrong_mass=fg_mass - same_mass,
                oracle=oracle, oracle_valid=oracle_mass > 0)


def project(x):
    return F.interpolate(torch.as_tensor(np.array(x, copy=True))[None, None].float(),
                         (28, 28), mode="nearest")[0, 0].numpy()


def boundary_masks(truth):
    # Verbatim geometry of Phase-0: 8-connected FG-FG transitions, Euclidean EDT.
    boundary = np.zeros_like(truth, dtype=bool)
    h, w = truth.shape
    for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        a, b = truth[y0:y1, x0:x1], truth[y0+dy:y1+dy, x0+dx:x1+dx]
        transition = (a < 4) & (b < 4) & (a != b)
        boundary[y0:y1, x0:x1] |= transition
        boundary[y0+dy:y1+dy, x0+dx:x1+dx] |= transition
    distance = ndimage.distance_transform_edt(~boundary) if boundary.any() else np.full(truth.shape, np.inf)
    return {"boundary": (truth < 4) & (distance <= 7), "interior": (truth < 4) & (distance > 7)}


def populations(cache, truth):
    fg = (truth >= 0) & (truth < 4)
    raw, rect = cache["raw"], cache["rect"]
    groups = {"all": fg, "Corrected_by_CH": fg & (raw != truth) & (rect == truth),
              "Still_Wrong": fg & (raw != truth) & (rect != truth),
              "Harmed_by_CH": fg & (raw == truth) & (rect != truth),
              "Stable_Correct": fg & (raw == truth) & (rect == truth),
              "Top20": cache["top20"].astype(bool)}
    assert not (groups["Top20"] & ~fg).any()
    groups["Bottom80"] = fg & ~groups["Top20"]
    return groups


def binary_hist(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels, dtype=bool)
    idx = np.minimum((np.clip(scores, 0, 1) * BINS).astype(np.int64), BINS - 1)
    return np.stack([np.bincount(idx[labels], minlength=BINS), np.bincount(idx[~labels], minlength=BINS)])


def binary_metrics(hist):
    pos, neg = np.asarray(hist, dtype=np.float64)
    p, n = pos.sum(), neg.sum()
    auc = (pos * (np.cumsum(neg) - .5 * neg)).sum() / (p * n) if p and n else np.nan
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp > 0)
    ap = (precision * pos[::-1]).sum() / p if p else np.nan
    prevalence = p / (p+n) if p+n else np.nan
    return dict(auroc=auc, auprc=ap, prevalence=prevalence,
                auprc_over_prevalence=ap/prevalence if prevalence else np.nan,
                positive=int(p), negative=int(n))


def exact_binary_metrics(score, label):
    """Exact ties via sorting; independent of 4096-bin implementation."""
    order = np.argsort(score, kind="stable")
    s, y = np.asarray(score)[order], np.asarray(label, dtype=np.int64)[order]
    if len(s) == 0:
        return binary_metrics(np.zeros((2, 1)))
    starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    counts = np.diff(np.r_[starts, len(s)])
    pos = np.add.reduceat(y, starts)
    return binary_metrics(np.stack([pos, counts-pos]))


def confusion(truth, prediction, mask):
    return np.bincount(4 * truth[mask].astype(int) + prediction[mask].astype(int), minlength=16).reshape(4, 4)


def cm_metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    d = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-1) + cm.sum(-2)
    iou = np.divide(d, denom-d, out=np.full_like(d, np.nan), where=denom-d > 0)
    dice = np.divide(2*d, denom, out=np.full_like(d, np.nan), where=denom > 0)
    total = cm.sum(axis=(-2, -1))
    accuracy = np.divide(d.sum(-1), total, out=np.full_like(total, np.nan), where=total > 0)
    return dict(accuracy=accuracy, miou=np.nanmean(iou, axis=-1), dice=np.nanmean(dice, axis=-1), class_iou=iou)


def nanmean(x, axis=None):
    x = np.asarray(x, dtype=float)
    count = np.isfinite(x).sum(axis=axis)
    return np.divide(np.nansum(x, axis=axis), count, out=np.full(np.shape(count), np.nan), where=count > 0)


def bootstrap_means(values, resamples=10000, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    result = []
    for start in range(0, resamples, 50):
        indices = rng.integers(0, len(values), (min(50, resamples-start), len(values)), dtype=np.int32)
        result.append(nanmean(values[indices], axis=1))
    return np.concatenate(result)


def ci_row(name, observed, values):
    finite = np.asarray(values)[np.isfinite(values)]
    return dict(metric=name, observed=float(observed), ci95_low=float(np.quantile(finite, .025)) if len(finite) else np.nan,
                ci95_high=float(np.quantile(finite, .975)) if len(finite) else np.nan,
                finite_resamples=len(finite), resamples=10000, seed=42, unit="fraction")
