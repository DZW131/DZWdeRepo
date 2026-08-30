"""Frozen GT-blind dual-hypothesis equations and audit-only utility functions."""
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
Q_EDGES = np.array([.020935675129294395, .072734534740448, .163648784160614, .3369627296924591])
ESTIMATORS = ("shallow", "deep", "fixed_average", "anchor", "context_only")
HFRM_GROUPS = ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct")
SAFETY_BASES = ("all", "Top20", "Bottom80", "hard_disagreement", "Q1", "Q2", "Q3", "Q4", "Q5",
                "boundary", "interior", "class0", "class1", "class2", "class3")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def json_clean(x):
    if isinstance(x, dict): return {str(k): json_clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [json_clean(v) for v in x]
    if isinstance(x, np.ndarray): return json_clean(x.tolist())
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, x):
    Path(path).write_text(json.dumps(json_clean(x), indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader(); w.writerows(rows)


def phase0_js(p, q, dim=1):
    m = .5*(p+q)
    return .5*((p*((p+EPS).log()-(m+EPS).log())).sum(dim)
               +(q*((q+EPS).log()-(m+EPS).log())).sum(dim))


def neighbors(x):
    b,c,h,w = x.shape
    return F.unfold(x, 15, padding=7).reshape(b,c,225,h*w)


@torch.no_grad()
def compute_support(ps, pd):
    """No labels, masks, q, reliability or dataset access in this function."""
    assert ps.shape == pd.shape and ps.shape[1:] == (4,28,28)
    ps, pd = ps.float(), pd.float()
    source = neighbors(ps)
    valid = neighbors(torch.ones_like(ps[:,:1]))[:,0].bool()
    valid[:,112] = False
    c_s = (1-phase0_js(ps.flatten(2)[:,:,None], source)/math.log(2)).clamp(0,1)
    c_d = (1-phase0_js(pd.flatten(2)[:,:,None], source)/math.log(2)).clamp(0,1)
    count = valid.sum(1)
    ss = (c_s*valid).sum(1)/count
    sd = (c_d*valid).sum(1)/count
    delta = sd-ss
    wd = sd/(ss+sd+EPS)
    ws = 1-wd
    anchor = ws[:,None]*ps.flatten(2)+wd[:,None]*pd.flatten(2)
    ctx = (source*valid[:,None]).sum(2)/count[:,None]
    result = dict(ss=ss, sd=sd, delta=delta, wd=wd, ws=ws, anchor=anchor,
                  ctx=ctx, valid=valid, choose_deep=delta>0)
    assert all(torch.isfinite(v).all() for v in result.values())
    return result


def project(x):
    return F.interpolate(torch.as_tensor(np.array(x,copy=True))[None,None].float(), (28,28), mode="nearest")[0,0].numpy()


def populations(cache, truth):
    fg = (truth>=0)&(truth<4)
    raw, rect = cache["raw"], cache["rect"]
    masks = {"all":fg, "Corrected_by_CH":fg&(raw!=truth)&(rect==truth),
             "Still_Wrong":fg&(raw!=truth)&(rect!=truth), "Harmed_by_CH":fg&(raw==truth)&(rect!=truth),
             "Stable_Correct":fg&(raw==truth)&(rect==truth), "Top20":cache["top20"].astype(bool)}
    assert not (masks["Top20"]&~fg).any()
    masks["Bottom80"] = fg&~masks["Top20"]
    return masks


def boundary_mask(y):
    b = np.zeros_like(y,dtype=bool)
    h,w = y.shape
    for dy,dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        y0,y1 = max(0,-dy),min(h,h-dy)
        x0,x1 = max(0,-dx),min(w,w-dx)
        a,c = y[y0:y1,x0:x1],y[y0+dy:y1+dy,x0+dx:x1+dx]
        t = (a<4)&(c<4)&(a!=c)
        b[y0:y1,x0:x1] |= t
        b[y0+dy:y1+dy,x0+dx:x1+dx] |= t
    d = ndimage.distance_transform_edt(~b) if b.any() else np.full(y.shape,np.inf)
    return (y<4)&(d<=7)


def winner_labels(truth, shallow, deep):
    fg = (truth>=0)&(truth<4)
    sc, dc = shallow==truth, deep==truth
    eligible = fg&(shallow!=deep)&(sc^dc)
    return eligible, dc


def divide(a,b):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)


def nanmean(a,axis=None):
    a = np.asarray(a,dtype=float)
    return divide(np.nansum(a,axis=axis),np.isfinite(a).sum(axis=axis))


def binary_exact(score,label):
    score,label = np.asarray(score),np.asarray(label,dtype=np.int64)
    p,n = int(label.sum()),int(len(label)-label.sum())
    if len(score)==0:
        return dict(auroc=np.nan,auprc=np.nan,positive=0,negative=0,prevalence=np.nan)
    order = np.argsort(score,kind="stable")
    s,y = score[order],label[order]
    start = np.r_[0,np.flatnonzero(np.diff(s))+1]
    pos = np.add.reduceat(y,start).astype(float)
    neg = np.diff(np.r_[start,len(s)])-pos
    auc = (pos*(np.cumsum(neg)-.5*neg)).sum()/(p*n) if p and n else np.nan
    tp,fp = np.cumsum(pos[::-1]),np.cumsum(neg[::-1])
    ap = (tp/(tp+fp)*pos[::-1]).sum()/p if p else np.nan
    return dict(auroc=auc,auprc=ap,positive=p,negative=n,prevalence=p/(p+n))


def sign_scores(cm):
    cm = np.asarray(cm,dtype=float)
    d = np.diagonal(cm,axis1=-2,axis2=-1)
    recalls = divide(d,cm.sum(-1))
    denom = cm.sum(-1)+cm.sum(-2)
    return dict(accuracy=divide(d.sum(-1),cm.sum(axis=(-2,-1))),
                balanced_accuracy=(recalls[...,0]+recalls[...,1])/2,
                macro_f1=nanmean(divide(2*d,denom),axis=-1),
                deep_win_recall=recalls[...,1],shallow_win_recall=recalls[...,0])


def segmentation_scores(cm):
    cm = np.asarray(cm,dtype=float)
    d = np.diagonal(cm,axis1=-2,axis2=-1)
    denom = cm.sum(-1)+cm.sum(-2)
    iou = divide(d,denom-d)
    return dict(accuracy=divide(d.sum(-1),cm.sum(axis=(-2,-1))),miou=nanmean(iou,axis=-1),
                dice=nanmean(divide(2*d,denom),axis=-1),class_iou=iou)


def bootstrap_indices(n, count=10000, seed=42):
    rng = np.random.default_rng(seed)
    for start in range(0,count,50):
        yield rng.integers(0,n,(min(50,count-start),n),dtype=np.int32)


def decide(a,b,c,d):
    if not d: return "ADJUDICATION_DEEP_WRONG_UNSAFE"
    if not a or not b: return "RDDR_PHASE2B1_NOGO"
    if not c: return "ADJUDICATION_EXISTS_FUSION_UTILITY_FAIL"
    return "RDDR_PHASE2B1_GO"
