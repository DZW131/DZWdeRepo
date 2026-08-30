"""GT-blind frozen probes and separate audit-only definitions for Phase-2B1.5."""
import math
import numpy as np
import torch
from tools.rddr_phase2b1_common import (EPS, Q_EDGES, HFRM_GROUPS, compute_support,
    neighbors, phase0_js, divide)

CACHE_SHA = "767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a"
PREFIX = "rddr_phase2b15_"
SCORES = ("old", "dsrc", "sym", "ctx_S", "ctx_D", "ctx_sym")
ESTIMATORS = ("shallow", "deep", "fixed_average", "anchor_old", "anchor_sym", "ctx_S", "ctx_D", "ctx_sym")


@torch.no_grad()
def compute_four_way_support_matrix(ps, pd):
    """Two calls to the frozen kernel preserve its exact reduction order."""
    from_s = compute_support(ps, pd)
    from_d = compute_support(pd, ps)
    return dict(T_SS=from_s["ss"], T_DS=from_s["sd"], T_SD=from_d["sd"],
                T_DD=from_d["ss"], ctx_S=from_s["ctx"], ctx_D=from_d["ctx"])


def compute_same_family_bias(t):
    bs, bd = t["T_SS"]-t["T_SD"], t["T_DD"]-t["T_DS"]
    return dict(B_S=bs, B_D=bd, B_family=.5*(bs+bd))


def compute_delta_symmetric(t):
    ss, sd = .5*(t["T_SS"]+t["T_SD"]), .5*(t["T_DS"]+t["T_DD"])
    return ss, sd, sd-ss


@torch.no_grad()
def probes(ps, pd):
    """Only cached probabilities enter the deployable score probes."""
    ps, pd = ps.detach().float(), pd.detach().float()
    t = compute_four_way_support_matrix(ps, pd)
    t.update(compute_same_family_bias(t))
    ss, sd, delta = compute_delta_symmetric(t)
    wd = sd/(ss+sd+EPS)
    t.update(old=t["T_DS"]-t["T_SS"], dsrc=t["T_DD"]-t["T_SD"], sym=delta,
             wD_sym=wd, anchor_sym=(1-wd[:,None])*ps.flatten(2)+wd[:,None]*pd.flatten(2),
             ctx_sym=.5*(t["ctx_S"]+t["ctx_D"]))
    for name in ("ctx_S", "ctx_D", "ctx_sym"):
        t["delta_"+name] = phase0_js(t[name], ps.flatten(2))-phase0_js(t[name], pd.flatten(2))
    assert all(torch.isfinite(v).all() and not v.requires_grad for v in t.values())
    return t


@torch.no_grad()
def gt_context_diagnostic(truth, shallow, deep):
    """Audit only: GT is never accepted by probes() or any support function."""
    assert truth.shape[1:] == (28,28)
    assert torch.isin(truth, torch.tensor([0,1,2,3,4,255], device=truth.device)).all()
    valid = neighbors(torch.ones_like(truth[:,None],dtype=torch.float32))[:,0].bool()
    valid[:,112] = False
    count = valid.sum(1)
    channels = torch.stack([truth==k for k in (0,1,2,3,4,255)],1).float()
    hist = (neighbors(channels)*valid[:,None]).sum(2)/count[:,None]
    ys = torch.where(truth==255,5,truth).flatten(1).long()
    s,d = shallow.flatten(1).long(),deep.flatten(1).long()
    gather=lambda idx:hist.gather(1,idx[:,None])[:,0]
    sm,dm = gather(s),gather(d)
    other=hist[:,:4].sum(1)-sm-dm+torch.where(s==d,sm,0.)
    return dict(GT_same_fraction=gather(ys), GT_shallow_candidate_fraction=sm,
                GT_deep_candidate_fraction=dm, GT_other_fraction=other,
                GT_background_fraction=hist[:,4], GT_ignore_fraction=hist[:,5])


def make_groups(data):
    y=data["truth"]; fg=y<4
    s,d=data["ps"].argmax(1),data["pd"].argmax(1)
    sc,dc=s==y,d==y; top=data["top20"].astype(bool)&fg
    win=fg&(sc^dc); hard=fg&(s!=d)
    g=dict(all=fg,hard_disagreement=hard,adjudication=win,Deep_Win=win&dc,
           Shallow_Win=win&sc,Both_Wrong=fg&~sc&~dc,Both_Correct=fg&sc&dc,
           Top20=top,Bottom80=fg&~top,Top20_Both_Wrong=top&~sc&~dc)
    for name,m in (("Deep_Correct",dc),("Deep_Wrong",~dc),("Shallow_Correct",sc),("Shallow_Wrong",~sc)):
        g[name]=fg&m; g["Top20_"+name]=top&m
    for k in range(4): g[f"class{k}"]=y==k
    g["boundary"]=fg&data["boundary"].astype(bool); g["interior"]=fg&~data["boundary"].astype(bool)
    bins=np.searchsorted(Q_EDGES,data["q_feature"],side="left")
    for k in range(5): g[f"Q{k+1}"]=fg&(bins==k)
    for k,name in enumerate(HFRM_GROUPS): g[name]=fg&(data["hfrm"]==k)
    for a in range(4):
        for b in range(4):
            if a!=b:g[f"pair{a}_{b}"]=hard&(s==a)&(d==b)
    return g,win,dc,s,d


def third_class_metrics(truth, shallow, deep, context_pred, mask):
    different=(context_pred!=shallow)&(context_pred!=deep)&mask
    rescue=different&(context_pred==truth)
    harm=different&(context_pred!=truth)
    return dict(targets=int(mask.sum()),different_from_both=int(different.sum()),
                correct_third_class=int(rescue.sum()),wrong_third_class=int(harm.sum()),
                rescue_rate=float(divide(rescue.sum(),mask.sum())),
                rescue_precision=float(divide(rescue.sum(),different.sum())),
                intrusion_rate=float(divide(different.sum(),mask.sum())),
                harm_rate=float(divide(harm.sum(),mask.sum())))


def class_status(positive, negative, auc):
    if min(positive,negative)<500: return "UNDERPOWERED"
    return "PASS" if np.isfinite(auc) and auc>=.45 else "FAIL"


def aggregate_class_status(statuses):
    if "FAIL" in statuses: return "FAIL"
    return "UNDERPOWERED" if "UNDERPOWERED" in statuses else "PASS"


def decide(a,b,c,d):
    if not a: return "SAME_FAMILY_BIAS_HYPOTHESIS_NOT_SUPPORTED"
    if not b: return "THIRD_EVIDENCE_REQUIRED_FOR_NEXT_DESIGN" if d else "ADJUDICATION_BIAS_UNRESOLVED"
    if c=="FAIL": return "ADJUDICATION_BIAS_UNRESOLVED"
    if c=="UNDERPOWERED": return "SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED"
    return "SYMMETRIC_ADJUDICATION_BIAS_RESOLVED"
