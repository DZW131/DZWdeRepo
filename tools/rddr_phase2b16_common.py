"""GT-blind, detached training-time-only loss helpers and frozen audit utilities."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PREFIX = 'rddr_phase2b16_'
A0 = '4e9a2887b220d17e27649d72a3d13f32b7ebe8f9'
CKPT_SHA = '509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579'
NATIVE_SHA = '767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a'
DERIVED_SHA = '237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514'
EPS = 1e-8
Q_EDGES = np.array([.020935675129294395,.072734534740448,.163648784160614,.3369627296924591])
PARAMS = ('hfrm_28_1.context_conv.weight','hfrm_28_1.veto_mlp.0.weight',
          'hfrm_28_1.veto_mlp.2.weight','hfrm_28_1.gamma_context',
          'hfrm_28_1.gamma_veto','ic1.weight','ic1.bias')
PATHS = {'context':(0,3),'semantic':(1,2,4),'head':(5,6)}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,Path):return str(x)
    return x


def write_json(path,obj):
    Path(path).write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def js(p,q,dim=1):
    # Exact Phase0/Phase2B1 operator order (epsilon inside each logarithm).
    m=.5*(p+q)
    return .5*((p*((p+EPS).log()-(m+EPS).log())).sum(dim)
               +(q*((q+EPS).log()-(m+EPS).log())).sum(dim))


@torch.no_grad()
def two_support(source,target):
    b,c,h,w=source.shape
    assert (c,h,w)==(4,28,28) and target.shape==source.shape
    nei=F.unfold(source,15,padding=7).reshape(b,c,225,h*w)
    valid=F.unfold(torch.ones_like(source[:,:1]),15,padding=7).reshape(b,225,h*w).bool()
    valid[:,112]=False
    a=(1-js(source.flatten(2)[:,:,None],nei)/math.log(2)).clamp(0,1)
    z=(1-js(target.flatten(2)[:,:,None],nei)/math.log(2)).clamp(0,1)
    return (a*valid).sum(1)/valid.sum(1),(z*valid).sum(1)/valid.sum(1)


@torch.no_grad()
def detached_teacher(ps,pd):
    """No GT/mask/q/context source input; strictly two-hypothesis teacher."""
    ps,pd=ps.detach().float(),pd.detach().float()
    tss,tds=two_support(ps,pd); tdd,tsd=two_support(pd,ps)
    ss=.5*(tss+tsd); sd=.5*(tds+tdd)
    wd=sd/(ss+sd+EPS)
    teacher=(1-wd[:,None])*ps.flatten(2)+wd[:,None]*pd.flatten(2)
    return dict(T_SS=tss,T_SD=tsd,T_DS=tds,T_DD=tdd,sym=sd-ss,wD_sym=wd,
                anchor_sym=teacher,q=js(ps,pd).flatten(1)/math.log(2))


def loss_probe(logits,teacher,q,mode='CCA'):
    assert mode in ('U','FA','CCA')
    logits=logits.float()
    t=teacher.detach().float().reshape_as(logits)
    weight=q.detach().float().reshape(logits.shape[0],*logits.shape[2:])
    p=logits.softmax(1)
    kl=(t*((t+EPS).log()-(p+EPS).log())).sum(1)
    loss=kl.mean() if mode=='U' else (weight*kl).sum()/(weight.sum()+EPS)
    return loss,kl


def margin_direction(logits,gradient,truth):
    """GT audit only. Directional max uses all CURRENT tied competitors."""
    l,g=np.asarray(logits),np.asarray(gradient)
    y=np.asarray(truth).clip(0,3)
    true_mask=np.eye(4,dtype=bool)[y].transpose(0,2,1)
    competitors=np.where(true_mask,-np.inf,l)
    tied=competitors==competitors.max(1,keepdims=True)
    move=-g
    dgt=np.take_along_axis(move,y[:,None],axis=1)[:,0]
    dcomp=np.where(tied,move,-np.inf).max(1)
    return dgt-dcomp,tied.sum(1)>1


def strata(data,rect):
    y=data['truth']; fg=y<4; q=data['q_feature']
    top=fg&data['top20'].astype(bool)
    masks={'all':fg,'Top20':top,'Bottom80':fg&~top}
    quint=np.searchsorted(Q_EDGES,q,side='left')
    masks.update({f'Q{k+1}':fg&(quint==k) for k in range(5)})
    masks.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    masks.update({f'class{k}':fg&(y==k) for k in range(4)})
    dw=data['pd'].argmax(1)!=y; sw=data['ps'].argmax(1)!=y
    masks.update({'Deep-Wrong':fg&dw,'Shallow-Wrong':fg&sw,'Both-Wrong':fg&dw&sw,
                  'Rect_Correct':fg&(rect==y),'Rect_Wrong':fg&(rect!=y)})
    return masks


def scores(cm):
    cm=np.asarray(cm,dtype=np.float64)
    d=np.diagonal(cm,axis1=-2,axis2=-1)
    denom=cm.sum(-1)+cm.sum(-2)
    iou=np.divide(d,denom-d,out=np.full_like(d,np.nan),where=(denom-d)>0)
    dice=np.divide(2*d,denom,out=np.full_like(d,np.nan),where=denom>0)
    return dict(accuracy=d.sum(-1)/cm.sum(axis=(-2,-1)),miou=np.nanmean(iou,axis=-1),
                dice=np.nanmean(dice,axis=-1),class_iou=iou,class_dice=dice)


def bootstrap_indices(n,count=10000,seed=42):
    rng=np.random.default_rng(seed)
    for start in range(0,count,50):
        yield rng.integers(0,n,(min(50,count-start),n),dtype=np.int32)


def decision(a,b,c,d):
    if not d:return 'CCA_INTEGRATION_ENGINEERING_NOGO'
    if not a:return 'SYMMETRIC_TEACHER_NOT_SUPERIOR'
    if not b:return 'CONFLICT_WEIGHTING_LOCALIZATION_NOT_SUPPORTED'
    if c=='FAIL':return 'TEACHER_SIGNAL_PRESENT_GRADIENT_UNSAFE'
    if c=='UNDERPOWERED':return 'TRAINABILITY_GO_CLASS_SAFETY_UNDERPOWERED'
    return 'RDDR_PHASE2B16_READY_FOR_TRAINING'
