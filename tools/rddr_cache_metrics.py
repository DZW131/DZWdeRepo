"""Frozen scalar/ranking utilities reused from verified Phase2B1.10; no model code."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,np.bool_):return bool(x)
    if isinstance(x,Path):return str(x)
    return x

def write_json(path,x):Path(path).write_text(json.dumps(clean(x),indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def divide(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)

def mean(x):return float(np.mean(x)) if len(x) else np.nan

def margin_direction(logits,gradient,truth):
    y=truth.clip(0,3);l=np.asarray(logits);v=-np.asarray(gradient,dtype=float)
    mask=np.eye(4,dtype=bool)[y].transpose(0,2,1);other=np.where(mask,-np.inf,l);ties=other==other.max(1,keepdims=True)
    return np.take_along_axis(v,y[:,None],axis=1)[:,0]-np.where(ties,v,-np.inf).max(1)

def binary(score,label):
    """Inherited Phase2B1 tied-score AUROC / non-interpolated AP."""
    label=np.asarray(label,dtype=np.int64);p=int(label.sum());n=len(label)-p
    if not len(label):return dict(auroc=np.nan,auprc=np.nan,positive=0,negative=0,prevalence=np.nan)
    order=np.argsort(score,kind='stable');s=np.asarray(score)[order];y=label[order]
    start=np.r_[0,np.flatnonzero(np.diff(s))+1];pos=np.add.reduceat(y,start).astype(float);neg=np.diff(np.r_[start,len(s)])-pos
    auc=(pos*(np.cumsum(neg)-.5*neg)).sum()/(p*n) if p and n else np.nan
    tp,fp=np.cumsum(pos[::-1]),np.cumsum(neg[::-1]);ap=(tp/(tp+fp)*pos[::-1]).sum()/p if p else np.nan
    return dict(auroc=auc,auprc=ap,positive=p,negative=n,prevalence=p/(p+n))

def ranking(score,label,eligible):
    pooled=binary(score[eligible],label[eligible]);ia=np.full(len(score),np.nan)
    for i in range(len(score)):
        m=eligible[i];r=binary(score[i,m],label[i,m]);ia[i]=r['auroc']
    return dict(pooled,image_auroc=mean(ia[np.isfinite(ia)]),eligible_images=int(np.isfinite(ia).sum()),images_with_targets=int(eligible.any(1).sum())),ia

def class_power(positive,negative,images):return 'POWERED' if positive>=500 and negative>=500 and images>=30 else 'UNDERPOWERED'

def bootstrap_indices(n):
    rng=np.random.default_rng(42)
    for _ in range(200):yield rng.integers(0,n,(50,n),dtype=np.int32)
