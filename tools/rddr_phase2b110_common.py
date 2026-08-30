"""Frozen cache-only score construction and explicitly GT-only audit estimands."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

P='rddr_phase2b110_'
A0='4e9a2887b220d17e27649d72a3d13f32b7ebe8f9'
EPS=1e-8
HASHES=dict(native='767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a',
 derived='237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514',
 observations='d4f65c519920c010e307ba8f32fb8e110387e0e14db73baa7c43163072ad0f1a',
 checkpoint='509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579',
 previous_runtime='bb54ae356e1258baabd8795894c51d12d5532137ac352cf1d6d0c4c88d3f48a0',
 previous_summary='84e009170aa335c0f625afdc097f86369e709ea004903881d10b9c264c7a0eb7',
 previous_identity='4adffe179ac328db9ce922c7c7ab3f18de759c1cb0cff6bbd09631cde3cd6637')
SCORES=('S_D_sym','Delta_sym','q','deep_confidence_advantage','deep_entropy_advantage')

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

def frozen_scores(ps,pd,q,tss,tsd,tds,tdd):
    """No labels or population masks. No learned model, ranking selection or gate."""
    ss=.5*(tss+tsd);sd=.5*(tds+tdd)
    a=ps.astype(float);b=pd.astype(float)
    entropy_s=-(a*np.log(a+EPS)).sum(1);entropy_d=-(b*np.log(b+EPS)).sum(1)
    return dict(zip(SCORES,(sd,sd-ss,q,pd.max(1)-ps.max(1),entropy_s-entropy_d)))

def margin_direction(logits,gradient,truth):
    y=truth.clip(0,3);l=np.asarray(logits);v=-np.asarray(gradient,dtype=float)
    mask=np.eye(4,dtype=bool)[y].transpose(0,2,1);other=np.where(mask,-np.inf,l);ties=other==other.max(1,keepdims=True)
    return np.take_along_axis(v,y[:,None],axis=1)[:,0]-np.where(ties,v,-np.inf).max(1)

def population(truth,ps,pd,gate,udt_dm):
    fg=truth<4;s=ps.argmax(1);d=pd.argmax(1);rw=fg&(s!=truth);res=fg&~gate
    rrw=res&rw;rdw=rrw&(d==truth);rbw=rrw&(d!=truth);rsw=res&(s==truth)&(d!=truth)
    return dict(foreground=fg,Raw_Wrong=rw,Residual=res,R_RW=rrw,Rejected_Deep_Win=rdw,Rejected_Both_Wrong=rbw,Rejected_Shallow_Win=rsw,
        Residual_Beneficial=rrw&(udt_dm>0),Residual_Harmful=rrw&(udt_dm<0),Residual_Zero=rrw&(udt_dm==0))

def required_gap(n,b):
    assert isinstance(n,(int,np.integer)) and isinstance(b,(int,np.integer))
    return (2*int(n)+4)//5-int(b)

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

def diagnostic_quintiles(values,mask):
    edges=np.quantile(values[mask],[.2,.4,.6,.8],method='linear')
    bins=np.searchsorted(edges,values,side='left')
    return edges,[mask&(bins==k) for k in range(5)]

def context_metrics(prob,truth,shallow,deep,mask):
    pred=prob.argmax(1);y=truth[mask].astype(int);cp=pred[mask];cm=np.bincount(y*4+cp,minlength=16).reshape(4,4).astype(float)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1);iou=divide(tp,den-tp);dice=divide(2*tp,den)
    pp=prob.transpose(0,2,1)[mask].astype(float);diff=mask&(pred!=shallow)&(pred!=deep);correct=mask&(pred==truth)
    rescue=diff&correct;harm=diff&~correct
    return dict(targets=int(mask.sum()),accuracy=float(divide(correct.sum(),mask.sum())),miou=mean(iou[np.isfinite(iou)]),dice=mean(dice[np.isfinite(dice)]),
        nll=mean(-np.log(pp[np.arange(len(y)),y]+EPS)),brier=mean(((pp-np.eye(4)[y])**2).sum(1)),
        different_from_both=int(diff.sum()),correct_third_class=int(rescue.sum()),wrong_third_class=int(harm.sum()),
        rescue_rate=float(divide(rescue.sum(),mask.sum())),rescue_precision=float(divide(rescue.sum(),diff.sum())),
        intrusion_rate=float(divide(diff.sum(),mask.sum())),third_harm_rate=float(divide(harm.sum(),mask.sum())),
        **{f'iou_class{k}':iou[k] for k in range(4)})

def class_power(positive,negative,images):return 'POWERED' if positive>=500 and negative>=500 and images>=30 else 'UNDERPOWERED'
def cross_stratum(interior_auc,classes):
    good=sum(r['power']=='POWERED' and r['image_auroc']>.55 for r in classes);missing=sum(r['power']=='UNDERPOWERED' for r in classes)
    if interior_auc>.60 and good>=3:return 'PASS'
    if interior_auc>.60 and good+missing>=3:return 'UNDERPOWERED'
    return 'FAIL'
def decide(a,b,c,d,third):
    if not a:return 'RESIDUAL_COVERAGE_HEADROOM_INSUFFICIENT'
    if b and c and d=='PASS':return 'DUAL_RESIDUAL_RECOVERY_SIGNAL_SUPPORTED' if third else 'RESIDUAL_DEEP_RECOVERY_SIGNAL_SUPPORTED'
    return 'RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED' if third else 'RESIDUAL_COVERAGE_NOT_RECOVERABLE_WITH_FROZEN_EVIDENCE'

def bootstrap_indices(n):
    rng=np.random.default_rng(42)
    for _ in range(200):yield rng.integers(0,n,(50,n),dtype=np.int32)
