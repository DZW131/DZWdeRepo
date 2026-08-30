"""Frozen acceptance equations. No GT in construction or loss."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from tools.rddr_phase2b16_common import EPS,js,loss_probe,margin_direction,Q_EDGES

PREFIX='rddr_phase2b17_'
GRAD_SHA='5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5'
MODES=('U','CCA','HA','SA')
HFRM_GROUPS=('Corrected_by_CH','Still_Wrong','Harmed_by_CH','Stable_Correct')


@torch.no_grad()
def acceptance_support(ps,pd,rect,teacher):
    ps,pd,rect,teacher=[v.detach().float() for v in (ps,pd,rect,teacher)]
    assert ps.shape==pd.shape==rect.shape==teacher.shape and ps.shape[1:]==(4,28,28)
    b,c,h,w=ps.shape
    valid=F.unfold(torch.ones_like(ps[:,:1]),15,padding=7).reshape(b,225,h*w).bool()
    valid[:,112]=False
    result={}
    for source,suffix in ((ps,'S'),(pd,'D')):
        nei=F.unfold(source,15,padding=7).reshape(b,c,225,h*w)
        for target,prefix in ((rect,'R'),(teacher,'T')):
            sim=1-js(target.flatten(2)[:,:,None],nei)/math.log(2)
            result[prefix+'_'+suffix]=(sim*valid).sum(1)/valid.sum(1)
    result['S_R']=.5*(result['R_S']+result['R_D'])
    result['S_T']=.5*(result['T_S']+result['T_D'])
    result['delta']=result['S_T']-result['S_R']
    assert all(torch.isfinite(v).all() for v in result.values())
    return result


def acceptance_loss(logits,teacher,q,delta,mode):
    assert mode in MODES
    q=q.detach().float();delta=delta.detach().float().reshape_as(q)
    if mode=='HA':weight=q*(delta>0).detach()
    elif mode=='SA':weight=q*delta.relu().detach()
    else:weight=q
    return loss_probe(logits,teacher.detach(),weight,'U' if mode=='U' else 'CCA')


def groups(data,rect,teacher):
    y=data['truth'];fg=y<4
    tw=fg&(teacher==y)&(rect!=y);rw=fg&(rect==y)&(teacher!=y)
    masks={'all':fg,'Top20':fg&data['top20'].astype(bool),'Bottom80':fg&~data['top20'].astype(bool)}
    bins=np.searchsorted(Q_EDGES,data['q_feature'],side='left')
    masks.update({f'Q{k+1}':fg&(bins==k) for k in range(5)})
    masks.update({'Rect_Correct':fg&(rect==y),'Rect_Wrong':fg&(rect!=y),'Teacher-Win':tw,'Rect-Win':rw})
    masks.update({f'class{k}':fg&(y==k) for k in range(4)})
    masks.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    masks.update({name:fg&(data['hfrm']==k) for k,name in enumerate(HFRM_GROUPS)})
    return masks,tw,rw


def rank_metrics(score,label):
    score=np.asarray(score);label=np.asarray(label,dtype=np.int64)
    p=int(label.sum());n=len(label)-p
    if len(score)==0:return dict(positive=0,negative=0,prevalence=np.nan,auroc=np.nan,auprc=np.nan)
    order=np.argsort(score,kind='stable');s=score[order];y=label[order]
    start=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    tp=np.add.reduceat(y,start).astype(float);fp=np.diff(np.r_[start,len(y)])-tp
    auc=(tp*(np.cumsum(fp)-.5*fp)).sum()/(p*n) if p and n else np.nan
    ap=(np.cumsum(tp[::-1])/(np.cumsum(tp[::-1])+np.cumsum(fp[::-1]))*tp[::-1]).sum()/p if p else np.nan
    return dict(positive=p,negative=n,prevalence=p/len(y),auroc=auc,auprc=ap)


def sign_metrics(tp,fn,fp,tn):
    tp,fn,fp,tn=[np.asarray(v,dtype=float) for v in (tp,fn,fp,tn)]
    def div(x,y):return np.divide(x,y,out=np.full(np.broadcast_shapes(x.shape,y.shape),np.nan),where=y>0)
    pr=div(tp,tp+fn);nr=div(tn,tn+fp)
    return dict(balanced_accuracy=.5*(pr+nr),macro_f1=.5*(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn)),
                teacher_win_recall=pr,rect_win_recall=nr,correction_precision=div(tp,tp+fp))


def decide(a,b,c,d,engineering=True):
    if not engineering:return 'ACCEPTANCE_AUDIT_ENGINEERING_NOGO'
    if not a:return 'CONTEXTUAL_ACCEPTANCE_NOT_SUPPORTED'
    if not b:return 'WINNER_ACCEPTANCE_NOT_GRADIENT_ACCEPTANCE'
    if c=='FAIL':return 'ACCEPTANCE_SIGNAL_EXISTS_CONSUMPTION_UNSAFE'
    if not d:return 'ACCEPTANCE_PROTECTION_CAPACITY_FAIL'
    if c=='UNDERPOWERED':return 'ACCEPTANCE_SIGNAL_CLASS_SAFETY_UNDERPOWERED'
    return 'RDDR_PHASE2B17_ACCEPTANCE_GO'
