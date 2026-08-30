"""Frozen pre-rectification guidance and audit-only estimands. GT never enters loss."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from tools.rddr_phase2b16_common import EPS, Q_EDGES, js, loss_probe, scores, margin_direction

PREFIX='rddr_phase2b18_'
GRAD_SHA='5ab5a048e932f27726cea7507685354f984432fb66b542b87b805cea24a72bd5'
MODES=('Uraw','FAraw','PRG')
UPSTREAM=('b4','b4_1','b4_2','b4_3','b4_4','b4_5','bn45')


def upstream_name(name):
    return name.split('.')[0] in UPSTREAM


def student_logits(raw,head,shared=False):
    return F.conv2d(raw,head.weight if shared else head.weight.detach(),
                    head.bias if shared else head.bias.detach())


def guidance_loss(logits,teacher,fixed,q,mode='PRG'):
    assert mode in MODES
    return loss_probe(logits,fixed if mode=='FAraw' else teacher,q,'U' if mode=='Uraw' else 'CCA')[0]


def conflict_gradient(logits,deep):
    """Separate graph. Sum of pointwise q gives its block-diagonal per-pixel derivative."""
    leaf=logits.detach().float().requires_grad_(True)
    q=js(leaf.softmax(1),deep.detach().float())/math.log(2)
    g,=torch.autograd.grad(q.sum(),leaf)
    return q.detach(),g.detach()


def hierarchy(gq,gradient):
    a=np.asarray(gq,dtype=np.float64);v=-np.asarray(gradient,dtype=np.float64)
    dq=(a*v).sum(1)
    cos=dq/(np.linalg.norm(a,axis=1)*np.linalg.norm(v,axis=1)+EPS)
    return dq,cos


def groups(data):
    y=data['truth'];fg=y<4;s=data['ps'].argmax(1);d=data['pd'].argmax(1)
    sc=s==y;dc=d==y
    g={'all':fg,'Top20':fg&data['top20'].astype(bool),'Bottom80':fg&~data['top20'].astype(bool)}
    bins=np.searchsorted(Q_EDGES,data['q_feature'],side='left')
    g.update({f'Q{k+1}':fg&(bins==k) for k in range(5)})
    g.update(Raw_Correct=fg&sc,Raw_Wrong=fg&~sc)
    g.update({'Deep-Win':fg&~sc&dc,'Shallow-Win':fg&sc&~dc,'Both-Wrong':fg&~sc&~dc,'Stable-Correct':fg&sc&dc})
    g.update({f'class{k}':fg&(y==k) for k in range(4)})
    g.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    return g


def semantic_metrics(prob,truth):
    m=truth<4;y=truth[m].astype(np.int64);pred=prob.argmax(1)[m]
    cm=np.bincount(y*4+pred,minlength=16).reshape(4,4)
    sc=scores(cm);p=prob.transpose(0,2,1)[m].astype(np.float64)
    return dict(accuracy=float(sc['accuracy']),miou=float(sc['miou']),dice=float(sc['dice']),
                nll=float(-np.log(p[np.arange(len(y)),y]+EPS).mean()),
                brier=float(((p-np.eye(4)[y])**2).sum(1).mean()),
                **{f'iou_class{k}':float(sc['class_iou'][k]) for k in range(4)})


def decide(a,b,c,d,e):
    if not e:return 'PRERECT_GUIDANCE_ENGINEERING_NOGO'
    if not a:return 'SYMMETRIC_TEACHER_NOT_SUITABLE_FOR_RAW'
    if not b or not c:return 'TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE'
    if not d:return 'PRERECT_GUIDANCE_HIERARCHY_COLLAPSE_RISK'
    return 'RDDR_PHASE2B18_PRERECT_GUIDANCE_GO'
