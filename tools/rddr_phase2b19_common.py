"""Frozen GT-blind directional-transfer loss; audit utilities only."""
import numpy as np
import torch
from tools.rddr_phase2b16_common import EPS, two_support, js
from tools.rddr_phase2b18_common import student_logits, upstream_name, UPSTREAM, conflict_gradient, groups, hierarchy

PREFIX='rddr_phase2b19_'
PREVIOUS_SHA='740b2e80c9182e701509f5ed7a6fab4c8f1b6ddcd585df418ccefa9f288d8e52'
MODES=('UDT','RG','ADT','SDT')

def random_gate(delta):
    gate=np.asarray(delta)>0
    assert gate.ndim==2 and gate.shape[1]==784
    out=np.zeros_like(gate);rng=np.random.default_rng(42)
    for i,m in enumerate(gate):out[i,rng.choice(784,int(m.sum()),replace=False)]=True
    return out

@torch.no_grad()
def adjudicate(ps,pd):
    ps=ps.detach().float();pd=pd.detach().float()
    tss,tds=two_support(ps,pd);tdd,tsd=two_support(pd,ps)
    delta=.5*(tds+tdd)-.5*(tss+tsd)
    return dict(T_SS=tss,T_SD=tsd,T_DS=tds,T_DD=tdd,sym=delta,
                q=js(ps,pd).flatten(1)/np.log(2))

def direction_gate(delta):return (delta.detach()>0).detach()

def directional_loss(logits,deep,q,delta,mode,random=None):
    assert mode in MODES
    p=logits.float().softmax(1);d=deep.detach().float().reshape_as(p)
    shape=(p.shape[0],*p.shape[2:]);q=q.detach().float().reshape(shape);delta=delta.detach().reshape(shape)
    gate=torch.ones_like(q) if mode=='UDT' else random.detach().reshape(shape) if mode=='RG' else direction_gate(delta) if mode=='ADT' else delta.relu()
    w=q*gate;kl=(d*((d+EPS).log()-(p+EPS).log())).sum(1)
    return (w*kl).sum()/(w.sum()+EPS)

def decide(a,b,c,d,e,f,g):
    if not g:return 'DIRECTIONAL_TRANSFER_ENGINEERING_NOGO'
    if not a:return 'SYMMETRIC_ADJUDICATION_REPRODUCTION_FAIL'
    if not all((b,c,d,e)):return 'ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE'
    if not f:return 'DIRECTIONAL_TRANSFER_NOT_BETTER_THAN_RANDOM_SELECTION'
    return 'RDDR_PHASE2B19_DIRECTIONAL_TRANSFER_GO'
