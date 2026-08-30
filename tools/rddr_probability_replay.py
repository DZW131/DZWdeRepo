"""Frozen FP32 probability replay from Phase110; explicitly prohibits graph operations."""
import math
import time
import numpy as np
from unittest.mock import patch
EPS=1e-8

def forbidden(*a,**k):raise RuntimeError('model/optimizer/backward/checkpoint operation forbidden')

def replay_probabilities(data,old,raw_logits):
    """Same FP32 reduction order as Phase2B15; probability operations only."""
    import torch
    import torch.nn.functional as F
    def js(p,q):
        m=.5*(p+q)
        return .5*((p*((p+EPS).log()-(m+EPS).log())).sum(1)+(q*((q+EPS).log()-(m+EPS).log())).sum(1))
    def support(p,d):
        b,c,h,w=p.shape;nei=F.unfold(p,15,padding=7).reshape(b,c,225,h*w)
        valid=F.unfold(torch.ones_like(p[:,:1]),15,padding=7).reshape(b,225,h*w).bool();valid[:,112]=False;count=valid.sum(1)
        s=(1-js(p.flatten(2)[:,:,None],nei)/math.log(2)).clamp(0,1);z=(1-js(d.flatten(2)[:,:,None],nei)/math.log(2)).clamp(0,1)
        return (s*valid).sum(1)/count,(z*valid).sum(1)/count,(nei*valid[:,None]).sum(2)/count[:,None]
    errors={k:0. for k in ('T_SS','T_SD','T_DS','T_DD','ctx_S','ctx_D','ctx_sym','q','raw_probability')}
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),patch.object(torch.nn.Module,'__init__',forbidden),patch.object(torch.optim.Optimizer,'__init__',forbidden),patch.object(torch.Tensor,'backward',forbidden),patch.object(torch,'save',forbidden),patch.object(torch.autograd,'grad',forbidden):
        for i in range(len(data['names'])):
            p=torch.from_numpy(data['ps'][i].reshape(1,4,28,28)).cuda();d=torch.from_numpy(data['pd'][i].reshape(1,4,28,28)).cuda()
            rawp=torch.from_numpy(raw_logits[i].reshape(1,4,28,28)).cuda().softmax(1)
            errors['raw_probability']=max(errors['raw_probability'],float((rawp-p).abs().max()))
            ss,ds,cs=support(p,d);dd,sd,cd=support(d,p)
            vals=dict(T_SS=ss,T_SD=sd,T_DS=ds,T_DD=dd,ctx_S=cs,ctx_D=cd,ctx_sym=.5*(cs+cd),q=js(p,d).flatten(1)/math.log(2))
            for k,a in vals.items():
                b=data['q_feature'][i] if k=='q' else old[k][i];errors[k]=max(errors[k],float(np.abs(a.cpu().numpy()[0]-b).max()))
                assert torch.isfinite(a).all() and not a.requires_grad
    assert errors['raw_probability']<=1e-7 and errors['q']<=1e-7 and all(v==0 for k,v in errors.items() if k not in ('q','raw_probability')),errors
    torch.cuda.synchronize()
    return dict(errors=errors,seconds=time.perf_counter()-start,allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),model_instantiated=False,network_forward=False,backward=False)
