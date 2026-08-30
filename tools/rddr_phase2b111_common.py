"""GT-blind candidates and epsilon-KL algebra. No model/autograd dependencies."""
import numpy as np
from tools.rddr_cache_metrics import *

P='rddr_phase2b111_'
A0='4e9a2887b220d17e27649d72a3d13f32b7ebe8f9'
EPS=1e-8
Q_EDGES=np.array([.020935675129294395,.072734534740448,.163648784160614,.3369627296924591])
HASHES=dict(native='767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a',
 derived='237268197426464ff4be2bb4761afddd1f1644eaaf66906e47439119d3c5d514',
 observations='d4f65c519920c010e307ba8f32fb8e110387e0e14db73baa7c43163072ad0f1a',
 checkpoint='509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579',
 previous_summary='a2137dc73dabf57dc8b1e0457138108127be8bbbe167d04bc8dc213753307850',
 previous_runtime='fbd9ccd0741e0e25dff010af2bd7f2a95636cbea002395322c6d86e05f840bdf',
 previous_identity='9a5e40333ed7431da9838b2b903bb2e69affb8c16b98dbd5b579504fd4c2f071',
 previous_verification='a8d908bc450940ba295a3800be59065619910a43b369c2499a76bbee7d0dddf9')
SCORES=('M_alt','C_ctx','E_ctx','q','Delta_sym','D_hier')

def candidate(ps,pd,ctx,md):
    cs=ps.argmax(1);cd=pd.argmax(1);cc=ctx.argmax(1)
    value=lambda c:np.take_along_axis(ctx,c[:,None],1)[:,0]
    margin=value(cc)-np.maximum(value(cs),value(cd))
    alt=(cc!=cs)&(cc!=cd);universe=~np.asarray(md,dtype=bool)
    return dict(cs=cs,cd=cd,cc=cc,U_R=universe,a_alt=alt,M_alt=margin,A_alt=universe&alt&(margin>0))

def scores(ps,pd,ctx,q,delta,c):
    t=np.asarray(ctx,dtype=float)
    return dict(zip(SCORES,(c['M_alt'],ctx.max(1),(t*np.log(t+EPS)).sum(1),q,delta,1-np.maximum(ps.max(1),pd.max(1)))))

def context_gradient(ps,ctx,active):
    """Per-pixel KL derivative, no reduction or q weight; inactive exactly zero."""
    p=ps.transpose(0,2,1)[active].astype(float);t=ctx.transpose(0,2,1)[active].astype(float)
    r=t*p/(p+EPS);gg=p*r.sum(1,keepdims=True)-r
    kl=np.zeros(active.shape,dtype=float);g=np.zeros(ps.shape,dtype=float)
    kl[active]=(t*(np.log(t+EPS)-np.log(p+EPS))).sum(1)
    g.transpose(0,2,1)[active]=gg
    return kl,g

def masks(truth,c,dm):
    fg=(truth>=0)&(truth<4);u=fg&c['U_R'];a=fg&c['A_alt'];sc=c['cs']==truth;dc=c['cd']==truth;cc=c['cc']==truth
    return dict(foreground=fg,U_R=u,candidate=a,RawWrong=fg&~sc,RawCorrect=u&sc,RawWrongResidual=u&~sc,
        BothWrong=u&~sc&~dc,DeepWin=u&~sc&dc,ShallowWin=u&sc&~dc,StableCorrect=u&sc&dc,
        ThirdRescue=a&cc,AlternativeFailure=a&~cc,beneficial=a&(dm>0),harmful=a&(dm<0),zero=a&(dm==0),
        Repair=a&~sc&cc,Harm=a&sc&~cc,WrongToWrong=a&~sc&~cc,StableCorrectActivated=a&sc&cc)

def utility(dm,m):
    x=dm[m]
    return dict(count=int(m.sum()),beneficial=int((x>0).sum()),harmful=int((x<0).sum()),zero=int((x==0).sum()),
        benefit_rate=float(divide((x>0).sum(),len(x))),harm_rate=float(divide((x<0).sum(),len(x))),zero_rate=float(divide((x==0).sum(),len(x))),
        mean_dm=mean(x),median_dm=float(np.median(x)) if len(x) else np.nan)

def cross_gate(interior_auc,classes):
    good=sum(r['power']=='POWERED' and r['image_auroc']>.55 for r in classes)
    missing=sum(r['power']=='UNDERPOWERED' for r in classes)
    if interior_auc>.60 and good>=3:return 'PASS'
    if (not np.isfinite(interior_auc) or interior_auc>.60) and good+missing>=3:return 'UNDERPOWERED'
    return 'FAIL'

def decide(a,b,c,d,e,f):
    prefix='THIRD_EVIDENCE_'
    if not a:return prefix+'OPERATIONAL_HEADROOM_INSUFFICIENT'
    if not b or not c:return prefix+'EXISTS_BUT_NOT_SELECTABLE'
    if not d:return prefix+'HARD_RESCUE_BUT_GRADIENT_UNSAFE'
    if not e:return prefix+'SIGNAL_WITH_PROTECTION_FAILURE'
    if f=='UNDERPOWERED':return None
    if f!='PASS':return prefix+'SIGNAL_NOT_ROBUST'
    return prefix+'GTBLIND_FEASIBILITY_SUPPORTED'
