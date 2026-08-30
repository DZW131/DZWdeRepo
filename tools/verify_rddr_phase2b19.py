"""Independent verifier: no imports from primary audit/analysis modules."""
import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
P='rddr_phase2b19_';EPS=1e-8

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def auc(x,y):
    y=np.asarray(y,bool)
    if not y.any() or y.all():return np.nan
    ix=np.argsort(-x,kind='stable');xx=x[ix];yy=y[ix];ends=np.r_[np.flatnonzero(xx[1:]!=xx[:-1]),len(xx)-1]
    tp=np.r_[0,np.cumsum(yy)[ends]]/y.sum();fp=np.r_[0,np.cumsum(~yy)[ends]]/(~y).sum()
    return float(np.trapz(tp,fp))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--report',required=True);args=ap.parse_args()
    run=Path(args.run);report=Path(args.report)
    def jsn(folder,name):return json.loads((folder/(P+name+'.json')).read_text())
    def rows(name):
        with (report/(P+name+'.csv')).open(newline='') as f:return list(csv.DictReader(f))
    rt=jsn(run,'runtime');summary=jsn(report,'summary');obs=load(run/(P+'observations.npz'))
    data=load(rt['paths']['native']);old=load(rt['paths']['derived']);prev=load(rt['paths']['previous']);n=len(data['names']);checks={};errors={}
    checks['immutable_sources']=all(sha(rt['paths'][k])==v for k,v in rt['source_sha256'].items())
    checks['observation_hash']=sha(run/(P+'observations.npz'))==rt['observation_sha256']
    checks['full3418_order']=n==3418 and all(np.array_equal(data['names'],a['names']) for a in (old,prev,obs))
    delta=(old['T_DS']+old['T_DD'])*.5-(old['T_SS']+old['T_SD'])*.5;gate=delta>0
    checks['phase2b15_delta_exact']=np.array_equal(delta,old['sym']) and np.array_equal(gate,obs['direction_gate'])
    checks['phase2b18_raw_exact']=np.array_equal(obs['raw_logits'],prev['raw_logits'])
    checks['q_frozen_replay']=rt['parity']['q']<=1e-7 and np.array_equal(obs['q_gradients'],prev['q_gradients'])
    rng=np.random.default_rng(42);rand=np.zeros_like(gate)
    for i in range(n):rand[i,rng.choice(784,size=np.count_nonzero(gate[i]),replace=False)]=1
    checks['random_seed42_exact']=np.array_equal(rand,obs['random_gate']) and hashlib.sha256(rand.tobytes()).hexdigest()==rt['random_gate_sha256']
    checks['random_per_image_rate_match']=np.array_equal(rand.sum(1),gate.sum(1))
    gf32=gq32=gf64=gq64=loss_error=soft=0.
    for i in range(n):
        z=torch.tensor(obs['raw_logits'][i:i+1].reshape(1,4,28,28),device='cuda',requires_grad=True)
        d=torch.tensor(data['pd'][i:i+1].reshape(1,4,28,28),device='cuda');q=torch.tensor(data['q_feature'][i:i+1].reshape(1,28,28),device='cuda')
        dd=torch.tensor(delta[i:i+1].reshape(1,28,28),device='cuda');rr=torch.tensor(rand[i:i+1].reshape(1,28,28),device='cuda');p=z.softmax(1)
        soft=max(soft,float(np.abs(p.detach().flatten(2).cpu().numpy()[0]-data['ps'][i]).max()))
        for j,m in enumerate((torch.ones_like(q),rr,dd>0,dd.relu())):
            w=q*m;kl=(d*((d+EPS).log()-(p+EPS).log())).sum(1);loss=(w*kl).sum()/(w.sum()+EPS)
            g,=torch.autograd.grad(loss,z,retain_graph=True)
            gf32=max(gf32,float(np.abs(g.detach().flatten(2).cpu().numpy()[0]-obs['gradients'][i,j]).max()));loss_error=max(loss_error,abs(loss.item()-obs['losses'][i,j]))
            zz=z.detach().double().requires_grad_();pp=zz.softmax(1);target=d.double();ww=q.double()*m.double()
            k=(target*((target+EPS).log()-(pp+EPS).log())).sum(1);l=(ww*k).sum()/(ww.sum()+EPS);actual,=torch.autograd.grad(l,zz)
            a=target*pp/(pp+EPS);formula=(pp*a.sum(1,keepdim=True)-a)*(ww/(ww.sum()+EPS))[:,None]
            gf64=max(gf64,float((actual-formula).abs().max()))
        m=.5*(p+d);qq=.5*((p*((p+EPS).log()-(m+EPS).log())).sum(1)+(d*((d+EPS).log()-(m+EPS).log())).sum(1))/math.log(2)
        gq,=torch.autograd.grad(qq.sum(),z);gq32=max(gq32,float(np.abs(gq.detach().flatten(2).cpu().numpy()[0]-obs['q_gradients'][i]).max()))
        zz=z.detach().double().requires_grad_();pp=zz.softmax(1);tt=d.double();mm=.5*(pp+tt)
        jj=.5*((pp*((pp+EPS).log()-(mm+EPS).log())).sum(1)+(tt*((tt+EPS).log()-(mm+EPS).log())).sum(1))/math.log(2)
        a,=torch.autograd.grad(jj.sum(),zz);h=.5*((pp+EPS).log()-(mm+EPS).log()+pp/(pp+EPS)-.5*(pp+tt)/(mm+EPS))/math.log(2)
        gq64=max(gq64,float((a-pp*(h-(pp*h).sum(1,keepdim=True))).abs().max()))
    errors.update(FP32_loss=float(loss_error),FP32_gradient=gf32,FP32_q_gradient=gq32,FP64_KL_formula=gf64,FP64_JS_formula=gq64,raw_probability=soft)
    checks['four_losses_exact_FP32']=gf32==loss_error==0;checks['q_gradient_exact_FP32']=gq32==0
    checks['active_direction_analytic_FP64']=gf64<1e-12;checks['JS_analytic_FP64']=gq64<1e-12;checks['raw_probability_exact']=soft==0
    checks['all_finite']=all(np.isfinite(obs[k]).all() for k in ('gradients','q_gradients','losses','feature_sqsum','feature_maxabs','parameter_energy','parameter_maxabs'))
    y=data['truth'];fg=y<4;s=data['ps'].argmax(1);d=data['pd'].argmax(1);sc=s==y;dc=d==y;dw=fg&~sc&dc;sw=fg&sc&~dc;conf=dw|sw
    masks={'all':fg,'Top20':fg&data['top20'].astype(bool),'Bottom80':fg&~data['top20'].astype(bool)}
    bins=np.searchsorted([.020935675129294395,.072734534740448,.163648784160614,.3369627296924591],data['q_feature'],side='left')
    masks.update({f'Q{k+1}':fg&(bins==k) for k in range(5)});masks.update(Raw_Correct=fg&sc,Raw_Wrong=fg&~sc)
    masks.update({'Deep-Win':dw,'Shallow-Win':sw,'Both-Wrong':fg&~sc&~dc,'Stable-Correct':fg&sc&dc})
    masks.update({f'class{k}':fg&(y==k) for k in range(4)});masks.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    modes=('UDT','RG','ADT','SDT','PRG_previous');dm={};dq={};norm={};activity={};z=obs['raw_logits'].astype(float);gq=obs['q_gradients'].astype(float)
    for j,mode in enumerate(modes):
        v=-(obs['gradients'][:,j] if j<4 else prev['gradients'][:,2]).astype(float)
        other=np.where(np.arange(4)[None,:,None]==y[:,None],-np.inf,z);ties=other==other.max(1,keepdims=True)
        dm[mode]=np.take_along_axis(v,y.clip(0,3)[:,None],1)[:,0]-np.where(ties,v,-np.inf).max(1)
        dq[mode]=sum(v[:,k]*gq[:,k] for k in range(4));norm[mode]=np.sqrt(sum(v[:,k]**2 for k in range(4)))
        activity[mode]=rand if mode=='RG' else gate if mode in ('ADT','SDT') else np.ones_like(gate)
    checks['rejected_logit_feature_dQ_zero']=np.all(obs['gradients'][:,2].transpose(0,2,1)[~gate]==0) and np.all(obs['feature_sqsum'][~gate]==0) and np.all(dq['ADT'][~gate]==0) and np.all(dm['ADT'][~gate]==0)
    staterr=0.;utility={}
    for row in rows('gradient_controls'):
        mode=row['loss'];m=masks[row['stratum']];x=dm[mode][m];qv=dq[mode][m];gn=norm[mode][m]
        expected=dict(benefit_rate=(x>0).mean(),harm_rate=(x<0).mean(),zero_rate=(x==0).mean(),mean_dm=x.mean(),median_dm=np.median(x),
            active_gradient_fraction=(gn>0).mean(),mean_gradient_norm=gn.mean(),active_transfer_fraction=activity[mode][m].mean(),mean_dQ=qv.mean(),dQ_negative_fraction=(qv<0).mean(),dQ_zero_fraction=(qv==0).mean())
        staterr=max(staterr,max(abs(float(row[k])-v) for k,v in expected.items()));utility[mode,row['stratum']]=expected
    errors['strata_statistics']=staterr;checks['all_strata_margin_hierarchy']=staterr<1e-12
    adjerr=0.;imagauc={};refs={}
    for row in rows('adjudication_replay'):
        name=row['stratum'];m=masks[name];aa=m&dw;bb=m&sw;c=aa|bb
        ia=np.array([auc(delta[i,c[i]],aa[i,c[i]]) for i in range(n)]);imagauc[name]=ia
        capture=(aa&gate).sum()/aa.sum() if aa.any() else np.nan;protection=(bb&~gate).sum()/bb.sum() if bb.any() else np.nan
        expected=dict(image_auroc=np.nanmean(ia) if np.isfinite(ia).any() else np.nan,pooled_auroc=auc(delta[c],aa[c]),DeepCaptureRate=capture,ShallowProtectionRate=protection,BA=.5*(capture+protection),
            DeepSelectionPrecision=(aa&gate).sum()/(c&gate).sum() if (c&gate).any() else np.nan)
        for k,v in expected.items():
            got=float(row[k]);assert np.isnan(got)==np.isnan(v)
            if np.isfinite(v):adjerr=max(adjerr,abs(got-v))
        assert int(row['Deep_Win_count'])==aa.sum() and int(row['Shallow_Win_count'])==bb.sum() and int(row['dual_label_images'])==np.isfinite(ia).sum()
        assert row['power']==('POWERED' if aa.sum()>=500 and bb.sum()>=500 and np.isfinite(ia).sum()>=30 else 'UNDERPOWERED')
        refs[name]=expected
    errors['adjudication']=adjerr;checks['all_adjudication_and_power']=adjerr<1e-12
    quality=True
    for row in rows('selected_region_quality'):
        m=masks[row['stratum']]&(gate if row['selection']=='selected' else ~gate)
        r=(m&~sc&dc).sum();h=(m&sc&~dc).sum()
        quality &= int(row['repair'])==r and int(row['harm'])==h and int(row['net_repair'])==r-h
    checks['selected_rejected_quality']=bool(quality)
    recerr=0.
    for row in rows('brr_hhcr'):
        mode=row['loss'];m=masks[row['stratum']];on=activity[mode];a=m&dw;b=m&sw
        for key,v,den in [('DBR',a&on&(dm[mode]>0),a.sum()),('DCR',b&on&(dm[mode]<0),b.sum()),('BRR',a&on&(dm[mode]>0)&(dq[mode]<0),a.sum()),('HHCR',b&on&(dm[mode]<0)&(dq[mode]<0),b.sum())]:
            actual=v.sum()/den if den else np.nan;got=float(row[key]);assert np.isnan(actual)==np.isnan(got)
            if den:recerr=max(recerr,abs(got-actual))
    checks['BRR_HHCR_all_denominators']=recerr<1e-12
    fmask=dict(masks,active=fg&gate,rejected=fg&~gate,all784=np.ones_like(gate));fe=0.
    for row in rows('feature_gradient'):
        m=fmask[row['stratum']];en=obs['feature_sqsum'][m]
        for k,v in dict(rms=np.sqrt(en.sum()/(m.sum()*512)),mean_pixel_l2=np.sqrt(en).mean(),max_abs=obs['feature_maxabs'][m].max()).items():fe=max(fe,abs(float(row[k])-v))
    checks['feature_statistics']=fe<1e-12
    pe=obs['parameter_energy'];pm=obs['parameter_maxabs'];pn=obs['parameter_names'];num=obs['parameter_numel']
    checks['39_upstream_plus2_head']=len(pn)==41 and all(k.split('.')[0] in ('b4','b4_1','b4_2','b4_3','b4_4','b4_5','bn45') for k in pn[:-2])
    checks['frozen_head_zero']=np.all(pe[:,-2:]==0)
    checks['each_b4_conv_group_active']=all(pe[:,[j for j,k in enumerate(pn) if k.startswith(b+'.conv_')]].sum()>0 for b in ('b4','b4_1','b4_2','b4_3','b4_4','b4_5'))
    checks['parameter_energy_bound']=np.all(pe<=num[None]*pm**2+1e-12)
    terms={}
    def add(name,v,m):terms[name]=(np.where(m,v,0).sum(1,dtype=float),m.sum(1))
    terms['Delta_image_AUROC']=(np.nan_to_num(imagauc['all']),np.isfinite(imagauc['all']).astype(int))
    add('DeepCapture',gate,dw);add('ShallowProtection',~gate,sw);add('DeepSelectionPrecision',dw,conf&gate)
    for name in ('all','Top20','class0','class1','class2','class3'):add('ADT:'+name+':mean_dm',dm['ADT'],masks[name])
    add('ADT:Raw_Correct:harm_rate',dm['ADT']<0,masks['Raw_Correct']);add('ADT:Raw_Wrong:benefit_rate',dm['ADT']>0,masks['Raw_Wrong'])
    add('BRR_ADT',gate&(dm['ADT']>0)&(dq['ADT']<0),dw);add('HHCR_ADT',gate&(dm['ADT']<0)&(dq['ADT']<0),sw)
    add('ADT-RG:all:mean_dm',dm['ADT']-dm['RG'],fg);add('ADT-RG:Shallow-Win:harm_rate',(dm['ADT']<0).astype(float)-(dm['RG']<0),sw)
    add('ADT-RG:Deep-Win:benefit_rate',(dm['ADT']>0).astype(float)-(dm['RG']>0),dw)
    for name in ('all','Top20','Deep-Win','Shallow-Win'):
        add('ADT:'+name+':benefit_rate',dm['ADT']>0,masks[name]);add('ADT:'+name+':harm_rate',dm['ADT']<0,masks[name])
    keys=list(terms);ns=np.stack([terms[k][0] for k in keys],1);ds=np.stack([terms[k][1] for k in keys],1)
    rng=np.random.default_rng(42);rh=hashlib.sha256();reps=[]
    for _ in range(200):
        ix=rng.integers(0,n,(50,n),dtype=np.int32);rh.update(ix.tobytes());reps.append(ns[ix].sum(1)/ds[ix].sum(1))
    reps=np.concatenate(reps);existing=np.array([[float(r[k]) for k in keys] for r in rows('bootstrap_replicates')]);be=float(np.max(np.abs(reps-existing)))
    cis={r['metric']:r for r in rows('bootstrap')};ce=max(abs(float(cis[k][field])-np.quantile(reps[:,j],q)) for j,k in enumerate(keys) for field,q in [('ci_low',.025),('ci_high',.975)])
    errors.update(bootstrap_replicates=be,bootstrap_intervals=ce)
    checks['10000_paired_image_bootstrap']=be<1e-12 and ce<1e-12 and rh.hexdigest()==summary['bootstrap_rng_sha256']
    ident=jsn(run,'identity_audit');dt=jsn(run,'detach_audit');smoke=jsn(run,'bf16_smoke')
    checks['state_bn_checkpoint_identity']=ident['state_before']==ident['state_after'] and ident['bn_before']==ident['bn_after'] and ident['checkpoint_sha_before']==ident['checkpoint_sha_after']==rt['source_sha256']['checkpoint']
    checks['official_prediction_identity']=ident['prediction_before']==ident['prediction_after'] and ident['prediction_before']['images']==160
    checks['detach_no_forbidden_gradients']=all(dt[k] for k in ('q_detached','delta_detached','gate_detached','deep_source_detached','primary_ic1_none','hfrm_none','all_other_primary_gradients_none'))
    checks['BF16_batch20']=smoke['pass'] and smoke['all_finite'] and smoke['head_energy']==0 and smoke['upstream_conv_energy']>0 and smoke['reserved_bytes']<=22*1024**3
    checks['no_optimizer_test_luad']=not any(rt[k] for k in ('optimizer_created','optimizer_steps','checkpoint_written','test_access','luad_access','training_split_access'))
    checks['original_sources_unchanged']=not subprocess.check_output(['git','diff',rt['a0'],'--','network','tool','train_sshr.py'],cwd=Path(__file__).resolve().parents[1])
    pr={k:utility['ADT',k] for k in masks};ref=refs['all'];br=(gate&(dm['ADT']>0)&(dq['ADT']<0))[dw].mean();hh=(gate&(dm['ADT']<0)&(dq['ADT']<0))[sw].mean()
    a=ref['image_auroc']>=.75 and ref['DeepCaptureRate']>=.60 and ref['ShallowProtectionRate']>=.75 and ref['BA']>=.70
    b=all(pr[k]['benefit_rate']>pr[k]['harm_rate'] for k in ('all','Top20')) and sum(pr[k]['mean_dm']>0 for k in ('all','Top20','class0','class1','class2','class3'))>=5
    c=br>=.60 and pr['Deep-Win']['benefit_rate']>=.60 and pr['Deep-Win']['mean_dm']>0
    d=hh<=.30 and pr['Shallow-Win']['harm_rate']<=.30 and ref['ShallowProtectionRate']>=.70
    e=pr['Raw_Correct']['harm_rate']<=.30 and pr['Raw_Wrong']['benefit_rate']>=.40 and pr['all']['active_transfer_fraction']>=.10
    sig=np.quantile(reps[:,keys.index('ADT-RG:all:mean_dm')],.025)>0 or np.quantile(reps[:,keys.index('ADT-RG:Shallow-Win:harm_rate')],.975)<0 or np.quantile(reps[:,keys.index('ADT-RG:Deep-Win:benefit_rate')],.025)>0
    f=pr['all']['mean_dm']>utility['RG','all']['mean_dm'] and (pr['Shallow-Win']['harm_rate']<utility['RG','Shallow-Win']['harm_rate'] or pr['Deep-Win']['benefit_rate']>utility['RG','Deep-Win']['benefit_rate']) and sig
    g=all(checks[k] for k in ('all_finite','detach_no_forbidden_gradients','frozen_head_zero','each_b4_conv_group_active','BF16_batch20','no_optimizer_test_luad','state_bn_checkpoint_identity','official_prediction_identity'))
    decision='DIRECTIONAL_TRANSFER_ENGINEERING_NOGO' if not g else 'SYMMETRIC_ADJUDICATION_REPRODUCTION_FAIL' if not a else 'ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE' if not all((b,c,d,e)) else 'DIRECTIONAL_TRANSFER_NOT_BETTER_THAN_RANDOM_SELECTION' if not f else 'RDDR_PHASE2B19_DIRECTIONAL_TRANSFER_GO'
    checks['independent_gates_decision']=decision==summary['decision'] and all(summary['gate_'+k]==('PASS' if v else 'FAIL') for k,v in zip('ABCDEFG',(a,b,c,d,e,f,g)))
    soft=utility['SDT','all']['mean_dm']>pr['all']['mean_dm'] and utility['SDT','Shallow-Win']['harm_rate']<=pr['Shallow-Win']['harm_rate']+.05 and (gate&(dm['SDT']>0)&(dq['SDT']<0))[dw].mean()>=br-.05
    strong=ref['image_auroc']>=.78 and ref['DeepCaptureRate']>=.63 and ref['ShallowProtectionRate']>=.78 and br>=.63 and hh<=.22 and pr['Raw_Correct']['harm_rate']<=.20 and pr['Raw_Wrong']['benefit_rate']>=.50 and pr['all']['mean_dm']>0 and pr['Top20']['mean_dm']>0 and f
    checks['secondary_strong_flags']=bool(soft)==summary['SOFT_DIRECTIONAL_TRANSFER_PROMISING'] and bool(strong)==summary['STRONG_DIRECTIONAL_TRANSFER_SIGNAL']
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},errors=errors,images=n,resamples=10000,decision=decision,
        method='independent FP32 replay + FP64 epsilon-KL/JS proofs; trapezoidal AUROC; direct gather bootstrap',command=shlex.join([sys.executable,*sys.argv]),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    path=report/(P+'verification.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
