"""Independent verification: no imports from primary audit/analysis modules."""
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

P='rddr_phase2b18_';EPS=1e-8


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def stats(cm):
    cm=np.asarray(cm,dtype=np.float64);tp=np.diagonal(cm,axis1=-2,axis2=-1);s=cm.sum(-1)+cm.sum(-2)
    iou=np.divide(tp,s-tp,out=np.full_like(tp,np.nan),where=s>tp)
    dice=np.divide(2*tp,s,out=np.full_like(tp,np.nan),where=s>0)
    return tp.sum(-1)/cm.sum(axis=(-2,-1)),np.nanmean(iou,-1),np.nanmean(dice,-1),iou


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--report',required=True);args=ap.parse_args()
    run=Path(args.run);report=Path(args.report);rt=json.loads((run/(P+'runtime.json')).read_text());summ=json.loads((report/(P+'summary.json')).read_text())
    data=npz(rt['paths']['native']);old=npz(rt['paths']['derived']);obs=npz(run/(P+'observations.npz'));n=len(data['names'])
    def rows(name):
        with (report/(P+name+'.csv')).open(newline='') as f:return list(csv.DictReader(f))
    checks={};errors={}
    checks['immutable_sources']=all(sha(rt['paths'][k])==v for k,v in rt['source_sha256'].items())
    checks['observation_hash']=sha(run/(P+'observations.npz'))==rt['observation_sha256']
    checks['full3418_order']=n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],obs['names'])
    checks['frozen_replay']=max(rt['parity'].values())<=1e-7 and rt['parity']['raw_frozen_head_logits']==rt['parity']['rect_logits']==0
    gf32=gqf32=gf64=gqf64=soft=loss_error=0.
    for i in range(n):
        z=torch.tensor(obs['raw_logits'][i:i+1].reshape(1,4,28,28),device='cuda',requires_grad=True)
        t=torch.tensor(old['anchor_sym'][i:i+1].reshape(1,4,28,28),device='cuda')
        fixed=torch.tensor(data['fixed_average'][i:i+1].reshape(1,4,28,28),device='cuda')
        d=torch.tensor(data['pd'][i:i+1].reshape(1,4,28,28),device='cuda')
        q=torch.tensor(data['q_feature'][i:i+1].reshape(1,28,28),device='cuda')
        p=z.softmax(1);soft=max(soft,float(np.abs(p.detach().flatten(2).cpu().numpy()[0]-data['ps'][i]).max()))
        for j,mode in enumerate(('Uraw','FAraw','PRG')):
            target=fixed if j==1 else t
            kl=(target*((target+EPS).log()-(p+EPS).log())).sum(1)
            loss=kl.mean() if j==0 else (q*kl).sum()/(q.sum()+EPS)
            grad,=torch.autograd.grad(loss,z,retain_graph=True)
            gf32=max(gf32,float(np.abs(grad.detach().flatten(2).cpu().numpy()[0]-obs['gradients'][i,j]).max()))
            loss_error=max(loss_error,abs(loss.item()-float(obs['losses'][i,j])))
            zz=z.detach().double().requires_grad_();pp=zz.softmax(1);tt=target.double();qq=q.double()
            kk=(tt*((tt+EPS).log()-(pp+EPS).log())).sum(1)
            ll=kk.mean() if j==0 else (qq*kk).sum()/(qq.sum()+EPS)
            actual,=torch.autograd.grad(ll,zz)
            a=tt*pp/(pp+EPS);formula=pp*a.sum(1,keepdim=True)-a
            formula=formula/(28*28) if j==0 else formula*(qq/(qq.sum()+EPS))[:,None]
            gf64=max(gf64,float((actual-formula).abs().max()))
        m=.5*(p+d)
        js=.5*((p*((p+EPS).log()-(m+EPS).log())).sum(1)+(d*((d+EPS).log()-(m+EPS).log())).sum(1))/math.log(2)
        qgrad,=torch.autograd.grad(js.sum(),z)
        gqf32=max(gqf32,float(np.abs(qgrad.detach().flatten(2).cpu().numpy()[0]-obs['q_gradients'][i]).max()))
        zz=z.detach().double().requires_grad_();pp=zz.softmax(1);dd=d.double();mm=.5*(pp+dd)
        jj=.5*((pp*((pp+EPS).log()-(mm+EPS).log())).sum(1)+(dd*((dd+EPS).log()-(mm+EPS).log())).sum(1))/math.log(2)
        aa,=torch.autograd.grad(jj.sum(),zz)
        h=.5*((pp+EPS).log()-(mm+EPS).log()+pp/(pp+EPS)-.5*(pp+dd)/(mm+EPS))/math.log(2)
        formula=pp*(h-(pp*h).sum(1,keepdim=True));gqf64=max(gqf64,float((aa-formula).abs().max()))
    errors.update(FP32_loss=loss_error,FP32_loss_gradient=gf32,FP32_q_gradient=gqf32,FP64_KL_formula=gf64,FP64_q_formula=gqf64,raw_probability=soft)
    checks['independent_exact_FP32_losses_gradients']=gf32==loss_error==0
    checks['independent_exact_FP32_q_derivative']=gqf32==0
    checks['FP64_analytic_KL_and_JS']=gf64<1e-12 and gqf64<1e-12
    checks['raw_probability_exact']=soft==0
    checks['all_observation_gradients_finite']=all(np.isfinite(obs[k]).all() for k in ('gradients','q_gradients','feature_sqsum','feature_maxabs','parameter_energy','parameter_maxabs'))
    y=data['truth'];fg=y<4;raw=data['ps'].argmax(1);deep=data['pd'].argmax(1);teacher=old['anchor_sym'].argmax(1)
    sc=raw==y;dc=deep==y;tc=teacher==y
    masks={'all':fg,'Top20':fg&data['top20'].astype(bool),'Bottom80':fg&~data['top20'].astype(bool)}
    edges=[.020935675129294395,.072734534740448,.163648784160614,.3369627296924591]
    bins=np.searchsorted(edges,data['q_feature'],side='left')
    masks.update({f'Q{k+1}':fg&(bins==k) for k in range(5)})
    masks.update(Raw_Correct=fg&sc,Raw_Wrong=fg&~sc)
    masks.update({'Deep-Win':fg&~sc&dc,'Shallow-Win':fg&sc&~dc,'Both-Wrong':fg&~sc&~dc,'Stable-Correct':fg&sc&dc})
    masks.update({f'class{k}':fg&(y==k) for k in range(4)})
    masks.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    dm=[];dq=[];cos=[];norm=[];zz=obs['raw_logits'].astype(float);gq=obs['q_gradients'].astype(float)
    for j in range(3):
        v=-obs['gradients'][:,j].astype(float);other=np.where(np.arange(4)[None,:,None]==y[:,None],-np.inf,zz)
        winners=other==other.max(1,keepdims=True)
        change=np.take_along_axis(v,np.clip(y,0,3)[:,None],1)[:,0]-np.where(winners,v,-np.inf).max(1)
        dot=sum(gq[:,k]*v[:,k] for k in range(4));vn=np.sqrt(sum(v[:,k]**2 for k in range(4)));qn=np.sqrt(sum(gq[:,k]**2 for k in range(4)))
        dm.append(change);dq.append(dot);cos.append(dot/(qn*vn+EPS));norm.append(vn)
    stat_error=0.
    for r in rows('raw_gradient'):
        j=('Uraw','FAraw','PRG').index(r['loss']);m=masks[r['stratum']]
        expected=dict(benefit_rate=(dm[j][m]>0).mean(),harm_rate=(dm[j][m]<0).mean(),zero_rate=(dm[j][m]==0).mean(),
                      mean_dm=dm[j][m].mean(),median_dm=np.median(dm[j][m]),mean_gradient_norm=norm[j][m].mean(),active_fraction=(norm[j][m]>0).mean())
        stat_error=max(stat_error,max(abs(float(r[k])-v) for k,v in expected.items()))
    for r in rows('hierarchy_direction'):
        j=('Uraw','FAraw','PRG').index(r['loss']);m=masks[r['stratum']]
        for name,a in (('dQ',dq[j][m]),('CosCollapse',cos[j][m])):
            for k,v in dict(mean=a.mean(),median=np.median(a),positive_fraction=(a>0).mean(),negative_fraction=(a<0).mean(),zero_fraction=(a==0).mean()).items():
                stat_error=max(stat_error,abs(float(r[name+'_'+k])-v))
    errors['independent_margin_hierarchy_statistics']=stat_error
    checks['all_strata_margin_hierarchy']=stat_error<1e-12
    repair=~sc&tc;harm=sc&~tc;net=repair.astype(np.int64)-harm
    transition_ok=True
    for r in rows('teacher_raw_transition'):
        m=masks[r['stratum']]
        transition_ok &= int(r['repair'])==int((repair&m).sum()) and int(r['harm'])==int((harm&m).sum()) and int(r['net_repair'])==int(net[m].sum())
        transition_ok &= abs(float(r['teacher_accuracy'])-tc[m].mean())<1e-12
    checks['teacher_repair_harm_all_strata']=bool(transition_ok)
    semantic=[]
    for prob in (data['ps'],data['fixed_average'],old['anchor_sym']):
        cm=np.bincount((y*4+prob.argmax(1))[fg],minlength=16).reshape(4,4);a,mi,di,ci=stats(cm)
        pp=prob.transpose(0,2,1)[fg].astype(float);yy=y[fg].astype(int)
        semantic.append(dict(accuracy=a,miou=mi,dice=di,nll=-np.log(pp[np.arange(len(yy)),yy]+EPS).mean(),brier=((pp-np.eye(4)[yy])**2).sum(1).mean(),
                             **{f'iou_class{k}':ci[k] for k in range(4)}))
    errors['semantic_metrics']=max(abs(float(r[k])-s[k]) for r,s in zip(rows('teacher_raw_metrics'),semantic) for k in s)
    checks['semantic_confusion_nll_brier']=errors['semantic_metrics']<1e-12
    feature_error=0.
    for r in rows('feature_gradient'):
        m=masks[r['stratum']];en=obs['feature_sqsum'][m]
        expected=dict(rms=np.sqrt(en.sum()/(m.sum()*512)),mean_pixel_l2=np.sqrt(en).mean(),max_abs=obs['feature_maxabs'][m].max())
        feature_error=max(feature_error,max(abs(float(r[k])-v) for k,v in expected.items()))
    checks['feature_gradient_aggregation']=feature_error<1e-12
    checks['frozen_shared_feature_same']=float(np.max(np.abs(obs['feature_sqsum'].sum(1)-obs['shared_feature_energy'])))<1e-12
    pe=obs['parameter_energy'];pnames=obs['parameter_names'].tolist();numel=obs['parameter_numel'];pmax=obs['parameter_maxabs']
    checks['39_upstream_plus_2_head']=len(pnames)==41 and all(x.split('.')[0] in ('b4','b4_1','b4_2','b4_3','b4_4','b4_5','bn45') for x in pnames[:-2])
    checks['frozen_head_zero']=np.all(pe[:,0,-2:]==0)
    checks['each_b4_conv_group_active']=all(pe[:,0,[j for j,k in enumerate(pnames) if k.startswith(b+'.conv_')]].sum()>0 for b in ('b4','b4_1','b4_2','b4_3','b4_4','b4_5'))
    checks['parameter_energy_bound']=bool(np.all(pe<=numel[None,None]*pmax**2+1e-12))
    fraction=pe[:,1,-2:].sum()/pe[:,1].sum()
    checks['shared_energy_denominator']=abs(fraction-summ['shared_head_parameter_energy_fraction'])<1e-12
    # Rebuild per-image sufficient statistics independently, then gather rows directly for bootstrap.
    image=np.broadcast_to(np.arange(n)[:,None],y.shape)
    cm=np.stack([np.bincount((image*16+y*4+pp)[fg],minlength=n*16).reshape(n,4,4) for pp in (raw,teacher)],1)
    terms={}
    def add(name,value,m,kind='ratio'):terms[name]=(np.where(m,value,0).sum(1,dtype=float),m.sum(1),kind)
    add('teacher-raw_accuracy_delta',net,fg);add('teacher_NetRepair_count',net,fg,'sum');add('teacher_NetRepair_rate',net,fg)
    for k in ('all','Top20','class0','class1','class2','class3'):add('PRG:'+k+':mean_dm',dm[2],masks[k])
    add('PRG:Raw_Correct:harm_rate',dm[2]<0,masks['Raw_Correct']);add('PRG:Raw_Wrong:benefit_rate',dm[2]>0,masks['Raw_Wrong'])
    add('Deep-Win:BRR',(dm[2]>0)&(dq[2]<0),masks['Deep-Win']);add('Shallow-Win:HHCR',(dm[2]<0)&(dq[2]<0),masks['Shallow-Win'])
    add('Shallow-Win:teacher_accuracy',tc,masks['Shallow-Win'])
    for k in ('all','Top20'):
        add('PRG:'+k+':benefit_rate',dm[2]>0,masks[k]);add('PRG:'+k+':harm_rate',dm[2]<0,masks[k])
    bootrows=rows('bootstrap_replicates');keys=list(bootrows[0]);existing=np.array([[float(r[k]) for k in keys] for r in bootrows])
    want=[];rng=np.random.default_rng(42);rnghash=hashlib.sha256()
    ns=np.stack([terms[k][0] for k in keys[:-1]],1);ds=np.stack([terms[k][1] for k in keys[:-1]],1);is_sum=np.array([terms[k][2]=='sum' for k in keys[:-1]])
    for start in range(0,10000,50):
        ix=rng.integers(0,n,(50,n),dtype=np.int32);rnghash.update(ix.tobytes())
        numerator=ns[ix].sum(1);denominator=ds[ix].sum(1);value=numerator/denominator;value[:,is_sum]=numerator[:,is_sum]
        _,mi,_,_=stats(cm[ix].sum(1));want.append(np.column_stack((value,mi[:,1]-mi[:,0])))
    want=np.concatenate(want);errors['bootstrap_replicates']=float(np.max(np.abs(want-existing)))
    errors['bootstrap_intervals']=max(abs(float(r[k])-np.quantile(want[:,j],q)) for j,r in enumerate(rows('bootstrap')) for k,q in (('ci_low',.025),('ci_high',.975)))
    checks['10000_paired_image_bootstrap']=errors['bootstrap_replicates']<1e-7 and errors['bootstrap_intervals']<1e-7 and rnghash.hexdigest()==summ['bootstrap_rng_sha256']
    br=float(((dm[2]>0)&(dq[2]<0))[masks['Deep-Win']].mean());hh=float(((dm[2]<0)&(dq[2]<0))[masks['Shallow-Win']].mean())
    checks['BRR_HHCR_denominators']=abs(br-summ['BRR'])<1e-12 and abs(hh-summ['HHCR'])<1e-12
    identity=json.loads((run/(P+'identity_audit.json')).read_text());detach=json.loads((run/(P+'detach_audit.json')).read_text());smoke=json.loads((run/(P+'bf16_smoke.json')).read_text())
    checks['state_bn_checkpoint_identity']=identity['state_before']==identity['state_after'] and identity['bn_unchanged'] and identity['checkpoint_sha_before']==identity['checkpoint_sha_after']==rt['source_sha256']['checkpoint']
    checks['official_prediction_identity']=identity['prediction_before']==identity['prediction_after'] and identity['prediction_before']['images']==160
    checks['detach_and_no_forbidden_gradients']=all(detach[k] for k in ('teacher_detached','q_detached','deep_source_detached','primary_ic1_none','hfrm_none','all_other_primary_gradients_none'))
    checks['BF16_batch20']=smoke['pass'] and smoke['all_finite'] and smoke['head_energy']==0 and smoke['upstream_conv_energy']>0 and smoke['reserved_bytes']<=22*1024**3
    checks['no_optimizer_test_luad']=not any(rt[k] for k in ('optimizer_created','optimizer_steps','checkpoint_written','test_access','luad_access','training_split_access'))
    checks['original_sources_unchanged']=not subprocess.check_output(['git','diff',rt['a0'],'--','network','tool','train_sshr.py'],cwd=Path(__file__).resolve().parents[1])
    means=[dm[2][masks[k]].mean() for k in ('all','Top20','class0','class1','class2','class3')]
    a=semantic[2]['accuracy']>semantic[0]['accuracy'] and semantic[2]['miou']>semantic[0]['miou'] and net[fg].sum()>0 and (np.quantile(want[:,keys.index('teacher-raw_accuracy_delta')],.025)>0 or np.quantile(want[:,-1],.025)>0)
    b=all((dm[2][masks[k]]>0).mean()>(dm[2][masks[k]]<0).mean() for k in ('all','Top20')) and sum(v>0 for v in means)>=5
    c=(dm[2][masks['Raw_Wrong']]>0).mean()>=.70 and (dm[2][masks['Raw_Correct']]<0).mean()<=.50 and net[fg].sum()>0
    d=br>=.60 and hh<=.30 and tc[masks['Shallow-Win']].mean()>=.60 and norm[2][masks['Q5']].mean()>norm[2][masks['Q1']].mean()
    e=all(checks[k] for k in ('frozen_replay','all_observation_gradients_finite','frozen_head_zero','each_b4_conv_group_active','state_bn_checkpoint_identity','official_prediction_identity','detach_and_no_forbidden_gradients','BF16_batch20','no_optimizer_test_luad'))
    decision='PRERECT_GUIDANCE_ENGINEERING_NOGO' if not e else 'SYMMETRIC_TEACHER_NOT_SUITABLE_FOR_RAW' if not a else 'TEACHER_BETTER_THAN_RAW_BUT_GRADIENT_UNSAFE' if not b or not c else 'PRERECT_GUIDANCE_HIERARCHY_COLLAPSE_RISK' if not d else 'RDDR_PHASE2B18_PRERECT_GUIDANCE_GO'
    checks['independent_all_gates_decision']=decision==summ['decision'] and all(summ['gate_'+k]==('PASS' if v else 'FAIL') for k,v in zip('ABCDE',(a,b,c,d,e)))
    loc=norm[2][masks['Top20']].mean()/norm[2][masks['Bottom80']].mean()>norm[0][masks['Top20']].mean()/norm[0][masks['Bottom80']].mean()
    strong=semantic[2]['miou']-semantic[0]['miou']>=.10 and means[0]>0 and means[1]>0 and (dm[2][masks['Raw_Wrong']]>0).mean()>=.80 and (dm[2][masks['Raw_Correct']]<0).mean()<=.35 and br>=.70 and hh<=.20
    checks['secondary_and_strong_flags']=bool(loc)==summ['CONFLICT_LOCALIZATION_CONFIRMED'] and bool(fraction>.5)==summ['SHARED_HEAD_ABSORPTION_RISK'] and bool(strong)==summ['STRONG_PRERECT_GUIDANCE_SIGNAL']
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},errors=errors,
                images=n,resamples=10000,decision=decision,method='independent FP32 replay + FP64 analytic epsilon-KL/JS; NumPy explicit derivatives/confusion/gather bootstrap',
                code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),command=shlex.join([sys.executable,*sys.argv]))
    path=report/(P+'verification.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
