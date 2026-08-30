"""Independent NumPy recomputation: no imports from loss/analyzer/common helpers."""
import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np

P='rddr_phase2b16_'
EPS=1e-8


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def read_np(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def read_json(path):return json.loads(Path(path).read_text())
def rows(root,name):return list(csv.DictReader((root/(P+name+'.csv')).open()))


def metric(c):
    diag=c.diagonal(axis1=-2,axis2=-1)
    denom=c.sum(-1)+c.sum(-2)
    return diag.sum(-1)/c.sum(axis=(-2,-1)),np.nanmean(np.where(denom-diag>0,diag/(denom-diag),np.nan),axis=-1),np.nanmean(np.where(denom>0,2*diag/denom,np.nan),axis=-1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--report',required=True)
    args=ap.parse_args();run=Path(args.run);report=Path(args.report);dest=report/(P+'verification.json')
    if dest.exists():raise FileExistsError(dest)
    runtime=read_json(run/(P+'runtime.json'));summary=read_json(report/(P+'summary.json'))
    data=read_np(runtime['native']);old=read_np(runtime['derived']);obs=read_np(run/(P+'gradient_observations.npz'))
    checks={};errors={};n=len(data['names']);y=data['truth'];fg=y<4
    checks['input_hashes']=all(digest(runtime[k])==runtime[k+'_sha256'] for k in ('native','derived','checkpoint'))
    checks['observation_hash']=digest(run/(P+'gradient_observations.npz'))==runtime['observations_sha256']
    checks['names_count']=n==3418 and np.array_equal(data['names'],obs['names']) and int(fg.sum())==2479143
    ss=(old['T_SS']+old['T_SD'])*.5;dd=(old['T_DS']+old['T_DD'])*.5
    wd=dd/(ss+dd+EPS);teacher=(1-wd[:,None])*data['ps']+wd[:,None]*data['pd']
    errors['teacher_formula_max_abs']=float(np.abs(teacher-old['anchor_sym']).max())
    errors['wd_max_abs']=float(np.abs(wd-old['wD_sym']).max())
    checks['teacher_parity']=max(errors.values())<=1e-7 and max(runtime['parity_max_abs'].values())<=1e-7
    checks['source_forward_parity']=max(runtime['forward_source_parity'].values())<=1e-7
    ps=data['ps'].astype(float);pd=data['pd'].astype(float);mid=(ps+pd)/2
    q64=((ps*np.log((ps+EPS)/(mid+EPS))).sum(1)+(pd*np.log((pd+EPS)/(mid+EPS))).sum(1))/(2*np.log(2))
    errors['q_float64_vs_frozen_max_abs']=float(np.abs(q64-data['q_feature']).max())
    checks['q_parity']=runtime['parity_max_abs']['q']<=1e-7 and errors['q_float64_vs_frozen_max_abs']<1e-6
    l=obs['logits'];e=np.exp(l.astype(float)-l.max(1)[:,None]);p=e/e.sum(1)[:,None]
    probs={'rect':p,'fixed':data['fixed_average'].astype(float),'teacher':teacher.astype(float)}
    preds={k:v.argmax(1) for k,v in probs.items()}
    masks={'all':fg,'Top20':fg&data['top20'].astype(bool),'Bottom80':fg&~data['top20'].astype(bool)}
    edges=np.array([.020935675129294395,.072734534740448,.163648784160614,.3369627296924591])
    bins=np.searchsorted(edges,data['q_feature'],side='left')
    masks.update({f'Q{k+1}':fg&(bins==k) for k in range(5)})
    masks.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    masks.update({f'class{k}':fg&(y==k) for k in range(4)})
    dw=pd.argmax(1)!=y;sw=ps.argmax(1)!=y
    masks.update({'Deep-Wrong':fg&dw,'Shallow-Wrong':fg&sw,'Both-Wrong':fg&dw&sw,'Rect_Correct':fg&(preds['rect']==y),'Rect_Wrong':fg&(preds['rect']!=y)})
    cms={};gtprob={};score={}
    for key,value in probs.items():
        # A confusion matrix assembled by explicit true/pred masks, unlike bincount in analyzer.
        cms[key]=np.array([[((y==a)&(preds[key]==b)&fg).sum(1) for b in range(4)] for a in range(4)]).transpose(2,0,1)
        score[key]=metric(cms[key].sum(0))
        gtprob[key]=np.take_along_axis(value,y.clip(0,3)[:,None],axis=1)[:,0]
    metric_error=0.
    for row in rows(report,'teacher_metrics'):
        key=row['estimator'];v=probs[key]
        own=[*score[key],-np.log(gtprob[key][fg]+EPS).mean(),((v*v).sum(1)-2*gtprob[key]+1)[fg].mean()]
        metric_error=max(metric_error,max(abs(a-float(row[b])) for a,b in zip(own,('accuracy','miou','dice','nll','brier'))))
    errors['teacher_metric_max_abs']=metric_error;checks['teacher_metrics']=metric_error<1e-12
    transition_error=0.;advantage_error=0.
    for row in rows(report,'teacher_transition'):
        mask=masks[row['stratum']];t=preds[row['teacher']]==y;r=preds['rect']==y
        repair=int((mask&t&~r).sum());harm=int((mask&~t&r).sum())
        assert int(row['repair'])==repair and int(row['harm'])==harm
        transition_error=max(transition_error,abs((repair-harm)/mask.sum()-float(row['net_repair_rate'])))
    checks['transition_counts']=transition_error<1e-14
    for row in rows(report,'teacher_advantage'):
        k=row['teacher'];mask=masks[row['stratum']]
        adv=(gtprob['teacher']-gtprob['fixed']) if k=='teacher-minus-fixed' else gtprob[k]-gtprob['rect']
        vals=[adv[mask].mean(),np.median(adv[mask]),(adv[mask]>0).mean(),(adv[mask]<0).mean()]
        advantage_error=max(advantage_error,max(abs(v-float(row[k])) for v,k in zip(vals,('mean','median','positive_fraction','negative_fraction'))))
    errors['teacher_advantage_max_abs']=advantage_error;checks['teacher_advantage']=advantage_error<1e-12
    q=data['q_feature'].astype(float);weight=q/(q.sum(1)[:,None]+EPS);dm={};norm={};loss_err=grad_err=0.
    for j,name in enumerate(('U','FA','CCA')):
        t=probs['fixed'] if name=='FA' else probs['teacher']
        w=1/784 if name=='U' else weight
        kl=(t*(np.log(t+EPS)-np.log(p+EPS))).sum(1)
        expected=(kl*w).sum(1)
        loss_err=max(loss_err,float(np.abs(expected-obs['losses'][:,j]).max()))
        # Exact derivative of the approved epsilon-inside-log formula, not p-t approximation.
        a=t*p/(p+EPS)
        analytical=(p*a.sum(1)[:,None]-a)*(w if name=='U' else w[:,None])
        g=obs['gradients'][:,j]
        grad_err=max(grad_err,float(np.abs(analytical-g).max()))
        # Independent directional derivative via gather over non-GT class index table.
        other=np.array([[b for b in range(4) if b!=a] for a in range(4)])
        idx=other[y.clip(0,3)].transpose(0,2,1)
        comp_l=np.take_along_axis(l,idx,axis=1);comp_move=np.take_along_axis(-g,idx,axis=1)
        largest=comp_l.max(1,keepdims=True);tied=comp_l==largest
        target_move=np.take_along_axis(-g,y.clip(0,3)[:,None],axis=1)[:,0]
        dm[name]=target_move-np.max(np.where(tied,comp_move,-np.inf),axis=1)
        norm[name]=np.linalg.norm(g.astype(float),axis=1)
    errors['loss_float64_vs_autograd_max_abs']=loss_err;errors['analytic_gradient_max_abs']=grad_err
    # Independent FP64 vs original BF16-forward/FP32-softmax loss has rounding error.
    checks['loss_recompute']=loss_err<5e-6;checks['analytical_logit_gradient']=grad_err<2e-8
    checks['all_gradients_finite']=np.isfinite(obs['gradients']).all() and np.isfinite(obs['feature_norm']).all()
    util_error=0.;um={}
    for row in rows(report,'gradient_semantic_utility'):
        mode=row['loss'];group=row['stratum'];mask=masks[group];v=dm[mode][mask]
        own=[(v>0).mean(),(v<0).mean(),v.mean(dtype=float),np.median(v),norm[mode][mask].mean()]
        keys=('benefit_rate','harm_rate','mean_dm','median_dm','gradient_norm')
        util_error=max(util_error,max(abs(a-float(row[b])) for a,b in zip(own,keys)))
        assert int(row['targets'])==mask.sum() and int(row['images'])==mask.any(1).sum()
        um[mode,group]=dict(zip(keys,own))
    errors['semantic_utility_max_abs']=util_error;checks['semantic_utility']=util_error<1e-12
    checks['tied_competitor_count']=int(((tied.sum(1)>1)&fg).sum())==summary['mathematical_identities']['tied_competitor_pixels']
    checks['positive_scalar_sign_identity']=int(((np.sign(dm['U'])!=np.sign(dm['CCA']))&fg&(q>0)).sum())==summary['mathematical_identities']['cca_u_dm_sign_mismatch']
    boot=rows(report,'bootstrap');keys=[r['metric'] for r in boot];numerators=[];denominators=[]
    for key in keys[1:]:
        if ':' not in key:
            if key=='teacher-fixed_accuracy':v=(preds['teacher']==y).astype(int)-(preds['fixed']==y)
            elif key=='teacher-vs-rect_NetRepair':v=(preds['teacher']==y).astype(int)-(preds['rect']==y)
            else:v=(preds['fixed']==y).astype(int)-(preds['rect']==y)
            mask=fg
        else:
            mode,group,kind=key.split(':');mask=masks[group]
            def values(mode):return (dm[mode]>0).astype(float) if kind=='benefit_rate' else dm[mode].astype(float)
            if '-' in mode:a,b=mode.split('-');v=values(a)-values(b)
            else:v=values(mode)
        numerators.append((v*mask).sum(1,dtype=float));denominators.append(mask.sum(1))
    nums=np.array(numerators).T;dens=np.array(denominators).T
    rng=np.random.default_rng(42);rh=hashlib.sha256();rep=[]
    for _ in range(200):
        ids=rng.integers(0,n,(50,n),dtype=np.int32);rh.update(ids.tobytes())
        # Gather-and-sum implementation independent of analyzer's multiplicity matrix products.
        mi=metric(cms['teacher'][ids].sum(1))[1]-metric(cms['fixed'][ids].sum(1))[1]
        rep.append(np.c_[mi,nums[ids].sum(1)/dens[ids].sum(1)])
    rep=np.concatenate(rep)
    saved=np.array([[float(r[k]) for k in keys] for r in rows(report,'bootstrap_replicates')])
    errors['bootstrap_replicate_max_abs']=float(np.abs(rep-saved).max())
    checks['bootstrap_reproducible']=rh.hexdigest()==summary['bootstrap_rng_sha256'] and errors['bootstrap_replicate_max_abs']<1e-12
    cierr=0.
    for j,r in enumerate(boot):
        ci=np.quantile(rep[:,j],[.025,.975]);cierr=max(cierr,abs(ci[0]-float(r['ci_low'])),abs(ci[1]-float(r['ci_high'])))
    errors['bootstrap_ci_max_abs']=cierr;checks['bootstrap_intervals']=cierr<1e-12
    smoke=read_json(run/(P+'bf16_smoke.json'));ident=read_json(run/(P+'identity_audit.json'));detach=read_json(run/(P+'detach_audit.json'))
    checks['teacher_q_detached']=detach['teacher_detached'] and detach['q_detached'] and smoke['ps_teacher_grad_none'] and smoke['pd_teacher_grad_none']
    checks['checkpoint_unchanged']=ident['checkpoint_sha_before']==ident['checkpoint_sha_after']==runtime['checkpoint_sha256']
    checks['inference_off_exact']=ident['official_before']==ident['official_after'] and ident['fixed160_forward_tensors_exact']
    checks['zero_update_state_identity']=ident['state_before']==ident['state_after'] and ident['optimizer_steps']==0 and not runtime['optimizer_created']
    checks['no_test_luad_access']=not runtime['test_access'] and not runtime['luad_access'] and not runtime['train_access']
    checks['batch20_bf16_backward']=smoke['batch']==20 and smoke['finite'] and smoke['feature_gradient_nonzero'] and smoke['budget_pass']
    checks['selected_parameter_paths_nonzero']=all(float(r['pooled_L2'])>0 for r in rows(report,'gradient_path_attribution'))
    checks['feature_gradient_nonzero']=bool((obs['feature_norm'].max(1)>0).all())
    changed=subprocess.check_output(['git','diff','--name-only',runtime['a0_commit'],'HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).splitlines()
    checks['original_model_training_inference_unchanged']=all(p.startswith(('docs/','tools/','tests/','audit/')) for p in changed)
    ca=score['teacher'][1]>score['fixed'][1] and score['teacher'][0]>score['fixed'][0] and any(float(r['ci_low'])>0 for r in boot if r['metric'] in ('teacher-fixed_accuracy','teacher-fixed_miou'))
    cb=norm['CCA'][masks['Top20']].mean()>norm['CCA'][masks['Bottom80']].mean() and norm['CCA'][masks['Top20']].mean()/norm['CCA'][masks['Bottom80']].mean()>norm['U'][masks['Top20']].mean()/norm['U'][masks['Bottom80']].mean() and norm['CCA'][masks['Q5']].mean()>norm['CCA'][masks['Q1']].mean()
    names=('all','Top20','class0','class1','class2','class3')
    sufficient=all(masks[g].sum()>=500 and masks[g].any(1).sum()>=30 for g in names)
    cc=all(um['CCA',g]['benefit_rate']>um['CCA',g]['harm_rate'] for g in ('all','Top20')) and sum(um['CCA',g]['mean_dm']>0 for g in names)>=5
    cd=all(checks[k] for k in ('all_gradients_finite','teacher_q_detached','checkpoint_unchanged','inference_off_exact','zero_update_state_identity','batch20_bf16_backward','selected_parameter_paths_nonzero','feature_gradient_nonzero'))
    gates=[bool(ca),bool(cb),'PASS' if cc and sufficient else ('UNDERPOWERED' if not sufficient else 'FAIL'),bool(cd)]
    checks['independent_gates']=gates==[summary['gate_A']=='PASS',summary['gate_B']=='PASS',summary['gate_C'],summary['gate_D']=='PASS']
    preferred=um['CCA','all']['benefit_rate']>=um['FA','all']['benefit_rate'] and um['CCA','Top20']['mean_dm']>=um['FA','Top20']['mean_dm']
    checks['independent_preference']=bool(preferred)==summary['adjudication_teacher_preferred']
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},errors=errors,
                independent_gates=gates,independent_numpy_only=True,images=n,resamples=10000,
                method='FP64 epsilon-KL analytical derivative, independent confusion masks/tied max gather, gather-sum paired image bootstrap',
                command=shlex.join([sys.executable,*sys.argv]),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['status']=='PASS'


if __name__=='__main__':main()
