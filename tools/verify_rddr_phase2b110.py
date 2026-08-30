"""Independent residual audit verifier. No imports from primary modules."""
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
from scipy.ndimage import uniform_filter
P='rddr_phase2b110_';EPS=1e-8

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def div(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)
def roc_ap(score,label):
    y=np.asarray(label,bool);positive=int(y.sum());negative=len(y)-positive
    if not len(y):return np.nan,np.nan
    ix=np.argsort(-score,kind='stable');x=score[ix];yy=y[ix];end=np.r_[np.flatnonzero(x[1:]!=x[:-1]),len(x)-1]
    tp=np.cumsum(yy)[end];fp=np.cumsum(~yy)[end]
    auc=float(np.trapz(np.r_[0,tp]/positive,np.r_[0,fp]/negative)) if positive and negative else np.nan
    ap=float(np.sum(np.diff(np.r_[0,tp])*tp/(tp+fp))/positive) if positive else np.nan
    return auc,ap
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();out=Path(args.run)
    def js(name):return json.loads((out/(P+name+'.json')).read_text())
    def rows(name):
        with (out/(P+name+'.csv')).open(newline='') as f:return list(csv.DictReader(f))
    rt=js('runtime');s=js('summary');ident=js('identity_audit');checks={};errors={}
    assert not (out/(P+'verification.json')).exists()
    checks['immutable_source_hashes']=all(sha(p)==rt['source_sha256'][k] for k,p in rt['paths'].items())
    data=load(rt['paths']['native']);old=load(rt['paths']['derived']);obs=load(rt['paths']['observations']);n=len(data['names']);y=data['truth'];fg=y<4
    checks['full3418_order']=n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],obs['names'])
    ss=(old['T_SS']+old['T_SD'])/2;sd=(old['T_DS']+old['T_DD'])/2;delta=sd-ss;gate=delta>0
    checks['sD_delta_gate_exact']=np.array_equal(delta,old['sym']) and np.array_equal(gate,obs['direction_gate'])
    checks['q_exact_cache_replay']=rt['replay']['errors']['q']<=1e-7
    raw=data['ps'].argmax(1);deep=data['pd'].argmax(1);rw=fg&(raw!=y);rrw=rw&~gate;rdw=rrw&(deep==y);rbw=rrw&(deep!=y);rsw=fg&~gate&(raw==y)&(deep!=y)
    z=obs['raw_logits'].astype(float);other=np.where(np.arange(4)[None,:,None]==y[:,None],-np.inf,z);tied=other==other.max(1,keepdims=True)
    dms=[]
    for j in (0,2):
        v=-obs['gradients'][:,j].astype(float);dms.append(np.take_along_axis(v,y.clip(0,3)[:,None],1)[:,0]-np.where(tied,v,-np.inf).max(1))
    udm,adm=dms;ben=rrw&(udm>0);harm=rrw&(udm<0);zero=rrw&(udm==0)
    checks['residual_partition']=np.array_equal(rrw,rdw|rbw) and np.array_equal(rrw,ben|harm|zero) and not (ben&harm).any()
    counts={'foreground':fg,'Raw_Wrong':rw,'Residual':fg&~gate,'R_RW':rrw,'Rejected_Deep_Win':rdw,'Rejected_Both_Wrong':rbw,'Rejected_Shallow_Win':rsw,
        'Residual_Beneficial':ben,'Residual_Harmful':harm,'Residual_Zero':zero}
    checks['counts_all_populations']=all(int(r['count'])==counts[r['population']].sum() and int(r['images'])==counts[r['population']].any(1).sum() for r in rows('residual_counts'))
    checks['phase2b19_frozen_counts']=(rw.sum(),rrw.sum(),rdw.sum(),rbw.sum(),rsw.sum())==(708407,435185,113204,321981,144662)
    nrw=int(rw.sum());b=int((rw&(adm>0)).sum());target=math.ceil(nrw*2/5);gap=target-b;head=s['headroom']
    checks['exact_integer_gap']=head['required_additional']==gap==31266 and head['ADT_beneficial']==b==252097 and head['target_beneficial']==target
    checks['headroom_arithmetic']=head['residual_beneficial']==int(ben.sum()) and head['headroom_over_gap']==int(ben.sum())-gap and abs(head['coverage_headroom']-ben.sum()/nrw)<1e-15
    a=data['ps'].astype(float);d=data['pd'].astype(float)
    score={'S_D_sym':sd,'Delta_sym':delta,'q':data['q_feature'],'deep_confidence_advantage':data['pd'].max(1)-data['ps'].max(1),
           'deep_entropy_advantage':-(a*np.log(a+EPS)).sum(1)+(d*np.log(d+EPS)).sum(1)}
    checks['finite_scores_derivatives']=all(np.isfinite(v).all() for v in (*score.values(),udm,adm))
    # Independent separable float64 box filter, compared with frozen FP32 exact-order replay.
    den=uniform_filter(np.ones((1,1,28,28)),size=(1,1,15,15),mode='constant')*225-1
    contexts=[];context_error=0.
    for name,p in (('ctx_S',a),('ctx_D',d)):
        x=p.reshape(n,4,28,28);c=((uniform_filter(x,size=(1,1,15,15),mode='constant')*225-x)/den).reshape(n,4,784)
        context_error=max(context_error,float(np.abs(c-old[name]).max()));contexts.append(c)
    errors['independent_FP64_context']=context_error
    checks['ctx_exact_primary_and_independent']=all(rt['replay']['errors'][k]==0 for k in ('ctx_S','ctx_D','ctx_sym')) and np.array_equal(old['ctx_sym'],(old['ctx_S']+old['ctx_D'])*.5) and context_error<1e-6
    strata={'all':np.ones_like(rrw),'Top20':data['top20'].astype(bool),'Bottom80':~data['top20'].astype(bool),**{f'class{k}':y==k for k in range(4)},
        'boundary':data['boundary'].astype(bool),'interior':~data['boundary'].astype(bool)}
    images={};rankmap={};re=0.
    for r in rows('score_controls'):
        group=r['group'];name=r['score'];pop=r['population'];valid=(ben|harm) if pop=='residual_utility' else (rdw|rsw);positive=ben if pop=='residual_utility' else rdw
        m=valid&strata[group];sc=score[name];auc,ap=roc_ap(sc[m],positive[m]);ia=np.full(n,np.nan)
        for i in range(n):ia[i]=roc_ap(sc[i,m[i]],positive[i,m[i]])[0]
        imageauc=np.nanmean(ia) if np.isfinite(ia).any() else np.nan
        expected=dict(auroc=auc,auprc=ap,image_auroc=imageauc,prevalence=positive[m].mean() if m.any() else np.nan)
        for k,v in expected.items():
            got=float(r[k]);assert np.isnan(got)==np.isnan(v)
            if np.isfinite(v):re=max(re,abs(got-v))
        assert int(r['positive'])==(positive&m).sum() and int(r['negative'])==(~positive&m).sum() and int(r['eligible_images'])==np.isfinite(ia).sum()
        images[pop,group,name]=ia;rankmap[pop,group,name]=dict(expected,positive=int((positive&m).sum()),negative=int((~positive&m).sum()),eligible_images=int(np.isfinite(ia).sum()))
    errors['rank_metrics']=re;checks['all_primary_control_rankings']=re<1e-12
    checks['five_scores_no_substitution']=len(rows('score_controls'))==90 and {r['score'] for r in rows('residual_utility_ranking')}=={'S_D_sym'} and {r['score'] for r in rows('rejected_winner_ranking')}=={'S_D_sym'}
    ce=0.
    for file,pop in (('beneficial_composition',ben),('harmful_composition',harm),('zero_composition',zero)):
        for r in rows(file):
            m=pop&counts[r['source']];assert int(r['count'])==m.sum()
            values=dict(fraction_of_utility=div(m.sum(),pop.sum()),mean_q=score['q'][m].mean() if m.any() else np.nan,
                mean_S_D=sd[m].mean() if m.any() else np.nan,mean_Delta=delta[m].mean() if m.any() else np.nan,mean_deep_confidence=data['pd'].max(1)[m].mean() if m.any() else np.nan)
            for k,v in values.items():
                got=float(r[k]);assert np.isnan(got)==np.isnan(v)
                if np.isfinite(v):ce=max(ce,abs(got-v))
    checks['beneficial_harmful_composition']=ce<1e-12
    for file,key in (('delta_quintiles','Delta_sym'),('deep_support_quintiles','S_D_sym')):
        sorted_values=np.sort(score[key][rrw]);loc=(len(sorted_values)-1)*np.array([.2,.4,.6,.8]);lower=np.floor(loc).astype(int);upper=np.ceil(loc).astype(int)
        edges=sorted_values[lower]+(sorted_values[upper]-sorted_values[lower])*(loc-lower)
        # np.quantile linear uses a numerically stable interpolation branch; tolerate its ulp only.
        np.testing.assert_allclose(edges,s['quintile_edges'][file],rtol=0,atol=1e-7)
        frozen_edges=np.asarray(s['quintile_edges'][file]);bins=np.searchsorted(frozen_edges,score[key],side='left')
        for r in rows(file):
            m=rrw&(bins==int(r['quintile'])-1)
            assert int(r['count'])==m.sum() and int(r['beneficial_count'])==(m&ben).sum() and int(r['harmful_count'])==(m&harm).sum() and int(r['zero_count'])==(m&zero).sum()
        assert sum(int(r['count']) for r in rows(file))==rrw.sum()
    checks['quintile_partitions_no_tie_splitting']=True
    classrows={};group_error=0.
    for file in ('per_class','boundary_interior','top20_bottom80'):
        for r in rows(file):
            group=r['group'];m=rrw&strata[group];rank=rankmap['residual_utility',group,'S_D_sym'];power='POWERED' if rank['positive']>=500 and rank['negative']>=500 and rank['eligible_images']>=30 else 'UNDERPOWERED'
            assert r['power']==power and int(r['count'])==m.sum() and int(r['rejected_Deep_Win'])==(m&rdw).sum() and int(r['rejected_Both_Wrong'])==(m&rbw).sum()
            assert abs(float(r['beneficial_rate'])-(m&ben).sum()/m.sum())<1e-12
            group_error=max(group_error,abs(float(r['image_auroc'])-rank['image_auroc']))
            if file=='per_class':classrows[group]=dict(rank,power=power)
    checks['cross_stratum_and_power']=group_error<1e-12
    ctx=old['ctx_sym'];cp=ctx.argmax(1);diff=(cp!=raw)&(cp!=deep);correct=cp==y;cmerr=0.
    for r in rows('third_evidence_bothwrong')+rows('third_evidence_harm_control'):
        m=counts[r['population']];yy=y[m].astype(int);pp=ctx.transpose(0,2,1)[m].astype(float)
        matrix=np.bincount(yy*4+cp[m],minlength=16).reshape(4,4);tp=np.diag(matrix);denom=matrix.sum(0)+matrix.sum(1)
        iou=div(tp,denom-tp);dice=div(2*tp,denom);expected=dict(accuracy=correct[m].mean(),miou=np.nanmean(iou),dice=np.nanmean(dice),
            nll=-np.log(pp[np.arange(len(yy)),yy]+EPS).mean(),brier=((pp-np.eye(4)[yy])**2).sum(1).mean(),
            rescue_rate=(correct&diff)[m].mean(),rescue_precision=div((correct&diff&m).sum(),(diff&m).sum()),intrusion_rate=diff[m].mean(),third_harm_rate=(~correct&diff)[m].mean())
        for k,v in expected.items():cmerr=max(cmerr,abs(float(r[k])-v))
        assert int(r['correct_third_class'])==(m&diff&correct).sum() and int(r['wrong_third_class'])==(m&diff&~correct).sum()
    errors['context_metrics']=cmerr;checks['context_confusion_nll_brier_rescue']=cmerr<1e-12
    checks['third_rescue_identity']=np.array_equal((correct&rbw),(diff&correct&rbw))
    # Independently rebuild per-image terms and gather sampled rows instead of matrix multiplication.
    terms={}
    def add(name,v,m):terms[name]=(np.where(m,v,0).sum(1,dtype=float),m.sum(1))
    add('ResidualBeneficial_prevalence',ben,rrw);add('ResidualBeneficial_binary_prevalence',ben,ben|harm);add('CoverageHeadroom_rate',ben,rw)
    add('ResidualBeneficial_count_equivalent',ben.astype(float)*nrw,rw)
    for key,ia in images.items():
        if key[1]=='all' or key[2]=='S_D_sym':terms[':'.join(key)+':image_AUROC']=(np.nan_to_num(ia),np.isfinite(ia).astype(int))
    add('ctx_sym_rejected_BothWrong_accuracy',correct,rbw);add('ThirdClassRescueRate',correct&diff,rbw);add('ThirdClassRescuePrecision',correct&diff,rbw&diff)
    for name in ('Rejected_Deep_Win','Rejected_Shallow_Win'):
        add(name+':ctx_accuracy',correct,counts[name]);add(name+':third_intrusion',diff,counts[name])
    boot=rows('bootstrap_replicates');keys=list(boot[0]);assert set(keys)==set(terms)
    ns=np.stack([terms[k][0] for k in keys],1);ds=np.stack([terms[k][1] for k in keys],1);rng=np.random.default_rng(42);rh=hashlib.sha256();reps=[]
    for _ in range(200):
        ix=rng.integers(0,n,(50,n),dtype=np.int32);rh.update(ix.tobytes());reps.append(div(ns[ix].sum(1),ds[ix].sum(1)))
    reps=np.concatenate(reps);actual=np.array([[float(r[k]) for k in keys] for r in boot]);err=float(np.nanmax(np.abs(reps-actual)))
    cierr=0.;cis={}
    for r in rows('bootstrap'):
        j=keys.index(r['metric']);vv=reps[:,j];valid=np.isfinite(vv);lo,hi=np.quantile(vv[valid],[.025,.975]);estimate=ns[:,j].sum()/ds[:,j].sum()
        cierr=max(cierr,abs(lo-float(r['ci_low'])),abs(hi-float(r['ci_high'])),abs(estimate-float(r['estimate'])));assert int(r['valid_resamples'])==valid.sum()
        cis[r['metric']]=(estimate,lo,hi)
    errors.update(bootstrap_replicates=err,bootstrap_intervals=cierr)
    checks['10000_paired_image_bootstrap']=err<1e-9 and cierr<1e-9 and rh.hexdigest()==s['bootstrap_rng_sha256']
    checks['headroom_bootstrap_denominator']=abs(cis['ResidualBeneficial_count_equivalent'][1]-head['count_equivalent_ci_low'])<1e-9 and abs(cis['ResidualBeneficial_count_equivalent'][1]-cis['CoverageHeadroom_rate'][1]*nrw)<1e-8
    aok=ben.sum()>=gap and cis['ResidualBeneficial_count_equivalent'][1]>=gap
    ua=rankmap['residual_utility','all','S_D_sym']['image_auroc'];wa=rankmap['rejected_winner','all','S_D_sym']['image_auroc']
    bok=ua>=.65 and cis['residual_utility:all:S_D_sym:image_AUROC'][1]>.5;cok=wa>=.65 and cis['rejected_winner:all:S_D_sym:image_AUROC'][1]>.5
    interior=rankmap['residual_utility','interior','S_D_sym']['image_auroc'];good=sum(r['power']=='POWERED' and r['image_auroc']>.55 for r in classrows.values());missing=sum(r['power']=='UNDERPOWERED' for r in classrows.values())
    dstatus='PASS' if interior>.6 and good>=3 else 'UNDERPOWERED' if interior>.6 and good+missing>=3 else 'FAIL'
    third=correct[rbw].mean()>=.25 and cis['ctx_sym_rejected_BothWrong_accuracy'][1]>.2 and (correct&diff)[rbw].mean()>=.2
    decision='RESIDUAL_COVERAGE_HEADROOM_INSUFFICIENT' if not aok else ('DUAL_RESIDUAL_RECOVERY_SIGNAL_SUPPORTED' if third else 'RESIDUAL_DEEP_RECOVERY_SIGNAL_SUPPORTED') if bok and cok and dstatus=='PASS' else ('RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED' if third else 'RESIDUAL_COVERAGE_NOT_RECOVERABLE_WITH_FROZEN_EVIDENCE')
    checks['independent_gates_decision']=decision==s['decision'] and all(s['gate_'+k]==('PASS' if v else 'FAIL') for k,v in zip('ABC',(aok,bok,cok))) and s['gate_D']==dstatus
    strong=aok and bok and cok and dstatus=='PASS' and ua>=.75 and wa>=.75
    checks['independent_secondary_flags']=bool(third)==s['RESIDUAL_THIRD_EVIDENCE_SIGNAL'] and bool(strong)==s['STRONG_RESIDUAL_DEEP_RECOVERY_SIGNAL']
    forbidden=('model_instantiated','network_forward','backward','optimizer_created','optimizer_steps','checkpoint_written','new_recovery_gate','threshold_search','test_access','luad_access','training_split_access')
    checks['no_training_gate_test_luad']=not any(rt[k] for k in forbidden)
    checks['identity_evidence_honest']=not ident['new_checks']['new_state_bn_prediction_test'] and ident['new_checks']['checkpoint_sha_before']==ident['new_checks']['checkpoint_sha_after']==rt['source_sha256']['checkpoint']
    prior=json.loads(Path(rt['paths']['previous_identity']).read_text());checks['inherited_identity_exact']=ident['inherited_phase2b19_identity']==prior
    root=Path(__file__).resolve().parents[1];checks['pure_A0_unchanged']=not subprocess.check_output(['git','diff',rt['a0'],'--','network','tool','train_sshr.py'],cwd=root)
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},errors=errors,images=n,resamples=10000,decision=decision,
        method='independent tied ROC trapezoid/AP, explicit margin, FP64 separable box filter, direct-gather bootstrap and decision',
        command=shlex.join([sys.executable,*sys.argv]),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
    (out/(P+'verification.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
