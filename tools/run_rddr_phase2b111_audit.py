"""Full validation-cache-only third-evidence audit. No training, autograd or new gate design."""
import argparse
import hashlib
import json
import resource
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b111_common import *
from tools.rddr_probability_replay import replay_probabilities

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in (*HASHES,'output'):ap.add_argument('--'+k.replace('_','-'),required=True)
    args=ap.parse_args();out=Path(args.output);tick=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    paths={k:Path(getattr(args,k)) for k in HASHES}
    assert all(sha(p)==HASHES[k] for k,p in paths.items())
    assert not subprocess.check_output(['git','diff',A0,'--','network','tool','train_sshr.py'],cwd=ROOT)
    def audit_open(event,items):
        if event=='open' and isinstance(items[0],(str,bytes)):
            assert '/reseg-data/' not in str(items[0]).replace('\\','/').lower()
    sys.addaudithook(audit_open)
    data,old,obs=(loadnp(paths[k]) for k in ('native','derived','observations'))
    prev,prt,pid,pver=(json.loads(paths[k].read_text()) for k in ('previous_summary','previous_runtime','previous_identity','previous_verification'))
    assert prev['decision']=='RESIDUAL_THIRD_EVIDENCE_ROUTE_SUPPORTED' and pver['status']=='PASS'
    n=len(data['names']);assert n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],obs['names'])
    out.mkdir(parents=True)
    def wc(name,r):write_csv(out/(P+name+'.csv'),r)
    replay=replay_probabilities(data,old,obs['raw_logits'])
    sd=.5*(old['T_DS']+old['T_DD']);ss=.5*(old['T_SS']+old['T_SD']);delta=sd-ss;md=delta>0
    assert np.array_equal(delta,old['sym']) and np.array_equal(md,obs['direction_gate'])
    assert np.array_equal(old['ctx_sym'],.5*(old['ctx_S']+old['ctx_D']))
    # All-native score/candidate functions are called BEFORE any truth is used.
    c=candidate(data['ps'],data['pd'],old['ctx_sym'],md)
    scs=scores(data['ps'],data['pd'],old['ctx_sym'],data['q_feature'],delta,c)
    kl,g=context_gradient(data['ps'],old['ctx_sym'],c['A_alt'])
    y=data['truth'];assert set(np.unique(y))<={0,1,2,3,4,255}
    dm=margin_direction(obs['raw_logits'],g,y);m=masks(y,c,dm)
    assert all(np.isfinite(x).all() for x in (*scs.values(),g,dm,kl))
    assert (g.transpose(0,2,1)[~c['A_alt']]==0).all() and (dm[~c['A_alt']]==0).all()
    expected={'RawWrong':708407,'RawWrongResidual':435185,'DeepWin':113204,'BothWrong':321981,'ShallowWin':144662}
    assert all(int(m[k].sum())==v for k,v in expected.items())
    udm=margin_direction(obs['raw_logits'],obs['gradients'][:,0],y)
    adm=margin_direction(obs['raw_logits'],obs['gradients'][:,2],y)
    assert int((m['RawWrongResidual']&(udm>0)).sum())==177865
    badt=int((m['RawWrong']&(adm>0)).sum());nrw=int(m['RawWrong'].sum());gap=(2*nrw+4)//5-badt
    assert gap==prev['headroom']['required_additional']==31266
    legacy=[]
    for key,ref in zip(('BothWrong','DeepWin','ShallowWin'),prev['third_evidence']):
        pop=m[key];a=pop&c['a_alt'];good=a&(c['cc']==y);bad=a&(c['cc']!=y)
        assert int(a.sum())==ref['different_from_both'] and int(good.sum())==ref['correct_third_class'] and int(bad.sum())==ref['wrong_third_class']
        legacy.append(dict(group=key,old_alternative=int(a.sum()),old_rescue=int(good.sum()),old_wrong=int(bad.sum()),
            strict_alternative=int((pop&c['A_alt']).sum()),strict_rescue=int((pop&m['ThirdRescue']).sum()),
            rejected_zero_margin=int((a&(c['M_alt']==0)).sum())))
    wc('legacy_replay',legacy)
    counts=[]
    for name,pop in (('all_native',np.ones_like(md)),('foreground',m['foreground']),('background4',y==4),('ignore255',y==255)):
        u=pop&c['U_R'];a=pop&c['A_alt'];legacy_alt=u&c['a_alt']
        counts.append(dict(population=name,total=int(pop.sum()),universe=int(u.sum()),candidate=int(a.sum()),
            candidate_rate_in_universe=float(divide(a.sum(),u.sum())),legacy_alternative=int(legacy_alt.sum()),
            strict_zero_rejected=int((legacy_alt&(c['M_alt']==0)).sum()),
            raw_argmax_tie=int((pop&((data['ps']==data['ps'].max(1,keepdims=True)).sum(1)>1)).sum()),
            deep_argmax_tie=int((pop&((data['pd']==data['pd'].max(1,keepdims=True)).sum(1)>1)).sum()),
            ctx_argmax_tie=int((pop&((old['ctx_sym']==old['ctx_sym'].max(1,keepdims=True)).sum(1)>1)).sum()),
            rescue=int((a&(c['cc']==y)).sum()) if name=='foreground' else None,
            failure=int((a&(c['cc']!=y)).sum()) if name=='foreground' else None))
    wc('candidate_counts',counts)
    a=m['candidate'];u=m['U_R'];fg=m['foreground'];rescue=m['ThirdRescue'];fail=m['AlternativeFailure']
    compm={'BothWrong_ctx_correct':m['BothWrong']&rescue,'BothWrong_ctx_wrong':m['BothWrong']&fail,
        'DeepWin_intrusion':m['DeepWin']&a,'ShallowWin_intrusion':m['ShallowWin']&a,'StableCorrect_intrusion':m['StableCorrect']&a}
    covered=np.logical_or.reduce(list(compm.values()));compm['other']=a&~covered
    assert not compm['other'].any() and sum(int(x.sum()) for x in compm.values())==int(a.sum())
    wc('candidate_composition',[dict(group=k,count=int(z.sum()),fraction=float(divide(z.sum(),a.sum()))) for k,z in compm.items()])
    hard=[];final=np.where(c['A_alt'],c['cc'],c['cs']);repair=m['Repair'];harm=m['Harm'];net=repair.astype(int)-harm.astype(int)
    for name,pop in (('U_R',u),('all_foreground',fg)):
        hard.append(dict(population=name,count=int(pop.sum()),candidate=int((pop&a).sum()),Repair=int((pop&repair).sum()),Harm=int((pop&harm).sum()),
            WrongToWrong_activated=int((pop&m['WrongToWrong']).sum()),StableCorrect_activated=int((pop&m['StableCorrectActivated']).sum()),
            WrongToWrong_full=int((pop&(c['cs']!=y)&(final!=y)).sum()),StableCorrect_full=int((pop&(c['cs']==y)&(final==y)).sum()),
            NetRepair=int(net[pop].sum()),raw_accuracy=float((c['cs'][pop]==y[pop]).mean()),diagnostic_accuracy=float((final[pop]==y[pop]).mean()),
            hard_accuracy_delta=float(net[pop].mean())))
    wc('hard_effect',hard)
    print(json.dumps(dict(phase='replay_and_candidates_complete',images=n,foreground_candidates=int(a.sum()),rescue=int(rescue.sum()),failure=int(fail.sum()),probability_replay=replay)),flush=True)
    bins=np.searchsorted(Q_EDGES,data['q_feature'],side='left')
    strata={'all':np.ones_like(a),'Top20':data['top20'].astype(bool),'Bottom80':~data['top20'].astype(bool),
        **{f'Q{k+1}':bins==k for k in range(5)},**{f'class{k}':y==k for k in range(4)},
        'boundary':data['boundary'].astype(bool),'interior':~data['boundary'].astype(bool)}
    rankrows=[];ias={};lookup={}
    for task,label,eligible in (('rescue',rescue,a),('bothwrong',m['BothWrong'],u),('gradient',m['beneficial'],m['beneficial']|m['harmful'])):
        for group,st in strata.items():
            for score,values in scs.items():
                if group!='all' and score!='M_alt':continue
                val=np.where(c['A_alt'],values,0) if task=='bothwrong' and score=='M_alt' else values
                r,ia=ranking(val,label,eligible&st)
                row=dict(task=task,group=group,score=score,**r,targets=int((eligible&st).sum()),
                    zero_excluded=int((m['zero']&st).sum()) if task=='gradient' else 0)
                rankrows.append(row);lookup[task,group,score]=row;ias[task,group,score]=ia
    for file,task in (('rescue_ranking','rescue'),('bothwrong_detection','bothwrong'),('gradient_ranking','gradient')):
        wc(file,[r for r in rankrows if r['task']==task and r['score']=='M_alt'])
    wc('score_controls',[r for r in rankrows if r['group']=='all'])
    gradrows=[]
    gm=dict(strata,ThirdRescue=rescue,AlternativeFailure=fail,BothWrong=m['BothWrong'],DeepWin_intrusion=m['DeepWin'],
        ShallowWin_intrusion=m['ShallowWin'],RawCorrect=m['RawCorrect'],RawWrong=m['RawWrongResidual'],StableCorrect_intrusion=m['StableCorrect'])
    for name,mask in gm.items():gradrows.append(dict(group=name,**utility(dm,a&mask)))
    wc('context_gradient',gradrows)
    protections={}
    for key in ('RawCorrect','DeepWin','ShallowWin','BothWrong','StableCorrect'):
        pop=m[key];on=pop&a
        r=dict(group=key,denominator='full_residual_group',population_count=int(pop.sum()),candidate=int(on.sum()),
            activation_rate=float(divide(on.sum(),pop.sum())),hard_harm_count=int((pop&harm).sum()),hard_harm_rate=float(divide((pop&harm).sum(),pop.sum())),
            third_intrusion_rate=float(divide(on.sum(),pop.sum())),rescue_count=int((pop&rescue).sum()),rescue_rate=float(divide((pop&rescue).sum(),pop.sum())),
            candidate_precision=float(divide((pop&rescue).sum(),on.sum())),unconditional_ctx_accuracy=float(divide((pop&(c['cc']==y)).sum(),pop.sum())),
            **utility(dm,pop))
        active=dict(group=key,denominator='active_only',population_count=int(on.sum()),candidate=int(on.sum()),
            activation_rate=1. if on.any() else np.nan,hard_harm_count=int((on&harm).sum()),hard_harm_rate=float(divide((on&harm).sum(),on.sum())),
            third_intrusion_rate=1. if on.any() else np.nan,rescue_count=int((on&rescue).sum()),rescue_rate=float(divide((on&rescue).sum(),on.sum())),
            candidate_precision=float(divide((on&rescue).sum(),on.sum())),unconditional_ctx_accuracy=float(divide((on&(c['cc']==y)).sum(),on.sum())),**utility(dm,on))
        protections[key]=r
        file={'RawCorrect':'rawcorrect_protection','DeepWin':'deepwin_protection','ShallowWin':'shallowwin_protection','BothWrong':'bothwrong_rescue','StableCorrect':'stablecorrect_intrusion'}[key]
        wc(file,[r,active])
    grouped={}
    for group,st in strata.items():
        sub=u&st;on=a&st;rc=m['RawCorrect']&st;r=lookup['rescue',group,'M_alt'];gr=lookup['gradient',group,'M_alt']
        grouped[group]=dict(group=group,universe=int(sub.sum()),candidate=int(on.sum()),candidate_rate=float(divide(on.sum(),sub.sum())),
            rescue=int((on&rescue).sum()),failure=int((on&fail).sum()),precision=float(divide((on&rescue).sum(),on.sum())),
            image_auroc=r['image_auroc'],pooled_auroc=r['auroc'],auprc=r['auprc'],eligible_images=r['eligible_images'],
            power=class_power(r['positive'],r['negative'],r['eligible_images']),gradient_image_auroc=gr['image_auroc'],
            gradient_power=class_power(gr['positive'],gr['negative'],gr['eligible_images']),
            rawcorrect=int(rc.sum()),rawcorrect_hard_harm=float(divide((rc&harm).sum(),rc.sum())),
            rawcorrect_gradient_harm=float(divide((rc&(dm<0)).sum(),rc.sum())),**utility(dm,on))
    wc('q_strata',[grouped[k] for k in ('Top20','Bottom80','Q1','Q2','Q3','Q4','Q5')])
    wc('per_class',[grouped[f'class{k}'] for k in range(4)])
    wc('boundary_interior',[grouped[k] for k in ('boundary','interior')])
    bw_a=float(divide((m['BothWrong']&a).sum(),a.sum()));bw_u=float(divide(m['BothWrong'].sum(),u.sum()))
    names=[];nums=[];dens=[]
    def term(name,value,pop):
        names.append(name);nums.append(np.where(pop,value,0).sum(1,dtype=float));dens.append(pop.sum(1))
    term('CandidatePrecision',rescue,a);term('CandidateRate',a,u)
    term('ThirdRescue_count_equivalent',rescue.astype(float)*nrw,m['RawWrong'])
    term('ThirdRescue_to_gap',rescue.astype(float)*nrw/gap,m['RawWrong'])
    term('Hard_NetRepair_count_equivalent',net*int(u.sum()),u)
    term('Hard_accuracy_delta_UR',net,u);term('Hard_accuracy_delta_foreground',net,fg)
    term('RawCorrect_hard_HarmRate',harm,m['RawCorrect']);term('RawCorrect_gradient_HarmRate',dm<0,m['RawCorrect'])
    term('Candidate_gradient_BenefitRate',dm>0,a);term('Candidate_gradient_HarmRate',dm<0,a);term('Candidate_gradient_Mean_dM',dm,a)
    term('BW_prevalence_candidate',m['BothWrong'],a);term('BW_prevalence_UR',m['BothWrong'],u)
    term('DeepWin_intrusion',a,m['DeepWin']);term('ShallowWin_intrusion',a,m['ShallowWin'])
    for key,ia in ias.items():
        names.append(':'.join(key)+':image_AUROC');nums.append(np.nan_to_num(ia));dens.append(np.isfinite(ia).astype(int))
    nums=np.stack(nums,1);dens=np.stack(dens,1);base=divide(nums.sum(0),dens.sum(0));reps=[];rh=hashlib.sha256();bt=time.perf_counter()
    for ix in bootstrap_indices(n):
        rh.update(ix.tobytes());w=np.stack([np.bincount(row,minlength=n) for row in ix]);reps.append(divide(w@nums,w@dens))
    reps=np.concatenate(reps);ci=[]
    for j,name in enumerate(names):
        valid=np.isfinite(reps[:,j]);lo,hi=np.quantile(reps[valid,j],[.025,.975]) if valid.any() else (np.nan,np.nan)
        ci.append(dict(metric=name,estimate=base[j],ci_low=lo,ci_high=hi,resamples=10000,valid_resamples=int(valid.sum()),seed=42))
    wc('bootstrap',ci);wc('bootstrap_replicates',[dict(zip(names,row)) for row in reps])
    np.savez_compressed(out/(P+'image_statistics.npz'),names=np.array(names),numerators=nums,denominators=dens)
    cis={r['metric']:r for r in ci};primary={task:lookup[task,'all','M_alt'] for task in ('rescue','bothwrong','gradient')}
    precision=float(divide(rescue.sum(),a.sum()));head=dict(RequiredGap=gap,RawWrong=nrw,ThirdRescueCount=int(rescue.sum()),
        rescue_to_gap=float(rescue.sum()/gap),count_equivalent_ci_low=cis['ThirdRescue_count_equivalent']['ci_low'],
        count_equivalent_ci_high=cis['ThirdRescue_count_equivalent']['ci_high'],CandidatePrecision=precision,BW_prevalence_candidate=bw_a,BW_prevalence_UR=bw_u)
    wc('coverage_headroom',[head])
    ga=rescue.sum()>=gap and head['count_equivalent_ci_low']>=gap and precision>=.65 and cis['CandidatePrecision']['ci_low']>.55
    gb=primary['rescue']['image_auroc']>=.65 and cis['rescue:all:M_alt:image_AUROC']['ci_low']>.50
    gc=primary['bothwrong']['image_auroc']>=.65 and cis['bothwrong:all:M_alt:image_AUROC']['ci_low']>.50 and bw_a>bw_u
    gu=utility(dm,a);gd=gu['benefit_rate']>gu['harm_rate'] and gu['mean_dm']>0 and primary['gradient']['image_auroc']>=.60
    rc=protections['RawCorrect'];ge=rc['hard_harm_rate']<=.08 and rc['harm_rate']<=.15 and protections['DeepWin']['activation_rate']<=.12 and protections['ShallowWin']['activation_rate']<=.10
    gf=cross_gate(grouped['interior']['image_auroc'],[grouped[f'class{k}'] for k in range(4)])
    stronger=lookup['rescue','all','C_ctx']['image_auroc']>primary['rescue']['image_auroc']
    strong=all((ga,gb,gc,gd,ge,gf=='PASS')) and precision>=.75 and primary['rescue']['image_auroc']>=.75 and primary['gradient']['image_auroc']>=.70 and rc['hard_harm_rate']<=.05 and rescue.sum()>=2*gap
    assert all(sha(p)==HASHES[k] for k,p in paths.items())
    assert not subprocess.check_output(['git','diff',A0,'--','network','tool','train_sshr.py'],cwd=ROOT)
    summary=dict(images=n,headroom=head,primary=primary,candidate_gradient=gu,protection=protections,classes={k:grouped[k] for k in ('class0','class1','class2','class3')},
        interior=grouped['interior'],hard_effect=hard,legacy=legacy,candidate_counts=counts,
        **{'gate_'+k:('PASS' if value else 'FAIL') for k,value in zip('ABCDE',(ga,gb,gc,gd,ge))},gate_F=gf,
        CONTEXT_CONFIDENCE_DIAGNOSTIC_STRONGER=stronger,STRONG_THIRD_EVIDENCE_SIGNAL=strong,
        decision=decide(ga,gb,gc,gd,ge,gf),bootstrap_rng_sha256=rh.hexdigest(),bootstrap_seconds=time.perf_counter()-bt,
        no_training=True,no_test_luad=True,no_new_recovery_gate=True,q_edges=Q_EDGES,prior_decision=prev['decision'])
    flags=dict(model_instantiated=False,network_forward=False,backward=False,autograd=False,optimizer_created=False,optimizer_steps=0,
        checkpoint_written=False,threshold_search=False,score_fusion=False,classifier_fit=False,test_access=False,luad_access=False,training_split_access=False,new_gate_design=False)
    identity=dict(new_checks=dict(all_input_sha_unchanged=True,original_sources_unchanged=True,checkpoint_sha=HASHES['checkpoint'],new_model_state_bn_prediction_test=False,**flags),
        inherited_phase110_identity=pid,inherited_sha256=HASHES['previous_identity'],note='Prior model/state/BN/prediction evidence is inherited, not a new model test.')
    runtime=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),a0=A0,
        command=shlex.join([sys.executable,*sys.argv]),paths=paths,source_sha256=HASHES,contract_sha256=sha(ROOT/'docs/rddr_phase2b111_contract.md'),
        images=n,replay=replay,**flags,all_tensors_finite=True,analytic_gradient='r=t*p/(p+eps); g=p*sum(r)-r; frozen FP32 p/t promoted to FP64; no q/reduction',
        total_seconds=time.perf_counter()-tick,peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,numpy=np.__version__)
    for name,value in (('summary',summary),('identity_audit',identity),('runtime',runtime)):write_json(out/(P+name+'.json'),value)
    print(json.dumps(clean({k:summary[k] for k in ('gate_A','gate_B','gate_C','gate_D','gate_E','gate_F','decision','headroom','primary')})),flush=True)

if __name__=='__main__':main()
