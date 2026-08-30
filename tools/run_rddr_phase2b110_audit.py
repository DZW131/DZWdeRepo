"""Cache-only residual recoverability audit. No model, network inference, optimizer or training."""
import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b110_common import *

def forbidden(*a,**k):raise RuntimeError('model/optimizer/backward/checkpoint operation forbidden')

def replay_probabilities(data,old):
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
    errors={k:0. for k in ('T_SS','T_SD','T_DS','T_DD','ctx_S','ctx_D','ctx_sym','q')}
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(),patch.object(torch.nn.Module,'__init__',forbidden),patch.object(torch.optim.Optimizer,'__init__',forbidden),patch.object(torch.Tensor,'backward',forbidden),patch.object(torch,'save',forbidden):
        for i in range(len(data['names'])):
            p=torch.from_numpy(data['ps'][i].reshape(1,4,28,28)).cuda();d=torch.from_numpy(data['pd'][i].reshape(1,4,28,28)).cuda()
            ss,ds,cs=support(p,d);dd,sd,cd=support(d,p)
            vals=dict(T_SS=ss,T_SD=sd,T_DS=ds,T_DD=dd,ctx_S=cs,ctx_D=cd,ctx_sym=.5*(cs+cd),q=js(p,d).flatten(1)/math.log(2))
            for k,a in vals.items():
                b=data['q_feature'][i] if k=='q' else old[k][i];errors[k]=max(errors[k],float(np.abs(a.cpu().numpy()[0]-b).max()))
                assert torch.isfinite(a).all() and not a.requires_grad
    assert errors['q']<=1e-7 and all(v==0 for k,v in errors.items() if k!='q'),errors
    torch.cuda.synchronize()
    return dict(errors=errors,seconds=time.perf_counter()-start,allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),model_instantiated=False,network_forward=False,backward=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in (*HASHES,'output'):ap.add_argument('--'+k.replace('_','-'),required=True)
    args=ap.parse_args();out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    paths={k:Path(getattr(args,k)) for k in HASHES};assert all(sha(p)==HASHES[k] for k,p in paths.items())
    assert not subprocess.check_output(['git','diff',A0,'--','network','tool','train_sshr.py'],cwd=ROOT)
    def access_guard(event,items):
        if event=='open' and isinstance(items[0],(str,bytes)):
            assert '/reseg-data/' not in str(items[0]).replace('\\','/').lower(),'No dataset split files needed; frozen caches only'
    sys.addaudithook(access_guard)
    data=loadnp(paths['native']);old=loadnp(paths['derived']);obs=loadnp(paths['observations'])
    prev=json.loads(paths['previous_summary'].read_text());pid=json.loads(paths['previous_identity'].read_text());prt=json.loads(paths['previous_runtime'].read_text())
    n=len(data['names']);assert n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],obs['names'])
    assert prt['observation_sha256']==HASHES['observations'] and prev['decision']=='ADJUDICATION_VALID_DIRECTIONAL_TRANSFER_UNSAFE'
    out.mkdir(parents=True)
    def wc(name,r):write_csv(out/(P+name+'.csv'),r)
    replay=replay_probabilities(data,old)
    scores=frozen_scores(data['ps'],data['pd'],data['q_feature'],old['T_SS'],old['T_SD'],old['T_DS'],old['T_DD'])
    delta=scores['Delta_sym'];gate=delta>0
    assert np.array_equal(delta,old['sym']) and np.array_equal(gate,obs['direction_gate'])
    assert np.array_equal(old['ctx_sym'],.5*(old['ctx_S']+old['ctx_D']))
    udm=margin_direction(obs['raw_logits'],obs['gradients'][:,0],data['truth']);adm=margin_direction(obs['raw_logits'],obs['gradients'][:,2],data['truth'])
    pop=population(data['truth'],data['ps'],data['pd'],gate,udm);rw=pop['Raw_Wrong'];rrw=pop['R_RW'];ben=pop['Residual_Beneficial'];harm=pop['Residual_Harmful'];zero=pop['Residual_Zero']
    assert all(np.isfinite(x).all() for x in (*scores.values(),udm,adm))
    expected={'Raw_Wrong':708407,'R_RW':435185,'Rejected_Deep_Win':113204,'Rejected_Both_Wrong':321981,'Rejected_Shallow_Win':144662}
    assert all(int(pop[k].sum())==v for k,v in expected.items())
    nrw=int(rw.sum());badt=int((rw&(adm>0)).sum());gap=required_gap(nrw,badt)
    assert badt/nrw==prev['primary']['Raw_Wrong']['benefit_rate'] and gap==math.ceil(nrw*(.4-badt/nrw))
    wc('residual_counts',[dict(population=k,count=int(m.sum()),images=int(m.any(1).sum())) for k,m in pop.items()])
    print(json.dumps(dict(phase='frozen_replay_pass',replay=replay,RawWrong=nrw,ADT_beneficial=badt,required_additional=gap)),flush=True)
    strata={'all':np.ones_like(rrw),'Top20':data['top20'].astype(bool),'Bottom80':~data['top20'].astype(bool),
        **{f'class{k}':data['truth']==k for k in range(4)},'boundary':data['boundary'].astype(bool),'interior':~data['boundary'].astype(bool)}
    rankrows=[];iauc={};lookup={}
    for population_name,eligible,label in (('residual_utility',ben|harm,ben),('rejected_winner',pop['Rejected_Deep_Win']|pop['Rejected_Shallow_Win'],pop['Rejected_Deep_Win'])):
        for group,m in strata.items():
            for name,score in scores.items():
                result,ia=ranking(score,label,eligible&m)
                r=dict(population=population_name,group=group,score=name,**result,targets=int((eligible&m).sum()),
                    zero_excluded=int((zero&m).sum()) if population_name=='residual_utility' else 0)
                rankrows.append(r);lookup[population_name,group,name]=r;iauc[population_name,group,name]=ia
    wc('residual_utility_ranking',[r for r in rankrows if r['population']=='residual_utility' and r['score']=='S_D_sym'])
    wc('rejected_winner_ranking',[r for r in rankrows if r['population']=='rejected_winner' and r['score']=='S_D_sym'])
    wc('score_controls',rankrows)
    composition={}
    for kind,z in (('beneficial',ben),('harmful',harm),('zero',zero)):
        cr=[]
        for source in ('Rejected_Deep_Win','Rejected_Both_Wrong'):
            m=z&pop[source]
            cr.append(dict(utility=kind,source=source,count=int(m.sum()),fraction_of_utility=float(divide(m.sum(),z.sum())),mean_q=mean(scores['q'][m]),
                mean_S_D=mean(scores['S_D_sym'][m]),mean_Delta=mean(delta[m]),mean_deep_confidence=mean(data['pd'].max(1)[m])))
        composition[kind]=cr;wc(kind+'_composition',cr)
    def describe(m):
        return dict(count=int(m.sum()),beneficial_count=int((m&ben).sum()),harmful_count=int((m&harm).sum()),zero_count=int((m&zero).sum()),
            beneficial_rate=float(divide((m&ben).sum(),m.sum())),harmful_rate=float(divide((m&harm).sum(),m.sum())),zero_rate=float(divide((m&zero).sum(),m.sum())),
            rejected_Deep_Win=int((m&pop['Rejected_Deep_Win']).sum()),rejected_Both_Wrong=int((m&pop['Rejected_Both_Wrong']).sum()),
            Deep_Win_fraction=float(divide((m&pop['Rejected_Deep_Win']).sum(),m.sum())),Both_Wrong_fraction=float(divide((m&pop['Rejected_Both_Wrong']).sum(),m.sum())),
            mean_S_D=mean(scores['S_D_sym'][m]),mean_q=mean(scores['q'][m]),mean_Delta=mean(delta[m]))
    qmeta={}
    for file,score in (('delta_quintiles','Delta_sym'),('deep_support_quintiles','S_D_sym')):
        edges,masks=diagnostic_quintiles(scores[score],rrw);qmeta[file]=edges
        wc(file,[dict(quintile=i+1,score=score,**describe(m)) for i,m in enumerate(masks)])
    grouped={}
    for group,m in strata.items():
        r=lookup['residual_utility',group,'S_D_sym'];grouped[group]=dict(group=group,**describe(rrw&m),**{k:r[k] for k in ('image_auroc','auroc','auprc','positive','negative','prevalence','eligible_images')},
            power=class_power(r['positive'],r['negative'],r['eligible_images']))
    for file,groups in (('per_class',[f'class{k}' for k in range(4)]),('boundary_interior',['boundary','interior']),('top20_bottom80',['Top20','Bottom80'])):wc(file,[grouped[k] for k in groups])
    crows=[];yp=data['truth'];sp=data['ps'].argmax(1);dp=data['pd'].argmax(1);ctx=old['ctx_sym'];cp=ctx.argmax(1)
    for key in ('Rejected_Both_Wrong','Rejected_Deep_Win','Rejected_Shallow_Win'):
        crows.append(dict(population=key,**context_metrics(ctx,yp,sp,dp,pop[key])))
    assert crows[0]['accuracy']==crows[0]['rescue_rate']
    assert all(r['intrusion_rate']==r['third_harm_rate'] for r in crows[1:])
    wc('third_evidence_bothwrong',crows[:1]);wc('third_evidence_harm_control',crows[1:])
    # Primary denominators never use residual-only prevalence for total-coverage headroom.
    names=[];nums=[];dens=[]
    def term(name,value,m):
        names.append(name);nums.append(np.where(m,value,0).sum(1,dtype=float));dens.append(m.sum(1))
    term('ResidualBeneficial_prevalence',ben,rrw);term('ResidualBeneficial_binary_prevalence',ben,ben|harm)
    term('CoverageHeadroom_rate',ben,rw)
    term('ResidualBeneficial_count_equivalent',ben.astype(float)*nrw,rw)
    for key,ia in iauc.items():
        # Overall controls plus primary for all frozen strata, all with same image draws.
        if key[1]=='all' or key[2]=='S_D_sym':
            names.append(':'.join(key)+':image_AUROC');nums.append(np.nan_to_num(ia));dens.append(np.isfinite(ia).astype(int))
    rbw=pop['Rejected_Both_Wrong'];different=(cp!=sp)&(cp!=dp);correct=cp==yp
    term('ctx_sym_rejected_BothWrong_accuracy',correct,rbw);term('ThirdClassRescueRate',correct&different,rbw)
    term('ThirdClassRescuePrecision',correct&different,rbw&different)
    for key in ('Rejected_Deep_Win','Rejected_Shallow_Win'):
        term(key+':ctx_accuracy',correct,pop[key]);term(key+':third_intrusion',different,pop[key])
    nums=np.stack(nums,1);dens=np.stack(dens,1);base=divide(nums.sum(0),dens.sum(0));rep=[];rh=hashlib.sha256();bt=time.perf_counter()
    for ix in bootstrap_indices(n):
        rh.update(ix.tobytes());w=np.stack([np.bincount(row,minlength=n) for row in ix]);rep.append(divide(w@nums,w@dens))
    rep=np.concatenate(rep);ci=[]
    for j,name in enumerate(names):
        v=rep[:,j];good=np.isfinite(v);lo,hi=np.quantile(v[good],[.025,.975]) if good.any() else (np.nan,np.nan)
        ci.append(dict(metric=name,estimate=base[j],ci_low=lo,ci_high=hi,resamples=10000,valid_resamples=int(good.sum()),seed=42))
    wc('bootstrap',ci);wc('bootstrap_replicates',[dict(zip(names,row)) for row in rep]);cimap={r['metric']:r for r in ci}
    np.savez_compressed(out/(P+'image_statistics.npz'),names=np.array(names),numerators=nums,denominators=dens)
    head=dict(RawWrong=nrw,ADT_beneficial=badt,ADT_benefit_rate=badt/nrw,target_beneficial=(2*nrw+4)//5,required_additional=gap,required_additional_rate=gap/nrw,
        residual_beneficial=int(ben.sum()),residual_harmful=int(harm.sum()),residual_zero=int(zero.sum()),coverage_headroom=float(ben.sum()/nrw),
        count_equivalent_ci_low=cimap['ResidualBeneficial_count_equivalent']['ci_low'],count_equivalent_ci_high=cimap['ResidualBeneficial_count_equivalent']['ci_high'],
        headroom_over_gap=int(ben.sum())-gap,rejected_Deep_Win=int(pop['Rejected_Deep_Win'].sum()),required_fraction_of_rejected_Deep_Win=gap/pop['Rejected_Deep_Win'].sum())
    wc('coverage_headroom',[head])
    ru=lookup['residual_utility','all','S_D_sym'];win=lookup['rejected_winner','all','S_D_sym']
    a=ben.sum()>=gap and head['count_equivalent_ci_low']>=gap
    b=ru['image_auroc']>=.65 and cimap['residual_utility:all:S_D_sym:image_AUROC']['ci_low']>.50
    c=win['image_auroc']>=.65 and cimap['rejected_winner:all:S_D_sym:image_AUROC']['ci_low']>.50
    d=cross_stratum(grouped['interior']['image_auroc'],[grouped[f'class{k}'] for k in range(4)])
    third=crows[0]['accuracy']>=.25 and cimap['ctx_sym_rejected_BothWrong_accuracy']['ci_low']>.20 and crows[0]['rescue_rate']>=.20
    strong=a and b and c and d=='PASS' and ru['image_auroc']>=.75 and win['image_auroc']>=.75
    assert all(sha(p)==HASHES[k] for k,p in paths.items())
    unchanged=not subprocess.check_output(['git','diff',A0,'--','network','tool','train_sshr.py'],cwd=ROOT)
    identity=dict(new_checks=dict(all_input_files_sha_unchanged=True,checkpoint_sha_before=HASHES['checkpoint'],checkpoint_sha_after=sha(paths['checkpoint']),original_sources_unchanged=unchanged,
        model_instantiated=False,network_forward=False,backward=False,optimizer_created=False,optimizer_steps=0,checkpoint_written=False,new_state_bn_prediction_test=False),
        inherited_phase2b19_identity=pid,inherited_identity_sha256=HASHES['previous_identity'],note='Prior state/BN/prediction hashes are inherited evidence, not newly rerun tests. Current process never creates a model.')
    write_json(out/(P+'identity_audit.json'),identity)
    summary=dict(images=n,headroom=head,primary_utility=ru,primary_winner=win,classes={k:grouped[k] for k in ('class0','class1','class2','class3')},interior=grouped['interior'],
        gate_A='PASS' if a else 'FAIL',gate_B='PASS' if b else 'FAIL',gate_C='PASS' if c else 'FAIL',gate_D=d,
        RESIDUAL_THIRD_EVIDENCE_SIGNAL=third,STRONG_RESIDUAL_DEEP_RECOVERY_SIGNAL=strong,decision=decide(a,b,c,d,third),
        third_evidence=crows,prior_phase2b19_decision=prev['decision'],prior_phase2b19_gates={k:prev['gate_'+k] for k in 'ABCDEFG'},
        bootstrap_rng_sha256=rh.hexdigest(),bootstrap_seconds=time.perf_counter()-bt,bootstrap_resamples=10000,quintile_edges=qmeta,
        no_recovery_gate=True,no_training=True,no_test_luad=True,protocol='approved Phase2B1.10 contract; native28 local-gradient diagnostic, not model performance')
    write_json(out/(P+'summary.json'),summary)
    runtime=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),a0=A0,command=shlex.join([sys.executable,*sys.argv]),
        paths=paths,source_sha256=HASHES,contract_sha256=sha(ROOT/'docs/rddr_phase2b110_contract.md'),images=n,replay=replay,
        all_finite=True,model_instantiated=False,network_forward=False,backward=False,optimizer_created=False,optimizer_steps=0,checkpoint_written=False,
        new_recovery_gate=False,threshold_search=False,test_access=False,luad_access=False,training_split_access=False,total_seconds=time.perf_counter()-start,numpy=np.__version__)
    write_json(out/(P+'runtime.json'),runtime)
    print(json.dumps(clean({k:summary[k] for k in ('gate_A','gate_B','gate_C','gate_D','decision','primary_utility','primary_winner','RESIDUAL_THIRD_EVIDENCE_SIGNAL')})),flush=True)

if __name__=='__main__':main()
