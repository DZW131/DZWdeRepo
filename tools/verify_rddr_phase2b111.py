"""Independent CPU/FP64/cache verification; never imports the primary audit implementation."""
import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter

P='rddr_phase2b111_';EPS=1e-8
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def readnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def div(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)
def avg(x):return float(np.mean(x)) if len(x) else np.nan
def auc_ap(x,y):
    y=np.asarray(y,bool);pos=int(y.sum());neg=len(y)-pos
    if not len(y):return np.nan,np.nan
    order=np.argsort(-x,kind='stable');x=x[order];yy=y[order];end=np.r_[np.flatnonzero(np.diff(x))+1,len(x)]-1
    tp=np.r_[0,np.cumsum(yy)[end]];fp=np.r_[0,np.cumsum(~yy)[end]]
    auc=float(np.trapz(tp/pos,fp/neg)) if pos and neg else np.nan
    ap=float(np.sum(np.diff(tp)/pos*div(tp[1:],tp[1:]+fp[1:]))) if pos else np.nan
    return auc,ap
def rank(x,label,mask):
    auc,ap=auc_ap(x[mask],label[mask]);ia=np.full(len(x),np.nan)
    for i in range(len(x)):ia[i]=auc_ap(x[i,mask[i]],label[i,mask[i]])[0]
    pos=int(label[mask].sum());neg=int(mask.sum())-pos
    return dict(auroc=auc,auprc=ap,positive=pos,negative=neg,prevalence=float(div(pos,pos+neg)),
        image_auroc=avg(ia[np.isfinite(ia)]),eligible_images=int(np.isfinite(ia).sum()),images_with_targets=int(mask.any(1).sum()),targets=int(mask.sum())),ia
def softmax(z):
    p=np.exp(z-z.max(-1,keepdims=True));return p/p.sum(-1,keepdims=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();out=Path(args.run)
    target=out/(P+'verification.json')
    if target.exists():raise FileExistsError(target)
    def js(name):return json.loads((out/(P+name+'.json')).read_text())
    def rows(name):
        with (out/(P+name+'.csv')).open(newline='') as f:return list(csv.DictReader(f))
    rt=js('runtime');s=js('summary');checks={};errors={}
    paths=rt['paths'];checks['input_sha_unchanged']=all(sha(path)==rt['source_sha256'][k] for k,path in paths.items())
    data,old,obs=(readnp(paths[k]) for k in ('native','derived','observations'))
    previous=json.loads(Path(paths['previous_summary']).read_text());previous_identity=json.loads(Path(paths['previous_identity']).read_text())
    n=len(data['names']);checks['full3418_order']=n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],obs['names'])
    ps=data['ps'];pd=data['pd'];ctx=old['ctx_sym'];delta=(old['T_DS']+old['T_DD'])*.5-(old['T_SS']+old['T_SD'])*.5
    md=delta>0;u_all=~md;cs=ps.argmax(1);cd=pd.argmax(1);cc=ctx.argmax(1)
    support_ctx=np.take_along_axis(ctx,cc[:,None],1)[:,0];support_cs=np.take_along_axis(ctx,cs[:,None],1)[:,0];support_cd=np.take_along_axis(ctx,cd[:,None],1)[:,0]
    margin=support_ctx-np.maximum(support_cs,support_cd);different=(cc!=cs)&(cc!=cd);active=u_all&different&(margin>0)
    checks['mD_exact_replay']=np.array_equal(md,obs['direction_gate']) and np.array_equal(delta,old['sym'])
    checks['q_exact_replay']=rt['replay']['errors']['q']<=1e-7 and rt['replay']['errors']['raw_probability']<=1e-7
    # Independent separable FP64 box-filter context, not the FP32 unfold implementation.
    ctx_error=0.;valid=np.ones((28,28));den=uniform_filter(valid,15,mode='constant')*225-1
    for i in range(n):
        pp=ps[i].reshape(4,28,28).astype(float);dd=pd[i].reshape(4,28,28).astype(float)
        c1=(uniform_filter(pp,size=(1,15,15),mode='constant')*225-pp)/den
        c2=(uniform_filter(dd,size=(1,15,15),mode='constant')*225-dd)/den
        ctx_error=max(ctx_error,float(np.max(np.abs((c1+c2)*.5-ctx[i].reshape(4,28,28)))))
    errors['FP64_context']=ctx_error
    checks['ctx_exact_and_independent_replay']=ctx_error<1e-6 and all(rt['replay']['errors'][k]==0 for k in ('ctx_S','ctx_D','ctx_sym','T_SS','T_SD','T_DS','T_DD'))
    # Independent full softmax-Jacobian contraction of dKL/dp. No autograd.
    p=ps.astype(float);t=ctx.astype(float);dKdp=-t/(p+EPS);g=np.zeros_like(p)
    for k in range(4):
        for j in range(4):g[:,k]+=dKdp[:,j]*p[:,j]*((1. if j==k else 0.)-p[:,k])
    g*=active[:,None];v=-g;truth=data['truth'];fg=(truth>=0)&(truth<4)
    raw=obs['raw_logits'];gt=np.take_along_axis(v,truth.clip(0,3)[:,None],1)[:,0]
    non_gt=np.where(np.arange(4)[None,:,None]==truth[:,None],-np.inf,raw);ties=non_gt==non_gt.max(1,keepdims=True)
    dm=gt-np.max(np.where(ties,v,-np.inf),axis=1)
    # Formula regrouping may round near zero: reconstruct primary arithmetic as a comparator,
    # keep exact signs checked; never relabel with an epsilon.
    ratio=t*p/(p+EPS);g_formula=(p*ratio.sum(1,keepdims=True)-ratio)*active[:,None]
    v0=-g_formula;dm0=np.take_along_axis(v0,truth.clip(0,3)[:,None],1)[:,0]-np.max(np.where(ties,v0,-np.inf),axis=1)
    errors['jacobian_vs_formula']=float(np.abs(g-g_formula).max());checks['context_kl_gradient_formula']=errors['jacobian_vs_formula']<1e-12
    checks['gradient_labels_exact']=np.array_equal(np.sign(dm[fg&active]),np.sign(dm0[fg&active]))
    # Fixed128 evenly spaced candidate locations, every channel; FP64 finite differences.
    ids=np.flatnonzero(active.ravel());sample=ids[np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int)]
    z=raw.transpose(0,2,1).reshape(-1,4)[sample].astype(float);tt=t.transpose(0,2,1).reshape(-1,4)[sample]
    original=g_formula.transpose(0,2,1).reshape(-1,4)[sample];step=1e-5;num=np.zeros_like(z)
    for k in range(4):
        plus=z.copy();minus=z.copy();plus[:,k]+=step;minus[:,k]-=step
        lp=(tt*(np.log(tt+EPS)-np.log(softmax(plus)+EPS))).sum(1);lm=(tt*(np.log(tt+EPS)-np.log(softmax(minus)+EPS))).sum(1)
        num[:,k]=(lp-lm)/(2*step)
    errors['real128_finite_difference']=float(np.abs(num-original).max());checks['real128_finite_difference']=errors['real128_finite_difference']<1e-6
    u=u_all&fg;a=active&fg;sc=cs==truth;dc=cd==truth;correct=cc==truth
    bw=u&~sc&~dc;dw=u&~sc&dc;sw=u&sc&~dc;stable=u&sc&dc;rc=u&sc;rw=fg&~sc
    rescue=a&correct;fail=a&~correct;repair=a&~sc&correct;harm=a&sc&~correct;net=repair.astype(int)-harm.astype(int)
    ben=a&(dm>0);bad=a&(dm<0);zero=a&(dm==0)
    checks['all_tensors_finite']=all(np.isfinite(x).all() for x in (ps,pd,ctx,delta,margin,g,dm))
    checks['inactive_zero']=not g.transpose(0,2,1)[~active].any() and not dm[~active].any()
    checks['phase2b110_counts_exact_replay']=[int(x.sum()) for x in (rw,u&~sc,dw,bw,sw)]==[708407,435185,113204,321981,144662]
    # Keep complete previous hard-proposal counts, separately checking strict-margin ties.
    checks['legacy_third_rescue']=all(int((mask&different&correct).sum())==ref['correct_third_class'] and
        int((mask&different&~correct).sum())==ref['wrong_third_class'] for mask,ref in zip((bw,dw,sw),previous['third_evidence']))
    err=0.
    def compare(got,expected):
        nonlocal err
        for k,val in expected.items():
            if isinstance(val,str):assert got[k]==val,(k,got[k],val);continue
            v=float(got[k]) if got[k] is not None and got[k]!='' else np.nan
            if not np.isfinite(val):assert not np.isfinite(v),(k,v,val)
            else:assert np.isfinite(v);err=max(err,abs(v-val));assert abs(v-val)<1e-10,(k,v,val)
    for r in rows('candidate_counts'):
        pop={'all_native':np.ones_like(a),'foreground':fg,'background4':truth==4,'ignore255':truth==255}[r['population']]
        uu=pop&u_all;aa=pop&active
        expected=dict(total=pop.sum(),universe=uu.sum(),candidate=aa.sum(),candidate_rate_in_universe=float(div(aa.sum(),uu.sum())),
            legacy_alternative=(uu&different).sum(),strict_zero_rejected=(uu&different&(margin==0)).sum(),
            raw_argmax_tie=(pop&((ps==ps.max(1,keepdims=True)).sum(1)>1)).sum(),deep_argmax_tie=(pop&((pd==pd.max(1,keepdims=True)).sum(1)>1)).sum(),
            ctx_argmax_tie=(pop&((ctx==ctx.max(1,keepdims=True)).sum(1)>1)).sum())
        if r['population']=='foreground':expected.update(rescue=rescue.sum(),failure=fail.sum())
        compare(r,expected)
    checks['candidate_counts_and_ties']=True
    for r in rows('legacy_replay'):
        pop={'BothWrong':bw,'DeepWin':dw,'ShallowWin':sw}[r['group']]
        compare(r,dict(old_alternative=(pop&different).sum(),old_rescue=(pop&different&correct).sum(),old_wrong=(pop&different&~correct).sum(),
            strict_alternative=(pop&active).sum(),strict_rescue=(pop&rescue).sum(),rejected_zero_margin=(pop&different&(margin==0)).sum()))
    cm={'BothWrong_ctx_correct':bw&rescue,'BothWrong_ctx_wrong':bw&fail,'DeepWin_intrusion':dw&a,'ShallowWin_intrusion':sw&a,'StableCorrect_intrusion':stable&a,'other':np.zeros_like(a)}
    for r in rows('candidate_composition'):compare(r,dict(count=cm[r['group']].sum(),fraction=float(div(cm[r['group']].sum(),a.sum()))))
    checks['candidate_composition_partition']=sum(int(x.sum()) for x in cm.values())==int(a.sum())
    final=np.where(active,cc,cs)
    for r in rows('hard_effect'):
        pop=u if r['population']=='U_R' else fg
        compare(r,dict(count=pop.sum(),candidate=(pop&a).sum(),Repair=(pop&repair).sum(),Harm=(pop&harm).sum(),
            WrongToWrong_activated=(pop&a&~sc&~correct).sum(),StableCorrect_activated=(pop&a&sc&correct).sum(),
            WrongToWrong_full=(pop&~sc&(final!=truth)).sum(),StableCorrect_full=(pop&sc&(final==truth)).sum(),NetRepair=net[pop].sum(),
            raw_accuracy=float(sc[pop].mean()),diagnostic_accuracy=float((final[pop]==truth[pop]).mean()),hard_accuracy_delta=float(net[pop].mean())))
    checks['hard_repair_harm_formula']=True
    scores={'M_alt':margin,'C_ctx':ctx.max(1),'E_ctx':(t*np.log(t+EPS)).sum(1),'q':data['q_feature'],'Delta_sym':delta,'D_hier':1-np.maximum(ps.max(1),pd.max(1))}
    edges=[.020935675129294395,.072734534740448,.163648784160614,.3369627296924591];qbin=np.searchsorted(edges,data['q_feature'],side='left')
    strata={'all':np.ones_like(a),'Top20':data['top20'].astype(bool),'Bottom80':~data['top20'].astype(bool),**{f'Q{k+1}':qbin==k for k in range(5)},
        **{f'class{k}':truth==k for k in range(4)},'boundary':data['boundary'].astype(bool),'interior':~data['boundary'].astype(bool)}
    label={'rescue':rescue,'bothwrong':bw,'gradient':ben};eligible={'rescue':a,'bothwrong':u,'gradient':ben|bad};ranked={};iaucs={}
    for file in ('rescue_ranking','bothwrong_detection','gradient_ranking','score_controls'):
        for r in rows(file):
            key=(r['task'],r['group'],r['score']);task,group,score=key
            if key not in ranked:
                val=np.where(active,margin,0) if task=='bothwrong' and score=='M_alt' else scores[score]
                ranked[key],iaucs[key]=rank(val,label[task],eligible[task]&strata[group])
            compare(r,ranked[key]);compare(r,dict(zero_excluded=(zero&strata[group]).sum() if task=='gradient' else 0))
    checks['all_three_task_rankings']=True
    checks['controls_frozen_no_substitution']=all({r['score'] for r in rows('score_controls') if r['task']==task}==set(scores) for task in label)
    def ustats(pop):
        x=dm[pop]
        return dict(count=pop.sum(),beneficial=(x>0).sum(),harmful=(x<0).sum(),zero=(x==0).sum(),benefit_rate=float(div((x>0).sum(),len(x))),
            harm_rate=float(div((x<0).sum(),len(x))),zero_rate=float(div((x==0).sum(),len(x))),mean_dm=avg(x),median_dm=float(np.median(x)) if len(x) else np.nan)
    gm=dict(strata,ThirdRescue=rescue,AlternativeFailure=fail,BothWrong=bw,DeepWin_intrusion=dw,ShallowWin_intrusion=sw,RawCorrect=rc,RawWrong=u&~sc,StableCorrect_intrusion=stable)
    for r in rows('context_gradient'):compare(r,ustats(a&gm[r['group']]))
    prot={};populations={'RawCorrect':rc,'DeepWin':dw,'ShallowWin':sw,'BothWrong':bw,'StableCorrect':stable}
    for file in ('rawcorrect_protection','deepwin_protection','shallowwin_protection','bothwrong_rescue','stablecorrect_intrusion'):
        for r in rows(file):
            pop=populations[r['group']];pop=pop&a if r['denominator']=='active_only' else pop;on=pop&a
            expected=dict(population_count=pop.sum(),candidate=on.sum(),activation_rate=float(div(on.sum(),pop.sum())),hard_harm_count=(pop&harm).sum(),
                hard_harm_rate=float(div((pop&harm).sum(),pop.sum())),third_intrusion_rate=float(div(on.sum(),pop.sum())),rescue_count=(pop&rescue).sum(),
                rescue_rate=float(div((pop&rescue).sum(),pop.sum())),candidate_precision=float(div((pop&rescue).sum(),on.sum())),unconditional_ctx_accuracy=float(div((pop&correct).sum(),pop.sum())),**ustats(pop))
            compare(r,expected)
            if r['denominator']=='full_residual_group':prot[r['group']]=expected
    checks['gradient_and_full_active_protection']=True
    grouped={}
    for file in ('q_strata','per_class','boundary_interior'):
        for r in rows(file):
            group=r['group'];st=strata[group];on=a&st;sub=u&st;rs=rc&st;rankr=ranked['rescue',group,'M_alt'];gr=ranked['gradient',group,'M_alt']
            power=lambda v:'POWERED' if v['positive']>=500 and v['negative']>=500 and v['eligible_images']>=30 else 'UNDERPOWERED'
            expected=dict(universe=sub.sum(),candidate=on.sum(),candidate_rate=float(div(on.sum(),sub.sum())),rescue=(on&rescue).sum(),failure=(on&fail).sum(),
                precision=float(div((on&rescue).sum(),on.sum())),image_auroc=rankr['image_auroc'],pooled_auroc=rankr['auroc'],auprc=rankr['auprc'],eligible_images=rankr['eligible_images'],
                power=power(rankr),gradient_image_auroc=gr['image_auroc'],gradient_power=power(gr),rawcorrect=rs.sum(),rawcorrect_hard_harm=float(div((rs&harm).sum(),rs.sum())),
                rawcorrect_gradient_harm=float(div((rs&(dm<0)).sum(),rs.sum())),**ustats(on))
            compare(r,expected);grouped[group]=expected
    checks['q_strata_replay']=np.array_equal(edges,s['q_edges'])
    checks['boundary_replay']=np.array_equal(strata['boundary'],data['boundary'].astype(bool))
    checks['per_class_power']=True;errors['all_tables']=err
    stat=readnp(out/(P+'image_statistics.npz'));names=list(stat['names']);terms={}
    def term(name,value,pop):terms[name]=(np.where(pop,value,0).sum(1,dtype=float),pop.sum(1))
    term('CandidatePrecision',rescue,a);term('CandidateRate',a,u);term('ThirdRescue_count_equivalent',rescue.astype(float)*708407,rw)
    term('ThirdRescue_to_gap',rescue.astype(float)*708407/31266,rw);term('Hard_NetRepair_count_equivalent',net*int(u.sum()),u)
    term('Hard_accuracy_delta_UR',net,u);term('Hard_accuracy_delta_foreground',net,fg);term('RawCorrect_hard_HarmRate',harm,rc);term('RawCorrect_gradient_HarmRate',dm<0,rc)
    term('Candidate_gradient_BenefitRate',dm>0,a);term('Candidate_gradient_HarmRate',dm<0,a);term('Candidate_gradient_Mean_dM',dm,a)
    term('BW_prevalence_candidate',bw,a);term('BW_prevalence_UR',bw,u);term('DeepWin_intrusion',a,dw);term('ShallowWin_intrusion',a,sw)
    for key,ia in iaucs.items():terms[':'.join(key)+':image_AUROC']=(np.nan_to_num(ia),np.isfinite(ia).astype(int))
    assert set(names)==set(terms)
    nn=np.stack([terms[k][0] for k in names],1);dd=np.stack([terms[k][1] for k in names],1)
    errors['bootstrap_image_terms']=max(float(np.abs(nn-stat['numerators']).max()),float(np.abs(dd-stat['denominators']).max()))
    # Alternative per-draw gather summation, no primary bincount matrix path.
    rng=np.random.default_rng(42);reps=[];rh=hashlib.sha256()
    for _ in range(200):
        ix=rng.integers(0,n,(50,n),dtype=np.int32);rh.update(ix.tobytes());reps.append(div(nn[ix].sum(1),dd[ix].sum(1)))
    reps=np.concatenate(reps);original=np.genfromtxt(out/(P+'bootstrap_replicates.csv'),delimiter=',',skip_header=1)
    errors['bootstrap_replicates']=float(np.nanmax(np.abs(reps-original)))
    checks['bootstrap_reproducible']=rh.hexdigest()==s['bootstrap_rng_sha256'] and np.array_equal(np.isfinite(reps),np.isfinite(original)) and errors['bootstrap_replicates']<1e-8
    ci={}
    for r in rows('bootstrap'):
        j=names.index(r['metric']);vals=reps[:,j];ok=np.isfinite(vals);lo,hi=np.quantile(vals[ok],[.025,.975]) if ok.any() else (np.nan,np.nan)
        expected=dict(estimate=float(div(nn[:,j].sum(),dd[:,j].sum())),ci_low=lo,ci_high=hi,resamples=10000,valid_resamples=ok.sum(),seed=42)
        compare(r,expected);ci[r['metric']]=expected
    checks['bootstrap_denominators_and_CIs']=True
    precision=float(div(rescue.sum(),a.sum()));bwa=float(div((bw&a).sum(),a.sum()));bwu=float(div(bw.sum(),u.sum()));gu=ustats(a)
    head=dict(RequiredGap=31266,RawWrong=708407,ThirdRescueCount=rescue.sum(),rescue_to_gap=float(rescue.sum()/31266),
        count_equivalent_ci_low=ci['ThirdRescue_count_equivalent']['ci_low'],count_equivalent_ci_high=ci['ThirdRescue_count_equivalent']['ci_high'],
        CandidatePrecision=precision,BW_prevalence_candidate=bwa,BW_prevalence_UR=bwu)
    compare(rows('coverage_headroom')[0],head);compare(s['headroom'],head);compare(s['candidate_gradient'],gu)
    for task in label:compare(s['primary'][task],ranked[task,'all','M_alt'])
    checks['summary_headroom_and_primary']=True
    ra=ranked['rescue','all','M_alt']['image_auroc'];ba=ranked['bothwrong','all','M_alt']['image_auroc'];gra=ranked['gradient','all','M_alt']['image_auroc']
    A=rescue.sum()>=31266 and ci['ThirdRescue_count_equivalent']['ci_low']>=31266 and precision>=.65 and ci['CandidatePrecision']['ci_low']>.55
    B=ra>=.65 and ci['rescue:all:M_alt:image_AUROC']['ci_low']>.50
    C=ba>=.65 and ci['bothwrong:all:M_alt:image_AUROC']['ci_low']>.50 and bwa>bwu
    D=gu['benefit_rate']>gu['harm_rate'] and gu['mean_dm']>0 and gra>=.60
    E=prot['RawCorrect']['hard_harm_rate']<=.08 and prot['RawCorrect']['harm_rate']<=.15 and prot['DeepWin']['activation_rate']<=.12 and prot['ShallowWin']['activation_rate']<=.10
    interior=grouped['interior']['image_auroc'];classes=[grouped[f'class{k}'] for k in range(4)]
    good=sum(c['power']=='POWERED' and c['image_auroc']>.55 for c in classes);missing=sum(c['power']=='UNDERPOWERED' for c in classes)
    F='PASS' if interior>.60 and good>=3 else 'UNDERPOWERED' if (not np.isfinite(interior) or interior>.60) and good+missing>=3 else 'FAIL'
    decision='THIRD_EVIDENCE_OPERATIONAL_HEADROOM_INSUFFICIENT' if not A else 'THIRD_EVIDENCE_EXISTS_BUT_NOT_SELECTABLE' if not B or not C else 'THIRD_EVIDENCE_HARD_RESCUE_BUT_GRADIENT_UNSAFE' if not D else 'THIRD_EVIDENCE_SIGNAL_WITH_PROTECTION_FAILURE' if not E else None if F=='UNDERPOWERED' else 'THIRD_EVIDENCE_SIGNAL_NOT_ROBUST' if F=='FAIL' else 'THIRD_EVIDENCE_GTBLIND_FEASIBILITY_SUPPORTED'
    checks['independent_gates_decision']=all(s['gate_'+k]==('PASS' if v else 'FAIL') for k,v in zip('ABCDE',(A,B,C,D,E))) and s['gate_F']==F and s['decision']==decision
    strong=bool(all((A,B,C,D,E,F=='PASS')) and precision>=.75 and ra>=.75 and gra>=.70 and prot['RawCorrect']['hard_harm_rate']<=.05 and rescue.sum()>=62532)
    checks['secondary_flags']=bool(ranked['rescue','all','C_ctx']['image_auroc']>ra)==s['CONTEXT_CONFIDENCE_DIAGNOSTIC_STRONGER'] and strong==s['STRONG_THIRD_EVIDENCE_SIGNAL']
    ident=js('identity_audit');checks['identity_unchanged']=ident['inherited_phase110_identity']==previous_identity and not ident['new_checks']['new_model_state_bn_prediction_test'] and all(sha(path)==rt['source_sha256'][k] for k,path in paths.items())
    flags=('model_instantiated','network_forward','backward','autograd','optimizer_created','optimizer_steps','checkpoint_written','threshold_search','score_fusion','classifier_fit','test_access','luad_access','training_split_access','new_gate_design')
    checks['no_forbidden_operations']=not any(rt[k] for k in flags)
    checks['original_A0_unchanged']=not subprocess.check_output(['git','diff',rt['a0'],'--','network','tool','train_sshr.py'],cwd=Path(__file__).resolve().parents[1])
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},errors=errors,images=n,resamples=10000,decision=decision,
        finite_difference_candidates=len(sample),method='independent context box-filter, softmax Jacobian, finite difference, tied ROC/AP, table arithmetic, direct-gather bootstrap and gates',
        command=shlex.join([sys.executable,*sys.argv]),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
