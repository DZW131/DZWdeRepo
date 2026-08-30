"""Frozen native28 PRG statistics, hierarchy safety and paired image bootstrap."""
import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b16_common import sha256,write_json,write_csv,scores,margin_direction,bootstrap_indices,clean
from tools.rddr_phase2b18_common import PREFIX as P,MODES,groups,hierarchy,semantic_metrics,decide


def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def divide(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)


def describe(a):
    a=np.asarray(a,dtype=float)
    if not len(a):return dict(mean=np.nan,median=np.nan,positive_fraction=np.nan,negative_fraction=np.nan,zero_fraction=np.nan)
    return dict(mean=a.mean(),median=np.median(a),positive_fraction=(a>0).mean(),negative_fraction=(a<0).mean(),zero_fraction=(a==0).mean())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    rt=json.loads((run/(P+'runtime.json')).read_text());assert sha256(run/(P+'observations.npz'))==rt['observation_sha256']
    data=loadnp(rt['paths']['native']);old=loadnp(rt['paths']['derived']);obs=loadnp(run/(P+'observations.npz'))
    assert np.array_equal(data['names'],obs['names']) and len(obs['names'])==3418
    out.mkdir(parents=True)
    def wc(name,r):write_csv(out/(P+name+'.csv'),r)
    y=data['truth'];n=len(y);mask=groups(data);fg=mask['all'];raw=data['ps'].argmax(1);deep=data['pd'].argmax(1);teacher=old['anchor_sym'].argmax(1)
    rawok=raw==y;teachok=teacher==y;repair=~rawok&teachok;harm=rawok&~teachok;net=repair.astype(np.int64)-harm
    probabilities=(data['ps'],data['fixed_average'],old['anchor_sym'])
    semantics=[dict(model=name,**semantic_metrics(prob,y)) for name,prob in zip(('raw','FixedAvg','teacher'),probabilities)]
    wc('teacher_raw_metrics',semantics)
    original=json.loads((run/(P+'identity_audit.json')).read_text());detach=json.loads((run/(P+'detach_audit.json')).read_text());smoke=json.loads((run/(P+'bf16_smoke.json')).read_text())
    dm={};dq={};cos={};gnorm={}
    for j,mode in enumerate(MODES):
        grad=obs['gradients'][:,j].astype(np.float64);dm[mode],ties=margin_direction(obs['raw_logits'],grad,y)
        dq[mode],cos[mode]=hierarchy(obs['q_gradients'],grad);gnorm[mode]=np.linalg.norm(grad,axis=1)
    assert all(np.isfinite(v).all() for d in (dm,dq,cos,gnorm) for v in d.values())
    py=np.clip(y,0,3)[:,None]
    adv=(np.take_along_axis(old['anchor_sym'],py,1)-np.take_along_axis(data['ps'],py,1))[:,0].astype(np.float64)
    transition=[];advrows=[];gradrows=[];hierrows=[];feature=[];reconcile=[];utility={};trmap={};hmap={}
    for name,m in mask.items():
        count=int(m.sum());den=max(count,1)
        tr=dict(stratum=name,targets=count,images=int(m.any(1).sum()),raw_accuracy=float(rawok[m].mean()) if count else np.nan,
                teacher_accuracy=float(teachok[m].mean()) if count else np.nan,repair=int((m&repair).sum()),harm=int((m&harm).sum()),
                net_repair=int(net[m].sum()),net_repair_rate=float(net[m].mean()) if count else np.nan)
        transition.append(tr);trmap[name]=tr;advrows.append(dict(stratum=name,targets=count,**describe(adv[m])))
        for mode in MODES:
            desc=describe(dm[mode][m]);r=dict(stratum=name,loss=mode,targets=count,benefit_rate=desc['positive_fraction'],harm_rate=desc['negative_fraction'],
                zero_rate=desc['zero_fraction'],mean_dm=desc['mean'],median_dm=desc['median'],mean_gradient_norm=gnorm[mode][m].mean() if count else np.nan,
                active_fraction=(gnorm[mode][m]>0).mean() if count else np.nan)
            gradrows.append(r);utility[mode,name]=r
            hr=dict(stratum=name,loss=mode,targets=count,**{'dQ_'+k:v for k,v in describe(dq[mode][m]).items()},
                    **{'CosCollapse_'+k:v for k,v in describe(cos[mode][m]).items()})
            hierrows.append(hr);hmap[mode,name]=hr
        dw=m&mask['Deep-Win'];sw=m&mask['Shallow-Win']
        rr=dict(stratum=name,Deep_Win_targets=int(dw.sum()),Shallow_Win_targets=int(sw.sum()),
                BRR=float(divide((dw&(dm['PRG']>0)&(dq['PRG']<0)).sum(),dw.sum())),
                HHCR=float(divide((sw&(dm['PRG']<0)&(dq['PRG']<0)).sum(),sw.sum())))
        reconcile.append(rr)
        if name in ('all','Top20','Bottom80','Q1','Q2','Q3','Q4','Q5','class0','class1','class2','class3'):
            en=obs['feature_sqsum'][m]
            feature.append(dict(stratum=name,targets=count,channels=512,rms=float(np.sqrt(en.sum()/(den*512))) if count else np.nan,
                                mean_pixel_l2=float(np.sqrt(en).mean()) if count else np.nan,max_abs=float(obs['feature_maxabs'][m].max()) if count else np.nan,
                                finite=bool(np.isfinite(en).all())))
    wc('teacher_raw_transition',transition);wc('teacher_advantage',advrows);wc('raw_gradient',gradrows);wc('hierarchy_direction',hierrows)
    wc('reconciliation_collapse',reconcile);wc('feature_gradient',feature)
    def combined(names):
        result=[]
        for name in names:
            result.append(dict(trmap[name],**{k:v for k,v in utility['PRG',name].items() if k not in ('stratum','targets','loss')},
                               mean_dQ=hmap['PRG',name]['dQ_mean'],**{k:v for k,v in next(r for r in reconcile if r['stratum']==name).items() if k!='stratum'}))
        return result
    wc('correct_wrong_safety',combined(['Raw_Correct','Raw_Wrong']))
    wc('deep_shallow_win',combined(['Deep-Win','Shallow-Win','Both-Wrong','Stable-Correct']))
    wc('q_quintiles',combined([f'Q{i}' for i in range(1,6)]));wc('per_class',combined([f'class{i}' for i in range(4)]))
    wc('boundary_interior',combined(['boundary','interior']))
    localization=[]
    for mode in MODES:
        top=utility[mode,'Top20']['mean_gradient_norm'];bottom=utility[mode,'Bottom80']['mean_gradient_norm']
        localization.append(dict(loss=mode,Top20_mean_G=top,Bottom80_mean_G=bottom,ratio=float(divide(top,bottom))))
    wc('gradient_localization',localization)
    parameters=[];pn=obs['parameter_names'].tolist();pe=obs['parameter_energy'];pm=obs['parameter_maxabs'];numel=obs['parameter_numel']
    for j,mode in enumerate(('PRG_frozen','PRG_sharedhead')):
        for k,name in enumerate(pn):
            parameters.append(dict(loss=mode,parameter=name,images=n,numel=int(numel[k]),total_squared_energy=pe[:,j,k].sum(),
                                   rms=np.sqrt(pe[:,j,k].sum()/(n*numel[k])),max_abs=pm[:,j,k].max(),nonzero_images=int((pe[:,j,k]>0).sum()),
                                   zero_images=int((pe[:,j,k]==0).sum()),finite=bool(np.isfinite(pe[:,j,k]).all() and np.isfinite(pm[:,j,k]).all())))
    wc('parameter_gradient',parameters)
    he=pe[:,1,-2:].sum(1);ue=pe[:,1,:-2].sum(1);shares=divide(he,he+ue)
    head_fraction=float(divide(he.sum(),he.sum()+ue.sum()))
    shared=[]
    for j,mode in enumerate(('PRG_frozen','PRG_sharedhead')):
        fe=obs['feature_sqsum'].sum() if j==0 else obs['shared_feature_energy'].sum()
        ee=pe[:,j];head=ee[:,-2:].sum();up=ee[:,:-2].sum()
        shared.append(dict(mode=mode,images=n,feature_squared_energy=fe,feature_l2=np.sqrt(fe),ic1_squared_energy=head,ic1_l2=np.sqrt(head),
                           upstream_squared_energy=up,upstream_l2=np.sqrt(up),head_parameter_energy_fraction=float(divide(head,head+up)),
                           head_fraction_per_image_mean=0. if j==0 else np.nanmean(shares),head_fraction_per_image_median=0. if j==0 else np.nanmedian(shares),
                           denominator='ic1 + approved upstream PARAMETERS only; features excluded'))
    wc('shared_head_diagnostic',shared)
    wc('per_image_losses',[dict(image_id=str(name),**{mode:float(obs['losses'][i,j]) for j,mode in enumerate((*MODES,'PRG_sharedhead'))}) for i,name in enumerate(obs['names'])])
    # Per-image sufficient statistics, paired bootstrap with unchanged image draws.
    image=np.broadcast_to(np.arange(n)[:,None],y.shape)
    cms=np.stack([np.bincount((image*16+y*4+p)[fg],minlength=n*16).reshape(n,4,4) for p in (raw,teacher)],axis=1)
    names=[];nums=[];dens=[];kinds=[]
    def add(name,values,m,kind='ratio'):
        names.append(name);nums.append(np.where(m,values,0).sum(1,dtype=np.float64));dens.append(m.sum(1));kinds.append(kind)
    add('teacher-raw_accuracy_delta',net,fg);add('teacher_NetRepair_count',net,fg,'sum');add('teacher_NetRepair_rate',net,fg)
    for name in ('all','Top20','class0','class1','class2','class3'):add('PRG:'+name+':mean_dm',dm['PRG'],mask[name])
    add('PRG:Raw_Correct:harm_rate',dm['PRG']<0,mask['Raw_Correct'])
    add('PRG:Raw_Wrong:benefit_rate',dm['PRG']>0,mask['Raw_Wrong'])
    add('Deep-Win:BRR',(dm['PRG']>0)&(dq['PRG']<0),mask['Deep-Win'])
    add('Shallow-Win:HHCR',(dm['PRG']<0)&(dq['PRG']<0),mask['Shallow-Win'])
    add('Shallow-Win:teacher_accuracy',teachok,mask['Shallow-Win'])
    add('PRG:all:benefit_rate',dm['PRG']>0,fg);add('PRG:all:harm_rate',dm['PRG']<0,fg)
    add('PRG:Top20:benefit_rate',dm['PRG']>0,mask['Top20']);add('PRG:Top20:harm_rate',dm['PRG']<0,mask['Top20'])
    nums=np.stack(nums,1);dens=np.stack(dens,1);sumkind=np.array(kinds)=='sum'
    base=divide(nums.sum(0),dens.sum(0));base[sumkind]=nums.sum(0)[sumkind]
    ci_values=[];rnghash=hashlib.sha256()
    for indices in bootstrap_indices(n):
        rnghash.update(indices.tobytes());mult=np.zeros((len(indices),n),np.int32)
        for i,idx in enumerate(indices):mult[i]=np.bincount(idx,minlength=n)
        bn=mult@nums;bd=mult@dens;scalar=divide(bn,bd);scalar[:,sumkind]=bn[:,sumkind]
        bc=(mult@cms.reshape(n,-1)).reshape(len(indices),2,4,4);ss=scores(bc)
        ci_values.append(np.column_stack((scalar,ss['miou'][:,1]-ss['miou'][:,0])))
    ci_values=np.concatenate(ci_values);names.append('teacher-raw_mIoU_delta')
    base=np.r_[base,semantics[2]['miou']-semantics[0]['miou']]
    ci=[]
    for j,name in enumerate(names):
        v=ci_values[:,j];valid=np.isfinite(v);lo,hi=np.quantile(v[valid],[.025,.975])
        ci.append(dict(metric=name,estimate=base[j],ci_low=lo,ci_high=hi,resamples=10000,valid_resamples=int(valid.sum()),seed=42))
    wc('bootstrap',ci);wc('bootstrap_replicates',[dict(zip(names,row)) for row in ci_values])
    np.savez_compressed(out/(P+'image_statistics.npz'),names=np.array(names[:-1]),numerators=nums,denominators=dens,sumkind=sumkind,confusion=cms)
    low={r['metric']:r['ci_low'] for r in ci};pr={name:utility['PRG',name] for name in mask}
    rec=reconcile[0];positive=sum(pr[x]['mean_dm']>0 for x in ('all','Top20','class0','class1','class2','class3'))
    a=semantics[2]['accuracy']>semantics[0]['accuracy'] and semantics[2]['miou']>semantics[0]['miou'] and trmap['all']['net_repair']>0 and (low['teacher-raw_accuracy_delta']>0 or low['teacher-raw_mIoU_delta']>0)
    b=all(pr[x]['benefit_rate']>pr[x]['harm_rate'] for x in ('all','Top20')) and positive>=5
    c=pr['Raw_Wrong']['benefit_rate']>=.70 and pr['Raw_Correct']['harm_rate']<=.50 and trmap['all']['net_repair']>0
    d=rec['BRR']>=.60 and rec['HHCR']<=.30 and trmap['Shallow-Win']['teacher_accuracy']>=.60 and pr['Q5']['mean_gradient_norm']>pr['Q1']['mean_gradient_norm']
    e=all((rt['all_finite'],not rt['optimizer_created'],rt['optimizer_steps']==0,not rt['test_access'],smoke['pass'],original['state_unchanged'],original['prediction_unchanged'],
           detach['teacher_detached'],detach['q_detached'],detach['deep_source_detached'],detach['primary_ic1_none'],detach['hfrm_none'],detach['upstream_conv_nonzero']))
    strong=(semantics[2]['miou']-semantics[0]['miou']>=.10 and pr['all']['mean_dm']>0 and pr['Top20']['mean_dm']>0 and
            pr['Raw_Wrong']['benefit_rate']>=.80 and pr['Raw_Correct']['harm_rate']<=.35 and rec['BRR']>=.70 and rec['HHCR']<=.20)
    summary=dict(images=n,foreground_targets=int(fg.sum()),semantics=semantics,teacher_raw_transition=trmap['all'],primary=pr,
                 hierarchy={name:hmap['PRG',name] for name in mask},BRR=rec['BRR'],HHCR=rec['HHCR'],Shallow_Win_teacher_accuracy=trmap['Shallow-Win']['teacher_accuracy'],
                 gate_A='PASS' if a else 'FAIL',gate_B='PASS' if b else 'FAIL',gate_C='PASS' if c else 'FAIL',gate_D='PASS' if d else 'FAIL',gate_E='PASS' if e else 'FAIL',
                 positive_mean_dm_strata=positive,decision=decide(a,b,c,d,e),CONFLICT_LOCALIZATION_CONFIRMED=localization[2]['ratio']>localization[0]['ratio'],
                 SHARED_HEAD_ABSORPTION_RISK=head_fraction>.50,shared_head_parameter_energy_fraction=head_fraction,STRONG_PRERECT_GUIDANCE_SIGNAL=strong,
                 localization=localization,raw_tied_competitor_pixels=int((fg&ties).sum()),gradient_zero_pixels=int((fg&(dm['PRG']==0)).sum()),
                 bootstrap_rng_sha256=rnghash.hexdigest(),bootstrap_resamples=10000,optimizer_steps=0,test_access=False,full25_started=False,
                 analysis_seconds=time.perf_counter()-start,run=str(run),command=shlex.join([sys.executable,*sys.argv]))
    write_json(out/(P+'summary.json'),summary)
    print(json.dumps(clean({k:summary[k] for k in ('decision','gate_A','gate_B','gate_C','gate_D','gate_E','BRR','HHCR','shared_head_parameter_energy_fraction')})),flush=True)


if __name__=='__main__':main()
