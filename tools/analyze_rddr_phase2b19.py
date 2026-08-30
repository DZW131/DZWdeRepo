"""Frozen directional-transfer strata, hierarchy safety, paired image-bootstrap gates."""
import argparse
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b16_common import sha256,write_json,write_csv,margin_direction,bootstrap_indices,clean
from tools.rddr_phase2b19_common import PREFIX as P,MODES,groups,hierarchy,decide

def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def div(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)

def auc(score,label):
    label=np.asarray(label,bool);pos=label.sum();neg=len(label)-pos
    if not pos or not neg:return np.nan
    order=np.argsort(score,kind='stable');x=np.asarray(score)[order];y=label[order]
    starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1];ends=np.r_[starts[1:],len(x)]
    rp=np.add.reduceat(y.astype(float),starts)
    return float((np.sum(rp*(starts+ends+1)/2)-pos*(pos+1)/2)/(pos*neg))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    rt=json.loads((run/(P+'runtime.json')).read_text());assert sha256(run/(P+'observations.npz'))==rt['observation_sha256']
    data=loadnp(rt['paths']['native']);old=loadnp(rt['paths']['derived']);prev=loadnp(rt['paths']['previous']);obs=loadnp(run/(P+'observations.npz'))
    out.mkdir(parents=True)
    def wc(name,rows):write_csv(out/(P+name+'.csv'),rows)
    y=data['truth'];n=len(y);mask=groups(data);fg=mask['all'];dw=mask['Deep-Win'];sw=mask['Shallow-Win'];gate=obs['direction_gate'];rand=obs['random_gate']
    raw=data['ps'].argmax(1);deep=data['pd'].argmax(1);sc=raw==y;dc=deep==y;delta=old['sym'];conf=dw|sw
    modes=(*MODES,'PRG_previous');dm={};dq={};norm={};active={};util={};recs={};adjudication=[];amap={};auc_images={}
    for j,mode in enumerate(modes):
        grad=(prev['gradients'][:,2] if mode=='PRG_previous' else obs['gradients'][:,j]).astype(float)
        dm[mode],ties=margin_direction(obs['raw_logits'],grad,y);dq[mode],_=hierarchy(obs['q_gradients'],grad);norm[mode]=np.linalg.norm(grad,axis=1)
        active[mode]=np.ones_like(gate) if mode in ('UDT','PRG_previous') else rand if mode=='RG' else gate
    assert np.all(dm['ADT'][~gate]==0) and np.all(dq['ADT'][~gate]==0)
    for name,m in mask.items():
        a=m&dw;b=m&sw;both=m&conf
        ia=np.array([auc(delta[i,both[i]],a[i,both[i]]) for i in range(n)])
        capture=float(div((a&gate).sum(),a.sum()));protect=float(div((b&~gate).sum(),b.sum()))
        r=dict(stratum=name,targets=int(m.sum()),Deep_Win_count=int(a.sum()),Shallow_Win_count=int(b.sum()),
               dual_label_images=int(np.isfinite(ia).sum()),image_auroc=float(np.nanmean(ia)) if np.isfinite(ia).any() else np.nan,
               pooled_auroc=auc(delta[both],a[both]),DeepCaptureRate=capture,ShallowProtectionRate=protect,BA=.5*(capture+protect),
               DeepSelectionPrecision=float(div((a&gate).sum(),(both&gate).sum())),zero_ties=int((both&(delta==0)).sum()),
               power='POWERED' if a.sum()>=500 and b.sum()>=500 and np.isfinite(ia).sum()>=30 else 'UNDERPOWERED')
        adjudication.append(r);amap[name]=r;auc_images[name]=ia
    reference=amap['all']
    assert reference['Deep_Win_count']==314730 and reference['Shallow_Win_count']==182899
    for k,v in dict(image_auroc=.7848415501407355,BA=.7156265907137409,DeepCaptureRate=.6403139198678233,ShallowProtectionRate=.7909392615596587).items():assert abs(reference[k]-v)<1e-12
    wc('adjudication_replay',adjudication)
    gradrows=[];coverage=[];quality=[];recrows=[];feature=[];activeonly=[];randomrows=[]
    for name,m in mask.items():
        count=int(m.sum())
        coverage.append(dict(stratum=name,targets=count,selected=int((m&gate).sum()),rejected=int((m&~gate).sum()),ActiveTransferFraction=float(div((m&gate).sum(),count))))
        for label,g in (('selected',gate),('rejected',~gate)):
            z=m&g;repair=z&~sc&dc;harm=z&sc&~dc
            quality.append(dict(stratum=name,selection=label,targets=int(z.sum()),raw_accuracy=float(div((z&sc).sum(),z.sum())),
                deep_accuracy=float(div((z&dc).sum(),z.sum())),deep_raw_delta=float(div(repair.sum()-harm.sum(),z.sum())),
                repair=int(repair.sum()),harm=int(harm.sum()),net_repair=int(repair.sum()-harm.sum())))
        for mode in modes:
            d=dm[mode][m];qq=dq[mode][m];gg=norm[mode][m]
            r=dict(stratum=name,loss=mode,targets=count,benefit_rate=float((d>0).mean()),harm_rate=float((d<0).mean()),zero_rate=float((d==0).mean()),
                mean_dm=float(d.mean()),median_dm=float(np.median(d)),active_gradient_fraction=float((gg>0).mean()),mean_gradient_norm=float(gg.mean()),
                active_transfer_fraction=float(active[mode][m].mean()),mean_dQ=float(qq.mean()),dQ_negative_fraction=float((qq<0).mean()),dQ_zero_fraction=float((qq==0).mean()))
            util[mode,name]=r;gradrows.append(r)
            a=m&dw;b=m&sw;on=active[mode];bp=on&(dm[mode]>0);hp=on&(dm[mode]<0)
            rr=dict(stratum=name,loss=mode,Deep_Win_targets=int(a.sum()),Shallow_Win_targets=int(b.sum()),
                DBR=float(div((a&bp).sum(),a.sum())),DCR=float(div((b&hp).sum(),b.sum())),
                BRR=float(div((a&bp&(dq[mode]<0)).sum(),a.sum())),HHCR=float(div((b&hp&(dq[mode]<0)).sum(),b.sum())))
            recrows.append(rr);recs[mode,name]=rr
            if mode in MODES:
                z=m&on;dd=dm[mode][z]
                activeonly.append(dict(stratum=name,loss=mode,targets=int(z.sum()),benefit_rate=float((dd>0).mean()) if len(dd) else np.nan,
                    harm_rate=float((dd<0).mean()) if len(dd) else np.nan,mean_dm=float(dd.mean()) if len(dd) else np.nan))
        randomrows.append(dict(stratum=name,targets=count,ADT_rate=float(gate[m].mean()),RG_rate=float(rand[m].mean()),
            rate_difference=float(rand[m].mean()-gate[m].mean()),RG_mean_dm=util['RG',name]['mean_dm'],
            RG_benefit=util['RG',name]['benefit_rate'],RG_harm=util['RG',name]['harm_rate']))
    wc('transfer_coverage',coverage);wc('selected_region_quality',quality);wc('gradient_controls',gradrows);wc('brr_hhcr',recrows);wc('active_only',activeonly)
    randomrows.insert(0,dict(stratum='ALL784_GT_BLIND',targets=gate.size,ADT_rate=float(gate.mean()),RG_rate=float(rand.mean()),rate_difference=float(rand.mean()-gate.mean()),
        per_image_exact_parity=bool(np.array_equal(gate.sum(1),rand.sum(1))),seed=42,mask_sha256=rt['random_gate_sha256']))
    wc('random_gate',randomrows)
    def combined(names):
        return [dict(amap[name],**{k:v for k,v in util['ADT',name].items() if k not in ('stratum','targets')},
                     **{k:v for k,v in recs['ADT',name].items() if k not in ('stratum','loss')},
                     UDT_harm=util['UDT',name]['harm_rate'],RG_harm=util['RG',name]['harm_rate'],UDT_benefit=util['UDT',name]['benefit_rate'],RG_benefit=util['RG',name]['benefit_rate']) for name in names]
    for file,names in [('deepwin',['Deep-Win']),('shallowwin',['Shallow-Win']),('raw_correct_wrong',['Raw_Correct','Raw_Wrong']),
                       ('bothwrong_stablecorrect',['Both-Wrong','Stable-Correct']),('per_class',[f'class{k}' for k in range(4)]),('boundary_interior',['boundary','interior'])]:wc(file,combined(names))
    wc('adjudication_precision',[{k:r[k] for k in ('stratum','DeepSelectionPrecision','DeepCaptureRate','ShallowProtectionRate','Deep_Win_count','Shallow_Win_count')} for r in adjudication])
    grid=[]
    for k in range(1,6):
        for on in (0,1):
            z=mask[f'Q{k}']&(gate==on);d=dm['ADT'][z]
            grid.append(dict(quintile=k,gate=on,targets=int(z.sum()),raw_accuracy=float(sc[z].mean()),deep_accuracy=float(dc[z].mean()),
                deep_raw_delta=float(dc[z].mean()-sc[z].mean()),benefit_rate=float((d>0).mean()),harm_rate=float((d<0).mean()),mean_dm=float(d.mean()),mean_dQ=float(dq['ADT'][z].mean())))
    wc('q_direction_grid',grid)
    loc=[]
    for mode in MODES:
        top=util[mode,'Top20']['mean_gradient_norm'];bottom=util[mode,'Bottom80']['mean_gradient_norm']
        loc.append(dict(loss=mode,Top20_mean_G=top,Bottom80_mean_G=bottom,ratio=float(div(top,bottom)),**{f'Q{k}_mean_G':util[mode,f'Q{k}']['mean_gradient_norm'] for k in range(1,6)}))
    wc('gradient_localization',loc)
    fmask=dict(mask,active=fg&gate,rejected=fg&~gate,all784=np.ones_like(gate))
    for name,m in fmask.items():
        en=obs['feature_sqsum'][m]
        feature.append(dict(stratum=name,targets=int(m.sum()),channels=512,rms=float(np.sqrt(div(en.sum(),m.sum()*512))),
            mean_pixel_l2=float(np.sqrt(en).mean()),max_abs=float(obs['feature_maxabs'][m].max()),finite=bool(np.isfinite(en).all())))
    wc('feature_gradient',feature)
    pe=obs['parameter_energy'];pm=obs['parameter_maxabs'];pn=obs['parameter_names'];numel=obs['parameter_numel']
    wc('parameter_gradient',[dict(parameter=str(name),numel=int(numel[j]),images=n,total_squared_energy=pe[:,j].sum(),rms=np.sqrt(pe[:,j].sum()/(n*numel[j])),
        max_abs=pm[:,j].max(),nonzero_images=int((pe[:,j]>0).sum()),finite=bool(np.isfinite(pe[:,j]).all() and np.isfinite(pm[:,j]).all())) for j,name in enumerate(pn)])
    wc('soft_directional',[dict(util['SDT',name],**{k:v for k,v in recs['SDT',name].items() if k not in ('stratum','loss')}) for name in mask])
    wc('historical_comparison',[dict(util[mode,name],**{k:v for k,v in recs[mode,name].items() if k not in ('stratum','loss')}) for name in ('all','Top20','Deep-Win','Shallow-Win') for mode in ('PRG_previous','ADT')])
    wc('per_image_losses',[dict(image_id=str(name),**{mode:float(obs['losses'][i,j]) for j,mode in enumerate(MODES)},ADT_active_count=int(gate[i].sum()),RG_active_count=int(rand[i].sum())) for i,name in enumerate(obs['names'])])
    names=[];nums=[];dens=[]
    def add(name,value,m):
        names.append(name);nums.append(np.where(m,value,0).sum(1,dtype=float));dens.append(m.sum(1))
    ia=auc_images['all'];names.append('Delta_image_AUROC');nums.append(np.nan_to_num(ia));dens.append(np.isfinite(ia).astype(int))
    add('DeepCapture',gate,dw);add('ShallowProtection',~gate,sw);add('DeepSelectionPrecision',dw,conf&gate)
    for name in ('all','Top20','class0','class1','class2','class3'):add('ADT:'+name+':mean_dm',dm['ADT'],mask[name])
    add('ADT:Raw_Correct:harm_rate',dm['ADT']<0,mask['Raw_Correct']);add('ADT:Raw_Wrong:benefit_rate',dm['ADT']>0,mask['Raw_Wrong'])
    add('BRR_ADT',gate&(dm['ADT']>0)&(dq['ADT']<0),dw);add('HHCR_ADT',gate&(dm['ADT']<0)&(dq['ADT']<0),sw)
    add('ADT-RG:all:mean_dm',dm['ADT']-dm['RG'],fg)
    add('ADT-RG:Shallow-Win:harm_rate',(dm['ADT']<0).astype(float)-(dm['RG']<0),sw)
    add('ADT-RG:Deep-Win:benefit_rate',(dm['ADT']>0).astype(float)-(dm['RG']>0),dw)
    for name in ('all','Top20','Deep-Win','Shallow-Win'):
        add('ADT:'+name+':benefit_rate',dm['ADT']>0,mask[name]);add('ADT:'+name+':harm_rate',dm['ADT']<0,mask[name])
    nums=np.stack(nums,1);dens=np.stack(dens,1);estimate=div(nums.sum(0),dens.sum(0));rep=[];rh=hashlib.sha256()
    for ix in bootstrap_indices(n):
        rh.update(ix.tobytes());weights=np.stack([np.bincount(row,minlength=n) for row in ix]);rep.append(div(weights@nums,weights@dens))
    rep=np.concatenate(rep);ci=[]
    for j,name in enumerate(names):
        finite=np.isfinite(rep[:,j]);lo,hi=np.quantile(rep[finite,j],[.025,.975])
        ci.append(dict(metric=name,estimate=estimate[j],ci_low=lo,ci_high=hi,resamples=10000,valid_resamples=int(finite.sum()),seed=42))
    wc('bootstrap',ci);wc('bootstrap_replicates',[dict(zip(names,row)) for row in rep]);cimap={r['metric']:r for r in ci}
    np.savez_compressed(out/(P+'image_statistics.npz'),names=np.array(names),numerators=nums,denominators=dens)
    ident=json.loads((run/(P+'identity_audit.json')).read_text());detach=json.loads((run/(P+'detach_audit.json')).read_text());smoke=json.loads((run/(P+'bf16_smoke.json')).read_text())
    pr={name:util['ADT',name] for name in mask};rc=recs['ADT','all'];ref=reference
    positive=sum(pr[name]['mean_dm']>0 for name in ('all','Top20','class0','class1','class2','class3'))
    sig=(cimap['ADT-RG:all:mean_dm']['ci_low']>0 or cimap['ADT-RG:Shallow-Win:harm_rate']['ci_high']<0 or cimap['ADT-RG:Deep-Win:benefit_rate']['ci_low']>0)
    a=ref['image_auroc']>=.75 and ref['DeepCaptureRate']>=.60 and ref['ShallowProtectionRate']>=.75 and ref['BA']>=.70
    b=all(pr[name]['benefit_rate']>pr[name]['harm_rate'] for name in ('all','Top20')) and positive>=5
    c=rc['BRR']>=.60 and pr['Deep-Win']['benefit_rate']>=.60 and pr['Deep-Win']['mean_dm']>0
    d=rc['HHCR']<=.30 and pr['Shallow-Win']['harm_rate']<=.30 and ref['ShallowProtectionRate']>=.70
    e=pr['Raw_Correct']['harm_rate']<=.30 and pr['Raw_Wrong']['benefit_rate']>=.40 and pr['all']['active_transfer_fraction']>=.10
    f=pr['all']['mean_dm']>util['RG','all']['mean_dm'] and (pr['Shallow-Win']['harm_rate']<util['RG','Shallow-Win']['harm_rate'] or pr['Deep-Win']['benefit_rate']>util['RG','Deep-Win']['benefit_rate']) and sig
    g=all((rt['all_finite'],not rt['optimizer_created'],rt['optimizer_steps']==0,not rt['test_access'],smoke['pass'],ident['state_unchanged'],ident['bn_unchanged'],ident['prediction_unchanged'],
        *[detach[k] for k in ('q_detached','delta_detached','gate_detached','deep_source_detached','primary_ic1_none','hfrm_none','upstream_conv_nonzero','all_other_primary_gradients_none')]))
    soft=util['SDT','all']['mean_dm']>pr['all']['mean_dm'] and util['SDT','Shallow-Win']['harm_rate']<=pr['Shallow-Win']['harm_rate']+.05 and recs['SDT','all']['BRR']>=rc['BRR']-.05
    strong=ref['image_auroc']>=.78 and ref['DeepCaptureRate']>=.63 and ref['ShallowProtectionRate']>=.78 and rc['BRR']>=.63 and rc['HHCR']<=.22 and pr['Raw_Correct']['harm_rate']<=.20 and pr['Raw_Wrong']['benefit_rate']>=.50 and pr['all']['mean_dm']>0 and pr['Top20']['mean_dm']>0 and f
    summary=dict(images=n,foreground_targets=int(fg.sum()),adjudication=ref,primary=pr,BRR=rc['BRR'],HHCR=rc['HHCR'],positive_mean_dm_strata=positive,
        **{'gate_'+k:'PASS' if v else 'FAIL' for k,v in zip('ABCDEFG',(a,b,c,d,e,f,g))},decision=decide(a,b,c,d,e,f,g),
        class_power={name:amap[name]['power'] for name in ('class0','class1','class2','class3')},random_comparison_significant=sig,
        SOFT_DIRECTIONAL_TRANSFER_PROMISING=soft,STRONG_DIRECTIONAL_TRANSFER_SIGNAL=strong,
        bootstrap_rng_sha256=rh.hexdigest(),bootstrap_resamples=10000,raw_tied_competitor_pixels=int((fg&ties).sum()),
        optimizer_steps=0,test_access=False,full25_started=False,analysis_seconds=time.perf_counter()-start,run=str(run),command=shlex.join([sys.executable,*sys.argv]))
    write_json(out/(P+'summary.json'),summary)
    print(json.dumps(clean({k:v for k,v in summary.items() if k.startswith('gate_') or k in ('decision','BRR','HHCR','class_power')})),flush=True)

if __name__=='__main__':main()
