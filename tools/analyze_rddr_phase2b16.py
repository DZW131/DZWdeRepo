"""Pooled native28 diagnostics and paired image bootstrap; never selects hyperparameters."""
import argparse
import csv
import json
import shlex
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b16_common import *


def load_np(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def describe(x):
    return dict(mean=float(x.mean()),median=float(np.median(x)),positive_fraction=float((x>0).mean()),
                negative_fraction=float((x<0).mean())) if x.size else dict(mean=np.nan,median=np.nan,positive_fraction=np.nan,negative_fraction=np.nan)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    runtime=json.loads((run/(PREFIX+'runtime.json')).read_text())
    assert sha256(runtime['native'])==NATIVE_SHA and sha256(runtime['derived'])==DERIVED_SHA
    assert sha256(run/(PREFIX+'gradient_observations.npz'))==runtime['observations_sha256']
    data=load_np(runtime['native']);derived=load_np(runtime['derived']);obs=load_np(run/(PREFIX+'gradient_observations.npz'))
    assert np.array_equal(obs['names'],data['names']) and len(obs['names'])==3418
    n=len(obs['names']);y=data['truth'];valid=y<4;logits=obs['logits']
    ex=np.exp(logits.astype(np.float64)-logits.max(1,keepdims=True));rect=ex/ex.sum(1,keepdims=True)
    prob={'rect':rect,'fixed':data['fixed_average'].astype(np.float64),'teacher':derived['anchor_sym'].astype(np.float64)}
    pred={k:p.argmax(1) for k,p in prob.items()};groups=strata(data,pred['rect'])
    gtprob={k:np.take_along_axis(p,y.clip(0,3)[:,None],axis=1)[:,0] for k,p in prob.items()}
    kl={k:(p*(np.log(p+EPS)-np.log(rect+EPS))).sum(1) for k,p in prob.items() if k!='rect'}
    out.mkdir(parents=True)
    def wc(name,rows):write_csv(out/(PREFIX+name+'.csv'),rows)
    metrics=[];cms={};metric_map={}
    for k,p in prob.items():
        cm=np.stack([np.bincount((y[i,valid[i]]*4+pred[k][i,valid[i]]),minlength=16).reshape(4,4) for i in range(n)])
        cms[k]=cm;s=scores(cm.sum(0));metric_map[k]=s
        brier=p.square() if hasattr(p,'square') else p*p
        brier=brier.sum(1)-2*gtprob[k]+1
        row=dict(estimator=k,targets=int(valid.sum()),accuracy=s['accuracy'],miou=s['miou'],dice=s['dice'],
                 nll=-np.log(gtprob[k][valid]+EPS).mean(),brier=brier[valid].mean())
        row.update({f'class{c}_iou':s['class_iou'][c] for c in range(4)})
        row.update({f'class{c}_dice':s['class_dice'][c] for c in range(4)})
        metrics.append(row)
    wc('teacher_metrics',metrics)
    transitions=[];advantages=[];qrows=[];identity_error=0.
    for name,mask in groups.items():
        count=int(mask.sum());r=pred['rect']==y
        for k in ('fixed','teacher'):
            correct=pred[k]==y;repair=int((mask&~r&correct).sum());harm=int((mask&r&~correct).sum())
            net=(repair-harm)/count if count else np.nan
            accgap=float(correct[mask].mean()-r[mask].mean()) if count else np.nan
            identity_error=max(identity_error,abs(net-accgap))
            transitions.append(dict(stratum=name,teacher=k,targets=count,images=int(mask.any(1).sum()),repair=repair,harm=harm,
                                    net_repair=repair-harm,repair_rate=repair/count,harm_rate=harm/count,net_repair_rate=net,
                                    accuracy_difference=accgap))
            adv=gtprob[k]-gtprob['rect']
            advantages.append(dict(stratum=name,teacher=k,targets=count,**describe(adv[mask])))
            if name.startswith('Q'):
                qrows.append(dict(stratum=name,teacher=k,targets=count,net_repair_rate=net,
                                  teacher_advantage=adv[mask].mean(),mean_kl=kl[k][mask].mean()))
        advantages.append(dict(stratum=name,teacher='teacher-minus-fixed',targets=count,
                               **describe((gtprob['teacher']-gtprob['fixed'])[mask])))
    wc('teacher_transition',transitions);wc('teacher_advantage',advantages);wc('q_strata',qrows)
    utility=[];logrows=[];localization=[];feature=[];param_summary=[];attribution=[]
    dm={};gn={};ties=None;utility_map={};local_map={}
    for j,mode in enumerate(('U','FA','CCA')):
        g=obs['gradients'][:,j]
        dm[mode],ties=margin_direction(logits,g,y)
        gn[mode]=np.sqrt(np.square(g.astype(np.float64)).sum(1))
        for name,mask in groups.items():
            count=int(mask.sum());images=int(mask.any(1).sum());d=dm[mode][mask];norm=gn[mode][mask]
            row=dict(loss=mode,stratum=name,targets=count,images=images,benefit_rate=float((d>0).mean()),
                     harm_rate=float((d<0).mean()),zero_rate=float((d==0).mean()),mean_dm=float(d.mean(dtype=np.float64)),
                     median_dm=float(np.median(d)),gradient_norm=float(norm.mean()),
                     power='SUFFICIENT' if count>=500 and images>=30 else 'UNDERPOWERED')
            utility.append(row);utility_map[mode,name]=row
            logrows.append(dict(loss=mode,stratum=name,targets=count,mean_G=norm.mean(),rms=float(np.sqrt(np.square(norm).mean()/4)),
                                max_abs=float(np.abs(g.transpose(0,2,1)[mask]).max()),finite=True))
        top=gn[mode][groups['Top20']].mean();bottom=gn[mode][groups['Bottom80']].mean()
        row=dict(loss=mode,Top20_mean_G=top,Bottom80_mean_G=bottom,Top20_over_Bottom80=top/bottom,
                 **{f'Q{k}_mean_G':gn[mode][groups[f'Q{k}']].mean() for k in range(1,6)})
        localization.append(row);local_map[mode]=row
    wc('gradient_semantic_utility',utility);wc('logit_gradient',logrows);wc('gradient_localization',localization)
    wc('per_class_gradient',[r for r in utility if r['stratum'].startswith('class')])
    for name,mask in groups.items():
        norm=obs['feature_norm'][mask].astype(np.float64)
        feature.append(dict(stratum=name,targets=int(mask.sum()),rms=np.sqrt(np.square(norm).mean()/512),
                            mean_pixel_L2=norm.mean(),max_abs=obs['feature_max'][mask].max(),finite=True))
    wc('feature_gradient',feature)
    prows=list(csv.DictReader((run/(PREFIX+'parameter_per_image.csv')).open()))
    sums=np.zeros((n,len(PARAMS)))
    for j,k in enumerate(PARAMS):
        rows=[r for r in prows if r['parameter']==k];assert len(rows)==n
        sq=np.array([float(r['sumsq']) for r in rows]);sums[:,j]=sq
        elements=int(rows[0]['elements'])
        param_summary.append(dict(parameter=k,elements=elements,images=n,pooled_rms=np.sqrt(sq.sum()/(n*elements)),
                                  max_abs=max(float(r['max_abs']) for r in rows),finite=all(r['finite']=='True' for r in rows),
                                  mean_nonzero_fraction=np.mean([float(r['nonzero_fraction']) for r in rows]),
                                  zero_gradient_images=int((sq==0).sum())))
    total=np.sqrt(sums.sum())
    for name,indices in PATHS.items():
        sq=sums[:,indices].sum(1);g=np.sqrt(sq.sum())
        attribution.append(dict(path=name,pooled_L2=g,L2_fraction=g/total,squared_energy_share=g*g/total**2,
                                mean_per_image_L2=np.sqrt(sq).mean(),zero_gradient_images=int((sq==0).sum())))
    wc('parameter_gradient',param_summary);wc('gradient_path_attribution',attribution)
    # Sufficient statistics are sums per image; resampling uses paired image multiplicities.
    stats={};denoms={}
    for name,mask in groups.items():
        denoms[name]=mask.sum(1).astype(float)
        for mode in ('U','FA','CCA'):
            stats[f'{mode}:{name}:mean_dm']=(dm[mode]*mask).sum(1,dtype=np.float64)
            stats[f'{mode}:{name}:benefit_rate']=((dm[mode]>0)&mask).sum(1).astype(float)
    count=valid.sum(1).astype(float)
    net_teacher=(((pred['teacher']==y).astype(int)-(pred['rect']==y))*valid).sum(1)
    net_fixed=(((pred['fixed']==y).astype(int)-(pred['rect']==y))*valid).sum(1)
    bspec=[('teacher-fixed_accuracy',net_teacher-net_fixed,count),('teacher-vs-rect_NetRepair',net_teacher,count),
           ('fixed-vs-rect_NetRepair',net_fixed,count)]
    for group in ('all','Top20','class0','class1','class2','class3'):
        bspec.append((f'CCA:{group}:mean_dm',stats[f'CCA:{group}:mean_dm'],denoms[group]))
    bspec.append(('CCA:all:benefit_rate',stats['CCA:all:benefit_rate'],denoms['all']))
    for other in ('U','FA'):
        for group,metric in (('all','benefit_rate'),('all','mean_dm'),('Top20','mean_dm')):
            bspec.append((f'CCA-{other}:{group}:{metric}',stats[f'CCA:{group}:{metric}']-stats[f'{other}:{group}:{metric}'],denoms[group]))
    keys=['teacher-fixed_miou']+[x[0] for x in bspec];rep=[];rng_hash=__import__('hashlib').sha256()
    num=np.stack([x[1] for x in bspec],1);den=np.stack([x[2] for x in bspec],1)
    for ids in bootstrap_indices(n):
        rng_hash.update(ids.tobytes())
        weights=np.stack([np.bincount(i,minlength=n) for i in ids]).astype(np.float64)
        dif=scores((weights@cms['teacher'].reshape(n,16)).reshape(-1,4,4))['miou']-scores((weights@cms['fixed'].reshape(n,16)).reshape(-1,4,4))['miou']
        rep.append(np.c_[dif,(weights@num)/(weights@den)])
    rep=np.concatenate(rep);point=np.r_[metric_map['teacher']['miou']-metric_map['fixed']['miou'],num.sum(0)/den.sum(0)]
    bootstrap=[]
    for j,key in enumerate(keys):
        lo,hi=np.quantile(rep[:,j],[.025,.975]);bootstrap.append(dict(metric=key,estimate=point[j],ci_low=lo,ci_high=hi,resamples=10000,seed=42))
    wc('bootstrap',bootstrap);wc('bootstrap_replicates',(dict(draw=i,**dict(zip(keys,row))) for i,row in enumerate(rep)))
    np.savez_compressed(out/(PREFIX+'sufficient_statistics.npz'),cm_rect=cms['rect'],cm_fixed=cms['fixed'],cm_teacher=cms['teacher'],
                        numerator=num,denominator=den,keys=np.array(keys),bootstrap=rep)
    # Mathematical identities: disclose numerical sign exceptions; never tune a sign threshold.
    scale=data['q_feature']*784/(data['q_feature'].sum(1,keepdims=True)+EPS)
    identity_grad=float(np.abs(obs['gradients'][:,2]-obs['gradients'][:,0]*scale[:,None]).max())
    pos=valid&(data['q_feature']>0)
    sign_mismatch=pos&(np.sign(dm['CCA'])!=np.sign(dm['U']))
    sign_dict=dict(q_positive_targets=int(pos.sum()),q_zero_targets=int((valid&~(data['q_feature']>0)).sum()),
                   cca_u_gradient_positive_scaling_max_abs=identity_grad,cca_u_dm_sign_mismatch=int(sign_mismatch.sum()),
                   mismatch_max_abs_u_dm=float(np.abs(dm['U'][sign_mismatch]).max()) if sign_mismatch.any() else 0.,
                   mismatch_max_abs_cca_dm=float(np.abs(dm['CCA'][sign_mismatch]).max()) if sign_mismatch.any() else 0.,
                   net_repair_accuracy_identity_max_abs=identity_error,tied_competitor_pixels=int((ties&valid).sum()),
                   tie_counts={k:int((ties&m).sum()) for k,m in groups.items()})
    write_json(out/(PREFIX+'mathematical_identities.json'),sign_dict)
    smoke=json.loads((run/(PREFIX+'bf16_smoke.json')).read_text());ident=json.loads((run/(PREFIX+'identity_audit.json')).read_text())
    det=json.loads((run/(PREFIX+'detach_audit.json')).read_text())
    a=metric_map['teacher']['miou']>metric_map['fixed']['miou'] and metric_map['teacher']['accuracy']>=metric_map['fixed']['accuracy']
    a=bool(a and net_teacher.sum()>net_fixed.sum() and any(r['ci_low']>0 for r in bootstrap if r['metric'] in ('teacher-fixed_miou','teacher-fixed_accuracy')))
    loc=local_map['CCA'];u=local_map['U']
    b=bool(loc['Top20_mean_G']>loc['Bottom80_mean_G'] and loc['Top20_over_Bottom80']>u['Top20_over_Bottom80'] and loc['Q5_mean_G']>loc['Q1_mean_G'])
    critical=('all','Top20','class0','class1','class2','class3')
    positives=sum(utility_map['CCA',g]['mean_dm']>0 for g in critical)
    rates_ok=all(utility_map['CCA',g]['benefit_rate']>utility_map['CCA',g]['harm_rate'] for g in ('all','Top20'))
    insufficient=[g for g in critical if utility_map['CCA',g]['power']=='UNDERPOWERED']
    c='PASS' if rates_ok and positives>=5 else 'FAIL'
    if insufficient and rates_ok:
        known_positive=sum(utility_map['CCA',g]['mean_dm']>0 for g in critical if g not in insufficient)
        c='UNDERPOWERED' if known_positive+len(insufficient)>=5 else 'FAIL'
    d=bool(smoke['finite'] and smoke['budget_pass'] and smoke['feature_gradient_nonzero'] and
           all(r['finite'] for r in param_summary) and all(r['pooled_L2']>0 for r in attribution) and
           all(sum(row['sumsq'] for row in smoke['parameter_gradients'] if row['parameter'] in [PARAMS[i] for i in indices])>0 for indices in PATHS.values()) and
           det['teacher_detached'] and det['q_detached'] and ident['all_parameters_buffers_equal'] and ident['official_predictions_exact'] and
           ident['checkpoint_sha_after']==CKPT_SHA and ident['optimizer_steps']==0 and runtime['numerical_stability']['all_finite'])
    preferred=utility_map['CCA','all']['benefit_rate']>=utility_map['FA','all']['benefit_rate'] and utility_map['CCA','Top20']['mean_dm']>=utility_map['FA','Top20']['mean_dm']
    summary=dict(images=n,foreground_targets=int(valid.sum()),run=str(run),native_metric='pooled native28 four foreground classes; absent union excluded; not official fullres TTA score',
                 gate_A='PASS' if a else 'FAIL',gate_B='PASS' if b else 'FAIL',gate_C=c,gate_D='PASS' if d else 'FAIL',
                 decision=decision(a,b,c,d),adjudication_teacher_preferred=bool(preferred),positive_mean_dm_strata=int(positives),
                 class_support={g:dict(targets=utility_map['CCA',g]['targets'],images=utility_map['CCA',g]['images'],power=utility_map['CCA',g]['power']) for g in critical[2:]},
                 teacher_metrics=metrics,loss_summary={mode:dict(mean=float(obs['losses'][:,j].mean()),min=float(obs['losses'][:,j].min()),max=float(obs['losses'][:,j].max())) for j,mode in enumerate(('U','FA','CCA'))},
                 mathematical_identities=sign_dict,bootstrap_rng_sha256=rng_hash.hexdigest(),bootstrap_resamples=10000,
                 report_analysis_seconds=time.perf_counter()-start,analysis_command=shlex.join([sys.executable,*sys.argv]),
                 test_access=False,luad_access=False,optimizer_steps=0,full25_started=False,lambda_selected=False)
    write_json(out/(PREFIX+'summary.json'),summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
