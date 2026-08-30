"""Fixed-population acceptance statistics and paired image bootstrap. No search."""
import argparse
import csv
import hashlib
import json
import shlex
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b16_common import sha256,write_json,write_csv,margin_direction,bootstrap_indices,clean
from tools.rddr_phase2b17_common import PREFIX,MODES,HFRM_GROUPS,groups,rank_metrics,sign_metrics,decide


def load_np(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def ratio(a,b):return float(a/b) if b else np.nan
def mean(a):return float(np.mean(a,dtype=np.float64)) if a.size else np.nan


def ranked_row(score,label,mask):
    n=mask.shape[0];row=rank_metrics(score[mask],label[mask]);auc=np.full(n,np.nan)
    for i in range(n):
        y=label[i,mask[i]]
        if y.any() and (~y).any():auc[i]=rank_metrics(score[i,mask[i]],y)['auroc']
    row.update(image_auroc=float(np.nanmean(auc)) if np.isfinite(auc).any() else np.nan,
               eligible_images=int(np.isfinite(auc).sum()),nonempty_images=int(mask.any(1).sum()),
               excluded_no_dual_label_images=int(n-np.isfinite(auc).sum()))
    return row,auc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();run=Path(args.run);out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    rt=json.loads((run/(PREFIX+'runtime.json')).read_text())
    assert all(sha256(path)==rt['source_sha256'][key] for key,path in rt['paths'].items())
    assert sha256(run/(PREFIX+'observations.npz'))==rt['observation_sha256']
    data=load_np(rt['paths']['native']);old=load_np(rt['paths']['derived']);prev=load_np(rt['paths']['previous']);obs=load_np(run/(PREFIX+'observations.npz'))
    assert np.array_equal(data['names'],obs['names']) and np.array_equal(data['names'],prev['names'])
    y=data['truth'];n=len(y);rect=obs['p_rect'].argmax(1);teacher=old['anchor_sym'].argmax(1)
    masks,tw,rw=groups(data,rect,teacher);winner=tw|rw;fg=masks['all'];delta=obs['delta'];accepted=delta>0
    assert n==3418 and int(tw.sum())==88290 and int(rw.sum())==168626
    allg={'U':prev['gradients'][:,0],'CCA':prev['gradients'][:,2],'HA':obs['gradients'][:,0],'SA':obs['gradients'][:,1]}
    dm={};norm={}
    for mode,g in allg.items():
        dm[mode],ties=margin_direction(prev['logits'],g,y)
        norm[mode]=np.linalg.norm(g.astype(np.float64),axis=1)
    grad_label=dm['CCA']>0;grad_eligible=fg&(dm['CCA']!=0)
    out.mkdir(parents=True)
    def wc(name,rows):write_csv(out/(PREFIX+name+'.csv'),rows)
    support_rows=[];population=[];quality=[];utility=[];selective=[];qgrid=[];perclass=[];hfrm=[]
    winner_rows=[];grad_rows=[];image_auc={};wr={};gr={};quality_map={};um={}
    for name,mask in masks.items():
        count=int(mask.sum());acc=mask&accepted;rej=mask&~accepted
        population.append(dict(stratum=name,targets=count,images=int(mask.any(1).sum()),accepted=int(acc.sum()),rejected=int(rej.sum()),
                               acceptance_rate=ratio(acc.sum(),count),zero_delta=int((mask&(delta==0)).sum())))
        for key in ('R_S','R_D','T_S','T_D','S_R','S_T','delta'):
            a=obs[key][mask]
            support_rows.append(dict(stratum=name,quantity=key,count=count,mean=mean(a),std=float(a.std(dtype=np.float64)),
                                     median=float(np.median(a)),min=float(a.min()),max=float(a.max())))
        wmask=mask&winner
        row,aucs=ranked_row(delta,tw,wmask);image_auc['winner:'+name]=aucs
        tp=int((wmask&tw&accepted).sum());fn=int((wmask&tw&~accepted).sum());fp=int((wmask&rw&accepted).sum());tn=int((wmask&rw&~accepted).sum())
        row=dict(stratum=name,**row,TP=tp,FN=fn,FP=fp,TN=tn,**sign_metrics(tp,fn,fp,tn))
        winner_rows.append(row);wr[name]=row
        selective.append(dict(stratum=name,Teacher_Win=tp+fn,Rect_Win=fp+tn,accepted_Teacher_Win=tp,accepted_Rect_Win=fp,
                               correction_precision=ratio(tp,tp+fp),correction_recall=ratio(tp,tp+fn),rect_protection_rate=ratio(tn,fp+tn)))
        row,aucs=ranked_row(delta,grad_label,mask&grad_eligible)
        row=dict(stratum=name,**row,zero_dm=int((mask&(dm['CCA']==0)).sum()));grad_rows.append(row);gr[name]=row;image_auc['gradient:'+name]=aucs
        for region,m in (('Accepted',acc),('Rejected',rej)):
            nc=int(m.sum());repair=int((m&tw).sum());harm=int((m&rw).sum())
            tc=int((m&(teacher==y)).sum());rc=int((m&(rect==y)).sum())
            row=dict(stratum=name,region=region,targets=nc,teacher_accuracy=ratio(tc,nc),rect_accuracy=ratio(rc,nc),
                     accuracy_delta=ratio(tc-rc,nc),repair=repair,harm=harm,net_repair=repair-harm,net_repair_rate=ratio(repair-harm,nc))
            quality.append(row);quality_map[name,region]=row
        for mode in MODES:
            values=dm[mode][mask];active=norm[mode][mask]>0
            row=dict(stratum=name,loss=mode,targets=count,benefit_rate=mean(values>0),harm_rate=mean(values<0),zero_rate=mean(values==0),
                     mean_dm=mean(values),median_dm=float(np.median(values)),active_fraction=mean(active),mean_gradient_norm=mean(norm[mode][mask]),
                     accepted_conditional_benefit=mean(dm[mode][acc]>0),accepted_conditional_harm=mean(dm[mode][acc]<0),
                     denominator='all valid foreground in stratum, including rejected zero gradients')
            utility.append(row);um[mode,name]=row
        if name.startswith('Q'):
            for region,m in (('Accept',acc),('Reject',rej)):
                qrow=quality_map[name,'Accepted' if region=='Accept' else 'Rejected']
                qgrid.append(dict(quintile=name,acceptance=region,targets=int(m.sum()),accuracy_delta=qrow['accuracy_delta'],
                                  net_repair=qrow['net_repair'],net_repair_rate=qrow['net_repair_rate'],CCA_mean_dm=mean(dm['CCA'][m]),HA_mean_dm=mean(dm['HA'][m])))
        if name.startswith('class'):
            power='SUFFICIENT' if wr[name]['positive']>=500 and wr[name]['negative']>=500 and wr[name]['eligible_images']>=30 else 'UNDERPOWERED'
            perclass.append(dict(stratum=name,Teacher_Win=wr[name]['positive'],Rect_Win=wr[name]['negative'],eligible_images=wr[name]['eligible_images'],
                                 image_auroc=wr[name]['image_auroc'],balanced_accuracy=wr[name]['balanced_accuracy'],acceptance_rate=ratio(acc.sum(),count),
                                 accepted_accuracy_delta=quality_map[name,'Accepted']['accuracy_delta'],HA_benefit_rate=um['HA',name]['benefit_rate'],
                                 HA_harm_rate=um['HA',name]['harm_rate'],HA_mean_dm=um['HA',name]['mean_dm'],power=power))
        if name in HFRM_GROUPS:
            hfrm.append(dict(stratum=name,targets=count,acceptance_rate=ratio(acc.sum(),count),mean_delta=mean(delta[mask]),
                             teacher_rect_accuracy_delta=mean((teacher[mask]==y[mask]).astype(int)-(rect[mask]==y[mask])),
                             HA_mean_dm=um['HA',name]['mean_dm'],definition='frozen raw -> FULL HFRM transition, not isolated CH'))
    for name,rows in (('support_rect_teacher',support_rows),('acceptance_winner',winner_rows),('gradient_discrimination',grad_rows),
                      ('acceptance_population',population),('accepted_teacher_quality',quality),('selective_correction',selective),
                      ('q_acceptance_grid',qgrid),('per_class',perclass),('hfrm_groups',hfrm)):
        wc(name,rows)
    wc('ha_gradient',[r for r in utility if r['loss']=='HA']);wc('sa_gradient',[r for r in utility if r['loss']=='SA'])
    wc('all_gradient_controls',utility)
    wc('correct_wrong_safety',[r for r in utility if r['stratum'] in ('all','Top20','Rect_Correct','Rect_Wrong','class0','class1','class2','class3')])
    wc('gradient_coverage',[{k:r[k] for k in ('stratum','loss','targets','active_fraction','zero_rate','mean_gradient_norm')} for r in utility])
    wc('boundary_interior',[dict(stratum=name,winner_image_auroc=wr[name]['image_auroc'],winner_pooled_auroc=wr[name]['auroc'],
                                HA_benefit_rate=um['HA',name]['benefit_rate'],HA_harm_rate=um['HA',name]['harm_rate'],HA_mean_dm=um['HA',name]['mean_dm'],
                                accepted_accuracy_delta=quality_map[name,'Accepted']['accuracy_delta']) for name in ('boundary','interior')])
    # Fixed score orientation, diagnostic only. Natural-log entropy, no score reversal.
    pt=old['anchor_sym'].astype(float);pr=obs['p_rect'].astype(float);mid=.5*(pt+pr)
    controls={'q':data['q_feature'],'delta_accept':delta,'teacher_maxconf_minus_rect_maxconf':pt.max(1)-pr.max(1),
              'teacher_entropy_minus_rect_entropy':-(pt*np.log(pt+1e-8)).sum(1)+(pr*np.log(pr+1e-8)).sum(1),
              'JS_teacher_rect':.5*((pt*np.log((pt+1e-8)/(mid+1e-8))).sum(1)+(pr*np.log((pr+1e-8)/(mid+1e-8))).sum(1))}
    control_rows=[]
    for name,value in controls.items():
        row,aucs=ranked_row(value,tw,winner);control_rows.append(dict(score=name,**row));image_auc['control:'+name]=aucs
    wc('q_vs_acceptance',[r for r in control_rows if r['score'] in ('q','delta_accept')]);wc('confidence_controls',control_rows[2:])
    # Ratios are represented by per-image numerator/denominator. Same resampling for every estimate.
    stats={};point={}
    def register(key,num,den):
        stats[key]=(np.asarray(num,dtype=float),np.asarray(den,dtype=float));point[key]=ratio(np.sum(num),np.sum(den))
    for key in ('winner:all','gradient:all',*[f'winner:class{k}' for k in range(4)]):
        values=image_auc[key];register(key+':image_auroc',np.nan_to_num(values),np.isfinite(values))
    av=image_auc['control:delta_accept'];qv=image_auc['control:q'];eligible=np.isfinite(av)&np.isfinite(qv)
    register('delta-minus-q:image_auroc',np.where(eligible,av-qv,0),eligible)
    for name in ('all','Top20','class0','class1','class2','class3'):
        m=masks[name];register('HA:'+name+':mean_dm',(dm['HA']*m).sum(1,dtype=float),m.sum(1))
    for key,mode,name,kind in (('HA:Rect_Correct:harm_rate','HA','Rect_Correct','harm'),('HA:Rect_Wrong:benefit_rate','HA','Rect_Wrong','benefit'),
                               ('HA:all:active_fraction','HA','all','active')):
        vals=dm[mode]<0 if kind=='harm' else (dm[mode]>0 if kind=='benefit' else norm[mode]>0)
        m=masks[name];register(key,(vals&m).sum(1),m.sum(1))
    for name in ('all','Top20','Rect_Correct','Rect_Wrong'):
        m=masks[name]
        register('HA-CCA:'+name+':mean_dm',((dm['HA'].astype(float)-dm['CCA'])*m).sum(1),m.sum(1))
        register('HA-CCA:'+name+':harm_rate',(((dm['HA']<0).astype(int)-(dm['CCA']<0))*m).sum(1),m.sum(1))
    ac=accepted&fg;net=(tw.astype(int)-rw)*ac
    register('Accepted:teacher-rect_accuracy_delta',net.sum(1),ac.sum(1))
    register('Accepted:NetRepair_rate',net.sum(1),ac.sum(1))
    # Mean denominator 1/n makes this replicate the total count under image resampling.
    register('Accepted:NetRepair_count',net.sum(1),np.full(n,1/n))
    tp=(tw&accepted).sum(1);fn=(tw&~accepted).sum(1);fp=(rw&accepted).sum(1);tn=(rw&~accepted).sum(1)
    register('winner:Teacher-Win_recall',tp,tp+fn);register('winner:Rect-Win_recall',tn,tn+fp)
    keys=list(stats);nums=np.stack([stats[k][0] for k in keys],1);dens=np.stack([stats[k][1] for k in keys],1)
    bs=[];rng_sha=hashlib.sha256()
    for indices in bootstrap_indices(n):
        rng_sha.update(indices.tobytes());w=np.stack([np.bincount(idx,minlength=n) for idx in indices]).astype(float)
        den=w@dens;sample=np.divide(w@nums,den,out=np.full_like(den,np.nan),where=den>0)
        sig=sign_metrics(w@tp,w@fn,w@fp,w@tn);bs.append(np.c_[sample,sig['balanced_accuracy']])
    bs=np.concatenate(bs);keys.append('winner:zero-sign_BA');point[keys[-1]]=float(wr['all']['balanced_accuracy'])
    boot=[]
    for j,key in enumerate(keys):
        finite=np.isfinite(bs[:,j]);ci=np.quantile(bs[finite,j],[.025,.975]) if finite.any() else [np.nan,np.nan]
        boot.append(dict(metric=key,estimate=point[key],ci_low=ci[0],ci_high=ci[1],resamples=10000,valid_resamples=int(finite.sum()),seed=42))
    wc('bootstrap',boot);wc('bootstrap_replicates',[dict(draw=k,**dict(zip(keys,row))) for k,row in enumerate(bs)])
    np.savez_compressed(out/(PREFIX+'sufficient_statistics.npz'),numerators=nums,denominators=dens,keys=np.array(keys),bootstrap=bs,
                        image_auc_keys=np.array(list(image_auc)),image_aucs=np.stack(list(image_auc.values()),1),tp=tp,fn=fn,fp=fp,tn=tn)
    ci={r['metric']:r for r in boot}
    a=bool(wr['all']['image_auroc']>=.65 and ci['winner:all:image_auroc']['ci_low']>.5 and wr['all']['balanced_accuracy']>=.60 and wr['all']['teacher_win_recall']>=.55 and wr['all']['rect_win_recall']>=.55)
    b=bool(gr['all']['image_auroc']>=.65 and ci['gradient:all:image_auroc']['ci_low']>.5)
    critical=('all','Top20','class0','class1','class2','class3');under=[x['stratum'] for x in perclass if x['power']=='UNDERPOWERED']
    positive=sum(um['HA',name]['mean_dm']>0 for name in critical)
    rates=all(um['HA',name]['benefit_rate']>um['HA',name]['harm_rate'] for name in ('all','Top20'))
    c='PASS' if rates and positive>=5 else 'FAIL'
    if under and rates:
        known=sum(um['HA',name]['mean_dm']>0 for name in critical if name not in under)
        c='UNDERPOWERED' if known+len(under)>=5 else 'FAIL'
    d=bool(um['HA','Rect_Correct']['harm_rate']<=.5*um['CCA','Rect_Correct']['harm_rate'] and um['HA','Rect_Wrong']['benefit_rate']>=.60 and um['HA','all']['active_fraction']>=.10)
    soft=bool(um['SA','all']['mean_dm']>um['HA','all']['mean_dm'] and um['SA','Rect_Correct']['harm_rate']<=um['HA','Rect_Correct']['harm_rate']+.05)
    strong=bool(wr['all']['image_auroc']>=.75 and gr['all']['image_auroc']>=.75 and um['HA','all']['mean_dm']>0 and um['HA','Top20']['mean_dm']>0 and um['HA','Rect_Correct']['harm_rate']<=.25 and um['HA','Rect_Wrong']['benefit_rate']>=.70)
    identity=json.loads((run/(PREFIX+'identity_audit.json')).read_text());detach=json.loads((run/(PREFIX+'detach_audit.json')).read_text());smoke=json.loads((run/(PREFIX+'bf16_smoke.json')).read_text())
    engineering=bool(rt['all_finite'] and max(rt['parity'].values())<=1e-7 and identity['all_parameters_buffers_unchanged'] and identity['official_predictions_exact'] and identity['max_logit_replay_difference']==0 and
                     all(detach[k] for k in ('teacher_detached','q_detached','delta_detached','m_detached','a_detached','all_other_parameter_gradients_none')) and smoke['all_finite'] and smoke['budget_pass'] and rt['optimizer_steps']==0)
    identities={}
    q=data['q_feature'];zcca=q.sum(1,keepdims=True)+1e-8
    for mode in ('HA','SA'):
        weight=(delta>0).astype(np.float32) if mode=='HA' else np.maximum(delta,0)
        scale=weight*zcca/((q*weight).sum(1,keepdims=True)+1e-8)
        identities[mode]=dict(positive_scaling_max_abs=float(np.abs(allg[mode]-allg['CCA']*scale[:,None]).max()),
                             accepted_dm_sign_mismatches=int((fg&accepted&(np.sign(dm[mode])!=np.sign(dm['CCA']))).sum()),
                             rejected_nonzero_gradient_pixels=int((fg&~accepted&(norm[mode]>0)).sum()),
                             all_rejected_images=int((~accepted.any(1)).sum()))
    write_json(out/(PREFIX+'gradient_identities.json'),identities)
    summary=dict(images=n,foreground_targets=int(fg.sum()),teacher_win=int(tw.sum()),rect_win=int(rw.sum()),winner=wr['all'],gradient_discrimination=gr['all'],
                 gate_A='PASS' if a else 'FAIL',gate_B='PASS' if b else 'FAIL',gate_C=c,gate_D='PASS' if d else 'FAIL',engineering='PASS' if engineering else 'FAIL',
                 decision=decide(a,b,c,d,engineering),soft_acceptance_promising=soft,strong_acceptance_signal=strong,HA_positive_mean_dm_strata=int(positive),
                 HA_key={name:um['HA',name] for name in ('all','Top20','Rect_Correct','Rect_Wrong')},per_class=perclass,
                 accepted_teacher_quality=quality_map['all','Accepted'],rejected_teacher_quality=quality_map['all','Rejected'],acceptance_rate=ratio((fg&accepted).sum(),fg.sum()),
                 frozen_dm_ties=int((fg&(dm['CCA']==0)).sum()),tied_competitor_pixels=int((fg&ties).sum()),gradient_identities=identities,
                 bootstrap_rng_sha256=rng_sha.hexdigest(),bootstrap_resamples=10000,analysis_seconds=time.perf_counter()-start,
                 command=shlex.join([sys.executable,*sys.argv]),run=str(run),optimizer_steps=0,test_access=False,full25_started=False,search=False)
    write_json(out/(PREFIX+'summary.json'),summary)
    print(json.dumps(clean({k:summary[k] for k in ('decision','gate_A','gate_B','gate_C','gate_D','engineering','winner','gradient_discrimination','HA_key','acceptance_rate')}),indent=2),flush=True)


if __name__=='__main__':main()
