"""Independent verification: explicit neighbor indexing, SciPy ranks, NumPy gradients/metrics/bootstrap.
No imports from the implementation or analyzer.
"""
import argparse
import csv
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import rankdata

P='rddr_phase2b17_'


def read_np(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)
def rjson(path):return json.loads(Path(path).read_text())
def rows(root,name):return list(csv.DictReader((root/(P+name+'.csv')).open()))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def rank_independent(score,label):
    score=np.asarray(score);label=np.asarray(label,dtype=bool);pos=int(label.sum());neg=len(label)-pos
    auc=(rankdata(score,method='average')[label].sum()-pos*(pos+1)/2)/(pos*neg) if pos and neg else np.nan
    if pos:
        order=np.argsort(-score,kind='stable');s=score[order];lab=label[order]
        ends=np.r_[np.flatnonzero(s[1:]!=s[:-1]),len(s)-1]
        cum=np.cumsum(lab)[ends];ap=np.sum((cum/ (ends+1))*np.diff(np.r_[0,cum]))/pos
    else:ap=np.nan
    return auc,ap


def compare(a,b,tol=1e-12):
    a,b=float(a),float(b)
    return (np.isnan(a) and np.isnan(b)) or abs(a-b)<=tol


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--report',required=True)
    args=ap.parse_args();run=Path(args.run);report=Path(args.report);dest=report/(P+'verification.json')
    if dest.exists():raise FileExistsError(dest)
    rt=rjson(run/(P+'runtime.json'));summary=rjson(report/(P+'summary.json'))
    data=read_np(rt['paths']['native']);old=read_np(rt['paths']['derived']);prev=read_np(rt['paths']['previous']);obs=read_np(run/(P+'observations.npz'))
    checks={};errors={};n=len(data['names']);y=data['truth'];fg=y<4;delta=obs['delta'];accept=delta>0
    checks['immutable_source_hashes']=all(sha(path)==rt['source_sha256'][k] for k,path in rt['paths'].items())
    checks['observation_hash']=sha(run/(P+'observations.npz'))==rt['observation_sha256']
    checks['all3418_order']=n==3418 and np.array_equal(data['names'],obs['names']) and np.array_equal(data['names'],prev['names'])
    checks['frozen_replay']=max(rt['parity'].values())<=1e-7
    # Full independent support recomputation: explicit neighbor-coordinate gather, no unfold or helper imports.
    yy,xx=np.indices((28,28));cy=yy.reshape(-1,1);cx=xx.reshape(-1,1)
    offsets=np.array([(dy,dx) for dy in range(-7,8) for dx in range(-7,8) if (dy,dx)!=(0,0)])
    ny=cy+offsets[:,0];nx=cx+offsets[:,1];valid=(ny>=0)&(ny<28)&(nx>=0)&(nx<28)
    indexes=np.clip(ny,0,27)*28+np.clip(nx,0,27)
    idx=torch.as_tensor(indexes,device='cuda');mask=torch.as_tensor(valid,device='cuda')
    max_support=0.;sign_difference=0;max_ambiguous=0.;softmax_err=0.
    with torch.no_grad():
        for i in range(n):
            pr=torch.from_numpy(prev['logits'][i]).cuda().softmax(0)
            softmax_err=max(softmax_err,float(np.abs(pr.cpu().numpy()-obs['p_rect'][i]).max()))
            targets={'R':torch.from_numpy(obs['p_rect'][i]).cuda(),'T':torch.from_numpy(old['anchor_sym'][i]).cuda()}
            values={}
            for suffix,key in (('S','ps'),('D','pd')):
                source=torch.from_numpy(data[key][i]).cuda()[:,idx]
                for prefix,target in targets.items():
                    t=target[:,:,None];mid=(t+source)*.5
                    divergence=.5*((t*(torch.log(t+1e-8)-torch.log(mid+1e-8))).sum(0)+(source*(torch.log(source+1e-8)-torch.log(mid+1e-8))).sum(0))
                    values[prefix+'_'+suffix]=(((1-divergence/np.log(2))*mask).sum(-1)/mask.sum(-1)).cpu().numpy()
            values['S_R']=(values['R_S']+values['R_D'])*.5;values['S_T']=(values['T_S']+values['T_D'])*.5
            values['delta']=values['S_T']-values['S_R']
            for k,v in values.items():max_support=max(max_support,float(np.abs(v-obs[k][i]).max()))
            mismatch=(values['delta']>0)!=(delta[i]>0);sign_difference+=int(mismatch.sum())
            if mismatch.any():max_ambiguous=max(max_ambiguous,float(np.abs(delta[i,mismatch]).max()))
    errors.update(independent_support_max_abs=max_support,independent_support_sign_mismatches=sign_difference,
                  max_original_abs_delta_at_sign_mismatch=max_ambiguous,rect_softmax_max_abs=softmax_err)
    checks['full3418_independent_support']=max_support<1e-6
    checks['rect_fp32_reconstruction']=softmax_err<=1e-7
    print(json.dumps(dict(phase='independent_support_pass',errors=errors)),flush=True)
    rect=obs['p_rect'].argmax(1);teacher=old['anchor_sym'].argmax(1)
    tw=fg&(teacher==y)&(rect!=y);rw=fg&(teacher!=y)&(rect==y);winner=tw|rw
    checks['frozen_winner_counts']=int(tw.sum())==88290 and int(rw.sum())==168626
    masks={'all':fg,'Top20':fg&data['top20'].astype(bool),'Bottom80':fg&~data['top20'].astype(bool),
           'Rect_Correct':fg&(rect==y),'Rect_Wrong':fg&(rect!=y),'Teacher-Win':tw,'Rect-Win':rw}
    edges=[.020935675129294395,.072734534740448,.163648784160614,.3369627296924591]
    bins=np.searchsorted(edges,data['q_feature'],side='left');masks.update({f'Q{k+1}':fg&(bins==k) for k in range(5)})
    masks.update({f'class{k}':fg&(y==k) for k in range(4)})
    masks.update(boundary=fg&data['boundary'].astype(bool),interior=fg&~data['boundary'].astype(bool))
    for k,name in enumerate(('Corrected_by_CH','Still_Wrong','Harmed_by_CH','Stable_Correct')):masks[name]=fg&(data['hfrm']==k)
    log=prev['logits'];exp=np.exp(log.astype(float)-log.max(1,keepdims=True));p=exp/exp.sum(1,keepdims=True)
    pt=old['anchor_sym'].astype(float);q=data['q_feature'].astype(float)
    kl=(pt*(np.log(pt+1e-8)-np.log(p+1e-8))).sum(1)
    raw={'U':prev['gradients'][:,0],'CCA':prev['gradients'][:,2],'HA':obs['gradients'][:,0],'SA':obs['gradients'][:,1]};dm={};norm={}
    loss_error=grad_error=0.
    others=np.array([[j for j in range(4) if j!=c] for c in range(4)])[y.clip(0,3)].transpose(0,2,1)
    lc=np.take_along_axis(log,others,axis=1);tied=lc==lc.max(1,keepdims=True)
    for j,(mode,g) in enumerate(raw.items()):
        w=q.copy()
        if mode=='HA':w*=accept
        if mode=='SA':w*=np.maximum(delta,0)
        w=w/(w.sum(1,keepdims=True)+1e-8) if mode!='U' else np.ones_like(w)/784
        expected_loss=(w*kl).sum(1);a=pt*p/(p+1e-8);expected_grad=(p*a.sum(1,keepdims=True)-a)*w[:,None]
        loss_error=max(loss_error,float(np.abs(expected_loss-obs['losses'][:,j]).max()))
        grad_error=max(grad_error,float(np.abs(expected_grad-g).max()))
        move=-g;true_move=np.take_along_axis(move,y.clip(0,3)[:,None],axis=1)[:,0]
        other_move=np.take_along_axis(move,others,axis=1)
        dm[mode]=true_move-np.where(tied,other_move,-np.inf).max(1);norm[mode]=np.sqrt((g.astype(float)**2).sum(1))
    errors.update(analytic_gradient_max_abs=grad_error,loss_max_abs=loss_error)
    checks['independent_ha_sa_analytic_gradient']=grad_error<2e-8;checks['independent_loss_denominator']=loss_error<1e-5
    checks['all_gradients_finite']=all(np.isfinite(g).all() for g in raw.values())
    checks['rejected_zero_gradient']=all(not np.any(g.transpose(0,2,1)[~accept]!=0) for k,g in raw.items() if k in ('HA','SA'))
    checks['accepted_direction_preserved']=all(not np.any((np.sign(dm[k])!=np.sign(dm['CCA']))&accept&fg) for k in ('HA','SA'))
    def image_aucs(score,label,mask):
        out=np.full(n,np.nan)
        for i in range(n):out[i]=rank_independent(score[i,mask[i]],label[i,mask[i]])[0]
        return out
    ia={};ranks_ok=True
    for filename,label,elig,prefix in (('acceptance_winner',tw,winner,'winner'),('gradient_discrimination',dm['CCA']>0,fg&(dm['CCA']!=0),'gradient')):
        for row in rows(report,filename):
            name=row['stratum'];m=masks[name]&elig;auc,ap=rank_independent(delta[m],label[m]);image=image_aucs(delta,label,m);ia[prefix+':'+name]=image
            ranks_ok &= compare(auc,row['auroc']) and compare(ap,row['auprc']) and compare(np.nanmean(image) if np.isfinite(image).any() else np.nan,row['image_auroc'])
            ranks_ok &= int(row['eligible_images'])==int(np.isfinite(image).sum())
    checks['winner_and_gradient_rank_metrics']=ranks_ok
    coverage_ok=quality_ok=utility_ok=True;um={}
    for row in rows(report,'acceptance_population'):
        m=masks[row['stratum']];coverage_ok &= int(row['targets'])==int(m.sum()) and int(row['accepted'])==int((m&accept).sum())
    for row in rows(report,'accepted_teacher_quality'):
        m=masks[row['stratum']]&(accept if row['region']=='Accepted' else ~accept)
        correct_t=int(((teacher==y)&m).sum());correct_r=int(((rect==y)&m).sum());cnt=m.sum()
        quality_ok &= int(row['repair'])==int((tw&m).sum()) and int(row['harm'])==int((rw&m).sum())
        quality_ok &= compare((correct_t-correct_r)/cnt if cnt else np.nan,row['accuracy_delta'])
    for row in rows(report,'all_gradient_controls'):
        m=masks[row['stratum']];v=dm[row['loss']][m];no=norm[row['loss']][m]
        vals={'benefit_rate':(v>0).mean(),'harm_rate':(v<0).mean(),'zero_rate':(v==0).mean(),
              'mean_dm':v.mean(dtype=float),'median_dm':np.median(v),'active_fraction':(no>0).mean()}
        um[row['loss'],row['stratum']]=vals
        utility_ok &= all(compare(val,row[k]) for k,val in vals.items())
    checks.update(acceptance_coverage=coverage_ok,accepted_rejected_quality=quality_ok,all_denominator_gradient_utility=utility_ok)
    # Confidence controls remain in specified orientation.
    pr=obs['p_rect'].astype(float);mid=(pt+pr)/2
    controls={'q':q,'delta_accept':delta,'teacher_maxconf_minus_rect_maxconf':pt.max(1)-pr.max(1),
              'teacher_entropy_minus_rect_entropy':(-pt*np.log(pt+1e-8)).sum(1)-(-pr*np.log(pr+1e-8)).sum(1),
              'JS_teacher_rect':((pt*(np.log(pt+1e-8)-np.log(mid+1e-8))).sum(1)+(pr*(np.log(pr+1e-8)-np.log(mid+1e-8))).sum(1))/2}
    control_ok=True
    for row in rows(report,'q_vs_acceptance')+rows(report,'confidence_controls'):
        value=controls[row['score']];auc,ap=rank_independent(value[winner],tw[winner]);image=image_aucs(value,tw,winner);ia['control:'+row['score']]=image
        control_ok &= compare(auc,row['auroc'],1e-7) and compare(ap,row['auprc'],1e-7) and compare(np.nanmean(image),row['image_auroc'],1e-7)
    checks['fixed_direction_confidence_controls']=control_ok
    bootstrap=rows(report,'bootstrap');keys=[r['metric'] for r in bootstrap];nums=[];dens=[]
    for key in keys[:-1]:
        if key.endswith(':image_auroc'):
            if key=='delta-minus-q:image_auroc':
                a,b=ia['control:delta_accept'],ia['control:q'];valid=np.isfinite(a)&np.isfinite(b);v=np.where(valid,a-b,0)
            else:v=ia[key.removesuffix(':image_auroc')];valid=np.isfinite(v);v=np.nan_to_num(v)
            num,den=v,valid
        elif key.startswith('Accepted:'):
            m=fg&accept;num=((tw.astype(int)-rw)*m).sum(1);den=np.full(n,1/n) if key.endswith('count') else m.sum(1)
        elif key=='winner:Teacher-Win_recall':num=(tw&accept).sum(1);den=tw.sum(1)
        elif key=='winner:Rect-Win_recall':num=(rw&~accept).sum(1);den=rw.sum(1)
        else:
            mode,name,kind=key.split(':');m=masks[name]
            def field(mode):return dm[mode].astype(float) if kind=='mean_dm' else ((dm[mode]<0).astype(float) if kind=='harm_rate' else ((dm[mode]>0).astype(float) if kind=='benefit_rate' else (norm[mode]>0).astype(float)))
            if '-' in mode:a,b=mode.split('-');v=field(a)-field(b)
            else:v=field(mode)
            num=(v*m).sum(1);den=m.sum(1)
        nums.append(num);dens.append(den)
    nums=np.asarray(nums).T;dens=np.asarray(dens).T;tp=(tw&accept).sum(1);fn=(tw&~accept).sum(1);fp=(rw&accept).sum(1);tn=(rw&~accept).sum(1)
    rng=np.random.default_rng(42);rng_hash=hashlib.sha256();rep=[]
    for _ in range(200):
        ids=rng.integers(0,n,(50,n),dtype=np.int32);rng_hash.update(ids.tobytes())
        den=dens[ids].sum(1);result=nums[ids].sum(1)/den
        ba=.5*(tp[ids].sum(1)/(tp+fn)[ids].sum(1)+tn[ids].sum(1)/(tn+fp)[ids].sum(1))
        rep.append(np.c_[result,ba])
    rep=np.concatenate(rep);saved=np.array([[float(row[k]) for k in keys] for row in rows(report,'bootstrap_replicates')])
    errors['bootstrap_replicate_max_abs']=float(np.nanmax(np.abs(rep-saved)))
    checks['paired_bootstrap_all_replicates']=errors['bootstrap_replicate_max_abs']<1e-7 and rng_hash.hexdigest()==summary['bootstrap_rng_sha256']
    cimax=0.
    for j,row in enumerate(bootstrap):
        ci=np.nanquantile(rep[:,j],[.025,.975]);cimax=max(cimax,abs(ci[0]-float(row['ci_low'])),abs(ci[1]-float(row['ci_high'])))
    errors['bootstrap_ci_max_abs']=cimax;checks['bootstrap_intervals']=cimax<1e-7
    ident=rjson(run/(P+'identity_audit.json'));det=rjson(run/(P+'detach_audit.json'));smoke=rjson(run/(P+'bf16_smoke.json'))
    checks['state_bn_checkpoint_identity']=ident['state_before']==ident['state_after'] and ident['checkpoint_sha_before']==ident['checkpoint_sha_after'] and ident['bn_buffers_unchanged']
    checks['prediction_identity']=ident['prediction_before']==ident['prediction_after'] and ident['max_logit_replay_difference']==0
    checks['teacher_acceptance_q_detached']=all(det[k] for k in ('teacher_detached','q_detached','delta_detached','m_detached','a_detached','ps_source_gradient_none','rect_acceptance_branch_gradient_none'))
    checks['no_optimizer_test_luad']=not rt['optimizer_created'] and rt['optimizer_steps']==0 and not rt['test_access'] and not rt['luad_access']
    checks['bf16_batch20']=smoke['batch']==20 and smoke['all_finite'] and smoke['budget_pass']
    pclasses=rows(report,'per_class');power_ok=True
    for row in pclasses:
        m=masks[row['stratum']];pos=(m&tw).sum();neg=(m&rw).sum();eligible=((m&tw).any(1)&(m&rw).any(1)).sum()
        power_ok &= (row['power']=='SUFFICIENT')==(pos>=500 and neg>=500 and eligible>=30)
    checks['class_power']=power_ok
    au=np.nanmean(ia['winner:all']);gu=np.nanmean(ia['gradient:all']);bootmap={row['metric']:row for row in bootstrap}
    recall=tp.sum()/(tp+fn).sum();protect=tn.sum()/(tn+fp).sum();ba=(recall+protect)/2
    ga=au>=.65 and float(bootmap['winner:all:image_auroc']['ci_low'])>.5 and ba>=.6 and recall>=.55 and protect>=.55
    gb=gu>=.65 and float(bootmap['gradient:all:image_auroc']['ci_low'])>.5
    positive=sum(um['HA',g]['mean_dm']>0 for g in ('all','Top20','class0','class1','class2','class3'))
    gc='PASS' if all(um['HA',g]['benefit_rate']>um['HA',g]['harm_rate'] for g in ('all','Top20')) and positive>=5 else 'FAIL'
    if gc=='PASS' and any(row['power']=='UNDERPOWERED' for row in pclasses):gc='UNDERPOWERED'
    gd=um['HA','Rect_Correct']['harm_rate']<=.5*um['CCA','Rect_Correct']['harm_rate'] and um['HA','Rect_Wrong']['benefit_rate']>=.6 and um['HA','all']['active_fraction']>=.1
    checks['independent_gate_decision']=[bool(ga),bool(gb),gc,bool(gd)]==[summary['gate_A']=='PASS',summary['gate_B']=='PASS',summary['gate_C'],summary['gate_D']=='PASS']
    soft=um['SA','all']['mean_dm']>um['HA','all']['mean_dm'] and um['SA','Rect_Correct']['harm_rate']<=um['HA','Rect_Correct']['harm_rate']+.05
    strong=au>=.75 and gu>=.75 and um['HA','all']['mean_dm']>0 and um['HA','Top20']['mean_dm']>0 and um['HA','Rect_Correct']['harm_rate']<=.25 and um['HA','Rect_Wrong']['benefit_rate']>=.7
    checks['secondary_flags']=bool(soft)==summary['soft_acceptance_promising'] and bool(strong)==summary['strong_acceptance_signal']
    changed=subprocess.check_output(['git','diff','--name-only',rt['a0'],'HEAD'],text=True).splitlines()
    checks['original_sources_unchanged']=all(p.startswith(('docs/','tools/','tests/','audit/')) for p in changed)
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks={k:bool(v) for k,v in checks.items()},errors=errors,
                method='all3418 independent explicit-neighbor CUDA gather; SciPy average ranks/AP; FP64 analytical epsilon-KL; NumPy gather-sum bootstrap',
                support_sign_note='Independent reduction order can differ near FP32 zero. Primary fixed Delta is never replaced or retuned.',
                images=n,resamples=10000,command=shlex.join([sys.executable,*sys.argv]),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['status']=='PASS'


if __name__=='__main__':main()
