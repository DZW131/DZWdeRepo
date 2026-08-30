"""Full fixed-stratum analysis and 10k paired image bootstrap; no model code."""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b1_common import (EPS,sha256,write_json,write_csv,binary_exact,
    divide,nanmean,sign_scores,segmentation_scores,bootstrap_indices)
from tools.rddr_phase2b15_common import (CACHE_SHA,PREFIX,SCORES,ESTIMATORS,
    make_groups,third_class_metrics,class_status,aggregate_class_status,decide)

SUPPORT=("T_SS","T_SD","T_DS","T_DD","B_S","B_D","B_family","old","dsrc","sym","wD_sym")
CONTEXTS=("ctx_S","ctx_D","ctx_sym")


def describe(x):
    x=np.asarray(x,dtype=np.float64)
    if not x.size:return dict(mean=np.nan,std=np.nan,median=np.nan,p05=np.nan,p25=np.nan,p50=np.nan,p75=np.nan,p95=np.nan)
    q=np.quantile(x,[.05,.25,.5,.75,.95])
    return dict(mean=float(x.mean()),std=float(x.std()),median=float(q[2]),
                **{key:float(v) for key,v in zip(("p05","p25","p50","p75","p95"),q)})


def ci_row(key,observed,values,n,aggregation):
    values=np.asarray(values);f=values[np.isfinite(values)]
    lo,hi=np.quantile(f,[.025,.975]) if len(f) else (np.nan,np.nan)
    return dict(metric=key,observed=float(observed),ci95_low=float(lo),ci95_high=float(hi),
                resamples=10000,finite_resamples=len(f),seed=42,resampling_images=n,aggregation=aggregation)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native",required=True);p.add_argument("--derived",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args();tick=time.perf_counter();out=Path(args.output);derived=Path(args.derived)
    if out.exists():raise FileExistsError(out)
    runtime=json.loads((derived/(PREFIX+"runtime.json")).read_text())
    assert runtime["images"]==3418 and not runtime["smoke"]
    assert sha256(args.native)==CACHE_SHA
    assert sha256(derived/(PREFIX+"derived_observations.npz"))==runtime["derived_sha256"]
    with np.load(args.native,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(derived/(PREFIX+"derived_observations.npz"),allow_pickle=False) as z:r={k:z[k] for k in z.files}
    assert np.array_equal(data["names"],r["names"])
    groups,eligible,label,s,d=make_groups(data);gn=list(groups);gi={g:i for i,g in enumerate(gn)}
    n=len(s);ng=len(gn);y=data["truth"].astype(np.int64)
    probabilities=[data["ps"],data["pd"],data["fixed_average"],data["anchor"],r["anchor_sym"],r["ctx_S"],r["ctx_D"],r["ctx_sym"]]
    preds=[x.argmax(1) for x in probabilities]
    scores=[r[k] if k in ("old","dsrc","sym") else r["delta_"+k] for k in SCORES]
    image_auc=np.full((n,ng,6),np.nan);image_ap=np.full_like(image_auc,np.nan)
    sign_cm=np.zeros((n,ng,6,2,2),np.int64);cm=np.zeros((n,ng,8,4,4),np.int64)
    count=np.zeros((n,ng),np.int64);sums=np.zeros((n,ng,len(SUPPORT)),np.float64)
    support_rows=[];adj_rows=[];anchor_rows=[];all_rows=[];context_rows=[]
    imgid=np.broadcast_to(np.arange(n,dtype=np.int64)[:,None],y.shape)
    encseg=[imgid*16+y*4+pred for pred in preds]
    encsign=[imgid*4+label.astype(np.int64)*2+(sc>0) for sc in scores]
    nll=[];brier=[]
    for prob in probabilities:
        pgt=np.take_along_axis(prob,np.clip(y,0,3)[:,None],axis=1)[:,0].astype(np.float64)
        nll.append(-np.log(pgt+EPS))
        brier.append(sum((prob[:,k].astype(np.float64)-(y==k))**2 for k in range(4)))
    for j,(g,m) in enumerate(groups.items()):
        cnt=int(m.sum());count[:,j]=m.sum(1);win=m&eligible
        merged=dict(group=g,targets=cnt,deep_win_count=int((win&label).sum()),
                    shallow_win_count=int((win&~label).sum()),both_wrong_count=int((m&groups["Both_Wrong"]).sum()))
        for k,key in enumerate(SUPPORT):
            sums[:,j,k]=np.bincount(imgid[m],weights=r[key][m],minlength=n)
            stat=describe(r[key][m]);support_rows.append(dict(group=g,field=key,targets=cnt,**stat))
            merged[key+"_mean"]=stat["mean"]
            if key in ("old","sym"):merged[key+"_median"]=stat["median"]
        merged["bias_shrink"]=1-abs(merged["sym_mean"])/(abs(merged["old_mean"])+EPS)
        for k,name in enumerate(SCORES):
            sign_cm[:,j,k]=np.bincount(encsign[k][win],minlength=n*4).reshape(n,2,2)
            for i in range(n):
                labels=label[i,win[i]]
                if labels.size and labels.any():
                    if labels.all():image_ap[i,j,k]=1.
                    else:
                        exact=binary_exact(scores[k][i,win[i]],labels)
                        image_auc[i,j,k]=exact["auroc"];image_ap[i,j,k]=exact["auprc"]
            pooled=binary_exact(scores[k][win],label[win]);sign=sign_scores(sign_cm[:,j,k].sum(0))
            arow=dict(group=g,score=name,targets=cnt,winner_targets=int(win.sum()),
                      mean_score=float(nanmean(scores[k][m])),**pooled,
                      image_auroc=float(nanmean(image_auc[:,j,k])),image_auprc=float(nanmean(image_ap[:,j,k])),
                      auc_images=int(np.isfinite(image_auc[:,j,k]).sum()),
                      **{key:float(val) for key,val in sign.items()},zero_ties=int((scores[k][win]==0).sum()))
            adj_rows.append(arow)
            for key in ("auroc","image_auroc","balanced_accuracy","deep_win_recall","shallow_win_recall"):
                merged[name+"_"+key]=arow[key]
        for k,name in enumerate(ESTIMATORS):
            cm[:,j,k]=np.bincount(encseg[k][m],minlength=n*16).reshape(n,4,4)
            sm=segmentation_scores(cm[:,j,k].sum(0))
            row=dict(group=g,estimator=name,targets=cnt,accuracy=float(sm["accuracy"]),miou=float(sm["miou"]),
                     dice=float(sm["dice"]),nll=float(nanmean(nll[k][m])),brier=float(nanmean(brier[k][m])))
            row.update({f"iou_class{c}":float(sm["class_iou"][c]) for c in range(4)})
            anchor_rows.append(row)
            merged[name+"_accuracy"]=row["accuracy"];merged[name+"_miou"]=row["miou"]
        for k,name in enumerate(CONTEXTS):
            cp=preds[k+5]
            third=third_class_metrics(y,s,d,cp,m)
            context_rows.append(dict(group=g,context=name,**third,
                equals_shallow=float(divide(((cp==s)&m).sum(),cnt)),equals_deep=float(divide(((cp==d)&m).sum(),cnt)),
                accuracy=merged[name+"_accuracy"],miou=merged[name+"_miou"]))
        for ref in ("fixed_average","anchor_old"):
            for metric in ("accuracy","miou"):
                merged[f"anchor_sym_minus_{ref}_{metric}"]=merged["anchor_sym_"+metric]-merged[ref+"_"+metric]
        merged["pair_support"]="LOW_SUPPORT" if g.startswith("pair") and min(merged["deep_win_count"],merged["shallow_win_count"])<100 else "DEFINED"
        all_rows.append(merged)
        if j%10==0:print(json.dumps(dict(analysis_group=g,index=j,total=ng)),flush=True)
    lookup={(x["group"],x["estimator"]):x for x in anchor_rows}
    ag={(x["group"],x["score"]):x for x in adj_rows}
    cr={(x["group"],x["context"]):x for x in context_rows}
    ar={x["group"]:x for x in all_rows}
    # Freeze supplementary diagnostics; no resulting value feeds a probe.
    confidence=[]
    for g in ["all"]+[f"class{k}" for k in range(4)]:
        m=groups[g];ct=int(m.sum())
        for branch,prob,pred in (("shallow",data["ps"],s),("deep",data["pd"],d)):
            entropy=-(prob.astype(np.float64)*np.log(prob.astype(np.float64)+EPS)).sum(1)
            row=dict(group=g,branch=branch,targets=ct,mean_max_confidence=float(nanmean(prob.max(1)[m])),
                     entropy_nats=float(nanmean(entropy[m])),entropy_normalized=float(nanmean(entropy[m]))/np.log(4))
            for k in range(4):
                row[f"predicted_frequency_class{k}"]=float(divide(((pred==k)&m).sum(),ct))
                row[f"mean_probability_class{k}"]=float(nanmean(prob[:,k][m]))
            confidence.append(row)
    mass={}
    for name in ("S","D"):
        ctx=r["ctx_"+name]
        mass["M_s_"+name]=np.take_along_axis(ctx,s[:,None],axis=1)[:,0]
        mass["M_d_"+name]=np.take_along_axis(ctx,d[:,None],axis=1)[:,0]
        mass["margin_"+name]=mass["M_d_"+name]-mass["M_s_"+name]
    diag_groups={g:groups[g]&groups["hard_disagreement"] for g in gn}
    for c in (2,3):
        for state in ("Deep_Win","Shallow_Win","Both_Wrong"):
            diag_groups[f"class{c}_{state}"]=groups[f"class{c}"]&groups[state]&groups["hard_disagreement"]
        for a in range(4):
            for b in range(4):
                if a!=b:diag_groups[f"class{c}_pair{a}_{b}"]=groups[f"class{c}"]&groups[f"pair{a}_{b}"]
    mass_rows=[];gt_rows=[]
    gtkeys=[k for k in r if k.startswith("GT_")]
    for g,m in diag_groups.items():
        countg=int(m.sum())
        mass_rows.append(dict(group=g,targets=countg,**{key:float(nanmean(v[m])) for key,v in mass.items()}))
        gt_rows.append(dict(group=g,targets=countg,**{key:float(nanmean(r[key][m])) for key in gtkeys}))
    mass_lookup={x["group"]:x for x in mass_rows};gt_lookup={x["group"]:x for x in gt_rows}
    # Context-rescue identities hold on these exact populations; not independent endpoints.
    for name in CONTEXTS:
        assert cr["Both_Wrong",name]["rescue_rate"]==lookup["Both_Wrong",name]["accuracy"]
        assert cr["adjudication",name]["intrusion_rate"]==cr["adjudication",name]["harm_rate"]
    gsum=sum(r[k] for k in ("GT_shallow_candidate_fraction","GT_deep_candidate_fraction","GT_other_fraction","GT_background_fraction","GT_ignore_fraction"))
    assert np.max(abs(gsum[groups["hard_disagreement"]]-1))<3e-7
    # Bootstrap sufficient statistics: image clusters, not individual targets.
    third=np.zeros((n,2,3),np.int64) # populations BothWrong, oneCorrect; n, rescue, harm
    cp=preds[7];different=(cp!=s)&(cp!=d)
    for k,g in enumerate(("Both_Wrong","adjudication")):
        m=groups[g];third[:,k,0]=m.sum(1)
        third[:,k,1]=(m&different&(cp==y)).sum(1)
        third[:,k,2]=(m&different&(cp!=y)).sum(1)
    observed={};aggregation={};samples={}
    def register(key,val,agg):observed[key]=float(val);aggregation[key]=agg;samples[key]=[]
    for g in gn:
        for key in ("B_S","B_D","B_family"):
            register(g+"__mean_"+key,ar[g][key+"_mean"],"pooled_target_mean")
    for score in ("old","sym"):
        register(score+"_image_auroc",ag["all",score]["image_auroc"],"image_balanced")
        for key in ("balanced_accuracy","deep_win_recall","shallow_win_recall"):
            register(score+"_"+key,ag["all",score][key],"pooled_sign_confusion")
    register("sym_minus_old_image_auroc",observed["sym_image_auroc"]-observed["old_image_auroc"],"paired_image_means")
    for key in ("balanced_accuracy","deep_win_recall","shallow_win_recall"):
        register("sym_minus_old_"+key,observed["sym_"+key]-observed["old_"+key],"paired_pooled_sign_confusion")
    for c in (2,3):register(f"class{c}_sym_image_auroc",ag[f"class{c}","sym"]["image_auroc"],"image_balanced")
    for ref in ("fixed_average","anchor_old"):
        for key in ("accuracy","miou"):
            register(f"anchor_sym_minus_{ref}_{key}",lookup["all","anchor_sym"][key]-lookup["all",ref][key],"paired_pooled_segmentation")
    register("ctx_sym_Both_Wrong_accuracy",lookup["Both_Wrong","ctx_sym"]["accuracy"],"pooled_target_ratio")
    register("ctx_sym_ThirdClassRescueRate",cr["Both_Wrong","ctx_sym"]["rescue_rate"],"pooled_target_ratio")
    register("ctx_sym_ThirdClassHarmRate",cr["adjudication","ctx_sym"]["harm_rate"],"pooled_target_ratio")
    register("mean_Delta_old",ar["all"]["old_mean"],"pooled_target_mean")
    register("mean_Delta_sym",ar["all"]["sym_mean"],"pooled_target_mean")
    register("BiasShrink",ar["all"]["bias_shrink"],"ratio_of_paired_target_means")
    bidx=[SUPPORT.index(k) for k in ("B_S","B_D","B_family")]
    for indices in bootstrap_indices(n):
        w=np.array([np.bincount(ix,minlength=n) for ix in indices],dtype=np.float64)
        res_count=w@count;res_sums=(w@sums[:,:,bidx].reshape(n,-1)).reshape(len(w),ng,3)
        means=divide(res_sums,res_count[:,:,None])
        chunk={}
        for j,g in enumerate(gn):
            for k,key in enumerate(("B_S","B_D","B_family")):chunk[g+"__mean_"+key]=means[:,j,k]
        for score,k in (("old",0),("sym",2)):
            vals=image_auc[:,0,k]
            chunk[score+"_image_auroc"]=divide(w@np.nan_to_num(vals),w@np.isfinite(vals).astype(float))
            rs=sign_scores((w@sign_cm[:,0,k].reshape(n,4)).reshape(-1,2,2))
            for key in ("balanced_accuracy","deep_win_recall","shallow_win_recall"):chunk[score+"_"+key]=rs[key]
        chunk["sym_minus_old_image_auroc"]=chunk["sym_image_auroc"]-chunk["old_image_auroc"]
        for key in ("balanced_accuracy","deep_win_recall","shallow_win_recall"):
            chunk["sym_minus_old_"+key]=chunk["sym_"+key]-chunk["old_"+key]
        for c in (2,3):
            vals=image_auc[:,gi[f"class{c}"],2]
            chunk[f"class{c}_sym_image_auroc"]=divide(w@np.nan_to_num(vals),w@np.isfinite(vals).astype(float))
        seg=segmentation_scores((w@cm[:,0].reshape(n,-1)).reshape(-1,8,4,4))
        for ref in ("fixed_average","anchor_old"):
            for key in ("accuracy","miou"):
                chunk[f"anchor_sym_minus_{ref}_{key}"]=seg[key][:,4]-seg[key][:,ESTIMATORS.index(ref)]
        tt=(w@third.reshape(n,-1)).reshape(-1,2,3)
        chunk["ctx_sym_Both_Wrong_accuracy"]=divide(tt[:,0,1],tt[:,0,0])
        chunk["ctx_sym_ThirdClassRescueRate"]=chunk["ctx_sym_Both_Wrong_accuracy"].copy()
        chunk["ctx_sym_ThirdClassHarmRate"]=divide(tt[:,1,2],tt[:,1,0])
        old_sym=w@sums[:,0,[SUPPORT.index("old"),SUPPORT.index("sym")]]
        old_sym=divide(old_sym,res_count[:,0,None])
        chunk["mean_Delta_old"],chunk["mean_Delta_sym"]=old_sym[:,0],old_sym[:,1]
        chunk["BiasShrink"]=1-abs(old_sym[:,1])/(abs(old_sym[:,0])+EPS)
        assert set(chunk)==set(samples)
        for key in samples:samples[key].extend(chunk[key])
    boot=[ci_row(k,observed[k],samples[k],n,aggregation[k]) for k in samples];ci={r["metric"]:r for r in boot}
    statuses={f"class{c}":class_status(ag[f"class{c}","sym"]["positive"],ag[f"class{c}","sym"]["negative"],ag[f"class{c}","sym"]["image_auroc"]) for c in (2,3)}
    c=aggregate_class_status(list(statuses.values()))
    a=all(ci["all__mean_"+key]["observed"]>0 and ci["all__mean_"+key]["ci95_low"]>0 for key in ("B_S","B_D"))
    sym=ag["all","sym"]
    b=(sym["image_auroc"]>=.70 and sym["balanced_accuracy"]>=.62 and sym["deep_win_recall"]>=.55
       and sym["shallow_win_recall"]>=.55 and abs(ar["all"]["sym_mean"])<.5*abs(ar["all"]["old_mean"]))
    bw=ci["ctx_sym_Both_Wrong_accuracy"]
    dg=bw["observed"]>=.25 and bw["ci95_low"]>.20 and observed["ctx_sym_ThirdClassRescueRate"]>=.15 and observed["ctx_sym_ThirdClassHarmRate"]<=.10
    strong_sym=sym["image_auroc"]>=.75 and sym["balanced_accuracy"]>=.65 and min(sym["deep_win_recall"],sym["shallow_win_recall"])>=.60
    strong_third=bw["observed"]>=.30 and observed["ctx_sym_ThirdClassRescueRate"]>=.20 and observed["ctx_sym_ThirdClassHarmRate"]<=.08
    summary=dict(decision=decide(a,b,c,dg),gates=dict(A=bool(a),B=bool(b),C=c,D=bool(dg)),class_status=statuses,
                 strong_symmetry_signal=bool(strong_sym),strong_third_evidence_signal=bool(strong_third),third_evidence_supported=bool(dg),
                 images=n,foreground_targets=int(groups["all"].sum()),winner_targets=int(eligible.sum()),
                 old_primary=ag["all","old"],sym_primary=sym,all_group=ar["all"],ci=ci,
                 old_decision_unchanged="RDDR_PHASE2B1_NOGO",identities=dict(rescue_equals_both_wrong_accuracy=True,intrusion_equals_one_correct_harm=True),
                 source_cache_sha256=CACHE_SHA,no_training=True,no_test=True,no_search=True,stop_after_report=True)
    rootcause=[]
    for cnum in (2,3):
        g=f"class{cnum}";cg=groups[g];hard=cg&groups["hard_disagreement"]
        for name,m in [(g,cg)]+[(f"{g}_pair{a}_{b}",cg&groups[f"pair{a}_{b}"]) for a in range(4) for b in range(4) if a!=b]:
            win=m&eligible;ct=int(m.sum());row=dict(group=name,gt_class=cnum,targets=ct,
                deep_win_count=int((win&label).sum()),shallow_win_count=int((win&~label).sum()),
                hard_pair_fraction=float(divide((m&hard).sum(),hard.sum())),class_evidence_status=statuses[g])
            for key in SUPPORT:row[key+"_mean"]=float(nanmean(r[key][m]))
            for key in ("old","sym"):
                bx=binary_exact(r[key][win],label[win]);row[key+"_pooled_auroc"]=bx["auroc"]
                av=[]
                for i in range(n):
                    yy=label[i,win[i]]
                    if yy.any() and not yy.all():av.append(binary_exact(r[key][i,win[i]],yy)["auroc"])
                row[key+"_image_auroc"]=float(nanmean(av))
            row.update({k:v for k,v in mass_lookup[name].items() if k not in ("group","targets")})
            row.update({k:v for k,v in gt_lookup[name].items() if k not in ("group","targets")})
            row["mass_gt_targets"]=mass_lookup[name]["targets"]
            row["pair_support"]="LOW_SUPPORT" if "pair" in name and min(row["deep_win_count"],row["shallow_win_count"])<100 else "NOT_LOW_SUPPORT"
            rootcause.append(row)
    for row in all_rows:
        if row["group"].startswith("pair"):
            row["hard_pair_prevalence"]=float(divide(row["targets"],groups["hard_disagreement"].sum()))
    out.mkdir(parents=True)
    write_json(out/(PREFIX+"summary.json"),summary)
    for key,rows in (("support_matrix",[x for x in support_rows if x["field"].startswith("T_")]),
                     ("same_family_bias",[dict(x,ci95_low=ci[x["group"]+"__mean_"+x["field"]]["ci95_low"],ci95_high=ci[x["group"]+"__mean_"+x["field"]]["ci95_high"]) for x in support_rows if x["field"].startswith("B_")]),
                     ("source_branch_reversal",[x for x in adj_rows if x["score"] in ("old","dsrc")]),
                     ("symmetric_adjudication",[x for x in adj_rows if x["score"] in ("old","sym")]),
                     ("symmetric_anchor",anchor_rows),("class_prior_confidence",confidence),
                     ("candidate_mass",mass_rows),("gt_context_availability",gt_rows),("class23_root_cause",rootcause),
                     ("context_sources",context_rows),("both_wrong_rescue",[x for x in context_rows if x["group"] in ("Both_Wrong","Top20_Both_Wrong")]),
                     ("one_correct_intrusion",[x for x in context_rows if x["group"]=="adjudication"]),
                     ("three_state_roles",[x for x in all_rows if x["group"] in ("Both_Correct","Deep_Win","Shallow_Win","Both_Wrong")]),
                     ("context_winner",[x for x in adj_rows if x["score"] in CONTEXTS]),
                     ("all_groups",all_rows),("bootstrap",boot)):
        write_csv(out/(PREFIX+key+".csv"),rows)
    for key,names in (("safety_strata",[g for g in gn if g.startswith(("Deep_Correct","Deep_Wrong","Shallow_Correct","Shallow_Wrong","Top20_Deep","Top20_Shallow"))]),
                      ("per_class",[f"class{k}" for k in range(4)]),("ordered_pairs",[g for g in gn if g.startswith("pair")]),
                      ("boundary_interior",["boundary","interior"]),("quintiles",[f"Q{k}" for k in range(1,6)])):
        rows=[]
        for g in names:
            rows.append(dict(ar[g],ctx_sym_Both_Wrong_rescue=float(divide(((cp==y)&groups[g]&groups["Both_Wrong"]).sum(),(groups[g]&groups["Both_Wrong"]).sum()))))
        write_csv(out/(PREFIX+key+".csv"),rows)
    write_csv(out/(PREFIX+"bootstrap_replicates.csv"),[dict(replicate=i,**{k:v[i] for k,v in samples.items()}) for i in range(10000)])
    per=[]
    for i,name in enumerate(data["names"]):
        row=dict(image_id=name,foreground_targets=int(count[i,0]))
        for j,g in enumerate(gn):
            row[g+"_targets"]=int(count[i,j])
            for key in ("B_S","B_D","B_family","old","sym"):row[g+"_"+key+"_sum"]=sums[i,j,SUPPORT.index(key)]
        for key,k in (("old",0),("sym",2)):
            row[key+"_image_auroc"]=image_auc[i,0,k]
            row[key+"_sign_confusion"]=json.dumps(sign_cm[i,0,k].tolist(),separators=(",",":"))
        for cnum in (2,3):row[f"class{cnum}_sym_image_auroc"]=image_auc[i,gi[f"class{cnum}"],2]
        for k,name in enumerate(ESTIMATORS):row[name+"_confusion"]=json.dumps(cm[i,0,k].tolist(),separators=(",",":"))
        for k,g in enumerate(("Both_Wrong","adjudication")):
            row[g+"_ctx_sym_third_counts"]=json.dumps(third[i,k].tolist(),separators=(",",":"))
        per.append(row)
    write_csv(out/(PREFIX+"per_image.csv"),per)
    statfile=out/(PREFIX+"sufficient_statistics.npz")
    np.savez_compressed(statfile,names=data["names"],groups=np.array(gn),support_fields=np.array(SUPPORT),count=count,
                        sums=sums,image_auc=image_auc,image_ap=image_ap,sign_cm=sign_cm,cm=cm,third=third)
    runtime.update(analysis_seconds=time.perf_counter()-tick,analysis_command=shlex.join([sys.executable,*sys.argv]),
                   analysis_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                   statistics_sha256=sha256(statfile))
    write_json(out/(PREFIX+"runtime.json"),runtime)
    print(json.dumps(dict(decision=summary["decision"],gates=summary["gates"],class_status=statuses,
                         sym_primary=sym,primary_ci={k:ci[k] for k in ci if not "__" in k}),indent=2),flush=True)


if __name__=="__main__":main()
