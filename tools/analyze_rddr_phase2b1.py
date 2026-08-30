"""Exact native-grid adjudication, frozen strata and paired image bootstrap."""
from __future__ import annotations
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
from tools.rddr_phase2b1_common import (EPS,Q_EDGES,HFRM_GROUPS,ESTIMATORS,SAFETY_BASES,
    sha256,write_json,write_csv,winner_labels,binary_exact,nanmean,divide,sign_scores,
    segmentation_scores,bootstrap_indices,decide)


def make_groups(data,preds):
    y=data["truth"]
    fg=(y>=0)&(y<4)
    sc,dc=preds[0]==y,preds[1]==y
    eligible,label=winner_labels(y,preds[0],preds[1])
    groups={"all":fg,"Top20":fg&data["top20"].astype(bool),"Bottom80":fg&~data["top20"].astype(bool),
            "hard_disagreement":fg&(preds[0]!=preds[1]),"adjudication":eligible,
            "Deep_Win":eligible&label,"Shallow_Win":eligible&~label,
            "Deep_Correct":fg&dc,"Deep_Wrong":fg&~dc,
            "Top20_Deep_Correct":fg&data["top20"].astype(bool)&dc,
            "Top20_Deep_Wrong":fg&data["top20"].astype(bool)&~dc,
            "Shallow_Correct":fg&sc,"Shallow_Wrong":fg&~sc,
            "Top20_Shallow_Correct":fg&data["top20"].astype(bool)&sc,
            "Top20_Shallow_Wrong":fg&data["top20"].astype(bool)&~sc,
            "Both_Wrong":fg&~sc&~dc,"Both_Correct":fg&sc&dc}
    groups.update({name:fg&(data["hfrm"]==i) for i,name in enumerate(HFRM_GROUPS)})
    quintile=np.searchsorted(Q_EDGES,data["q_feature"],side="left")
    groups.update({f"Q{i+1}":fg&(quintile==i) for i in range(5)})
    groups["boundary"]=fg&data["boundary"].astype(bool)
    groups["interior"]=fg&~data["boundary"].astype(bool)
    groups.update({f"class{i}":y==i for i in range(4)})
    strength=np.abs(data["delta"])
    edges=np.quantile(strength[fg],[.2,.4,.6,.8],method="higher")
    sq=np.searchsorted(edges,strength,side="left")
    groups.update({f"Strength{i+1}":fg&(sq==i) for i in range(5)})
    for base in SAFETY_BASES: groups["DW__"+base]=groups[base]&~dc
    return groups,eligible,label,edges


def js_numpy(p,q):
    m=.5*(p+q)
    return .5*((p*(np.log(p+EPS)-np.log(m+EPS))).sum(1)+(q*(np.log(q+EPS)-np.log(m+EPS))).sum(1))


def ci_row(key,observed,samples,eligible,n,aggregation):
    samples=np.asarray(samples)
    finite=samples[np.isfinite(samples)]
    lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else (np.nan,np.nan)
    return dict(metric=key,observed=float(observed),ci95_low=float(lo),ci95_high=float(hi),
                resamples=10000,finite_resamples=len(finite),seed=42,eligible_images=int(eligible),
                resampling_images=n,aggregation=aggregation,unit="fraction")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True); p.add_argument("--output",required=True)
    args=p.parse_args(); tick=time.perf_counter()
    source,out=Path(args.input),Path(args.output)
    if out.exists(): raise FileExistsError(out)
    runtime=json.loads((source/"rddr_phase2b1_runtime.json").read_text(encoding="utf-8"))
    manifest=json.loads((source/"rddr_phase2b1_population_manifest.json").read_text(encoding="utf-8"))
    assert runtime["images"]==3418 and not runtime["smoke"] and runtime["unchanged_model_state"]
    assert runtime["frozen_q_max_abs_difference"]==0
    native=source/"rddr_phase2b1_native_observations.npz"
    assert sha256(native)==runtime["native_observations_sha256"]
    with np.load(native) as archive: data={k:archive[k] for k in archive.files}
    n=len(data["names"]); y=data["truth"].astype(int)
    probs=[data[k] for k in ("ps","pd","fixed_average","anchor","ctx")]
    preds=np.stack([a.argmax(1) for a in probs])
    assert np.array_equal(data["delta"],data["sd"]-data["ss"])
    wd_cpu=data["sd"]/(data["ss"]+data["sd"]+EPS)
    anchor_cpu=(1-data["wd"][:,None])*data["ps"]+data["wd"][:,None]*data["pd"]
    numerical=dict(wd_cpu_cuda_max_difference=float(np.abs(wd_cpu-data["wd"]).max()),
                   anchor_cpu_cuda_max_difference=float(np.abs(anchor_cpu-data["anchor"]).max()),
                   fixed_average_max_difference=float(np.abs(.5*data["ps"]+.5*data["pd"]-data["fixed_average"]).max()))
    assert max(numerical.values())<2e-7,numerical
    del wd_cpu,anchor_cpu
    groups,eligible,label,strength_edges=make_groups(data,preds)
    group_names=list(groups); gi={g:i for i,g in enumerate(group_names)}
    gnum=len(groups)
    cm=np.zeros((n,gnum,5,4,4),np.int64)
    sign_cm=np.zeros((n,gnum,2,2),np.int64)
    image_auc=np.full((n,gnum),np.nan)
    image_ap=np.full((n,gnum),np.nan)
    target_counts=np.zeros((n,gnum),np.int64)
    support_keys=("SS","SD","Delta","wD","JS_ctx_shallow","JS_ctx_deep")
    support_values=[data["ss"],data["sd"],data["delta"],data["wd"],js_numpy(data["ctx"],data["ps"]),js_numpy(data["ctx"],data["pd"])]
    support_mean=np.full((n,gnum,len(support_keys)),np.nan)
    img_id=np.broadcast_to(np.arange(n,dtype=np.int64)[:,None],y.shape)
    enc=[img_id*16+y*4+pred for pred in preds]
    encsign=img_id*4+label.astype(np.int64)*2+(data["delta"]>0).astype(np.int64)
    yclip=np.clip(y,0,3)
    nll,brier=[],[]
    for prob in probs:
        gtprob=np.take_along_axis(prob,yclip[:,None],axis=1)[:,0]
        nll.append(-np.log(gtprob+EPS))
        # Direct nonnegative squared errors avoid cancellation near pGT=1.
        brier.append(sum((prob[:,k]-(y==k))**2 for k in range(4)))
    adjudication,sign_rows,anchor_rows,support_rows,echo_rows=[],[],[],[],[]
    for gidx,(g,mask) in enumerate(groups.items()):
        assert not (mask&~groups["all"]).any()
        target_counts[:,gidx]=mask.sum(1)
        winner=mask&eligible
        pooled=binary_exact(data["delta"][winner],label[winner])
        sign_cm[:,gidx]=np.bincount(encsign[winner],minlength=n*4).reshape(n,2,2)
        for i in range(n):
            m=winner[i]
            counts=sign_cm[i,gidx].sum(1)
            if counts[0] and counts[1]:
                exact=binary_exact(data["delta"][i,m],label[i,m])
                image_auc[i,gidx]=exact["auroc"]; image_ap[i,gidx]=exact["auprc"]
            elif counts[1]: image_ap[i,gidx]=1.
        sign_total=sign_cm[:,gidx].sum(0)
        sm=sign_scores(sign_total)
        adjudication.append(dict(group=g,targets=int(mask.sum()),winner_targets=int(winner.sum()),
                                  **pooled,image_balanced_auroc=float(nanmean(image_auc[:,gidx])),
                                  image_balanced_auprc=float(nanmean(image_ap[:,gidx])),
                                  auroc_eligible_images=int(np.isfinite(image_auc[:,gidx]).sum()),
                                  auroc_excluded_images=int((~np.isfinite(image_auc[:,gidx])).sum()),
                                  hard_disagreement_prevalence=float(divide((mask&groups["hard_disagreement"]).sum(),mask.sum()))))
        sign_rows.append(dict(group=g,targets=int(winner.sum()),**{k:float(v) for k,v in sm.items()},
                              image_balanced_accuracy=float(nanmean(sign_scores(sign_cm[:,gidx])["balanced_accuracy"])),
                              shallow_win_count=int(sign_total[0].sum()),deep_win_count=int(sign_total[1].sum()),
                              tn=int(sign_total[0,0]),fp=int(sign_total[0,1]),fn=int(sign_total[1,0]),tp=int(sign_total[1,1]),
                              exact_zero_ties=int((winner&(data["delta"]==0)).sum())))
        for ei,e in enumerate(ESTIMATORS):
            cm[:,gidx,ei]=np.bincount(enc[ei][mask],minlength=n*16).reshape(n,4,4)
            metrics=segmentation_scores(cm[:,gidx,ei].sum(0))
            anchor_rows.append(dict(group=g,estimator=e,targets=int(mask.sum()),
                                    **{k:float(v) for k,v in metrics.items() if k!="class_iou"},
                                    **{f"class{k}_iou":float(v) for k,v in enumerate(metrics["class_iou"])},
                                    nll=float(divide(nll[ei][mask].sum(dtype=np.float64),mask.sum())),
                                    brier=float(divide(brier[ei][mask].sum(dtype=np.float64),mask.sum()))))
        row=dict(group=g,targets=int(mask.sum()))
        for fi,(key,values) in enumerate(zip(support_keys,support_values)):
            support_mean[:,gidx,fi]=divide(np.where(mask,values,0).sum(1,dtype=np.float64),mask.sum(1))
            row[key+"_pixel_mean"]=float(divide(values[mask].sum(dtype=np.float64),mask.sum()))
            row[key+"_image_mean"]=float(nanmean(support_mean[:,gidx,fi]))
            row[key+"_std"]=float(values[mask].std(dtype=np.float64)) if mask.any() else np.nan
        row["consensus_closer_deep_fraction"]=float(divide((mask&(support_values[5]<support_values[4])).sum(),mask.sum()))
        row["consensus_js_exact_ties"]=int((mask&(support_values[5]==support_values[4])).sum())
        support_rows.append(row)
        echo_rows.append(dict(group=g,targets=int(mask.sum()),
                              anchor_equals_deep=float(divide((mask&(preds[3]==preds[1])).sum(),mask.sum())),
                              anchor_equals_shallow=float(divide((mask&(preds[3]==preds[0])).sum(),mask.sum())),
                              anchor_differs_both_count=int((mask&(preds[3]!=preds[0])&(preds[3]!=preds[1])).sum()),
                              anchor_correct_third_class_count=int((mask&(preds[3]!=preds[0])&(preds[3]!=preds[1])&(preds[3]==y)).sum())))
        if (gidx+1)%10==0: print(f"ANALYZE_GROUPS {gidx+1}/{gnum}",flush=True)
    lookup={(r["group"],r["estimator"]):r for r in anchor_rows}
    for row in anchor_rows:
        for ref in ("fixed_average","shallow","deep"):
            reference=lookup[(row["group"],ref)]
            row["accuracy_minus_"+ref]=row["accuracy"]-reference["accuracy"]
            row["miou_minus_"+ref]=row["miou"]-reference["miou"]
    joined=[]
    for g in group_names:
        a=next(r for r in adjudication if r["group"]==g)
        s=next(r for r in sign_rows if r["group"]==g)
        row=dict(a)
        row.update({"sign_"+k:v for k,v in s.items() if k not in ("group","targets")})
        for e in ESTIMATORS:
            m=lookup[g,e]
            row.update({e+"_"+k:m[k] for k in ("accuracy","miou","dice","nll","brier")})
        row["anchor_fixed_accuracy_delta"]=lookup[g,"anchor"]["accuracy_minus_fixed_average"]
        row["anchor_fixed_miou_delta"]=lookup[g,"anchor"]["miou_minus_fixed_average"]
        row.update({k:v for k,v in next(r for r in support_rows if r["group"]==g).items() if k not in ("group","targets")})
        if "DW__"+g in gi:
            dw=lookup["DW__"+g,"anchor"]
            row["deep_wrong_targets"]=dw["targets"]
            row["deep_wrong_anchor_fixed_accuracy_delta"]=dw["accuracy_minus_fixed_average"]
        joined.append(row)
    safety=[]
    for base in SAFETY_BASES:
        row=lookup["DW__"+base,"anchor"]
        delta=row["accuracy_minus_fixed_average"]
        safety.append(dict(stratum=base,targets=row["targets"],anchor_accuracy=row["accuracy"],
                           fixed_average_accuracy=lookup["DW__"+base,"fixed_average"]["accuracy"],
                           delta=delta,hard_line=-.10,hard_line_failed=bool(np.isfinite(delta) and delta<=-.10),
                           status="DEFINED" if row["targets"] else "UNDEFINED_EMPTY"))
    calibration=[]
    bucket=np.searchsorted([.2,.4,.6,.8],data["wd"],side="right")
    for b in range(5):
        m=eligible&(bucket==b)
        calibration.append(dict(bin=b,low=b*.2,high=(b+1)*.2,upper_inclusive=b==4,
                                targets=int(m.sum()),deep_win=int(label[m].sum()),shallow_win=int((~label[m]).sum()),
                                mean_wD=float(nanmean(data["wd"][m])),empirical_deep_win_probability=float(divide(label[m].sum(),m.sum()))))
    samples={k:[] for k in ("adjudication_image_auroc","sign_balanced_accuracy","anchor_fixed_accuracy_delta",
                            "anchor_fixed_miou_delta","Top20_anchor_fixed_accuracy_delta","Deep_Wrong_anchor_fixed_accuracy_delta",
                            "Top20_Deep_Wrong_anchor_fixed_accuracy_delta")}
    for idx in bootstrap_indices(n):
        samples["adjudication_image_auroc"].extend(nanmean(image_auc[idx,0],axis=1))
        samples["sign_balanced_accuracy"].extend(sign_scores(sign_cm[idx,0].sum(1))["balanced_accuracy"])
        for g,key in (("all","anchor_fixed_accuracy_delta"),("Top20","Top20_anchor_fixed_accuracy_delta"),
                      ("Deep_Wrong","Deep_Wrong_anchor_fixed_accuracy_delta"),("Top20_Deep_Wrong","Top20_Deep_Wrong_anchor_fixed_accuracy_delta")):
            selected=cm[:,gi[g],2:4]
            scores=segmentation_scores(selected[idx].sum(1))
            samples[key].extend(scores["accuracy"][:,1]-scores["accuracy"][:,0])
            if g=="all": samples["anchor_fixed_miou_delta"].extend(scores["miou"][:,1]-scores["miou"][:,0])
    boot=[]
    for key,value in samples.items():
        if key=="adjudication_image_auroc":
            observed=adjudication[0]["image_balanced_auroc"]; count=np.isfinite(image_auc[:,0]).sum(); aggregation="image_balanced"
        elif key=="sign_balanced_accuracy":
            observed=sign_rows[0]["balanced_accuracy"]; count=(sign_cm[:,0].sum((1,2))>0).sum(); aggregation="pooled_sign_confusion"
        else:
            g="Top20_Deep_Wrong" if key.startswith("Top20_Deep") else "Deep_Wrong" if key.startswith("Deep_") else "Top20" if key.startswith("Top20_") else "all"
            observed=lookup[g,"anchor"]["miou_minus_fixed_average" if "miou" in key else "accuracy_minus_fixed_average"]
            count=(target_counts[:,gi[g]]>0).sum(); aggregation="pooled_segmentation_confusion"
        boot.append(ci_row(key,observed,value,count,n,aggregation))
    ci={r["metric"]:r for r in boot}
    a=ci["adjudication_image_auroc"]; b=sign_rows[0]
    ca,cm_i=ci["anchor_fixed_accuracy_delta"],ci["anchor_fixed_miou_delta"]
    dw=lookup["Deep_Wrong","anchor"]["accuracy_minus_fixed_average"]
    tdw=lookup["Top20_Deep_Wrong","anchor"]["accuracy_minus_fixed_average"]
    hard_fail=any(r["hard_line_failed"] for r in safety)
    gates=dict(A=a["observed"]>=.65 and a["ci95_low"]>.50,
               B=b["balanced_accuracy"]>=.60 and b["deep_win_recall"]>=.55 and b["shallow_win_recall"]>=.55,
               C=ca["observed"]>0 and cm_i["observed"]>0 and max(ca["ci95_low"],cm_i["ci95_low"])>0,
               D=dw>=-.02 and tdw>=-.03 and not hard_fail)
    decision=decide(**{k.lower():v for k,v in gates.items()})
    strong=a["observed"]>=.70 and cm_i["observed"]>=.01 and dw>=0
    oracle=np.where(label,preds[1],preds[0])
    assert np.all(oracle[eligible]==y[eligible])
    summary=dict(decision=decision,gates=gates,strong_signal=bool(strong),hard_safety_failed=hard_fail,
                  ci=ci,sign_primary=sign_rows[0],images=n,foreground_targets=int(groups["all"].sum()),
                  adjudication_targets=int(eligible.sum()),deep_win_count=int(label[eligible].sum()),
                  shallow_win_count=int((~label[eligible]).sum()),hard_disagreement_targets=int(groups["hard_disagreement"].sum()),
                  strength_quintile_edges=strength_edges,q_quintile_edges=Q_EDGES,
                  checkpoint_sha256=runtime["checkpoint_sha256"],extraction_commit=runtime["commit"],
                  analysis_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  exact_analysis_command=shlex.join([sys.executable,*sys.argv]),numerical_replay=numerical,
                  native_observations_sha256=runtime["native_observations_sha256"],
                  oracle_winner_accuracy=1.,adjudication_gap_to_oracle=1-sign_rows[0]["accuracy"],
                  no_training=True,no_test=True,no_search=True,stop_after_report=True)
    out.mkdir(parents=True)
    write_json(out/"rddr_phase2b1_population_manifest.json",manifest)
    write_json(out/"rddr_phase2b1_summary.json",summary)
    for name,rows in (("adjudication",adjudication),("sign_decision",sign_rows),("anchor_metrics",anchor_rows),
                      ("support_diagnostics",support_rows),("all_groups",joined),("deep_wrong_safety",safety),
                      ("calibration",calibration),("echo",echo_rows),("bootstrap",boot)):
        write_csv(out/f"rddr_phase2b1_{name}.csv",rows)
    subsets={"conflict_strata":("all","hard_disagreement","adjudication","Top20"),
             "deep_strata":("Deep_Correct","Deep_Wrong","Top20_Deep_Correct","Top20_Deep_Wrong"),
             "shallow_strata":("Shallow_Correct","Shallow_Wrong","Top20_Shallow_Correct","Top20_Shallow_Wrong"),
             "both_wrong":("Both_Wrong",),"hfrm_groups":HFRM_GROUPS,"top20_bottom80":("Top20","Bottom80"),
             "quintiles":tuple(f"Q{i}" for i in range(1,6)),"strength_quintiles":tuple(f"Strength{i}" for i in range(1,6)),
             "boundary_interior":("boundary","interior"),"per_class":tuple(f"class{i}" for i in range(4)),
             "context_consensus":("all","Deep_Win","Shallow_Win","Both_Wrong","Top20")}
    for key,names in subsets.items(): write_csv(out/f"rddr_phase2b1_{key}.csv",[r for r in joined if r["group"] in names])
    write_csv(out/"rddr_phase2b1_bootstrap_replicates.csv",[dict(replicate=i,**{k:v[i] for k,v in samples.items()}) for i in range(10000)])
    per_image=[]
    for i,name in enumerate(data["names"]):
        row=dict(image_id=name,foreground_targets=int(target_counts[i,0]),image_auroc=image_auc[i,0],
                 image_auprc=image_ap[i,0],sign_confusion=json.dumps(sign_cm[i,0].tolist(),separators=(",",":")))
        for eidx,e in enumerate(ESTIMATORS): row[e+"_confusion"]=json.dumps(cm[i,0,eidx].tolist(),separators=(",",":"))
        for g in ("Top20","Deep_Wrong","Top20_Deep_Wrong"):
            for eidx,e in ((2,"fixed_average"),(3,"anchor")):
                row[g+"_"+e+"_confusion"]=json.dumps(cm[i,gi[g],eidx].tolist(),separators=(",",":"))
        for gidx,g in enumerate(group_names):
            row[g+"_targets"]=int(target_counts[i,gidx])
            row[g+"_image_auroc"]=image_auc[i,gidx]
            if g.startswith("DW__"):
                row[g+"_fixed_correct"]=int(np.trace(cm[i,gidx,2]))
                row[g+"_anchor_correct"]=int(np.trace(cm[i,gidx,3]))
        per_image.append(row)
    write_csv(out/"rddr_phase2b1_per_image.csv",per_image)
    np.savez_compressed(out/"rddr_phase2b1_sufficient_statistics.npz",names=data["names"],group_names=np.array(group_names),
                        cm=cm,sign_cm=sign_cm,image_auc=image_auc,image_ap=image_ap,target_counts=target_counts,support_mean=support_mean)
    runtime.update(analysis_seconds=time.perf_counter()-tick,analysis_command=summary["exact_analysis_command"],
                   analysis_commit=summary["analysis_commit"],sufficient_statistics_sha256=sha256(out/"rddr_phase2b1_sufficient_statistics.npz"))
    write_json(out/"rddr_phase2b1_runtime.json",runtime)
    print(json.dumps(dict(decision=decision,gates=gates,strong_signal=bool(strong),ci=ci),indent=2),flush=True)


if __name__=="__main__": main()
