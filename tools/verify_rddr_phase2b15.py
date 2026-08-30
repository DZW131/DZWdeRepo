"""Independent NumPy/SciPy reconstruction; no imports from audit helpers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

P="rddr_phase2b15_"


def read(root,name):
    with (root/(P+name+".csv")).open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
    return h.hexdigest()


def auc(score,positive):
    m=int(positive.sum());k=len(positive)-m
    if not m or not k:return np.nan
    return (rankdata(score,method="average")[positive].sum()-m*(m+1)/2)/(m*k)


def ap(score,label):
    if not label.sum():return np.nan
    order=np.argsort(-score,kind="stable");s=score[order];y=label[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(y,dtype=np.int64)[ends]
    return (np.diff(np.r_[0,tp])*tp/(ends+1)).sum()/label.sum()


def ratio(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)


def seg(cm):
    cm=np.asarray(cm,float);d=np.diagonal(cm,axis1=-2,axis2=-1)
    u=cm.sum(-1)+cm.sum(-2)-d
    return dict(accuracy=ratio(d.sum(-1),cm.sum(axis=(-2,-1))),miou=np.nanmean(ratio(d,u),axis=-1))


def js(p,q):
    p,q=np.asarray(p,np.float64),np.asarray(q,np.float64);m=.5*(p+q)
    return .5*((p*(np.log(p+1e-8)-np.log(m+1e-8))).sum(-1)+(q*(np.log(q+1e-8)-np.log(m+1e-8))).sum(-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",required=True);parser.add_argument("--native",required=True)
    parser.add_argument("--derived",required=True)
    args=parser.parse_args();root=Path(args.report);target=root/(P+"verification.json")
    if target.exists():raise FileExistsError(target)
    required=("summary.json","per_image.csv","support_matrix.csv","same_family_bias.csv","source_branch_reversal.csv",
              "symmetric_adjudication.csv","symmetric_anchor.csv","safety_strata.csv","per_class.csv","ordered_pairs.csv",
              "class_prior_confidence.csv","candidate_mass.csv","gt_context_availability.csv","class23_root_cause.csv",
              "context_sources.csv","both_wrong_rescue.csv","one_correct_intrusion.csv","three_state_roles.csv",
              "context_winner.csv","boundary_interior.csv","quintiles.csv","bootstrap.csv","runtime.json")
    assert all((root/(P+x)).is_file() for x in required)
    summary=json.loads((root/(P+"summary.json")).read_text());rt=json.loads((root/(P+"runtime.json")).read_text())
    assert sha(args.native)==rt["native_sha256"] and sha(args.derived)==rt["derived_sha256"]
    assert sha(root/(P+"sufficient_statistics.npz"))==rt["statistics_sha256"]
    with np.load(args.native,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    with np.load(args.derived,allow_pickle=False) as z:r={k:z[k] for k in z.files}
    with np.load(root/(P+"sufficient_statistics.npz"),allow_pickle=False) as z:st={k:z[k] for k in z.files}
    n=len(data["names"]);assert n==3418 and np.array_equal(data["names"],r["names"])
    y=data["truth"];fg=y<4;s=data["ps"].argmax(1);d=data["pd"].argmax(1)
    sc,dc=s==y,d==y;hard=fg&(s!=d);win=fg&(sc!=dc);top=fg&data["top20"].astype(bool)
    g=dict(all=fg,hard_disagreement=hard,adjudication=win,Deep_Win=win&dc,Shallow_Win=win&sc,
           Both_Wrong=fg&~sc&~dc,Both_Correct=fg&sc&dc,Top20=top,Bottom80=fg&~top,Top20_Both_Wrong=top&~sc&~dc)
    for name,m in (("Deep_Correct",dc),("Deep_Wrong",~dc),("Shallow_Correct",sc),("Shallow_Wrong",~sc)):
        g[name]=fg&m;g["Top20_"+name]=top&m
    for c in range(4):g[f"class{c}"]=y==c
    g["boundary"]=fg&data["boundary"].astype(bool);g["interior"]=fg&~data["boundary"].astype(bool)
    q=np.searchsorted([.020935675129294395,.072734534740448,.163648784160614,.3369627296924591],data["q_feature"],side="left")
    for k in range(5):g[f"Q{k+1}"]=fg&(q==k)
    for k,name in enumerate(("Corrected_by_CH","Still_Wrong","Harmed_by_CH","Stable_Correct")):g[name]=fg&(data["hfrm"]==k)
    for a in range(4):
        for b in range(4):
            if a!=b:g[f"pair{a}_{b}"]=hard&(s==a)&(d==b)
    assert list(g)==list(st["groups"])
    for j,name in enumerate(g):np.testing.assert_array_equal(g[name].sum(1),st["count"][:,j])
    np.testing.assert_array_equal(sum(g[f"pair{a}_{b}"].astype(int) for a in range(4) for b in range(4) if a!=b),hard.astype(int))
    for key,old in (("T_SS","ss"),("T_DS","sd"),("old","delta"),("ctx_S","ctx")):
        assert np.max(abs(r[key]-data[old]))<=1e-7
    assert np.array_equal(r["B_S"],r["T_SS"]-r["T_SD"])
    assert np.array_equal(r["B_D"],r["T_DD"]-r["T_DS"])
    assert np.array_equal(r["B_family"],.5*(r["B_S"]+r["B_D"]))
    syms=.5*(r["T_SS"]+r["T_SD"]);symd=.5*(r["T_DS"]+r["T_DD"])
    np.testing.assert_array_equal(r["sym"],symd-syms)
    wd=symd/(syms+symd+1e-8)
    np.testing.assert_array_equal(wd,r["wD_sym"])
    np.testing.assert_allclose(r["anchor_sym"],(1-wd[:,None])*data["ps"]+wd[:,None]*data["pd"],rtol=0,atol=0)
    np.testing.assert_array_equal(r["ctx_sym"],.5*(r["ctx_S"]+r["ctx_D"]))
    support_errors=[];context_errors=[];gt_errors=[]
    for i in (0,1708,3417):
        for ty,tx in ((0,0),(14,14),(27,10)):
            loc=ty*28+tx
            ids=[yy*28+xx for yy in range(max(0,ty-7),min(28,ty+8)) for xx in range(max(0,tx-7),min(28,tx+8)) if (yy,xx)!=(ty,tx)]
            for source,key in (("ps","S"),("pd","D")):
                evidence=data[source][i][:,ids].T
                context_errors.append(float(abs(evidence.astype(float).mean(0)-r["ctx_"+key][i,:,loc]).max()))
                for hypothesis,letter in (("ps","S"),("pd","D")):
                    v=np.clip(1-js(data[hypothesis][i,:,loc],evidence)/np.log(2),0,1).mean()
                    support_errors.append(abs(v-r["T_"+letter+key][i,loc]))
            labels=y[i,ids];sv,dv=s[i,loc],d[i,loc]
            expected=dict(GT_same_fraction=np.mean(labels==y[i,loc]),GT_shallow_candidate_fraction=np.mean(labels==sv),
                          GT_deep_candidate_fraction=np.mean(labels==dv),GT_other_fraction=np.mean((labels<4)&(labels!=sv)&(labels!=dv)),
                          GT_background_fraction=np.mean(labels==4),GT_ignore_fraction=np.mean(labels==255))
            gt_errors.extend(abs(value-r[key][i,loc]) for key,value in expected.items())
    assert max(support_errors)<5e-7 and max(context_errors)<2e-7 and max(gt_errors)<3e-7
    table={(x["group"],x["score"]):x for x in read(root,"symmetric_adjudication")+read(root,"context_winner")+read(root,"source_branch_reversal")}
    scores={key:r[key] for key in ("old","dsrc","sym")}
    scores.update({key:r["delta_"+key] for key in ("ctx_S","ctx_D","ctx_sym")})
    verified_auc_images=0
    for group in ["all","class2","class3"]+[f"pair{a}_{b}" for a in range(4) for b in range(4) if a!=b]:
        j=list(g).index(group);mask=g[group]&win
        for key in (scores if group=="all" else ("old","sym")):
            score=scores[key];k=list(scores).index(key)
            row=table[group,key];direct=auc(score[mask],dc[mask])
            np.testing.assert_allclose(direct,float(row["auroc"]),rtol=0,atol=1e-12,equal_nan=True)
            np.testing.assert_allclose(ap(score[mask],dc[mask]),float(row["auprc"]),rtol=0,atol=1e-12,equal_nan=True)
            direct_image=np.array([auc(score[i,mask[i]],dc[i,mask[i]]) for i in range(n)])
            np.testing.assert_allclose(direct_image,st["image_auc"][:,j,k],rtol=0,atol=1e-12,equal_nan=True)
            verified_auc_images+=n
            mat=np.bincount(dc[mask].astype(int)*2+(score[mask]>0),minlength=4).reshape(2,2)
            np.testing.assert_array_equal(mat,st["sign_cm"][:,j,k].sum(0))
    probs={"shallow":data["ps"],"deep":data["pd"],"fixed_average":data["fixed_average"],"anchor_old":data["anchor"],
           "anchor_sym":r["anchor_sym"],"ctx_S":r["ctx_S"],"ctx_D":r["ctx_D"],"ctx_sym":r["ctx_sym"]}
    rows={(x["group"],x["estimator"]):x for x in read(root,"symmetric_anchor")}
    prediction_cache={est:prob.argmax(1) for est,prob in probs.items()}
    for j,(name,m) in enumerate(g.items()):
        for k,(est,prob) in enumerate(probs.items()):
            pred=prediction_cache[est]
            mat=np.bincount(4*y[m].astype(int)+pred[m],minlength=16).reshape(4,4)
            np.testing.assert_array_equal(mat,st["cm"][:,j,k].sum(0))
            met=seg(mat)
            for metric,val in met.items():np.testing.assert_allclose(val,float(rows[name,est][metric]),rtol=0,atol=1e-12,equal_nan=True)
            if name=="all":
                pgt=np.take_along_axis(prob,np.clip(y,0,3)[:,None],axis=1)[:,0]
                nll=-np.log(pgt[m].astype(float)+1e-8).mean()
                brier=sum((prob[:,c].astype(float)-(y==c))**2 for c in range(4))[m].mean()
                assert abs(nll-float(rows[name,est]["nll"]))<1e-12 and abs(brier-float(rows[name,est]["brier"]))<1e-12
    for row in read(root,"same_family_bias")+read(root,"support_matrix"):
        vals=r[row["field"]][g[row["group"]]].astype(float)
        if len(vals):
            np.testing.assert_allclose([vals.mean(),vals.std(),*np.quantile(vals,[.05,.25,.5,.75,.95])],
                [float(row[k]) for k in ("mean","std","p05","p25","p50","p75","p95")],rtol=0,atol=1e-12)
    for ctx in ("ctx_S","ctx_D","ctx_sym"):
        cp=r[ctx].argmax(1);diff=(cp!=s)&(cp!=d)
        assert ((cp==y)&g["Both_Wrong"]&~diff).sum()==0
        assert (diff&win&(cp==y)).sum()==0
    cp=r["ctx_sym"].argmax(1);diff=(cp!=s)&(cp!=d)
    third=np.array([np.stack([m.sum(1),(m&diff&(cp==y)).sum(1),(m&diff&(cp!=y)).sum(1)],1) for m in (g["Both_Wrong"],win)]).transpose(1,0,2)
    np.testing.assert_array_equal(third,st["third"])
    # Reconstruct 32 replicates from image indices, not the analyzer's matrix-multiply method.
    idx=np.random.default_rng(42).integers(0,n,(32,n),dtype=np.int32)
    check={}
    for j,group in enumerate(g):
        cnt=st["count"][idx,j].sum(1)
        for key in ("B_S","B_D","B_family"):
            v=(r[key].astype(float)*g[group]).sum(1)
            check[group+"__mean_"+key]=ratio(v[idx].sum(1),cnt)
    for key,k in (("old",0),("sym",2)):
        check[key+"_image_auroc"]=np.nanmean(st["image_auc"][idx,0,k],axis=1)
        mat=st["sign_cm"][idx,0,k].sum(1);rec=ratio(np.diagonal(mat,axis1=-2,axis2=-1),mat.sum(-1))
        check[key+"_balanced_accuracy"]=rec.mean(-1);check[key+"_deep_win_recall"]=rec[:,1];check[key+"_shallow_win_recall"]=rec[:,0]
    for key in ("image_auroc","balanced_accuracy","deep_win_recall","shallow_win_recall"):
        check["sym_minus_old_"+key]=check["sym_"+key]-check["old_"+key]
    for c in (2,3):check[f"class{c}_sym_image_auroc"]=np.nanmean(st["image_auc"][idx,list(g).index(f"class{c}"),2],axis=1)
    met=seg(st["cm"][idx,0].sum(1))
    for ref,k in (("fixed_average",2),("anchor_old",3)):
        for key in ("accuracy","miou"):check[f"anchor_sym_minus_{ref}_{key}"]=met[key][:,4]-met[key][:,k]
    tt=third[idx].sum(1)
    check["ctx_sym_Both_Wrong_accuracy"]=ratio(tt[:,0,1],tt[:,0,0])
    check["ctx_sym_ThirdClassRescueRate"]=check["ctx_sym_Both_Wrong_accuracy"].copy()
    check["ctx_sym_ThirdClassHarmRate"]=ratio(tt[:,1,2],tt[:,1,0])
    for name in ("old","sym"):
        sums=(r[name].astype(float)*fg).sum(1)
        check["mean_Delta_"+name]=ratio(sums[idx].sum(1),fg.sum(1)[idx].sum(1))
    check["BiasShrink"]=1-abs(check["mean_Delta_sym"])/(abs(check["mean_Delta_old"])+1e-8)
    reps=read(root,"bootstrap_replicates");assert len(reps)==10000
    errors={}
    for key,value in check.items():
        original=np.array([float(row[key]) for row in reps[:32]])
        np.testing.assert_allclose(value,original,atol=1e-12,rtol=0,equal_nan=True)
        errors[key]=float(np.nanmax(abs(value-original)))
    for row in read(root,"bootstrap"):
        values=np.array([float(x[row["metric"]]) for x in reps])
        f=values[np.isfinite(values)];ci=np.quantile(f,[.025,.975]) if len(f) else [np.nan,np.nan]
        np.testing.assert_allclose(ci,[float(row["ci95_low"]),float(row["ci95_high"])],rtol=0,atol=1e-12,equal_nan=True)
    cstates={}
    for c in (2,3):
        row=table[f"class{c}","sym"]
        cstates[f"class{c}"]="UNDERPOWERED" if min(int(row["positive"]),int(row["negative"]))<500 else "PASS" if float(row["image_auroc"])>=.45 else "FAIL"
    cg="FAIL" if "FAIL" in cstates.values() else "UNDERPOWERED" if "UNDERPOWERED" in cstates.values() else "PASS"
    ci=summary["ci"];sym=table["all","sym"]
    a=all(ci["all__mean_"+key]["observed"]>0 and ci["all__mean_"+key]["ci95_low"]>0 for key in ("B_S","B_D"))
    b=(float(sym["image_auroc"])>=.70 and float(sym["balanced_accuracy"])>=.62 and float(sym["deep_win_recall"])>=.55 and float(sym["shallow_win_recall"])>=.55
       and abs(r["sym"][fg].astype(float).mean())<.5*abs(r["old"][fg].astype(float).mean()))
    rescue=float(((cp==y)&g["Both_Wrong"]).sum()/g["Both_Wrong"].sum());harm=float((diff&win).sum()/win.sum())
    dg=rescue>=.25 and ci["ctx_sym_Both_Wrong_accuracy"]["ci95_low"]>.20 and rescue>=.15 and harm<=.10
    gates=dict(A=bool(a),B=bool(b),C=cg,D=bool(dg));assert gates==summary["gates"] and cstates==summary["class_status"]
    decision=("SAME_FAMILY_BIAS_HYPOTHESIS_NOT_SUPPORTED" if not a else
              ("THIRD_EVIDENCE_REQUIRED_FOR_NEXT_DESIGN" if dg else "ADJUDICATION_BIAS_UNRESOLVED") if not b else
              "ADJUDICATION_BIAS_UNRESOLVED" if cg=="FAIL" else
              "SYMMETRY_PROMISING_CLASS_EVIDENCE_UNDERPOWERED" if cg=="UNDERPOWERED" else "SYMMETRIC_ADJUDICATION_BIAS_RESOLVED")
    assert decision==summary["decision"]
    strong_sym=float(sym["image_auroc"])>=.75 and float(sym["balanced_accuracy"])>=.65 and min(float(sym["deep_win_recall"]),float(sym["shallow_win_recall"]))>=.60
    strong_third=rescue>=.30 and rescue>=.20 and harm<=.08
    assert strong_sym==summary["strong_symmetry_signal"] and strong_third==summary["strong_third_evidence_signal"]
    result=dict(status="PASS",images=n,required_artifacts_before_verification=len(required),all_45_group_counts_exact=True,
                all_groups_native_confusions_exact=True,all_support_distribution_statistics_verified=True,
                native_rankdata_image_computations=verified_auc_images,pooled_auc_and_ap_verified=True,
                independent_real_neighbor_positions=9,max_support_error=float(max(support_errors)),max_context_error=float(max(context_errors)),
                max_gt_composition_error=float(max(gt_errors)),old_score_parity_max=rt["parity_max_abs"],
                rescue_intrusion_identities_verified=True,independent_bootstrap_replicates=32,
                bootstrap_columns_verified=len(check),max_bootstrap_error=max(errors.values()),all_10000_ci_quantiles_exact=True,
                gates=gates,class_status=cstates,decision=decision,no_shared_helper_imports=True)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
