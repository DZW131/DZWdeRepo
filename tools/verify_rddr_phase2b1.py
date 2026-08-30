"""Independent CSV/native observation audit using NumPy/SciPy, no shared helpers."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata


def read_rows(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))


def exact_rank_auc(score,positive):
    positive=np.asarray(positive,bool)
    m,k=int(positive.sum()),int((~positive).sum())
    if m==0 or k==0: return np.nan
    ranks=rankdata(score,method="average")
    return (ranks[positive].sum()-m*(m+1)/2)/(m*k)


def metrics(cm):
    cm=np.asarray(cm,dtype=float)
    d=np.diagonal(cm,axis1=-2,axis2=-1)
    u=cm.sum(-1)+cm.sum(-2)-d
    iou=np.divide(d,u,out=np.full_like(d,np.nan),where=u>0)
    return d.sum(-1)/cm.sum(axis=(-2,-1)),np.nanmean(iou,axis=-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",required=True); p.add_argument("--native",required=True)
    p.add_argument("--output",help="Optional unique verification output path")
    args=p.parse_args(); root=Path(args.report)
    expected=("summary.json","per_image.csv","adjudication.csv","sign_decision.csv","anchor_metrics.csv",
              "conflict_strata.csv","deep_wrong_safety.csv","shallow_strata.csv","both_wrong.csv","hfrm_groups.csv",
              "top20_bottom80.csv","quintiles.csv","boundary_interior.csv","per_class.csv","calibration.csv",
              "echo.csv","bootstrap.csv","runtime.json","population_manifest.json")
    assert all((root/("rddr_phase2b1_"+x)).is_file() for x in expected)
    summary=json.loads((root/"rddr_phase2b1_summary.json").read_text(encoding="utf-8"))
    rows=read_rows(root/"rddr_phase2b1_per_image.csv")
    assert len(rows)==len({r["image_id"] for r in rows})==3418
    vec=lambda key:np.array([float(r[key]) for r in rows])
    fgcounts=vec("foreground_targets")
    for partition in (("Top20","Bottom80"),("Corrected_by_CH","Still_Wrong","Harmed_by_CH","Stable_Correct"),
                      ("Q1","Q2","Q3","Q4","Q5"),("boundary","interior"),("Deep_Correct","Deep_Wrong"),
                      ("Shallow_Correct","Shallow_Wrong"),("Strength1","Strength2","Strength3","Strength4","Strength5"),
                      ("class0","class1","class2","class3")):
        assert np.array_equal(sum(vec(g+"_targets") for g in partition),fgcounts),partition
    assert np.array_equal(vec("Deep_Win_targets")+vec("Shallow_Win_targets"),vec("adjudication_targets"))
    with np.load(args.native) as archive:
        data={k:archive[k] for k in archive.files}
    assert np.array_equal(data["names"],np.array([r["image_id"] for r in rows]))
    y=data["truth"].astype(int); fg=y<4
    s,d=data["ps"].argmax(1),data["pd"].argmax(1)
    sc,dc=s==y,d==y
    eligible=fg&(s!=d)&(sc!=dc)
    score=data["sd"]-data["ss"]
    assert np.array_equal(score,data["delta"])
    assert int(eligible.sum())==summary["adjudication_targets"]
    direct_cm=np.bincount((dc[eligible].astype(int)*2+(score[eligible]>0)),minlength=4).reshape(2,2)
    sign=np.array([json.loads(r["sign_confusion"]) for r in rows],dtype=np.int64)
    assert np.array_equal(direct_cm,sign.sum(0))
    auc=np.array([exact_rank_auc(score[i,eligible[i]],dc[i,eligible[i]]) for i in range(len(rows))])
    np.testing.assert_allclose(auc,vec("image_auroc"),rtol=0,atol=1e-12,equal_nan=True)
    pooled=exact_rank_auc(score[eligible],dc[eligible])
    pooled_row=next(r for r in read_rows(root/"rddr_phase2b1_adjudication.csv") if r["group"]=="all")
    assert abs(pooled-float(pooled_row["auroc"]))<1e-12
    # AP via descending thresholds/end-of-tie precision, unlike the analyzer's
    # ascending grouped-positive accumulation.
    ds=score[eligible]; dl=dc[eligible]
    order=np.argsort(-ds,kind="stable")
    ordered_score,ordered_label=ds[order],dl[order]
    ends=np.r_[np.flatnonzero(np.diff(ordered_score)),len(ordered_score)-1]
    tp=np.cumsum(ordered_label,dtype=np.int64)[ends]
    ap=float((np.diff(np.r_[0,tp])*(tp/(ends+1))).sum()/dl.sum())
    assert abs(ap-float(pooled_row["auprc"]))<1e-12
    refs={k:np.array([json.loads(r[k+"_confusion"]) for r in rows],dtype=np.int64)
          for k in ("shallow","deep","fixed_average","anchor","context_only")}
    all_metrics=read_rows(root/"rddr_phase2b1_anchor_metrics.csv")
    prob_keys=dict(shallow="ps",deep="pd",fixed_average="fixed_average",anchor="anchor",context_only="ctx")
    for name,cm in refs.items():
        assert np.array_equal(cm.sum((1,2)),fgcounts)
        probs=data[prob_keys[name]]
        pred=probs.argmax(1)
        native_cm=np.bincount(4*y[fg]+pred[fg],minlength=16).reshape(4,4)
        assert np.array_equal(native_cm,cm.sum(0))
        a,i=metrics(native_cm)
        r=next(r for r in all_metrics if r["group"]=="all" and r["estimator"]==name)
        assert abs(a-float(r["accuracy"]))<1e-12 and abs(i-float(r["miou"]))<1e-12
        gt_prob=np.take_along_axis(probs,np.clip(y,0,3)[:,None],axis=1)[:,0]
        ref_nll=-np.log(gt_prob[fg].astype(np.float64)+1e-8).mean()
        ref_brier=sum((probs[:,k].astype(np.float64)-(y==k))**2 for k in range(4))[fg].mean()
        assert abs(ref_nll-float(r["nll"]))<1e-6
        assert abs(ref_brier-float(r["brier"]))<1e-7
    # mIoU is nonlinear: positive deltas in disjoint strata need not imply a
    # positive delta after confusion matrices are merged. Verify the partitions.
    top=data["top20"].astype(bool)&fg
    qbin=np.searchsorted(summary["q_quintile_edges"],data["q_feature"],side="left")
    subgroup_rows={r["group"]:r for r in read_rows(root/"rddr_phase2b1_all_groups.csv")}
    for partition in (([("Top20",top),("Bottom80",fg&~top)]),[(f"Q{k+1}",fg&(qbin==k)) for k in range(5)]):
        for name in ("fixed_average","anchor"):
            reconstructed=np.zeros((4,4),np.int64)
            pred=data[name].argmax(1)
            for group,mask in partition:
                mat=np.bincount(4*y[mask]+pred[mask],minlength=16).reshape(4,4)
                reconstructed+=mat
                _,iou=metrics(mat)
                assert abs(iou-float(subgroup_rows[group][name+"_miou"]))<1e-12
            assert np.array_equal(reconstructed,refs[name].sum(0))
    # Independent nested-neighborhood check on fixed real image/position triples.
    support_errors=[]
    def js(p,q):
        m=.5*(p+q)
        return .5*((p*(np.log(p+1e-8)-np.log(m+1e-8))).sum(-1)+(q*(np.log(q+1e-8)-np.log(m+1e-8))).sum(-1))
    for ii in (0,1708,3417):
        for ty,tx in ((0,0),(14,14),(27,10)):
            evidence=[]
            for sy in range(max(0,ty-7),min(28,ty+8)):
                for sx in range(max(0,tx-7),min(28,tx+8)):
                    if (sy,sx)!=(ty,tx): evidence.append(data["ps"][ii,:,sy*28+sx])
            evidence=np.asarray(evidence)
            for key,prob in (("ss","ps"),("sd","pd")):
                val=np.clip(1-js(data[prob][ii,:,ty*28+tx],evidence)/np.log(2),0,1).mean()
                support_errors.append(abs(float(val)-float(data[key][ii,ty*28+tx])))
    assert max(support_errors)<5e-7
    reps=read_rows(root/"rddr_phase2b1_bootstrap_replicates.csv")
    assert len(reps)==10000
    idx=np.random.default_rng(42).integers(0,len(rows),(32,len(rows)),dtype=np.int32)
    sample_sign=sign[idx].sum(1)
    recalls=np.diagonal(sample_sign,axis1=-2,axis2=-1)/sample_sign.sum(-1)
    check=dict(adjudication_image_auroc=np.nanmean(auc[idx],axis=1),sign_balanced_accuracy=recalls.mean(-1))
    for group,prefix in (("all",""),("Top20","Top20_"),("Deep_Wrong","Deep_Wrong_"),("Top20_Deep_Wrong","Top20_Deep_Wrong_")):
        if group=="all": avg,anchor=refs["fixed_average"],refs["anchor"]
        else:
            avg=np.array([json.loads(r[prefix+"fixed_average_confusion"]) for r in rows])
            anchor=np.array([json.loads(r[prefix+"anchor_confusion"]) for r in rows])
        a0,i0=metrics(avg[idx].sum(1)); a1,i1=metrics(anchor[idx].sum(1))
        check[prefix+"anchor_fixed_accuracy_delta"]=a1-a0
        if group=="all": check["anchor_fixed_miou_delta"]=i1-i0
    errors={}
    for key,val in check.items():
        original=np.array([float(r[key]) for r in reps[:32]])
        errors[key]=float(np.max(np.abs(val-original)))
        assert errors[key]<1e-12,(key,errors[key])
    for r in read_rows(root/"rddr_phase2b1_bootstrap.csv"):
        sample=np.array([float(x[r["metric"]]) for x in reps])
        lo,hi=np.nanquantile(sample,[.025,.975])
        assert abs(lo-float(r["ci95_low"]))<1e-12 and abs(hi-float(r["ci95_high"]))<1e-12
    safety=read_rows(root/"rddr_phase2b1_deep_wrong_safety.csv")
    expected_bases=("all","Top20","Bottom80","hard_disagreement","Q1","Q2","Q3","Q4","Q5","boundary","interior","class0","class1","class2","class3")
    assert tuple(r["stratum"] for r in safety)==expected_bases
    hard=False
    for r in safety:
        base="DW__"+r["stratum"]
        count=vec(base+"_targets").sum()
        delta=(vec(base+"_anchor_correct").sum()-vec(base+"_fixed_correct").sum())/count if count else np.nan
        assert int(count)==int(r["targets"])
        assert (np.isnan(delta) and r["status"]=="UNDEFINED_EMPTY") or abs(delta-float(r["delta"]))<1e-12
        hard|=bool(delta<=-.1)
    ci=summary["ci"]
    a=ci["adjudication_image_auroc"]; ba=recall=direct_cm.diagonal()/direct_cm.sum(1)
    ca,mi=ci["anchor_fixed_accuracy_delta"],ci["anchor_fixed_miou_delta"]
    dw=float(next(r for r in safety if r["stratum"]=="all")["delta"])
    tdw=float(next(r for r in safety if r["stratum"]=="Top20")["delta"])
    gates=dict(A=a["observed"]>=.65 and a["ci95_low"]>.5,
               B=bool(ba.mean()>=.60 and (recall>=.55).all()),
               C=ca["observed"]>0 and mi["observed"]>0 and max(ca["ci95_low"],mi["ci95_low"])>0,
               D=dw>=-.02 and tdw>=-.03 and not hard)
    assert gates==summary["gates"]
    decision=("ADJUDICATION_DEEP_WRONG_UNSAFE" if not gates["D"] else "RDDR_PHASE2B1_NOGO" if not gates["A"] or not gates["B"] else
              "ADJUDICATION_EXISTS_FUSION_UTILITY_FAIL" if not gates["C"] else "RDDR_PHASE2B1_GO")
    assert decision==summary["decision"]
    strong=a["observed"]>=.70 and mi["observed"]>=.01 and dw>=0
    assert strong==summary["strong_signal"]
    result=dict(status="PASS",images=3418,required_files=len(expected),all_partitions_exact=True,
                native_all_image_rankdata_AUROC_exact=True,pooled_rankdata_AUROC=pooled,independent_pooled_AP=ap,
                independent_proper_scores_pass=True,
                independent_real_support_points=9,max_support_error=max(support_errors),
                independent_bootstrap_replicates=32,bootstrap_max_errors=errors,all_10000_CI_quantiles_exact=True,
                all_15_safety_strata_verified=True,nonlinear_partition_miou_verified=True,
                gates=gates,decision=decision,strong_signal=bool(strong))
    target=Path(args.output) if args.output else root/"rddr_phase2b1_independent_verification.json"
    if target.exists(): raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
