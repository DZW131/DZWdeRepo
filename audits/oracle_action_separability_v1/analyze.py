"""Predeclared univariate and grouped Oracle-supervised information probes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

A_ABLATIONS={
    "A0": ["a_disagreement", "a_js", "a_cosine", "a_u5_mean", "a_d4_mean"],
    "A1": ["a_energy", "a_local_positive", "a_local_negative"],
    "A2": ["a_deep_entropy", "a_local_entropy", "a_entropy_delta", "a_deep_top_prob", "a_local_top_prob", "a_query_agree"],
    "A3": ["a_c5_", "a_c4_", "a_c3_", "a_three_same", "a_margin_gain_53", "a_cam_margin"],
    "A4": ["a_tta_", "a_gate_view_std"],
    "A5": ["a_log_area", "a_perimeter_area", "a_compactness", "a_bbox_aspect", "a_border_touch", "a_same_class_count"],
}
G_ABLATIONS={
    "G0": ["g_deep_score", "g_gate_threshold", "g_gate_margin", "g_abs_margin", "g_deep_rank"],
    "G1": ["g_tta_"],
    "G2": ["g_c4_mean", "g_c4_max", "g_c4_q", "g_c4_top0", "g_c3_mean", "g_c3_max", "g_c3_q", "g_c3_top0", "g_support_", "g_largest_support", "g_spatial_entropy"],
    "G3": ["g_c4_c3_", "g_c3_c4_", "g_c4_top_class_same", "g_c3_top_class_same", "g_contradiction", "g_local_deep_ratio"],
    "G4": ["g_query_"],
}


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y: np.ndarray, s: np.ndarray, weight=None) -> dict:
    y=np.asarray(y,dtype=int); s=np.asarray(s,dtype=float)
    weight=None if weight is None else np.asarray(weight,dtype=float)
    valid=np.isfinite(s)
    y=y[valid]; s=s[valid]
    if weight is not None: weight=weight[valid]
    if len(y)==0 or len(np.unique(y))<2:
        return {"n":len(y),"prevalence":float(np.mean(y)) if len(y) else None,
                "auroc":None,"auprc":None,"enrichment":None,
                **{f"recall_p{p}":0. for p in (80,90,95)},
                **{f"precision_top{p}":None for p in (1,5,10,20)}}
    prev=float(np.average(y,weights=weight))
    order=np.argsort(-s,kind="stable")
    yo=y[order]; wo=np.ones(len(y)) if weight is None else weight[order]
    tp=np.cumsum(yo*wo); total=np.cumsum(wo)
    precision=tp/total; recall=tp/max(float((y*(np.ones(len(y)) if weight is None else weight)).sum()),1e-8)
    ranked=s[order]
    boundaries=np.r_[ranked[1:]!=ranked[:-1],True]
    result={"n":len(y),"prevalence":prev,
            "auroc":float(roc_auc_score(y,s,sample_weight=weight)),
            "auprc":float(average_precision_score(y,s,sample_weight=weight))}
    result["enrichment"]=result["auprc"]/max(prev,1e-8)
    for p in (80,90,95):
        eligible=recall[boundaries & (precision>=p/100)]
        result[f"recall_p{p}"]=float(eligible.max()) if len(eligible) else 0.
    for p in (1,5,10,20):
        count=max(1,int(np.ceil(len(y)*p/100)))
        end=int(np.searchsorted(-ranked,-ranked[count-1],side="right"))-1
        result[f"precision_top{p}"]=float(tp[end]/total[end])
    return result


def groups_from_ids(ids: pd.Series) -> np.ndarray:
    def patient(s):
        parts=str(s).split("_")[0].split("-")
        return "-".join(parts[:3]) if len(parts)>=3 else str(s).split("_")[0]
    return ids.map(patient).to_numpy()


def probe(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kind: str) -> np.ndarray:
    folds=GroupKFold(n_splits=5)
    scores=np.full(len(y),np.nan)
    for train,test in folds.split(x,y,groups):
        if len(np.unique(y[train]))<2:
            raise AssertionError("Single-class grouped training fold")
        if kind=="linear":
            model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=False)),
                            ("scale",StandardScaler()),
                            ("model",LogisticRegression(C=1.,max_iter=2000))])
        else:
            model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=False)),
                            ("model",DecisionTreeClassifier(max_depth=3,
                                min_samples_leaf=max(50,int(np.ceil(.01*len(train)))),random_state=42))])
        model.fit(x.iloc[train],y[train])
        scores[test]=model.predict_proba(x.iloc[test])[:,1]
    if not np.isfinite(scores).all(): raise AssertionError("Incomplete OOF probe scores")
    return scores


def ablation_features(columns: list[str], ablations: dict) -> dict[str,list[str]]:
    cumulative=[]; sets={}
    for stage,prefixes in ablations.items():
        cumulative.extend(prefixes)
        selected=[name for name in columns if any(name.startswith(p) for p in cumulative)]
        if not selected: raise AssertionError(stage)
        sets[stage]=selected
    if set(sets[list(sets)[-1]])!=set(columns):
        raise AssertionError(f"Unassigned features: {set(columns)-set(sets[list(sets)[-1]])}")
    return sets


def univariate(frame: pd.DataFrame, features: list[str], y: np.ndarray, task: str) -> pd.DataFrame:
    rows=[]
    for name in features:
        values=frame[name].to_numpy(dtype=float)
        if np.isfinite(values).sum()==0 or np.nanstd(values)==0:
            continue
        raw=metrics(y,values); inverse=metrics(y,-values)
        chosen=raw if raw["auroc"]>=inverse["auroc"] else inverse
        rows.append({"task":task,"feature":name,"orientation":"+" if chosen is raw else "-",**chosen})
    return pd.DataFrame(rows).sort_values("auroc",ascending=False)


def gain_ranking(y: np.ndarray, score: np.ndarray, gain: np.ndarray, weights=None) -> dict:
    rho=spearmanr(score,gain).statistic
    select=np.argsort(-score)[:max(1,int(np.ceil(.1*len(score))))]
    weight=np.ones(len(score)) if weights is None else np.asarray(weights,dtype=float)
    overall=float(np.average(gain,weights=weight))
    top=float(np.average(gain[select],weights=weight[select]))
    return {"spearman":float(rho),"top10_mean_gain":top,"overall_mean_gain":overall,
            "top10_gain_enrichment":float(top/overall) if overall>0 else None,
            "top10_action_precision":float(np.average(y[select],weights=weight[select]))}


def subgroup(name: str, values: pd.Series, y: np.ndarray, scores: np.ndarray) -> list[dict]:
    rows=[]
    for value in values.dropna().unique():
        mask=(values==value).to_numpy()
        if mask.sum()<20: continue
        rows.append({"subgroup":name,"value":str(value),**metrics(y[mask],scores[mask])})
    return rows


def class_specific(frame: pd.DataFrame, field: str) -> tuple[bool,float]:
    subset=frame[(frame.subgroup==field)&frame.auroc.notna()]
    contribution=(subset.n.to_numpy(dtype=float)*np.maximum(subset.auroc.to_numpy(dtype=float)-0.5,0))
    share=float(contribution.max()/contribution.sum()) if contribution.sum()>0 else 0.
    return share>.70,share


def decision_a(ap1: dict, ap2: dict) -> str:
    if (ap1["auroc"] is None or ap2["auroc"] is None or ap1["auroc"]<.70 or ap2["auroc"]<.70
        or min(ap1["recall_p80"],ap2["recall_p80"])<.10):
        return "ARBITRATION_ACTION_NOT_SEPARABLE"
    if (min(ap1["auroc"],ap2["auroc"])>=.88 and
        min(ap1["recall_p80"],ap2["recall_p80"])>=.35 and
        min(ap1["enrichment"],ap2["enrichment"])>=2):
        return "STRONG_ARBITRATION_GO"
    if (min(ap1["auroc"],ap2["auroc"])>=.80 and
        min(ap1["recall_p80"],ap2["recall_p80"])>=.20 and
        min(ap1["enrichment"],ap2["enrichment"])>=2):
        return "ARBITRATION_SEPARABLE"
    return "ARBITRATION_WEAK"


def decision_g(m: dict) -> str:
    if m["auroc"] is None or m["auroc"]<.80 or m["recall_p80"]<.10:
        return "GATE_RESCUE_NOT_SEPARABLE"
    if m["auroc"]>=.92 and m["recall_p90"]>=.20 and m["enrichment"]>=3 and m["recall_p80"]>=.20:
        return "STRONG_GATE_GO"
    if m["auroc"]>=.88 and m["enrichment"]>=3 and m["recall_p80"]>=.20 and m["recall_p90"]>=.10:
        return "GATE_RESCUE_SEPARABLE"
    return "GATE_WEAK"


def permutation(x: pd.DataFrame,y: np.ndarray,groups: np.ndarray,seed=20260920) -> list[dict]:
    rng=np.random.default_rng(seed)
    rows=[]
    for repeat in range(10):
        shuffled=y.copy()
        for group in np.unique(groups):
            index=np.flatnonzero(groups==group)
            shuffled[index]=rng.permutation(shuffled[index])
        prediction=probe(x,shuffled,groups,"linear")
        rows.append({"permutation":repeat,"auroc":metrics(shuffled,prediction)["auroc"]})
    return rows


def analyze_track(features: pd.DataFrame, labels: pd.DataFrame, task: str,
                  names: list[str], ablations: dict, target: str, gain_col: str,
                  out: Path, permutation_on: bool=True) -> dict:
    if not np.array_equal(labels.feature_row.to_numpy(),np.arange(len(features))):
        raise AssertionError("Feature-label row identity mismatch")
    y=labels[target].to_numpy(dtype=int)
    groups=groups_from_ids(features.image_id)
    if len(np.unique(groups))<5: raise AssertionError("<5 patient groups")
    uni=univariate(features,names,y,task)
    suffix="_ap2" if task=="AP2" else ""
    uni.to_csv(out/f"univariate_metrics{suffix}.csv",index=False)
    sets=ablation_features(names,ablations)
    linear=[]; shallow=[]; score_cache={}
    for stage,selected in sets.items():
        for kind,collector in (("linear",linear),("shallow",shallow)):
            scores=probe(features[selected],y,groups,kind)
            score_cache[(stage,kind)]=scores
            collector.append({"task":task,"ablation":stage,"n_features":len(selected),
                              "features":";".join(selected),**metrics(y,scores)})
            print(json.dumps({"event":"probe","task":task,"stage":stage,"kind":kind,
                              "auroc":collector[-1]["auroc"]}),flush=True)
    pd.DataFrame(linear).to_csv(out/f"linear_probe_cv{suffix}.csv",index=False)
    pd.DataFrame(shallow).to_csv(out/f"shallow_probe_cv{suffix}.csv",index=False)
    final=score_cache[(list(sets)[-1],"linear")]
    pd.DataFrame({"feature_row":np.arange(len(final)),"oof_score":final}).to_parquet(out/f"{task}_oof_scores.parquet",index=False)
    gain=labels[gain_col].to_numpy(dtype=float)
    ranking=gain_ranking(y,final,gain)
    (out/f"gain_ranking{suffix}.json").write_text(json.dumps(ranking,indent=2),encoding="utf-8")
    perm=permutation(features[names],y,groups) if permutation_on else []
    return {"task":task,"univariate_best":uni.iloc[0].to_dict(),
            "linear_final":linear[-1],"shallow_final":shallow[-1],
            "gain_ranking":ranking,"permutations":perm,"oof_score":final,
            "features":names,"groups":len(np.unique(groups))}


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output
    manifest=json.loads((out/"feature_manifest_final.json").read_text())
    leakage=json.loads((out/"leakage_audit.json").read_text())
    if not leakage["pass"]: raise AssertionError("Oracle label integrity failed")
    for relative,expected in manifest["sha256"].items():
        if sha(out/relative)!=expected: raise AssertionError(f"Feature hash changed: {relative}")
    a=pd.read_parquet(out/"arbitration/observable_features.parquet")
    al=pd.read_parquet(out/"arbitration/oracle_action_labels.parquet")
    g=pd.read_parquet(out/"gate/gate_off_pairs.parquet")
    query=pd.read_parquet(out/"gate/gate_query_features.parquet")
    if not np.array_equal(g.image_id,query.image_id) or not np.array_equal(g.candidate_class,query.candidate_class):
        raise AssertionError("GT-free query feature row identity mismatch")
    g=pd.concat([g,query.drop(columns=["image_id","candidate_class"])],axis=1)
    gl=pd.read_parquet(out/"gate/rescue_oracle_labels.parquet")
    for frame,feature_names in ((a,manifest["features_A"]),(g,manifest["features_G"])):
        if any(name not in frame or not pd.api.types.is_numeric_dtype(frame[name]) for name in feature_names):
            raise AssertionError("Non-numeric or missing predictor")
        if any(bad in feature_names for bad in ("image_id","component_id","candidate_class","predicted_class","oracle_gain_pixels","true_class")):
            raise AssertionError("Forbidden predictor")
    a1=analyze_track(a,al,"AP1",manifest["features_A"],A_ABLATIONS,
                     "intervene","oracle_gain_pixels",out/"arbitration")
    actionable=al.action!="KEEP"
    a2=analyze_track(a.loc[actionable].reset_index(drop=True),
                     al.loc[actionable].reset_index(drop=True).assign(feature_row=np.arange(actionable.sum())),
                     "AP2",manifest["features_A"],A_ABLATIONS,
                     "direction_up","oracle_gain_pixels",out/"arbitration")
    gate=analyze_track(g,gl,"G",manifest["features_G"],G_ABLATIONS,
                       "beneficial_rescue","gate_gain_pixels",out/"gate")
    pd.DataFrame([{"probe":"linear",**a2["linear_final"]},
                  {"probe":"shallow_tree",**a2["shallow_final"]}]).to_csv(
                      out/"arbitration/action_direction.csv",index=False)
    pd.DataFrame([{"probe":"linear",**gate["linear_final"]},
                  {"probe":"shallow_tree",**gate["shallow_final"]},
                  {"probe":"best_univariate",**gate["univariate_best"]}]).to_csv(
                      out/"gate/operating_points.csv",index=False)
    pd.DataFrame([{"task":"AP1",**a1["gain_ranking"]},
                  {"task":"AP2",**a2["gain_ranking"]}]).to_csv(
                      out/"arbitration/gain_ranking.csv",index=False)
    pd.DataFrame([{"task":"G",**gate["gain_ranking"]}]).to_csv(
                      out/"gate/gain_ranking.csv",index=False)
    a_score=a1["oof_score"];g_score=gate["oof_score"]
    pd.DataFrame([{"weighting":"component",**metrics(al.intervene,a_score)},
                  {"weighting":"area",**metrics(al.intervene,a_score,al.area)}]).to_csv(
                      out/"arbitration/weighted_metrics.csv",index=False)
    benefit_weight=np.where(gl.gate_gain_pixels>0,gl.gate_gain_pixels.clip(lower=1),
                            (-gl.gate_gain_pixels).clip(lower=1))
    pd.DataFrame([{"weighting":"pair",**metrics(gl.beneficial_rescue,g_score)},
                  {"weighting":"benefit_or_harm_pixels",**metrics(gl.beneficial_rescue,g_score,benefit_weight)}]).to_csv(
                      out/"gate/weighted_metrics.csv",index=False)
    aq=pd.qcut(a.area,4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
    a_sub=[]
    for name,values in (("area_quartile",aq),("predicted_class",a.predicted_class),
                        ("gate_preserved",a.gate_preserved),("m1_subset",al.m1_subset)):
        a_sub.extend(subgroup(name,values,al.intervene.to_numpy(),a_score))
    pd.DataFrame(a_sub).to_csv(out/"arbitration/subgroup_metrics.csv",index=False)
    gq=pd.qcut(g.g_gate_score,4,labels=["Q1","Q2","Q3","Q4"],duplicates="drop")
    g_sub=[]
    for name,values in (("candidate_class",g.candidate_class),("deep_score_quartile",gq),
                        ("tta_any_above",g.g_tta_above>0),("m1_related",gl.m1_related)):
        g_sub.extend(subgroup(name,values,gl.beneficial_rescue.to_numpy(),g_score))
    pd.DataFrame(g_sub).to_csv(out/"gate/subgroup_metrics.csv",index=False)
    ad=decision_a(a1["linear_final"],a2["linear_final"])
    gd=decision_g(gate["linear_final"])
    a_specific,a_share=class_specific(pd.DataFrame(a_sub),"predicted_class")
    g_specific,g_share=class_specific(pd.DataFrame(g_sub),"candidate_class")
    for track,decision,primary,specific,share in (("arbitration",ad,a1,a_specific,a_share),
                                                  ("gate",gd,gate,g_specific,g_share)):
        perm=primary["permutations"]
        perm_max=max(row["auroc"] for row in perm)
        confidence="LOW_LEAKAGE_REVIEW" if perm_max>.55 else "LOW_CLASS_SPECIFIC" if specific else "MODERATE_SINGLE_SEED"
        result={"decision":decision,"confidence":confidence,"permutation_max_auroc":perm_max,
                "class_specific":specific,"largest_class_discrimination_share":share,
                "primary_linear":primary["linear_final"],"primary_univariate":primary["univariate_best"]}
        (out/track/"decision.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    audit={"pass":True,"feature_hashes_still_match":True,
           "permutation_AP1":a1["permutations"],"permutation_AP2":a2["permutations"],
           "permutation_G":gate["permutations"],
           "permutation_any_above_055":any(r["auroc"]>.55 for run in (a1,a2,gate) for r in run["permutations"]),
           "grouping":"TCGA case/patient identifier used only for GroupKFold, not predictor",
           "fold_preprocessing":"train-only median imputation and standardization",
           "parameter_updates":0}
    (out/"leakage_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps({"event":"ANALYSIS_COMPLETE","A":ad,"G":gd,
                      "AP1_AUROC":a1["linear_final"]["auroc"],
                      "AP2_AUROC":a2["linear_final"]["auroc"],
                      "G_AUROC":gate["linear_final"]["auroc"]}),flush=True)


if __name__=="__main__":main()
