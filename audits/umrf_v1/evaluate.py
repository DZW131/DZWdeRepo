"""Open GT only after the UMRF evidence freeze and evaluate fixed rules."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from audits.umrf_v1.core import CHAINS, decision, metric_block, sha256
from audits.umrf_v1.freeze_evidence import MAP_KEYS
from network.cirv import extract_regions


def apply_component_rules(frame: pd.DataFrame, chain: str) -> None:
    h5=frame[f"{chain}5_pred"].to_numpy(); h4=frame[f"{chain}4_pred"].to_numpy(); h3=frame[f"{chain}3_pred"].to_numpy()
    frame[f"{chain}_r1_pred"] = np.where(h4==h3,h4,h5).astype(np.uint8)
    frame[f"{chain}_r2_pred"] = np.where(h5==h4,h5,np.where(h5==h3,h5,np.where(h4==h3,h4,h5))).astype(np.uint8)
    probs=np.stack([sum(frame[f"{chain}{s}_p{c}"].to_numpy() for s in ("5","4","3"))/3 for c in range(4)],1)
    frame[f"{chain}_r3_pred"] = probs.argmax(1).astype(np.uint8)


def block_from_frame(frame: pd.DataFrame, chain: str, weight: str | None = None) -> dict:
    return metric_block(*(frame[f"{chain}{s}_pred"].to_numpy() for s in ("5","4","3")),
                        frame.true_class.to_numpy(), None if weight is None else frame[weight].to_numpy())


def count_vector(h5, h4, h3, truth, rule_predictions: dict[str,np.ndarray]) -> dict[str,float]:
    trigger=(h4==h3)&(h4!=h5); wrong=h5!=truth; right=~wrong
    return {"total":float(len(truth)),"trigger":float(trigger.sum()),"correct_trigger":float((trigger&(h4==truth)).sum()),
            "h5_wrong":float(wrong.sum()),"corrected":float((trigger&wrong&(h4==truth)).sum()),
            "h5_right":float(right.sum()),"harmed":float((trigger&right&(h4!=truth)).sum()),
            **{f"gain_{k}":float(((v==truth).astype(np.int64)-(h5==truth).astype(np.int64)).sum()) for k,v in rule_predictions.items()}}


def metrics_from_counts(c: dict[str,float]) -> dict[str,float]:
    harmed=c["harmed"]
    return {"count_or_area":c["total"],"trigger":c["trigger"],"trigger_rate":c["trigger"]/max(c["total"],1),
            "precision":c["correct_trigger"]/max(c["trigger"],1),"coverage":c["corrected"]/max(c["h5_wrong"],1),
            "h5_correct_harm_rate":harmed/max(c["h5_right"],1),"corrected":c["corrected"],"harmed":harmed,
            "nce":c["corrected"]/harmed if harmed else (float("inf") if c["corrected"] else 0.0),
            **{k:c[k]/max(c["total"],1) for k in c if k.startswith("gain_")}}


def bootstrap_components(frame: pd.DataFrame, chain: str, resamples: int=2000) -> dict:
    rng=np.random.default_rng(42); n=len(frame); values=[]
    columns=[f"{chain}{s}_pred" for s in ("5","4","3")]
    h=frame[columns].to_numpy(np.int16); truth=frame.true_class.to_numpy(np.int16); area=frame.area.to_numpy(np.float64)
    for _ in range(resamples):
        ix=rng.integers(0,n,n); m=metric_block(h[ix,0],h[ix,1],h[ix,2],truth[ix],area[ix])
        values.append([m["precision"],m["coverage"],m["nce"],m["h5_correct_harm_rate"]])
    a=np.asarray(values); names=("precision","coverage","nce","h5_correct_harm_rate")
    return {name:{"median":float(np.nanmedian(a[:,i])),"ci95":[float(np.nanquantile(a[:,i],.025)),float(np.nanquantile(a[:,i],.975))]} for i,name in enumerate(names)} | {"resamples":resamples,"seed":42,"unit":"component"}


def bootstrap_images(rows: list[dict], chain: str, resamples: int=2000) -> dict:
    keys=("total","trigger","correct_trigger","h5_wrong","corrected","h5_right","harmed","gain_r1","gain_r2","gain_r3")
    a=np.asarray([[row[f"{chain}_{k}"] for k in keys] for row in rows],np.float64); rng=np.random.default_rng(42); samples=[]
    for _ in range(resamples):
        c=dict(zip(keys,a[rng.integers(0,len(a),len(a))].sum(0))); m=metrics_from_counts(c)
        samples.append([m["precision"],m["coverage"],m["nce"],m["h5_correct_harm_rate"],m["gain_r1"],m["gain_r2"],m["gain_r3"]])
    b=np.asarray(samples); names=("precision","coverage","nce","h5_correct_harm_rate","gain_r1","gain_r2","gain_r3")
    return {name:{"median":float(np.nanmedian(b[:,i])),"ci95":[float(np.nanquantile(b[:,i],.025)),float(np.nanquantile(b[:,i],.975))]} for i,name in enumerate(names)} | {"resamples":resamples,"seed":42,"unit":"image"}


def rescue_matrix(frame: pd.DataFrame, chain: str, weight: str) -> dict:
    t=frame.true_class.to_numpy(); h5=frame[f"{chain}5_pred"].to_numpy(); h4=frame[f"{chain}4_pred"].to_numpy(); h3=frame[f"{chain}3_pred"].to_numpy(); w=frame[weight].to_numpy()
    base=h5!=t
    masks={"h4_true_h3_true":(h4==t)&(h3==t),"h4_true_h3_false":(h4==t)&(h3!=t),
           "h4_false_h3_true":(h4!=t)&(h3==t),"both_false_same_rival":(h4!=t)&(h3!=t)&(h4==h3),
           "both_false_different":(h4!=t)&(h3!=t)&(h4!=h3)}
    denom=float(w[base].sum())
    return {k:{"mass":float(w[base&m].sum()),"fraction":float(w[base&m].sum()/denom) if denom else 0.} for k,m in masks.items()}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--val-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--historical-labels",type=Path); args=p.parse_args()
    out=args.output; manifest=json.loads((out/"01_evidence_freeze_manifest.json").read_text())
    for name, digest in manifest["sha256"].items():
        if sha256(out/name)!=digest: raise AssertionError(f"Frozen artifact changed: {name}")
    frame=pd.read_parquet(out/"component_evidence.parquet"); maps=np.load(out/"gt_free_prediction_maps.uint8.npy",mmap_mode="r")
    ids=np.load(out/"image_ids.npy",allow_pickle=False).astype(str); baseline_index=MAP_KEYS.index("baseline")
    labels=[]; image_rows=[]
    for image_index,image_id in enumerate(ids):
        truth=np.asarray(Image.open(args.val_root/"mask"/f"{image_id}.png")); prediction=maps[baseline_index,image_index]
        # extract_regions numbers connected components independently per class;
        # preserve its class-major emission order and use the class in the key.
        sub=frame[frame.image_index==image_index]
        regions=extract_regions(prediction)
        if len(sub)!=len(regions): raise AssertionError(f"Component replay mismatch: {image_id}")
        for (_,row),region in zip(sub.iterrows(),regions):
            valid=truth[region["mask"]]; valid=valid[valid<4]
            counts=np.bincount(valid.astype(np.int64),minlength=4) if len(valid) else np.zeros(4,np.int64)
            true_class=int(counts.argmax()); purity=float(counts.max()/max(counts.sum(),1))
            labels.append({"image_id":image_id,"baseline_class":int(region["class_id"]),
                           "component_id":int(region["component_id"]),"true_class":true_class,
                           "purity":purity,"m1":bool(not np.any(valid==int(region["class_id"]))),
                           "baseline_correct":bool(int(region["class_id"])==true_class)})
        valid=truth<4; row={"image_id":image_id}
        for chain in CHAINS:
            h5=maps[MAP_KEYS.index(f"{chain}5"),image_index][valid]; h4=maps[MAP_KEYS.index(f"{chain}4"),image_index][valid]; h3=maps[MAP_KEYS.index(f"{chain}3"),image_index][valid]; t=truth[valid]
            rules={r:maps[MAP_KEYS.index(f"{chain}_{r}"),image_index][valid] for r in ("r1","r2","r3")}
            counts=count_vector(h5,h4,h3,t,rules)
            row.update({f"{chain}_{k}":v for k,v in counts.items()})
        image_rows.append(row)
        if (image_index+1)%500==0: print(json.dumps({"event":"umrf_gt_progress","images":image_index+1}),flush=True)
    label_frame=pd.DataFrame(labels); frame=frame.merge(label_frame,on=["image_id","baseline_class","component_id"],validate="one_to_one")
    for chain in CHAINS: apply_component_rules(frame,chain)
    q4=float(frame.loc[frame.m1,"area"].quantile(.75)) if frame.m1.any() else float(frame.area.quantile(.75)); frame["m1_high_purity"]=frame.m1&(frame.purity>=.70); frame["m1_q4"]=frame.m1&(frame.area>=q4)
    historical={"expected_count":4440,"exact_replay_count":int(frame.m1.sum())}
    if args.historical_labels and args.historical_labels.exists():
        old=pd.read_parquet(args.historical_labels); keys=set(zip(old.loc[old.m1_historical_key,"image_id"].astype(str),old.loc[old.m1_historical_key,"predicted_class"].astype(int),old.loc[old.m1_historical_key,"component_id"].astype(int))) if "m1_historical_key" in old else set()
        now=set(zip(frame.loc[frame.m1,"image_id"].astype(str),frame.loc[frame.m1,"baseline_class"].astype(int),frame.loc[frame.m1,"component_id"].astype(int)))
        historical.update({"historical_key_count":len(keys),"key_overlap":len(keys&now),"key_union":len(keys|now)})
    summaries={}; pixel={}
    for chain in CHAINS:
        summaries[chain]={"component":block_from_frame(frame,chain),"area":block_from_frame(frame,chain,"area"),
                          "baseline_correct_area":block_from_frame(frame[frame.baseline_correct],chain,"area"),
                          "m1_area":block_from_frame(frame[frame.m1],chain,"area"),
                          "m1_high_purity_area":block_from_frame(frame[frame.m1_high_purity],chain,"area"),
                          "m1_q4_area":block_from_frame(frame[frame.m1_q4],chain,"area"),
                          "rescue_matrix_area":rescue_matrix(frame,chain,"area"),
                          "per_class_area":{str(c):block_from_frame(frame[frame.true_class==c],chain,"area") for c in range(4)}}
        for rule in ("r1","r2","r3"):
            gain=((frame[f"{chain}_{rule}_pred"]==frame.true_class).astype(int)-(frame[f"{chain}5_pred"]==frame.true_class).astype(int))
            summaries[chain][f"{rule}_component_net_gain"]=int(gain.sum()); summaries[chain][f"{rule}_area_net_gain"]=float((gain*frame.area).sum())
        total={k:sum(row[f"{chain}_{k}"] for row in image_rows) for k in ("total","trigger","correct_trigger","h5_wrong","corrected","h5_right","harmed","gain_r1","gain_r2","gain_r3")}
        pixel[chain]=metrics_from_counts(total); summaries[chain]["decision"]=decision(summaries[chain]["area"])
        summaries[chain]["bootstrap_component_area"]=bootstrap_components(frame,chain); summaries[chain]["bootstrap_pixel_image"]=bootstrap_images(image_rows,chain)
    common=summaries["common"]["decision"]; sequential=summaries["sequential"]["decision"]
    if common in ("GO","STRONG_GO") and sequential=="NOGO": diagnosis="DEEP_CONDITIONING_ERASES_LOCAL_EVIDENCE"
    elif common in ("GO","STRONG_GO") and sequential in ("GO","STRONG_GO"): diagnosis="LOCAL_EVIDENCE_UNDERUTILIZED"
    elif common=="NOGO" and sequential=="NOGO": diagnosis="INSUFFICIENT_MULTILEVEL_CORRECTIVE_EVIDENCE"
    else: diagnosis="MIXED_OR_WEAK_MULTILEVEL_EVIDENCE"
    labeled=out/"component_evidence_with_gt.parquet"; frame.to_parquet(labeled,index=False)
    pd.DataFrame(image_rows).to_parquet(out/"per_image_pixel_counts.parquet",index=False)
    result={"protocol":"UMRF-v1","feature_freeze_verified":True,"parameter_updates":0,"historical_m1":historical,"m1_q4_area_threshold":q4,
            "components":len(frame),"summaries":summaries,"pixel":pixel,"primary_chain":"common","primary_decision":common,
            "secondary_decision":sequential,"final_diagnosis":diagnosis,
            "downstream_counterfactual":{"status":"SKIP","reason":"Frozen HQMR exposes no legal pre-decoder class-responsibility injection interface; adding one would redefine the model."},
            "labeled_table_sha256":sha256(labeled)}
    (out/"02_evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=True),encoding="utf-8")
    print(json.dumps({"event":"UMRF_EVALUATED","primary_decision":common,"secondary_decision":sequential,"diagnosis":diagnosis}),flush=True)


if __name__=="__main__": main()
