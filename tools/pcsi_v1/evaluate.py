"""E5-only PCSI evaluation against the exact frozen UMRF P0 component bank."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from audits.umrf_v1.core import softmax_class
from network.cirv import extract_regions
from network.pcsi_net import PCSINet
from tools.eval_gcqm_full25_bcss_seed42 import TTA,foreground_confusion,normalize_cam,prediction_from_cam,presence,scores_from_confusion
from tools.pdsr_vpca_phase0.common import CommonEvalDataset,sha256,write_json
from tools.pdsr_vpca_phase0.evaluate import cohort_metrics,tp_safety
from tools.pcsi_v1.identity import collect


def resize_unflip(x,flip):
    x=F.interpolate(x.float(),size=(224,224),mode="bilinear",align_corners=False)
    return torch.flip(x,dims=tuple(d+1 for d in flip)) if flip else x


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.inference_mode()
def run_variant(model,loader,ids,truth_root,baseline_by_id,frame,variant,outdir):
    predictions=[]; hists=[]; responsibility=[]; drift={}; hashes=[]; image_count=0
    by_image={str(k):g for k,g in frame.groupby(frame.image_id.astype(str))}
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for names,raw in loader:
        raw=raw.cuda(non_blocking=True); dummy=torch.ones((raw.shape[0],4),device=raw.device)
        cams=[]; gates=[]; evidence=[]
        for view_index,(input_flip,output_flip) in enumerate(TTA):
            value=torch.flip(raw,dims=input_flip) if input_flip else raw
            with torch.autocast("cuda",dtype=torch.bfloat16): output=model(value,dummy)
            cams.append(resize_unflip(output["primary_output"],output_flip).cpu())
            gates.append(output["deep_gate"].float().cpu())
            item=output["stages"][2]["hqmr"]
            l5=torch.einsum("bqc,bqhw->bchw",item["weights"].float(),item["logits5"].float().sigmoid()).clamp(0,1)
            evidence.append(resize_unflip(l5,output_flip).cpu())
            if view_index==0 and image_count<32:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    reference=model.base((value-model.hqmr_mean.to(value))/model.hqmr_std.to(value),dummy,step=29275)
                base_tensors=collect(reference); new_tensors=collect(output)
                for key in base_tensors:
                    delta=(base_tensors[key].float()-new_tensors[key].float()).detach()
                    row=drift.setdefault(key,{"max_abs":0.,"sum_square":0.,"count":0})
                    row["max_abs"]=max(row["max_abs"],float(delta.abs().max()))
                    row["sum_square"]+=float(delta.square().sum()); row["count"]+=delta.numel()
                    hashes.append({"image_batch_first":str(names[0]),"key":key,
                                   "baseline_sha256":tensor_digest(base_tensors[key]),"new_sha256":tensor_digest(new_tensors[key])})
        cam=np.stack([x.numpy() for x in cams]).mean(0); gate=np.stack([x.numpy() for x in gates]).mean(0)
        l5_evidence=np.stack([x.numpy() for x in evidence]).mean(0)
        for i,name in enumerate(names):
            image_id=str(name); baseline=baseline_by_id[image_id]
            pred=prediction_from_cam(normalize_cam(cam[i]),presence(gate[i]),np.empty((224,224))).astype(np.uint8)
            truth=np.asarray(Image.open(truth_root/f"{image_id}.png"))
            predictions.append(pred); hists.append(foreground_confusion(truth,pred))
            probs=softmax_class(l5_evidence[i])
            regions={(int(r["class_id"]),int(r["component_id"])):r["mask"] for r in extract_regions(baseline)}
            rows=by_image.get(image_id)
            if rows is not None:
                for _,row in rows.iterrows():
                    key=(int(row.baseline_class),int(row.component_id)); mask=regions[key]
                    pooled=probs[:,mask].mean(1); new_class=int(pooled.argmax()); base_l5=int(row.sequential5_pred)
                    true=int(row.true_class) if bool(row.evaluable) else -1
                    responsibility.append({"image_id":image_id,"component_id":key[1],"baseline_class":key[0],
                                           "true_class":true,"evaluable":bool(row.evaluable),"hard_m1":bool(row.hard_m1),
                                           "m1":bool(row.m1),"area":int(row.area),"baseline_l5":base_l5,
                                           "new_l5":new_class,"new_margin_true_rival":float(pooled[true]-pooled[key[0]]) if true>=0 else np.nan,
                                           "new_true_probability":float(pooled[true]) if true>=0 else np.nan,
                                           "new_rival_probability":float(pooled[key[0]])})
        image_count+=len(names)
        if image_count%400<len(names): print(json.dumps({"event":"PCSI_EVAL_PROGRESS","variant":variant,"images":image_count}),flush=True)
    if image_count!=len(ids): raise AssertionError("Validation count changed")
    pred_array=np.stack(predictions); hist_array=np.stack(hists); table=pd.DataFrame(responsibility)
    outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(outdir/"e5_predictions.npz",image_ids=ids,predictions=pred_array,confusions=hist_array)
    table.to_parquet(outdir/"e5_responsibility.parquet",index=False)
    drift={key:{"max_abs":row["max_abs"],"rms":(row["sum_square"]/row["count"])**.5,
                "n_elements":row["count"]} for key,row in drift.items()}
    write_json(outdir/"posttrain_tensor_audit.json",{"tensor_drift":drift,"hashes":hashes,
               "query1_exact":drift["query1"]["max_abs"]==0,"context_pre_exact":drift["context_pre"]["max_abs"]==0,
               "ccra_responsibility_changed":drift["ccra2_responsibility"]["max_abs"]>0,
               "l5_changed":drift["hqmr5_logits"]["max_abs"]>0})
    runtime={"images":image_count,"seconds":time.perf_counter()-started,"tta_views":3,
             "peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}
    return pred_array,hist_array,table,runtime,drift


def responsibility_summary(table):
    evaluable=table[table.evaluable]
    def rate(rows,condition):
        if len(rows)==0:return {"components":0,"component_rate":None,"area_weighted_rate":None}
        x=condition(rows).astype(float)
        return {"components":len(rows),"component_rate":float(x.mean()),
                "area_weighted_rate":float(np.average(x,weights=rows.area))}
    wrong=evaluable[evaluable.baseline_l5!=evaluable.true_class]
    hard=evaluable[evaluable.hard_m1]
    exit_=hard[hard.new_l5!=hard.baseline_class]
    return {"RCR":rate(wrong,lambda x:x.new_l5==x.true_class),
            "HRCR":rate(hard,lambda x:x.new_l5==x.true_class),
            "RER":rate(hard,lambda x:x.new_l5!=x.baseline_class),
            "CorrectRER":rate(exit_,lambda x:x.new_l5==x.true_class),
            "true_rival_margin_hard_mean":float(hard.new_margin_true_rival.mean()) if len(hard) else None,
            "true_rival_margin_hard_median":float(hard.new_margin_true_rival.median()) if len(hard) else None,
            "per_class_HRCR":{str(c):rate(hard[hard.true_class==c],lambda x:x.new_l5==x.true_class) for c in range(4)}}


def bootstrap(hist,cohort,safety,responsibility,ids,n=2000):
    rng=np.random.default_rng(42); values={k:[] for k in ("C1-C0_mIoU_pp","C2-C1_mIoU_pp","C2-C0_mIoU_pp","C1_HardM1CR","C2_HardM1CR","C1_HRCR","C2_HRCR","C1_TPHarm","C2_TPHarm")}
    indexes={str(x):i for i,x in enumerate(ids)}
    resp={v:t.assign(image_index=t.image_id.astype(str).map(indexes)).query("hard_m1 and evaluable") for v,t in responsibility.items()}
    for _ in range(n):
        ix=rng.integers(0,len(ids),len(ids)); counts=np.bincount(ix,minlength=len(ids))
        score={v:scores_from_confusion(h[ix].sum(0))["mIoU"] for v,h in hist.items()}
        for a,b in (("C1","C0"),("C2","C1"),("C2","C0")):
            values[f"{a}-{b}_mIoU_pp"].append(100*(score[a]-score[b]))
        for v in ("C1","C2"):
            c=cohort[v]["hard_m1"]["per_image"].iloc[ix]
            values[f"{v}_HardM1CR"].append(float(c.corrected.sum()/max(c.denom.sum(),1)))
            t=safety[v]["per_image"].iloc[ix]
            values[f"{v}_TPHarm"].append(float(t.harmed.sum()/max(t.right.sum(),1)))
            r=resp[v]; weights=counts[r.image_index.to_numpy()]
            correct=(r.new_l5==r.true_class).to_numpy()
            values[f"{v}_HRCR"].append(float(np.dot(weights,correct)/max(weights.sum(),1)))
    return {k:{"mean":float(np.mean(x)),"ci95":[float(np.quantile(x,.025)),float(np.quantile(x,.975))]} for k,x in values.items()}|{"resamples":n,"seed":42,"unit":"paired image"}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--plip",type=Path,required=True); p.add_argument("--concept-cache",type=Path,required=True)
    p.add_argument("--val-root",type=Path,required=True); p.add_argument("--umrf",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--num-workers",type=int,default=4); a=p.parse_args()
    for v,d in (("C1","C1_VLM_LAST"),("C2","C2_STATIC_PDSR")):
        runtime=json.loads((a.output/d/"runtime.json").read_text())
        if runtime["status"]!="TRAINING_COMPLETE" or runtime["optimizer_steps"]!=5855:
            raise AssertionError(f"{v} E5 not complete")
    dataset=CommonEvalDataset(a.val_root/"img")
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True)
    ids=np.asarray([p.stem for p in dataset.files]); bank_ids=np.load(a.umrf/"image_ids.npy",allow_pickle=False).astype(str)
    bank=np.load(a.umrf/"gt_free_prediction_maps.uint8.npy",mmap_mode="r")
    baseline_by_id={x:bank[0,i] for i,x in enumerate(bank_ids)}
    if len(ids)!=3418 or set(ids)!=set(bank_ids): raise AssertionError("C0 frozen image universe mismatch")
    p0=np.stack([baseline_by_id[x] for x in ids]); hist0=np.stack([foreground_confusion(np.asarray(Image.open(a.val_root/"mask"/f"{x}.png")),p0[i]) for i,x in enumerate(ids)])
    frame=pd.read_parquet(a.umrf/"component_evidence_with_gt.parquet")
    frame["hard_m1"]=frame.evaluable&(frame.sequential5_pred!=frame.true_class)&(frame.sequential4_pred!=frame.true_class)&(frame.sequential3_pred!=frame.true_class)
    cache=torch.load(a.concept_cache,map_location="cpu",weights_only=False)
    hist={"C0":hist0}; cohort={}; safety={}; responsibility={}; runtime={}; drift={}; checkpoints={}
    for v,d in (("C1","C1_VLM_LAST"),("C2","C2_STATIC_PDSR")):
        model=PCSINet(a.checkpoint,a.plip,cache["embeddings"],v).cuda().eval()
        state_path=a.output/d/f"{v.lower()}_e5_adapter.pth"
        model.load_trainable_state_dict(torch.load(state_path,map_location="cpu",weights_only=False))
        checkpoints[v]={"path":str(state_path),"sha256":sha256(state_path),
                        "gamma_abs_mean":float(model.gamma.detach().abs().mean())}
        predictions,hist[v],responsibility[v],runtime[v],drift[v]=run_variant(model,loader,ids,a.val_root/"mask",baseline_by_id,frame,v,a.output/d)
        cohort[v]={"hard_m1":cohort_metrics(frame,predictions,ids,a.val_root/"mask",baseline_by_id,"hard_m1"),
                   "m1":cohort_metrics(frame,predictions,ids,a.val_root/"mask",baseline_by_id,"m1")}
        safety[v]=tp_safety(predictions,ids,a.val_root/"mask",baseline_by_id)
        del model; torch.cuda.empty_cache()
    scores={v:scores_from_confusion(x.sum(0)) for v,x in hist.items()}
    resp={v:responsibility_summary(t) for v,t in responsibility.items()}
    ci=bootstrap(hist,cohort,safety,responsibility,ids)
    c1_min=(resp["C1"]["HRCR"]["component_rate"] or 0)>=.05 and cohort["C1"]["hard_m1"]["pixel_area_weighted"]>=.01 and safety["C1"]["nce"]>1
    c2_advantage=((resp["C2"]["HRCR"]["component_rate"] or 0)-(resp["C1"]["HRCR"]["component_rate"] or 0)>=.05 or
                  cohort["C2"]["hard_m1"]["pixel_area_weighted"]-cohort["C1"]["hard_m1"]["pixel_area_weighted"]>=.03)
    c2_advantage=c2_advantage and scores["C2"]["mIoU"]>=scores["C1"]["mIoU"] and sum(scores["C2"]["class_iou"][str(c)]>=scores["C1"]["class_iou"][str(c)] for c in range(4))>=3
    strong=(100*(scores["C2"]["mIoU"]-scores["C0"]["mIoU"])>=.30 and
            (resp["C2"]["HRCR"]["component_rate"] or 0)>=.10 and safety["C2"]["nce"]>=1.5)
    decision="PRE_CCRA_PDSR_STRONG_GO" if strong else ("PRE_CCRA_PDSR_MECHANISM_GO" if c1_min and c2_advantage else "PRE_CCRA_VLM_SEMANTIC_NOGO")
    compact_cohort={v:{k:{x:y for x,y in block.items() if x not in ("per_image","components_table")} for k,block in cohorts.items()} for v,cohorts in cohort.items()}
    compact_safety={v:{k:x for k,x in s.items() if k!="per_image"} for v,s in safety.items()}
    result={"FINAL_DECISION":decision,"FULL25_GO":False,"C1_MINIMUM_GO":c1_min,"C2_PDSR_GO":c2_advantage,
            "C2_STRONG_GO":strong,"segmentation":scores,"responsibility":resp,"cohort":compact_cohort,
            "tp_safety":compact_safety,"bootstrap":ci,"runtime":runtime,"posttrain_tensor_drift":drift,
            "checkpoints":checkpoints,"exact_C0_bank":str(a.umrf/"gt_free_prediction_maps.uint8.npy"),
            "hard_m1_components":int(frame.hard_m1.sum()),"m1_components":int(frame.m1.sum()),
            "validation_only_after_e5":True,"vpca_v1_status":"CLOSED"}
    out=a.output/"metrics"; out.mkdir(parents=True,exist_ok=True)
    write_json(out/"final_result.json",result); write_json(out/"bootstrap_ci.json",ci)
    pd.DataFrame([{"variant":v,"mIoU":scores[v]["mIoU"],"mDice":scores[v]["mDice"],
                   **{f"IoU_C{c}":scores[v]["class_iou"][str(c)] for c in range(4)}} for v in ("C0","C1","C2")]).to_csv(out/"segmentation.csv",index=False)
    pd.DataFrame([{"variant":v,**{k:r["component_rate"] for k,r in resp[v].items() if isinstance(r,dict) and "component_rate" in r}} for v in ("C1","C2")]).to_csv(out/"responsibility.csv",index=False)
    print(json.dumps({"event":"PCSI_EVALUATION_COMPLETE","decision":decision,
                      "miou":{v:scores[v]["mIoU"] for v in scores},
                      "hrcr":{v:resp[v]["HRCR"]["component_rate"] for v in resp}}),flush=True)


if __name__=="__main__": main()
