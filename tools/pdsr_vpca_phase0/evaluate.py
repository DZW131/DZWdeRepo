"""Unified E5-only evaluation for P0/P1/P2/P3 and preregistered mechanism gates."""
from __future__ import annotations
import argparse,json,math,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from audits.ucrf_v1.gate import load_model
from network.cirv import extract_regions
from network.pdsr_vpca_hqmr import PDSRVPCAHQMR
from tools.eval_gcqm_full25_bcss_seed42 import TTA,foreground_confusion,normalize_cam,prediction_from_cam,presence,scores_from_confusion
from tools.pdsr_vpca_phase0.common import CommonEvalDataset,sha256,write_json

VARIANT_DIR={"P1":"P1_VLM_LAST","P2":"P2_STATIC_PDSR","P3":"P3_VPCA_PDSR"}

def counted_flops(callable_forward):
    """Return profiler-counted FLOPs for one image/view (unsupported ops remain uncounted)."""
    torch.cuda.synchronize()
    with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],with_flops=True) as prof:
        callable_forward()
    torch.cuda.synchronize()
    return float(sum((event.flops or 0) for event in prof.key_averages()))

@torch.inference_mode()
def profile_p0(checkpoint,raw):
    model=load_model(checkpoint).eval(); labels=torch.ones((1,4),device=raw.device); mean=raw.new_tensor([.485,.456,.406])[None,:,None,None]; std=raw.new_tensor([.229,.224,.225])[None,:,None,None]
    def forward():
        with torch.autocast("cuda",dtype=torch.bfloat16): model((raw-mean)/std,labels,step=29275)
    flops=counted_flops(forward); torch.cuda.reset_peak_memory_stats(); forward(); torch.cuda.synchronize(); started=time.perf_counter()
    for _ in range(9): forward()
    torch.cuda.synchronize(); seconds=time.perf_counter()-started; peak=torch.cuda.max_memory_allocated()/1024**3
    del model; torch.cuda.empty_cache()
    return {"flops_per_view":flops,"gflops_per_view":flops/1e9,"flops_note":"torch.profiler counted FLOPs; unsupported operators are not imputed","estimated_3view_fps":9/(seconds*3),"peak_vram_gib":peak}

def resize_unflip_batch(value,size,dims):
    x=F.interpolate(value.float(),size=size,mode="bilinear",align_corners=False)
    return torch.flip(x,dims=tuple(d+1 for d in dims)) if dims else x

@torch.inference_mode()
def infer_variant(model,loader,val_root:Path,variant:str):
    predictions=[]; ids=[]; hist=[]; betas=[]; evidences=[]; layer_evidences=[]; qs=[]; rhos=[]; omegas=[]; energies=[]; rnorm=[]; unorm=[]
    started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    for batch_index,(names,raw) in enumerate(loader):
        raw=raw.cuda(non_blocking=True); dummy=torch.ones((raw.shape[0],4),device=raw.device)
        view_cam=[]; view_gate=[]; view_beta=[]; view_evidence=[]; view_layer_evidence=[]; view_q=[]; view_rho=[]; view_omega=[]; view_energy=[]; view_r=[]; view_u=[]
        for input_flip,output_flip in TTA:
            value=torch.flip(raw,dims=input_flip) if input_flip else raw
            with torch.autocast("cuda",dtype=torch.bfloat16): out=model(value,dummy)
            view_cam.append(resize_unflip_batch(out["primary_output"],(224,224),output_flip).cpu())
            view_gate.append(out["deep_gate"].float().cpu())
            phase=out["phase0"]; energy=phase["semantic"].float().norm(dim=1,keepdim=True)
            view_energy.append(resize_unflip_batch(energy,energy.shape[-2:],output_flip).cpu())
            aux=phase["pdsr"]; provider=phase["provider"]
            if aux:
                view_beta.append(resize_unflip_batch(aux["layer_weights"],aux["layer_weights"].shape[-2:],output_flip).cpu())
                view_evidence.append(resize_unflip_batch(aux["class_semantic_evidence"],aux["class_semantic_evidence"].shape[-2:],output_flip).cpu())
                layer=aux["layer_class_evidence"].flatten(0,1)
                layer=resize_unflip_batch(layer,layer.shape[-2:],output_flip).reshape(raw.shape[0],3,4,*layer.shape[-2:])
                view_layer_evidence.append(layer.cpu())
                view_r.append(aux["reconstruction_norm"].float().cpu()); view_u.append(aux["visual_norm"].float().cpu())
            if "q" in provider:
                view_q.append(provider["q"].float().cpu()); view_rho.append(provider["rho"].float().cpu()); view_omega.append(provider["omega"].float().cpu())
        cams=torch.stack(view_cam).mean(0).numpy(); gates=torch.stack(view_gate).mean(0).numpy()
        if view_beta:
            betas.extend(torch.stack(view_beta).mean(0).numpy().astype(np.float16)); evidences.extend(torch.stack(view_evidence).mean(0).numpy().astype(np.float16)); layer_evidences.extend(torch.stack(view_layer_evidence).mean(0).numpy().astype(np.float16))
            rnorm.extend(torch.stack(view_r).mean(0).numpy()); unorm.extend(torch.stack(view_u).mean(0).numpy())
        if view_q:
            qs.extend(torch.stack(view_q).mean(0).numpy().astype(np.float16)); rhos.extend(torch.stack(view_rho).mean(0).numpy().astype(np.float16)); omegas.extend(torch.stack(view_omega).mean(0).numpy().astype(np.float16))
        energies.extend(torch.stack(view_energy).mean(0)[:,0].numpy().astype(np.float16))
        for i,name in enumerate(names):
            cam=normalize_cam(cams[i]); pred=prediction_from_cam(cam,presence(gates[i]),np.empty((224,224))).astype(np.uint8)
            truth=np.asarray(Image.open(val_root/"mask"/f"{name}.png")); predictions.append(pred); hist.append(foreground_confusion(truth,pred)); ids.append(str(name))
        if (batch_index+1)%100==0: print(json.dumps({"event":"pdsr_eval_progress","variant":variant,"images":len(ids)}),flush=True)
    runtime={"seconds":time.perf_counter()-started,"images":len(ids),"tta_views":3,"fps":len(ids)/(time.perf_counter()-started),"peak_vram_gib":torch.cuda.max_memory_allocated()/1024**3}
    mechanism={"beta":np.asarray(betas),"class_evidence":np.asarray(evidences),"layer_class_evidence":np.asarray(layer_evidences),"q":np.asarray(qs),"rho":np.asarray(rhos),"omega":np.asarray(omegas),"energy":np.asarray(energies),"reconstruction_norm":np.asarray(rnorm),"visual_norm":np.asarray(unorm)}
    return np.asarray(ids),np.stack(predictions),np.stack(hist),mechanism,runtime

def cohort_for_image(frame:pd.DataFrame,image_id:str,baseline:np.ndarray):
    regions={(int(x["class_id"]),int(x["component_id"])):x["mask"] for x in extract_regions(baseline)}
    rows=[]
    for _,r in frame[frame.image_id.astype(str)==str(image_id)].iterrows():
        key=(int(r.baseline_class),int(r.component_id));
        if key not in regions: raise AssertionError(f"Frozen component key missing: {image_id} {key}")
        rows.append((r,regions[key]))
    return rows

def cohort_metrics(frame,predictions,ids,truth_root,baseline_by_id,cohort_column,mechanism=None):
    component=[]; image_rows=[]; margins=[]
    index={str(x):i for i,x in enumerate(ids)}
    for image_id in ids:
        i=index[str(image_id)]; base=baseline_by_id[str(image_id)]; new=predictions[i]; truth=np.asarray(Image.open(truth_root/f"{image_id}.png")); valid=truth<4
        corrected=denom=wrong_rival=0
        for r,mask in cohort_for_image(frame[frame[cohort_column]],str(image_id),base):
            wrong=mask&valid&(base!=truth); c=int((wrong&(new==truth)).sum()); d=int(wrong.sum()); corrected+=c; denom+=d; wrong_rival+=int((wrong&(new==int(r.baseline_class))).sum())
            component.append({"image_id":str(image_id),"class":int(r.true_class),"area":int(r.area),"corrected":c,"denom":d,"rate":c/max(d,1),"wrong_rival":int((wrong&(new==int(r.baseline_class))).sum())})
            if mechanism is not None and len(mechanism.get("class_evidence",[])):
                ev=torch.from_numpy(mechanism["class_evidence"][i].astype(np.float32))[None]
                ev=F.interpolate(ev,size=(224,224),mode="bilinear",align_corners=False)[0].numpy()
                margins.append(float((ev[int(r.true_class)]-ev[int(r.baseline_class)])[mask].mean()))
        image_rows.append({"image_id":str(image_id),"corrected":corrected,"denom":denom,"wrong_rival":wrong_rival})
    c=pd.DataFrame(component); im=pd.DataFrame(image_rows)
    return {"pixel_area_weighted":float(c.corrected.sum()/max(c.denom.sum(),1)),"component_weighted":float(c.rate.mean()) if len(c) else 0.,
            "components":len(c),"area":int(c.area.sum()) if len(c) else 0,"wrong_rival_persistence":float(c.wrong_rival.sum()/max(c.denom.sum(),1)),
            "per_class":{str(k):float(g.corrected.sum()/max(g.denom.sum(),1)) for k,g in c.groupby("class")},
            "semantic_margin":{"mean":float(np.mean(margins)) if margins else None,"median":float(np.median(margins)) if margins else None,"positive_fraction":float(np.mean(np.asarray(margins)>0)) if margins else None},
            "per_image":im,"components_table":c}

def tp_safety(predictions,ids,truth_root,baseline_by_id):
    rows=[]
    for i,image_id in enumerate(ids):
        truth=np.asarray(Image.open(truth_root/f"{image_id}.png")); valid=truth<4; base=baseline_by_id[str(image_id)]; new=predictions[i]
        wrong=valid&(base!=truth); right=valid&(base==truth); corrected=int((wrong&(new==truth)).sum()); harmed=int((right&(new!=truth)).sum())
        rows.append({"image_id":str(image_id),"corrected":corrected,"harmed":harmed,"wrong":int(wrong.sum()),"right":int(right.sum())})
    x=pd.DataFrame(rows); return {"corrected":int(x.corrected.sum()),"harmed":int(x.harmed.sum()),"tp_harm":float(x.harmed.sum()/x.right.sum()),
        "nce":float(x.corrected.sum()/max(x.harmed.sum(),1)),"per_image":x}

def paired_bootstrap(seg_hist,cohort,tp,resamples=2000):
    rng=np.random.default_rng(42); names=list(seg_hist); n=len(next(iter(seg_hist.values()))); values={}
    pairs=(("P1","P0"),("P2","P1"),("P3","P2"),("P3","P0"))
    for a,b in pairs: values[f"miou_{a}_{b}"]=[]
    for v in ("P1","P2","P3"): values[f"hmcr_{v}"]=[]; values[f"m1cr_{v}"]=[]; values[f"tpharm_{v}"]=[]
    for _ in range(resamples):
        ix=rng.integers(0,n,n)
        scores={v:scores_from_confusion(seg_hist[v][ix].sum(0))["mIoU"] for v in names}
        for a,b in pairs: values[f"miou_{a}_{b}"].append(100*(scores[a]-scores[b]))
        for v in ("P1","P2","P3"):
            for label,key in (("hmcr","hard_m1"),("m1cr","m1")):
                x=cohort[v][key]["per_image"].iloc[ix]; values[f"{label}_{v}"].append(float(x.corrected.sum()/max(x.denom.sum(),1)))
            x=tp[v]["per_image"].iloc[ix]; values[f"tpharm_{v}"].append(float(x.harmed.sum()/max(x.right.sum(),1)))
    return {k:{"mean":float(np.mean(v)),"ci95":[float(np.quantile(v,.025)),float(np.quantile(v,.975))]} for k,v in values.items()}|{"resamples":resamples,"seed":42,"unit":"paired image"}

def mechanism_summary(mechanism,labels_by_id,ids):
    result={}
    if len(mechanism.get("beta",[])):
        beta=mechanism["beta"].astype(np.float32); dominant=(beta>.9).mean((0,2,3)); result["beta_mean"]=beta.mean((0,2,3)).tolist(); result["beta_std"]=beta.std((0,2,3)).tolist(); result["beta_gt_0_9_fraction"]=dominant.tolist(); result["layer_collapse"]=bool(dominant.max()>.9)
        r=mechanism["reconstruction_norm"]; u=mechanism["visual_norm"]; result["reconstruction_norm_mean"]=r.mean(0).tolist(); result["visual_norm_mean"]=u.mean(0).tolist(); result["reconstruction_visual_ratio_mean"]=(r/np.maximum(u,1e-8)).mean(0).tolist()
    if len(mechanism.get("q",[])):
        q=mechanism["q"].astype(np.float32); per={}; collapse=False
        for c in range(4):
            selected=np.asarray([labels_by_id[str(x)][c] for x in ids],bool); qc=q[selected,c]
            top=np.bincount(qc.argmax(1),minlength=8)/max(len(qc),1); entropy=-(qc*np.log(np.maximum(qc,1e-8))).sum(1)
            order=np.sort(qc,axis=1); js=[]
            for x,y in zip(qc[::2],qc[1::2]):
                m=.5*(x+y); js.append(.5*((x*np.log(np.maximum(x,1e-8)/np.maximum(m,1e-8))).sum()+(y*np.log(np.maximum(y,1e-8)/np.maximum(m,1e-8))).sum()))
            flag=bool(top.max()>.8 and entropy.mean()<.5*math.log(8)); collapse|=flag
            per[str(c)]={"images":int(selected.sum()),"entropy_mean":float(entropy.mean()),"top1_frequency_max":float(top.max()),"top2_mass_mean":float(order[:,-2:].sum(1).mean()),"inter_image_js_mean":float(np.mean(js)) if js else 0.,"collapse":flag}
        result["concept_per_class"]=per; result["concept_collapse"]=collapse
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--plip",type=Path,required=True); p.add_argument("--val-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--umrf",type=Path,required=True); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--num-workers",type=int,default=4); a=p.parse_args()
    out=a.output; prep=json.loads((out/"manifests/preparation_complete.json").read_text());
    for v,d in VARIANT_DIR.items():
        runtime=json.loads((out/d/"runtime.json").read_text());
        if runtime["status"]!="TRAINING_COMPLETE" or runtime["optimizer_steps"]!=5855: raise AssertionError(f"{v} E5 missing")
    cache=torch.load(out/"concept_bank/concept_embeddings.pt",map_location="cpu",weights_only=False)
    dataset=CommonEvalDataset(a.val_root/"img"); loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers,pin_memory=True)
    ids=np.asarray([p.stem for p in dataset.files]); labels_by_id={x:(np.bincount(np.asarray(Image.open(a.val_root/"mask"/f"{x}.png")).ravel(),minlength=5)[:4]>0).astype(int) for x in ids}
    profile_raw=dataset[0][1][None].cuda(); compute={"P0":profile_p0(a.checkpoint,profile_raw)}
    bank=np.load(a.umrf/"gt_free_prediction_maps.uint8.npy",mmap_mode="r"); bank_ids=np.load(a.umrf/"image_ids.npy",allow_pickle=False).astype(str); baseline_by_id={x:bank[0,i] for i,x in enumerate(bank_ids)}
    p0=np.stack([baseline_by_id[x] for x in ids]); hist0=np.stack([foreground_confusion(np.asarray(Image.open(a.val_root/"mask"/f"{x}.png")),p0[i]) for i,x in enumerate(ids)])
    predictions={"P0":p0}; hist={"P0":hist0}; mechanisms={}; runtimes={"P0":{"source":"exact UMRF replay"}}; ccra_identity=None
    for variant,directory in VARIANT_DIR.items():
        model=PDSRVPCAHQMR(a.checkpoint,a.plip,cache["embeddings"],variant).cuda().eval(); state=torch.load(out/directory/f"{variant.lower()}_e5_adapter.pth",map_location="cpu",weights_only=False); model.load_trainable_state_dict(state)
        labels=torch.ones((1,4),device=profile_raw.device)
        compute[variant]={"flops_per_view":counted_flops(lambda: model(profile_raw,labels)),"flops_note":"torch.profiler counted FLOPs; unsupported operators are not imputed"}; compute[variant]["gflops_per_view"]=compute[variant]["flops_per_view"]/1e9
        if variant=="P3":
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
                fused=model(profile_raw,labels); reference=model.base((profile_raw-model.hqmr_mean.to(profile_raw))/model.hqmr_std.to(profile_raw),labels,step=29275)
            drifts=[]
            for fs,rs in zip(fused["stages"],reference["stages"]):
                if "detail" in fs and "responsibility_class" in fs["detail"]:
                    drifts.append(float((fs["detail"]["responsibility_class"]-rs["detail"]["responsibility_class"]).abs().max()))
            ccra_identity={"sample":str(ids[0]),"stage_max_abs_drift":drifts,"max_abs_drift":max(drifts,default=float("inf")),"exact_upstream_identity":bool(drifts) and all(x==0 for x in drifts)}
            if not ccra_identity["exact_upstream_identity"]: raise AssertionError(f"CCRA upstream identity failed: {ccra_identity}")
        vid,vpred,vhist,mech,runtime=infer_variant(model,loader,a.val_root,variant)
        if not np.array_equal(vid,ids): raise AssertionError("Validation ordering changed")
        predictions[variant]=vpred; hist[variant]=vhist; mechanisms[variant]=mech; runtimes[variant]=runtime
        np.savez_compressed(out/directory/"e5_predictions.npz",image_ids=vid,predictions=vpred,confusions=vhist)
        np.savez_compressed(out/directory/"e5_mechanism.npz",**mech)
        del model; torch.cuda.empty_cache()
    scores={v:scores_from_confusion(h.sum(0)) for v,h in hist.items()}
    frame=pd.read_parquet(a.umrf/"component_evidence_with_gt.parquet"); frame["hard_m1"]=frame.evaluable&(frame.sequential5_pred!=frame.true_class)&(frame.sequential4_pred!=frame.true_class)&(frame.sequential3_pred!=frame.true_class)
    cohort={}; safety={}
    for v in ("P1","P2","P3"):
        cohort[v]={"hard_m1":cohort_metrics(frame,predictions[v],ids,a.val_root/"mask",baseline_by_id,"hard_m1",mechanisms.get(v)),
                   "m1":cohort_metrics(frame,predictions[v],ids,a.val_root/"mask",baseline_by_id,"m1",mechanisms.get(v))}
        safety[v]=tp_safety(predictions[v],ids,a.val_root/"mask",baseline_by_id)
    bootstrap=paired_bootstrap(hist,cohort,safety)
    mechanism={v:mechanism_summary(mechanisms[v],labels_by_id,ids) for v in ("P1","P2","P3")}
    gamma={}
    for v,d in VARIANT_DIR.items():
        state=torch.load(out/d/f"{v.lower()}_e5_adapter.pth",map_location="cpu",weights_only=False); g=state["gamma"].float(); gamma[v]={"mean":float(g.mean()),"std":float(g.std()),"abs_mean":float(g.abs().mean()),"min":float(g.min()),"max":float(g.max()),"vlm_ignored":bool(g.abs().mean()<1e-3)}
    deltas={v:{str(c):100*(scores[v]["class_iou"][str(c)]-scores["P0"]["class_iou"][str(c)]) for c in range(4)} for v in ("P1","P2","P3")}
    pdsr_gain=cohort["P2"]["hard_m1"]["pixel_area_weighted"]-cohort["P1"]["hard_m1"]["pixel_area_weighted"]
    pdsr_m1=cohort["P2"]["m1"]["pixel_area_weighted"]-cohort["P1"]["m1"]["pixel_area_weighted"]
    pdsr_go=pdsr_gain>=.05 and pdsr_m1>=.03 and sum(x>=0 for x in deltas["P2"].values())>=3 and not mechanism["P2"].get("layer_collapse",False)
    m2=cohort["P2"]["hard_m1"]["semantic_margin"]["positive_fraction"] or 0; m3=cohort["P3"]["hard_m1"]["semantic_margin"]["positive_fraction"] or 0
    vpca_gain=cohort["P3"]["hard_m1"]["pixel_area_weighted"]-cohort["P2"]["hard_m1"]["pixel_area_weighted"]
    vpca_go=vpca_gain>=.05 and m3-m2>=.05 and sum((scores["P3"]["class_iou"][str(c)]-scores["P2"]["class_iou"][str(c)])>=0 for c in range(4))>=3 and not mechanism["P3"].get("concept_collapse",False)
    delta_p3=100*(scores["P3"]["mIoU"]-scores["P0"]["mIoU"]); class_damage=any(x< -1 for x in deltas["P3"].values()); ccra_collapse=not ccra_identity["exact_upstream_identity"]
    full25=delta_p3>=.5 and pdsr_go and vpca_go and safety["P3"]["nce"]>1.5 and not class_damage and not ccra_collapse and not mechanism["P3"].get("concept_collapse",False)
    if full25: final="FULL25_GO"
    elif (pdsr_go or vpca_go) and .2<=delta_p3<.5: final="MECHANISM_GO"
    else: final="FULL_MODEL_NOGO"
    compact_cohort={v:{k:{x:y for x,y in block.items() if x not in ("per_image","components_table")} for k,block in value.items()} for v,value in cohort.items()}
    compact_safety={v:{k:x for k,x in value.items() if k!="per_image"} for v,value in safety.items()}
    result={"FINAL_DECISION":final,"FULL25_GO":full25,"segmentation":scores,"delta_miou_pp":{"P1-P0":100*(scores['P1']['mIoU']-scores['P0']['mIoU']),"P2-P1":100*(scores['P2']['mIoU']-scores['P1']['mIoU']),"P3-P2":100*(scores['P3']['mIoU']-scores['P2']['mIoU']),"P3-P0":delta_p3},
            "cohort":compact_cohort,"tp_safety":compact_safety,"mechanism":mechanism,"gamma":gamma,"per_class_delta_vs_p0_pp":deltas,
            "PDSR_DECISION":"GO" if pdsr_go else "NOGO","VPCA_DECISION":"GO" if vpca_go else "NOGO","pdsr_hmcr_gain":pdsr_gain,"pdsr_m1cr_gain":pdsr_m1,"vpca_hmcr_gain":vpca_gain,"vpca_margin_gain":m3-m2,
            "LAYER_COLLAPSE":mechanism["P2"].get("layer_collapse",False) or mechanism["P3"].get("layer_collapse",False),"CONCEPT_COLLAPSE":mechanism["P3"].get("concept_collapse",False),
            "CCRA_COLLAPSE":ccra_collapse,"ccra_health":{"effective_query_ratio":1.0,"responsibility_overlap_change":0.0,"query_peak_diversity_change":0.0,"identity_audit":ccra_identity,"reason":"CCRA tensors are exactly identical because the frozen CCRA computation is structurally upstream of the only H5 residual injection"},
            "CLASS_DAMAGE":class_damage,"VLM_IGNORED":gamma["P3"]["vlm_ignored"],"bootstrap":bootstrap,"runtime":runtimes,"parameter_counts":json.loads((out/"manifests/parameter_counts.json").read_text()),
            "hard_m1":{"count":int(frame.hard_m1.sum()),"area":int(frame.loc[frame.hard_m1,"area"].sum())},"m1":{"historical":4440,"exact":int(frame.m1.sum())},"validation_only_after_e5":True}
    write_json(out/"metrics/final_result.json",result)
    pd.DataFrame([{"variant":v,"mIoU":scores[v]["mIoU"],"mDice":scores[v]["mDice"],**{f"IoU_C{c}":scores[v]["class_iou"][str(c)] for c in range(4)}} for v in ("P0","P1","P2","P3")]).to_csv(out/"metrics/segmentation.csv",index=False)
    # Machine-readable metric tables mirror the preregistered output contract.
    pd.DataFrame([{"variant":v,**{k:value for k,value in cohort[v]["hard_m1"].items() if k not in ("per_image","components_table","per_class","semantic_margin")}} for v in ("P1","P2","P3")]).to_csv(out/"metrics/hard_m1.csv",index=False)
    pd.DataFrame([{"variant":v,**{k:value for k,value in cohort[v]["m1"].items() if k not in ("per_image","components_table","per_class","semantic_margin")}} for v in ("P1","P2","P3")]).to_csv(out/"metrics/m1.csv",index=False)
    pd.DataFrame([{"variant":v,"cohort":c,**cohort[v][c]["semantic_margin"]} for v in ("P1","P2","P3") for c in ("hard_m1","m1")]).to_csv(out/"metrics/semantic_margin.csv",index=False)
    pd.DataFrame([{"variant":v,"cohort":c,"wrong_rival_persistence":cohort[v][c]["wrong_rival_persistence"]} for v in ("P1","P2","P3") for c in ("hard_m1","m1")]).to_csv(out/"metrics/rival_persistence.csv",index=False)
    pd.DataFrame([{"variant":v,**{k:value for k,value in safety[v].items() if k!="per_image"}} for v in ("P1","P2","P3")]).to_csv(out/"metrics/tp_safety.csv",index=False)
    pd.DataFrame([{"variant":v,"class":c,"iou":scores[v]["class_iou"][str(c)],"dice":scores[v]["class_dice"][str(c)]} for v in ("P0","P1","P2","P3") for c in range(4)]).to_csv(out/"metrics/per_class.csv",index=False)
    pd.DataFrame([{"variant":v,"layer":layer,"beta_mean":mechanism[v].get("beta_mean",[None]*3)[layer],"beta_std":mechanism[v].get("beta_std",[None]*3)[layer],"beta_gt_0_9_fraction":mechanism[v].get("beta_gt_0_9_fraction",[None]*3)[layer],"reconstruction_visual_ratio":mechanism[v].get("reconstruction_visual_ratio_mean",[None]*3)[layer]} for v in ("P2","P3") for layer in range(3)]).to_csv(out/"metrics/pdsr_mechanism.csv",index=False)
    pd.DataFrame([{"class":int(c),**values} for c,values in mechanism["P3"].get("concept_per_class",{}).items()]).to_csv(out/"metrics/vpca_concepts.csv",index=False)
    pd.DataFrame([result["ccra_health"]|{"collapse":result["CCRA_COLLAPSE"]}]).to_csv(out/"metrics/ccra_health.csv",index=False)
    for v in ("P1","P2","P3"):
        compute[v].update({"eval_fps_3view":runtimes[v]["fps"],"eval_peak_vram_gib":runtimes[v]["peak_vram_gib"],**result["parameter_counts"][v]})
    pd.DataFrame([{"variant":v,**compute[v]} for v in ("P0","P1","P2","P3")]).to_csv(out/"metrics/computational_cost.csv",index=False)
    gradients={v:json.loads((out/d/"gradient_manifest.json").read_text()) for v,d in VARIANT_DIR.items()}; write_json(out/"manifests/gradient_manifest.json",gradients)
    write_json(out/"metrics/bootstrap_ci.json",bootstrap); write_json(out/"metrics/computational_cost.json",{"runtime":runtimes,"parameters":result["parameter_counts"],"profiles":compute})
    print(json.dumps({"event":"PDSR_VPCA_EVALUATION_COMPLETE","decision":final,"delta":result["delta_miou_pp"],"PDSR":result["PDSR_DECISION"],"VPCA":result["VPCA_DECISION"]}),flush=True)
if __name__=="__main__": main()
