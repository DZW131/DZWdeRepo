"""Render fixed P1/P2/P3 case banks after the unified E5 evaluation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from PIL import Image
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from network.cirv import extract_regions
from tools.pdsr_vpca_phase0.common import load_concepts

DIR={"P1":"P1_VLM_LAST","P2":"P2_STATIC_PDSR","P3":"P3_VPCA_PDSR"}; CMAP=ListedColormap(["#d73027","#1a9850","#4575b4","#984ea3","#f5f5f5"])

def colored(ax,value,title,mask=None):
    ax.imshow(value,cmap=CMAP,vmin=0,vmax=4,interpolation="nearest");
    if mask is not None: ax.contour(mask.astype(float),[.5],colors="yellow",linewidths=.7)
    ax.set_title(title); ax.axis("off")

def region_for(pred,row):
    matches=[r for r in extract_regions(pred) if int(r["class_id"])==int(row.baseline_class) and int(r["component_id"])==int(row.component_id)]
    if len(matches)!=1: raise AssertionError("Component lookup failed")
    return matches[0]["mask"]

def candidates(frame,p0,pnew,ids,val_root):
    index={str(x):i for i,x in enumerate(ids)}; rows=[]
    # Region extraction is image-level work. Cache all components once per image;
    # otherwise the 11k-component audit repeatedly labels the same prediction map.
    region_cache={}
    for image_id in frame.image_id.astype(str).unique():
        i=index.get(image_id)
        if i is not None:
            region_cache[image_id]={(int(x["class_id"]),int(x["component_id"])):x["mask"] for x in extract_regions(p0[i])}
    for _,r in frame.iterrows():
        image_id=str(r.image_id); i=index[image_id]; truth=np.asarray(Image.open(val_root/"mask"/f"{image_id}.png")); mask=region_cache[image_id][(int(r.baseline_class),int(r.component_id))]; valid=truth<4
        wrong=mask&valid&(p0[i]!=truth); right=mask&valid&(p0[i]==truth)
        rows.append({**r.to_dict(),"corrected":int((wrong&(pnew[i]==truth)).sum()),"wrong":int(wrong.sum()),"harmed":int((right&(pnew[i]!=truth)).sum()),"index":i})
    return pd.DataFrame(rows)

def render_standard(row,variant,p0,pnew,mech,ids,val_root,path):
    i=int(row["index"]); image_id=str(ids[i]); image=np.asarray(Image.open(val_root/"img"/f"{image_id}.png").convert("RGB")); truth=np.asarray(Image.open(val_root/"mask"/f"{image_id}.png")); mask=region_for(p0[i],pd.Series(row))
    energy=mech["energy"][i].astype(np.float32); error=np.zeros((*truth.shape,3),np.uint8); valid=truth<4; error[valid&(p0[i]!=truth)&(pnew[i]==truth)]=(0,210,255); error[valid&(p0[i]==truth)&(pnew[i]!=truth)]=(255,60,60); error[valid&(p0[i]!=truth)&(pnew[i]!=truth)]=(255,180,0)
    fig,axes=plt.subplots(2,3,figsize=(11,7)); axes[0,0].imshow(image); axes[0,0].set_title("Image"); axes[0,0].axis("off"); colored(axes[0,1],truth,"GT",mask); colored(axes[0,2],p0[i],"P0 HQMR",mask); colored(axes[1,0],pnew[i],variant,mask); axes[1,1].imshow(error); axes[1,1].set_title("cyan recovered / red harmed"); axes[1,1].axis("off"); axes[1,2].imshow(energy,cmap="magma"); axes[1,2].set_title("VLM/PDSR energy"); axes[1,2].axis("off")
    fig.suptitle(f"{image_id} C{int(row['true_class'])} rival C{int(row['baseline_class'])} corrected={int(row['corrected'])}/{int(row['wrong'])} harmed={int(row['harmed'])}"); fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(path,dpi=130); plt.close(fig)

def render_grounding(row,variant,p0,pnew,mech,ids,val_root,concepts,path):
    i=int(row["index"]); image_id=str(ids[i]); image=np.asarray(Image.open(val_root/"img"/f"{image_id}.png").convert("RGB")); truth=np.asarray(Image.open(val_root/"mask"/f"{image_id}.png")); mask=region_for(p0[i],pd.Series(row)); true=int(row["true_class"]); rival=int(row["baseline_class"])
    evidence=mech["class_evidence"][i].astype(np.float32); layers=mech["layer_class_evidence"][i].astype(np.float32); beta=mech["beta"][i].astype(np.float32)
    fig,axes=plt.subplots(4,3,figsize=(12,13)); axes[0,0].imshow(image); axes[0,0].axis("off"); axes[0,0].set_title("Image"); colored(axes[0,1],truth,"GT",mask); colored(axes[0,2],pnew[i],variant,mask)
    for j in range(3):
        axes[1,j].imshow(layers[j,true],cmap="magma"); axes[1,j].set_title(f"block {(4,8,12)[j]} C{true} response"); axes[1,j].axis("off")
        axes[2,j].imshow(beta[j],vmin=0,vmax=1,cmap="viridis"); axes[2,j].set_title(f"beta block {(4,8,12)[j]}"); axes[2,j].axis("off")
    axes[3,0].imshow(evidence[true],cmap="magma"); axes[3,0].set_title(("static concept" if variant=="P2" else "VPCA")+f" true C{true}"); axes[3,0].axis("off"); axes[3,1].imshow(evidence[rival],cmap="magma"); axes[3,1].set_title(f"rival C{rival} semantic"); axes[3,1].axis("off")
    if len(mech["q"]):
        q=mech["q"][i,true].astype(float); omega=mech["omega"][i].reshape(4,8)[true].astype(float); top=np.argsort(-q)[:5]; text="\n".join(f"q={q[k]:.3f} ω={omega[k]:.3f}  {concepts[8*true+k]}" for k in top)+f"\nrho={float(mech['rho'][i,true]):.3f}"
    else: text="Static concept participation: 1/32 each"
    axes[3,2].text(0,.95,text,va="top",wrap=True,fontsize=9); axes[3,2].axis("off"); fig.suptitle(f"{image_id}: pathology concept grounding"); fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(path,dpi=130); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--val-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--umrf",type=Path,required=True); p.add_argument("--concepts",type=Path,required=True); a=p.parse_args()
    frame=pd.read_parquet(a.umrf/"component_evidence_with_gt.parquet"); frame=frame[frame.evaluable].copy(); frame["hard_m1"]=(frame.sequential5_pred!=frame.true_class)&(frame.sequential4_pred!=frame.true_class)&(frame.sequential3_pred!=frame.true_class)
    ids=np.load(a.umrf/"image_ids.npy",allow_pickle=False).astype(str); p0=np.load(a.umrf/"gt_free_prediction_maps.uint8.npy",mmap_mode="r")[0]; id_to_p0={x:p0[i] for i,x in enumerate(ids)}
    concepts,_=load_concepts(a.concepts); manifest={}
    for variant,directory in DIR.items():
        bank=np.load(a.output/directory/"e5_predictions.npz",allow_pickle=False); vids=bank["image_ids"].astype(str); pred=bank["predictions"]; order=np.asarray([np.where(ids==x)[0][0] for x in vids]); base=p0[order]
        mech=dict(np.load(a.output/directory/"e5_mechanism.npz",allow_pickle=False)); c=candidates(frame,base,pred,vids,a.val_root)
        groups={"hard_m1_recovered":c[c.hard_m1&(c.corrected>0)].sort_values("corrected",ascending=False),"hard_m1_unrecovered":c[c.hard_m1&(c.wrong>0)&(c.corrected==0)].sort_values("area",ascending=False),"baseline_correct_harmed":c[(c.harmed>0)&c.baseline_correct].sort_values("harmed",ascending=False)}
        counts={}; root=a.output/"visualizations"/variant
        for name,group in groups.items():
            selected=group.head(20); d=root/name; d.mkdir(parents=True,exist_ok=True)
            for rank,(_,row) in enumerate(selected.iterrows(),1): render_standard(row,variant,base,pred,mech,vids,a.val_root,d/f"{rank:02d}_{row.image_id}_c{int(row.component_id)}.png")
            counts[name]={"available":len(group),"rendered":len(selected)}
        if variant in ("P2","P3"):
            selected=groups["hard_m1_recovered"].head(20); d=root/"concept_grounding"; d.mkdir(parents=True,exist_ok=True)
            for rank,(_,row) in enumerate(selected.iterrows(),1): render_grounding(row,variant,base,pred,mech,vids,a.val_root,concepts,d/f"{rank:02d}_{row.image_id}_c{int(row.component_id)}.png")
            counts["concept_grounding"]={"available":len(groups["hard_m1_recovered"]),"rendered":len(selected)}
        manifest[variant]=counts
    (a.output/"visualizations/visualization_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); print(json.dumps({"event":"PDSR_VISUALIZATION_COMPLETE","counts":manifest}),flush=True)
if __name__=="__main__": main()
