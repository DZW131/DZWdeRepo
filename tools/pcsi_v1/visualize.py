"""Pre-registered A/B/C PCSI examples from frozen UMRF components after E5."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from network.cirv import extract_regions


def locate_image(root:Path,image_id:str):
    for suffix in (".png",".jpg"):
        path=root/f"{image_id}{suffix}"
        if path.exists(): return path
    raise FileNotFoundError(image_id)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--umrf",type=Path,required=True); p.add_argument("--val-root",type=Path,required=True)
    a=p.parse_args()
    pred={}
    for v,d in (("C1","C1_VLM_LAST"),("C2","C2_STATIC_PDSR")):
        z=np.load(a.output/d/"e5_predictions.npz",allow_pickle=False)
        pred[v]={str(name):z["predictions"][i] for i,name in enumerate(z["image_ids"].astype(str))}
    bank_ids=np.load(a.umrf/"image_ids.npy",allow_pickle=False).astype(str)
    bank=np.load(a.umrf/"gt_free_prediction_maps.uint8.npy",mmap_mode="r")
    baseline={str(name):bank[0,i] for i,name in enumerate(bank_ids)}
    resp=pd.read_parquet(a.output/"C2_STATIC_PDSR/e5_responsibility.parquet")
    candidates={"A":[],"B":[],"C":[]}
    for image_id,rows in resp.groupby(resp.image_id.astype(str)):
        b=baseline[image_id]; c2=pred["C2"][image_id]
        gt=np.asarray(Image.open(a.val_root/"mask"/f"{image_id}.png")); valid=gt<4
        regions={(int(r["class_id"]),int(r["component_id"])):r["mask"] for r in extract_regions(b)}
        for _,row in rows[rows.evaluable].iterrows():
            key=(int(row.baseline_class),int(row.component_id)); mask=regions[key]
            wrong=mask&valid&(b!=gt); corrected=int((wrong&(c2==gt)).sum()); denom=int(wrong.sum())
            right=mask&valid&(b==gt); harmed=int((right&(c2!=gt)).sum())
            if bool(row.hard_m1) and int(row.new_l5)==int(row.true_class):
                group="A" if corrected>0 else "B"
                score=corrected/max(denom,1) if group=="A" else denom
                candidates[group].append((score,image_id,key))
            if harmed>0:
                candidates["C"].append((harmed,image_id,key))
    palette=np.array([[180,35,45],[70,130,180],[255,180,50],[110,65,135],[255,255,255]],dtype=np.uint8)
    manifest=[]
    for group,rows in candidates.items():
        rows.sort(reverse=True); used=set(); selected=[]
        for score,image_id,key in rows:
            if image_id in used: continue
            used.add(image_id); selected.append((score,image_id,key))
            if len(selected)==20: break
        folder=a.output/"visualizations"/group; folder.mkdir(parents=True,exist_ok=True)
        for index,(score,image_id,key) in enumerate(selected,1):
            raw=np.asarray(Image.open(locate_image(a.val_root/"img",image_id)).convert("RGB"))
            gt=np.asarray(Image.open(a.val_root/"mask"/f"{image_id}.png"))
            b=baseline[image_id]; region=next(r["mask"] for r in extract_regions(b) if (int(r["class_id"]),int(r["component_id"]))==key)
            fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
            panes=(("RGB",raw),("Ground truth",palette[np.minimum(gt,4)]),("C0",palette[np.minimum(b,4)]),
                   ("C1",palette[np.minimum(pred["C1"][image_id],4)]),("C2",palette[np.minimum(pred["C2"][image_id],4)]),
                   ("Frozen component",np.where(region[...,None],raw,np.clip(raw.astype(float)*.25,0,255).astype(np.uint8))))
            for ax,(title,image) in zip(axes.flat,panes):
                ax.imshow(image); ax.set_title(title); ax.axis("off")
            fig.suptitle(f"PCSI {group} | {image_id} | C{key[0]} component {key[1]} | score {score:.3f}")
            path=folder/f"{index:02d}_{image_id}.png"; fig.savefig(path,dpi=110); plt.close(fig)
            manifest.append({"group":group,"image_id":image_id,"component_id":key[1],"baseline_class":key[0],"score":score,"path":str(path)})
    pd.DataFrame(manifest).to_csv(a.output/"visualizations/manifest.csv",index=False)
    print({"A":sum(x["group"]=="A" for x in manifest),"B":sum(x["group"]=="B" for x in manifest),
           "C":sum(x["group"]=="C" for x in manifest)},flush=True)


if __name__=="__main__": main()
