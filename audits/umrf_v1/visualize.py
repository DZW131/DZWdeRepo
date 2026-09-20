"""Render the four prespecified UMRF case banks and summary figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import ListedColormap

from audits.umrf_v1.freeze_evidence import MAP_KEYS
from network.cirv import extract_regions

CMAP=ListedColormap(["#d73027","#1a9850","#4575b4","#984ea3","#f2f2f2"])


def case_groups(frame: pd.DataFrame) -> dict[str,pd.DataFrame]:
    t=frame.true_class
    c5,c4,c3=(frame[f"common{s}_pred"] for s in ("5","4","3"))
    s4,s3=frame.sequential4_pred,frame.sequential3_pred
    return {
        "V1_H5_wrong_H4_true_H3_true":frame[(c5!=t)&(c4==t)&(c3==t)],
        "V2_H5_correct_shallow_wrong_consensus":frame[(c5==t)&(c4==c3)&(c4!=t)],
        "V3_all_same_wrong_rival":frame[(c5==c4)&(c4==c3)&(c5!=t)],
        "V4_common_rescue_sequential_failure":frame[(c5!=t)&(c4==t)&(c3==t)&~((s4==t)&(s3==t))],
    }


def render_case(row, val_root:Path, maps, output:Path) -> None:
    idx=int(row.image_index); image_id=str(row.image_id)
    image=np.asarray(Image.open(val_root/"img"/f"{image_id}.png").convert("RGB")); truth=np.asarray(Image.open(val_root/"mask"/f"{image_id}.png"))
    baseline=maps[MAP_KEYS.index("baseline"),idx]; regions=extract_regions(baseline)
    matches=[region for region in regions if int(region["class_id"])==int(row.baseline_class) and int(region["component_id"])==int(row.component_id)]
    if len(matches)!=1: raise AssertionError(f"Component key mismatch for {image_id}")
    mask=matches[0]["mask"]
    panels=[("Image",image), ("GT",truth), ("HQMR final",baseline)]
    for chain in ("common","sequential"):
        for stage in ("5","4","3"): panels.append((f"{chain[:3].upper()} C{stage}",maps[MAP_KEYS.index(f"{chain}{stage}"),idx]))
    fig,axes=plt.subplots(3,3,figsize=(12,11))
    for ax,(title,value) in zip(axes.ravel(),panels):
        if value.ndim==3: ax.imshow(value)
        else: ax.imshow(value,cmap=CMAP,vmin=0,vmax=4,interpolation="nearest")
        ax.contour(mask.astype(float),levels=[.5],colors=["yellow"],linewidths=.8); ax.set_title(title); ax.axis("off")
    fig.suptitle(f"{image_id} comp={int(row.component_id)} GT=C{int(row.true_class)} | CQ 5/4/3={int(row.common5_pred)}/{int(row.common4_pred)}/{int(row.common3_pred)} | SEQ={int(row.sequential5_pred)}/{int(row.sequential4_pred)}/{int(row.sequential3_pred)}\nCQ margins={row.common5_margin:.3f}/{row.common4_margin:.3f}/{row.common3_margin:.3f}; SEQ={row.sequential5_margin:.3f}/{row.sequential4_margin:.3f}/{row.sequential3_margin:.3f}",fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(output,dpi=130); plt.close(fig)


def summary_figures(result:dict, frame:pd.DataFrame, output:Path) -> None:
    output.mkdir(parents=True,exist_ok=True)
    metrics=("precision","coverage","nce","h5_correct_harm_rate")
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(metrics)); width=.35
    for i,chain in enumerate(("common","sequential")): ax.bar(x+(i-.5)*width,[result["summaries"][chain]["area"][m] for m in metrics],width,label=chain)
    ax.set_xticks(x,metrics,rotation=15); ax.legend(); ax.set_title("Area-weighted consensus health"); fig.tight_layout(); fig.savefig(output/"summary_01_consensus_health.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); cats=list(result["summaries"]["common"]["rescue_matrix_area"]); x=np.arange(len(cats))
    for i,chain in enumerate(("common","sequential")): ax.bar(x+(i-.5)*width,[result["summaries"][chain]["rescue_matrix_area"][c]["fraction"] for c in cats],width,label=chain)
    ax.set_xticks(x,[c.replace("_","\n") for c in cats],fontsize=8); ax.legend(); ax.set_title("H5-wrong rescue matrix"); fig.tight_layout(); fig.savefig(output/"summary_02_rescue_matrix.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(4)
    for i,chain in enumerate(("common","sequential")): ax.bar(x+(i-.5)*width,[result["summaries"][chain]["per_class_area"][str(c)]["coverage"] for c in range(4)],width,label=chain)
    ax.set_xticks(x,[f"C{c}" for c in range(4)]); ax.set_ylabel("coverage"); ax.legend(); ax.set_title("Per-class corrective coverage"); fig.tight_layout(); fig.savefig(output/"summary_03_class_coverage.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); labels=[]; values=[]
    for chain in ("common","sequential"):
        for rule in ("r1","r2","r3"): labels.append(f"{chain[:3]}-{rule}"); values.append(result["pixel"][chain][f"gain_{rule}"]*100)
    ax.bar(labels,values,color=["#4575b4"]*3+["#d73027"]*3); ax.axhline(0,color="black",lw=.8); ax.set_ylabel("net corrected pixels (%)"); ax.set_title("Fixed-rule pixel net gain over H5"); fig.tight_layout(); fig.savefig(output/"summary_04_rule_net_gain.png",dpi=180); plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--val-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--per-category",type=int,default=20); args=p.parse_args()
    frame=pd.read_parquet(args.output/"component_evidence_with_gt.parquet"); maps=np.load(args.output/"gt_free_prediction_maps.uint8.npy",mmap_mode="r")
    result=json.loads((args.output/"02_evaluation.json").read_text()); root=args.output/"visualizations"; root.mkdir(parents=True,exist_ok=True)
    counts={}
    for name,group in case_groups(frame).items():
        selected=group.sort_values(["area","purity"],ascending=False).head(args.per_category); directory=root/name; directory.mkdir(parents=True,exist_ok=True)
        for rank,(_,row) in enumerate(selected.iterrows(),1): render_case(row,args.val_root,maps,directory/f"{rank:02d}_{row.image_id}_c{int(row.component_id)}.png")
        counts[name]={"available":len(group),"rendered":len(selected)}
    summary_figures(result,frame,root/"summary")
    (root/"visualization_manifest.json").write_text(json.dumps({"requested_per_category":args.per_category,"categories":counts,"summary_figures":4},indent=2),encoding="utf-8")
    print(json.dumps({"event":"UMRF_VISUALIZED","categories":counts}),flush=True)


if __name__=="__main__": main()
