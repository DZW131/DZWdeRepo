"""Frozen-model qualitative examples selected from OOF scores, never probe inputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from audits.oracle_action_separability_v1.freeze_observables import infer
from audits.ucrf_v1.gate import load_model
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import prediction_from_cam

ALPHAS=[0,.25,.5,1,1.5,2,3,4]


def select_rows(a,al,g,gl,aps,gs):
    data=al.copy()
    data["score"]=aps
    actionable=data.action!="KEEP"
    ap2=pd.read_parquet(select_rows.output/"arbitration/AP2_oof_scores.parquet")
    data["direction_score"]=np.nan
    data.loc[actionable,"direction_score"]=ap2.oof_score.to_numpy()
    data["up_score"]=data.score*data.direction_score.fillna(0)
    data["down_score"]=data.score*(1-data.direction_score.fillna(0))
    gate=gl.copy();gate["score"]=gs
    selections={
      "A_true_UP_high":data[data.action=="UP"].sort_values("up_score",ascending=False).head(20),
      "A_true_DOWN_high":data[data.action=="DOWN"].sort_values("down_score",ascending=False).head(20),
      "A_KEEP_false_selected":data[data.action=="KEEP"].sort_values("score",ascending=False).head(20),
      "A_action_missed":data[data.action!="KEEP"].sort_values("score").head(20),
      "G_beneficial_high":gate[gate.beneficial_rescue==1].sort_values("score",ascending=False).head(20),
      "G_harmful_false_rescue":gate[gate.gate_gain_pixels<0].sort_values("score",ascending=False).head(20),
      "G_high_score_FP":gate[gate.beneficial_rescue==0].sort_values("score",ascending=False).head(20),
      "G_beneficial_missed":gate[gate.beneficial_rescue==1].sort_values("score").head(20),
    }
    return selections


def render_a(path, image, bundle, row):
    cls=int(row.predicted_class);curve=json.loads(row.alpha_gain_curve)
    fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    axes[0,0].imshow(image);axes[0,0].set_title("BCSS image")
    axes[0,1].imshow(bundle["prediction"],vmin=0,vmax=3,cmap="tab10");axes[0,1].set_title("Frozen baseline prediction")
    axes[0,2].plot(ALPHAS,curve,"o-");axes[0,2].axvline(1,color="gray",ls="--")
    axes[0,2].set_title(f"Oracle gain curve; {row.action} alpha={row.oracle_alpha}")
    axes[0,2].set_xlabel("alpha");axes[0,2].set_ylabel("correct-pixel delta")
    axes[1,0].imshow(bundle["maps"]["u5"].mean(0),cmap="coolwarm");axes[1,0].set_title("GT-free deep query logit mean")
    axes[1,1].imshow(bundle["maps"]["d4"].mean(0),cmap="coolwarm");axes[1,1].set_title("GT-free local query logit mean")
    axes[1,2].imshow(bundle["maps"]["cam"][cls],vmin=0,vmax=1,cmap="viridis")
    axes[1,2].set_title(f"C{cls} CAM | AP1={row.score:.3f} | gain={row.oracle_gain_pixels}")
    for ax in (axes[0,0],axes[0,1],axes[1,0],axes[1,1],axes[1,2]):ax.axis("off")
    fig.savefig(path,dpi=110);plt.close(fig)


def render_g(path,image,bundle,row):
    cls=int(row.candidate_class)
    force=bundle["label"].copy();force[cls]=1
    forced=prediction_from_cam(bundle["maps"]["cam"],force,bundle["prediction"])
    fig,axes=plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
    axes[0,0].imshow(image);axes[0,0].set_title("BCSS image")
    axes[0,1].imshow(bundle["prediction"],vmin=0,vmax=3,cmap="tab10");axes[0,1].set_title("Baseline")
    axes[0,2].imshow(forced,vmin=0,vmax=3,cmap="tab10");axes[0,2].set_title(f"Force ON C{cls}; net gain={row.gate_gain_pixels}")
    axes[1,0].imshow(bundle["maps"]["c4"][cls],cmap="viridis");axes[1,0].set_title("C4 candidate evidence")
    axes[1,1].imshow(bundle["maps"]["c3"][cls],cmap="viridis");axes[1,1].set_title("C3 candidate evidence")
    axes[1,2].imshow(bundle["maps"]["cam"][cls],vmin=0,vmax=1,cmap="viridis")
    views=", ".join(f"{v:.3f}" for v in bundle["gate_views"][:,cls])
    axes[1,2].set_title(f"CAM | OOF={row.score:.3f}\nDeep views: {views}")
    for ax in axes.flat:ax.axis("off")
    fig.savefig(path,dpi=110);plt.close(fig)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    for name in ("checkpoint","val-root","output"):
        p.add_argument(f"--{name}",type=Path,required=True)
    args=p.parse_args();out=args.output
    a=pd.read_parquet(out/"arbitration/observable_features.parquet")
    al=pd.read_parquet(out/"arbitration/oracle_action_labels.parquet")
    g=pd.read_parquet(out/"gate/gate_off_pairs.parquet")
    gl=pd.read_parquet(out/"gate/rescue_oracle_labels.parquet")
    aps=pd.read_parquet(out/"arbitration/AP1_oof_scores.parquet").oof_score.to_numpy()
    gs=pd.read_parquet(out/"gate/G_oof_scores.parquet").oof_score.to_numpy()
    select_rows.output=out
    choices=select_rows(a,al,g,gl,aps,gs)
    dataset=Stage1_InferDataset(str(args.val_root/"img"),img_size=224)
    image_index={str(dataset[i][0]):i for i in range(len(dataset))}
    model=load_model(args.checkpoint)
    manifest=[]
    cache={}
    for category,frame in choices.items():
        directory=out/"visualizations"/category;directory.mkdir(parents=True,exist_ok=True)
        for ordinal,row in enumerate(frame.itertuples(),1):
            image_id=str(row.image_id)
            if image_id not in cache:
                _,tensor=dataset[image_index[image_id]]
                bundle=infer(model,tensor[None].cuda())
                image=np.asarray(Image.open(args.val_root/"img"/f"{image_id}.png").convert("RGB"))
                if len(cache)>=4:cache.pop(next(iter(cache)))
                cache[image_id]=(image,bundle)
            image,bundle=cache[image_id]
            name=f"{ordinal:03d}_{image_id}_{'component'+str(row.component_id) if category.startswith('A') else 'class'+str(row.candidate_class)}.png"
            path=directory/name
            if category.startswith("A"):render_a(path,image,bundle,row)
            else:render_g(path,image,bundle,row)
            manifest.append({"category":category,"image_id":image_id,"path":str(path),
                             "score":float(row.score)})
        print(json.dumps({"event":"visualized","category":category,"count":len(frame)}),flush=True)
    (out/"visualizations/manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"event":"VISUALIZATION_COMPLETE","count":len(manifest)}),flush=True)


if __name__=="__main__":main()
