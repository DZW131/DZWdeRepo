#!/usr/bin/env python3
"""Fixed-E5 evaluation, mechanism audit, visualization, and report for RACC-v1 Phase0."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from network.hqmr_net import HQMRNet
from network.racc_net import RACCNet
from tool.GenDataset import Stage1_InferDataset
from tools.eval_gcqm_full25_bcss_seed42 import THRESHOLDS, TTA, foreground_confusion, load_state, normalize_cam, prediction_from_cam, presence, resize_unflip, scores_from_confusion
from tools.hqrf_phase0_io import sha256, write_csv, write_json

HQMR_MIOU, SSHR_MIOU, ORACLE_GAIN = .6557244403737567, .6669670591172749, 3.5908
HQMR_SHA256 = "84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
PALETTE = np.asarray([[255,0,0],[0,255,0],[0,0,255],[153,0,255],[255,255,255]], dtype=np.uint8)


def load_group(path: Path) -> RACCNet:
    model = RACCNet().cuda(); model.load_state_dict(load_state(path), strict=True); model.eval(); return model


@torch.no_grad()
def infer_hqmr(model: HQMRNet, image: torch.Tensor, original_hw) -> dict:
    maps, deep = [], []; dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16): output = model(value, dummy, step=29275)
        maps.append(resize_unflip(output["primary_output"], original_hw, cam_flip).float().cpu())
        deep.append(output["deep_gate"].float().cpu())
    cam = normalize_cam(torch.stack(maps).mean(0).numpy()); deep_probability = torch.stack(deep).mean(0).numpy()[0]
    label = presence(deep_probability)
    return {"cam": cam, "deep_probability": deep_probability, "deep_label": label, "label": label,
            "prediction": prediction_from_cam(cam, label, np.empty(original_hw))}


@torch.no_grad()
def infer(model: RACCNet, image: torch.Tensor, original_hw, enable_a: bool, enable_g: bool, detail=False) -> dict:
    maps, deep, local = [], [], []; canonical = None
    dummy = torch.ones((1, 4), device=image.device)
    for input_flip, cam_flip in TTA:
        value = torch.flip(image, dims=input_flip) if input_flip else image
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(value, dummy, step=29275, enable_arbitration=enable_a, enable_presence=enable_g)
        maps.append(resize_unflip(output["primary_output"], original_hw, cam_flip).float().cpu())
        deep.append(output["deep_gate"].float().cpu()); local.append(output["racc"]["local_presence"]["probability"].float().cpu())
        if not input_flip: canonical = output
    cam = normalize_cam(torch.stack(maps).mean(0).numpy())
    deep_probability = torch.stack(deep).mean(0).numpy()[0]
    local_probability = torch.stack(local).mean(0).numpy()[0]
    deep_label = presence(deep_probability)
    label = np.logical_or(deep_label > 0, local_probability > .5).astype(np.float32) if enable_g else deep_label
    result = {"cam": cam, "deep_probability": deep_probability, "local_probability": local_probability,
              "deep_label": deep_label, "label": label, "prediction": prediction_from_cam(cam, label, np.empty(original_hw))}
    if detail:
        stage = canonical["stages"][2]["hqmr"]
        result.update({"alpha": None if stage["alpha4"] is None else stage["alpha4"][0].float().cpu().numpy(),
                       "logits5": stage["logits5"][0].float().cpu().numpy(),
                       "direct4": stage["direct4"][0].float().cpu().numpy(),
                       "weights": stage["weights"][0].float().cpu().numpy()})
    return result


class AlphaAudit:
    def __init__(self):
        self.count = 0; self.total = 0.; self.square = 0.; self.hist = np.zeros(400, np.int64)
        self.class_sum = np.zeros(4); self.class_count = np.zeros(4); self.spatial_variance = []
        self.conflict_sum = 0.; self.conflict_count = 0; self.agree_sum = 0.; self.agree_count = 0

    def add(self, payload):
        alpha = payload["alpha"].astype(np.float64); values = alpha.ravel(); self.count += values.size
        self.total += values.sum(); self.square += np.square(values).sum(); self.hist += np.histogram(values, bins=400, range=(0,4))[0]
        self.spatial_variance.extend(alpha.reshape(alpha.shape[0], -1).var(1).tolist())
        qmean, weights = alpha.mean((1,2)), payload["weights"]
        for cls in range(4):
            denom = weights[:,cls].sum() + 1e-12; self.class_sum[cls] += float((qmean * weights[:,cls]).sum() / denom); self.class_count[cls] += 1
        coarse = F.interpolate(torch.from_numpy(payload["logits5"])[None], size=alpha.shape[-2:], mode="bilinear", align_corners=False)[0].numpy()
        direct = payload["direct4"]; conflict = (coarse * direct < 0) & (np.abs(1/(1+np.exp(-coarse))-1/(1+np.exp(-direct))) > .25)
        agree = (coarse * direct > 0) & (~conflict)
        if conflict.any(): self.conflict_sum += alpha[conflict].sum(); self.conflict_count += int(conflict.sum())
        if agree.any(): self.agree_sum += alpha[agree].sum(); self.agree_count += int(agree.sum())

    def summary(self):
        mean = self.total / max(self.count,1); variance = max(0., self.square/max(self.count,1)-mean*mean); std = math.sqrt(variance)
        centers = (np.arange(400)+.5)/100; near = float(self.hist[np.abs(centers-mean)<.2].sum()/max(self.count,1))
        edges = ((0,.5),(.5,.8),(.8,1.2),(1.2,2),(2,3),(3,4.0001)); names=("lt_0.5","0.5_0.8","0.8_1.2","1.2_2","2_3","gt_3")
        distribution = {name: float(self.hist[(centers>=lo)&(centers<hi)].sum()/max(self.count,1)) for name,(lo,hi) in zip(names,edges)}
        return {"mean": mean, "std": std, "fraction_within_0.2_of_mean": near,
                "global_weight_collapse": bool(std < .10 and near >= .90), "distribution": distribution,
                "per_class_mean": {str(c): float(self.class_sum[c]/max(self.class_count[c],1)) for c in range(4)},
                "conflict_mean": self.conflict_sum/max(self.conflict_count,1), "agreement_mean": self.agree_sum/max(self.agree_count,1),
                "mean_spatial_variance": float(np.mean(self.spatial_variance))}


def image_label(truth): return np.asarray([(truth == cls).any() for cls in range(4)], dtype=np.uint8)


def pixel_audit(truth, base, candidate):
    valid = truth < 4; correct = (base == truth) & valid; wrong = (base != truth) & valid
    corrected = int((wrong & (candidate == truth)).sum()); harmed = int((correct & (candidate != truth)).sum())
    return {"baseline_wrong": int(wrong.sum()), "baseline_correct": int(correct.sum()), "corrected": corrected, "harmed": harmed}


def safe_auc(y, p):
    try: return float(roc_auc_score(y, p))
    except ValueError: return None


def build_report(result):
    metrics, delta, gate, alpha = result["metrics"], result["delta_pp"], result["gate"], result["alpha"]
    home = (f"# RACC-v1 BCSS Seed42 Phase0 Final Report\n\n"
            f"**DECISION = {result['decision']}**\n\n"
            f"- HQMR = {100*metrics['P0_HQMR']['mIoU']:.4f}\n- RACC-A = {100*metrics['P1_RACC_A']['mIoU']:.4f}\n"
            f"- RACC-G = {100*metrics['P2_RACC_G']['mIoU']:.4f}\n- RACC-Joint = {100*metrics['P3_RACC_JOINT']['mIoU']:.4f}\n"
            f"- SSHR = {100*SSHR_MIOU:.4f}\n- Joint delta = {delta['P3_RACC_JOINT']:+.4f} pp\n"
            f"- Oracle recovery ratio = {result['oracle_recovery_ratio']:.2%}\n- Flags = {', '.join(result['flags']) or 'None'}\n")
    rows = "\n".join(f"| {name} | {100*m['mIoU']:.4f} | {100*m['mDice']:.4f} | {delta[name]:+.4f} |" for name,m in metrics.items())
    class_rows = "\n".join(f"| C{c} | " + " | ".join(f"{100*metrics[n]['class_iou'][str(c)]:.4f}" for n in metrics) + " |" for c in range(4))
    sections = [
        ("Executive Decision", f"预注册结论为 **{result['decision']}**；FINAL_CANDIDATE={result['final_candidate']}；CLASS_DAMAGE={result['class_damage']}。"),
        ("Baseline Reproduction", f"P0={100*metrics['P0_HQMR']['mIoU']:.4f}%，冻结目标=65.5724%，reproduction={result['baseline_reproduction']}。"),
        ("Identity Tests", f"四项恒等检查全部通过：{result['identity']}。"),
        ("Architecture", "RACC-A 仅以 sigmoid(U(L5))、sigmoid(D4)、绝对差和乘积产生 alpha∈(0,4)；RACC-G 仅以 C4/C3 top-5% mean、max 和跨层差产生局部 presence。"),
        ("Parameter Count/FLOPs", f"{result['complexity']}。HQMR FLOPs 由 torch.profiler 实测；controller FLOPs 为固定 224 输入解析估计，乘加按 2 FLOPs 计。"),
        ("Training Protocol", "冻结 HQMR-v1；P1/P2/P3 各 Seed42、batch20、BF16、5 epochs/5855 steps；PolyOptimizer；仅 image-level labels；固定 E5，不做验证集选点和阈值 sweep。"),
        ("RACC-A Results", f"P1 mIoU={100*metrics['P1_RACC_A']['mIoU']:.4f}%，Δ={delta['P1_RACC_A']:+.4f} pp。"),
        ("Alpha Distribution", f"P1={alpha['P1_RACC_A']}；P3={alpha['P3_RACC_JOINT']}。"),
        ("Deep-Dominance Repair", f"{result['deep_dominance']}。"),
        ("RACC-G Results", f"P2 mIoU={100*metrics['P2_RACC_G']['mIoU']:.4f}%，Δ={delta['P2_RACC_G']:+.4f} pp。"),
        ("Presence Rescue Precision/Recall", f"P2={gate['P2_RACC_G']}；P3={gate['P3_RACC_JOINT']}。安全线为 rescue precision≥80%。"),
        ("Gate-Excluded Recovery", f"{result['gate_excluded']}。类别 rescue 与实际 segmentation correction 分开统计。"),
        ("Joint RACC Results", "\n| Model | mIoU | mDice | ΔHQMR pp |\n|---|---:|---:|---:|\n" + rows),
        ("M1 Recovery", f"复核 UCRF 原始 M1 components={result['m1']['components']}；{result['m1']['by_model']}。"),
        ("TP Harm", f"{result['pixel_change']}。NCE=corrected/(harmed+ε)。"),
        ("Per-Class Results", "\n| Class | P0 | P1 | P2 | P3 |\n|---|---:|---:|---:|---:|\n" + class_rows),
        ("Oracle Recovery Ratio", f"({100*metrics['P3_RACC_JOINT']['mIoU']:.4f}-65.5724)/3.5908={result['oracle_recovery_ratio']:.2%}。"),
        ("Representative Cases", f"已按六类各选择并渲染最多20例；manifest={result['visualizations']}。"),
        ("Failure Analysis", result["failure_analysis"]),
        ("GO/NOGO", f"阈值审计：{result['decision_matrix']}。"),
        ("Exact Next Step", result["next_step"]),
    ]
    return home + "\n" + "\n\n".join(f"## {i}. {title}\n\n{body}" for i,(title,body) in enumerate(sections,1)) + f"\n\nDECISION = {result['decision']}\n"


def parse_args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--val-root",required=True);p.add_argument("--hqmr-checkpoint",required=True)
    p.add_argument("--experiment",required=True);p.add_argument("--ucrf-components",required=True);p.add_argument("--ucrf-masks",required=True)
    p.add_argument("--num-workers",type=int,default=8);return p.parse_args()


def main():
    args=parse_args(); valroot,experiment=Path(args.val_root).resolve(),Path(args.experiment).resolve(); hqmr=Path(args.hqmr_checkpoint).resolve()
    if sha256(hqmr)!=HQMR_SHA256: raise AssertionError("HQMR identity mismatch")
    runtime=json.loads((experiment/"provenance/runtime.json").read_text()); identity=json.loads((experiment/"identity_tests/identity_manifest.json").read_text())
    if runtime["status"]!="RACC_PHASE0_TRAINING_COMPLETE" or not identity["passed"]: raise AssertionError("Unsealed training")
    paths={name:Path(runtime["groups"][name]["checkpoint"]) for name in ("P1_RACC_A","P2_RACC_G","P3_RACC_JOINT")}
    for name,path in paths.items():
        if sha256(path)!=runtime["groups"][name]["sha256"]: raise AssertionError(name+" seal mismatch")
    models={name:load_group(path) for name,path in paths.items()}
    baseline=HQMRNet().cuda();baseline.load_state_dict(load_state(hqmr),strict=True);baseline.eval()
    table=pd.read_parquet(args.ucrf_components); packed=np.load(args.ucrf_masks)["packed"]; mask_shape=tuple(np.load(args.ucrf_masks)["shape"].tolist())
    m1=table[table.cohort=="M1"].copy(); m1["source_index"]=m1.index; by_image=defaultdict(list)
    for row in m1.itertuples(): by_image[row.image_id].append(row)
    if len(m1)!=4440: raise AssertionError(f"Expected 4440 M1 components, got {len(m1)}")
    loader=DataLoader(Stage1_InferDataset(str(valroot/"img"),img_size=224),batch_size=1,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    names=("P0_HQMR","P1_RACC_A","P2_RACC_G","P3_RACC_JOINT"); hist={n:[] for n in names}; predictions={n:{} for n in names}
    alpha={"P1_RACC_A":AlphaAudit(),"P3_RACC_JOINT":AlphaAudit()}; presence_rows=[]; pixel={n:defaultdict(int) for n in names[1:]}
    m1_stats={n:defaultdict(int) for n in names[1:]}; gate_subset={n:defaultdict(int) for n in ("P2_RACC_G","P3_RACC_JOINT")}; deep_subset={n:defaultdict(float) for n in ("P1_RACC_A","P3_RACC_JOINT")}
    image_rows=[]; started=time.perf_counter()
    for index,(ids,image) in enumerate(loader,1):
        image_id=ids[0]; truth=np.asarray(Image.open(valroot/"mask"/f"{image_id}.png")); image=image.cuda(non_blocking=True); hw=truth.shape
        p0=infer_hqmr(baseline,image,hw);p2=infer(models["P2_RACC_G"],image,hw,False,True,True); p1=infer(models["P1_RACC_A"],image,hw,True,False,True); p3=infer(models["P3_RACC_JOINT"],image,hw,True,True,True)
        bundle={"P0_HQMR":p0,"P1_RACC_A":p1,"P2_RACC_G":p2,"P3_RACC_JOINT":p3}; y=image_label(truth)
        for n,b in bundle.items(): hist[n].append(foreground_confusion(truth,b["prediction"])); predictions[n][image_id]=b["prediction"].astype(np.uint8)
        alpha["P1_RACC_A"].add(p1);alpha["P3_RACC_JOINT"].add(p3)
        for n,b in (("P2_RACC_G",p2),("P3_RACC_JOINT",p3)):
            for cls in range(4): presence_rows.append({"image_id":image_id,"model":n,"class":cls,"truth":int(y[cls]),"deep":int(b["deep_label"][cls]),"local_probability":float(b["local_probability"][cls]),"local":int(b["local_probability"][cls]>.5),"rescued":int(not b["deep_label"][cls] and b["local_probability"][cls]>.5)})
        row={"image_id":image_id}
        for n in names[1:]:
            audit=pixel_audit(truth,p0["prediction"],bundle[n]["prediction"])
            for k,v in audit.items():pixel[n][k]+=v
            row[n+"__delta_correct_pixels"]=audit["corrected"]-audit["harmed"]
        p2_rescued=(p2["deep_label"]==0)&(p2["local_probability"]>.5);p3_rescued=(p3["deep_label"]==0)&(p3["local_probability"]>.5)
        row["P2_true_rescues"]=int((p2_rescued&y.astype(bool)).sum());row["P2_false_rescues"]=int((p2_rescued&~y.astype(bool)).sum())
        row["P3_true_rescues"]=int((p3_rescued&y.astype(bool)).sum());row["P3_false_rescues"]=int((p3_rescued&~y.astype(bool)).sum());image_rows.append(row)
        for component in by_image.get(image_id,[]):
            mask=np.unpackbits(packed[component.source_index],bitorder="big")[:np.prod(mask_shape)].reshape(mask_shape).astype(bool); cls=int(component.true_class); area=int(mask.sum())
            for n in names[1:]:
                fixed=int((bundle[n]["prediction"][mask]==cls).sum());m1_stats[n]["pixels"]+=area;m1_stats[n]["corrected"]+=fixed;m1_stats[n]["components"]+=1;m1_stats[n]["components_recovered"]+=int(fixed>area/2)
            excluded=p0["deep_label"][cls]==0
            if excluded:
                for n,b in (("P2_RACC_G",p2),("P3_RACC_JOINT",p3)):
                    rescued=b["label"][cls]>0;gate_subset[n]["components"]+=1;gate_subset[n]["class_rescued"]+=int(rescued);gate_subset[n]["segmentation_corrected_pixels"]+=int((b["prediction"][mask]==cls).sum());gate_subset[n]["pixels"]+=area
            if component.quadrant=="Q3_deep_rival_direct_true":
                rival=int(component.predicted_class)
                for n,b in (("P1_RACC_A",p1),("P3_RACC_JOINT",p3)):
                    margin=float((b["cam"][cls][mask]-b["cam"][rival][mask]).mean());base_margin=float((p0["cam"][cls][mask]-p0["cam"][rival][mask]).mean())
                    deep_subset[n]["components"]+=1;deep_subset[n]["margin_gain_sum"]+=margin-base_margin;deep_subset[n]["corrected_pixels"]+=int((b["prediction"][mask]==cls).sum());deep_subset[n]["pixels"]+=area
        if index%100==0 or index==len(loader):print(f"RACC_EVAL_PROGRESS={index}/{len(loader)}",flush=True)
    metrics={n:scores_from_confusion(np.stack(h).sum(0)) for n,h in hist.items()}
    if abs(metrics["P0_HQMR"]["mIoU"]-HQMR_MIOU)>1e-12: raise AssertionError(metrics["P0_HQMR"]["mIoU"])
    delta={n:100*(m["mIoU"]-metrics["P0_HQMR"]["mIoU"]) for n,m in metrics.items()};alpha_summary={n:a.summary() for n,a in alpha.items()}
    presence=pd.DataFrame(presence_rows);gate={}
    for n in ("P2_RACC_G","P3_RACC_JOINT"):
        d=presence[presence.model==n]; y=d.truth.to_numpy(); pred=d.local.to_numpy(); rescued=d.rescued.astype(bool).to_numpy(); precision,recall,f1,_=precision_recall_fscore_support(y,pred,average="binary",zero_division=0)
        gate[n]={"local_precision":float(precision),"local_recall":float(recall),"local_f1":float(f1),"local_auroc":safe_auc(y,d.local_probability),"rescue_count":int(rescued.sum()),"rescue_precision":float(y[rescued].mean()) if rescued.any() else 0.,"false_rescues":int(((y==0)&rescued).sum()),"true_deep_exclusions":int(((y==1)&(d.deep==0)).sum()),"true_exclusion_recall":float(((y==1)&rescued).sum()/max(((y==1)&(d.deep==0)).sum(),1))}
    pix={n:{**v,"m1_recovery":v["corrected"]/max(v["baseline_wrong"],1),"tp_harm":v["harmed"]/max(v["baseline_correct"],1),"NCE":v["corrected"]/max(v["harmed"],1)} for n,v in pixel.items()}
    m1_result={n:{**v,"pixel_recovery":v["corrected"]/max(v["pixels"],1),"component_recovery":v["components_recovered"]/max(v["components"],1)} for n,v in m1_stats.items()}
    gate_result={n:{**v,"class_rescue_rate":v["class_rescued"]/max(v["components"],1),"segmentation_correction_rate":v["segmentation_corrected_pixels"]/max(v["pixels"],1)} for n,v in gate_subset.items()}
    deep_result={n:{**v,"mean_margin_gain":v["margin_gain_sum"]/max(v["components"],1),"pixel_repair":v["corrected_pixels"]/max(v["pixels"],1)} for n,v in deep_subset.items()}
    class_damage=any(100*(metrics["P3_RACC_JOINT"]["class_iou"][str(c)]-metrics["P0_HQMR"]["class_iou"][str(c)]) < -1 for c in range(4)); rescue_safe=gate["P3_RACC_JOINT"]["rescue_precision"]>=.8; collapse=alpha_summary["P3_RACC_JOINT"]["global_weight_collapse"]; nce=pix["P3_RACC_JOINT"]["NCE"]
    if delta["P3_RACC_JOINT"]<.2 or delta["P3_RACC_JOINT"]<0 or not rescue_safe or collapse:decision="NOGO"
    elif delta["P3_RACC_JOINT"]<.5:decision="WEAK"
    elif delta["P3_RACC_JOINT"]>=.8 and sum(100*(metrics["P3_RACC_JOINT"]["class_iou"][str(c)]-metrics["P0_HQMR"]["class_iou"][str(c)])>=-1 for c in range(4))>=3 and nce>1.5:decision="STRONG_GO"
    elif nce>1.5:decision="GO"
    else:decision="NOGO"
    df=pd.DataFrame(image_rows); selections={"arbitration_success":df.nlargest(20,"P1_RACC_A__delta_correct_pixels").image_id.tolist(),"arbitration_harm":df.nsmallest(20,"P1_RACC_A__delta_correct_pixels").image_id.tolist(),"gate_rescue_success":df.sort_values(["P2_true_rescues","P2_RACC_G__delta_correct_pixels"],ascending=False).head(20).image_id.tolist(),"false_rescue":df.sort_values(["P2_false_rescues","P2_RACC_G__delta_correct_pixels"],ascending=[False,True]).head(20).image_id.tolist(),"joint_recovery":df.nlargest(20,"P3_RACC_JOINT__delta_correct_pixels").image_id.tolist(),"joint_harm":df.nsmallest(20,"P3_RACC_JOINT__delta_correct_pixels").image_id.tolist()}
    write_json(experiment/"visualizations/selection.json",selections)
    for category,ids in selections.items():
        folder=experiment/"visualizations"/category;folder.mkdir(parents=True,exist_ok=True)
        for rank,image_id in enumerate(ids,1):
            pil=Image.open(valroot/"img"/f"{image_id}.png").convert("RGB"); image=np.asarray(pil);truth=np.asarray(Image.open(valroot/"mask"/f"{image_id}.png"))
            tensor=TF.normalize(TF.to_tensor(pil),mean=[.485,.456,.406],std=[.229,.224,.225])[None].cuda()
            detail=infer(models["P3_RACC_JOINT"],tensor,truth.shape,True,True,True); true_cls=int(np.bincount(truth[truth<4],minlength=4).argmax()); rival_cls=int(np.bincount(predictions["P0_HQMR"][image_id][truth==true_cls],minlength=4).argmax())
            weights=detail["weights"][:,true_cls]; amap=(detail["alpha"]*weights[:,None,None]).sum(0)/(weights.sum()+1e-12); amap=F.interpolate(torch.from_numpy(amap)[None,None],size=truth.shape,mode="bilinear",align_corners=False)[0,0].numpy()
            fig,axes=plt.subplots(1,11,figsize=(27.5,3))
            panels=[image,PALETTE[truth],PALETTE[predictions["P0_HQMR"][image_id]],PALETTE[predictions["P1_RACC_A"][image_id]],PALETTE[predictions["P2_RACC_G"][image_id]],PALETTE[predictions["P3_RACC_JOINT"][image_id]],detail["deep_probability"][None],detail["local_probability"][None],amap,detail["cam"][true_cls],detail["cam"][rival_cls]]
            titles=["Image","GT","HQMR","RACC-A","RACC-G","Joint","Deep gate","Local presence",f"Alpha C{true_cls}",f"True CAM C{true_cls}",f"Rival CAM C{rival_cls}"]
            for ax,value,title in zip(axes,panels,titles):
                if title in ("Deep gate","Local presence"): ax.imshow(value,aspect="auto",vmin=0,vmax=1,cmap="viridis");ax.set_xticks(range(4));ax.set_yticks([])
                elif value.ndim==2: ax.imshow(value,cmap="viridis");ax.axis("off")
                else: ax.imshow(value);ax.axis("off")
                ax.set_title(title)
            fig.suptitle(f"{category} | {image_id}");fig.tight_layout();fig.savefig(folder/f"{rank:03d}_{image_id}.png",dpi=120);plt.close(fig)
    base_params=sum(p.numel() for p in HQMRNet().parameters());counts=json.loads((experiment/"metrics/parameter_counts.json").read_text());a_flops=2*2*196*28*28*(4*8+8);g_flops=2*4*(5*8+8);hqmr_flops=214_910_692_676;complexity={**counts,"racc_a_extra_flops":a_flops,"racc_g_extra_flops":g_flops,"racc_total_extra_flops":a_flops+g_flops,"hqmr_flops":hqmr_flops,"extra_flops_percent":100*(a_flops+g_flops)/hqmr_flops,"hqmr_flops_method":"torch.profiler CUDA+CPU operator FLOPs, fixed 1x3x224x224 forward"}
    flags=[]
    if class_damage:flags.append("CLASS_DAMAGE")
    if collapse:flags.append("GLOBAL_WEIGHT_COLLAPSE")
    if not rescue_safe:flags.append("GATE_MODULE_UNSAFE")
    final_candidate=metrics["P3_RACC_JOINT"]["mIoU"]>=.667
    if final_candidate:flags.append("FINAL_CANDIDATE")
    decision_matrix={"joint_delta_ge_0.50":delta["P3_RACC_JOINT"]>=.5,"rescue_precision_ge_0.80":rescue_safe,"NCE_gt_1.5":nce>1.5,"no_global_weight_collapse":not collapse,"strong_delta_ge_0.80":delta["P3_RACC_JOINT"]>=.8}
    failure=("P1/P2/P3 均按固定 E5 评价。" + ("联合结果低于单模块，提示联合训练干扰。" if metrics["P3_RACC_JOINT"]["mIoU"]<max(metrics["P1_RACC_A"]["mIoU"],metrics["P2_RACC_G"]["mIoU"]) else "联合结果未出现负互作。"))
    next_step="停止于 Phase0；等待人工选择 Full25 边界。" if decision in ("GO","STRONG_GO") else "停止 Full25；按失败模块做一次最小审计，不扩展结构或扫参。"
    result={"decision":decision,"flags":flags,"final_candidate":final_candidate,"class_damage":class_damage,"baseline_reproduction":"PASS","metrics":metrics,"delta_pp":delta,"oracle_recovery_ratio":delta["P3_RACC_JOINT"]/ORACLE_GAIN,"identity":identity,"complexity":complexity,"alpha":alpha_summary,"gate":gate,"pixel_change":pix,"m1":{"components":len(m1),"by_model":m1_result},"gate_excluded":gate_result,"deep_dominance":deep_result,"visualizations":{k:len(v) for k,v in selections.items()},"failure_analysis":failure,"decision_matrix":decision_matrix,"next_step":next_step,"runtime":{"seconds":time.perf_counter()-started,"images":len(loader)},"evaluation_source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()}
    write_json(experiment/"metrics/final_result.json",result);write_csv(experiment/"metrics/segmentation_summary.csv",[{"model":n,"delta_pp":delta[n],**{k:v for k,v in metrics[n].items() if k not in ("confusion","class_iou","class_dice")}} for n in names]);write_csv(experiment/"metrics/per_class_iou.csv",[{"class":c,**{n:metrics[n]["class_iou"][str(c)] for n in names}} for c in range(4)]);write_csv(experiment/"metrics/gate_rescue_metrics.csv",[{"model":n,**v} for n,v in gate.items()]);write_csv(experiment/"metrics/alpha_statistics.csv",[{"model":n,**{k:v for k,v in s.items() if not isinstance(v,dict)},**{"distribution":json.dumps(s["distribution"]),"per_class_mean":json.dumps(s["per_class_mean"])}} for n,s in alpha_summary.items()]);write_json(experiment/"metrics/decision_matrix.json",decision_matrix)
    report=experiment/"RACC_v1_Phase0_Final_Report.md";report.write_text(build_report(result),encoding="utf-8");print(json.dumps({"decision":decision,"metrics":{n:100*m["mIoU"] for n,m in metrics.items()},"report":str(report)},indent=2));print(f"DECISION = {decision}")


if __name__=="__main__":main()
