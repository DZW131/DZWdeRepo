"""HQRF Phase-0 query-mask health gate: BCSS train-only, Seed42, max 3 epochs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import train_sshr as official
from network.hqrf_net import HQRFNet
from tool.GenDataset import Stage1_TrainDataset
from tool.torchutils import PolyOptimizer
from tools.hqrf_diagnostics import apply_epoch2_screen, apply_final_gate, batch_health, summarize
from tools.hqrf_phase0_io import check_train_path, install_train_access_guard, protected_sources, sha256, write_csv, write_json
from tools.hqrf_report import render_report


ROOT = Path(__file__).resolve().parent
FULL25_STEPS = 29_275
PHASE0_STEPS = 3_513
PCRE_COMMIT = "d89029439c2676ed98035a77bc94157afd0c777c"
CONFIG = {
    "experiment": "HQRF-Net Innovation 1 v2 Phase-0 Query-Mask Health Gate",
    "dataset": "BCSS training only", "seed": 42, "gpu": "RTX4090D", "batch": 20,
    "epochs_max": 3, "steps_per_epoch": 1171, "phase0_steps_max": PHASE0_STEPS,
    "full25_schedule_denominator": FULL25_STEPS, "amp": "bf16", "image_size": 224,
    "query_grid": [14,14], "query_count": 196, "patch_size": 16, "dimension": 256,
    "decoder_layers": 2, "layers_per_stage": 1, "heads": 8, "ffn_hidden": 1024,
    "dropout": .1, "activation": "GELU", "memory_detach": "detach CNN FD/F5 before trainable projection",
    "chpf": {"F5_channels":256,"F4_channels":128,"kernel":15,"padding":7,"bias":False,"init":1/225,"gamma_init":0},
    "mask_mlp": [256,256,256], "loss_weights": {"deep":.50,"PCA":.25,"mask":.25},
    "pseudo": {"positive_floor":.60,"top_ratio":.15,"class_margin":.10,"background_ceiling":.10,"positive_dilation":1},
    "locality": {"radii":[1,5],"query_to_mask_scale":4,"schedule":"1 then cosine 1->5 then 5","denominator":FULL25_STEPS},
    "mask_min_labeled_pixels":4, "pmec":{"tau_bin":.70,"tau_low":.40,"tau_high":.50,"T":5,"ror_denominator":"reference mask area"},
    "optimizer":{"base_lr":.01,"weight_decay":.0005,"multipliers":[1,2,10,20],"poly_max_step":PHASE0_STEPS},
    "monitor_snapshots":[250,500,1000,1171,2342,3513], "validation_access":False,
    "absent_dominance_definition":"more than 50% of top20 max-joint-confidence mass assigned to absent classes; healthy image fraction <5%",
    "pc_re_source_commit":PCRE_COMMIT,
    "pc_re_source_sha256": {
        "model/query_decoder.py":"d612c448f72230ffc97692de7313030119af5548fe1e02150b143b79f299c5d4",
        "model/model_wsddn.py":"39bee5fec4e9144a7909fbafdb6c73a2d1a3dfe0248a08ab95bacb0ec0087f00",
        "model/losses.py":"429b1b00b79746debea4572bde665fc6dc83ebd133c7a24e70b21819ec08eb6a",
        "model/wsddn_layer.py":"e344e92fd50555e3f1783c9e88add875b8575a9a6c2d6136ce0c1a584027cc38",
    },
}


class Tee:
    def __init__(self, *streams): self.streams=streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
        return len(value)
    def flush(self):
        for stream in self.streams: stream.flush()


class MonitorDataset(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        path,label=self.rows[index]
        image=Image.open(path).convert("RGB")
        if image.size != (224,224): image=TF.resize(image,[224,224])
        value=TF.normalize(TF.to_tensor(image),[.485,.456,.406],[.229,.224,.225])
        return Path(path).stem,value,label


def select_monitor_cohort(objects):
    rows=sorted(objects,key=lambda value: str(value[0]))
    single=[row for row in rows if int(row[1].sum())==1]
    multi=[row for row in rows if int(row[1].sum())>=2]
    generator=np.random.default_rng(42)
    def take(pool,count):
        if not pool or count<=0:return []
        chosen=generator.choice(len(pool),size=min(count,len(pool)),replace=False)
        return [pool[int(index)] for index in chosen]
    chosen=take(single,8)+take(multi,16)
    used={str(row[0]) for row in chosen}
    remainder=[row for row in rows if str(row[0]) not in used]
    chosen+=take(remainder,32-len(chosen))
    if len(chosen)!=32: raise RuntimeError("Unable to freeze a 32-image monitoring cohort")
    return chosen


def gradient_norms(model):
    groups={name:[] for name in ("backbone","deep","query","pixel","mask","pca")}
    for name,parameter in model.named_parameters():
        if parameter.grad is None: continue
        group="backbone" if name.startswith("backbone.") else "deep" if name.startswith("deep_head.") else "query" if any(name.startswith(prefix) for prefix in ("patch_queries.","semantic_projection.","decoder1.","context_projection.","f5_chpf.","decoder2.")) else "pixel" if name.startswith("pixel_decoder.") else "mask" if name.startswith("mask_embedding.") else "pca"
        groups[group].append(parameter.grad.detach().float().square().sum())
    return {key:float(torch.stack(values).sum().sqrt()) if values else 0. for key,values in groups.items()}


def save_visualizations(directory, snapshot, names, images, labels, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target=Path(directory)/"visualizations"/snapshot; target.mkdir(parents=True,exist_ok=True)
    mean=images.new_tensor([.485,.456,.406])[None,:,None,None]; std=images.new_tensor([.229,.224,.225])[None,:,None,None]
    rgb=(images*std+mean).clamp(0,1).permute(0,2,3,1).float().cpu().numpy()
    joint=output["confidence"]["joint"].float(); top=joint.max(-1).values.argsort(dim=1,descending=True,stable=True)[:,:5]
    probability=output["mask_logits"].float().sigmoid(); assigned=output["confidence"]["p_class"].argmax(-1)
    for image in range(min(8,len(names))):
        cls=int(torch.where(labels[image].bool())[0][0]); tri=output["targets"][image,cls].float().cpu().numpy()
        cam=output["normalized_cam"][image].max(0).values.float().cpu().numpy()
        panels=[(rgb[image],"input"),(cam,"deep CAM"),(tri,"tri-state")]
        panels += [(probability[image,index].float().cpu().numpy(),f"top query {rank+1}") for rank,index in enumerate(top[image].tolist())]
        panels += [(assigned[image].reshape(14,14).float().cpu().numpy(),"PCA assignment"),(output["pmec_region"][image].amax(0).float().cpu().numpy(),"PMEC regions")]
        figure,axes=plt.subplots(2,5,figsize=(15,6))
        for axis,(value,title) in zip(axes.flat,panels):
            axis.imshow(value,cmap=None if value.ndim==3 else "viridis",vmin=None if value.ndim==3 else (-1 if title=="tri-state" else 0),vmax=None if value.ndim==3 else (1 if title=="tri-state" else 1)); axis.set_title(title); axis.axis("off")
        figure.suptitle(f"{names[image]} | present={torch.where(labels[image].bool())[0].tolist()}")
        figure.tight_layout(); figure.savefig(target/f"{names[image]}.png",dpi=130); plt.close(figure)


@torch.no_grad()
def monitor(model, loader, step, snapshot, histories, output_dir, visualize=False):
    was_training=model.training; model.eval(); batches=[]; image_rows=[]; pmec_rows=[]; first=None
    for names,images,labels in loader:
        images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=step,run_pmec=True)
        values,rows=batch_health(result,labels); batches.append(values); image_rows.extend(rows); pmec_rows.extend(result["pmec_rows"])
        if first is None: first=(names,images.detach(),labels.detach(),result)
    summary=summarize(snapshot,batches,image_rows,pmec_rows,model)
    file_map={"mask_area":"hqrf_phase0_mask_area.csv","mask_logits":"hqrf_phase0_mask_logits.csv","query_diversity":"hqrf_phase0_query_diversity.csv","semantic_selectivity":"hqrf_phase0_semantic_selectivity.csv","pca_health":"hqrf_phase0_pca_health.csv","pmec_health":"hqrf_phase0_pmec_health.csv","chpf":"hqrf_phase0_chpf.csv","catastrophic":"hqrf_phase0_catastrophic_screen.csv"}
    for key,filename in file_map.items(): histories[key].append(summary[key]); write_csv(Path(output_dir)/filename,histories[key])
    if visualize and first is not None: save_visualizations(output_dir,snapshot,*first)
    if was_training:model.train()
    print("HQRF_MONITOR "+json.dumps({"snapshot":snapshot,"mask_area":summary["mask_area"],"diversity":summary["query_diversity"],"selectivity":summary["semantic_selectivity"],"pmec":summary["pmec_health"]}),flush=True)
    return summary


def finalize_checkpoint(model, output, epoch):
    path=Path(output)/f"hqrf_phase0_epoch{epoch}.pth"; torch.save(model.state_dict(),path)
    digest=sha256(path); (Path(output)/f"hqrf_phase0_epoch{epoch}_sha256.txt").write_text(digest+"\n",encoding="utf-8")
    return path,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--trainroot",required=True); parser.add_argument("--weights",required=True); parser.add_argument("--output",required=True); parser.add_argument("--smoke-steps",type=int,choices=(0,2),default=0); args=parser.parse_args()
    check_train_path(args.trainroot); output=Path(args.output).resolve()
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("Native CUDA BF16 required")
    if "4090" not in torch.cuda.get_device_name(0): raise RuntimeError("Registered RTX4090D required")
    protected=protected_sources(ROOT); accesses=install_train_access_guard(); official.set_seed(42)
    output.mkdir(parents=True); log_handle=(output/"hqrf_phase0_train.log").open("w",encoding="utf-8",buffering=1); sys.stdout=Tee(sys.__stdout__,log_handle); sys.stderr=Tee(sys.__stderr__,log_handle)
    source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); config={**CONFIG,"source_commit":source_commit,"smoke_steps":args.smoke_steps,"trainroot":str(Path(args.trainroot).resolve())}
    write_json(output/"hqrf_phase0_config.json",config); (output/"hqrf_phase0_source_commit.txt").write_text(source_commit+"\n",encoding="utf-8")
    model=HQRFNet().cuda(); init=model.backbone.load_official_initialization(args.weights); write_json(output/"hqrf_phase0_init_identity.json",init)
    dataset=Stage1_TrainDataset(args.trainroot,transform=transforms.Compose([transforms.ToTensor()]),dataset="bcss",img_size=224)
    if len(dataset)!=23422: raise RuntimeError("Expected 23,422 BCSS training images")
    cohort=select_monitor_cohort(dataset.object); write_json(output/"hqrf_phase0_monitor_cohort.json",{"seed":42,"images":[{"path":str(Path(path).resolve()),"label":[int(v) for v in label.tolist()]} for path,label in cohort]})
    monitor_loader=DataLoader(MonitorDataset(cohort),batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
    generator=torch.Generator().manual_seed(42); loader=DataLoader(dataset,batch_size=20,shuffle=True,num_workers=8,pin_memory=True,drop_last=True,worker_init_fn=official.seed_worker,generator=generator)
    if len(loader)!=1171: raise RuntimeError("Expected 1,171 steps per epoch")
    groups=model.get_parameter_groups(); optimizer=PolyOptimizer([{"params":group,"lr":.01*multiplier,"weight_decay":decay} for group,multiplier,decay in zip(groups,(1,2,10,20),(.0005,0,.0005,0))],lr=.01,weight_decay=.0005,max_step=PHASE0_STEPS)
    if model.f5_chpf.gamma.detach().item()!=0 or model.pixel_decoder.f4_chpf.gamma.detach().item()!=0: raise RuntimeError("CHPF gamma must start at exactly zero")
    hashes={str(path.relative_to(ROOT)):sha256(path) for path in list((ROOT/"network").glob("hqrf*.py"))+list((ROOT/"tools").glob("hqrf*.py"))}; hashes["train_hqrf_phase0.py"]=sha256(ROOT/"train_hqrf_phase0.py")
    write_json(output/"hqrf_phase0_provenance.json",{"source_commit":source_commit,"implementation_sha256":hashes,"protected_source_sha256":protected,"environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0)},"pc_re_commit":PCRE_COMMIT})
    histories={key:[] for key in ("mask_area","mask_logits","query_diversity","semantic_selectivity","pca_health","pmec_health","chpf","catastrophic")}; losses=[]; completed=0; final_summary=None; gate=None; last_grad={}; started=time.perf_counter(); torch.cuda.reset_peak_memory_stats(); all_finite=True
    print("HQRF_PROTOCOL "+json.dumps(config),flush=True)
    try:
        stop=False
        for epoch in range(1,4):
            model.train(); epoch_values=[]
            for _,images,labels in loader:
                images=images.cuda(non_blocking=True); labels=labels.cuda(non_blocking=True)
                with torch.autocast("cuda",dtype=torch.bfloat16): result=model(images,labels,step=optimizer.global_step)
                values=result["losses"]
                if not all(torch.isfinite(value).all() for value in values.values()): raise FloatingPointError("Non-finite loss")
                optimizer.zero_grad(); values["loss"].backward(); last_grad=gradient_norms(model)
                if not all(math.isfinite(value) for value in last_grad.values()): raise FloatingPointError("Non-finite gradient")
                optimizer.step(); epoch_values.append({key:float(value.detach()) for key,value in values.items()})
                if optimizer.global_step%100==0 or args.smoke_steps:
                    row={"epoch":epoch,"step":optimizer.global_step,**epoch_values[-1],"lr":optimizer.param_groups[0]["lr"]}; losses.append(row); write_csv(output/"hqrf_phase0_losses.csv",losses)
                    print("HQRF_STEP "+json.dumps({**row,"gradients":last_grad,"peak_memory":torch.cuda.max_memory_allocated(),"elapsed_seconds":time.perf_counter()-started}),flush=True)
                if not args.smoke_steps and optimizer.global_step in (250,500,1000):
                    final_summary=monitor(model,monitor_loader,optimizer.global_step,f"step{optimizer.global_step:04d}",histories,output,visualize=optimizer.global_step in (500,1000))
                if args.smoke_steps and optimizer.global_step>=2: stop=True; break
            completed=epoch
            if args.smoke_steps: break
            aggregate={key:float(np.mean([row[key] for row in epoch_values])) for key in epoch_values[0]}; losses.append({"epoch":epoch,"step":optimizer.global_step,**aggregate,"lr":optimizer.param_groups[0]["lr"]}); write_csv(output/"hqrf_phase0_losses.csv",losses)
            final_summary=monitor(model,monitor_loader,optimizer.global_step,f"epoch{epoch}",histories,output,visualize=epoch>=2)
            print("HQRF_EPOCH "+json.dumps({"epoch":epoch,"step":optimizer.global_step,**aggregate,"gradients":last_grad}),flush=True)
            if epoch==2:
                gate=apply_epoch2_screen(final_summary); write_json(output/"hqrf_phase0_epoch2_screen.json",gate)
                if gate["decision"]=="HQRF_QUERY_REGION_NOGO": stop=True; break
        elapsed=time.perf_counter()-started
        if args.smoke_steps:
            runtime={"smoke":True,"steps":optimizer.global_step,"epochs":completed,"all_finite":True,"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"gradients":last_grad,"validation_accessed":False,"training_paths_opened":len(accesses)}; write_json(output/"hqrf_phase0_runtime.json",runtime); print("HQRF_SMOKE_PASS "+json.dumps(runtime),flush=True); return
        if stop and completed==2:
            decision="HQRF_QUERY_REGION_NOGO"
        else:
            gate=apply_final_gate(final_summary); decision=gate["decision"]
        checkpoint,digest=finalize_checkpoint(model,output,completed)
        runtime={"smoke":False,"steps":optimizer.global_step,"epochs":completed,"all_finite":all_finite,"train_seconds":elapsed,"peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),"peak_cuda_memory_gib":torch.cuda.max_memory_allocated()/1024**3,"gradients":last_grad,"validation_accessed":False,"test_accessed":False,"luad_accessed":False,"training_paths_opened":len(accesses),"decision":decision,"checkpoint":str(checkpoint),"checkpoint_sha256":digest}; write_json(output/"hqrf_phase0_runtime.json",runtime)
        result={**runtime,"source_commit":source_commit,"final_summary":final_summary,"gate":gate}; write_json(output/"hqrf_phase0_gate_result.json",result); report=render_report(output,result); print("HQRF_FINAL "+json.dumps({"decision":decision,"report":str(report),"gate":gate}),flush=True); print("DECISION = "+decision,flush=True)
    except Exception as error:
        all_finite=False; failure={"decision":"HQRF_ENGINEERING_BLOCKED","error":str(error),"step":optimizer.global_step,"epoch":completed,"validation_accessed":False}; write_json(output/"hqrf_phase0_engineering_failure.json",failure)
        render_report(output,{**failure,"epochs":completed,"steps":optimizer.global_step,"all_finite":False,"source_commit":source_commit,"training_paths_opened":len(accesses),"final_summary":final_summary or {},"gate":{}})
        raise


if __name__=="__main__": main()
