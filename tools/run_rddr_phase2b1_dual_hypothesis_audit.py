"""Extract frozen native-grid probabilities/supports, validation only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b1_common import (A0,CKPT_SHA,Q_EDGES,HFRM_GROUPS,sha256,write_json,
    populations,project,boundary_mask,compute_support,phase0_js)


def validate_root(path):
    path = Path(path).resolve()
    if path.name!="val" or path.parent.name!="BCSS-WSSS":
        raise ValueError("Only BCSS-WSSS/val is allowed")
    assert (path/"img").is_dir() and (path/"mask").is_dir()
    return path


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def preflight(args,dataset):
    root=Path(args.population_cache)
    manifest_path=root/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"]=="PASS" and manifest["images"]==3418
    assert manifest["checkpoint_sha256"]==CKPT_SHA
    assert manifest["source_commit"]=="586f402a30f446c409c625b55953e329cc041dcc"
    hashes={r["image_id"]:r["sha256"] for r in manifest["files"]}
    with (Path(args.phase0_results)/"rddr_phase0_per_image.csv").open(newline="") as f:
        expected={r["image_id"]:r for r in csv.DictReader(f)}
    ids=[Path(p).stem for p in dataset.object]
    assert len(ids)==len(set(ids))==3418
    assert set(ids)==set(hashes)==set(expected)=={p.stem for p in (args.val_root/"mask").glob("*.png")}
    full,small,files,qvalues={},{},[],[]
    for name in ids:
        cache_path=root/(name+".npz")
        assert sha256(cache_path)==hashes[name],name
        mask_path=args.val_root/"mask"/(name+".png")
        y=np.asarray(Image.open(mask_path),dtype=np.uint8)
        assert y.shape==(224,224) and set(np.unique(y)) <= {0,1,2,3,4,255}
        with np.load(cache_path) as cache:
            masks=populations(cache,y)
            for g,m in masks.items():
                full[g]=full.get(g,0)+int(m.sum())
                small[g]=small.get(g,0)+int(project(m).sum())
                if g in HFRM_GROUPS:
                    assert int(m.sum())==int(expected[name][f"ch_{g}_count"])
            assert int(masks["Top20"].sum())==int(expected[name]["S_JS_top20_flagged"])
            qvalues.append(cache["q_feature"][project(masks["all"]).astype(bool)])
        files.append(dict(image_id=name,cache_sha256=hashes[name],mask_sha256=sha256(mask_path)))
    for k,v in manifest["counts"].items(): assert full[k]==v
    edges=np.quantile(np.concatenate(qvalues),[.2,.4,.6,.8],method="higher")
    assert np.array_equal(edges,Q_EDGES),(edges,Q_EDGES)
    return dict(status="PASS",source_manifest=manifest_path,source_manifest_sha256=sha256(manifest_path),
                checkpoint_sha256=CKPT_SHA,source_commit=manifest["source_commit"],images=3418,
                all_cache_hashes_verified=True,all_original_per_image_counts_exact=True,
                historical_original_pixel_hash_available=False,full_resolution_counts=full,
                projected_counts=small,q_quintile_edges=edges,files=files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--val-root",required=True)
    p.add_argument("--population-cache",required=True)
    p.add_argument("--phase0-results",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--smoke-images",type=int,default=0,choices=(0,2))
    args=p.parse_args()
    tick=time.perf_counter()
    args.val_root=validate_root(args.val_root)
    out=Path(args.output)
    if out.exists(): raise FileExistsError(out)
    assert sha256(args.checkpoint)==CKPT_SHA
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    changed=subprocess.check_output(["git","diff","--name-only",A0],cwd=ROOT,text=True).splitlines()
    assert all(n.startswith(("tools/","tests/","docs/","audit/")) for n in changed),changed
    assert not subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip()
    torch.backends.cudnn.benchmark=False
    assert torch.backends.cuda.matmul.fp32_precision=="none"
    assert torch.backends.cudnn.conv.fp32_precision=="tf32"
    torch.set_num_threads(4)
    from network.resnet38_cls import Net
    from tool.GenDataset import Stage1_InferDataset
    dataset=Stage1_InferDataset(str(args.val_root/"img"),img_size=224)
    dataset.object=sorted(dataset.object)
    manifest=preflight(args,dataset)
    print("PREFLIGHT_PASS: 3418 cache hashes/counts; frozen q quintiles exact",flush=True)
    selected=list(range(3418)) if args.smoke_images==0 else [0,3417]
    n=len(selected)
    loader=DataLoader(Subset(dataset,selected),batch_size=1,shuffle=False,num_workers=4,pin_memory=True)
    model=Net(4).cuda()
    load=model.load_state_dict(torch.load(args.checkpoint,map_location="cpu",weights_only=False),strict=True)
    model.eval()  # A0 train() override returns None: do not chain.
    model.requires_grad_(False)
    assert not any(m.training for m in model.modules())
    before=state_digest(model)
    capture={}
    def hook(module,inputs,result): capture.update(raw=inputs[0],deep=inputs[1])
    handle=model.hfrm_28_1.register_forward_hook(hook)
    prob={k:np.empty((n,4,784),np.float32) for k in ("ps","pd","ctx","anchor","fixed_average")}
    scalar={k:np.empty((n,784),np.float32) for k in ("ss","sd","delta","wd","q_feature")}
    labels={k:np.empty((n,784),np.uint8) for k in ("truth","hfrm","top20","boundary")}
    names=[]
    max_q_diff=max_rounding=max_prob_sum_error=0.
    gpu_seconds=0.
    out.mkdir(parents=True)
    write_json(out/"rddr_phase2b1_population_manifest.json",manifest)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i,(image_ids,image) in enumerate(loader):
            name=image_ids[0]; names.append(name)
            start=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16):
                outputs=model(image.cuda(non_blocking=True))
                assert capture["raw"].shape==(1,512,28,28) and capture["deep"].shape==(1,4096,28,28)
                ls=model.ic1(capture["raw"])
                ld=outputs[8]
            ps,pd=ls.float().softmax(1),ld.float().softmax(1)
            result=compute_support(ps,pd)
            q=phase0_js(ps,pd)[0].cpu().numpy()/math.log(2)
            q_tensor=(phase0_js(ps,pd)/math.log(2)).clamp(0,1)[0].cpu().numpy()
            max_rounding=max(max_rounding,float(np.abs(q-q_tensor).max()))
            for value in (ps.flatten(2),pd.flatten(2),result["anchor"],result["ctx"]):
                error=float((value.sum(1)-1).abs().max())
                max_prob_sum_error=max(max_prob_sum_error,error)
                assert torch.isfinite(value).all() and error<2e-6
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            torch.cuda.synchronize()
            gpu_seconds+=time.perf_counter()-start
            # Support/anchor are finalized before GT enters the per-image audit.
            y=np.asarray(Image.open(args.val_root/"mask"/(name+".png")),dtype=np.uint8)
            with np.load(Path(args.population_cache)/(name+".npz")) as cache:
                masks=populations(cache,y)
                native_q=cache["q_feature"].copy()
            diff=float(np.abs(q-native_q).max())
            max_q_diff=max(max_q_diff,diff)
            assert diff==0.,(name,diff)
            group_code=np.full((28,28),255,np.uint8)
            for j,g in enumerate(HFRM_GROUPS): group_code[project(masks[g]).astype(bool)]=j
            labels["truth"][i]=project(y).astype(np.uint8).ravel()
            labels["hfrm"][i]=group_code.ravel()
            labels["top20"][i]=project(masks["Top20"]).astype(np.uint8).ravel()
            labels["boundary"][i]=project(boundary_mask(y)).astype(np.uint8).ravel()
            prob["ps"][i]=ps[0].flatten(1).cpu().numpy()
            prob["pd"][i]=pd[0].flatten(1).cpu().numpy()
            prob["ctx"][i]=result["ctx"][0].cpu().numpy()
            prob["anchor"][i]=result["anchor"][0].cpu().numpy()
            prob["fixed_average"][i]=(.5*ps[0]+.5*pd[0]).flatten(1).cpu().numpy()
            scalar["ss"][i]=result["ss"][0].cpu().numpy()
            scalar["sd"][i]=result["sd"][0].cpu().numpy()
            scalar["delta"][i]=result["delta"][0].cpu().numpy()
            scalar["wd"][i]=result["wd"][0].cpu().numpy()
            scalar["q_feature"][i]=native_q.ravel()
            if (i+1)%200==0 or i+1==n: print(f"EXTRACT {i+1}/{n} elapsed={time.perf_counter()-tick:.1f}s q_exact=True",flush=True)
    handle.remove()
    assert state_digest(model)==before
    assert sha256(args.checkpoint)==CKPT_SHA
    cache_path=out/"rddr_phase2b1_native_observations.npz"
    np.savez_compressed(cache_path,names=np.array(names),**prob,**scalar,**labels)
    runtime=dict(commit=commit,a0_commit=A0,checkpoint=args.checkpoint,checkpoint_sha256=CKPT_SHA,
                 command=shlex.join([sys.executable,*sys.argv]),working_directory=ROOT,images=n,smoke=bool(args.smoke_images),
                 checkpoint_missing_keys=load.missing_keys,checkpoint_unexpected_keys=load.unexpected_keys,
                 model_state_digest_before_after=before,unchanged_model_state=True,frozen_q_max_abs_difference=max_q_diff,
                 torch_numpy_q_rounding_max_difference=max_rounding,max_probability_sum_error=max_prob_sum_error,
                 forward_support_seconds=gpu_seconds,total_seconds=time.perf_counter()-tick,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                 gpu=torch.cuda.get_device_name(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                 benchmark=torch.backends.cudnn.benchmark,matmul_precision=torch.backends.cuda.matmul.fp32_precision,
                 conv_precision=torch.backends.cudnn.conv.fp32_precision,checkpoint_written=False,optimizer_created=False,
                 test_access=False,luad_access=False,requires_grad=False,native_observations_sha256=sha256(cache_path),
                 native_observations_bytes=cache_path.stat().st_size)
    write_json(out/"rddr_phase2b1_runtime.json",runtime)
    print("EXTRACTION_COMPLETE",flush=True)


if __name__=="__main__": main()
