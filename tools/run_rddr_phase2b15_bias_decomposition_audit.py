"""Frozen probability-cache audit: no network loading, training, or data split access."""
import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b1_common import A0,CKPT_SHA,sha256,write_json,compute_support
from tools.rddr_phase2b15_common import CACHE_SHA,PREFIX,probes,gt_context_diagnostic,make_groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",required=True)
    parser.add_argument("--previous-report",required=True)
    parser.add_argument("--checkpoint",required=True,help="SHA256 read only, never loaded")
    parser.add_argument("--output",required=True)
    parser.add_argument("--smoke-images",type=int,default=0)
    args=parser.parse_args(); out=Path(args.output); start=time.perf_counter()
    if out.exists(): raise FileExistsError(out)
    assert sha256(args.native)==CACHE_SHA
    assert sha256(args.checkpoint)==CKPT_SHA
    old=json.loads((Path(args.previous_report)/"rddr_phase2b1_summary.json").read_text())
    assert old["decision"]=="RDDR_PHASE2B1_NOGO" and old["images"]==3418
    with np.load(args.native,allow_pickle=False) as archive: data={k:archive[k] for k in archive.files}
    assert len(data["names"])==len(set(data["names"]))==3418
    groups,win,label,sp,dp=make_groups(data)
    assert groups["all"].sum()==old["foreground_targets"]
    assert win.sum()==old["adjudication_targets"]
    assert (win&label).sum()==old["deep_win_count"]
    assert (win&~label).sum()==old["shallow_win_count"]
    oldrows=__import__("csv").DictReader((Path(args.previous_report)/"rddr_phase2b1_per_image.csv").open())
    for i,row in enumerate(oldrows):
        assert str(data["names"][i])==row["image_id"]
        for g in ("all","Top20","Bottom80","boundary","interior","Q1","Q2","Q3","Q4","Q5"):
            assert groups[g][i].sum()==int(row[(g+"_targets") if g!="all" else "foreground_targets"])
    assert i==3417
    n=args.smoke_images or 3418
    assert 1<=n<=3418
    device=torch.device("cuda")
    torch.set_grad_enabled(False); torch.backends.cudnn.benchmark=False
    torch.cuda.reset_peak_memory_stats()
    def tensor(key,i):return torch.from_numpy(data[key][i:i+1].reshape(1,4,28,28)).to(device)
    parity={k:0. for k in ("T_SS","T_DS","Delta_old","ctx_S")}
    # Finish full old-score parity before any new probe/GT-neighborhood analysis.
    for i in range(n):
        r=compute_support(tensor("ps",i),tensor("pd",i))
        for key,new,oldkey in (("T_SS","ss","ss"),("T_DS","sd","sd"),("Delta_old","delta","delta"),("ctx_S","ctx","ctx")):
            err=float(np.max(np.abs(r[new].cpu().numpy()[0]-data[oldkey][i])))
            parity[key]=max(parity[key],err)
        assert max(parity.values())<=1e-7,dict(image=i,parity=parity)
    parity_seconds=time.perf_counter()-start
    print(json.dumps(dict(phase="old_exact_parity_pass",images=n,max_abs=parity)),flush=True)
    derived={}; compute_start=time.perf_counter()
    for i in range(n):
        result=probes(tensor("ps",i),tensor("pd",i))
        y=torch.from_numpy(data["truth"][i:i+1].reshape(1,28,28).astype(np.int64)).to(device)
        s=torch.from_numpy(sp[i:i+1].reshape(1,28,28)).to(device)
        d=torch.from_numpy(dp[i:i+1].reshape(1,28,28)).to(device)
        gt=gt_context_diagnostic(y,s,d)
        for key,value in {**result,**gt}.items():
            arr=value.cpu().numpy()[0]
            if key not in derived:derived[key]=np.empty((n,*arr.shape),np.float32)
            derived[key][i]=arr
        if (i+1)%500==0: print(json.dumps(dict(images=i+1,total=n)),flush=True)
    torch.cuda.synchronize()
    compute_seconds=time.perf_counter()-compute_start
    assert max(float(abs(derived["T_SS"]-data["ss"][:n]).max()),float(abs(derived["T_DS"]-data["sd"][:n]).max()),
               float(abs(derived["old"]-data["delta"][:n]).max()))<=1e-7
    assert sha256(args.native)==CACHE_SHA and sha256(args.checkpoint)==CKPT_SHA
    out.mkdir(parents=True)
    output=out/(PREFIX+"derived_observations.npz")
    np.savez_compressed(output,names=data["names"][:n],**derived)
    runtime=dict(images=n,smoke=bool(args.smoke_images),a0_commit=A0,
                 code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                 command=shlex.join([sys.executable,*sys.argv]),native=str(Path(args.native).resolve()),
                 native_sha256=CACHE_SHA,checkpoint=args.checkpoint,checkpoint_sha256=CKPT_SHA,
                 previous_report=args.previous_report,old_result=old["decision"],
                 all_frozen_counts_verified=True,parity_max_abs=parity,
                 parity_seconds=parity_seconds,probe_seconds=compute_seconds,total_seconds=time.perf_counter()-start,
                 gpu=torch.cuda.get_device_name(),python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,
                 input_precision="cached FP32 probabilities from frozen BF16 forward",probe_precision="FP32",batch=1,
                 peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                 derived_sha256=sha256(output),derived_bytes=output.stat().st_size,
                 network_forward=False,checkpoint_loaded=False,checkpoint_written=False,optimizer_created=False,
                 gradients=False,test_access=False,luad_access=False,search=False,inputs_unchanged=True)
    write_json(out/(PREFIX+"runtime.json"),runtime)
    print(json.dumps(runtime,indent=2),flush=True)


if __name__=="__main__":main()
