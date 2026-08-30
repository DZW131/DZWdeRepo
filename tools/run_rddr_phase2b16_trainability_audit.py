"""Validation-only, zero-update audit. Never constructs an optimizer or saves weights."""
import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b16_common import *
from network.resnet38_cls import Net,Net_CAM
from tool.GenDataset import Stage1_InferDataset


def tensor_hash(values):
    h=hashlib.sha256()
    for name,value in values:
        v=value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_hash(model):return tensor_hash(model.state_dict().items())


def forbid(*args,**kwargs):raise RuntimeError('optimizer/checkpoint-write forbidden in zero-update audit')


def official_identity(model,dataset,indices):
    """Execute unchanged infer up to raw decoded predictions; no metric overwrite."""
    from tool import infer_fun
    selected=[dataset.object[int(i)] for i in indices]
    def factory(*args,**kwargs):
        ds=Stage1_InferDataset(*args,**kwargs)
        ds.object=selected
        return ds
    def capture(gt,pred,n_class):
        assert len(gt)==len(pred)==len(indices)
        h=hashlib.sha256()
        for p in pred:h.update(np.asarray(p).tobytes())
        return dict(images=len(pred),pixels=sum(p.size for p in pred),prediction_sha256=h.hexdigest())
    args=SimpleNamespace(dataset='bcss',img_size=224,amp_dtype='bf16',num_workers=0)
    with patch.object(infer_fun,'Stage1_InferDataset',factory),patch.object(infer_fun.iouutils,'scores',capture):
        r=infer_fun.infer(model,str(Path(dataset.data_path).parent),4,args)
    assert r is not None
    return r


def param_stats(parameters):
    result=[]
    for name,p in parameters:
        assert p.grad is not None,name
        g=p.grad.detach().float(); finite=bool(torch.isfinite(g).all())
        result.append(dict(parameter=name,elements=p.numel(),sumsq=float(g.double().square().sum()),
                           rms=float(g.double().square().mean().sqrt()),max_abs=float(g.abs().max()),
                           finite=finite,nonzero_fraction=float((g!=0).float().mean())))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native',required=True);ap.add_argument('--derived',required=True)
    ap.add_argument('--checkpoint',required=True);ap.add_argument('--val-images',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args();start=time.perf_counter();out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    assert Path(args.val_images).as_posix().endswith('/BCSS-WSSS/val/img')
    assert sha256(args.native)==NATIVE_SHA and sha256(args.derived)==DERIVED_SHA
    assert sha256(args.checkpoint)==CKPT_SHA
    # Runtime guard is independent of the Dataset's unused train/test class definitions.
    accessed=set()
    def audit_open(event,params):
        if event!='open' or not isinstance(params[0],(str,bytes)):return
        p=str(params[0]).replace('\\','/').lower()
        if '/reseg-data/' in p:
            assert '/bcss-wsss/val/' in p,('forbidden dataset access',p)
            accessed.add(p)
    sys.addaudithook(audit_open)
    with np.load(args.native,allow_pickle=False) as z:data={k:z[k] for k in ('names','ps','pd','q_feature','truth','top20','boundary','fixed_average')}
    with np.load(args.derived,allow_pickle=False) as z:derived={k:z[k] for k in ('names','T_SS','T_SD','T_DS','T_DD','sym','wD_sym','anchor_sym')}
    n=len(data['names']);assert n==3418 and np.array_equal(data['names'],derived['names'])
    ds=Stage1_InferDataset(args.val_images,img_size=224);ds.object=sorted(ds.object)
    assert [Path(p).stem for p in ds.object]==data['names'].tolist()
    fixed=np.linspace(0,n-1,32,dtype=int)
    random=np.random.default_rng(42).choice(np.setdiff1d(np.arange(n),fixed),128,replace=False)
    stability=np.r_[fixed,random];batch20=fixed[:20]
    out.mkdir(parents=True)
    write_json(out/(PREFIX+'selection.json'),dict(deterministic32=fixed,random128=random,batch20=batch20,
               deterministic_names=data['names'][fixed],random_names=data['names'][random],
               batch20_names=data['names'][batch20],contract_sha256=sha256(ROOT/'docs/rddr_phase2b16_contract.md')))
    torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False
    assert torch.backends.cuda.matmul.fp32_precision=='none'
    assert torch.backends.cudnn.conv.fp32_precision=='tf32'
    torch.cuda.reset_peak_memory_stats()
    def t(arr):return torch.from_numpy(np.ascontiguousarray(arr)).cuda()
    parity={k:0. for k in derived if k!='names'};parity['q']=0.
    for i in range(n):
        r=detached_teacher(t(data['ps'][i:i+1].reshape(1,4,28,28)),t(data['pd'][i:i+1].reshape(1,4,28,28)))
        for k in parity:
            old=data['q_feature'][i:i+1] if k=='q' else derived[k][i:i+1]
            parity[k]=max(parity[k],float(np.abs(r[k].cpu().numpy()-old).max()))
        assert max(parity.values())<=1e-7,(i,parity)
        if (i+1)%500==0:print(json.dumps(dict(phase='teacher_parity',images=i+1)),flush=True)
    parity_seconds=time.perf_counter()-start
    print(json.dumps(dict(phase='teacher_parity_pass',max_abs=parity)),flush=True)
    model=Net_CAM(4).cuda();load=model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=False),strict=True)
    model.eval();model.requires_grad_(False)
    before=state_hash(model)
    holder={}
    def hook(_m,inputs,output):holder.update(raw=inputs[0],deep=inputs[1],rect=output)
    handle=model.hfrm_28_1.register_forward_hook(hook)
    selected_params=[(k,dict(model.named_parameters())[k]) for k in PARAMS]
    with patch.object(torch.optim.Optimizer,'__init__',forbid),patch.object(torch,'save',forbid):
        identity_start=time.perf_counter()
        official_before=official_identity(model,ds,stability)
        identity_seconds=time.perf_counter()-identity_start
        for _,p in selected_params:p.requires_grad_(True)
        assert tuple(k for k,p in model.named_parameters() if p.requires_grad)==tuple(k for k,p in model.named_parameters() if k in PARAMS)
        assert not any(m.training for m in model.modules())
        logits=np.empty((n,4,784),np.float32)
        grads=np.empty((n,3,4,784),np.float32)
        losses=np.empty((n,3),np.float32)
        feature_norm=np.empty((n,784),np.float32);feature_max=np.empty_like(feature_norm)
        param_rows=[];base_hash={};source_parity={'ps':0.,'pd':0.};all_finite=True
        loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True)
        gradient_start=time.perf_counter()
        for i,(names,x) in enumerate(loader):
            assert names[0]==data['names'][i]
            x=x.cuda(non_blocking=True);model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=Net.forward(model,x)
                with torch.no_grad():ps=model.ic1(holder['raw']).float().softmax(1)
            pd=result[8].detach().float().softmax(1)
            for k,value in (('ps',ps),('pd',pd)):
                err=float(np.abs(value.flatten(2).cpu().numpy()-data[k][i:i+1]).max())
                source_parity[k]=max(source_parity[k],err)
            assert max(source_parity.values())<=1e-7,(i,source_parity)
            L=result[6].float();logits[i]=L.detach().flatten(2).cpu().numpy()[0]
            teacher=t(derived['anchor_sym'][i:i+1].reshape(1,4,28,28))
            fixed_teacher=t(data['fixed_average'][i:i+1].reshape(1,4,28,28))
            q=t(data['q_feature'][i:i+1].reshape(1,28,28))
            holder['rect'].retain_grad()
            for j,mode in enumerate(('U','FA','CCA')):
                loss,_=loss_probe(L,fixed_teacher if mode=='FA' else teacher,q,mode)
                g,=torch.autograd.grad(loss,L,retain_graph=True)
                assert g.shape==L.shape
                grads[i,j]=g.detach().flatten(2).cpu().numpy()[0];losses[i,j]=loss.item()
                assert torch.isfinite(loss) and torch.isfinite(g).all() and torch.isfinite(L).all()
            loss.backward()
            fg=holder['rect'].grad.detach().float()
            assert torch.isfinite(fg).all() and fg.abs().max()>0
            feature_norm[i]=fg.square().sum(1).sqrt().flatten().cpu().numpy()
            feature_max[i]=fg.abs().max(1).values.flatten().cpu().numpy()
            rows=param_stats(selected_params)
            assert all(r['finite'] for r in rows)
            param_rows.extend(dict(image_id=names[0],index=i,**r) for r in rows)
            assert all(p.grad is None for k,p in model.named_parameters() if k not in PARAMS)
            if i in stability:base_hash[i]=tensor_hash((str(k),v) for k,v in enumerate(result))
            if (i+1)%100==0:print(json.dumps(dict(phase='full_gradient_audit',images=i+1,seconds=time.perf_counter()-gradient_start)),flush=True)
        torch.cuda.synchronize();gradient_seconds=time.perf_counter()-gradient_start
        # Fixed stability images were all audited above; repeat tensor outputs without gradient to test identity.
        exact=True;replay_start=time.perf_counter()
        with torch.no_grad():
            for i in stability:
                _,x=ds[int(i)]
                with torch.autocast('cuda',dtype=torch.bfloat16):r=Net.forward(model,x[None].cuda())
                exact &= tensor_hash((str(k),v) for k,v in enumerate(r))==base_hash[int(i)]
        assert exact
        stability_seconds=time.perf_counter()-replay_start
        # Construct teacher from this real batch's probabilities, detached from reused CAM heads.
        holder.clear();model.zero_grad(set_to_none=True)
        del result,L,g,loss,fg,ps,pd,teacher,fixed_teacher,q,x
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();bs=time.perf_counter()
        x=torch.stack([ds[int(i)][1] for i in batch20]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            r=Net.forward(model,x)
            ps=model.ic1(holder['raw']).float().softmax(1)
        pd=r[8].float().softmax(1)
        ps.retain_grad() if ps.requires_grad else None
        pd.retain_grad() if pd.requires_grad else None
        teach=detached_teacher(ps,pd);teacher=teach['anchor_sym'].reshape(20,4,28,28);q=teach['q'].reshape(20,28,28)
        loss,_=loss_probe(r[6].float(),teacher,q)
        holder['rect'].retain_grad();loss.backward();torch.cuda.synchronize()
        smoke_params=param_stats(selected_params)
        smoke=dict(batch=20,indices=batch20,loss=loss.item(),loss_dtype=str(loss.dtype),logit_dtype=str(r[6].dtype),
                   seconds=time.perf_counter()-bs,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),reserved_budget_bytes=22*1024**3,
                   finite=bool(torch.isfinite(loss) and all(x['finite'] for x in smoke_params) and torch.isfinite(holder['rect'].grad).all()),
                   parameter_gradients=smoke_params,feature_gradient_nonzero=bool(holder['rect'].grad.abs().max()>0),
                   teacher_detached=not teacher.requires_grad,q_detached=not q.requires_grad,
                   ps_teacher_grad_none=ps.grad is None,pd_teacher_grad_none=pd.grad is None,
                   optimizer_created=False,optimizer_steps=0,scope='selected HFRM28_1/ic1 backward; NOT full-unfrozen training memory')
        smoke['budget_pass']=smoke['peak_cuda_reserved_bytes']<=smoke['reserved_budget_bytes']
        assert smoke['finite'] and smoke['teacher_detached'] and smoke['q_detached'] and smoke['ps_teacher_grad_none'] and smoke['pd_teacher_grad_none']
        write_json(out/(PREFIX+'bf16_smoke.json'),smoke)
        after=state_hash(model)
        model.zero_grad(set_to_none=True)
        official_after=official_identity(model,ds,stability)
        assert official_before==official_after and before==after
        assert not any(m.training for m in model.modules())
        assert sha256(args.checkpoint)==CKPT_SHA
    handle.remove()
    observations=out/(PREFIX+'gradient_observations.npz')
    np.savez_compressed(observations,names=data['names'],logits=logits,gradients=grads,losses=losses,
                        feature_norm=feature_norm,feature_max=feature_max)
    write_csv(out/(PREFIX+'parameter_per_image.csv'),param_rows)
    write_json(out/(PREFIX+'detach_audit.json'),dict(teacher_detached=True,q_detached=True,teacher_source_grad_none=True,
              q_source_grad_none=True,student_only_parameters=list(PARAMS),other_model_grads_none=True,
              shared_ic1_note='ic1 has legitimate student gradients; raw-probe teacher branch is detached',
              no_gt_in_loss=True,no_third_evidence=True,optimizer_constructor_guard=True,checkpoint_write_guard=True))
    write_json(out/(PREFIX+'identity_audit.json'),dict(state_before=before,state_after=after,all_parameters_buffers_equal=before==after,
               checkpoint_sha_before=CKPT_SHA,checkpoint_sha_after=sha256(args.checkpoint),checkpoint_written=False,
               official_before=official_before,official_after=official_after,official_predictions_exact=True,
               fixed160_forward_tensors_exact=exact,model_eval_all_modules=True,optimizer_steps=0,
               missing_keys=load.missing_keys,unexpected_keys=load.unexpected_keys,
               inference_scope='original infer/forward_cam, fixed160, original TTA/presence/normalization/fusion/decoding, raw predictions before metric overwrite'))
    runtime=dict(images=n,foreground_targets=int((data['truth']<4).sum()),a0_commit=A0,
                 code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 command=shlex.join([sys.executable,*sys.argv]),checkpoint=args.checkpoint,checkpoint_sha256=CKPT_SHA,
                 native=args.native,native_sha256=NATIVE_SHA,derived=args.derived,derived_sha256=DERIVED_SHA,
                 observations_sha256=sha256(observations),val_images=args.val_images,
                 parity_max_abs=parity,forward_source_parity=source_parity,parity_seconds=parity_seconds,
                 gradient_seconds=gradient_seconds,initial_identity_seconds=identity_seconds,stability_replay_seconds=stability_seconds,
                 total_seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__,
                 main_batch=1,amp='BF16 forward/FP32 loss',benchmark=False,matmul_precision='none',conv_precision='tf32',
                 numerical_stability=dict(deterministic32=32,random128=128,all3418=True,all_finite=all_finite),
                 dataset_files_accessed=len(accessed),test_access=False,luad_access=False,train_access=False,
                 checkpoint_written=False,optimizer_created=False,optimizer_steps=0)
    write_json(out/(PREFIX+'runtime.json'),runtime)
    print(json.dumps(dict(phase='complete',runtime=runtime)),flush=True)


if __name__=='__main__':main()
