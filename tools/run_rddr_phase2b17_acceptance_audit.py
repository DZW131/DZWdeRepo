"""Full validation acceptance audit: no optimizer, updates, checkpoint writes or test access."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.rddr_phase2b16_common import A0,CKPT_SHA,NATIVE_SHA,DERIVED_SHA,EPS,PARAMS,PATHS,sha256,write_json,write_csv,detached_teacher,loss_probe
from tools.rddr_phase2b17_common import PREFIX,GRAD_SHA,MODES,acceptance_support,acceptance_loss
from network.resnet38_cls import Net,Net_CAM
from tool.GenDataset import Stage1_InferDataset


def array_tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def digest_tensors(items):
    h=hashlib.sha256()
    for name,value in items:
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def reject_mutation(*args,**kwargs):raise RuntimeError('optimizer/weight write forbidden')


def official_identity(model,ds,selection):
    from tool import infer_fun
    files=[ds.object[int(i)] for i in selection]
    def factory(*args,**kwargs):
        result=Stage1_InferDataset(*args,**kwargs);result.object=files;return result
    def hash_predictions(gt,pred,n_class):
        assert len(gt)==len(pred)==160 and all(np.isfinite(p).all() for p in pred)
        h=hashlib.sha256()
        for value in pred:h.update(value.tobytes())
        return dict(images=len(pred),pixels=sum(v.size for v in pred),prediction_sha256=h.hexdigest())
    args=SimpleNamespace(dataset='bcss',img_size=224,amp_dtype='bf16',num_workers=0)
    with patch.object(infer_fun,'Stage1_InferDataset',factory),patch.object(infer_fun.iouutils,'scores',hash_predictions):
        result=infer_fun.infer(model,str(Path(ds.data_path).parent),4,args)
    assert result is not None
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('native','derived','previous','checkpoint','val-images','output'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    paths={k:Path(getattr(args,k)) for k in ('native','derived','previous','checkpoint')}
    expected={'native':NATIVE_SHA,'derived':DERIVED_SHA,'previous':GRAD_SHA,'checkpoint':CKPT_SHA}
    assert all(sha256(paths[k])==value for k,value in expected.items())
    assert Path(args.val_images).as_posix().endswith('/BCSS-WSSS/val/img')
    accessed=set()
    def access_guard(event,params):
        if event=='open' and isinstance(params[0],(str,bytes)):
            path=str(params[0]).replace('\\','/').lower()
            if '/reseg-data/' in path:
                assert '/bcss-wsss/val/' in path,('forbidden dataset path',path)
                accessed.add(path)
    sys.addaudithook(access_guard)
    with np.load(paths['native'],allow_pickle=False) as z:data={k:z[k] for k in ('names','ps','pd','q_feature')}
    with np.load(paths['derived'],allow_pickle=False) as z:old={k:z[k] for k in ('names','T_SS','T_SD','T_DS','T_DD','sym','wD_sym','anchor_sym')}
    with np.load(paths['previous'],allow_pickle=False) as z:prev={k:z[k] for k in ('names','logits','gradients','losses')}
    n=len(data['names']);assert n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],prev['names'])
    ds=Stage1_InferDataset(args.val_images,img_size=224);ds.object=sorted(ds.object)
    assert [Path(p).stem for p in ds.object]==data['names'].tolist()
    fixed=np.linspace(0,n-1,32,dtype=int);random=np.random.default_rng(42).choice(np.setdiff1d(np.arange(n),fixed),128,replace=False)
    selection=np.r_[fixed,random]
    out.mkdir(parents=True)
    write_json(out/(PREFIX+'selection.json'),dict(fixed32=fixed,random128=random,batch20=fixed[:20],names=data['names'][selection],
               contract_sha256=sha256(ROOT/'docs/rddr_phase2b17_contract.md')))
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    assert torch.backends.cuda.matmul.fp32_precision=='none' and torch.backends.cudnn.conv.fp32_precision=='tf32'
    torch.cuda.reset_peak_memory_stats()
    parity={k:0. for k in old if k!='names'};parity.update(q=0.,U_gradient=0.,CCA_gradient=0.,U_loss=0.,CCA_loss=0.)
    p_rect=np.empty((n,4,784),np.float32)
    # No new acceptance scores or GT labels are examined until full frozen parity passes.
    for i in range(n):
        ps=array_tensor(data['ps'][i:i+1].reshape(1,4,28,28));pd=array_tensor(data['pd'][i:i+1].reshape(1,4,28,28))
        t=detached_teacher(ps,pd)
        for key in old:
            if key=='names':continue
            assert torch.isfinite(t[key]).all()
            parity[key]=max(parity[key],float(np.abs(t[key].cpu().numpy()-old[key][i:i+1]).max()))
        parity['q']=max(parity['q'],float(np.abs(t['q'].cpu().numpy()-data['q_feature'][i:i+1]).max()))
        L=array_tensor(prev['logits'][i:i+1].reshape(1,4,28,28)).requires_grad_()
        p_rect[i]=L.detach().softmax(1).flatten(2).cpu().numpy()[0]
        teacher=array_tensor(old['anchor_sym'][i:i+1].reshape(1,4,28,28));q=array_tensor(data['q_feature'][i:i+1].reshape(1,28,28))
        for name,j in (('U',0),('CCA',2)):
            loss,_=loss_probe(L,teacher,q,name);g,=torch.autograd.grad(loss,L)
            assert torch.isfinite(loss) and torch.isfinite(g).all()
            parity[name+'_loss']=max(parity[name+'_loss'],abs(loss.item()-float(prev['losses'][i,j])))
            parity[name+'_gradient']=max(parity[name+'_gradient'],float(np.abs(g.flatten(2).cpu().numpy()[0]-prev['gradients'][i,j]).max()))
        assert max(parity.values())<=1e-7,(i,parity)
    print(json.dumps(dict(phase='frozen_parity_pass',images=n,parity=parity)),flush=True)
    parity_seconds=time.perf_counter()-start
    support={k:np.empty((n,784),np.float32) for k in ('R_S','R_D','T_S','T_D','S_R','S_T','delta')}
    support_start=time.perf_counter()
    for i in range(n):
        tensors=[array_tensor(a[i:i+1].reshape(1,4,28,28)) for a in (data['ps'],data['pd'],p_rect,old['anchor_sym'])]
        values=acceptance_support(*tensors)
        for k,v in values.items():support[k][i]=v.cpu().numpy()[0]
    support_seconds=time.perf_counter()-support_start
    print(json.dumps(dict(phase='GT_blind_support_complete',images=n)),flush=True)
    model=Net_CAM(4).cuda();load=model.load_state_dict(torch.load(paths['checkpoint'],map_location='cpu',weights_only=False),strict=True)
    model.eval();model.requires_grad_(False);initial_state=digest_tensors(model.state_dict().items())
    captured={}
    handle=model.hfrm_28_1.register_forward_hook(lambda module,inp,res:captured.update(raw=inp[0],rect=res))
    params=[(k,dict(model.named_parameters())[k]) for k in PARAMS]
    gradients=np.empty((n,2,4,784),np.float32);losses=np.empty((n,4),np.float32);param_rows=[];raw_hash={};logit_error=0.
    with patch.object(torch.optim.Optimizer,'__init__',reject_mutation),patch.object(torch,'save',reject_mutation):
        pred_before=official_identity(model,ds,selection)
        for _,p in params:p.requires_grad_(True)
        assert {k for k,p in model.named_parameters() if p.requires_grad}==set(PARAMS)
        run_start=time.perf_counter()
        for i,(names,x) in enumerate(DataLoader(ds,batch_size=1,num_workers=0,shuffle=False,pin_memory=True)):
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):outputs=Net.forward(model,x.cuda(non_blocking=True))
            assert all(torch.isfinite(value).all() for value in outputs)
            L=outputs[6].float()
            err=float(np.abs(L.detach().flatten(2).cpu().numpy()[0]-prev['logits'][i]).max());logit_error=max(logit_error,err)
            assert err==0.,(i,err)
            teacher=array_tensor(old['anchor_sym'][i:i+1].reshape(1,4,28,28));q=array_tensor(data['q_feature'][i:i+1].reshape(1,28,28))
            delta=array_tensor(support['delta'][i:i+1].reshape(1,28,28))
            for j,mode in enumerate(MODES):
                loss,_=acceptance_loss(L,teacher,q,delta,mode)
                losses[i,j]=loss.item();assert torch.isfinite(loss)
                if mode in ('U','CCA'):continue
                g,=torch.autograd.grad(loss,L,retain_graph=True)
                gradients[i,j-2]=g.detach().flatten(2).cpu().numpy()[0]
                assert torch.isfinite(g).all()
                model.zero_grad(set_to_none=True);loss.backward(retain_graph=(mode=='HA'))
                for key,p in params:
                    assert p.grad is not None and torch.isfinite(p.grad).all(),(i,key)
                    grad=p.grad.detach().float()
                    param_rows.append(dict(index=i,image_id=names[0],loss=mode,parameter=key,
                                           rms=float(grad.double().square().mean().sqrt()),max_abs=float(grad.abs().max()),
                                           sumsq=float(grad.double().square().sum()),finite=True,
                                           nonzero_fraction=float((grad!=0).float().mean())))
                assert all(p.grad is None for k,p in model.named_parameters() if k not in PARAMS)
            if i in selection:raw_hash[i]=digest_tensors((str(j),value) for j,value in enumerate(outputs))
            if (i+1)%500==0:print(json.dumps(dict(phase='real_zero_update_gradients',images=i+1,seconds=time.perf_counter()-run_start)),flush=True)
        torch.cuda.synchronize();gradient_seconds=time.perf_counter()-run_start
        with torch.no_grad():
            for i in selection:
                _,x=ds[int(i)]
                with torch.autocast('cuda',dtype=torch.bfloat16):again=Net.forward(model,x[None].cuda())
                assert digest_tensors((str(j),v) for j,v in enumerate(again))==raw_hash[int(i)]
        # Real batch20 path: build acceptance from this batch, then detach from student.
        captured.clear();model.zero_grad(set_to_none=True)
        del outputs,L,g,loss,x,teacher,q,delta,again
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();smoke_start=time.perf_counter()
        x=torch.stack([ds[int(i)][1] for i in fixed[:20]]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=Net.forward(model,x);ls=model.ic1(captured['raw'])
        ps=ls.float().softmax(1);pd=outputs[8].float().softmax(1);pr=outputs[6].float().softmax(1)
        if ps.requires_grad:ps.retain_grad()
        if pr.requires_grad:pr.retain_grad()
        t=detached_teacher(ps,pd);teacher=t['anchor_sym'].reshape(20,4,28,28);q=t['q'].reshape(20,28,28)
        new=acceptance_support(ps,pd,pr,teacher);delta=new['delta'].reshape(20,28,28)
        smoke_losses={}
        for mode in ('HA','SA'):
            model.zero_grad(set_to_none=True);loss,_=acceptance_loss(outputs[6].float(),teacher,q,delta,mode)
            loss.backward(retain_graph=mode=='HA');assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in params)
            smoke_losses[mode]=dict(loss=loss.item(),parameter_nonzero={k:bool(p.grad.abs().max()>0) for k,p in params})
        torch.cuda.synchronize()
        smoke=dict(batch=20,losses=smoke_losses,seconds=time.perf_counter()-smoke_start,
                   allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),
                   budget_bytes=22*1024**3,all_finite=True,ps_grad_none=ps.grad is None,pr_acceptance_grad_none=pr.grad is None,
                   teacher_detached=not teacher.requires_grad,q_detached=not q.requires_grad,delta_detached=not delta.requires_grad,
                   scope='selected HFRM28_1/ic1 backward, NOT full-unfrozen training memory')
        smoke['budget_pass']=smoke['reserved_bytes']<=smoke['budget_bytes']
        assert smoke['ps_grad_none'] and smoke['pr_acceptance_grad_none'] and all(smoke[k] for k in ('teacher_detached','q_detached','delta_detached','budget_pass'))
        final_state=digest_tensors(model.state_dict().items());model.zero_grad(set_to_none=True)
        pred_after=official_identity(model,ds,selection)
        assert initial_state==final_state and pred_before==pred_after
        assert not any(m.training for m in model.modules())
    handle.remove()
    assert all(sha256(paths[k])==value for k,value in expected.items())
    obs=out/(PREFIX+'observations.npz')
    np.savez_compressed(obs,names=data['names'],p_rect=p_rect,gradients=gradients,losses=losses,**support)
    write_csv(out/(PREFIX+'parameter_gradients.csv'),param_rows)
    write_json(out/(PREFIX+'bf16_smoke.json'),smoke)
    write_json(out/(PREFIX+'detach_audit.json'),dict(teacher_detached=True,q_detached=True,delta_detached=True,m_detached=True,a_detached=True,
              ps_source_gradient_none=smoke['ps_grad_none'],rect_acceptance_branch_gradient_none=smoke['pr_acceptance_grad_none'],
              allowed_parameters=list(PARAMS),all_other_parameter_gradients_none=True,optimizer_created=False,steps=0,
              note='shared ic1 has legitimate student gradients, not teacher/acceptance-branch gradients'))
    write_json(out/(PREFIX+'identity_audit.json'),dict(state_before=initial_state,state_after=final_state,all_parameters_buffers_unchanged=True,
              checkpoint_sha_before=CKPT_SHA,checkpoint_sha_after=sha256(paths['checkpoint']),checkpoint_written=False,
              prediction_before=pred_before,prediction_after=pred_after,official_predictions_exact=True,raw_forward_fixed160_exact=True,
              max_logit_replay_difference=logit_error,model_all_eval=True,bn_buffers_unchanged=True,missing_keys=load.missing_keys,unexpected_keys=load.unexpected_keys))
    runtime=dict(images=n,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),a0=A0,
                 command=shlex.join([sys.executable,*sys.argv]),paths={k:str(p) for k,p in paths.items()},source_sha256=expected,
                 observation_sha256=sha256(obs),parity=parity,parity_seconds=parity_seconds,support_seconds=support_seconds,
                 gradient_seconds=gradient_seconds,total_seconds=time.perf_counter()-start,torch=torch.__version__,numpy=np.__version__,
                 gpu=torch.cuda.get_device_name(),precision='BF16 network / FP32 softmax, support and loss',batch=1,
                 optimizer_created=False,optimizer_steps=0,checkpoint_written=False,test_access=False,luad_access=False,other_seed=False,
                 dataset_files_accessed=len(accessed),original_source_changed=False,all_finite=True)
    write_json(out/(PREFIX+'runtime.json'),runtime)
    print(json.dumps(dict(phase='complete',runtime=runtime)),flush=True)


if __name__=='__main__':main()
