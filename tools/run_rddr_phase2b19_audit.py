"""Full validation directional transfer audit. No optimizer, training, weight write, test or search."""
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
from tools.rddr_phase2b16_common import A0,CKPT_SHA,NATIVE_SHA,DERIVED_SHA,sha256,write_json,write_csv
from tools.rddr_phase2b19_common import PREFIX as P,PREVIOUS_SHA,MODES,UPSTREAM,upstream_name,student_logits,directional_loss,conflict_gradient,adjudicate,random_gate,direction_gate
from network.resnet38_cls import Net,Net_CAM
from tool.GenDataset import Stage1_InferDataset


def tensor(a):return torch.from_numpy(np.ascontiguousarray(a)).cuda()
def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def digest(items):
    h=hashlib.sha256()
    for k,v in items:
        v=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def forbidden(*args,**kwargs):raise RuntimeError('optimizer/checkpoint write prohibited')


def identity(model,ds,indices):
    from tool import infer_fun
    files=[ds.object[int(i)] for i in indices]
    def factory(*a,**kw):
        d=Stage1_InferDataset(*a,**kw);d.object=files;return d
    def score(gt,pred,n_class):
        assert len(gt)==len(pred)==160
        h=hashlib.sha256()
        for x in pred:assert np.isfinite(x).all();h.update(x.tobytes())
        return dict(images=160,pixels=sum(x.size for x in pred),sha256=h.hexdigest())
    with patch.object(infer_fun,'Stage1_InferDataset',factory),patch.object(infer_fun.iouutils,'scores',score):
        r=infer_fun.infer(model,str(Path(ds.data_path).parent),4,SimpleNamespace(dataset='bcss',img_size=224,amp_dtype='bf16',num_workers=0))
    assert r is not None
    return r


def parameter_stats(params):
    zero=torch.zeros((),device='cuda',dtype=torch.float64);energy=[];maximum=[]
    for name,p in params:
        if p.grad is None:energy.append(zero);maximum.append(zero)
        else:
            assert torch.isfinite(p.grad).all(),name
            g=p.grad.detach().double();energy.append(g.square().sum());maximum.append(g.abs().max())
    return torch.stack(energy).cpu().numpy(),torch.stack(maximum).cpu().numpy()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('native','derived','previous','checkpoint','val-images','output'):ap.add_argument('--'+key,required=True)
    args=ap.parse_args();out=Path(args.output);start=time.perf_counter()
    if out.exists():raise FileExistsError(out)
    paths={k:Path(getattr(args,k)) for k in ('native','derived','previous','checkpoint')}
    hashes=dict(native=NATIVE_SHA,derived=DERIVED_SHA,previous=PREVIOUS_SHA,checkpoint=CKPT_SHA)
    assert all(sha256(paths[k])==h for k,h in hashes.items())
    assert not subprocess.check_output(['git','diff',A0,'--','network','tool','train_sshr.py'],cwd=ROOT)
    assert str(args.val_images).endswith('/BCSS-WSSS/val/img')
    accessed=set()
    def audit(event,items):
        if event=='open' and isinstance(items[0],(str,bytes)):
            p=str(items[0]).replace('\\','/').lower()
            if '/reseg-data/' in p:
                assert '/bcss-wsss/val/' in p,p
                accessed.add(p)
    sys.addaudithook(audit)
    data=loadnp(paths['native']);old=loadnp(paths['derived']);prev=loadnp(paths['previous'])
    n=len(data['names']);assert n==3418 and np.array_equal(data['names'],old['names']) and np.array_equal(data['names'],prev['names'])
    delta=.5*(old['T_DS']+old['T_DD'])-.5*(old['T_SS']+old['T_SD'])
    assert np.array_equal(delta,old['sym'])
    rg=random_gate(delta);gate=delta>0
    assert np.array_equal(rg.sum(1),gate.sum(1)) and np.array_equal(rg,random_gate(delta))
    ds=Stage1_InferDataset(args.val_images,img_size=224);ds.object=sorted(ds.object)
    assert [Path(p).stem for p in ds.object]==data['names'].tolist()
    fixed=np.linspace(0,n-1,32,dtype=int);random=np.random.default_rng(42).choice(np.setdiff1d(np.arange(n),fixed),128,replace=False)
    selection=np.r_[fixed,random];out.mkdir(parents=True)
    write_json(out/(P+'selection.json'),dict(fixed32=fixed,random128=random,batch20=fixed[:20],names=data['names'][selection],contract_sha256=sha256(ROOT/'docs/rddr_phase2b19_contract.md')))
    torch.manual_seed(42);torch.backends.cudnn.benchmark=False
    assert torch.backends.cuda.matmul.fp32_precision=='none' and torch.backends.cudnn.conv.fp32_precision=='tf32'
    model=Net_CAM(4).cuda();ld=model.load_state_dict(torch.load(paths['checkpoint'],map_location='cpu',weights_only=False),strict=True)
    model.eval();model.requires_grad_(False);before=digest(model.state_dict().items())
    bn_before=digest((k,v) for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k)
    capture={};hook=model.hfrm_28_1.register_forward_hook(lambda m,inp,r:capture.update(raw=inp[0],deep=inp[1]))
    rawlogits=np.empty((n,4,784),np.float32)
    parity={k:0. for k in ('ps','pd','q','raw_frozen_head_logits','raw_previous_logits','delta_stored','delta_recomputed','supports')}
    recomputed_gate_mismatches=0
    with patch.object(torch.optim.Optimizer,'__init__',forbidden),patch.object(torch.optim.Optimizer,'step',forbidden),patch.object(torch,'save',forbidden):
        pred_before=identity(model,ds,selection)
        with torch.no_grad():
            for i,(_,x) in enumerate(DataLoader(ds,batch_size=1,num_workers=0,shuffle=False)):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=Net.forward(model,x.cuda());original=model.ic1(capture['raw']);frozen=student_logits(capture['raw'],model.ic1)
                assert capture['raw'].shape==(1,512,28,28) and capture['deep'].shape==(1,4096,28,28)
                ps=original.float().softmax(1);pd=outputs[8].float().softmax(1);t=adjudicate(ps,pd)
                rawlogits[i]=original.float().flatten(2).cpu().numpy()[0]
                vals={'ps':(ps.flatten(2).cpu().numpy()[0],data['ps'][i]),'pd':(pd.flatten(2).cpu().numpy()[0],data['pd'][i]),
                      'q':(t['q'].cpu().numpy()[0],data['q_feature'][i]),
                      'raw_frozen_head_logits':(original.float().cpu().numpy(),frozen.float().cpu().numpy()),
                      'raw_previous_logits':(rawlogits[i],prev['raw_logits'][i]),
                      'delta_recomputed':(t['sym'].cpu().numpy()[0],delta[i])}
                for k in ('T_SS','T_SD','T_DS','T_DD'):
                    parity['supports']=max(parity['supports'],float(np.abs(t[k].cpu().numpy()[0]-old[k][i]).max()))
                recomputed_gate_mismatches+=int(np.count_nonzero((t['sym'].cpu().numpy()[0]>0)!=gate[i]))
                for k,(a,b) in vals.items():parity[k]=max(parity[k],float(np.max(np.abs(a-b))))
                assert max(parity.values())<=1e-7,parity
                assert parity['raw_frozen_head_logits']==parity['raw_previous_logits']==0.
        print(json.dumps(dict(phase='frozen_replay_pass',parity=parity,recomputed_gate_mismatches=recomputed_gate_mismatches)),flush=True)
        parity_seconds=time.perf_counter()-start
        for name,p in model.named_parameters():p.requires_grad_(upstream_name(name))
        upstream=[(k,p) for k,p in model.named_parameters() if upstream_name(k)]
        params=upstream+[('ic1.weight',model.ic1.weight),('ic1.bias',model.ic1.bias)]
        assert len(upstream)==39
        g=np.empty((n,4,4,784),np.float32);gq=np.empty((n,4,784),np.float32);losses=np.empty((n,4),np.float32)
        energy=np.zeros((n,len(params)),np.float64);pmax=np.zeros_like(energy)
        feature_sqsum=np.empty((n,784),np.float64);feature_maxabs=np.empty((n,784),np.float32)
        gradient_logit_error=0.;qgrad_replay_error=0.;tick=time.perf_counter()
        for i,(_,x) in enumerate(DataLoader(ds,batch_size=1,num_workers=0,shuffle=False)):
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=Net.forward(model,x.cuda());raw=capture['raw'];L=student_logits(raw,model.ic1)
            deep=capture['deep'];deep.retain_grad();L=L.float()
            gradient_logit_error=max(gradient_logit_error,float(np.abs(L.detach().flatten(2).cpu().numpy()[0]-rawlogits[i]).max()))
            assert gradient_logit_error==0.
            q=tensor(data['q_feature'][i:i+1].reshape(1,28,28));dlt=tensor(delta[i:i+1].reshape(1,28,28))
            pd=tensor(data['pd'][i:i+1].reshape(1,4,28,28));rand=tensor(rg[i:i+1].reshape(1,28,28))
            _,qgrad=conflict_gradient(L,pd);gq[i]=qgrad.flatten(2).cpu().numpy()[0]
            qgrad_replay_error=max(qgrad_replay_error,float(np.abs(gq[i]-prev['q_gradients'][i]).max()))
            for j,mode in enumerate(MODES):
                loss=directional_loss(L,pd,q,dlt,mode,rand);gg,=torch.autograd.grad(loss,L,retain_graph=True)
                assert torch.isfinite(loss) and torch.isfinite(gg).all()
                losses[i,j]=loss.item();g[i,j]=gg.detach().flatten(2).cpu().numpy()[0]
                if mode in ('ADT','SDT'):assert np.all(g[i,j,:,~gate[i]]==0)
                if mode=='ADT':primary=loss
            fg,=torch.autograd.grad(primary,raw,retain_graph=True);assert torch.isfinite(fg).all()
            feature_sqsum[i]=fg.detach().double().square().sum(1).flatten().cpu().numpy()
            feature_maxabs[i]=fg.detach().float().abs().amax(1).flatten().cpu().numpy()
            assert np.all(feature_sqsum[i,~gate[i]]==0)
            primary.backward()
            assert deep.grad is None and all(p.grad is None for k,p in model.named_parameters() if not upstream_name(k))
            assert all(p.grad is not None for _,p in upstream)
            energy[i],pmax[i]=parameter_stats(params)
            if (i+1)%500==0:print(json.dumps(dict(phase='real_ADT_upstream_backward',images=i+1,seconds=time.perf_counter()-tick)),flush=True)
        torch.cuda.synchronize();gradient_seconds=time.perf_counter()-tick
        assert qgrad_replay_error==0
        assert all(np.isfinite(a).all() for a in (g,gq,energy,feature_sqsum,feature_maxabs,losses))
        conv_idx=[j for j,(k,p) in enumerate(params) if upstream_name(k) and '.conv_' in k]
        assert len(conv_idx)==13 and energy[:,conv_idx].sum()>0 and np.all(energy[:,-2:]==0)
        for block in UPSTREAM[:-1]:
            ix=[j for j,(k,p) in enumerate(params) if k.startswith(block+'.conv_')]
            assert energy[:,ix].sum()>0,block
        del outputs,raw,L,fg,primary,loss,gg,x,deep
        capture.clear();model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        x=torch.stack([ds[int(i)][1] for i in fixed[:20]]).cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=Net.forward(model,x);raw=capture['raw'];L=student_logits(raw,model.ic1).float()
        ps=L.softmax(1);pd=outputs[8].float().softmax(1);ps.retain_grad();pd.retain_grad();capture['deep'].retain_grad()
        t=adjudicate(ps,pd);q=t['q'];dlt=t['sym'];m=direction_gate(dlt)
        loss=directional_loss(L,pd,q,dlt,'ADT');loss.backward();se,sm=parameter_stats(params);torch.cuda.synchronize()
        assert torch.isfinite(loss) and ps.grad is None and pd.grad is None and capture['deep'].grad is None
        assert all(p.grad is None for k,p in model.named_parameters() if not upstream_name(k))
        smoke=dict(batch=20,loss=loss.item(),active_transfer_fraction=m.float().mean().item(),seconds=time.perf_counter()-tick,
                   allocated_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.max_memory_reserved(),budget_bytes=22*1024**3,
                   all_finite=True,upstream_conv_energy=float(se[conv_idx].sum()),head_energy=float(se[-2:].sum()),
                   q_detached=not q.requires_grad,delta_detached=not dlt.requires_grad,gate_detached=not m.requires_grad,
                   source_ps_grad_none=ps.grad is None,deep_source_grad_none=pd.grad is None,
                   scope='temporary b4-stage/bn45 backward; live batch20 adjudication, not main batch1 cache; NOT full training memory')
        smoke['pass']=smoke['reserved_bytes']<=smoke['budget_bytes'] and smoke['upstream_conv_energy']>0 and smoke['head_energy']==0
        assert smoke['pass'];model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for i in selection:
                _,x=ds[int(i)]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    Net.forward(model,x[None].cuda());check=model.ic1(capture['raw']).float()
                assert np.array_equal(check.flatten(2).cpu().numpy()[0],rawlogits[int(i)])
        after=digest(model.state_dict().items())
        bn_after=digest((k,v) for k,v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k)
        pred_after=identity(model,ds,selection)
        assert before==after and bn_before==bn_after and pred_before==pred_after and all(not m.training for m in model.modules())
    hook.remove();assert all(sha256(paths[k])==h for k,h in hashes.items())
    obs=out/(P+'observations.npz')
    np.savez_compressed(obs,names=data['names'],raw_logits=rawlogits,gradients=g,q_gradients=gq,losses=losses,random_gate=rg,direction_gate=gate,
                        parameter_names=np.array([k for k,p in params]),parameter_numel=np.array([p.numel() for k,p in params]),
                        parameter_energy=energy,parameter_maxabs=pmax,feature_sqsum=feature_sqsum,feature_maxabs=feature_maxabs)
    write_json(out/(P+'bf16_smoke.json'),smoke)
    write_json(out/(P+'detach_audit.json'),dict(q_detached=True,delta_detached=True,gate_detached=True,deep_source_detached=True,primary_ic1_none=True,hfrm_none=True,
        approved_upstream=[k for k,p in upstream],upstream_conv_nonzero=True,all_other_primary_gradients_none=True,
        rejected_feature_zero=True,rejected_logit_zero=True,optimizer_created=False,optimizer_steps=0,checkpoint_written=False))
    write_json(out/(P+'identity_audit.json'),dict(state_before=before,state_after=after,state_unchanged=before==after,
        bn_before=bn_before,bn_after=bn_after,bn_unchanged=bn_before==bn_after,
        checkpoint_sha_before=CKPT_SHA,checkpoint_sha_after=sha256(paths['checkpoint']),prediction_before=pred_before,prediction_after=pred_after,
        prediction_unchanged=pred_before==pred_after,raw_fixed160_exact=True,raw_gradient_replay_max_abs=gradient_logit_error,
        missing_keys=ld.missing_keys,unexpected_keys=ld.unexpected_keys))
    runtime=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),a0=A0,images=n,
        command=shlex.join([sys.executable,*sys.argv]),paths={k:str(p) for k,p in paths.items()},source_sha256=hashes,observation_sha256=sha256(obs),
        random_gate_sha256=hashlib.sha256(rg.tobytes()).hexdigest(),qgrad_replay_max_abs=qgrad_replay_error,
        parity=parity,recomputed_gate_mismatches=recomputed_gate_mismatches,parity_seconds=parity_seconds,gradient_seconds=gradient_seconds,total_seconds=time.perf_counter()-start,
        torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),precision='BF16 network; FP32 loss/logit/q; FP64 statistics',
        optimizer_created=False,optimizer_steps=0,checkpoint_written=False,test_access=False,luad_access=False,training_split_access=False,
        original_sources_changed=False,all_finite=True,dataset_files_accessed=len(accessed))
    write_json(out/(P+'runtime.json'),runtime)
    print(json.dumps(dict(phase='complete',runtime=runtime)),flush=True)

if __name__=='__main__':main()

