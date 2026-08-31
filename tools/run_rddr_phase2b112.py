"""Frozen500-step three-arm ADT audit; no test, tuning, resume or Full25 mode."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from network.resnet38_cls import Net, Net_CAM
from tool.GenDataset import Stage1_TrainDataset
from train_sshr import set_seed, seed_worker
from tools.rddr_phase2b112_common import (A0, CKPT_SHA, NATIVE_SHA, P, MAX_STEP, FINAL_LRS,
    SNAPSHOTS, INTERACTIONS, sha256, digest, write_json, write_csv, make_optimizer, optimizer_record,
    gradients_for_batch, rng_state, restore_rng, move_arm, bn_digest, upstream_name)

C0_RUN = Path('/home/duyanhong/sshr-official-25ep-final-retry2-20260815')
CHECKPOINT = C0_RUN/'runs/bcss_seed42/checkpoints/stage1_last.pth'
NATIVE = Path('/home/duyanhong/experiments/RDDR_PHASE2B1/formal_r1/rddr_phase2b1_native_observations.npz')
DATA_ROOT = Path('/home/duyanhong/reseg-data/raw/BCSS-WSSS')


class RecordedTrainDataset(Stage1_TrainDataset):
    """Observe official augmentation RNG without consuming additional random draws."""
    def __getitem__(self, index):
        state = random.getstate()
        shadow = random.Random(); shadow.setstate(state)
        metadata = dict(index=int(index), worker_seed=int(torch.initial_seed()),
                        augmentation_rng_sha256=hashlib.sha256(repr(state).encode()).hexdigest(),
                        horizontal_flip=shadow.random() > .5, vertical_flip=shadow.random() > .5)
        name, image, label = super().__getitem__(index)
        return name, image, label, json.dumps(metadata, sort_keys=True)


def loader(dataset):
    generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True,
                      drop_last=True, worker_init_fn=seed_worker, generator=generator)


def metadata(batch, step):
    names, image, label, encoded = batch
    return dict(step=step, names=list(names), augmentation=[json.loads(s) for s in encoded],
                tensor_sha256=digest((('image', image), ('label', label))))


def preflight(out):
    assert not subprocess.check_output(['git', 'diff', A0, '--', 'network', 'tool', 'train_sshr.py'], cwd=ROOT)
    files = [C0_RUN/'environment.tsv', C0_RUN/'runs/bcss_seed42/status.tsv',
             C0_RUN/'runs/bcss_seed42/train.log', Path('/home/duyanhong/run_official_25ep_retry2.sh')]
    environment = dict(line.split('\t', 1) for line in files[0].read_text().splitlines())
    assert environment['repo_commit'] == A0
    launcher = files[3].read_text(); log = files[2].read_text(); status = files[1].read_text()
    assert '--max_epoches 25' in launcher and '--amp-dtype bf16' in launcher
    assert '--num_workers 4' in launcher and '--seed "$seed"' in launcher
    assert '--lr ' not in launcher and '--wt_dec ' not in launcher and '--batch_size ' not in launcher
    assert 'Iter:29200/29275' in log and 'Total Training Time:' in log
    assert f'checkpoint_sha256\t{CKPT_SHA}' in status and 'status\tcomplete' in status
    assert 'seed\t42' in status and 'dataset\tbcss' in status
    assert sha256(CHECKPOINT) == CKPT_SHA and sha256(NATIVE) == NATIVE_SHA
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert len(state) == 260 and all(torch.is_tensor(v) for v in state.values())
    parsed_dataset = Stage1_TrainDataset(str(DATA_ROOT/'training'), dataset='bcss', img_size=224)
    assert len(parsed_dataset) == 23422
    source_paths = files + [ROOT/'train_sshr.py', ROOT/'tool/torchutils.py', ROOT/'network/resnet38d.py',
                           ROOT/'network/resnet38_cls.py', ROOT/'tool/GenDataset.py', ROOT/'tool/infer_fun.py', ROOT/'tool/iouutils.py']
    arms = {}
    hashes = {}
    for arm in ('B', 'A', 'R'):
        model = Net_CAM(4)
        loaded = model.load_state_dict(state, strict=True)
        assert not loaded.missing_keys and not loaded.unexpected_keys
        optimizer = make_optimizer(model)
        hashes[arm] = digest(model.state_dict().items())
        arms[arm] = (model, optimizer)
    assert len(set(hashes.values())) == 1
    groups = optimizer_record(*arms['B'])
    assert all(groups == optimizer_record(*arms[k]) for k in ('A', 'R'))
    provenance = dict(resolved=True, state_policy='fresh_optimizer_state', checkpoint=str(CHECKPOINT),
                      checkpoint_sha256=CKPT_SHA, checkpoint_bytes=CHECKPOINT.stat().st_size,
                      model_state_sha256=hashes['B'], source_hashes={str(p): sha256(p) for p in source_paths},
                      original_environment=environment, global_step=MAX_STEP, max_step=MAX_STEP,
                      last_applied_step=MAX_STEP-1, lr_power=.9, groups=groups,
                      scheduler_rule='Original PolyOptimizer: after max_step retain last applied LR; no restart',
                      optimizer_buffers_recovered=False, initial_optimizer_states_empty=True,
                      training_samples=len(parsed_dataset),batch_size=20,num_workers=4,seed=42,
                      specification_sha256=sha256(ROOT/'docs/rddr_phase2b112_specification.md'),
                      contract_sha256=sha256(ROOT/'docs/rddr_phase2b112_execution_contract.md'))
    write_json(out/(P+'optimizer_provenance.json'), provenance)
    write_json(out/(P+'identity_step0.json'), dict(three_arms_bitwise_equal=True, strict_load=True,
        initial_state_sha256=hashes['B'], state_hashes=hashes, original_sources_unchanged=True,
        initial_prediction_sha256=None, main_forward_parity=None, status='weights_verified_forward_pending'))
    print(json.dumps(dict(phase='optimizer_provenance_pass',state_policy=provenance['state_policy'],
                         lr=list(FINAL_LRS),momentum=.0005,model_hash=hashes['B'])), flush=True)
    return arms, provenance


def gpu_admission():
    free, total = torch.cuda.mem_get_info()
    # Resource admission only, not a protocol change; one resident arm at a time.
    minimum = 18*1024**3
    record = dict(free_bytes=free, total_bytes=total, minimum_free_bytes=minimum,
                  admitted=free >= minimum, policy='No other processes stopped; no smaller batch or accumulation fallback')
    return record


def run(out, arms, provenance):
    from tools.rddr_phase2b112_evaluation import evaluate_snapshot
    start = time.perf_counter()
    access = set()
    phase = ['training_setup']
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            p = os.fsdecode(args[0]).replace('\\', '/').lower()
            if '/reseg-data/' in p:
                assert '/bcss-wsss/training/' in p or '/bcss-wsss/val/' in p, p
                if phase[0] in ('calibration', 'training', 'interaction'):
                    assert '/bcss-wsss/val/' not in p, 'Validation data leaked into optimization'
                access.add(p)
    sys.addaudithook(audit)
    dataset = RecordedTrainDataset(str(DATA_ROOT/'training'), dataset='bcss', img_size=224)
    assert len(dataset) == 23422 and len(dataset)//20*25 == MAX_STEP
    set_seed(42)
    train_rng = rng_state()
    iterator = iter(loader(dataset))
    calibration_batches = [next(iterator) for _ in range(32)]
    del iterator
    calibration_manifest = [metadata(b, i+1) for i, b in enumerate(calibration_batches)]
    diagnostic_batch = calibration_batches[0]
    identity = json.loads((out/(P+'identity_step0.json')).read_text())
    prediction_hashes = {}
    for arm, (model, optimizer) in arms.items():
        move_arm(model, optimizer, 'cuda'); model.eval()
        restore_rng(train_rng)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictions = Net.forward(model, diagnostic_batch[1].cuda())
        prediction_hashes[arm] = digest((str(i), v) for i, v in enumerate(predictions))
        del predictions
        move_arm(model, optimizer, 'cpu'); torch.cuda.empty_cache()
    assert len(set(prediction_hashes.values())) == 1
    identity.update(initial_prediction_sha256=prediction_hashes['B'], prediction_hashes=prediction_hashes,
                    main_forward_parity=True, status='weights_and_forward_verified')
    write_json(out/(P+'identity_step0.json'), identity)
    phase[0] = 'calibration'
    model, optimizer = arms['A']; move_arm(model, optimizer, 'cuda')
    initial_bn = bn_digest(model); initial_state = digest(model.state_dict().items())
    restore_rng(train_rng)
    calibration_rows = []
    torch.cuda.reset_peak_memory_stats()
    for i, batch in enumerate(calibration_batches):
        r, _ = gradients_for_batch(model, batch[1].cuda(), batch[2].cuda(), 'A', 0.)
        assert r['aux_grad_norm'] > 0 and r['main_grad_norm'] > 0
        calibration_rows.append(dict(batch=i+1, main_norm=r['main_grad_norm'], aux_norm=r['aux_grad_norm'],
                                     ratio=r['main_grad_norm']/(r['aux_grad_norm']+1e-8)))
        print(json.dumps(dict(phase='calibration',**calibration_rows[-1])), flush=True)
    assert digest(model.state_dict().items()) == initial_state and bn_digest(model) == initial_bn
    assert optimizer.global_step == MAX_STEP and not optimizer.state
    ratios = [r['ratio'] for r in calibration_rows]; median = float(np.median(ratios)); lam = .1*median
    calibration = dict(batches=32, seed=42, lambda_value=lam, ratios=ratios, r_median=median,
                       rows=calibration_rows, no_optimizer_step=True, state_unchanged=True,
                       budget=.1, recalibrations=0, use_validation=False)
    write_json(out/(P+'lambda_calibration.json'), calibration)
    model.zero_grad(set_to_none=True)
    peak_allocated = torch.cuda.max_memory_allocated(); peak_reserved = torch.cuda.max_memory_reserved()
    move_arm(model, optimizer, 'cpu'); torch.cuda.empty_cache(); restore_rng(train_rng)
    # Save a single shared step0: all three arms are byte-identical and optimizers empty.
    torch.save(dict(model=arms['B'][0].state_dict(), optimizer=arms['B'][1].state_dict(),
                    step=0, global_step=MAX_STEP, lambda_value=lam, rng=train_rng), out/'checkpoint_step0000_shared.pth')
    rows = []; interactions = []; representation = []; train_manifest = []; evaluation = []
    random_rng = np.random.default_rng(42)
    baseline_bn = {k: bn_digest(m) for k, (m, _) in arms.items()}
    baseline_parameters = {k: p.detach().clone() for k, p in arms['B'][0].named_parameters()}
    def evaluate(step):
        nonlocal peak_allocated, peak_reserved
        saved = rng_state(); phase[0] = 'validation'
        for arm, (m, opt) in arms.items():
            move_arm(m, opt, 'cuda')
            result = evaluate_snapshot(m, DATA_ROOT/'val', NATIVE, out, arm, step)
            representation.extend(result.pop('representation_rows'))
            evaluation.append(dict(arm=arm, step=step, **result))
            print(json.dumps(dict(phase='validation_complete',**result)),flush=True)
            peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated())
            peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
            move_arm(m,opt,'cpu'); torch.cuda.empty_cache()
        write_csv(out/(P+'representation_drift.csv'),representation)
        write_json(out/(P+'canonical_evaluations.json'),evaluation)
        restore_rng(saved)
    def interaction(step):
        saved=rng_state(); phase[0]='interaction'; counts=None
        diagnostic_random=np.random.default_rng(42)
        for arm in ('A','R','B'):
            m,opt=arms[arm]; move_arm(m,opt,'cuda'); restore_rng(train_rng)
            r, c=gradients_for_batch(m, diagnostic_batch[1].cuda(),diagnostic_batch[2].cuda(),
                                    arm,lam,random_counts=counts,random_rng=diagnostic_random)
            if arm=='A': counts=c
            interactions.append(dict(arm=arm,step=step,**r))
            assert opt.global_step==MAX_STEP+step
            m.zero_grad(set_to_none=True); move_arm(m,opt,'cpu'); torch.cuda.empty_cache()
        write_csv(out/(P+'gradient_interaction.csv'), interactions); restore_rng(saved)
    evaluate(0)
    with np.load(out/'snapshot_0000_B.npz') as base:
        for arm in ('A','R'):
            with np.load(out/f'snapshot_0000_{arm}.npz') as other:
                for key in base.files: assert np.array_equal(base[key],other[key]), (arm,key,'step0 mismatch')
    identity['full_validation_snapshot_bitwise_equal']=True
    write_json(out/(P+'identity_step0.json'),identity)
    interaction(0); restore_rng(train_rng)
    stream = iter(loader(dataset))
    for step in range(1,501):
        phase[0]='training'
        batch=next(stream); item=metadata(batch,step); train_manifest.append(item)
        if step<=32: assert item==calibration_manifest[step-1], 'Calibration/train stream mismatch'
        shared_rng=rng_state(); counts=None; next_rng=None
        for arm in ('B','A','R'):
            tick=time.perf_counter(); m,opt=arms[arm]
            move_arm(m,opt,'cuda'); restore_rng(shared_rng)
            r,c=gradients_for_batch(m,batch[1].cuda(),batch[2].cuda(),arm,lam,
                                  random_counts=counts,random_rng=random_rng)
            if arm=='B': next_rng=rng_state()
            if arm=='A': counts=c
            if arm=='R': assert r['active_fraction']==rows[-1]['active_fraction']
            opt.step()
            assert opt.global_step==MAX_STEP+step
            assert tuple(g['lr'] for g in opt.param_groups)==FINAL_LRS
            assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
            assert all(bool(torch.isfinite(v).all()) for s in opt.state.values()
                       for v in s.values() if torch.is_tensor(v))
            assert bn_digest(m)==baseline_bn[arm]
            r.update(arm=arm,step=step,lambda_value=lam,seconds=time.perf_counter()-tick,
                     **{f'lr{i}':g['lr'] for i,g in enumerate(opt.param_groups)})
            rows.append(r)
            peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated())
            peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
            m.zero_grad(set_to_none=True); move_arm(m,opt,'cpu'); torch.cuda.empty_cache()
        restore_rng(next_rng)
        if step%10==0:
            write_csv(out/(P+'training_curve.csv'),rows)
            print(json.dumps(dict(phase='training',step=step,lambda_value=lam,latest=rows[-3:])),flush=True)
        if step%50==0:
            write_json(out/(P+'batch_manifest.json'),dict(seed=42,batch_size=20,num_workers=4,
                calibration=calibration_manifest,training=train_manifest,same_batches_all_arms=True,
                policy='Official seeded4-worker DataLoader read once per step; tensor reused across all arms'))
        if step in SNAPSHOTS:
            evaluate(step)
            if step in INTERACTIONS: interaction(step)
        if step in (250,500):
            for arm,(m,opt) in arms.items():
                path=out/f'checkpoint_step{step:04d}_{arm}.pth'; assert not path.exists()
                torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),step=step,
                    global_step=opt.global_step,lambda_value=lam,rng=next_rng,
                    random_gate_rng=copy.deepcopy(random_rng.bit_generator.state)),path)
    del stream
    write_csv(out/(P+'training_curve.csv'),rows)
    movement=[]
    for arm,(m,opt) in arms.items():
        for name,p in m.named_parameters():
            initial=baseline_parameters[name]; change=(p.detach()-initial).double().norm().item()
            movement.append(dict(arm=arm,name=name,absolute_l2=change,
                                 relative_l2=change/(initial.double().norm().item()+1e-30),
                                 approved_aux=upstream_name(name)))
    write_csv(out/(P+'parameter_movement.csv'),movement)
    assert sha256(CHECKPOINT)==CKPT_SHA and sha256(NATIVE)==NATIVE_SHA
    runtime=dict(completed=True,steps_per_arm={k:o.global_step-MAX_STEP for k,(_,o) in arms.items()},
        checks=dict(all_finite=True,no_amp_skipped_step=True,no_unexpected_gradient_path=True,
                    no_state_corruption=True,bn_statistics_frozen=True,no_test_access=True,no_luad_access=True),
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),command=shlex.join([sys.executable,*sys.argv]),
        seconds=time.perf_counter()-start,peak_allocated_bytes=peak_allocated,peak_reserved_bytes=peak_reserved,
        clipping_events=0,amp_overflow_events=0,amp_skipped_steps=0,gradient_scaler_enabled=False,
        dataset_paths_accessed=len(access),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        lambda_value=lam,precision='Official BF16 network; FP32 support/loss; FP64 norm/statistics',
        test_access=False,luad_access=False,original_sources_changed=False)
    write_json(out/(P+'runtime.json'),runtime)
    print(json.dumps(dict(phase='training_and_validation_complete',seconds=runtime['seconds'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args(); out=args.output.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    torch.set_num_threads(4)
    arms,provenance=preflight(out)
    if args.preflight_only:
        write_json(out/(P+'preflight_status.json'),dict(status='CPU_PROVENANCE_PASS',training_started=False))
        return
    admission=gpu_admission(); write_json(out/(P+'resource_admission.json'),admission)
    if not admission['admitted']:
        write_json(out/(P+'runtime.json'),dict(completed=False,training_started=False,optimizer_steps=0,
            status='RESOURCE_BLOCKED',resource=admission,decision=None))
        print(json.dumps(dict(phase='resource_blocked',**admission)),flush=True)
        raise SystemExit(75)
    try:
        run(out,arms,provenance)
    except Exception as error:
        write_json(out/(P+'interrupted.json'),dict(completed=False,error_type=type(error).__name__,error=str(error),
            steps_per_arm={k:o.global_step-MAX_STEP for k,(_,o) in arms.items()},
            policy='No automatic retry, LR change, batch reduction or experiment reset'))
        raise


if __name__=='__main__': main()
