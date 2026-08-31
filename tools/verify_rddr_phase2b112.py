"""Independent artifact checks; refuses to certify a partial/resource-blocked run."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.rddr_phase2b112_common import A0,CKPT_SHA,P,MAX_STEP,FINAL_LRS,sha256,digest,write_json


def verify(run_dir,post_analysis=False):
    checks={}
    def check(name,condition): checks[name]=bool(condition)
    def read(name): return json.loads((run_dir/(P+name+'.json')).read_text())
    runtime=read('runtime')
    if not runtime.get('completed'):
        raise RuntimeError('Run not complete: no scientific readiness or outcome can be certified')
    provenance=read('optimizer_provenance'); identity=read('identity_step0'); calibration=read('lambda_calibration')
    manifest=read('batch_manifest')
    with (run_dir/(P+'training_curve.csv')).open() as f: curve=list(csv.DictReader(f))
    check('runtime_steps_exact',runtime['steps_per_arm']==dict(B=500,A=500,R=500))
    check('frozen_checkpoint',provenance['checkpoint_sha256']==CKPT_SHA and sha256(provenance['checkpoint'])==CKPT_SHA)
    check('fresh_state_explicit',provenance['state_policy']=='fresh_optimizer_state' and not provenance['optimizer_buffers_recovered'])
    check('source_files_unchanged',all(sha256(p)==h for p,h in provenance['source_hashes'].items()))
    check('original_sources_match_A0',not subprocess.check_output(['git','diff',A0,'--','network','tool','train_sshr.py'],cwd=ROOT))
    check('full_step0_identity',identity.get('full_validation_snapshot_bitwise_equal') is True)
    ratios=np.asarray(calibration['ratios'],float); lam=calibration['lambda_value']
    check('calibration_exact32',len(ratios)==32 and np.isfinite(ratios).all() and (ratios>0).all())
    check('calibration_exact_budget',lam==.1*float(np.median(ratios)))
    check('calibration_no_step',calibration['no_optimizer_step'] and calibration['state_unchanged'])
    check('manifest_500',len(manifest['training'])==500 and len(manifest['calibration'])==32)
    check('manifest_first32_exact',manifest['training'][:32]==manifest['calibration'])
    check('manifest_batch20',all(len(row['names'])==len(row['augmentation'])==20 for row in manifest['training']))
    grouped={a:[r for r in curve if r['arm']==a] for a in ('B','A','R')}
    for arm,rows in grouped.items():
        check(f'{arm}_steps',len(rows)==500 and [int(r['step']) for r in rows]==list(range(1,501)))
        check(f'{arm}_finite',all(r['finite']=='True' for r in rows))
        check(f'{arm}_lambda_frozen',all(float(r['lambda_value'])==lam for r in rows))
        check(f'{arm}_lrs',all(tuple(float(r[f'lr{i}']) for i in range(4))==FINAL_LRS for r in rows))
        check(f'{arm}_loss_identity',all(np.isclose(float(r['total_loss']),float(r['main_loss'])+lam*float(r['aux_loss']),rtol=1e-12,atol=1e-12) for r in rows))
    check('random_rate_matches_A_every_step',all(a['active_fraction']==r['active_fraction'] for a,r in zip(grouped['A'],grouped['R'])))
    check('B_no_aux_loss',all(float(r['aux_loss'])==0 for r in grouped['B']))
    state_hashes={}
    for step in (0,250,500):
        targets=['shared'] if step==0 else ['B','A','R']
        for arm in targets:
            path=run_dir/f'checkpoint_step{step:04d}_{arm}.pth'
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            check(f'{step}_{arm}_checkpoint_step',checkpoint['step']==step and checkpoint['global_step']==MAX_STEP+step)
            check(f'{step}_{arm}_checkpoint_lrs',tuple(g['lr'] for g in checkpoint['optimizer']['param_groups'])==FINAL_LRS)
            check(f'{step}_{arm}_checkpoint_momentum',all(g['momentum']==.0005 for g in checkpoint['optimizer']['param_groups']))
            check(f'{step}_{arm}_checkpoint_finite',all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()))
            state_hashes[f'{step}_{arm}']=dict(file_sha256=sha256(path),model_state_sha256=digest(checkpoint['model'].items()))
            if step==0: check('saved_step0_matches_C0',state_hashes['0_shared']['model_state_sha256']==identity['initial_state_sha256'])
            del checkpoint
    check('only_allowed_checkpoint_files',len(list(run_dir.glob('checkpoint_*.pth')))==7)
    snapshot_hashes={}
    reference=None
    for step in (0,50,100,250,500):
        for arm in ('B','A','R'):
            path=run_dir/f'snapshot_{step:04d}_{arm}.npz'; side=json.loads(path.with_suffix('.json').read_text())
            actual=sha256(path); snapshot_hashes[path.name]=actual
            check(f'{step}_{arm}_snapshot_sha',actual==side['snapshot_sha256'])
            check(f'{step}_{arm}_canonical_parity',side['official_cm_parity'] and side['state_unchanged'])
            with np.load(path) as z:
                check(f'{step}_{arm}_size',z['names'].shape==(3418,) and z['official_cm'].shape==(3418,5,5))
                check(f'{step}_{arm}_native_finite',all(np.isfinite(z[k]).all() for k in ('ps','pd','rect','raw_logits','deep_logits','q','delta')))
                if step==0:
                    if arm=='B': reference={k:z[k] for k in z.files}
                    else: check(f'{arm}_initial_array_identity',all(np.array_equal(z[k],v) for k,v in reference.items()))
    if post_analysis:
        required=('training_curve','loss_gradient_dynamics','gate_dynamics','gate_drift','representation_drift',
                  'official_metrics','native28_metrics','deepwin','shallowwin','stablecorrect','rawwrong',
                  'per_class','random_control','gradient_interaction','bootstrap')
        for name in required: check('artifact_'+name,(run_dir/(P+name+'.csv')).is_file())
        summary=read('summary')
        check('primary_step500',summary['primary_step']==500)
        check('bootstrap_10000',summary['settings']['bootstrap_replicates']==10000)
    failures=[name for name,passed in checks.items() if not passed]
    result=dict(passed=not failures,checks=checks,failures=failures,checkpoint_hashes=state_hashes,
                snapshot_hashes=snapshot_hashes,post_analysis=post_analysis)
    write_json(run_dir/(P+'verification.json'),result)
    if failures: raise RuntimeError('Artifact verification failed: '+','.join(failures))
    print(json.dumps(dict(passed=True,checks=len(checks),post_analysis=post_analysis)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--post-analysis',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(4); verify(args.run_dir,args.post_analysis)
