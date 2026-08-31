"""4090 migration smoke: one real batch, B/A/R backwards, ZERO optimizer steps.

This is an engineering check, not calibration, training, or validation inference.
The frozen experiment modules are imported unchanged. No test split is read.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from network.resnet38_cls import Net_CAM
from train_sshr import set_seed
from tools.rddr_phase2b112_common import (
    CKPT_SHA, NATIVE_SHA, bn_digest, digest, gradients_for_batch, make_optimizer,
    move_arm, optimizer_record, restore_rng, rng_state, sha256,
)
from tools.run_rddr_phase2b112 import (
    CHECKPOINT, DATA_ROOT, NATIVE, RecordedTrainDataset, gpu_admission, loader,
)


def run(report):
    set_seed(42)
    report.update(python=platform.python_version(), torch=torch.__version__,
                  cuda_runtime=torch.version.cuda, numpy=np.__version__,
                  cudnn=torch.backends.cudnn.version(),
                  gpu_query=subprocess.check_output([
                      'nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
                      '--format=csv,noheader'], text=True).strip(),
                  packages=sorted((d.metadata['Name'], d.version)
                                  for d in importlib.metadata.distributions()
                                  if d.metadata['Name']))
    report['cuda_available'] = torch.cuda.is_available()
    assert report['cuda_available']
    report['bf16_supported'] = torch.cuda.is_bf16_supported()
    assert report['bf16_supported']
    report['gpu_admission'] = gpu_admission()
    assert report['gpu_admission']['admitted']
    print('CUDA/BF16 available; free-memory admission PASS', flush=True)

    # Exercise CUDA kernels before the full-network check.
    x = torch.randn(16, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    y = (x @ x.T).float().square().mean()
    y.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y) and torch.isfinite(x.grad).all()
    report['bf16_matmul_backward_finite'] = True
    del x, y
    torch.cuda.empty_cache()

    report['checkpoint_sha256'] = sha256(CHECKPOINT)
    report['native_sha256'] = sha256(NATIVE)
    assert report['checkpoint_sha256'] == CKPT_SHA
    assert report['native_sha256'] == NATIVE_SHA
    dataset = RecordedTrainDataset(str(DATA_ROOT/'training'), dataset='bcss', img_size=224)
    images = {p.name for p in (DATA_ROOT/'val/img').glob('*.png')}
    masks = {p.name for p in (DATA_ROOT/'val/mask').glob('*.png')}
    report['parsed_training_count'] = len(dataset)
    report['validation_image_count'] = len(images)
    report['validation_mask_count'] = len(masks)
    report['validation_names_paired'] = images == masks
    assert len(dataset) == 23422 and len(images) == len(masks) == 3418 and images == masks
    batch = next(iter(loader(dataset)))
    report['batch_names'] = list(batch[0])
    report['batch_tensor_sha256'] = digest((('images', batch[1]), ('labels', batch[2])))
    report['batch_shape'] = list(batch[1].shape)
    assert tuple(batch[1].shape) == (20, 3, 224, 224)

    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = Net_CAM(4)
    loaded = model.load_state_dict(state, strict=True)
    report['missing_keys'] = list(loaded.missing_keys)
    report['unexpected_keys'] = list(loaded.unexpected_keys)
    before = digest(model.state_dict().items())
    optimizer = make_optimizer(model)
    move_arm(model, optimizer, 'cuda')
    x, labels = batch[1].cuda(), batch[2].cuda()
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        outputs = model.forward_cam(x)
    torch.cuda.synchronize()
    report['forward_cam'] = [dict(shape=list(v.shape), dtype=str(v.dtype),
                                  finite=bool(torch.isfinite(v).all())) for v in outputs]
    assert all(v['finite'] for v in report['forward_cam'])
    del outputs
    report['optimizer_groups'] = optimizer_record(model, optimizer)
    frozen_rng = rng_state()
    before_bn = bn_digest(model)
    records = []
    a_counts = None
    for arm in ('B', 'A', 'R'):
        restore_rng(frozen_rng)
        started = time.perf_counter()
        # Lambda is zero ONLY in this no-step smoke. It is NOT a calibrated
        # experiment value. Both auxiliary graphs and gradients are still built.
        record, counts = gradients_for_batch(
            model, x, labels, arm, 0.0,
            random_counts=a_counts if arm == 'R' else None,
            random_rng=np.random.default_rng(42) if arm == 'R' else None)
        torch.cuda.synchronize()
        if arm == 'A':
            a_counts = counts.copy()
        record.update(arm=arm, elapsed_s=time.perf_counter()-started,
                      optimizer_steps=0, smoke_lambda=0.0, lambda_calibrated=False)
        records.append(record)
        print(json.dumps(record), flush=True)
        assert not optimizer.state
        assert bn_digest(model) == before_bn
        model.zero_grad(set_to_none=True)
    assert len({r['main_loss'] for r in records}) == 1
    assert records[1]['active_fraction'] == records[2]['active_fraction']
    report['backward_smokes'] = records
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
    move_arm(model, optimizer, 'cpu')
    after = digest(model.state_dict().items())
    assert before == after and not optimizer.state
    report.update(model_state_before=before, model_state_after=after,
                  model_unchanged=True, bn_statistics_unchanged=True,
                  optimizer_buffers_empty=True, checkpoint_file_unchanged=sha256(CHECKPOINT) == CKPT_SHA,
                  status='MIGRATION_CUDA_BF16_BATCH20_PASS')
    assert report['checkpoint_file_unchanged']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', training_started=False, optimizer_steps=0,
                  calibration_run=False, validation_evaluated=False, test_accessed=False,
                  command=[sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        run(report)
    except Exception as e:
        report.update(status='MIGRATION_SMOKE_FAILED', error=repr(e), traceback=traceback.format_exc())
        raise
    finally:
        report['total_elapsed_s'] = time.perf_counter()-started
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write('\n')
        print(json.dumps({k: report[k] for k in ('status', 'optimizer_steps', 'total_elapsed_s')}), flush=True)


if __name__ == '__main__':
    main()
