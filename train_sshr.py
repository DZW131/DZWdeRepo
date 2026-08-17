import os
import math
import hashlib
import numpy as np
import argparse
import importlib
import json
import platform
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
import time
import random
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

cudnn.enabled = True
time_test = 0

# =========================================================================
# 0. PALETTE & UTILS
# =========================================================================
LUAD_PALETTE = [
    205, 51, 51,   # Class 0: Tumor
    0, 255, 0,     # Class 1: Stroma
    65, 105, 225,  # Class 2: Normal
    255, 165, 0,   # Class 3: Necrosis
    255, 255, 255  # Class 4: Background
]
LUAD_PALETTE += [0] * (256 * 3 - len(LUAD_PALETTE))

BCSS_PALETTE = [
    255, 0, 0,     # Class 0
    0, 255, 0,     # Class 1
    0, 0, 255,     # Class 2
    153, 0, 255,   # Class 3
    255, 255, 255  # Class 4: Background
]
BCSS_PALETTE += [0] * (256 * 3 - len(BCSS_PALETTE))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    cudnn.deterministic = False
    try:
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
    except AttributeError:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def get_amp_dtype(args):
    if args.amp_dtype == 'bf16':
        return torch.bfloat16
    if args.amp_dtype == 'fp16':
        return torch.float16
    return None

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_checkpoint_path(args):
    return os.path.join(args.save_folder, args.checkpoint_name)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit():
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_experiment_config(
    args, model, optimizer, train_dataset, max_step, load_result
):
    """Persist a self-contained run manifest without changing training."""
    weights_path = os.path.abspath(args.weights)
    config = {
        'git_commit': _git_commit(),
        'command_args': vars(args),
        'resolved_model_kwargs': get_model_kwargs(args),
        'dataset_size': len(train_dataset),
        'max_step': max_step,
        'model_parameter_count': sum(
            parameter.numel() for parameter in model.parameters()
        ),
        'trainable_parameter_count': sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        'optimizer_param_groups': [
            {
                'index': index,
                'parameter_tensor_count': len(group['params']),
                'parameter_count': sum(
                    parameter.numel() for parameter in group['params']
                ),
                'lr': group['lr'],
                'momentum': group.get('momentum', 0.0),
                'weight_decay': group.get('weight_decay', 0.0),
            }
            for index, group in enumerate(optimizer.param_groups)
        ],
        'pretrained_load': load_result,
        'pretrained_weights': {
            'path': weights_path,
            'size_bytes': (
                os.path.getsize(weights_path)
                if os.path.isfile(weights_path)
                else None
            ),
            'sha256': (
                _sha256_file(weights_path)
                if os.path.isfile(weights_path)
                else None
            ),
        },
        'environment': {
            'python': platform.python_version(),
            'pytorch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    if getattr(args, 'context_mode', 'ch') == 'fampr':
        config['fampr_config'] = model.hfrm_56.fampr_context.config.to_dict()
    if getattr(args, 'rectification_mode', 'uniform') == 'cdsr':
        config['cdsr_config'] = {
            'need_formula': 'R * (1 - (1-D) * (1-U))',
            'alpha_init': 0.10,
            'target_stages': ['stage1', 'stage2', 'stage3'],
            'shared_across_hierarchies': True,
            'new_learnable_scalars': 2,
        }
    path = os.path.join(args.save_folder, 'experiment_config.json')
    with open(path, 'w', encoding='utf-8') as output_file:
        json.dump(config, output_file, indent=2, sort_keys=True, default=str)
        output_file.write('\n')
    print('[ExperimentConfig]', path, flush=True)


def _gradient_record(parameter):
    gradient = parameter.grad
    if gradient is None:
        return {'present': False, 'finite': None, 'norm': None}
    gradient = gradient.detach().float()
    return {
        'present': True,
        'finite': bool(torch.isfinite(gradient).all().item()),
        'norm': gradient.norm().item(),
    }


def collect_fampr_epoch_record(model, diagnostics, epoch):
    """Summarize the first training batch for observability only."""
    stages = {
        'stage1': model.hfrm_56,
        'stage2': model.hfrm_28_1,
        'stage3': model.hfrm_28_2,
    }
    record = {'epoch': epoch, 'sample': 'first_training_batch', 'stages': {}}
    for stage, hfrm in stages.items():
        fampr = hfrm.fampr_context
        stage_diagnostics = diagnostics['fampr'][stage]
        record['stages'][stage] = {
            **stage_diagnostics['summary'],
            'gamma_veto': hfrm.gamma_veto.detach().float().item(),
            'gamma_context': hfrm.gamma_context.detach().float().item(),
            'gamma_veto_gradient': _gradient_record(hfrm.gamma_veto),
            'gamma_context_gradient': _gradient_record(hfrm.gamma_context),
            'anchor_logit_gradient': _gradient_record(fampr.anchor_logit),
            'band_predictor_gradient': _gradient_record(
                fampr.frequency_selector.band_weight_network[-1].weight
            ),
            'base_kernel_gradient': _gradient_record(
                fampr.adaptive_kernel.base_kernel
            ),
            'kernel_gate_gradient': _gradient_record(
                fampr.adaptive_kernel.gate_network[-1].weight
            ),
        }
    return record


def _tensor_summary(tensor):
    values = tensor.detach().float().flatten()
    quantiles = torch.quantile(
        values, torch.tensor([0.10, 0.50, 0.90], device=values.device)
    )
    return {
        'mean': values.mean().item(),
        'std': values.std(unbiased=False).item(),
        'p10': quantiles[0].item(),
        'p50': quantiles[1].item(),
        'p90': quantiles[2].item(),
        'finite': bool(torch.isfinite(values).all().item()),
    }


def collect_cdsr_epoch_record(model, diagnostics, epoch):
    """Summarize the first training batch for observability only."""
    stages = {
        'stage1': model.hfrm_56,
        'stage2': model.hfrm_28_1,
        'stage3': model.hfrm_28_2,
    }
    record = {'epoch': epoch, 'sample': 'first_training_batch', 'stages': {}}
    shared_gate = model.cdsr_selective_gate
    for stage, hfrm in stages.items():
        values = diagnostics['cdsr'][stage]
        need_map = values['need_map'].detach().float()
        record['stages'][stage] = {
            'disagreement': _tensor_summary(values['disagreement']),
            'stage_uncertainty': _tensor_summary(values['stage_uncertainty']),
            'deep_reliability': _tensor_summary(values['deep_reliability']),
            'need': _tensor_summary(need_map),
            'need_bins': {
                'lt_0_25': (need_map < 0.25).float().mean().item(),
                'ge_0_25_lt_0_50': (
                    (need_map >= 0.25) & (need_map < 0.50)
                ).float().mean().item(),
                'ge_0_50_lt_0_75': (
                    (need_map >= 0.50) & (need_map < 0.75)
                ).float().mean().item(),
                'ge_0_75': (need_map >= 0.75).float().mean().item(),
            },
            'alpha_sem': shared_gate.alpha_sem.detach().item(),
            'alpha_ctx': shared_gate.alpha_ctx.detach().item(),
            'gamma_sem': hfrm.gamma_veto.detach().float().item(),
            'gamma_context': hfrm.gamma_context.detach().float().item(),
            'gamma_sem_gradient': _gradient_record(hfrm.gamma_veto),
            'gamma_context_gradient': _gradient_record(hfrm.gamma_context),
            'alpha_sem_logit_gradient': _gradient_record(
                shared_gate.alpha_sem_logit
            ),
            'alpha_ctx_logit_gradient': _gradient_record(
                shared_gate.alpha_ctx_logit
            ),
            'semantic_gate': _tensor_summary(values['semantic_gate']),
            'context_gate': _tensor_summary(values['context_gate']),
            'effective_semantic_rms': values['effective_semantic']
            .detach().float().square().mean().sqrt().item(),
            'effective_context_rms': values['effective_context']
            .detach().float().square().mean().sqrt().item(),
            'all_finite': all(
                bool(torch.isfinite(value).all().item())
                for value in (
                    values['disagreement'],
                    values['stage_uncertainty'],
                    values['deep_reliability'],
                    values['need_map'],
                    values['semantic_gate'],
                    values['context_gate'],
                    values['effective_semantic'],
                    values['effective_context'],
                )
            ),
        }
    return record

def get_model_kwargs(args):
    """Build architecture arguments without changing the official defaults."""
    rectifier_type = getattr(args, 'rectifier', 'hfrm').lower()
    model_kwargs = {'rectifier_type': rectifier_type}
    context_mode = getattr(args, 'context_mode', 'ch').lower()
    if rectifier_type == 'hfrm':
        model_kwargs['context_mode'] = context_mode
        model_kwargs['rectification_mode'] = getattr(
            args, 'rectification_mode', 'uniform'
        ).lower()
    elif context_mode != 'ch':
        raise ValueError(
            "--context-mode=fampr requires --rectifier=hfrm; "
            "archived HST is isolated from FA-MPR"
        )
    if rectifier_type != 'hfrm' and getattr(
        args, 'rectification_mode', 'uniform'
    ).lower() != 'uniform':
        raise ValueError(
            "--rectification-mode=cdsr requires --rectifier=hfrm and "
            "--context-mode=ch"
        )
    if getattr(args, 'rectification_mode', 'uniform').lower() == 'cdsr' and \
            context_mode != 'ch':
        raise ValueError(
            "--rectification-mode=cdsr cannot be combined with FA-MPR"
        )
    if rectifier_type == 'hst':
        variant = getattr(args, 'hst_variant', 'a1').lower()
        transition_enabled = getattr(args, 'hst_transition_enabled', None)
        if transition_enabled is None:
            transition_enabled = variant in {'a2', 'a3'}
        hli_mode = getattr(args, 'hst_hli_mode', None)
        if hli_mode is None:
            hli_mode = 'mlp' if variant == 'a3' else 'identity'
        model_kwargs['hst_config'] = {
            'variant': variant,
            'latent_dim': getattr(args, 'hst_latent_dim', 256),
            'context_kernel': getattr(args, 'hst_context_kernel', 15),
            'transition_enabled': transition_enabled,
            'hli_mode': hli_mode,
        }
    return model_kwargs

def get_infer_thr(args):
    return args.infer_thr if args.infer_thr is not None else None

def get_cam_weights(args):
    return (args.cam_w_28_1, args.cam_w_28_2, args.cam_w_deep)

def get_loss_weights(args):
    return (args.loss_w_56, args.loss_w_28_1, args.loss_w_28_2, args.loss_w_deep)

def apply_palette(mask_np, dataset='luad'):
    mask_img = Image.fromarray(mask_np.astype(np.uint8))
    if dataset == 'bcss':
        mask_img.putpalette(BCSS_PALETTE)
    else:
        mask_img.putpalette(LUAD_PALETTE)
    return mask_img.convert('RGB')

def overlay_heatmap(img_np, cam_np):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)

def norm_np(cam_np):
    c_min = np.min(cam_np, axis=(1, 2), keepdims=True)
    c_max = np.max(cam_np, axis=(1, 2), keepdims=True)
    return (cam_np - c_min) / (c_max - c_min + 1e-8)

def spatial_normalize(cam_tensor):
    B_sz, C_sz, H_sz, W_sz = cam_tensor.shape
    cam_flat = cam_tensor.view(B_sz, C_sz, -1)
    c_min = cam_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    c_max = cam_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    denominator = (c_max - c_min).clamp_min(1e-5)
    return (cam_tensor - c_min) / denominator




class ClassificationEvalDataset(Dataset):
    def __init__(self, data_root, transform=None, img_size=224, n_class=4, dino_version="dino_v2"):
        self.img_dir = os.path.join(data_root, 'img')
        self.mask_dir = os.path.join(data_root, 'mask')
        self.bg_mask_dir = os.path.join(os.path.dirname(data_root.rstrip('/')), 'bg_mask_test')
        self.dino_feat_dir = os.path.join(os.path.dirname(data_root.rstrip('/')), f'test_feats_{dino_version}_{img_size}')
        
        self.img_size = img_size
        self.n_class = n_class
        self.transform = transform 
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if not f.startswith('.')]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(os.path.join(self.img_dir, img_id + '.png')).convert('RGB')
        
        mask_np = np.array(Image.open(os.path.join(self.mask_dir, img_id + '.png')))
        
        bg_mask = Image.open(os.path.join(self.bg_mask_dir, img_id + '.png')).convert('L') if os.path.exists(os.path.join(self.bg_mask_dir, img_id + '.png')) else Image.new('L', (self.img_size, self.img_size), 0)
        
        feat_path = os.path.join(self.dino_feat_dir, img_id + '.pt')
        dino_feat = torch.load(feat_path) if os.path.exists(feat_path) else torch.zeros((196, 384))
        if len(dino_feat.shape) == 3: dino_feat = dino_feat.view(-1, dino_feat.shape[-1])

        if img.size[0] != self.img_size or img.size[1] != self.img_size:
            img = TF.resize(img, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR)
            bg_mask = TF.resize(bg_mask, [self.img_size, self.img_size], interpolation=InterpolationMode.NEAREST)
            
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        bg_mask_tensor = (torch.from_numpy(np.array(bg_mask)).long() > 128).float()
        
        label_tensor = torch.zeros(self.n_class)
        for c in np.unique(mask_np):
            if c < self.n_class: label_tensor[c] = 1.0

        return img, label_tensor, bg_mask_tensor, dino_feat

    def __len__(self): return len(self.ids)

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = len(set(pred_labels) & set(gt_labels))
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    return round(pred_correct_count / union, 4) if union > 0 else 1.0


def train_phase(args):
    global time_test

    set_seed(args.seed)
    os.makedirs(args.save_folder, exist_ok=True)
    model = getattr(importlib.import_module(args.network), 'Net')(
        n_class=args.n_class, **get_model_kwargs(args)
    ).cuda()

    loss_weights = None
    amp_dtype = get_amp_dtype(args)
    use_amp = amp_dtype is not None
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp_dtype == 'fp16'))
    
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_eval = transforms.Compose([transforms.ToTensor()])

    train_dataset = Stage1_TrainDataset(data_path=args.trainroot, transform=transform_train, dataset=args.dataset, img_size=args.img_size)
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=data_generator,
    )
    


    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    
    param_groups = model.get_parameter_groups()
    optim_params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ]

    optimizer = torchutils.PolyOptimizer(optim_params, lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    
    if args.weights[-7:] == '.params':
        weights_dict = importlib.import_module('network.resnet38d').convert_mxnet_to_torch(args.weights)
        incompatible = model.load_state_dict(weights_dict, strict=False)
    elif args.weights[-4:] == '.pth':
        incompatible = model.load_state_dict(torch.load(args.weights), strict=False)
    else:
        raise ValueError(f'Unsupported weights file: {args.weights}')

    load_result = {
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
    }
    write_experiment_config(
        args, model, optimizer, train_dataset, max_step, load_result
    )
        
    avg_meter = pyutils.AverageMeter('loss_cls', 'loss_adapt', 'avg_ep_EM', 'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    best_val_miou = None
    eval_history = []
    os.makedirs(args.save_folder, exist_ok=True)

    for ep in range(args.max_epoches):
        model.train()
        ep_count = ep_EM = ep_acc = 0
        fampr_epoch_record = None
        cdsr_epoch_record = None

        
        for iter, (filename, img, label) in enumerate(train_data_loader):
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            collect_diagnostics = (
                (
                    getattr(args, 'context_mode', 'ch') == 'fampr'
                    or getattr(args, 'rectification_mode', 'uniform') == 'cdsr'
                )
                and iter == 0
            )
            
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                if collect_diagnostics:
                    model_outputs, model_diagnostics = \
                        model.forward_with_diagnostics(img)
                else:
                    model_outputs = model(img)
                    model_diagnostics = None
                out_56, out_28_1, out_28_2, out_deep, y_deep, cam_56, cam_28_1, cam_28_2, cam_deep, feat_56 = model_outputs
                loss_w_56, loss_w_28_1, loss_w_28_2, loss_w_deep = get_loss_weights(args)
                loss_cls = (
                    loss_w_56 * F.multilabel_soft_margin_loss(out_56, label, weight=loss_weights)
                    + loss_w_28_1 * F.multilabel_soft_margin_loss(out_28_1, label, weight=loss_weights)
                    + loss_w_28_2 * F.multilabel_soft_margin_loss(out_28_2, label, weight=loss_weights)
                    + loss_w_deep * F.multilabel_soft_margin_loss(out_deep, label, weight=loss_weights)
                )
                loss = loss_cls
            
            loss_adapt_val = torch.zeros((), device=img.device)
            
            # Metrics
            prob = y_deep.detach().float().cpu().numpy()
            gt = label.detach().float().cpu().numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > args.train_cls_thr)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls): ep_EM += 1
                ep_acc += compute_acc(pass_cls, true_cls)
            
            avg_meter.add({'loss_cls': loss_cls.item(), 'loss_adapt': loss_adapt_val.item()})
            
            optimizer.zero_grad()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if collect_diagnostics:
                    if getattr(args, 'context_mode', 'ch') == 'fampr':
                        fampr_epoch_record = collect_fampr_epoch_record(
                            model, model_diagnostics, ep + 1
                        )
                    if getattr(args, 'rectification_mode', 'uniform') == 'cdsr':
                        cdsr_epoch_record = collect_cdsr_epoch_record(
                            model, model_diagnostics, ep + 1
                        )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if collect_diagnostics:
                    if getattr(args, 'context_mode', 'ch') == 'fampr':
                        fampr_epoch_record = collect_fampr_epoch_record(
                            model, model_diagnostics, ep + 1
                        )
                    if getattr(args, 'rectification_mode', 'uniform') == 'cdsr':
                        cdsr_epoch_record = collect_cdsr_epoch_record(
                            model, model_diagnostics, ep + 1
                        )
                optimizer.step()

            if (optimizer.global_step) % 100 == 0 and (optimizer.global_step) != 0:
                timer.update_progress(optimizer.global_step / max_step)
                print('Epoch:%2d' % ep,
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'L_Cls:%.4f' % avg_meter.get('loss_cls'),
                      'L_Adpt:%.4f' % avg_meter.get('loss_adapt'),
                      'Tr_EM:%.4f' % round(ep_EM/ep_count, 4),
                      'Tr_Acc:%.4f' % round(ep_acc/ep_count, 4),
                      'lr: %.4f' % optimizer.param_groups[0]['lr'],
                      'Fin:%s' % timer.str_est_finish(), flush=True)


        if fampr_epoch_record is not None:
            diagnostics_path = os.path.join(
                args.save_folder, 'fampr_diagnostics.jsonl'
            )
            with open(diagnostics_path, 'a', encoding='utf-8') as output_file:
                output_file.write(
                    json.dumps(fampr_epoch_record, sort_keys=True) + '\n'
                )
            print(
                '[FAMPRDiagnostics]',
                json.dumps(fampr_epoch_record, sort_keys=True),
                flush=True,
            )

        if cdsr_epoch_record is not None:
            diagnostics_path = os.path.join(
                args.save_folder, 'cdsr_diagnostics.jsonl'
            )
            with open(diagnostics_path, 'a', encoding='utf-8') as output_file:
                output_file.write(
                    json.dumps(cdsr_epoch_record, sort_keys=True) + '\n'
                )
            print(
                '[CDSRDiagnostics]',
                json.dumps(cdsr_epoch_record, sort_keys=True),
                flush=True,
            )

        checkpoint_path = get_checkpoint_path(args)
        if args.save_checkpoints:
            torch.save(model.state_dict(), checkpoint_path)
        if args.save_last_k_checkpoints > 0 and (ep + 1) > args.max_epoches - args.save_last_k_checkpoints:
            epoch_checkpoint_path = os.path.join(args.save_folder, f"stage1_epoch_{ep + 1:04d}.pth")
            torch.save(model.state_dict(), epoch_checkpoint_path)
        do_eval = args.eval_every > 0 and ((ep + 1) % args.eval_every == 0)
        if do_eval:
            state_dict = None if args.save_checkpoints else model.state_dict()
            val_score = test_phase(args, dataroot=args.valroot, split_name='val', checkpoint_path=checkpoint_path, state_dict=state_dict)
            test_score = test_phase(args, dataroot=args.testroot, split_name='test', checkpoint_path=checkpoint_path, state_dict=state_dict)
            val_miou = val_score.get('Mean IoU') if val_score is not None else None
            test_miou = test_score.get('Mean IoU') if test_score is not None else None
            eval_record = {
                'epoch': ep + 1,
                'val_mean_iou': val_miou,
                'test_mean_iou': test_miou,
                'val_mean_dice': val_score.get('Mean Dice') if val_score is not None else None,
                'test_mean_dice': test_score.get('Mean Dice') if test_score is not None else None,
            }
            eval_history.append(eval_record)
            print('[Eval]', json.dumps(eval_record, sort_keys=True), flush=True)
            if val_miou is not None and (best_val_miou is None or val_miou > best_val_miou):
                best_val_miou = val_miou

    return {'best_val_miou': best_val_miou, 'eval_history': eval_history}


def test_phase(args, dataroot=None, split_name='test', checkpoint_path=None, state_dict=None):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(
        n_class=args.n_class, **get_model_kwargs(args)
    )
    model = model.cuda()
    if dataroot is None:
        dataroot = args.testroot
    if state_dict is None:
        if checkpoint_path is None:
            checkpoint_path = get_checkpoint_path(args)
        weights_dict = torch.load(checkpoint_path)
    else:
        weights_dict = state_dict
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    score = infer(model, dataroot, args.n_class, args, thr=get_infer_thr(args), cam_weights=get_cam_weights(args))
    print(f'[{split_name}] {score}', flush=True)
    return score

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--max_epoches", default=24, type=int)
    parser.add_argument("--network", default="network.resnet38_cls", type=str)
    parser.add_argument("--rectifier", default="hfrm", choices=["hfrm", "hst"])
    parser.add_argument(
        "--context_mode", "--context-mode",
        default="ch", choices=["ch", "fampr"],
        help=(
            "Contextual branch for HFRM. 'ch' is the exact official SSHR "
            "default; 'fampr' enables Full FA-MPR."
        ),
    )
    parser.add_argument(
        "--rectification_mode", "--rectification-mode",
        default="uniform", choices=["uniform", "cdsr"],
        help=(
            "Residual application mode. 'uniform' preserves exact SSHR; "
            "'cdsr' enables detached analytical selective rectification."
        ),
    )
    parser.add_argument(
        "--hst_variant", "--hst-variant",
        default="a1", choices=["a1", "a2", "a3"],
    )
    parser.add_argument("--hst_latent_dim", "--hst-latent-dim", default=256, type=int)
    parser.add_argument(
        "--hst_context_kernel", "--hst-context-kernel", default=15, type=int
    )
    parser.add_argument(
        "--hst_transition_enabled", "--hst-transition-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the transition default (A1=false, A2/A3=true).",
    )
    parser.add_argument(
        "--hst_hli_mode", "--hst-hli-mode",
        default=None,
        choices=["identity", "mlp"],
        help="Override the HLI default (A1/A2=identity, A3=mlp).",
    )
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--n_class", default=4, type=int)
    parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    parser.add_argument("--trainroot", default='datasets/LUAD-HistoSeg/training/', type=str)
    parser.add_argument("--testroot", default='datasets/LUAD-HistoSeg/test/', type=str)
    parser.add_argument("--valroot", default='datasets/LUAD-HistoSeg/val/', type=str)

    parser.add_argument("--dataset", default='luad', type=str)
    parser.add_argument("--img_size", default=224, type=int)

    parser.add_argument("--save_folder", default='checkpoints', type=str)
    parser.add_argument("--checkpoint_name", default='stage1_last.pth', type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--eval_every", default=1, type=int)
    parser.add_argument("--evaluate_only", "--evaluate-only", action="store_true")
    parser.add_argument("--infer_thr", default=None, type=float)
    parser.add_argument("--cam_w_28_1", default=0.6, type=float)
    parser.add_argument("--cam_w_28_2", default=0.2, type=float)
    parser.add_argument("--cam_w_deep", default=0.2, type=float)
    parser.add_argument("--loss_w_56", default=0.1, type=float)
    parser.add_argument("--loss_w_28_1", default=0.15, type=float)
    parser.add_argument("--loss_w_28_2", default=0.25, type=float)
    parser.add_argument("--loss_w_deep", default=0.5, type=float)
    parser.add_argument("--train_cls_thr", default=0.2, type=float)
    parser.add_argument("--amp_dtype", "--amp-dtype", default="bf16", choices=["none", "bf16", "fp16"], type=str)
    parser.add_argument("--save_checkpoints", "--save-checkpoints", dest="save_checkpoints", action="store_true", default=True)
    parser.add_argument("--no-save_checkpoints", "--no-save-checkpoints", dest="save_checkpoints", action="store_false")
    parser.add_argument("--save_last_k_checkpoints", "--save-last-k-checkpoints", default=5, type=int)
    args = parser.parse_args()
    if args.evaluate_only:
        set_seed(args.seed)
        test_phase(args, dataroot=args.testroot, split_name="test_tta", checkpoint_path=args.weights)
    else:
        os.makedirs(args.save_folder, exist_ok=True)
        start_time = time.time()
        train_phase(args)
        print(f"Total Training Time: {time.time() - start_time - time_test:.2f}s")
