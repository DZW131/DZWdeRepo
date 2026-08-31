"""Approved Phase2B1.12 helpers. Original network/tool/train sources are unmodified."""
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from tool.torchutils import PolyOptimizer

A0 = '4e9a2887b220d17e27649d72a3d13f32b7ebe8f9'
CKPT_SHA = '509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579'
NATIVE_SHA = '767aa0f97ce2d53db2cc70e4fece0e181a44919f693184c2541490077e07325a'
P = 'rddr_phase2b112_'
MAX_STEP = 29275
FINAL_LRS = tuple(x * (1 - (MAX_STEP - 1) / MAX_STEP) ** .9 for x in (.01, .02, .1, .2))
UPSTREAM = ('b4', 'b4_1', 'b4_2', 'b4_3', 'b4_4', 'b4_5', 'bn45')
SNAPSHOTS = (0, 50, 100, 250, 500)
INTERACTIONS = (0, 50, 250, 500)
Q_EDGES = np.array([.020935675129294395, .072734534740448, .163648784160614, .3369627296924591])
EPS = 1e-8


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def digest(items):
    h = hashlib.sha256()
    for name, value in items:
        v = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.bool_): return bool(x)
    if isinstance(x, Path): return str(x)
    return x


def write_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def upstream_name(name):
    return name.split('.')[0] in UPSTREAM


def js(p, q):
    m = .5 * (p + q)
    return .5 * ((p * ((p + EPS).log() - (m + EPS).log())).sum(1)
                 + (q * ((q + EPS).log() - (m + EPS).log())).sum(1))


@torch.no_grad()
def two_support(source, target):
    b, c, h, w = source.shape
    assert (c, h, w) == (4, 28, 28) and target.shape == source.shape
    nei = F.unfold(source, 15, padding=7).reshape(b, c, 225, h*w)
    valid = F.unfold(torch.ones_like(source[:, :1]), 15, padding=7).reshape(b, 225, h*w).bool()
    valid[:, 112] = False
    a = (1-js(source.flatten(2)[:, :, None], nei)/math.log(2)).clamp(0, 1)
    z = (1-js(target.flatten(2)[:, :, None], nei)/math.log(2)).clamp(0, 1)
    return (a*valid).sum(1)/valid.sum(1), (z*valid).sum(1)/valid.sum(1)


@torch.no_grad()
def adjudicate(ps, pd):
    ps, pd = ps.detach().float(), pd.detach().float()
    tss, tds = two_support(ps, pd); tdd, tsd = two_support(pd, ps)
    delta = .5*(tds+tdd)-.5*(tss+tsd)
    return dict(T_SS=tss, T_SD=tsd, T_DS=tds, T_DD=tdd, delta=delta,
                q=js(ps, pd).flatten(1)/math.log(2), gate=(delta > 0).detach())


def auxiliary_loss(logits, deep, q, gate):
    p = logits.float().softmax(1)
    d = deep.detach().float().reshape_as(p)
    weight = q.detach().float().reshape(p.shape[0], *p.shape[2:]) * gate.detach().reshape(p.shape[0], *p.shape[2:])
    kl = (d*((d+EPS).log()-(p+EPS).log())).sum(1)
    return (weight*kl).sum()/(weight.sum()+EPS)


def random_gate(counts, rng, device='cpu'):
    counts = np.asarray(counts, dtype=np.int64)
    assert counts.ndim == 1 and np.all((counts >= 0) & (counts <= 784))
    gate = np.zeros((len(counts), 784), dtype=bool)
    for i, n in enumerate(counts): gate[i, rng.choice(784, int(n), replace=False)] = True
    assert np.array_equal(gate.sum(1), counts)
    return torch.from_numpy(gate).to(device)


def make_optimizer(model):
    # Must be called BEFORE official model.train() freezes BN/early parameters.
    groups = model.get_parameter_groups()
    params = [dict(params=g, lr=lr, weight_decay=wd) for g, lr, wd in
              zip(groups, (.01, .02, .1, .2), (.0005, 0., .0005, 0.))]
    opt = PolyOptimizer(params, lr=.01, weight_decay=.0005, max_step=MAX_STEP)
    opt.global_step = MAX_STEP
    for g, lr in zip(opt.param_groups, FINAL_LRS): g['lr'] = lr
    assert not opt.state and all(g['momentum'] == .0005 for g in opt.param_groups)
    return opt


def optimizer_record(model, opt):
    names = {id(p): k for k, p in model.named_parameters()}
    return [dict(index=i, lr=g['lr'], momentum=g['momentum'], weight_decay=g['weight_decay'],
                 dampening=g['dampening'], nesterov=g['nesterov'],
                 names=[names[id(p)] for p in g['params']], numel=sum(p.numel() for p in g['params']))
            for i, g in enumerate(opt.param_groups)]


def auxiliary_forward(model, feat56):
    """Independent local graph; leaves share values, never main graph or gradients.

    BN leaves permit auxiliary-only affine gradients without unfreezing the main
    branch. Original optimizer already contains these tensors (created pre-freeze).
    No optimizer step or in-place tensor modification happens in this function.
    """
    leaves = {k: v.detach().requires_grad_(True) for k, v in model.named_parameters() if upstream_name(k)}
    assert len(leaves) == 39
    x = feat56.detach()
    for stage in UPSTREAM:
        module = getattr(model, stage)
        assert all(not m.training for m in module.modules() if isinstance(m, torch.nn.BatchNorm2d))
        state = {k[len(stage)+1:]: p for k, p in leaves.items() if k.startswith(stage+'.')}
        state.update({k: v.detach() for k, v in module.named_buffers()})
        x = functional_call(module, state, (x,), strict=True)
    x = F.relu(x)
    logits = F.conv2d(x, model.ic1.weight.detach(), model.ic1.bias.detach())
    return logits, leaves


def norm(grads):
    terms = [g.detach().double().square().sum() for g in grads if g is not None]
    return float(torch.stack(terms).sum().sqrt().item()) if terms else 0.


def gradients_for_batch(model, x, label, mode, lambda_value, random_counts=None, random_rng=None):
    """One official main backward plus scoped auxiliary backward. Never steps."""
    from network.resnet38_cls import Net
    assert mode in ('B', 'A', 'R') and x.shape == (20, 3, 224, 224)
    model.train(); model.zero_grad(set_to_none=True)
    capture = {}
    # ResBlock overrides __call__, bypassing module hooks; its first BN does not.
    handles = [model.b4.bn_branch2a.register_forward_pre_hook(lambda m, args: capture.update(feat56=args[0].detach())),
               model.hfrm_28_1.register_forward_hook(lambda m, args, out: capture.update(raw=args[0].detach()))]
    try:
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            outputs = Net.forward(model, x)
            main = sum(w*F.multilabel_soft_margin_loss(o, label, weight=None)
                       for w, o in zip((.1, .15, .25, .5), outputs[:4]))
            raw_logits = F.conv2d(capture['raw'], model.ic1.weight.detach(), model.ic1.bias.detach())
        deep = outputs[8].detach().float().softmax(1)
        evidence = adjudicate(raw_logits.detach().float().softmax(1), deep)
        counts = evidence['gate'].sum(1).cpu().numpy()
        assert torch.isfinite(main)
        main.backward()
        del outputs
    finally:
        for h in handles: h.remove()
    original = dict(model.named_parameters())
    approved = {k: p for k, p in original.items() if upstream_name(k)}
    main_norm = norm(p.grad for p in approved.values())
    assert all(p.grad is None for k, p in original.items() if not p.requires_grad)
    aux_value = aux_norm = cosine = 0.
    used_counts = counts
    if mode != 'B':
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            aux_logits, leaves = auxiliary_forward(model, capture['feat56'])
        # This also guards the raw-input/head probe correspondence.
        assert torch.equal(aux_logits.detach(), raw_logits.detach()), 'auxiliary local forward parity'
        gate = evidence['gate'] if mode == 'A' else random_gate(random_counts, random_rng, x.device)
        used_counts = gate.sum(1).cpu().numpy()
        aux = auxiliary_loss(aux_logits, deep, evidence['q'], gate)
        gradients = torch.autograd.grad(aux, tuple(leaves.values()), allow_unused=False)
        aux_norm = norm(gradients); aux_value = float(aux.detach())
        dot = sum((original[k].grad.detach().double()*g.detach().double()).sum()
                  for k, g in zip(leaves, gradients) if original[k].grad is not None)
        cosine = float(dot)/(main_norm*aux_norm+EPS)
        for k, g in zip(leaves, gradients):
            assert torch.isfinite(g).all(), k
            target = original[k]
            addition = g.detach()*lambda_value
            if target.grad is None: target.grad = addition.clone()
            else: target.grad.add_(addition)
    total_norm = norm(p.grad for p in model.parameters())
    finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert finite
    record = dict(main_loss=float(main.detach()), aux_loss=aux_value,
                  weighted_aux_loss=lambda_value*aux_value, total_loss=float(main.detach())+lambda_value*aux_value,
                  main_grad_norm=main_norm, aux_grad_norm=aux_norm,
                  weighted_gradient_ratio=lambda_value*aux_norm/(main_norm+EPS), gradient_cosine=cosine,
                  total_grad_norm=total_norm, active_fraction=float(used_counts.sum()/(20*784)), finite=finite,
                  adjudicated_active_fraction=float(counts.sum()/(20*784)),
                  main_bn_gradients_none=True, aux_parameter_count=39 if mode != 'B' else 0)
    return record, counts


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def move_arm(model, optimizer, device):
    old_names = {id(p): name for name, p in model.named_parameters()}
    group_names = [[old_names[id(p)] for p in g['params']] for g in optimizer.param_groups]
    states = {old_names[id(p)]: state for p, state in optimizer.state.items()}
    model.to(device)
    current = dict(model.named_parameters())
    # .to may replace Parameter objects on some torch versions/devices; preserve
    # optimizer ownership by exact names instead of relying on object identity.
    for group, names in zip(optimizer.param_groups, group_names):
        group['params'] = [current[name] for name in names]
    optimizer.state.clear()
    for name, state in states.items():
        for key, value in state.items():
            if torch.is_tensor(value): state[key] = value.to(device)
        optimizer.state[current[name]] = state


def bn_digest(model):
    return digest((k, v) for k, v in model.state_dict().items() if 'running_' in k or 'num_batches_tracked' in k)
