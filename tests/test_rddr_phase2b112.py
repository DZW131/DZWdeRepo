"""Independent Phase2B1.12 tests; CPU fixtures are not a formal CUDA run.

Run: python -m unittest discover -s tests -p test_rddr_phase2b112.py -v
The thin network uses the original ResBlock/BN/HFRM implementations, preserving
the 39-tensor auxiliary topology without allocating the full training network.
Full-model batch20/BF16/checkpoint/data identity remains an artifact/runtime gate.
"""
import ast
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import rddr_phase2b112_common as common
from tools import analyze_rddr_phase2b112 as statistics
from tools import rddr_phase2b112_evaluation as evaluation
from network.resnet38_cls import HFRM, Net
from network.resnet38d import Net as Backbone, ResBlock
from tool.torchutils import PolyOptimizer

torch.set_num_threads(min(2, torch.get_num_threads()))


class GroupedParameters(nn.Module):
    def __init__(self):
        super().__init__()
        self.values = nn.ParameterList([
            nn.Parameter(torch.tensor([1.25, -0.75], dtype=torch.float64))
            for _ in range(4)
        ])

    def get_parameter_groups(self):
        return tuple([p] for p in self.values)


class ThinNet(Backbone):
    """Original local graph and official freeze implementation, small channels."""
    get_parameter_groups = Net.get_parameter_groups

    def __init__(self):
        nn.Module.__init__(self)
        self.conv1a = nn.Conv2d(3, 2, 1, bias=False)
        self.b3 = nn.Conv2d(2, 2, 1, bias=False)
        self.b4 = ResBlock(2, 3, 3, stride=2)
        for name in common.UPSTREAM[1:-1]:
            setattr(self, name, ResBlock(3, 3, 3))
        self.bn45 = nn.BatchNorm2d(3)
        self.b5 = nn.Conv2d(3, 8, 1, bias=False)
        self.hfrm_28_1 = HFRM(3, deep_channels=8, context_kernel=15)
        self.ic_56 = nn.Conv2d(2, 4, 1)
        self.ic1 = nn.Conv2d(3, 4, 1)
        self.ic2 = nn.Conv2d(8, 4, 1)
        self.fc8 = nn.Conv2d(8, 4, 1, bias=False)
        self.dropout7 = nn.Dropout2d(.5)
        self.not_training = [self.conv1a]
        self.from_scratch_layers = [self.hfrm_28_1, self.ic_56, self.ic1, self.ic2, self.fc8]


def thin_forward(model, x):
    feat56 = model.b3(model.conv1a(F.avg_pool2d(x, 4)))
    raw = feat56
    for name in common.UPSTREAM:
        raw = getattr(model, name)(raw)
    raw = F.relu(raw)
    deep = F.relu(model.b5(raw))
    rect = model.hfrm_28_1(raw, deep)
    cams = [model.ic_56(feat56), model.ic1(rect), model.ic2(deep),
            model.fc8(model.dropout7(deep))]
    pooled = [cam.mean((2, 3)) for cam in cams]
    model.probe_feat56 = feat56.detach()
    model.probe_raw = raw.detach()
    model.probe_deep = cams[3].detach()
    return (*pooled, pooled[3].sigmoid(), *cams, feat56)


def official_main(model, x, labels):
    outputs = thin_forward(model, x)
    return sum(w * F.multilabel_soft_margin_loss(o, labels, weight=None)
               for w, o in zip((.1, .15, .25, .5), outputs[:4]))


def manual_js(p, q):
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    middle = (p + q) / 2
    return .5 * np.sum(p * (np.log(p + 1e-8) - np.log(middle + 1e-8))
                       + q * (np.log(q + 1e-8) - np.log(middle + 1e-8)), axis=0)


class FormulaTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_adt_formula(self):
        logits = torch.randn(2, 4, 3, 5, requires_grad=True)
        deep = torch.randn_like(logits).softmax(1).requires_grad_()
        q = torch.rand(2, 15, requires_grad=True)
        gate = torch.tensor([[1, 0, 0, 1, 1] * 3, [0, 1, 0, 1, 0] * 3], dtype=torch.float32,
                            requires_grad=True)
        actual = common.auxiliary_loss(logits, deep, q, gate)
        ps, pd = logits.detach().softmax(1).numpy(), deep.detach().numpy()
        weights = q.detach().numpy().reshape(2, 3, 5) * gate.detach().numpy().reshape(2, 3, 5)
        expected = (weights * (pd * (np.log(pd + 1e-8) - np.log(ps + 1e-8))).sum(1)).sum()
        expected /= weights.sum() + 1e-8
        self.assertAlmostEqual(float(actual.detach()), float(expected), places=6)
        actual.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertIsNone(deep.grad)
        self.assertIsNone(q.grad)
        self.assertIsNone(gate.grad)

    def test_aux_deep_detached(self):
        self.test_adt_formula()

    def test_aux_q_detached(self):
        self.test_adt_formula()

    def test_aux_gate_detached(self):
        self.test_adt_formula()

    def test_zero_gate_has_finite_zero_loss_and_zero_gradient(self):
        logits = torch.randn(2, 4, 3, 5, requires_grad=True)
        loss = common.auxiliary_loss(logits, torch.randn_like(logits).softmax(1),
                                     torch.rand(2, 15), torch.zeros(2, 15))
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_unselected_pixels_have_no_auxiliary_gradient(self):
        logits = torch.randn(1, 4, 2, 2, requires_grad=True)
        common.auxiliary_loss(logits, torch.randn_like(logits).softmax(1),
                              torch.ones(1, 4), torch.tensor([[1, 0, 0, 0]])).backward()
        self.assertTrue(torch.equal(logits.grad[:, :, 1, :], torch.zeros_like(logits.grad[:, :, 1, :])))
        self.assertTrue(torch.equal(logits.grad[:, :, 0, 1], torch.zeros_like(logits.grad[:, :, 0, 1])))

    def test_q_is_normalized_js_not_direction(self):
        ps, pd = torch.zeros(1, 4, 28, 28), torch.zeros(1, 4, 28, 28)
        ps[:, 0], pd[:, 1] = 1, 1
        result = common.adjudicate(ps, pd)
        torch.testing.assert_close(result['q'], torch.ones(1, 784), rtol=0, atol=1e-6)
        torch.testing.assert_close(result['delta'], torch.zeros(1, 784), rtol=0, atol=1e-6)
        self.assertFalse(result['gate'].any())

    def test_aux_delta_detached(self):
        ps = torch.randn(1, 4, 28, 28).softmax(1).requires_grad_()
        pd = torch.randn_like(ps).softmax(1).requires_grad_()
        result = common.adjudicate(ps, pd)
        for name, value in result.items():
            self.assertFalse(value.requires_grad, name)
            self.assertIsNone(value.grad_fn, name)
        self.assertTrue(torch.equal(result['gate'], result['delta'] > 0))

    def test_frozen_support_manual_oracle_at_corners_and_center(self):
        ps = torch.randn(1, 4, 28, 28).softmax(1)
        pd = torch.randn(1, 4, 28, 28).softmax(1)
        result = common.adjudicate(ps, pd)
        psn, pdn = ps[0].numpy(), pd[0].numpy()
        for row, col in [(0, 0), (0, 27), (7, 7), (14, 14), (27, 27)]:
            neighbors = [(r, c) for r in range(max(0, row-7), min(28, row+8))
                         for c in range(max(0, col-7), min(28, col+8)) if (r, c) != (row, col)]
            self.assertEqual(len(neighbors), 63 if (row, col) in [(0, 0), (0, 27), (27, 27)] else 224)
            for key, target, source in [('T_SS', psn, psn), ('T_SD', psn, pdn),
                                        ('T_DS', pdn, psn), ('T_DD', pdn, pdn)]:
                expected = np.mean([np.clip(1-manual_js(target[:, row, col], source[:, r, c]) / math.log(2), 0, 1)
                                    for r, c in neighbors])
                self.assertAlmostEqual(float(result[key][0, row*28+col]), float(expected), places=6, msg=key)

    def test_swap_symmetry_and_ties_excluded(self):
        ps, pd = torch.randn(1, 4, 28, 28).softmax(1), torch.randn(1, 4, 28, 28).softmax(1)
        original, swapped = common.adjudicate(ps, pd), common.adjudicate(pd, ps)
        torch.testing.assert_close(original['q'], swapped['q'], rtol=0, atol=0)
        torch.testing.assert_close(original['delta'], -swapped['delta'], rtol=0, atol=0)
        tied = common.adjudicate(ps, ps)
        self.assertFalse(tied['gate'].any())
        self.assertTrue(torch.equal(tied['q'], torch.zeros_like(tied['q'])))


class RandomGateTests(unittest.TestCase):
    def test_random_gate_rate_match(self):
        counts = np.array([0, 1, 13, 392, 783, 784])
        gate = common.random_gate(counts, np.random.default_rng(42))
        self.assertEqual(gate.shape, (6, 784))
        self.assertEqual(gate.dtype, torch.bool)
        np.testing.assert_array_equal(gate.sum(1).numpy(), counts)

    def test_random_gate_seed42(self):
        counts = np.array([11, 45, 345])
        first, second = np.random.default_rng(42), np.random.default_rng(42)
        a = common.random_gate(counts, first)
        self.assertTrue(torch.equal(a, common.random_gate(counts, second)))
        next_draw = common.random_gate(counts, first)
        self.assertFalse(torch.equal(a, next_draw))
        self.assertTrue(torch.equal(next_draw, common.random_gate(counts, second)))

    def test_random_gate_rng_is_independent(self):
        np.random.seed(42)
        torch.manual_seed(42)
        before_np, before_torch = np.random.get_state(), torch.get_rng_state().clone()
        common.random_gate([25, 70], np.random.default_rng(42))
        after_np = np.random.get_state()
        self.assertEqual(before_np[0], after_np[0])
        np.testing.assert_array_equal(before_np[1], after_np[1])
        self.assertEqual(before_np[2:], after_np[2:])
        self.assertTrue(torch.equal(before_torch, torch.get_rng_state()))

    def test_random_gate_rejects_invalid_count(self):
        for counts in [[-1], [785], [[2, 3]]]:
            with self.subTest(counts=counts), self.assertRaises(AssertionError):
                common.random_gate(counts, np.random.default_rng(42))


class OptimizerTests(unittest.TestCase):
    def test_optimizer_provenance_exact(self):
        model = GroupedParameters()
        optimizer = common.make_optimizer(model)
        self.assertIsInstance(optimizer, PolyOptimizer)
        self.assertEqual(optimizer.global_step, 29275)
        self.assertEqual(optimizer.max_step, 29275)
        self.assertEqual(optimizer.lr_power, .9)
        self.assertEqual(len(optimizer.state), 0)
        expected_lrs = [9.55328615544644e-7, 1.910657231089288e-6,
                        9.55328615544644e-6, 1.910657231089288e-5]
        np.testing.assert_allclose([g['lr'] for g in optimizer.param_groups], expected_lrs, rtol=1e-15)
        self.assertEqual([g['weight_decay'] for g in optimizer.param_groups], [.0005, 0., .0005, 0.])
        for group in optimizer.param_groups:
            self.assertEqual(group['momentum'], .0005)
            self.assertEqual(group['dampening'], 0)
            self.assertFalse(group['nesterov'])

    def test_last_used_lr_reconstructed_from_official_scheduler(self):
        model = GroupedParameters()
        params = [dict(params=g, lr=lr, weight_decay=wd) for g, lr, wd in zip(
            model.get_parameter_groups(), (.01, .02, .1, .2), (.0005, 0., .0005, 0.))]
        original = PolyOptimizer(params, lr=.01, weight_decay=.0005, max_step=29275)
        original.global_step = 29274
        original.step()
        self.assertEqual(tuple(g['lr'] for g in original.param_groups), common.FINAL_LRS)
        for _ in range(500):
            original.step()
            self.assertEqual(tuple(g['lr'] for g in original.param_groups), common.FINAL_LRS)
        self.assertEqual(original.global_step, 29775)

    def test_tail_updates_match_manual_sgd_weight_decay_and_momentum(self):
        model = GroupedParameters()
        optimizer = common.make_optimizer(model)
        expected = [p.detach().clone() for p in model.values]
        buffers = [None] * 4
        for step in range(3):
            for index, p in enumerate(model.values):
                p.grad = torch.tensor([.2 + step, -.3 - index], dtype=p.dtype)
                d = p.grad + optimizer.param_groups[index]['weight_decay'] * expected[index]
                buffers[index] = d.clone() if buffers[index] is None else .0005 * buffers[index] + d
                expected[index] -= common.FINAL_LRS[index] * buffers[index]
            optimizer.step()
            for index, p in enumerate(model.values):
                torch.testing.assert_close(p, expected[index], rtol=0, atol=2e-16)
            self.assertEqual(tuple(g['lr'] for g in optimizer.param_groups), common.FINAL_LRS)

    def test_three_arms_fresh_identical_optimizer_state(self):
        models = [GroupedParameters() for _ in range(3)]
        optimizers = [common.make_optimizer(model) for model in models]
        self.assertEqual(optimizers[0].state_dict(), optimizers[1].state_dict())
        self.assertEqual(optimizers[0].state_dict(), optimizers[2].state_dict())


class GradientScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.base = ThinNet()
        cls.x = torch.randn(20, 3, 224, 224)
        cls.labels = torch.randint(0, 2, (20, 4)).float()

    def local_aux(self):
        model = copy.deepcopy(self.base)
        model.train()
        feat56 = torch.randn(1, 2, 56, 56, requires_grad=True)
        before = common.digest(model.state_dict().items())
        logits, leaves = common.auxiliary_forward(model, feat56)
        deep = torch.randn_like(logits).softmax(1).requires_grad_()
        loss = common.auxiliary_loss(logits, deep, torch.ones(1, 784), torch.ones(1, 784))
        loss.backward()
        return model, feat56, logits, leaves, deep, before

    def run_arm(self, mode, strength=.2, counts=None):
        model = copy.deepcopy(self.base)
        torch.manual_seed(42)
        with patch.object(Net, 'forward', thin_forward):
            record, own_counts = common.gradients_for_batch(
                model, self.x, self.labels, mode, strength, random_counts=counts,
                random_rng=np.random.default_rng(42) if mode == 'R' else None)
        return model, record, own_counts

    def test_three_arms_bitwise_equal_step0(self):
        arms = [copy.deepcopy(self.base) for _ in range(3)]
        hashes = [common.digest(model.state_dict().items()) for model in arms]
        self.assertEqual(len(set(hashes)), 1)
        predictions = []
        for model in arms:
            model.train()
            torch.manual_seed(42)
            with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
                predictions.append(thin_forward(model, self.x)[8])
        self.assertTrue(torch.equal(predictions[0], predictions[1]))
        self.assertTrue(torch.equal(predictions[0], predictions[2]))

    def test_aux_feat56_detached(self):
        _, feat56, _, _, _, _ = self.local_aux()
        self.assertIsNone(feat56.grad)

    def test_aux_ic1_detached(self):
        model, _, _, _, _, _ = self.local_aux()
        self.assertIsNone(model.ic1.weight.grad)
        self.assertIsNone(model.ic1.bias.grad)

    def test_aux_only_b4_bn45_extra_grad(self):
        model, _, _, leaves, deep, before = self.local_aux()
        expected = {name for name, _ in model.named_parameters() if common.upstream_name(name)}
        self.assertEqual(set(leaves), expected)
        self.assertEqual(len(leaves), 39)
        self.assertTrue(all(leaf.is_leaf and leaf.grad is not None for leaf in leaves.values()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertIsNone(deep.grad)
        self.assertEqual(common.digest(model.state_dict().items()), before)

    def test_aux_forward_matches_original_local_subgraph(self):
        model = copy.deepcopy(self.base)
        model.train()
        feature = torch.randn(1, 2, 56, 56)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            expected = feature
            for stage in common.UPSTREAM:
                expected = getattr(model, stage)(expected)
            expected = F.conv2d(F.relu(expected), model.ic1.weight.detach(), model.ic1.bias.detach())
            actual, _ = common.auxiliary_forward(model, feature)
        self.assertTrue(torch.equal(actual, expected))

    def test_original_resblock_bypasses_block_hook_but_bn_hook_runs(self):
        block = ResBlock(2, 3, 3, stride=2)
        captured = {}
        handles = [block.register_forward_pre_hook(lambda m, args: captured.update(block=True)),
                   block.bn_branch2a.register_forward_pre_hook(lambda m, args: captured.update(bn=args[0]))]
        x = torch.randn(1, 2, 56, 56)
        try:
            block(x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertNotIn('block', captured)
        self.assertIs(captured['bn'], x)

    def test_main_loss_gradient_path_unchanged(self):
        model, record, _ = self.run_arm('B')
        reference = copy.deepcopy(self.base)
        reference.train()
        reference.zero_grad(set_to_none=True)
        torch.manual_seed(42)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = official_main(reference, self.x, self.labels)
        loss.backward()
        self.assertEqual(record['main_loss'], float(loss.detach()))
        for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
            self.assertEqual(actual.grad is None, expected.grad is None, name)
            if expected.grad is not None:
                self.assertTrue(torch.equal(actual.grad, expected.grad), name)
        self.assertEqual(record['aux_parameter_count'], 0)

    def test_aux_extra_gradient_scope_and_no_optimizer_step(self):
        baseline, _, _ = self.run_arm('B')
        treated, record, _ = self.run_arm('A')
        originals = dict(baseline.named_parameters())
        changed = []
        for name, actual in treated.named_parameters():
            expected = originals[name]
            if not common.upstream_name(name):
                self.assertEqual(actual.grad is None, expected.grad is None, name)
                if expected.grad is not None:
                    self.assertTrue(torch.equal(actual.grad, expected.grad), name)
            elif expected.grad is None or not torch.equal(actual.grad, expected.grad):
                changed.append(name)
        self.assertTrue(changed)
        self.assertEqual(record['aux_parameter_count'], 39)
        self.assertEqual(common.digest(treated.state_dict().items()), common.digest(self.base.state_dict().items()))
        self.assertEqual(common.bn_digest(baseline), common.bn_digest(treated))

    def test_main_bn_frozen_aux_bn_affine_allowed(self):
        baseline, _, _ = self.run_arm('B')
        treated, _, _ = self.run_arm('A')
        for model in [baseline, treated]:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    self.assertFalse(module.training)
                    self.assertFalse(module.weight.requires_grad)
                    self.assertFalse(module.bias.requires_grad)
        for module in baseline.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.assertIsNone(module.weight.grad)
                self.assertIsNone(module.bias.grad)
        for name, module in treated.named_modules():
            if isinstance(module, nn.BatchNorm2d) and common.upstream_name(name):
                self.assertIsNotNone(module.weight.grad, name)
                self.assertIsNotNone(module.bias.grad, name)
        self.assertEqual(common.bn_digest(treated), common.bn_digest(self.base))

    def test_random_arm_uses_supplied_A_counts_not_own_adjudication(self):
        requested = np.arange(20)
        _, record, own = self.run_arm('R', counts=requested)
        self.assertAlmostEqual(record['active_fraction'], requested.sum() / (20*784))
        self.assertFalse(np.array_equal(own, requested))
        self.assertAlmostEqual(record['adjudicated_active_fraction'], own.sum() / (20*784))

    def test_same_main_network_rng_across_arms(self):
        baseline, b_record, _ = self.run_arm('B')
        treated, a_record, counts = self.run_arm('A')
        control, r_record, _ = self.run_arm('R', counts=counts)
        self.assertEqual(b_record['main_loss'], a_record['main_loss'])
        self.assertEqual(a_record['main_loss'], r_record['main_loss'])
        self.assertTrue(torch.equal(baseline.probe_deep, treated.probe_deep))
        self.assertTrue(torch.equal(treated.probe_deep, control.probe_deep))


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = ast.parse((ROOT / 'tools/run_rddr_phase2b112.py').read_text(encoding='utf-8'))
        cls.run_node = next(node for node in cls.runner.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
        cls.training_loop = next(node for node in ast.walk(cls.run_node) if isinstance(node, ast.For)
                                 and ast.unparse(node.iter) == 'range(1, 501)')

    def extracted(self, name, namespace=None):
        node = next(node for node in ast.walk(self.runner)
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
        space = dict(torch=torch, random=random, json=json, hashlib=hashlib, digest=common.digest,
                     os=os, **(namespace or {}))
        exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), '<runner-unit>', 'exec'), space)
        return space[name]

    def test_immutable_provenance_and_schedules(self):
        self.assertEqual(common.A0, '4e9a2887b220d17e27649d72a3d13f32b7ebe8f9')
        self.assertEqual(common.CKPT_SHA, '509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579')
        self.assertEqual(common.SNAPSHOTS, (0, 50, 100, 250, 500))
        self.assertEqual(common.INTERACTIONS, (0, 50, 250, 500))
        self.assertEqual(common.UPSTREAM, ('b4', 'b4_1', 'b4_2', 'b4_3', 'b4_4', 'b4_5', 'bn45'))

    def test_gradient_helpers_never_step_optimizer(self):
        for function in [common.gradients_for_batch, common.auxiliary_forward, common.auxiliary_loss]:
            tree = ast.parse(inspect.getsource(function))
            forbidden = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Attribute) and node.func.attr in {'step', 'clip_grad_norm_', 'clip_grad_value_'}]
            self.assertEqual(forbidden, [], function.__name__)

    def test_rng_round_trip_restores_python_numpy_torch(self):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        state = common.rng_state()
        expected = random.random(), np.random.random(5), torch.rand(5)
        common.restore_rng(state)
        actual = random.random(), np.random.random(5), torch.rand(5)
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_loader_frozen_seed_batch_and_sampling(self):
        loader = self.extracted('loader', {'DataLoader': lambda dataset, **kwargs: (dataset, kwargs),
                                            'seed_worker': 'original_seed_worker'})
        sentinel = object()
        dataset, kwargs = loader(sentinel)
        self.assertIs(dataset, sentinel)
        self.assertEqual({key: kwargs[key] for key in ('batch_size', 'shuffle', 'num_workers', 'pin_memory', 'drop_last')},
                         dict(batch_size=20, shuffle=True, num_workers=4, pin_memory=True, drop_last=True))
        self.assertEqual(kwargs['generator'].initial_seed(), 42)
        self.assertEqual(kwargs['worker_init_fn'], 'original_seed_worker')
        self.assertTrue(torch.equal(torch.randperm(100, generator=kwargs['generator']),
                                    torch.randperm(100, generator=loader(sentinel)[1]['generator'])))

    def test_same_augmentation_seed_without_rng_consumption(self):
        class FakeOfficialDataset:
            def __getitem__(self, index):
                return 'image', torch.tensor([random.random() > .5, random.random() > .5]), torch.tensor([index])
        recorded = self.extracted('RecordedTrainDataset', {'Stage1_TrainDataset': FakeOfficialDataset})
        random.seed(42)
        expected_state = random.getstate()
        expected = FakeOfficialDataset()[7]
        state_after = random.getstate()
        random.setstate(expected_state)
        actual = recorded()[7]
        detail = json.loads(actual[3])
        self.assertEqual(random.getstate(), state_after)
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual([detail['horizontal_flip'], detail['vertical_flip']], expected[1].tolist())
        self.assertEqual(detail['augmentation_rng_sha256'], hashlib.sha256(repr(expected_state).encode()).hexdigest())
        self.assertEqual(detail['worker_seed'], torch.initial_seed())

    def test_same_batch_manifest_and_rng_reused_in_arm_loop(self):
        arm_loop = next(node for node in self.training_loop.body if isinstance(node, ast.For)
                        and ast.unparse(node.iter) == "('B', 'A', 'R')")
        source = ast.unparse(arm_loop)
        self.assertIn('restore_rng(shared_rng)', source)
        self.assertIn('batch[1].cuda(), batch[2].cuda()', source)
        self.assertIn('random_counts=counts, random_rng=random_rng', source)
        self.assertNotIn('next(', source)
        self.assertLess(source.index('restore_rng(shared_rng)'), source.index('gradients_for_batch('))
        self.assertIn("if arm == 'A':\n        counts = c", source)
        self.assertIn("item == calibration_manifest[step - 1]", ast.unparse(self.training_loop))
        source_batch = (['a', 'b'], torch.ones(2, 3, 2, 2), torch.zeros(2, 4),
                        [json.dumps({'worker_seed': 42}), json.dumps({'worker_seed': 42})])
        metadata = self.extracted('metadata')
        self.assertEqual(metadata(source_batch, 1), metadata(source_batch, 1))

    def test_lambda_calibration_32_batches(self):
        source = ast.unparse(self.run_node)
        self.assertIn('calibration_batches = [next(iterator) for _ in range(32)]', source)
        self.assertIn("gradients_for_batch(model, batch[1].cuda(), batch[2].cuda(), 'A', 0.0)", source)
        self.assertIn("r['main_grad_norm'] / (r['aux_grad_norm'] + 1e-08)", source)
        self.assertIn('median = float(np.median(ratios))', source)
        self.assertIn('lam = 0.1 * median', source)
        self.assertIn('optimizer.global_step == MAX_STEP and (not optimizer.state)', source)
        self.assertIn('digest(model.state_dict().items()) == initial_state', source)

    def test_lambda_frozen_after_step0(self):
        assignments = [node for node in ast.walk(self.run_node) if isinstance(node, ast.Name)
                       and node.id == 'lam' and isinstance(node.ctx, ast.Store)]
        self.assertEqual(len(assignments), 1)
        self.assertLess(assignments[0].lineno, self.training_loop.lineno)
        self.assertNotIn('np.median', ast.unparse(self.training_loop))

    def test_lambda_no_validation_use_and_no_test_luad(self):
        phase, access = ['calibration'], set()
        audit = self.extracted('audit', {'phase': phase, 'access': access})
        for phase_name in ['calibration', 'training', 'interaction']:
            phase[0] = phase_name
            audit('open', ('/home/duyanhong/reseg-data/raw/BCSS-WSSS/training/img/a.png',))
            with self.assertRaises(AssertionError):
                audit('open', ('/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img/a.png',))
        phase[0] = 'validation'
        audit('open', ('/home/duyanhong/reseg-data/raw/BCSS-WSSS/val/img/a.png',))
        for path in ['/home/duyanhong/reseg-data/raw/BCSS-WSSS/test/img/a.png',
                     '/home/duyanhong/reseg-data/raw/LUAD-HistoSeg/training/img/a.png']:
            with self.assertRaises(AssertionError):
                audit('open', (path,))

    def test_snapshot_schedule_exact_and_no_early_stop(self):
        self.assertEqual(common.SNAPSHOTS, (0, 50, 100, 250, 500))
        self.assertFalse(any(isinstance(node, (ast.Break, ast.Return)) for node in ast.walk(self.training_loop)))
        conditionals = [ast.unparse(node.test) for node in self.training_loop.body if isinstance(node, ast.If)]
        self.assertIn('step in SNAPSHOTS', conditionals)
        self.assertIn('step in (250, 500)', conditionals)

    def test_no_other_seed(self):
        seeds = []
        for node in ast.walk(self.runner):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func).split('.')[-1]
                if name in ('manual_seed', 'set_seed', 'default_rng'):
                    seeds.append(ast.literal_eval(node.args[0]))
        self.assertTrue(seeds)
        self.assertEqual(set(seeds), {42})

    def test_no_threshold_search_or_lambda_sweep(self):
        options = [ast.literal_eval(node.args[0]) for node in ast.walk(self.runner)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'add_argument']
        self.assertEqual(options, ['--output', '--preflight-only'])

    def test_no_third_evidence(self):
        self.assertEqual(tuple(inspect.signature(common.adjudicate).parameters), ('ps', 'pd'))
        self.assertEqual(tuple(inspect.signature(common.auxiliary_loss).parameters), ('logits', 'deep', 'q', 'gate'))
        support = ast.unparse(ast.parse(inspect.getsource(common.adjudicate)))
        self.assertNotIn('ctx', support)
        self.assertNotIn('alt', support)


class StatisticsTests(unittest.TestCase):
    def gate_fixture(self):
        bootstrap = [dict(step=step, comparison='A-B', metric='miou', population='official', delta=.003, ci_low=.001)
                     for step in (100, 250, 500)]
        bootstrap.append(dict(step=500, comparison='A-R', metric='miou', population='official', delta=.0015, ci_low=.0005))
        for population, accuracy, margin in [('Deep-Win_0', .02, .03), ('Shallow-Win_0', 0., -.05)]:
            bootstrap.extend([dict(step=500, comparison='A-B', metric='raw_accuracy', population=population,
                                   delta=accuracy, ci_low=.01 if accuracy else -.001),
                              dict(step=500, comparison='A-B', metric='raw_gt_margin', population=population,
                                   delta=margin, ci_low=margin-.01)])
        summary = dict(bootstrap=bootstrap,
                       official_metrics=[dict(arm=arm, step=500, **{f'iou_class{i}': .51 if arm == 'A' else .5 for i in range(4)})
                                         for arm in ('A', 'B')],
                       population_metrics=[dict(arm='B', step=0, population='Shallow-Win_0', raw_gt_margin=1.),
                                           *[dict(arm=arm, step=500, population='Stable-Correct_0', raw_accuracy=.9,
                                                  rect_accuracy=.9) for arm in ('A', 'B')]],
                       training_audit=dict(present=False, errors=[], arms={}),
                       gate_dynamics=[dict(arm='A', step=500, active_fraction_all=.2)],
                       snapshot_step0_bitwise_equal=True)
        evidence = {key: {} for key in ('optimizer_provenance', 'runtime', 'identity_step0', 'verification')}
        evidence['lambda_calibration'] = {'missing_file': 'not_executed'}
        return summary, evidence

    def test_bootstrap_reproducible_pooled_confusion_oracle(self):
        rng = np.random.RandomState(42)
        left = rng.randint(0, 100, (3, 5, 5))
        right = rng.randint(0, 100, (3, 5, 5))
        left[:, 4, :4] = right[:, 4, :4] = 0
        actual = statistics.bootstrap_confusions(left, right)
        self.assertEqual(actual, statistics.bootstrap_confusions(left, right))
        indices = np.random.RandomState(42).randint(0, 3, (10_000, 3))
        def independent_metrics(matrix):
            matrix = matrix.astype(np.float64).copy()
            matrix[..., 4, 4] = 0
            tp = np.diagonal(matrix, axis1=-2, axis2=-1)[..., :4]
            rows, columns = matrix.sum(-1)[..., :4], matrix.sum(-2)[..., :4]
            return {'miou': (tp / (rows + columns - tp)).mean(-1),
                    'mdice': (2 * tp / (rows + columns)).mean(-1)}
        pooled_left, pooled_right = independent_metrics(left[indices].sum(1)), independent_metrics(right[indices].sum(1))
        point_left, point_right = independent_metrics(left.sum(0)), independent_metrics(right.sum(0))
        for metric in ('miou', 'mdice'):
            expected = np.quantile(pooled_left[metric] - pooled_right[metric], [.025, .975])
            np.testing.assert_allclose([actual[metric]['ci_low'], actual[metric]['ci_high']], expected, rtol=0, atol=1e-15)
            self.assertAlmostEqual(actual[metric]['delta'], float(point_left[metric]-point_right[metric]), places=15)
            self.assertEqual(actual[metric]['bootstrap_replicates'], 10_000)
            self.assertEqual(actual[metric]['bootstrap_seed'], 42)
            self.assertEqual(actual[metric]['valid_replicates'], 10_000)

    def test_official_absent_class_and_background_semantics(self):
        matrix = np.zeros((5, 5))
        matrix[0, 0], matrix[0, 4], matrix[4, 4] = 8, 2, 1000
        result = statistics.confusion_metrics(matrix)
        self.assertAlmostEqual(float(result['miou']), .8)
        self.assertAlmostEqual(float(result['mdice']), (16/18)/4)
        self.assertTrue(np.isnan(result['iou'][1:]).all())
        np.testing.assert_array_equal(result['dice'][1:], np.zeros(3))
        self.assertEqual(matrix[4, 4], 1000)

    def test_original_official_scores_match_observed_confusions(self):
        from tool import iouutils
        rng = np.random.RandomState(42)
        truth = [rng.randint(0, 5, (9, 13)) for _ in range(3)]
        prediction = [rng.randint(0, 5, (9, 13)) for _ in range(3)]
        original = iouutils.scores(truth, copy.deepcopy(prediction), 4)
        matrices = []
        for labels, pred in zip(truth, prediction):
            pred = pred.copy(); pred[labels == 4] = 4
            matrices.append(iouutils._fast_hist(labels.flatten(), pred.flatten(), 5))
        observed = evaluation._metrics_from_cm(np.stack(matrices))
        analyzed = statistics.confusion_metrics(np.stack(matrices).sum(0))
        self.assertEqual(original['Mean IoU'], observed['miou'])
        self.assertEqual(original['Mean Dice'], observed['mdice'])
        self.assertEqual(original['Mean IoU'], float(analyzed['miou']))
        self.assertEqual(original['Mean Dice'], float(analyzed['mdice']))

    def test_official_evaluator_same_call_and_unmodified_output(self):
        tree = ast.parse(inspect.getsource(evaluation.evaluate_snapshot))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == 'infer_fun.infer']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].keywords, [])
        self.assertEqual(ast.unparse(calls[0].args[3]),
                         "SimpleNamespace(dataset='bcss', img_size=224, amp_dtype='bf16', num_workers=0)")
        observer = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'forward_cam_observer')
        returns = [ast.unparse(node.value) for node in ast.walk(observer) if isinstance(node, ast.Return)]
        self.assertEqual(returns, ['outputs'])
        self.assertEqual(evaluation.N_TTA, 3)
        self.assertEqual(evaluation.N_IMAGES, 3418)

    def test_representation_selection_fixed_160_seed42(self):
        actual = evaluation._reference_selection(3418)
        fixed = np.linspace(0, 3417, 32, dtype=int)
        remaining = np.setdiff1d(np.arange(3418), fixed)
        expected = np.r_[fixed, np.random.default_rng(42).choice(remaining, 128, replace=False)]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(set(actual)), 160)

    def test_gate_E_fixed_step0_margin_tolerance_and_strict_ci(self):
        summary, evidence = self.gate_fixture()
        self.assertEqual(statistics.evaluate_gates(summary, evidence)['gates']['E']['status'], 'PASS')
        margin = next(row for row in summary['bootstrap'] if row['population'] == 'Shallow-Win_0' and row['metric'] == 'raw_gt_margin')
        margin['delta'] = -.05000001
        self.assertEqual(statistics.evaluate_gates(summary, evidence)['gates']['E']['status'], 'FAIL')
        margin['delta'] = -.05
        accuracy = next(row for row in summary['bootstrap'] if row['population'] == 'Shallow-Win_0' and row['metric'] == 'raw_accuracy')
        accuracy['delta'], accuracy['ci_low'] = -.002, -.0049999
        self.assertEqual(statistics.evaluate_gates(summary, evidence)['gates']['E']['status'], 'PASS')
        accuracy['ci_low'] = -.005
        self.assertEqual(statistics.evaluate_gates(summary, evidence)['gates']['E']['status'], 'FAIL')

    def test_gate_G_checks_all_four_classes(self):
        summary, evidence = self.gate_fixture()
        summary['official_metrics'][0]['iou_class3'] = .49499
        result = statistics.evaluate_gates(summary, evidence)['gates']['G']
        self.assertEqual(result['status'], 'FAIL')
        self.assertEqual(result['facts']['classes_checked'], [0, 1, 2, 3])

    def test_gate_H_ratio_summary_uses_all500_not_snapshots(self):
        rows = []
        for arm in ('B', 'A', 'R'):
            for step in range(1, 501):
                ratio = (.4 if step <= 300 else .1) if arm == 'A' else .02
                aux, weighted = (0., 0.) if arm == 'B' else (.5, .1)
                rows.append(dict(arm=arm, step=step, main_loss=.8, aux_loss=aux, weighted_aux_loss=weighted,
                                 total_loss=.8+weighted, main_grad_norm=1., aux_grad_norm=2., weighted_gradient_ratio=ratio,
                                 gradient_cosine=.5, total_grad_norm=1., lr0=1e-6, lr1=2e-6, lr2=1e-5, lr3=2e-5,
                                 active_fraction=.2, seconds=1., finite=True))
        audit, _ = statistics.training_statistics(rows, {'lambda_value': .2})
        self.assertEqual(audit['errors'], [])
        self.assertEqual(audit['arms']['A']['weighted_gradient_ratio_median_all500'], .4)
        self.assertEqual(audit['arms']['R']['weighted_gradient_ratio_median_all500'], .02)
        self.assertTrue(audit['arms']['A']['step_sequence_exact'])
        incomplete, _ = statistics.training_statistics(rows[:-1], {'lambda_value': .2})
        self.assertTrue(incomplete['errors'])

    def test_step500_primary_and_bootstrap_constants(self):
        self.assertEqual(statistics.STEPS, (0, 50, 100, 250, 500))
        self.assertEqual(statistics.BOOTSTRAP_REPLICATES, 10_000)
        self.assertEqual(statistics.BOOTSTRAP_SEED, 42)
        self.assertIn("'primary_endpoint': step == 500", ast.unparse(ast.parse(inspect.getsource(statistics.analyze_snapshots))))

    def test_frozen_quintile_threshold_ties_go_left(self):
        truth = np.zeros((1, 4), dtype=int)
        ps = np.zeros((1, 4, 4)); ps[:, 0] = 1
        snapshot = dict(truth=truth, ps=ps, pd=ps.copy(), q=statistics.Q_EDGES[None],
                        top20=np.zeros_like(truth, dtype=bool), boundary=np.zeros_like(truth, dtype=bool))
        populations = statistics.frozen_populations(snapshot)
        for index in range(4):
            self.assertTrue(populations[f'Q{index+1}_q0'][0, index])
            self.assertFalse(populations[f'Q{index+2}_q0'][0, index])

    def test_decision_priority_and_missing_evidence_never_go(self):
        all_pass = {key: {'status': 'PASS'} for key in 'ABCDEFGH'}
        self.assertEqual(statistics.choose_decision('PASS', all_pass), 'RDDR_ADT_SHORT_HORIZON_DYNAMICS_GO')
        self.assertEqual(statistics.choose_decision('BLOCKED', all_pass), 'SHORT_HORIZON_OPTIMIZER_PROVENANCE_BLOCKED')
        for key in 'ABCDEFGH':
            pending = copy.deepcopy(all_pass); pending[key]['status'] = 'PENDING'
            self.assertIsNone(statistics.choose_decision('PASS', pending), key)
        failed = copy.deepcopy(all_pass)
        for key in 'ABCDEFGH': failed[key]['status'] = 'FAIL'
        self.assertEqual(statistics.choose_decision('PASS', failed), 'ADT_SHORT_HORIZON_ENGINEERING_NOGO')
        failed['H']['status'] = 'PASS'
        self.assertEqual(statistics.choose_decision('PASS', failed), 'ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION')
        failed['A']['status'] = failed['B']['status'] = 'PASS'
        self.assertEqual(statistics.choose_decision('PASS', failed), 'ADT_OPTIMIZATION_GAIN_WITH_SEMANTIC_SAFETY_REGRESSION')
        for key in 'DEFG': failed[key]['status'] = 'PASS'
        self.assertEqual(statistics.choose_decision('PASS', failed), 'SHORT_HORIZON_GAIN_NOT_CONTEXT_SPECIFIC')
        weak = copy.deepcopy(all_pass); weak['A']['status'] = 'WEAK_POSITIVE'
        self.assertEqual(statistics.choose_decision('PASS', weak), 'ADT_LOCAL_SIGNAL_NOT_TRANSLATING_TO_OPTIMIZATION')


if __name__ == '__main__':
    unittest.main(verbosity=2)
