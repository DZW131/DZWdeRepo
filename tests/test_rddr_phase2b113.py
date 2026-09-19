"""Independent CPU/control tests for the frozen Phase-2B1.13 audit."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tools import rddr_phase2b113_common as common
from tools import run_rddr_phase2b113 as runner


def vectors():
    main = {"b4.weight": torch.tensor([1.0, 2.0]), "bn45.bias": None}
    ctx = {"b4.weight": torch.tensor([2.0, -1.0]), "bn45.bias": torch.tensor([0.5])}
    rnd = {"b4.weight": torch.tensor([-1.0, 2.0]), "bn45.bias": torch.tensor([0.25])}
    params = {"b4.weight": torch.tensor([3.0, 4.0]), "bn45.bias": torch.tensor([0.0])}
    specs = {name: {"optimizer_group": index, "lr": 1e-3 * (index + 1), "weight_decay": 5e-4,
                    "momentum": 5e-4, "dampening": 0.0, "nesterov": False}
             for index, name in enumerate(params)}
    return main, ctx, rnd, params, specs


class FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = inspect.getsource(runner)
        cls.tree = ast.parse(cls.source)

    def test_phase2b112_identity_replay(self):
        source = inspect.getsource(runner._manual_training_batch)
        self.assertIn('actual == item["tensor_sha256"]', source)
        self.assertIn("TRACK_T_REPLAY_BLOCKED", source)

    def test_parameter_manifest_exact(self):
        source = inspect.getsource(common.parameter_manifest)
        self.assertIn("len(selected) == 39", source)
        self.assertIn("27_275_776", source)

    def test_only_approved_39_parameters(self):
        self.assertTrue(common.approved("b4.conv_branch2a.weight"))
        self.assertTrue(common.approved("bn45.bias"))
        for name in ("ic1.weight", "hfrm_28_1.gamma_context", "b5.conv.weight", "b3.conv.weight"):
            self.assertFalse(common.approved(name))

    def test_adt_formula_exact(self):
        source = inspect.getsource(runner._training_triplet)
        for token in ("adjudicate", "ctx_weight = q * evidence", "target = bundle[\"deep_probability\"]",
                      "(ctx_weight * kl).sum() / (ctx_weight.sum() + EPS)"):
            self.assertIn(token, source)

    def test_random_gate_rate_exact(self):
        rng = np.random.default_rng(42)
        gate = runner.random_gate(np.array([0, 1, 200, 784]), rng)
        np.testing.assert_array_equal(gate.sum(1).numpy(), [0, 1, 200, 784])

    def test_random_seed42_exact(self):
        self.assertGreaterEqual(self.source.count("np.random.default_rng(42)"), 2)

    def test_lambda_exact(self):
        self.assertEqual(common.LAMBDA_ADT, 0.027074256246554088)

    def test_main_gradient(self):
        self.assertIn("torch.autograd.grad", inspect.getsource(runner._main_gradients))

    def test_ctx_aux_gradient(self):
        self.assertIn("ctx = _aux_gradients", inspect.getsource(runner._training_triplet))

    def test_random_aux_gradient(self):
        self.assertIn("rnd = _aux_gradients", inspect.getsource(runner._training_triplet))

    def test_no_grad_outside_approved_params(self):
        source = inspect.getsource(runner._main_gradients)
        self.assertIn("if named[name].requires_grad", source)
        self.assertNotIn(".backward", self.source)

    def test_trackT_batch_count_128(self):
        self.assertEqual(common.TRAIN_BATCHES, 128)
        self.assertIn("[:TRAIN_BATCHES]", inspect.getsource(runner.run_track_t))

    def test_no_random_seed_sweep(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.For):
                self.assertNotIn("seed", ast.unparse(node.target).lower())

    def test_no_optimizer_step(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == "step" for node in calls))

    def test_no_checkpoint_write(self):
        self.assertNotIn("torch.save", self.source)

    def test_no_test(self):
        audit = inspect.getsource(runner.main)
        self.assertIn('"/test/" not in path', audit)
        self.assertNotIn('DATA_ROOT / "test"', self.source)

    def test_no_luad(self):
        audit = inspect.getsource(runner.main)
        self.assertIn('"luad" not in path', audit)
        self.assertNotIn("dataset=\"luad\"", self.source)

    def test_state_hash_unchanged(self):
        tensor = torch.tensor([1.0, 2.0])
        before = common.tensor_digest((("x", tensor),))
        _ = common.vector_norm({"b4.x": tensor})
        after = common.tensor_digest((("x", tensor),))
        self.assertEqual(before, after)


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.main, self.ctx, self.rnd, self.params, self.specs = vectors()

    def test_aux_cosine_formula(self):
        actual = common.vector_cosine(self.ctx, self.rnd)
        left = torch.cat([self.ctx["b4.weight"], self.ctx["bn45.bias"]]).double()
        right = torch.cat([self.rnd["b4.weight"], self.rnd["bn45.bias"]]).double()
        expected = float(torch.dot(left, right) / (left.norm() * right.norm()))
        self.assertAlmostEqual(actual, expected, places=14)

    def test_direction_difference_formula(self):
        direct, formula = common.direction_difference(self.ctx, self.rnd)
        self.assertAlmostEqual(direct, formula, places=15)
        self.assertAlmostEqual(direct * direct, 2 * (1 - common.vector_cosine(self.ctx, self.rnd)), places=14)

    def test_total_gradient_formula(self):
        total = common.add_gradients(self.main, self.ctx, common.LAMBDA_ADT)
        torch.testing.assert_close(total["b4.weight"],
                                   self.main["b4.weight"] + common.LAMBDA_ADT * self.ctx["b4.weight"])
        torch.testing.assert_close(total["bn45.bias"], common.LAMBDA_ADT * self.ctx["bn45.bias"])

    def test_rho_ctx_formula(self):
        metrics, _, _ = common.vector_metrics(self.main, self.ctx, self.rnd, self.params, self.specs)
        self.assertGreater(metrics["rho_ctx"], 0)

    def test_virtual_optimizer_matches_clone_step(self):
        total = common.add_gradients(self.main, self.ctx, common.LAMBDA_ADT)
        self.assertLessEqual(common.dry_run_clone_error(self.params, total, self.specs), 1e-12)

    def test_virtual_optimizer_no_state_write(self):
        before = {name: value.clone() for name, value in self.params.items()}
        total = common.add_gradients(self.main, self.ctx, common.LAMBDA_ADT)
        common.virtual_fresh_update(self.params, total, self.specs)
        for name in before:
            torch.testing.assert_close(self.params[name], before[name])

    def test_update_cosine_formula(self):
        metrics, update_a, update_r = common.vector_metrics(
            self.main, self.ctx, self.rnd, self.params, self.specs)
        self.assertAlmostEqual(metrics["C_update"], common.vector_cosine(update_a, update_r), places=15)

    def test_rho_update_formula(self):
        metrics, update_a, update_r = common.vector_metrics(
            self.main, self.ctx, self.rnd, self.params, self.specs)
        self.assertAlmostEqual(metrics["rho_update"], common.relative_difference(update_a, update_r), places=15)

    def test_oracle_gradient_formula(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        gradient, = torch.autograd.grad(loss, logits)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)

    def test_oracle_alignment_formula(self):
        oracle = {"b4.weight": torch.tensor([0.5, -0.5]), "bn45.bias": torch.tensor([1.0])}
        metrics = common.oracle_metrics(
            self.ctx, self.rnd, oracle,
            common.virtual_fresh_update(self.params, common.add_gradients(self.main, self.ctx), self.specs),
            common.virtual_fresh_update(self.params, common.add_gradients(self.main, self.rnd), self.specs))
        self.assertAlmostEqual(metrics["DeltaC_oracle"],
                               common.vector_cosine(self.ctx, oracle) - common.vector_cosine(self.rnd, oracle))

    def test_oracle_projection_formula(self):
        oracle = {"b4.weight": torch.tensor([0.5, -0.5]), "bn45.bias": torch.tensor([1.0])}
        expected = common.vector_dot(self.ctx, oracle) / (common.vector_dot(oracle, oracle) + runner.EPS)
        metrics = common.oracle_metrics(self.ctx, self.rnd, oracle,
                                        common.virtual_fresh_update(self.params, self.ctx, self.specs),
                                        common.virtual_fresh_update(self.params, self.rnd, self.specs))
        self.assertAlmostEqual(metrics["P_ctx"], expected)

    def test_first_order_oracle_change_formula(self):
        oracle = {"b4.weight": torch.tensor([0.5, -0.5]), "bn45.bias": torch.tensor([1.0])}
        update_a = common.virtual_fresh_update(self.params, self.ctx, self.specs)
        update_r = common.virtual_fresh_update(self.params, self.rnd, self.specs)
        metrics = common.oracle_metrics(self.ctx, self.rnd, oracle, update_a, update_r)
        self.assertAlmostEqual(metrics["Adv_oracle"],
                               -common.vector_dot(oracle, update_a) + common.vector_dot(oracle, update_r))


class PopulationAndBootstrapTests(unittest.TestCase):
    def test_oracle_uses_gt_only_for_diagnostic(self):
        source = inspect.getsource(runner.run_track_v)
        self.assertIn("truth < 4", source)
        self.assertIn("oracle_numerator", source)
        self.assertIn("None, names", source)
        self.assertNotIn("_parse_image_labels", source)

    def test_gate_is_gt_blind(self):
        source = inspect.getsource(runner.run_track_v)
        gate_position = source.index("frozen_gate")
        truth_position = source.index("truth =")
        self.assertLess(gate_position, truth_position)
        self.assertIn('snapshot["delta"]', source)

    def test_population_counts_exact(self):
        truth = np.array([[0, 1, 2, 3, 4, 255]])
        valid = truth < 4
        self.assertEqual(int(valid.sum()), 4)

    def test_population_partition_exhaustive(self):
        raw = np.array([[True, True, False, False]])
        deep = np.array([[True, False, True, False]])
        groups = (raw & deep, raw & ~deep, ~raw & deep, ~raw & ~deep)
        np.testing.assert_array_equal(sum(group.astype(np.int8) for group in groups), np.ones_like(raw, dtype=np.int8))

    def test_population_gradient_sum_identity(self):
        parts = [
            {"b4.x": torch.tensor([1.0, 0.0])}, {"b4.x": torch.tensor([0.0, 2.0])},
            {"b4.x": torch.tensor([-0.5, 0.0])}, {"b4.x": torch.tensor([0.0, -0.5])},
        ]
        total = parts[0]
        for part in parts[1:]:
            total = common.add_gradients(total, part)
        expected = {"b4.x": torch.tensor([0.5, 1.5])}
        self.assertLess(common.max_relative_error(total, expected), 1e-15)

    def test_population_cancellation_index(self):
        opposite = [{"b4.x": torch.tensor([1.0])}, {"b4.x": torch.tensor([-1.0])}]
        aligned = [{"b4.x": torch.tensor([1.0])}, {"b4.x": torch.tensor([1.0])}]
        self.assertGreater(common.cancellation_index(opposite), common.cancellation_index(aligned))
        self.assertAlmostEqual(common.cancellation_index(aligned), 0.0, places=8)

    def test_blockwise_sum_identity(self):
        gradient = {"b4.x": torch.tensor([1.0]), "b4_1.x": torch.tensor([2.0]),
                    "bn45.x": torch.tensor([3.0])}
        squared = sum(common.vector_norm(common.subset(gradient, block)) ** 2
                      for block in ("b4", "b4_1", "bn45"))
        self.assertAlmostEqual(squared, common.vector_norm(gradient) ** 2)

    def test_bootstrap_seed42(self):
        rows = [{"x": float(index), "y": float(index * 2)} for index in range(8)]
        left = common.bootstrap(rows, (("x", "mean"), ("x", "median"), ("y", "mean")))
        right = common.bootstrap(rows, (("x", "mean"), ("x", "median"), ("y", "mean")))
        self.assertEqual(left, right)
        self.assertTrue(all(row["bootstrap_seed"] == 42 and row["statistical_unit"] == "minibatch"
                            for row in left))


if __name__ == "__main__":
    unittest.main(verbosity=2)
