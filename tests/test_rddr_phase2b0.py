"""CPU mathematical/safety tests; real CUDA/cache checks are in the runner."""
import ast
import inspect
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import rddr_phase2b0_common as c
from tools.run_rddr_phase2b0_relation_audit import validate_root


class RelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.ps = torch.randn(1, 4, 28, 28).softmax(1)
        cls.pd = torch.randn(1, 4, 28, 28).softmax(1)
        cls.rel = c.build_relations(cls.ps, cls.pd)

    def test_js_matches_phase0_exactly(self):
        ps, pd = self.ps, self.pd
        m = .5*(ps+pd)
        expected = .5*((ps*((ps+1e-8).log()-(m+1e-8).log())).sum(1)
                       +(pd*((pd+1e-8).log()-(m+1e-8).log())).sum(1))
        self.assertTrue(torch.equal(c.phase0_js(ps, pd), expected))

    def test_q_range_0_1(self):
        self.assertTrue(((self.rel["q"] >= 0) & (self.rel["q"] <= 1)).all())
        self.assertLess(c.phase0_js(self.ps, self.ps).abs().max(), 1e-7)

    def test_srsc_no_gt_dependency(self):
        before = {k: v.clone() for k, v in self.rel.items()}
        c.relation_gt_metrics(self.rel, torch.zeros(28, 28).long())
        c.relation_gt_metrics(self.rel, torch.full((28, 28), 4).long())
        for k in before:
            self.assertTrue(torch.equal(before[k], self.rel[k]))

    def test_gt_used_only_in_metrics(self):
        args = list(inspect.signature(c.build_relations).parameters)
        self.assertEqual(args, ["ps", "pd"])
        tree = ast.parse(inspect.getsource(c.build_relations))
        identifiers = {v.id for v in ast.walk(tree) if isinstance(v, ast.Name)}
        self.assertFalse(identifiers & {"truth", "gt", "oracle", "mask", "labels"})

    def test_15x15_window_contract(self):
        v = self.rel["valid"][0]
        self.assertEqual(v[:, 0].sum(), 63)
        self.assertEqual(v[:, 14*28+14].sum(), 224)

    def test_self_edge_excluded(self):
        self.assertFalse(self.rel["valid"][:, 112].any())
        self.assertEqual(self.rel["weights"][:, :, 112].abs().sum(), 0)

    def test_relation_score_range(self):
        w = self.rel["weights"]
        self.assertTrue(((w >= 0) & (w <= 1)).all())
        self.assertTrue(torch.equal(w[:, 3], w[:, 1]*w[:, 2]))

    def test_uniform_relation_constant(self):
        self.assertTrue(torch.equal(self.rel["weights"][:, 0], self.rel["valid"].float()))

    def test_neighbor_distribution_sums_to_one(self):
        self.assertTrue(torch.allclose(self.rel["distribution"].sum(2), torch.ones(1, 4, 784), atol=1e-6))

    def test_oracle_empty_no_fallback(self):
        y = torch.full((28, 28), 4)
        y[0, 0] = 0
        result = c.relation_gt_metrics(self.rel, y)
        self.assertFalse(result["oracle_valid"].any())
        self.assertEqual(result["oracle"].abs().sum(), 0)
        self.assertTrue(torch.isfinite(self.rel["distribution"]).all())

    def test_frozen_population_replay_counts(self):
        truth = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        cache = dict(raw=np.array([[1, 2], [2, 3]]), rect=np.array([[0, 3], [1, 3]]), top20=np.array([[1, 0], [0, 0]], bool))
        groups = c.populations(cache, truth)
        for name in ("Corrected_by_CH", "Still_Wrong", "Harmed_by_CH", "Stable_Correct"):
            self.assertEqual(groups[name].sum(), 1)
            self.assertEqual(c.project(groups[name]).sum(), 196)

    def test_no_gradient(self):
        r = c.build_relations(self.ps.clone().requires_grad_(), self.pd)
        self.assertFalse(any(v.requires_grad for v in r.values()))

    def test_no_optimizer(self):
        p = Path(__file__).resolve().parents[1] / "tools/run_rddr_phase2b0_relation_audit.py"
        tree = ast.parse(p.read_text(encoding="utf-8"))
        calls = [ast.unparse(v.func) for v in ast.walk(tree) if isinstance(v, ast.Call)]
        self.assertFalse(any("optim." in v or v.endswith(".backward") for v in calls))

    def test_no_checkpoint_write(self):
        p = Path(__file__).resolve().parents[1] / "tools/run_rddr_phase2b0_relation_audit.py"
        tree = ast.parse(p.read_text(encoding="utf-8"))
        calls = [ast.unparse(v.func) for v in ast.walk(tree) if isinstance(v, ast.Call)]
        self.assertNotIn("torch.save", calls)

    def test_no_test_luad_access(self):
        for p in ("/tmp/BCSS-WSSS/test", "/tmp/LUAD-HistoSeg/val", "/tmp/BCSS-WSSS/training"):
            with self.assertRaises(ValueError):
                validate_root(p)

    def test_bootstrap_reproducible(self):
        x = np.array([1., 2., np.nan, 3.])
        self.assertTrue(np.array_equal(c.bootstrap_means(x, 100), c.bootstrap_means(x, 100), equal_nan=True))

    def test_pair_exact_hist_and_constant(self):
        score, label = np.array([.1, .1, .7, .8]), np.array([0, 1, 0, 1])
        a = c.binary_metrics(c.binary_hist(score, label))
        b = c.exact_binary_metrics(score, label)
        self.assertAlmostEqual(a["auroc"], b["auroc"])
        self.assertAlmostEqual(a["auprc"], b["auprc"])
        const = c.binary_metrics(c.binary_hist(np.ones(4), label))
        self.assertEqual(const["auroc"], .5)
        self.assertEqual(const["auprc"], .5)
        self.assertTrue(np.isnan(c.binary_metrics(c.binary_hist(score, np.ones(4)))["auroc"]))

    def test_confusion_absent_class(self):
        cm = np.zeros((4, 4))
        cm[0, 0] = 5
        m = c.cm_metrics(cm)
        self.assertEqual(m["miou"], 1.)
        self.assertTrue(np.isnan(m["class_iou"][1:]).all())

    def test_geometry_no_gt_padding_source(self):
        rel = c.build_relations(torch.full_like(self.ps, .25), torch.full_like(self.pd, .25))
        self.assertTrue(torch.allclose(rel["distribution"], torch.full_like(rel["distribution"], .25)))
        self.assertTrue(torch.allclose(rel["neff"][:, 0], rel["mass"][:, 0]))

    def test_source_target_orientation_against_explicit_neighbors(self):
        for ty, tx in ((0, 0), (14, 14), (27, 10)):
            total = torch.zeros(4)
            mass = 0.
            for sy in range(max(0, ty-7), min(28, ty+8)):
                for sx in range(max(0, tx-7), min(28, tx+8)):
                    if (sy, sx) == (ty, tx):
                        continue
                    p = self.ps[0, :, sy, sx]
                    dsource = self.pd[0, :, sy, sx]
                    dtarget = self.pd[0, :, ty, tx]
                    r = (1-c.phase0_js(p, dsource, dim=0)/math.log(2)).clamp(0, 1)
                    compatibility = (1-c.phase0_js(dtarget, p, dim=0)/math.log(2)).clamp(0, 1)
                    a = r*compatibility
                    total += a*p
                    mass += a
            expected = total/(mass+1e-8)
            self.assertTrue(torch.allclose(expected, self.rel["distribution"][0, 3, :, ty*28+tx], atol=5e-7))

    def test_boundary_width_is_full_resolution(self):
        y = np.zeros((224, 224), dtype=np.uint8)
        y[:, 112:] = 1
        m = c.boundary_masks(y)
        self.assertTrue(m["boundary"][:, 104:120].all())
        self.assertFalse(m["boundary"][:, :104].any())


if __name__ == "__main__":
    unittest.main(verbosity=2)
