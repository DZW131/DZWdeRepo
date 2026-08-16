"""Frozen engineering controls for SC-MPR v1.0."""

import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

from network.resnet38_cls import HFRM, Net, Net_CAM
from network.scmpr.compatibility_policy import SharedSCMPRPolicy
from network.scmpr.frequency_proposal import FixedFrequencyProposal, fixed_lowpass
from network.scmpr.scmpr_context import SCMPRContext
from network.scmpr.semantic_condition import StageSemanticCondition


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_WEIGHTS = (
    REPOSITORY_ROOT
    / "init_weights"
    / "ilsvrc-cls_rna-a1_cls1000_ep-0001.params"
)


def make_stage(channels=8, deep_channels=16, spatial=9, batch=2, device="cpu"):
    feature = torch.randn(
        batch, channels, spatial, spatial, device=device
    )
    deep = torch.randn(batch, deep_channels, 5, 5, device=device)
    logits = torch.randn(batch, 4, 5, 5, device=device)
    shared = SharedSCMPRPolicy(
        deep_channels=deep_channels, projection_dim=32
    ).to(device)
    context = SCMPRContext(channels).to(device)
    original_ch = torch.randn_like(feature)
    return feature, deep, logits, shared, context, original_ch


class SCMPRFrozenControlsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260816)

    def test_01_default_ch_is_exact_explicit_ch(self):
        torch.manual_seed(9)
        default = Net(n_class=4)
        torch.manual_seed(9)
        explicit = Net(n_class=4, context_mode="ch")
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for key in default.state_dict():
            self.assertTrue(torch.equal(default.state_dict()[key], explicit.state_dict()[key]))
        image = torch.randn(1, 3, 64, 64)
        default.eval(); explicit.eval()
        with torch.no_grad():
            left = default(image)
            right = explicit(image)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))

    def test_02_constant_input_has_zero_frequency_residuals(self):
        proposal = FixedFrequencyProposal()
        residuals, _ = proposal(torch.full((2, 5, 11, 13), 3.25))
        self.assertTrue(torch.equal(residuals["fine"], torch.zeros_like(residuals["fine"])))
        self.assertTrue(torch.equal(residuals["morphology"], torch.zeros_like(residuals["morphology"])))

    def test_03_fixed_lowpass_preserves_constants(self):
        feature = torch.full((2, 3, 10, 12), -1.75)
        for kernel in (3, 15):
            self.assertTrue(torch.equal(fixed_lowpass(feature, kernel), feature))

    def test_04_frequency_proposals_are_finite_and_normalized(self):
        residuals, qualities = FixedFrequencyProposal()(torch.randn(2, 7, 13, 11))
        for tensor in (*residuals.values(), *qualities.values()):
            self.assertTrue(torch.isfinite(tensor).all())
        for quality in qualities.values():
            self.assertGreaterEqual(quality.min().item(), 0.0)
            self.assertLessEqual(quality.max().item(), 5.0)
            self.assertAlmostEqual(quality.mean().item(), 1.0, delta=0.08)

    def _semantic_maps(self):
        target = torch.randn(2, 8, 9, 7)
        deep = torch.randn(2, 16, 5, 4)
        logits = torch.randn(2, 4, 5, 4)
        conditioner = StageSemanticCondition(8)
        deep_projector = torch.nn.Conv2d(16, 32, 1)
        return conditioner(target, deep, logits, deep_projector)

    def test_05_confidence_is_bounded(self):
        confidence = self._semantic_maps()["confidence"]
        self.assertGreaterEqual(confidence.min().item(), 0.0)
        self.assertLessEqual(confidence.max().item(), 1.0)

    def test_06_uncertainty_is_bounded(self):
        uncertainty = self._semantic_maps()["uncertainty"]
        self.assertGreaterEqual(uncertainty.min().item(), 0.0)
        self.assertLessEqual(uncertainty.max().item(), 1.0)

    def test_07_compatibility_is_bounded(self):
        compatibility = self._semantic_maps()["compatibility"]
        self.assertGreaterEqual(compatibility.min().item(), 0.0)
        self.assertLessEqual(compatibility.max().item(), 1.0)

    def test_08_semantic_inputs_are_stop_gradient_but_projectors_learn(self):
        target = torch.randn(2, 8, 9, 7, requires_grad=True)
        deep = torch.randn(2, 16, 5, 4, requires_grad=True)
        logits = torch.randn(2, 4, 5, 4, requires_grad=True)
        conditioner = StageSemanticCondition(8)
        deep_projector = torch.nn.Conv2d(16, 32, 1)
        maps = conditioner(target, deep, logits, deep_projector)
        maps["compatibility"].mean().backward()
        self.assertIsNone(target.grad)
        self.assertIsNone(deep.grad)
        self.assertIsNone(logits.grad)
        self.assertGreater(conditioner.target_projector.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(deep_projector.weight.grad.abs().sum().item(), 0.0)

    def test_09_gate_initialization_is_point_one(self):
        feature, deep, logits, shared, context, original_ch = make_stage()
        _, diagnostics = context(
            feature, original_ch, deep, logits, shared, return_diagnostics=True
        )
        self.assertTrue(torch.allclose(diagnostics["gate_fine"], torch.full_like(diagnostics["gate_fine"], 0.1), atol=1e-7))
        self.assertTrue(torch.allclose(diagnostics["gate_morphology"], torch.full_like(diagnostics["gate_morphology"], 0.1), atol=1e-7))

    def test_10_all_stages_use_one_shared_policy_object(self):
        model = Net(n_class=4, context_mode="sc-mpr")
        model.eval()
        with torch.no_grad():
            _, diagnostics = model.forward_with_diagnostics(torch.randn(1, 3, 64, 64))
        policy_ids = {
            diagnostics["scmpr"][stage]["shared_policy_id"]
            for stage in ("stage1", "stage2", "stage3")
        }
        self.assertEqual(policy_ids, {id(model.scmpr_shared)})

    def test_11_residual_is_spatially_demeaned(self):
        feature, deep, logits, shared, context, original_ch = make_stage()
        _, diagnostics = context(
            feature, original_ch, deep, logits, shared, return_diagnostics=True
        )
        spatial_mean = diagnostics["delta_zero_mean"].float().mean(dim=(-2, -1))
        self.assertLess(spatial_mean.abs().max().item(), 1e-6)

    def test_12_beta_initialization_and_bound(self):
        context = SCMPRContext(8)
        self.assertAlmostEqual(context.beta.item(), 0.1, places=7)
        self.assertAlmostEqual(context.beta_logit.item(), math.log(0.2 / 0.8), places=7)
        with torch.no_grad():
            context.beta_logit.fill_(100.0)
        self.assertGreaterEqual(context.beta.item(), 0.0)
        self.assertLessEqual(context.beta.item(), 0.5)

    def test_13_zero_beta_exactly_degenerates_to_ch(self):
        feature, deep, logits, shared, context, original_ch = make_stage()
        with torch.no_grad():
            context.beta_logit.fill_(-torch.inf)
        output = context(feature, original_ch, deep, logits, shared)
        self.assertTrue(torch.equal(output, original_ch))

    def test_14_real_hierarchy_shapes_are_preserved(self):
        deep = torch.randn(1, 4096, 4, 4)
        logits = torch.randn(1, 4, 4, 4)
        shared = SharedSCMPRPolicy()
        for channels, spatial in ((256, 8), (512, 4), (1024, 4)):
            hfrm = HFRM(channels, context_mode="sc-mpr")
            feature = torch.randn(1, channels, spatial, spatial)
            output = hfrm(feature, deep, logits, shared)
            self.assertEqual(output.shape, feature.shape)
            self.assertTrue(torch.isfinite(output).all())

    def test_15_all_public_cams_are_finite(self):
        model = Net_CAM(n_class=4, context_mode="sc-mpr")
        model.eval()
        with torch.no_grad():
            cams = model.forward_cam(torch.randn(1, 3, 64, 64))
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in cams))

    def test_16_optimizer_groups_cover_every_parameter_once(self):
        model = Net(n_class=4, context_mode="sc-mpr")
        groups = model.get_parameter_groups()
        grouped_ids = [id(parameter) for group in groups for parameter in group]
        trainable_ids = [id(parameter) for parameter in model.parameters() if parameter.requires_grad]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(trainable_ids))

    def test_17_backbone_and_batchnorm_freezing_is_unchanged(self):
        model = Net(n_class=4, context_mode="sc-mpr")
        running_mean = model.bn7.running_mean.detach().clone()
        model.train()
        with torch.no_grad():
            model(torch.randn(1, 3, 64, 64))
        self.assertFalse(model.conv1a.weight.requires_grad)
        self.assertFalse(model.bn7.weight.requires_grad)
        self.assertFalse(model.bn7.training)
        self.assertTrue(torch.equal(model.bn7.running_mean, running_mean))
        self.assertTrue(all(parameter.requires_grad for parameter in model.scmpr_shared.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_18_five_step_cuda_gradient_opening(self):
        device = "cuda"
        hfrm = HFRM(8, deep_channels=16, context_mode="sc-mpr").to(device)
        shared = SharedSCMPRPolicy(deep_channels=16).to(device)
        optimizer = torch.optim.SGD(
            list(hfrm.parameters()) + list(shared.parameters()), lr=0.05, momentum=0.9
        )
        feature = torch.randn(2, 8, 9, 9, device=device)
        deep = torch.randn(2, 16, 5, 5, device=device)
        logits = torch.randn(2, 4, 5, 5, device=device)
        active_step = None
        for step in range(1, 6):
            optimizer.zero_grad(set_to_none=True)
            output = hfrm(feature, deep, logits, shared)
            output.square().mean().backward()
            watched = (
                hfrm.scmpr_context.semantic_condition.target_projector.weight,
                shared.deep_projector.weight,
                shared.gate_policy[0].weight,
                shared.gate_policy[-1].weight,
                hfrm.scmpr_context.beta_logit,
            )
            for parameter in watched:
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            if all(parameter.grad is not None and parameter.grad.abs().sum().item() > 0 for parameter in watched):
                active_step = active_step or step
            optimizer.step()
        self.assertIsNotNone(active_step)
        self.assertLessEqual(active_step, 3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_19_batch20_bf16_official_path_smoke(self):
        model = Net_CAM(n_class=4, context_mode="sc-mpr").cuda()
        model.train()
        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=0.001,
            momentum=0.0005,
        )
        images = torch.randn(20, 3, 224, 224, device="cuda")
        labels = torch.randint(0, 2, (20, 4), device="cuda").float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(images)
            loss = sum(
                weight * F.multilabel_soft_margin_loss(output, labels)
                for weight, output in zip((0.10, 0.15, 0.25, 0.50), outputs[:4])
            )
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            cams = model.forward_cam(images[:1])
        self.assertTrue(all(torch.isfinite(tensor).all() for tensor in cams))
        del model, optimizer, images, labels, outputs, cams
        torch.cuda.empty_cache()

    def test_20_pretrained_conversion_has_no_new_backbone_missing_keys(self):
        if not PRETRAINED_WEIGHTS.is_file():
            self.skipTest("Released MXNet weights are not present")
        try:
            from network.resnet38d import convert_mxnet_to_torch
            converted = convert_mxnet_to_torch(str(PRETRAINED_WEIGHTS))
        except ImportError as error:
            self.skipTest(f"MXNet is unavailable: {error}")
        model = Net(n_class=4, context_mode="sc-mpr")
        incompatible = model.load_state_dict(converted, strict=False)
        allowed_prefixes = (
            "hfrm_56.", "hfrm_28_1.", "hfrm_28_2.", "scmpr_shared.",
            "ic_56.", "ic1.", "ic2.", "fc8.", "bn45.", "bn52.",
        )
        unexpected_backbone = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        self.assertEqual(unexpected_backbone, [])
        self.assertEqual(incompatible.unexpected_keys, [])


if __name__ == "__main__":
    unittest.main()
