import math
import unittest

import torch
import torch.nn.functional as F

from network.fampr.adaptive_kernel import AdaptiveKernelSpectrum
from network.fampr.adaptive_sampler import SpatiallyAdaptiveDepthwiseSampler
from network.fampr.fampr_context import FrequencyAdaptiveMorphologyContext
from network.fampr.frequency_selection import MultiBandFrequencySelector
from network.resnet38_cls import HFRM, Net, Net_CAM


class FAMPRComponentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260816)

    def test_telescoping_bands_reconstruct_input(self):
        selector = MultiBandFrequencySelector()
        feature = torch.randn(2, 7, 13, 11)
        bands = selector.decompose(feature)
        reconstruction = torch.stack(bands).sum(dim=0)
        self.assertLess((reconstruction - feature).abs().max().item(), 1e-6)

    def test_frequency_selector_is_exact_identity_at_initialization(self):
        selector = MultiBandFrequencySelector()
        feature = torch.randn(2, 7, 13, 11)
        selected, morphology, diagnostics = selector(feature)

        self.assertTrue(torch.equal(selected, feature))
        self.assertTrue(
            torch.equal(
                diagnostics["band_weights"],
                torch.ones_like(diagnostics["band_weights"]),
            )
        )
        self.assertGreaterEqual(morphology.min().item(), 0.0)
        self.assertLessEqual(morphology.max().item(), 1.0)

    def test_morphology_and_dilation_are_bounded_and_inverse(self):
        selector = MultiBandFrequencySelector()
        feature = torch.randn(2, 4, 12, 10)
        _, morphology, _ = selector(feature)
        dilation = 1.0 + (1.0 - morphology) * 6.0

        self.assertGreaterEqual(dilation.min().item(), 1.0)
        self.assertLessEqual(dilation.max().item(), 7.0)
        order = torch.argsort(morphology.flatten())
        sorted_dilation = dilation.flatten()[order]
        self.assertTrue(torch.all(sorted_dilation[:-1] >= sorted_dilation[1:]))

    def test_integer_sampler_matches_replicate_depthwise_convolution(self):
        channels = 5
        feature = torch.randn(2, channels, 9, 8)
        kernel_low = torch.randn(channels, 1, 3, 3)
        kernel_high = torch.randn(channels, 1, 3, 3)
        dilation = torch.ones(2, 1, 9, 8)
        sampler = SpatiallyAdaptiveDepthwiseSampler()

        sampled_low, sampled_high = sampler(
            feature, dilation, kernel_low, kernel_high
        )
        padded = F.pad(feature, (1, 1, 1, 1), mode="replicate")
        reference_low = F.conv2d(padded, kernel_low, groups=channels)
        reference_high = F.conv2d(padded, kernel_high, groups=channels)

        self.assertLess((sampled_low - reference_low).abs().max().item(), 1e-4)
        self.assertLess(
            (sampled_high - reference_high).abs().max().item(), 1e-4
        )

    def test_kernel_decomposition_and_neutral_gates(self):
        module = AdaptiveKernelSpectrum(channels=12)
        feature = torch.randn(3, 12, 7, 9)
        kernel_low, kernel_high, gate_low, gate_high = module(feature)

        self.assertTrue(
            torch.equal(kernel_low + kernel_high, module.base_kernel)
        )
        self.assertTrue(torch.equal(gate_low, torch.ones_like(gate_low)))
        self.assertTrue(torch.equal(gate_high, torch.ones_like(gate_high)))
        self.assertTrue(
            torch.allclose(
                module.base_kernel.sum(dim=(-2, -1)),
                torch.ones(12, 1),
            )
        )

    def test_zero_anchor_exactly_falls_back_to_original_ch(self):
        module = FrequencyAdaptiveMorphologyContext(channels=8)
        with torch.no_grad():
            module.anchor_logit.fill_(-torch.inf)
        feature = torch.randn(2, 8, 9, 7)
        original_ch = torch.randn_like(feature)
        output = module(feature, original_ch)
        self.assertTrue(torch.equal(output, original_ch))

    def test_fixed_anchor_initialization(self):
        module = FrequencyAdaptiveMorphologyContext(channels=8)
        self.assertAlmostEqual(module.anchor_lambda.item(), 0.25, places=7)
        self.assertAlmostEqual(
            module.anchor_logit.item(), math.log(0.25 / 0.75), places=7
        )


class FAMPRIntegrationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260816)

    def test_real_hierarchy_shapes_are_preserved(self):
        stages = (
            (256, 8, 8),
            (512, 4, 4),
            (1024, 4, 4),
        )
        deep = torch.randn(1, 4096, 4, 4)
        for channels, height, width in stages:
            module = HFRM(
                in_channels=channels,
                deep_channels=4096,
                context_mode="fampr",
            )
            feature = torch.randn(1, channels, height, width)
            output, diagnostics = module.forward_with_diagnostics(feature, deep)
            self.assertEqual(output.shape, feature.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(diagnostics["fampr"]["all_finite"])

    def test_fampr_model_does_not_instantiate_archived_hst(self):
        model = Net(n_class=4, rectifier_type="hfrm", context_mode="fampr")
        self.assertFalse(hasattr(model, "hst_rectifier"))
        self.assertFalse(
            any("hst" in name.lower() for name, _ in model.named_parameters())
        )
        self.assertEqual(model.hfrm_56.context_mode, "fampr")
        self.assertEqual(model.hfrm_28_1.context_mode, "fampr")
        self.assertEqual(model.hfrm_28_2.context_mode, "fampr")

    def test_archived_hst_rejects_fampr_context(self):
        with self.assertRaisesRegex(ValueError, "archived HST"):
            Net(n_class=4, rectifier_type="hst", context_mode="fampr")

    def test_optimizer_groups_cover_every_trainable_parameter_once(self):
        model = Net(n_class=4, rectifier_type="hfrm", context_mode="fampr")
        groups = model.get_parameter_groups()
        grouped_ids = [id(parameter) for group in groups for parameter in group]
        trainable_ids = [
            id(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), set(trainable_ids))

    def test_five_steps_open_frequency_adaptive_path(self):
        module = HFRM(
            in_channels=8,
            deep_channels=16,
            context_mode="fampr",
        )
        optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.9)
        feature = torch.randn(2, 8, 9, 9)
        deep = torch.randn(2, 16, 5, 5)
        active_by_step = None

        for step in range(1, 6):
            optimizer.zero_grad(set_to_none=True)
            output, diagnostics = module.forward_with_diagnostics(feature, deep)
            loss = output.square().mean()
            loss.backward()

            final_band = (
                module.fampr_context.frequency_selector.band_weight_network[-1]
            )
            final_gate = module.fampr_context.adaptive_kernel.gate_network[-1]
            watched = (
                final_band.weight,
                final_gate.weight,
                module.fampr_context.adaptive_kernel.base_kernel,
                module.fampr_context.anchor_logit,
            )
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(diagnostics["fampr"]["all_finite"])
            for parameter in watched:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

            if all(parameter.grad.abs().sum().item() > 0.0 for parameter in watched):
                active_by_step = active_by_step or step
            optimizer.step()

        self.assertIsNotNone(active_by_step)
        self.assertLessEqual(active_by_step, 3)

    def test_full_forward_diagnostics_and_forward_cam_are_finite(self):
        model = Net(n_class=4, context_mode="fampr")
        cam_model = Net_CAM(n_class=4, context_mode="fampr")
        cam_model.load_state_dict(model.state_dict())
        model.eval()
        cam_model.eval()
        image = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            outputs, diagnostics = model.forward_with_diagnostics(image)
            cams = cam_model.forward_cam(image)

        self.assertEqual(set(diagnostics["fampr"]), {"stage1", "stage2", "stage3"})
        for tensor in outputs:
            self.assertTrue(torch.isfinite(tensor).all())
        for tensor in cams:
            self.assertTrue(torch.isfinite(tensor).all())


if __name__ == "__main__":
    unittest.main()
