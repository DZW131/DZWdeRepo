import inspect

import torch

from network.fqrf_net import FQRFNet, STAGE_WEIGHTS
from network.fqrf_query import CrossFirstFocusDecoder, DynamicFocusPosition, FocusPatchQueries, previous_mask_visibility, sine_position_2d
from network.hqrf_backbone import FEATURE_TAPS, HQRFBackbone
from network.hqrf_pmec import MAX_REFERENCE_GROUPS, TAU_BIN, TAU_HIGH, TAU_LOW
from network.hqrf_query import CHPF, MaskEmbedding, PixelDecoder
from network.hqrf_targets import FULL25_STEPS, masked_query_bce, radius_at_step, tri_state_targets


def test_plain_resnet38_and_feature_taps():
    assert issubclass(HQRFBackbone, torch.nn.Module)
    assert {key: FEATURE_TAPS[key]["channels"] for key in ("FD", "F5", "F4", "F3")} == {"FD": 4096, "F5": 1024, "F4": 512, "F3": 256}


def test_patch_queries_196_and_base_position():
    content, base = FocusPatchQueries()(torch.randn(2, 3, 224, 224))
    assert content.shape == base.shape == (2, 196, 256)
    assert FocusPatchQueries().base_position.requires_grad


def test_parameter_free_memory_position():
    position = sine_position_2d(2, 7, 7, 256, torch.device("cpu"))
    assert position.shape == (2, 49, 256) and not position.requires_grad and torch.isfinite(position).all()


def test_all_decoders_are_cross_attention_first():
    model = FQRFNet()
    expected = ("cross_attention", "norm", "self_attention", "norm", "ffn", "norm")
    assert all(decoder.operation_order == expected for decoder in (model.decoder1, model.decoder2, model.decoder3))


def test_decoder_returns_cross_attention():
    decoder = CrossFirstFocusDecoder(dropout=0)
    output, detail = decoder(torch.randn(2, 6, 256), torch.randn(2, 9, 256))
    assert output.shape == (2, 6, 256) and detail["cross_attention"].shape == (2, 6, 9)


def test_dynamic_focus_uses_detached_attention_and_keeps_base():
    module = DynamicFocusPosition()
    attention = torch.rand(1, 4, 9, requires_grad=True)
    memory_position = torch.randn(1, 9, 256)
    base = torch.randn(1, 4, 256, requires_grad=True)
    result = module(attention, memory_position, base); result.sum().backward()
    assert result.shape == base.shape and attention.grad is None and base.grad is not None
    assert module.layers[0].weight.grad is not None


def test_previous_mask_threshold_and_detach():
    logits = torch.full((1, 2, 4, 4), -10.0, requires_grad=True)
    logits.data[0, 0, 0, 0] = 10.0
    visible, health = previous_mask_visibility(logits, (2, 2), threshold=.15)
    assert visible.shape == (1, 2, 4) and not visible.requires_grad
    assert bool(health["fallback_to_global"][0, 1]) and bool(visible[0, 1].all())


def test_wrong_attention_threshold_rejected():
    try: previous_mask_visibility(torch.zeros(1, 1, 2, 2), (1, 1), threshold=.2)
    except ValueError: pass
    else: raise AssertionError("non-frozen attention threshold accepted")


def test_fd_f5_f4_memories_detached_but_projections_trainable():
    model = FQRFNet().train()
    image = torch.randn(1, 3, 224, 224)
    fd = torch.randn(1, 4096, 2, 2, requires_grad=True)
    f5 = torch.randn(1, 1024, 2, 2, requires_grad=True)
    f4 = torch.randn(1, 128, 2, 2, requires_grad=True)
    initial = model.query_decode(image, fd, f5, f4)
    pixel = torch.randn(1, 256, 8, 8)
    stages, _ = model._finish_query_decode(initial, pixel, f4)
    stages[-1]["query"].square().mean().backward()
    assert fd.grad is None and f5.grad is None and f4.grad is None
    assert model.semantic_projection[0].weight.grad is not None
    assert model.context_projection[0].weight.grad is not None
    assert model.f4_memory_projection[0].weight.grad is not None


def test_chpf_and_raw_f3_contract():
    model = FQRFNet()
    assert model.f5_chpf.context.kernel_size == (15, 15) and model.f5_chpf.context.groups == 256
    assert model.pixel_decoder.f4_chpf.context.groups == 128 and not hasattr(model.pixel_decoder, "f3_chpf")
    assert CHPF(8).gamma.detach().item() == 0


def test_pixel_decoder_f4_f3():
    output, detail = PixelDecoder()(torch.randn(2, 512, 7, 7), torch.randn(2, 256, 14, 14))
    assert output.shape == (2, 256, 14, 14) and detail["F4_context"].shape == (2, 128, 7, 7)


def test_three_independent_mask_and_pca_heads():
    model = FQRFNet()
    assert len(model.mask_embeddings) == len(model.pca_heads) == 3
    assert len({id(head) for head in model.mask_embeddings}) == 3
    assert all(isinstance(head, MaskEmbedding) for head in model.mask_embeddings)


def test_stage_loss_weights():
    assert STAGE_WEIGHTS == (.20, .30, .50) and sum(STAGE_WEIGHTS) == 1.0


def test_tristate_supervision_unchanged():
    cam = torch.zeros(1, 4, 4, 4); cam[0, 0, 0, 0] = 1; cam[0, 1, 3, 3] = 1
    target, _ = tri_state_targets(cam, torch.tensor([[1, 1, 0, 0]], dtype=torch.bool))
    assert int(target[0, 0, 0, 0]) == 1 and int(target[0, 0, 3, 3]) == 0 and int(target[0, 0, 0, 3]) == 0


def test_mask_bce_uses_raw_logits():
    source = inspect.getsource(masked_query_bce)
    assert "binary_cross_entropy_with_logits(selected_logits" in source and "selected_logits.sigmoid" not in source


def test_locality_uses_full25_denominator():
    assert FULL25_STEPS == 29275 and radius_at_step(0) == 1
    try: radius_at_step(1, 3513)
    except ValueError: pass
    else: raise AssertionError("Phase-0 denominator accepted")


def test_pmec_frozen_and_stage3_only():
    assert (TAU_BIN, TAU_LOW, TAU_HIGH, MAX_REFERENCE_GROUPS) == (.70, .40, .50, 5)
    source = inspect.getsource(FQRFNet.forward)
    assert 'final = stages[-1]' in source and 'final["mask_logits"].detach()' in source


def test_no_prohibited_rescue_losses_or_hcrf_state():
    source = inspect.getsource(FQRFNet).lower()
    for token in ("diversity_loss", "repulsion", "orthogonal", "region_state", "hard_assembly", "island"):
        assert token not in source


def test_full_query_path_outputs_three_masks_and_attentions():
    model = FQRFNet().eval()
    image = torch.randn(1, 3, 224, 224)
    initial = model.query_decode(image, torch.randn(1, 4096, 4, 4), torch.randn(1, 1024, 4, 4), torch.randn(1, 128, 4, 4))
    stages, detail = model._finish_query_decode(initial, torch.randn(1, 256, 16, 16), torch.randn(1, 128, 4, 4))
    assert len(stages) == 3 and all(stage["mask_logits"].shape == (1, 196, 16, 16) for stage in stages)
    assert stages[1]["masked_attention"] is not None and detail["Qp3"].shape == (1, 196, 256)

