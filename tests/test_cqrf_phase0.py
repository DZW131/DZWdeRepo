import inspect
import torch

from network.cqrf_net import CQRFNet, STAGE_WEIGHTS
from network.cqrf_query import CCRALayer, CQRFPixelDecoder, FocusPatchQueries, sine_position_2d
from network.hqrf_targets import FULL25_STEPS


def _ccra_inputs(batch=2, queries=7, spatial=5):
    query=torch.randn(batch,queries,256,requires_grad=True); base=torch.randn(batch,queries,256)
    memory=torch.randn(batch,spatial,256); kp=torch.randn_like(memory)
    pclass=torch.softmax(torch.randn(batch,queries,4),-1).requires_grad_(); gate=torch.sigmoid(torch.randn(batch,4)).requires_grad_()
    return query,base,memory,kp,pclass,gate


def test_patch_query_and_position_contract():
    content,base=FocusPatchQueries()(torch.randn(2,3,224,224)); assert content.shape==base.shape==(2,196,256)
    kp=sine_position_2d(2,7,7,256,torch.device("cpu")); assert kp.shape==(2,49,256) and not kp.requires_grad


def test_responsibility_softmax_is_query_dimension():
    module=CCRALayer(dropout=0); inputs=_ccra_inputs(); _,detail=module(*inputs)
    assert detail["responsibility_class"].shape==(2,7,5,4)
    assert torch.allclose(detail["responsibility_class"].sum(1),torch.ones(2,5,4),atol=1e-6)
    assert detail["integrity_max"]<=1e-5


def test_class_prior_detached_but_ccra_trainable():
    module=CCRALayer(dropout=0); inputs=_ccra_inputs(); result,detail=module(*inputs); result.square().mean().backward()
    query,_,_,_,pclass,gate=inputs
    assert query.grad is not None and pclass.grad is None and gate.grad is None
    assert module.q_projection.weight.grad is not None and not detail["responsibility_class"].requires_grad


def test_all_gate_tiny_fallback():
    module=CCRALayer(); p=torch.softmax(torch.randn(2,7,4),-1); gate=torch.zeros(2,4)
    assert torch.allclose(module.class_prior(p,gate),p,atol=1e-6)


def test_no_stage2_or_stage3_self_attention():
    model=CQRFNet(); assert not hasattr(model.ccra2,"self_attention") and not hasattr(model.ccra3,"self_attention")
    source=inspect.getsource(CQRFNet).lower()
    for token in ("dynamicfocus", "previous_mask_visibility", "sinkhorn", "diversity_loss", "hard_assignment"):
        assert token not in source


def test_coherent_f4_and_raw_f3():
    module=CQRFPixelDecoder(); pixel,detail=module(torch.randn(2,512,7,7),torch.randn(2,256,14,14))
    assert pixel.shape==(2,256,14,14) and detail["F4_context"].shape==(2,128,7,7)
    assert module.f4_chpf.context.kernel_size==(15,15) and not hasattr(module,"f3_chpf")


def test_three_independent_heads_and_stage_weights():
    model=CQRFNet(); assert len(model.mask_embeddings)==len(model.pca_heads)==3
    assert len({id(x) for x in model.mask_embeddings})==3 and STAGE_WEIGHTS==(.20,.30,.50)


def test_full25_locality_and_pmec_stage3():
    assert FULL25_STEPS==29275
    source=inspect.getsource(CQRFNet.forward); assert 'final = stages[-1]' in source and 'final["mask_logits"].detach()' in source


def test_ccra_query_update_shapes():
    module=CCRALayer(dropout=0); result,detail=module(*_ccra_inputs(batch=1,queries=9,spatial=16))
    assert result.shape==(1,9,256) and detail["projected_update"].shape==result.shape and detail["query_delta"].shape==result.shape
