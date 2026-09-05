import inspect

import torch

from network.hqrf_net import HQRFNet
from network.hqrf_pmec import MAX_REFERENCE_GROUPS, TAU_BIN, TAU_HIGH, TAU_LOW, pmec, reference_overlap_ratio
from network.hqrf_query import CHPF, DualConfidenceAllocator, MaskEmbedding, PatchQueries, PixelDecoder, QueryDecoderLayer
from network.hqrf_targets import FULL25_STEPS, circular_locality, masked_query_bce, radius_at_step, tri_state_targets


def test_patch_embed_196_queries():
    module=PatchQueries(); assert module(torch.randn(2,3,224,224)).shape==(2,196,256)


def test_query_positional_embedding():
    module=PatchQueries(); assert module.position.shape==(1,196,256) and module.position.requires_grad


def test_decoder1_self_attention():
    module=QueryDecoderLayer(); output,detail=module(torch.randn(2,196,256),torch.randn(2,49,256)); assert output.shape==(2,196,256) and detail["self_attention"].shape==(2,196,196)


def test_decoder1_cross_attention_fd():
    module=QueryDecoderLayer(); _,detail=module(torch.randn(2,196,256),torch.randn(2,49,256)); assert detail["cross_attention"].shape==(2,196,49)


def test_fd_memory_detached():
    model=HQRFNet(); image=torch.randn(1,3,224,224); fd=torch.randn(1,4096,2,2,requires_grad=True); f5=torch.randn(1,1024,2,2,requires_grad=True)
    query,_=model.query_decode(image,fd,f5); query.sum().backward(); assert fd.grad is None and f5.grad is None and model.semantic_projection[0].weight.grad is not None


def test_decoder2_cross_attention_chf5():
    model=HQRFNet(); query,detail=model.query_decode(torch.randn(1,3,224,224),torch.randn(1,4096,2,2),torch.randn(1,1024,3,3)); assert query.shape==(1,196,256) and detail["decoder2_attention"]["cross_attention"].shape[-1]==9


def test_f5_memory_detached():
    model=HQRFNet(); f5=torch.randn(1,1024,2,2,requires_grad=True); query,_=model.query_decode(torch.randn(1,3,224,224),torch.randn(1,4096,2,2,requires_grad=True),f5); query.square().mean().backward(); assert f5.grad is None and model.context_projection[0].weight.grad is not None


def test_chpf_f5_dwconv15():
    module=CHPF(256); assert module.context.kernel_size==(15,15) and module.context.groups==256 and module.context.bias is None


def test_chpf_f4_dwconv15():
    module=PixelDecoder(); assert module.f4_chpf.context.kernel_size==(15,15) and module.f4_chpf.context.groups==128


def test_gamma_zero_init():
    assert CHPF(8).gamma.detach().item()==0


def test_no_chpf_f3():
    module=PixelDecoder(); assert not hasattr(module,"f3_chpf")


def test_pixel_decoder_f4_f3():
    module=PixelDecoder(); output,detail=module(torch.randn(2,512,7,7),torch.randn(2,256,14,14)); assert output.shape==(2,256,14,14) and detail["F3_raw"].shape==(2,128,14,14)


def test_mask_embed_shape():
    assert MaskEmbedding()(torch.randn(2,196,256)).shape==(2,196,256)


def test_query_mask_einsum():
    assert torch.einsum("bqd,bdhw->bqhw",torch.randn(2,196,256),torch.randn(2,256,14,14)).shape==(2,196,14,14)


def test_mask_logits_not_sigmoid_before_bce():
    source=inspect.getsource(masked_query_bce); assert "binary_cross_entropy_with_logits(selected_logits" in source and "selected_logits.sigmoid" not in source


def test_pca_class_softmax_dim():
    output=DualConfidenceAllocator()(torch.randn(2,196,256)); assert torch.allclose(output["p_class"].sum(-1),torch.ones(2,196),atol=1e-6)


def test_pca_query_softmax_dim():
    output=DualConfidenceAllocator()(torch.randn(2,196,256)); assert torch.allclose(output["p_patch"].sum(1),torch.ones(2,4),atol=1e-6)


def test_pca_sum_prediction():
    output=DualConfidenceAllocator()(torch.randn(2,196,256)); assert torch.allclose(output["joint"].sum(1),output["image_probability"],atol=1e-6)


def _target_fixture():
    cam=torch.zeros(1,4,4,4); cam[0,0,0,0]=1; cam[0,1,3,3]=1; return cam,torch.tensor([[1,1,0,0]],dtype=torch.bool)


def test_tristate_positive():
    target,_=tri_state_targets(*_target_fixture()); assert int(target[0,0,0,0])==1


def test_tristate_rival_negative():
    target,_=tri_state_targets(*_target_fixture()); assert int(target[0,0,3,3])==0


def test_tristate_background_negative():
    target,_=tri_state_targets(*_target_fixture()); assert int(target[0,0,0,3])==0


def test_uncertain_ignore():
    cam,present=_target_fixture(); cam[0,0,1,1]=.4; target,_=tri_state_targets(cam,present); assert int(target[0,0,1,1])==-1


def test_locality_radius():
    first=circular_locality(14,(56,56),1); fifth=circular_locality(14,(56,56),5); assert first.shape==(196,56,56) and torch.all(first<=fifth) and first.sum()<fifth.sum()


def test_radius_schedule_uses_full25_denominator():
    assert radius_at_step(0)==1 and radius_at_step(FULL25_STEPS)==5
    try: radius_at_step(1,3513)
    except ValueError: pass
    else: raise AssertionError("non-Full25 denominator accepted")


def test_pmec_tau_bin070(): assert TAU_BIN==.70
def test_pmec_ror_low040_high050(): assert (TAU_LOW,TAU_HIGH)==(.40,.50)
def test_pmec_T5(): assert MAX_REFERENCE_GROUPS==5


def test_pmec_reference_area_denominator():
    reference=torch.tensor([[1,1],[0,0]],dtype=torch.bool); candidates=torch.tensor([[[1,0],[0,0]],[[1,1],[1,1]]],dtype=torch.bool); assert torch.allclose(reference_overlap_ratio(candidates,reference),torch.tensor([.5,1.]))


def test_pmec_nontrivial_grouping():
    logits=torch.full((1,3,2,2),-10.); logits[0,0,0,:]=10; logits[0,1,:,0]=10; logits[0,2,1,:]=10
    p_class=torch.tensor([[[.9,.1,0,0],[.9,.1,0,0],[.9,.1,0,0]]]); joint=p_class/3; present=torch.tensor([[1,0,0,0]],dtype=torch.bool); locality=torch.ones(3,2,2,dtype=torch.bool)
    _,rows=pmec(logits,joint,p_class,present,locality); assert rows[0]["candidate_masks"]==3 and rows[0]["region_groups"]>=1


def test_no_hcrf_local_qk():
    source=inspect.getsource(HQRFNet); assert "local q" not in source.lower() and "masks_for" not in source


def test_no_cross_hierarchy_region_state():
    source=inspect.getsource(HQRFNet.forward); assert "region_state" not in source and "hard_assembly" not in source


def test_no_island_track_logic():
    source=inspect.getsource(HQRFNet); assert "island" not in source and "track" not in source


def test_full_model_forward_and_backward():
    model=HQRFNet().train(); image=torch.randn(1,3,224,224); labels=torch.tensor([[1.,1.,0.,0.]])
    output=model(image,labels,step=0,run_pmec=True)
    assert output["mask_logits"].shape==(1,196,56,56) and output["pmec_region"].shape==(1,4,56,56)
    output["losses"]["loss"].backward()
    assert model.patch_queries.projection.weight.grad is not None and model.semantic_projection[0].weight.grad is not None
