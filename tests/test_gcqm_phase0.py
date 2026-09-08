from pathlib import Path
import torch

from network.gcqm import gcqm_decode
from tools.fomd_counterfactuals import permutation_payload
from tools.gcqm_counterfactuals import materialize


ROOT=Path(__file__).resolve().parents[1]


def _decoded():
    torch.manual_seed(3); logits=torch.randn(2,7,5,5,requires_grad=True); r=torch.rand(2,7,4,4); locality=torch.ones(7,5,5,dtype=torch.bool)
    return logits,gcqm_decode(logits,r,(2,2),locality,materialize=True)


def _stage():
    _,g=_decoded(); joint=torch.rand(2,7,4); alpha=joint/joint.sum(1,keepdim=True); g["pca_reference"]=(alpha[...,None,None]*g["base_probability"].detach()[:,:,None]).sum(1); return {"gcqm":g}


def test_gcqm_spatial_mean_A():
    _,g=_decoded(); expected=g["routing"].mean((-2,-1)); expected/=expected.sum(1,keepdim=True); assert torch.allclose(g["weights"],expected)


def test_gcqm_w_query_renorm_and_sum_one():
    _,g=_decoded(); assert torch.allclose(g["weights"].sum(1),torch.ones(2,4),atol=1e-6) and g["weight_sum_error_max"]<=1e-6


def test_gcqm_w_detached(): assert not _decoded()[1]["weights"].requires_grad
def test_gcqm_B_original_mask_basis():
    logits,g=_decoded(); assert torch.allclose(g["base_probability"],logits.sigmoid())


def test_gcqm_F_equals_sum_wB():
    _,g=_decoded(); assert torch.allclose(g["mixture"],torch.einsum("bqc,bqhw->bchw",g["weights"],g["base_probability"]))


def test_gcqm_C_sum_F():
    _,g=_decoded(); assert torch.allclose(g["contribution"].sum(1),g["mixture"])


def test_gcqm_F_range():
    _,g=_decoded(); assert float(g["mixture"].min())>=0 and float(g["mixture"].max())<=1


def test_gcqm_stage2_stage3_primary_and_loss():
    source=(ROOT/"network/gcqm_net.py").read_text(); assert 'stage["gcqm"]=gcqm_decode' in source and 'mixture_class_bce(stage["gcqm"]["mixture"]' in source and '"primary_output":stages[-1]["gcqm"]["mixture"]' in source


def test_gcqm_references_detached():
    out=materialize(_stage(),permutation_payload(7)["permutations"]); assert all(not x.requires_grad for x in out.values())


def test_gcqm_no_oec_pmec_primary_or_aux_loss():
    source=(ROOT/"network/gcqm_net.py").read_text()+(ROOT/"train_gcqm_phase0.py").read_text(); assert "OECNet" not in source and "loss_oec" not in source and '"new_aux_loss":0' in source and '"pmec_region"' in source


def test_gcqm_zero_parameter_delta():
    source=(ROOT/"network/gcqm.py").read_text(); assert "nn.Parameter" not in source and "torch.nn" not in source


def test_gcqm_perm_bank_fixed_and_deranged():
    a=permutation_payload(20); b=permutation_payload(20); assert a==b and a["all_derangements"]


def test_gcqm_query_identity_metric_nonzero():
    out=materialize(_stage(),permutation_payload(7)["permutations"]); assert float((out["primary"][None]-out["perm"]).abs().mean())>0


def test_gcqm_only_primary_in_training_loss():
    source=(ROOT/"network/gcqm_net.py").read_text(); segment=source[source.index("mask_losses=[]"):source.index("output={")]; assert "pixel_reference" not in segment and "pca_reference" not in segment and "perm" not in segment


def test_gcqm_no_val_test_path():
    source=(ROOT/"train_gcqm_phase0.py").read_text(); assert "Stage1_TrainDataset" in source and "ValDataset" not in source and "TestDataset" not in source
