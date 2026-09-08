from pathlib import Path

import torch

from tools.fomd_counterfactuals import fixed_derangement, materialize, permutation_payload


ROOT=Path(__file__).resolve().parents[1]


def _stage():
    torch.manual_seed(4); a=torch.rand(2,7,4,5,5); a=a/a.sum(1,keepdim=True); b=torch.rand(2,7,5,5)
    full=(a*b[:,:,None]).sum(1); joint=torch.rand(2,7,4)
    return {"momd":{"routing":a,"base_probability":b,"mixture":full},"confidence":{"joint":joint}}


def test_fomd_no_training_equation_change():
    source=(ROOT/"train_fomd_phase0.py").read_text()
    assert "model=MOMDNet()" in source and "loss=result[\"losses\"][\"loss\"]" in source


def test_fomd_no_aux_loss():
    source=(ROOT/"train_fomd_phase0.py").read_text()
    assert '"auxiliary":False' in source


def test_fomd_no_oec():
    source=(ROOT/"train_fomd_phase0.py").read_text()
    assert "OECNet" not in source and "loss_oec" not in source


def test_fomd_full_equals_momd_F():
    stage=_stage(); out=materialize(stage,permutation_payload(7)["permutations"])
    assert torch.equal(out["full"],stage["momd"]["mixture"])


def test_perm_bank_fixed():
    assert permutation_payload(20)==permutation_payload(20)


def test_perm_is_derangement():
    bank=permutation_payload(30)
    assert bank["all_derangements"] and all(all(i!=v for i,v in enumerate(p)) for p in bank["permutations"])


def test_perm_breaks_query_identity():
    stage=_stage(); out=materialize(stage,permutation_payload(7)["permutations"])
    assert float((out["perm"]-out["full"][None]).abs().mean())>0


def test_perm_preserves_A_sum():
    stage=_stage(); p=fixed_derangement(7,1001)
    assert torch.allclose(stage["momd"]["routing"][:,p].sum(1),torch.ones(2,4,5,5),atol=1e-6)


def test_perm_detached():
    out=materialize(_stage(),permutation_payload(7)["permutations"])
    assert not out["perm"].requires_grad


def test_global_A_spatial_mean_and_query_renorm():
    stage=_stage(); out=materialize(stage,permutation_payload(7)["permutations"])
    expected=stage["momd"]["routing"].mean((-2,-1),keepdim=True); expected/=expected.sum(1,keepdim=True)
    assert torch.allclose(out["global_A"],expected) and torch.allclose(out["global_A"].sum(1),torch.ones_like(out["global_A"].sum(1)))


def test_global_detached():
    assert not materialize(_stage(),permutation_payload(7)["permutations"])["global"].requires_grad


def test_pca_reference_exact_and_detached():
    stage=_stage(); out=materialize(stage,permutation_payload(7)["permutations"]); j=stage["confidence"]["joint"]; alpha=j/j.sum(1,keepdim=True)
    expected=(alpha[...,None,None]*stage["momd"]["base_probability"][:,:,None]).sum(1)
    assert torch.allclose(out["pca"],expected) and not out["pca"].requires_grad


def test_uniform_reference():
    stage=_stage(); out=materialize(stage,permutation_payload(7)["permutations"])
    assert torch.allclose(out["uniform"][:,0],stage["momd"]["base_probability"].mean(1))


def test_no_naive_A_only():
    assert "A_only" not in (ROOT/"tools/fomd_counterfactuals.py").read_text()


def test_counterfactuals_not_in_loss():
    source=(ROOT/"train_fomd_phase0.py").read_text()
    assert "materialize" not in source[source.index("loss=result"):source.index("loss.backward")]


def test_smoke_materializes_counterfactual_audit():
    source=(ROOT/"train_fomd_phase0.py").read_text()
    assert 'snapshot=SNAPSHOTS.get(step,"smoke_step2")' in source


def test_zero_parameter_delta():
    assert not (ROOT/"network/fomd_net.py").exists()


def test_no_val_test_path():
    source=(ROOT/"train_fomd_phase0.py").read_text()
    assert "Stage1_TrainDataset" in source and "ValDataset" not in source and "TestDataset" not in source
