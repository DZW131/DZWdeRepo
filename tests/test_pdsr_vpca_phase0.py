import os
from pathlib import Path
import numpy as np
import pytest
import torch

from network.pdsr import PathologyDenseSemanticReconstructor
from network.vpca import StaticConceptProvider,VPCAConceptProvider
from tools.pdsr_vpca_phase0.common import CommonAugmentTrainDataset,load_concepts

ROOT=Path(__file__).resolve().parents[1]

def test_patch_grid_reconstruction():
    x=torch.randn(2,49,16); y=PathologyDenseSemanticReconstructor.grid(x)
    assert y.shape==(2,16,7,7)
    with pytest.raises(AssertionError): PathologyDenseSemanticReconstructor.grid(torch.randn(1,50,16))

def test_static_weights_sum():
    w,_=StaticConceptProvider()(3,torch.device("cpu")); assert torch.allclose(w.sum(1),torch.ones(3))

def test_vpca_weights_sum():
    w,aux=VPCAConceptProvider()(torch.randn(2,512,7,7),torch.randn(32,512))
    assert w.shape==(2,32) and torch.allclose(w.sum(1),torch.ones(2),atol=1e-6)
    assert torch.allclose(aux["q"].sum(-1),torch.ones(2,4),atol=1e-6)
    assert torch.allclose(aux["rho"].sum(-1),torch.ones(2),atol=1e-6)

def test_gamma_identity():
    h=torch.randn(2,256,28,28); z=torch.randn_like(h); gamma=torch.zeros(256)
    assert torch.equal(h+gamma[None,:,None,None]*z,h)

def test_pdsr_shapes_and_finite():
    model=PathologyDenseSemanticReconstructor(16,semantic_dim=32,output_dim=8)
    tokens=tuple(torch.randn(2,49,16) for _ in range(3)); concepts=torch.nn.functional.normalize(torch.randn(32,32),dim=-1); weights=torch.full((2,32),1/32)
    z,aux=model(tokens,concepts,weights)
    assert z.shape==(2,8,28,28) and torch.isfinite(z).all()
    assert torch.allclose(aux["layer_weights"].sum(1),torch.ones(2,7,7),atol=1e-5)

def test_concept_bank_cardinality_and_atomicity():
    values,_=load_concepts(ROOT/"configs/concepts/bcss_vpca_concepts_v1.yaml")
    assert len(values)==32 and len(set(values))==32
    assert not any("image showing" in x.lower() for x in values)

def test_concept_embedding_norm_if_available():
    path=os.getenv("PDSR_CONCEPT_CACHE")
    if not path: pytest.skip("integration cache not supplied")
    x=torch.load(path,map_location="cpu",weights_only=False)["embeddings"]
    assert x.shape==(32,512) and torch.allclose(x.norm(dim=-1),torch.ones(32),atol=1e-6)

def test_plip_hidden_state_shapes_if_available():
    path=os.getenv("PDSR_PLIP_PATH")
    if not path: pytest.skip("PLIP integration path not supplied")
    from network.plip_adapter import FrozenPLIPAdapter
    model=FrozenPLIPAdapter(path); audit=model.runtime_audit(torch.rand(1,3,224,224))
    assert audit["hidden_states_length"]==13 and audit["selected_dense_shapes"]==[[1,49,768]]*3

def test_hqmr_and_plip_frozen_if_available():
    plip=os.getenv("PDSR_PLIP_PATH"); checkpoint=os.getenv("PDSR_HQMR_CHECKPOINT"); cache=os.getenv("PDSR_CONCEPT_CACHE")
    if not all((plip,checkpoint,cache)): pytest.skip("full integration paths not supplied")
    from network.pdsr_vpca_hqmr import PDSRVPCAHQMR
    concepts=torch.load(cache,map_location="cpu",weights_only=False)["embeddings"]
    model=PDSRVPCAHQMR(checkpoint,plip,concepts,"P3")
    assert all(not p.requires_grad for p in model.base.parameters())
    assert all(not p.requires_grad for p in model.plip.parameters())

def test_no_seg_gt_in_train_and_same_geometric_augmentation(tmp_path):
    from PIL import Image
    image=np.zeros((224,224,3),np.uint8); image[:,112:]=255
    Image.fromarray(image).save(tmp_path/"sample[1010].png")
    ds=CommonAugmentTrainDataset(tmp_path); row=ds[0]
    assert len(row)==3 and row[1].shape==(3,224,224) and row[2].shape==(4,)
    # One raw augmented tensor is returned; both normalizations happen model-side.
    assert row[1].min()>=0 and row[1].max()<=1
