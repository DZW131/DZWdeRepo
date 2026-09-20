"""Verify adapters contain trainable tensors only and frozen tensors replay exactly."""
from __future__ import annotations
import argparse,hashlib,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from network.pcsi_net import PCSINet
from tools.pdsr_vpca_phase0.common import write_json


def digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--plip",type=Path,required=True); p.add_argument("--concept-cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    cache=torch.load(a.concept_cache,map_location="cpu",weights_only=False); result={}
    for variant,directory in (("C1","C1_VLM_LAST"),("C2","C2_STATIC_PDSR")):
        model=PCSINet(a.checkpoint,a.plip,cache["embeddings"],variant)
        base_before=digest(model.base.state_dict()); plip_before=digest(model.plip.state_dict())
        adapter=torch.load(a.output/directory/f"{variant.lower()}_e5_adapter.pth",map_location="cpu",weights_only=False)
        allowed=("gamma","last_projection","pdsr")
        if any(not name.startswith(allowed) for name in adapter): raise AssertionError("Frozen tensor in adapter checkpoint")
        model.load_trainable_state_dict(adapter)
        base_after=digest(model.base.state_dict()); plip_after=digest(model.plip.state_dict())
        if base_before!=base_after or plip_before!=plip_after: raise AssertionError("Frozen parameter hash changed")
        trainable={name for name,p in model.named_parameters() if p.requires_grad}
        expected=set(adapter)
        if trainable!=expected: raise AssertionError(f"Trainable/adapter mismatch: {trainable^expected}")
        result[variant]={"frozen_base_sha256":base_after,"frozen_plip_sha256":plip_after,
                         "base_hash_unchanged":True,"plip_hash_unchanged":True,
                         "adapter_tensors":sorted(adapter),"trainable_parameter_count":sum(p.numel() for p in model.parameters() if p.requires_grad)}
        del model
    write_json(a.output/"phaseC/parameter_audit.json",result)
    print("PCSI_PARAMETER_AUDIT_GO",flush=True)


if __name__=="__main__": main()
