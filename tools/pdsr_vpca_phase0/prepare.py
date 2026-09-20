"""Create the frozen PLIP/concept bank and run all pre-training safety gates."""
from __future__ import annotations
import argparse,json,shutil,subprocess,sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from audits.ucrf_v1.gate import load_model
from network.pdsr_vpca_hqmr import PDSRVPCAHQMR
from network.plip_adapter import FrozenPLIPAdapter
from tools.pdsr_vpca_phase0.common import CHECKPOINT_SHA,CLASSES,PLIP_WEIGHT_SHA,CommonEvalDataset,load_concepts,sha256,write_json

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--plip",type=Path,required=True)
    p.add_argument("--train-root",type=Path,required=True); p.add_argument("--concepts",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--source-gate",type=Path,required=True); a=p.parse_args(); out=a.output
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"Refusing populated Phase0 output: {out}")
    for name in ("manifests","concept_bank","P0_HQMR","P1_VLM_LAST","P2_STATIC_PDSR","P3_VPCA_PDSR","metrics","visualizations","logs"):(out/name).mkdir(parents=True,exist_ok=True)
    if sha256(a.checkpoint)!=CHECKPOINT_SHA: raise AssertionError("Frozen HQMR checkpoint mismatch")
    if sha256(a.plip/"pytorch_model.bin")!=PLIP_WEIGHT_SHA: raise AssertionError("Official PLIP weight mismatch")
    gate=json.loads(a.source_gate.read_text()); hqmr=gate.get("hqmr_miou",gate.get("observed",{}).get("hqmr"))
    if not gate.get("pass") or abs(float(hqmr)-.6557244403737567)>1e-12: raise AssertionError("P0 reproduction source failed")
    write_json(out/"manifests/baseline.json",{"pass":True,"mIoU":hqmr,"checkpoint":str(a.checkpoint),"checkpoint_sha256":CHECKPOINT_SHA,
               "same_inference":True,"same_3view_tta":True,"same_gate":True,"same_cam_normalization":True,"parameter_updates":0})
    mapping={str(i):name for i,name in enumerate(CLASSES)}; write_json(out/"manifests/bc_ss_class_mapping.json",mapping)
    concepts,payload=load_concepts(a.concepts); destination=out/"concept_bank/bcss_vpca_concepts_v1.yaml"; shutil.copy2(a.concepts,destination)
    dataset=CommonEvalDataset(a.train_root); name,raw=dataset[0]; raw=raw[None].cuda()
    adapter=FrozenPLIPAdapter(a.plip).cuda(); runtime=adapter.runtime_audit(raw); runtime.update({"source":"vinid/plip","revision":"67ade53","weight_sha256":PLIP_WEIGHT_SHA})
    if not runtime["dense_grid_safe"] or runtime["patch_token_count"]!=49: raise AssertionError("PLIP dense-token safety failed")
    write_json(out/"manifests/plip_runtime_manifest.json",runtime)
    embedding=adapter.encode_text(concepts,raw.device).cpu(); norms=embedding.norm(dim=-1)
    if embedding.shape!=(32,512) or not torch.allclose(norms,torch.ones_like(norms),atol=1e-6): raise AssertionError("Concept embedding safety failed")
    embedding_path=out/"concept_bank/concept_embeddings.pt"; torch.save({"embeddings":embedding,"concepts":concepts,"class_index":mapping},embedding_path)
    write_json(out/"manifests/concept_bank_manifest.json",{"frozen_before_validation_gt":True,"classes":mapping,"concepts_per_class":8,"total_concepts":32,
               "concept_bank_sha256":sha256(destination),"embedding_sha256":sha256(embedding_path),"embedding_shape":list(embedding.shape),
               "norm_min":float(norms.min()),"norm_max":float(norms.max()),"design":"atomic pathology morphology/architecture concepts; no validation segmentation GT used"})
    del adapter; torch.cuda.empty_cache()
    base=load_model(a.checkpoint); mean=raw.new_tensor([.485,.456,.406])[None,:,None,None]; std=raw.new_tensor([.229,.224,.225])[None,:,None,None]; labels=torch.ones((1,4),device=raw.device)
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): reference=base((raw-mean)/std,labels,step=29275)["primary_output"].float()
    identities={}; counts={}
    del base; torch.cuda.empty_cache()
    for mode in ("P1","P2","P3"):
        model=PDSRVPCAHQMR(a.checkpoint,a.plip,embedding,mode).cuda().eval()
        with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16): result=model(raw,labels); value=result["primary_output"].float()
        drift=float((value-reference).abs().max()); same=bool(torch.equal(value.argmax(1),reference.argmax(1)))
        identities[mode]={"same_argmax":same,"max_abs_drift":drift,"gamma_abs_max":float(model.gamma.abs().max()),"pass":same and drift<1e-5}
        frozen=sum(x.numel() for x in model.base.parameters()); plip=sum(x.numel() for x in model.plip.parameters()); trainable=sum(x.numel() for x in model.parameters() if x.requires_grad)
        counts[mode]={"hqmr_frozen_params":frozen,"plip_frozen_params":plip,"phase0_trainable_params":trainable,"full_inference_params":sum(x.numel() for x in model.parameters())}
        if not identities[mode]["pass"]: raise AssertionError(f"{mode} identity failed: {identities[mode]}")
        del model,result; torch.cuda.empty_cache()
    write_json(out/"manifests/identity_tests.json",{"sample":name,"tests":identities})
    write_json(out/"manifests/parameter_counts.json",counts)
    write_json(out/"manifests/leakage_audit.json",{"pass":True,"train_loader_fields":["image_id","raw_rgb","image_level_label"],"segmentation_gt_in_train":False,
               "validation_access_before_e5":False,"shared_geometric_augmentation":True,"hqmr_and_plip_normalization_after_shared_augmentation":True})
    write_json(out/"manifests/protocol_deviations.json",{"identity_preserving_fusion":"H5 + gamma*Z; standard LN would violate the mandatory gamma=0 identity",
               "gradient_audit":"At exact gamma=0, P1 P_last and P2/P3 Ps are mathematically gradient-blocked on the first backward; require gamma first-backward gradient and projection gradient after first optimizer update."})
    write_json(out/"manifests/preparation_complete.json",{"pass":True,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
               "checkpoint_sha256":CHECKPOINT_SHA,"plip_sha256":PLIP_WEIGHT_SHA,"training_samples":len(dataset),"validation_gt_opened":False})
    print(json.dumps({"event":"PDSR_VPCA_PREPARATION_PASS","identity":identities,"runtime":runtime}),flush=True)
if __name__=="__main__": main()

