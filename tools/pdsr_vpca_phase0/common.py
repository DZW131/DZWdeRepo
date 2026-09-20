from __future__ import annotations
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

CLASSES=("tumor","stroma","lymphocytic_infiltrate","necrosis")
CHECKPOINT_SHA="84dab82140eb79176bef3f518b6508b6167b328b6d55126d24efffa7467e4abb"
PLIP_WEIGHT_SHA="98a7f8d2a1f4a8fc8f6dedb3a16ff7efbe02a7ef67c93904c80bca9767c69630"
EPOCHS=5; STEPS_PER_EPOCH=1171; TOTAL_STEPS=5855; MICRO_BATCH=5; ACCUMULATION=4

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def write_json(path:Path,value): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,allow_nan=True),encoding="utf-8")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_concepts(path:Path)->tuple[list[str],dict]:
    payload=yaml.safe_load(path.read_text(encoding="utf-8")); mapping=payload["class_index"]
    if [mapping[i] for i in range(4)]!=list(CLASSES): raise AssertionError("BCSS class mapping changed")
    values=[]
    for name in CLASSES:
        rows=payload["concepts"][name]
        if len(rows)!=8: raise AssertionError(f"Expected 8 concepts for {name}")
        values.extend(rows)
    if len(set(values))!=32: raise AssertionError("Concept bank must contain 32 unique atomic concepts")
    return values,payload

def label_from_name(filename:str)->torch.Tensor:
    label=filename.split("]")[0].split("[")[-1]
    if len(label)<4 or any(x not in "01" for x in label[:4]): raise ValueError(filename)
    return torch.tensor([int(x) for x in label[:4]],dtype=torch.float32)

class CommonAugmentTrainDataset(Dataset):
    """One geometric augmentation, then model-side HQMR/PLIP normalization."""
    def __init__(self,root:Path):
        self.root=Path(root); self.files=sorted([*self.root.glob("*.png"),*self.root.glob("*.jpg")])
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        path=self.files[index]; image=Image.open(path).convert("RGB")
        if image.size!=(224,224): image=TF.resize(image,[224,224],InterpolationMode.BILINEAR)
        if random.random()>.5: image=TF.hflip(image)
        if random.random()>.5: image=TF.vflip(image)
        return path.stem,TF.to_tensor(image),label_from_name(path.name)

class CommonEvalDataset(Dataset):
    def __init__(self,root:Path): self.root=Path(root); self.files=sorted([*self.root.glob("*.png"),*self.root.glob("*.jpg")])
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        path=self.files[index]; image=Image.open(path).convert("RGB")
        if image.size!=(224,224): image=TF.resize(image,[224,224],InterpolationMode.BILINEAR)
        return path.stem,TF.to_tensor(image)

