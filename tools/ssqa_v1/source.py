"""Source-native zero-shot image/text encoders; no trainable adapters."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


class SemanticSource:
    def __init__(self, name: str, model_dir: Path, text_dir: Path | None = None):
        self.name = name
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "PLIP":
            from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast

            self.model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True).to(self.device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
            self.tokenizer = CLIPTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
            self.text_context = None
        elif name in ("QuiltNet", "BiomedCLIP"):
            if text_dir is None:
                raise ValueError("OpenCLIP PubMedBERT directory is required")
            import open_clip
            from open_clip.factory import _MODEL_CONFIGS

            cfg = json.loads((self.model_dir / "open_clip_config.json").read_text(encoding="utf-8"))
            model_cfg = cfg["model_cfg"]
            model_cfg["text_cfg"]["hf_model_name"] = str(text_dir)
            model_cfg["text_cfg"]["hf_tokenizer_name"] = str(text_dir)
            internal_name = f"ssqa_{name.lower()}_local"
            _MODEL_CONFIGS[internal_name] = model_cfg
            self.model, _, self.processor = open_clip.create_model_and_transforms(
                internal_name,
                pretrained=str(self.model_dir / "open_clip_pytorch_model.bin"),
                **{f"image_{key}": value for key, value in cfg["preprocess_cfg"].items()},
            )
            self.model = self.model.to(self.device).eval()
            self.tokenizer = open_clip.get_tokenizer(internal_name)
            self.text_context = int(model_cfg["text_cfg"]["context_length"])
        else:
            raise ValueError(f"Unsupported source {name}")
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        if self.name == "PLIP":
            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            value = self.model.get_text_features(**{key: item.to(self.device) for key, item in tokens.items()})
        else:
            tokens = self.tokenizer(texts, context_length=self.text_context).to(self.device)
            value = self.model.encode_text(tokens)
        return F.normalize(value.float(), dim=-1).cpu()

    @torch.inference_mode()
    def encode_image(self, images: list[Image.Image]) -> torch.Tensor:
        if self.name == "PLIP":
            # Tiny frozen components can yield 2x2 crops; explicit channels-last
            # prevents Transformers from mistaking spatial axes for channels.
            value = self.processor(images=images, return_tensors="pt", input_data_format="channels_last")["pixel_values"].to(self.device)
            embedding = self.model.get_image_features(pixel_values=value)
        else:
            value = torch.stack([self.processor(image) for image in images]).to(self.device)
            embedding = self.model.encode_image(value)
        return F.normalize(embedding.float(), dim=-1).cpu()
