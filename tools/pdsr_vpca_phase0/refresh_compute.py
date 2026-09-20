"""Refresh BF16 profiler costs without repeating the sealed E5 predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from network.pdsr_vpca_hqmr import PDSRVPCAHQMR
from tools.pdsr_vpca_phase0.common import CommonEvalDataset, write_json
from tools.pdsr_vpca_phase0.evaluate import VARIANT_DIR, counted_flops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plip", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = args.output / "metrics"
    cost = json.loads((metrics / "computational_cost.json").read_text())
    concepts = torch.load(args.output / "concept_bank/concept_embeddings.pt", map_location="cpu", weights_only=False)["embeddings"]
    raw = CommonEvalDataset(args.val_root / "img")[0][1][None].cuda()
    labels = torch.ones((1, 4), device=raw.device)
    for variant, directory in VARIANT_DIR.items():
        model = PDSRVPCAHQMR(args.checkpoint, args.plip, concepts, variant).cuda().eval()
        state = torch.load(args.output / directory / f"{variant.lower()}_e5_adapter.pth", map_location="cpu", weights_only=False)
        model.load_trainable_state_dict(state)

        def forward():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model(raw, labels)

        flops = counted_flops(forward)
        cost["profiles"][variant].update({
            "flops_per_view": flops,
            "gflops_per_view": flops / 1e9,
            "flops_note": "BF16 torch.profiler counted FLOPs; unsupported operators are not imputed",
        })
        print(f"{variant} BF16 profiler GFLOPs/view={flops / 1e9:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(metrics / "computational_cost.json", cost)
    pd.DataFrame([{"variant": variant, **values} for variant, values in cost["profiles"].items()]).to_csv(metrics / "computational_cost.csv", index=False)


if __name__ == "__main__":
    main()
