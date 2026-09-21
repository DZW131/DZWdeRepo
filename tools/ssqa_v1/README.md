# SSQA-v1 execution and reproduction

This is a no-training, frozen-source audit. It never changes HQMR, CCRA, PLIP or a foundation-model parameter. Use only official model weights; keep source caches and reports outside Git.

## Environment and data

Tested on Python 3.10, PyTorch 2.11 CUDA 12.8, `open_clip_torch==2.23.0`, `transformers==4.35.2`, `tokenizers==0.15.2`, plus the repository's NumPy, pandas, PyArrow, SciPy, Pillow, Matplotlib and PyYAML. A separate `venv --system-site-packages` was used so the historical HQMR environment was not modified.

Inputs: the frozen UMRF `component_evidence.parquet`, `gt_free_prediction_maps.uint8.npy`, `image_ids.npy`, `01_evidence_freeze_manifest.json`, and (only after extraction) `component_evidence_with_gt.parquet`. BCSS validation RGB images are under `val/img`. `val/mask` is not used to define any crop. PLIP is the same frozen `vinid/plip`; official QuiltNet-B-16-PMB and BiomedCLIP weights and their shared official PubMedBERT text files are kept in a separate model directory.

The official file SHA256 values and model revisions are frozen in `manifests/source_availability_manifest.json` under the output directory. Never substitute mirrors or unrecorded variants.

## Commands

From the repository root, with the SSQA Python environment active and paths adjusted:

```bash
python -m tools.ssqa_v1.probe --name PLIP --model-dir /path/to/vinid-plip
python -m tools.ssqa_v1.probe --name QuiltNet --model-dir /path/to/SSQA_v1/QuiltNet-B-16-PMB --text-dir /path/to/SSQA_v1/PubMedBERT
python -m tools.ssqa_v1.probe --name BiomedCLIP --model-dir /path/to/SSQA_v1/BiomedCLIP --text-dir /path/to/SSQA_v1/PubMedBERT
python -m tools.ssqa_v1.prepare --umrf /path/to/frozen-UMRF --val-root /path/to/BCSS-WSSS/val --model-root /path/to/models/SSQA_v1 --output /path/to/SSQA_v1_BCSS_Seed42
python -m tools.ssqa_v1.extract --name PLIP --model-root /path/to/models/SSQA_v1 --output /path/to/SSQA_v1_BCSS_Seed42
python -m tools.ssqa_v1.extract --name QuiltNet --model-root /path/to/models/SSQA_v1 --output /path/to/SSQA_v1_BCSS_Seed42
python -m tools.ssqa_v1.extract --name BiomedCLIP --model-root /path/to/models/SSQA_v1 --output /path/to/SSQA_v1_BCSS_Seed42
python -m tools.ssqa_v1.evaluate --umrf /path/to/frozen-UMRF --output /path/to/SSQA_v1_BCSS_Seed42
python -m tools.ssqa_v1.report --output /path/to/SSQA_v1_BCSS_Seed42
python -m tools.ssqa_v1.validate --output /path/to/SSQA_v1_BCSS_Seed42
```

Do not call `evaluate` before *all* READY source-cache manifests exist and pass their SHA checks. `prepare` freezes the 60/40 TCGA-patient split on GT-free predicted-class distribution because true class and Hard-M1 labels cannot be used before source embeddings freeze. The later `gt_unseal_manifest.json` records true-class balance. The source probes require finite, compatible 512-dimensional, normalized and deterministic image/text embeddings. The full extraction repeats a fixed 100-component subset twice and requires score drift <1e-5.

## Outputs and interpretation

The primary metric is Hard-M1 area-weighted true-vs-frozen-L5-rival positive rate on BBOX15. MASKED is secondary. The machine-readable decision is `decision.json`; the full human report is `SSQA_v1_External_Semantic_Source_Qualification_Final_Report.md`. `dev/`, `holdout/`, `bootstrap/`, `independence/` and `visualizations/` retain the evidence. Pair/multi-source oracles use GT and are **not deployable model results**. Only a source satisfying both the pre-registered DEV and HOLDOUT gates can motivate a *separate* integration experiment.
