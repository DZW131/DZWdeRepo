# FOMD Phase-0 reproducibility

FOMD keeps the archived MOMD-v1 network and loss exactly unchanged. It adds only detached monitor-time counterfactuals for query permutation, spatially global ownership, PCA-only mixing, and uniform region support.

Formal protocol: BCSS train split only, Seed42, official ResNet38 initialization, batch 20, BF16, RTX 4090D, 1,171 steps/epoch, at most 5 epochs. Run `python -m tools.audit_fomd_factorization`, the complete regression suite, a two-step smoke run, then `train_fomd_phase0.py`. Do not access validation/test data or tune after step 0.

The report is generated as `FOMD_Phase0_Factorized_Ownership_Mask_Decoding_Report.md`; all numerical evidence is retained in the adjacent JSON/CSV artifacts.
