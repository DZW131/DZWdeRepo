# GCQM Phase-0 reproducibility

GCQM is the zero-parameter global simplification of FOMD. It computes detached class-conditioned query weights from the spatial mean of frozen MOMD routing and uses `sum(w*B)` as the Stage2/3 primary mask.

Environment: the SSHR CUDA/PyTorch environment with BF16 on RTX 4090D. Data must be the 23,422-image BCSS training directory; validation, test and LUAD access are prohibited. Run the compatibility audit, full `pytest`, two-step smoke, and finally `train_gcqm_phase0.py` with the frozen 32-image cohort and official MXNet initialization. Metrics, endpoint hash, visualizations and the final report are emitted into the chosen output directory.
