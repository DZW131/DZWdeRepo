# OSMF Phase-0 Artifact Manifest

Decision: **`OSMF_PHASE0_NOGO`**  
Executed commit: `39fdf788aed6d0e31bd42108d87fc502a37d591a`  
Processed real BCSS batches: `2/128` (preregistered hard stop)  
Hard-stop reason: `PERSISTENT_SEM_GRADIENT_RATIO_GT_0_50`

Key entry points:

- [`summary.json`](summary.json): machine-readable decision and protocol state
- [`config/frozen_contract.json`](config/frozen_contract.json): command, hashes, optimizer groups, and scope guards
- [`docs/osmf_phase0_128batch_audit.md`](docs/osmf_phase0_128batch_audit.md): raw auto-generated report
- [`tables/main_summary.csv`](tables/main_summary.csv): main mechanism table
- [`tables/gradient_ratio.csv`](tables/gradient_ratio.csv): decisive gradient ratios
- [`tables/parameter_health_summary.csv`](tables/parameter_health_summary.csv): six-path gradient/update health
- [`run.log`](run.log): complete console log

Important SHA256 values:

- `summary.json`: `beae75a5b462c0cba08050b9f513daa230d290f35fde8be4c1a32ecfbfae0faa`
- `tables/gradient_ratio.csv`: `4e89afd18fb6781e2b33b16cd0418898b4295810f6b94cbe2c1f4397f5131fea`
- raw generated report: `ac8d4c2ff74e87c17854d613abc07519c5782210417c89357bb7eb8f97f2f61`

The raw report's sentence listing steps 1/2/4/8/16/32/64/96/128 describes the
planned schedule. Only steps 1 and 2 were actually measured because the hard
stop fired; the CSV row counts and `summary.json` are authoritative. Likewise,
the raw runtime overhead field is not estimable before the first scheduled
equivariance step. These presentation issues are corrected in the delivery
report and report generator without changing experimental data.

