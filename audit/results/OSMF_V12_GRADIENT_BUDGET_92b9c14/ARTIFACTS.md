# OSMF-v1.2 Artifact Manifest

Executed commit: `92b9c142a18a7c0d8bbc6406f3ff336b1ef7e7c4`

## Decisions

- Exact parity: `OSMF_V12_PARITY_PASS`
- Eight-batch readiness: `OSMF_V12_READINESS_PASS`
- Fresh 128-batch Phase 0: `OSMF_V12_PHASE0_REVIEW`
- Three-epoch pilot and all later experiments: not started

## Integrity

- `parity/summary.json`: `e2585306d7f43be4842ea39168ff2e5c7f2d64cf86e6332c1e11d8eccfef0dcf`
- `readiness_8b/summary.json`: `892d42e48eaffed3a7fbeaeb02e818430b504b2d49cc9650598cd7eb1ec70592`
- `phase0_128b/summary.json`: `095871dcfc31e967fc842c777b6c9d44edd3826ac2eb084e0123ebc587ab94b6`
- readiness gradient ratios: `10e107cc0fa6f039411795a313532261ed032878f5505cbbf4642efd6ab6fc70`
- Phase-0 gradient ratios: `37c914e68d92232f55a717cbc4b7a67f1a64670f6cf1e94f4f00ec08c2c0657b`
- readiness report: `f35c126715b000c348e976959c5000412bdd14b60d8828d3b5696256bf304703`
- Phase-0 report: `d25b40444521aeab1f29ebf01519b086dc33c66627e1a3454548b561d47ebd53`
- parity log: `e2585306d7f43be4842ea39168ff2e5c7f2d64cf86e6332c1e11d8eccfef0dcf`
- readiness log: `76e0007d30b007b3b6d7a464e6da086eac300fdac95fe4179e1725f966c19a9f`
- Phase-0 log: `7c7d61c8914bf7a34150564eb94b72b337a742d0f402f1ae2f60bc3db73e4fb2`

## Entry points

- [Parity summary](parity/summary.json)
- [Parity report](parity/osmf_v12_parity_report.md)
- [Parity console log](parity.log)
- [Readiness summary](readiness_8b/summary.json)
- [Readiness report](readiness_8b/docs/osmf_v12_semantic_readiness_report.md)
- [Readiness gradient ratios](readiness_8b/tables/gradient_ratio.csv)
- [Readiness console log](readiness_8b.log)
- [Phase-0 summary](phase0_128b/summary.json)
- [Phase-0 report](phase0_128b/docs/osmf_v12_phase0_report.md)
- [Phase-0 gradient ratios](phase0_128b/tables/gradient_ratio.csv)
- [Phase-0 representation health](phase0_128b/tables/representation_health.csv)
- [Phase-0 parameter health](phase0_128b/tables/parameter_health_summary.csv)
- [Phase-0 console log](phase0_128b.log)

All result files are raw downloads from the immutable server execution. The
validation split was used only for exact parity. No test data, LUAD data,
segmentation-ground-truth training signal, checkpoint selection, or
hyperparameter tuning was used.
