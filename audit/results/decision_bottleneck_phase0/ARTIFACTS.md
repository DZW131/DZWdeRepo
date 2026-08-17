# Decision Bottleneck Phase-0 artifacts

This directory contains the durable, reviewable outputs from the frozen BCSS
validation-only audit at audit commit
`f82eb0ef55377e1dc27a02f3e704667640a71c10`.

Included:

- exact protocol, environment, checkpoint identity, and parity records;
- the complete generated report;
- all 17 CSV evidence tables;
- five summary figures and 24 automatically ranked qualitative panels;
- CAM cache metadata, source-group assignments, and raw-value summaries;
- the complete run log.

The 12 GB numeric CAM/prediction cache and the 164 MB OOF prediction array are
intentionally retained on the 5090 server rather than committed to Git. They
are reproducible from the exact command in `config.json` and the checkpoint
whose SHA256 is recorded in `summary.json`.

The downloaded delivery bundle had SHA256
`54b596b1140d3462ba3147f6e8e51ac55730e5a9fb388128d990375b433bb94e`.
No BCSS test data was evaluated and no SSHR training was performed.
