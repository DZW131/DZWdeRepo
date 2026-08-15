# Upstream provenance

This repository uses the official SSHR implementation as its research baseline.

- Upstream repository: <https://github.com/trongduc-nguyen/SSHR>
- Imported revision: `8a0d7e271070ad4588d0fa6cfcd904505bee1189`
- Import date: 2026-08-15
- Layout: the upstream tracked-file snapshot is placed at the repository root.

Before adding this provenance file and the local repository-management rules, all
37 files tracked by the upstream revision were verified byte-for-byte against the
corresponding upstream Git blobs.

## Repository policy

- `main` is the reviewable baseline branch.
- Research changes are developed on dedicated branches and merged through pull
  requests after review.
- Datasets, pretrained weights, checkpoints, logs, and other generated experiment
  artifacts are not committed to Git.
- The official training and evaluation commands are retained in `README.md`.

## Legacy work

Earlier reproduction experiments and the A1/A2/A3 architecture-development work
remain in the legacy repository:
<https://github.com/DZW131/mytestrepo>.

They are intentionally not migrated into this repository. New experiments and
research innovations should start from the baseline recorded here.
