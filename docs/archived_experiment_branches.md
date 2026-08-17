# Archived experiment boundary

This branch starts a new, validation-only Decision Bottleneck audit from the
frozen official baseline commit `4e9a2887b220d17e27649d72a3d13f32b7ebe8f9`.

Earlier HST, FA-MPR, SC-MPR, CDSR, and CLRR experiment branches and their
artifacts are treated as archived, read-only research records. They are not
parents or dependencies of this audit branch. This archival boundary does not
delete, rewrite, close, merge, or otherwise change any earlier branch, pull
request, checkpoint, log, or result.

The new audit is intentionally restricted to the frozen A0 model and BCSS
validation split. It performs no SSHR retraining, no test evaluation, and no
modification of the network, training protocol, released inference rule, or
official metric.
