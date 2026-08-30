# Superseded diagnostic evaluation — not the final experiment report

This first full evaluation used the correct native-dtype official TTA, but
reconstructed historical Phase-0 risk/CH populations under current evaluation
backend settings. The independent artifact validator caught small count drift.

It is retained without overwrite for provenance. Do not use its mechanism
tables or decision as the final delivery. The verified report uses an immutable
cache replayed from original Phase-0 commit `586f402`, with all four CH-group
counts and Top20 counts matching the original CSV for **every validation image**.

Use `../rddr_phase2a/` and the report in `docs/` for final results. No model,
training checkpoint, inference threshold, fusion weight, or decision gate was
changed in this correction.
