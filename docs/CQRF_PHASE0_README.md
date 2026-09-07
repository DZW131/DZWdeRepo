# CQRF Phase-0 implementation boundary

This branch implements only the preregistered BCSS Seed42 five-epoch CCRA responsibility-allocation gate.

- Stage 1 is the frozen cross-first global query decoder.
- Stages 2 and 3 use class-conditioned responsibility competition with softmax over queries.
- The class prior is detached PCA probability times detached predicted deep gate; labels never route responsibility.
- `Kp` is the already-audited parameter-free normalized 2-D sine/cosine encoding because the plan fixes `X + Kp` but introduces no additional learned position mechanism.
- CNN memories are detached immediately before their trainable projections, so the new projection/CCRA path remains trainable.
- F4 receives one coherent 128-channel CHPF transformation. Its output feeds separate pixel and Stage-3 memory projections.
- No DFPQ, previous-mask attention mask, Stage2/3 self-attention, diversity loss, balancing, hard assignment, or rescue tuning is present.
- Validation/test images and segmentation ground truth are prohibited. The gate uses only a frozen 32-image train cohort.

Run tests with `pytest -q tests/test_cqrf_phase0.py tests/test_cqrf_protocol.py` and train with `train_cqrf_phase0.py`.
