"""Frozen OSMF-v1.3 readiness and Phase-0S audit contract."""

SEED = 20260817
BATCH_SIZE = 20
IMAGE_SIZE = 224
READINESS_BATCHES = 8
PHASE0S_BATCHES = 128
READINESS_AUDIT_STEPS = (0, 1, 2, 4, 8)
PHASE0S_AUDIT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128)
STRUCTURAL_STEPS = tuple(range(4, 129, 4))
FIXED_PROBE_STEPS = (0, 4, 8, 16, 32, 64, 96, 128)
PROBE_IMAGES = 64
PROBE_BATCH_SIZE = 8
PARAMETER_NAMES = (
    "p_sem.weight",
    "p_morph.weight",
    "u_sem.weight",
    "u_morph.weight",
)
MORPHOLOGY_PARAMETER_NAMES = ("p_morph.weight", "u_morph.weight")
OBJECTIVE_NAMES = ("sem_pres", "struct", "orth", "rec")
OBJECTIVE_WEIGHTS = {
    "sem_pres": 0.05,
    "struct": 0.05,
    "orth": 0.05,
    "rec": 0.10,
}
