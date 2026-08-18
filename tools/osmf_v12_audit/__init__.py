"""Frozen OSMF-v1.2 readiness and Phase-0 audit constants."""

SEED = 20260817
BATCH_SIZE = 20
IMAGE_SIZE = 224
FORMAL_EPOCHS_FOR_POLY_SCHEDULE = 25
READINESS_BATCHES = 8
PHASE0_BATCHES = 128
READINESS_AUDIT_STEPS = (0, 1, 2, 4, 8)
PHASE0_AUDIT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128)
PARAMETER_NAMES = (
    "p_sem.weight",
    "p_morph.weight",
    "u_sem.weight",
    "u_morph.weight",
)
OBJECTIVE_NAMES = ("sem_pres", "eq", "orth", "rec")
OBJECTIVE_WEIGHTS = {
    "sem_pres": 0.05,
    "eq": 0.05,
    "orth": 0.05,
    "rec": 0.10,
}
