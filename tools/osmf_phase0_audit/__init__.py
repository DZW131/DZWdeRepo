"""Frozen constants for the OSMF-v1.0 Phase-0 audit."""

AUDIT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128)
GRADIENT_STEPS = AUDIT_STEPS[1:]
NUM_REAL_BATCHES = 128
SEED = 20260817
BATCH_SIZE = 20
IMAGE_SIZE = 224
FORMAL_EPOCHS_FOR_POLY_SCHEDULE = 25

PARAMETER_NAMES = (
    "p_sem.weight",
    "p_morph.weight",
    "u_sem.weight",
    "u_morph.weight",
    "semantic_classifier.weight",
    "semantic_classifier.bias",
)

OBJECTIVE_WEIGHTS = {
    "sem": 0.20,
    "eq": 0.20,
    "orth": 0.05,
    "rec": 0.10,
}
