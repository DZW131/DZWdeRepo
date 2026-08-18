"""Frozen constants for the OSMF-v1.2 Phase-0M causal audit."""

SEED = 20260817
BATCH_SIZE = 20
IMAGE_SIZE = 224
AUTHORIZED_BATCHES = 128
PROBE_IMAGES = 64
PROBE_BATCH_SIZE = 8
EQUIVARIANCE_INTERVAL = 4
EQ_STEPS = tuple(range(4, AUTHORIZED_BATCHES + 1, EQUIVARIANCE_INTERVAL))
FIXED_PROBE_STEPS = (0, 4, 8, 16, 32, 64, 96, 128)
GRADIENT_STEPS = (4, 8, 16, 32, 64, 96, 128)
REPLICATION_AUDIT_STEPS = (1, 2, 4, 8, 16, 32, 64, 96, 128)
MORPHOLOGY_PARAMETER_NAMES = ("p_morph.weight", "u_morph.weight")
OBJECTIVE_WEIGHTS = {
    "sem_pres": 0.05,
    "eq": 0.05,
    "orth": 0.05,
    "rec": 0.10,
}
REPLICATION_REFERENCE = {
    "mean_r_sem": 0.16289584559225911,
    "mean_r_eq": 0.1076911730084808,
    "semantic_agreement_end": 0.9869547486305237,
    "reconstruction_cosine_end": 0.9980936050415039,
    "semantic_morphology_rms_ratio_end": 1.3556337555467697,
    "cross_covariance_end": 0.012997581623494625,
}
