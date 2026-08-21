"""Frozen constants for SSHR HMA-v0 mechanism autopsy."""

A0_COMMIT = "4e9a2887b220d17e27649d72a3d13f32b7ebe8f9"
EXPECTED_CHECKPOINT_SHA256 = (
    "509c264ec2df3b7dd6628b227d088db48300a2bc101ef3496d34ea6525911579"
)
EXPECTED_VAL_IMAGES = 3418
EXPECTED_VAL_SLIDES = 22
EXPECTED_TRAIN_IMAGES = 23422

SEED = 42
IMAGE_SIZE = 224
N_CLASS = 4
BACKGROUND = 4
PARITY_IMAGES = 32
GRADIENT_BATCHES = 32
GRADIENT_BATCH_SIZE = 20
BCSS_THRESHOLDS = (0.8, 0.9, 0.8, 0.6)
TTA_TRANSFORMS = (((), ()), ((3,), (2,)), ((2,), (1,)))
CAM_WEIGHTS = {"28_1": 0.6, "28_2": 0.2, "deep": 0.2}
LOSS_WEIGHTS = {"56": 0.10, "28_1": 0.15, "28_2": 0.25, "deep": 0.50}
STAGES = ("56", "28_1", "28_2")
VARIANTS = ("raw", "gsr", "ch", "full")
STAGE_CHANNELS = {"56": 256, "28_1": 512, "28_2": 1024}
CONTEXT_KERNEL = 15

FINAL_VARIANTS = {
    "official_full": ("full", "full"),
    "all_hfrm_off": ("raw", "raw"),
    "gsr_only": ("gsr", "gsr"),
    "ch_only": ("ch", "ch"),
    "hfrm_28_1_off": ("raw", "full"),
    "hfrm_28_2_off": ("full", "raw"),
    "gsr_28_1_off": ("ch", "full"),
    "ch_28_1_off": ("gsr", "full"),
    "gsr_28_2_off": ("full", "ch"),
    "ch_28_2_off": ("full", "gsr"),
}

