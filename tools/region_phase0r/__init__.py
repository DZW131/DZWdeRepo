"""Frozen constants for the SSHR region-centric Phase-0R audit."""

SEED = 20260817
IMAGE_SIZE = 224
N_FOREGROUND = 4
N_LABELS = 5
BACKGROUND = 4
PURITY_THRESHOLD = 0.80
PRIMARY_MIN_AREA = 8
SENSITIVITY_AREAS = (1, 32)
N_SPLITS = 5
MAX_CLUSTER_SAMPLE = 20000

REPRESENTATIONS = (
    "centroid",
    "bbox",
    "region",
    "geometry",
    "region_geometry",
)

GEOMETRY_COLUMNS = (
    "log_area",
    "area_fraction",
    "bbox_fill_ratio",
    "aspect_ratio",
    "compactness",
    "boundary_density",
    "component_rank",
    "same_class_component_count",
)
