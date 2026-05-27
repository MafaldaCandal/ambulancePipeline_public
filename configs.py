"""
configs.py

Project-wide configuration for the ambulance routing calibration pipeline.

This file stores methodological constants and non-path runtime settings. Path
resolution belongs in paths.py; dataset selection is supplied by orchestrator.py
through environment variables.
"""

from __future__ import annotations

import os


# =========================================================
# Small environment helpers
# =========================================================

def _normalise_input_name(value: str | None) -> str:
    if value is None or not str(value).strip():
        return "synthetic"
    return str(value).strip().removeprefix("input_")


def _infer_region_from_input(input_name: str) -> str:
    name = _normalise_input_name(input_name)
    if name.lower() == "synthetic":
        return "synthetic"
    return name.split("_", 1)[0]


def _infer_year_from_input(input_name: str, default: int = 2024) -> int:
    name = _normalise_input_name(input_name)
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return default


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return None


def _env_int(*names: str, default: int) -> int:
    value = _env_first(*names)
    if value is None:
        return default
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"Expected integer environment value for {names}, got {value!r}.") from exc


def _env_float(*names: str, default: float) -> float:
    value = _env_first(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected numeric environment value for {names}, got {value!r}.") from exc


def _env_bool(*names: str, default: bool) -> bool:
    value = _env_first(*names)
    if value is None:
        return default

    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Expected boolean environment value for {names}, got {value!r}.")


# =========================================================
# Runtime context
# =========================================================

# Usually set by orchestrator.py. Defaults to synthetic for public one-command
# reproducibility. These values are read by paths.py and methodology scripts.
ACTIVE_INPUT_NAME = _normalise_input_name(
    _env_first(
        "PIPELINE_INPUT_NAME",
        "PIPELINE_MAIN_INPUT",
        "PIPELINE_DATASET",
        "PIPELINE_MAIN_DATASET",
    )
)
ACTIVE_DATASET = ACTIVE_INPUT_NAME

ACTIVE_REGION = (
    _env_first("PIPELINE_ACTIVE_REGION", "PIPELINE_MAIN_REGION", "PIPELINE_REGION")
    or _infer_region_from_input(ACTIVE_INPUT_NAME)
).strip()

ACTIVE_YEAR = _env_int(
    "PIPELINE_ACTIVE_YEAR",
    default=_infer_year_from_input(ACTIVE_INPUT_NAME, default=2024),
)

# Backwards-compatible defaults only. The public orchestrator uses CLI flags for
# optional stages; leave these false unless running older scripts manually.
RUN_DIAGNOSTICS = _env_bool("PIPELINE_RUN_DIAGNOSTICS", default=False)
RUN_EXTENDED_TESTS = _env_bool("PIPELINE_RUN_EXTENDED_TESTS", default=False)


# =========================================================
# Region / OSM extract settings
# =========================================================

# Filenames are resolved relative to the active data/input_<dataset>/ folder.
# If the configured filename is absent and exactly one *.osm.pbf exists in the
# input folder, paths.py and orchestrator.py fall back to that single file.
OSM_PBF_FILES = {
    "synthetic": "synthetic.osm.pbf",
    "BN": "noord-brabant-260519.osm.pbf",
    "BZO": "noord-brabant-260519.osm.pbf",
    "BMW": "noord-brabant-260519.osm.pbf",
    # Legacy support only. This region is not used in the current analysis.
    "ZHZ": "zuid-holland-260512.osm.pbf",
}

# Legacy fallback for older extended-test scripts. Current extended-test dataset
# selection is inferred from data/input_<REGION>_<YEAR>/ and environment vars.
TRANSFER_REGIONS = ["BN", "BZO", "BMW"]


# =========================================================
# OSRM / Docker settings
# =========================================================

OSRM_URL = "http://127.0.0.1:5000"
OSRM_PROFILE = "driving"

OSRM_DOCKER_IMAGE = "osrm/osrm-backend@sha256:af5d4a83fb90086a43b1ae2ca22872e6768766ad5fcbb07a29ff90ec644ee409"
OSRM_CONTAINER_NAME = f"osrm_ambulance_{ACTIVE_REGION.lower()}"

GDAL_DOCKER_IMAGE = "ghcr.io/osgeo/gdal@sha256:d15d2ef116fde5bf32dbf094cfb007f8c8af1e283d687cd72ae88ad5bd786e66"

OSRM_TEST_ROUTES = {
    "synthetic": ((5.30, 51.60), (5.31, 51.61)),
    "BN": ((5.30, 51.60), (5.40, 51.65)),
    "BZO": ((5.45, 51.40), (5.50, 51.45)),
    "BMW": ((5.00, 51.55), (5.10, 51.60)),
    "NL": ((5.30, 51.60), (5.40, 51.65)),
    # Legacy support only.
    "ZHZ": ((4.65, 51.80), (4.75, 51.85)),
}


# =========================================================
# Coordinate systems / spatial processing
# =========================================================

EARTH_RADIUS_M = 6_371_000
PROJECTED_CRS = "EPSG:28992"

DENSIFY_INTERVAL_M = 20
MAX_ROAD_MATCH_DISTANCE_M = 25


# =========================================================
# Trip construction / GPS filtering
# =========================================================

MAX_TRIP_WINDOW_MINUTES = 45

# GPS data-quality thresholds. Some are diagnostic-only depending on the phase.
MAX_TRIP_GAP_SEC = _env_int("PIPELINE_MAX_TRIP_GAP_SEC", "MAX_TRIP_GAP_SEC", default=30)
MIN_OBSERVATIONS = _env_int("PIPELINE_MIN_OBSERVATIONS", "MIN_OBSERVATIONS", default=10)

# Movement / trip detection.
CONSISTENT_MOVEMENT_SECONDS = _env_int(
    "PIPELINE_CONSISTENT_MOVEMENT_SECONDS",
    "CONSISTENT_MOVEMENT_SECONDS",
    default=15,
)
MAX_START_GAP_SEC = _env_int("PIPELINE_MAX_START_GAP_SEC", "MAX_START_GAP_SEC", default=15)

# Behavioural thresholds.
MOVING_SPEED_THRESHOLD_KMH = _env_float(
    "PIPELINE_MOVING_SPEED_THRESHOLD_KMH",
    "MOVING_SPEED_THRESHOLD_KMH",
    default=2.0,
)
MOVING_AT_DISPATCH_WINDOW_SEC = _env_int(
    "PIPELINE_MOVING_AT_DISPATCH_WINDOW_SEC",
    "MOVING_AT_DISPATCH_WINDOW_SEC",
    default=15,
)
ARRIVAL_RADIUS_M = _env_int(
    "PIPELINE_ARRIVAL_RADIUS_M",
    "ARRIVAL_RADIUS_M",
    default=150,
)


# =========================================================
# Validation / evaluation settings
# =========================================================

RESPONSE_STANDARD_SEC = 15 * 60
LARGE_ERROR_SEC = 5 * 60

TRAINING_YEAR = 2024
VALIDATION_YEAR = 2025


# =========================================================
# OSRM Match / map-matching settings
# =========================================================

MATCH_CONFIDENCE_THRESHOLD = _env_float(
    "PIPELINE_MAP_MATCH_CONFIDENCE_THRESHOLD",
    "MAP_MATCH_CONFIDENCE_THRESHOLD",
    "OSRM_MATCH_CONFIDENCE_THRESHOLD",
    default=0.80,
)

MATCH_SAMPLE_INTERVAL_SEC = _env_int(
    "PIPELINE_MATCH_SAMPLE_INTERVAL_SEC",
    "MATCH_SAMPLE_INTERVAL_SEC",
    default=5,
)
MATCH_MAX_POINTS = _env_int("PIPELINE_MATCH_MAX_POINTS", "MATCH_MAX_POINTS", default=100)
MATCH_REQUEST_TIMEOUT_SEC = _env_int(
    "PIPELINE_MATCH_REQUEST_TIMEOUT_SEC",
    "MATCH_REQUEST_TIMEOUT_SEC",
    default=120,
)
MATCH_MIN_POINTS = _env_int("PIPELINE_MATCH_MIN_POINTS", "MATCH_MIN_POINTS", default=3)


# =========================================================
# Regression / calibration settings
# =========================================================

# Main thesis calibration applies only mapped coefficients that meet the
# configured significance threshold. This keeps profile changes tied to
# statistically supported residual patterns.
APPLY_ONLY_SIGNIFICANT_COEFS = _env_bool(
    "PIPELINE_APPLY_ONLY_SIGNIFICANT_COEFS",
    "APPLY_ONLY_SIGNIFICANT_COEFS",
    default=True,
)
P_VALUE_THRESHOLD = _env_float("PIPELINE_P_VALUE_THRESHOLD", "P_VALUE_THRESHOLD", default=0.10)

# If True, p3s8_run_regression.py excludes trips whose baseline prediction error
# lies outside [Q1 - IQR_OUTLIER_MULTIPLIER*IQR, Q3 + IQR_OUTLIER_MULTIPLIER*IQR].
EXCLUDE_IQR3_PREDICTION_ERROR_OUTLIERS = _env_bool(
    "PIPELINE_EXCLUDE_IQR3_PREDICTION_ERROR_OUTLIERS",
    "EXCLUDE_IQR3_PREDICTION_ERROR_OUTLIERS",
    default=False,
)
IQR_OUTLIER_MULTIPLIER = _env_float(
    "PIPELINE_IQR_OUTLIER_MULTIPLIER",
    "IQR_OUTLIER_MULTIPLIER",
    default=3.0,
)

MIN_CALIBRATED_SPEED_KMH = _env_float(
    "PIPELINE_MIN_CALIBRATED_SPEED_KMH",
    "MIN_CALIBRATED_SPEED_KMH",
    default=5.0,
)
MAX_CALIBRATED_SPEED_KMH = _env_float(
    "PIPELINE_MAX_CALIBRATED_SPEED_KMH",
    "MAX_CALIBRATED_SPEED_KMH",
    default=160.0,
)


# =========================================================
# OSM road classes
# =========================================================

ROAD_CLASSES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "residential",
    "unclassified",
    "service",
    "living_street",
]

ROAD_COEF_MAP = {
    "km_motorway": "motorway",
    "km_trunk": "trunk",
    "km_primary": "primary",
    "km_secondary": "secondary",
    "km_tertiary": "tertiary",
    "km_residential": "residential",
    "km_unclassified": "unclassified",
    "km_service": "service",
    "km_living_street": "living_street",
}


# =========================================================
# Baseline OSRM speeds
# =========================================================

BASE_ROAD_SPEED_KMH = {
    "motorway": 90,
    "trunk": 85,
    "primary": 65,
    "secondary": 55,
    "tertiary": 45,
    "residential": 30,
    "unclassified": 35,
    "service": 20,
    "living_street": 10,
}


# =========================================================
# Maximum plausible realised speeds
# =========================================================

MAX_SPEED_KMH_BY_ROAD = {
    "motorway": 191,
    "trunk": 189,
    "primary": 176,
    "secondary": 162,
    "tertiary": 138,
    "residential": 96,
    "unclassified": 90,
    "service": 90,
    "living_street": 68,
}