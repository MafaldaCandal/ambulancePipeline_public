"""
paths.py

Centralised path definitions for the core methodology pipeline.

Scope:
    This file defines only the path contract used by the methodology scripts:
    Phase 1 through Phase 4, the active input folder, and the active run folder.

    Diagnostics and extended tests should define their own report-specific paths
    in their own modules. They may still import the core methodology artefact
    paths below when they need to read outputs from a completed run.

Design:
    - Input paths point to data/input_<dataset>/.
    - The active input is resolved first from PIPELINE_INPUT_NAME, then from
      configs.ACTIVE_INPUT_NAME.
    - Run-specific paths require ACTIVE_RUN_DIR, normally set by orchestrator.py.
    - If ACTIVE_RUN_DIR is not set, the latest runs/RunXXX directory is used.
    - This file creates run output directories, but does not create data folders.
"""

from __future__ import annotations

import os
from pathlib import Path

import configs


# =========================================================
# Project directories
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
METHODOLOGY_DIR = PROJECT_ROOT / "methodology"
UTILS_DIR = PROJECT_ROOT / "utils"

# The run directory root is safe to create. Input folders are not created here,
# because missing data should fail visibly rather than being replaced by an empty
# directory.
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Active input directory
# =========================================================

def _normalise_input_name(name: str | None) -> str:
    """
    Accept both 'BN_2024' and 'input_BN_2024' style names.
    """
    if name is None or not str(name).strip():
        return "synthetic"
    return str(name).strip().removeprefix("input_")


def _infer_active_input_name() -> str:
    """
    Resolve the active input folder.

    Preferred:
        PIPELINE_INPUT_NAME=BN_2024

    Fallback:
        configs.ACTIVE_INPUT_NAME or configs.ACTIVE_DATASET

    Legacy fallback:
        configs.ACTIVE_REGION + configs.ACTIVE_YEAR
    """
    env_input = os.environ.get("PIPELINE_INPUT_NAME")
    if env_input:
        return _normalise_input_name(env_input)

    explicit = getattr(configs, "ACTIVE_INPUT_NAME", None)
    if explicit:
        return _normalise_input_name(explicit)

    dataset = getattr(configs, "ACTIVE_DATASET", None)
    if dataset:
        return _normalise_input_name(dataset)

    region = getattr(configs, "ACTIVE_REGION", None)
    if region is None:
        raise ValueError(
            "Could not infer active input. Set PIPELINE_INPUT_NAME or define "
            "ACTIVE_INPUT_NAME / ACTIVE_DATASET / ACTIVE_REGION in configs.py."
        )

    region = str(region).strip()
    if region.lower() == "synthetic":
        return "synthetic"

    year = getattr(configs, "ACTIVE_YEAR", 2024)
    return f"{region}_{year}"


ACTIVE_INPUT_NAME = _infer_active_input_name()
INPUT_DIR = DATA_DIR / f"input_{ACTIVE_INPUT_NAME}"

if ACTIVE_INPUT_NAME == "synthetic":
    INPUT_DIR = DATA_DIR / "input_synthetic"

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Active input directory does not exist: {INPUT_DIR}\n"
        f"Resolved ACTIVE_INPUT_NAME={ACTIVE_INPUT_NAME!r}. "
        "Check PIPELINE_INPUT_NAME, configs.py, or the data/ folder."
    )


# =========================================================
# Prepared methodology inputs
# =========================================================

# Core prepared input tables used by Phase 1.
DISPATCH_REGISTERS_PARQUET = INPUT_DIR / "dispatch_registers.parquet"
GPS_LOGS_PARQUET = INPUT_DIR / "gps_logs.parquet"

# Optional chunked GPS input contract. Some Phase 1 implementations stream
# parsed GPS chunks instead of reading one large gps_logs.parquet file.
GPS_RAW_DIR = INPUT_DIR / "gps_logs_raw_parquet"
GPS_RAW_PATTERN = GPS_RAW_DIR / "*.parquet"


# =========================================================
# Static OSM / OSRM files
# =========================================================

def _osrm_file_from_pbf(pbf_path: Path) -> Path:
    if pbf_path.name.endswith(".osm.pbf"):
        return pbf_path.with_name(pbf_path.name.removesuffix(".osm.pbf") + ".osrm")
    return pbf_path.with_suffix(".osrm")


def _path_in_active_input(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return INPUT_DIR / path


def _resolve_pbf_for_active_input(active_region: str) -> Path:
    """
    Resolve the OSM PBF for the active input.

    First tries configs.OSM_PBF_FILES[active_region]. If that configured filename
    is not present, falls back to exactly one *.osm.pbf in the active input
    directory. This keeps regional and national extracts reproducible while
    avoiding fragile failures from small filename changes.
    """
    if not hasattr(configs, "OSM_PBF_FILES"):
        raise AttributeError(
            "configs.py must define OSM_PBF_FILES, mapping region names to OSM .pbf filenames."
        )

    pbf_files = sorted(INPUT_DIR.glob("*.osm.pbf"))

    configured_name = configs.OSM_PBF_FILES.get(active_region)
    if configured_name:
        configured_path = _path_in_active_input(configured_name)
        if configured_path.exists():
            return configured_path

    if len(pbf_files) == 1:
        return pbf_files[0]

    if configured_name:
        expected = _path_in_active_input(configured_name)
        raise FileNotFoundError(
            f"Missing configured OSM PBF file for active region {active_region!r}: {expected}\n"
            f"Also expected exactly one fallback *.osm.pbf in {INPUT_DIR}, found {len(pbf_files)}."
        )

    raise ValueError(
        f"configs.OSM_PBF_FILES has no entry for active region {active_region!r}, "
        f"and fallback found {len(pbf_files)} *.osm.pbf files in {INPUT_DIR}."
    )


ACTIVE_REGION = getattr(configs, "ACTIVE_REGION", None)
if ACTIVE_REGION is None:
    raise ValueError("configs.py must define ACTIVE_REGION.")

ACTIVE_REGION = str(ACTIVE_REGION).strip()

OSM_PBF = _resolve_pbf_for_active_input(ACTIVE_REGION)
OSRM_FILE = _osrm_file_from_pbf(OSM_PBF)

# Compatibility mapping for scripts that still import OSM_PBF_BY_REGION.
OSM_PBF_BY_REGION = {
    region: _path_in_active_input(filename)
    for region, filename in getattr(configs, "OSM_PBF_FILES", {}).items()
}

# Default OSRM car profile copied from osrm/osrm-backend.
ORIGINAL_LUA = UTILS_DIR / "car.lua"


# =========================================================
# Active run directory
# =========================================================

def _find_latest_run_dir() -> Path:
    """
    Return the latest existing runs/RunXXX directory.

    Used when ACTIVE_RUN_DIR is not set, for example when running a single
    methodology script directly during development.
    """
    existing_runs = [
        p for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("Run") and p.name[3:].isdigit()
    ]

    if not existing_runs:
        raise RuntimeError(
            "ACTIVE_RUN_DIR is not set and no previous runs/RunXXX directory exists. "
            "Run orchestrator.py first to create a run."
        )

    return max(existing_runs, key=lambda p: int(p.name[3:]))


ACTIVE_RUN_DIR = os.environ.get("ACTIVE_RUN_DIR")

RUN_DIR = (
    Path(ACTIVE_RUN_DIR).resolve()
    if ACTIVE_RUN_DIR
    else _find_latest_run_dir().resolve()
)


def _run_dir(name: str) -> Path:
    directory = RUN_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _run_file(directory: Path, filename: str) -> Path:
    return directory / filename


OUT_DIR = _run_dir("outputs")
RESULTS_DIR = _run_dir("results")
REPORT_DIR = _run_dir("report")


# =========================================================
# Phase 1 — realised travel-time reconstruction
# =========================================================

TRIP_GPS_SEQUENCES_PARQUET = _run_file(
    OUT_DIR,
    "trip_gps_sequences.parquet",
)

CLEAN_TRAJECTORIES_PARQUET = _run_file(
    OUT_DIR,
    "clean_trajectories.parquet",
)

TRIP_SUMMARY_PARQUET = _run_file(
    OUT_DIR,
    "trip_summary.parquet",
)

TRIP_REJECTION_SUMMARY_PARQUET = _run_file(
    OUT_DIR,
    "trip_rejection_summary.parquet",
)


# =========================================================
# Phase 2 — baseline OSRM prediction and route features
# =========================================================

ACCESS_LUA = _run_file(
    RUN_DIR,
    "ambulance_nl.lua",
)

ACCESS_DIFF = _run_file(
    RUN_DIR,
    "ambulance_profile.diff",
)

ROUTES_PARQUET = _run_file(
    OUT_DIR,
    "routes.parquet",
)

ROUTES_GEOJSON = _run_file(
    OUT_DIR,
    "routes.geojson",
)

ROUTE_REJECTION_SUMMARY_PARQUET = _run_file(
    OUT_DIR,
    "route_rejection_summary.parquet",
)

ROADS_GPKG = _run_file(
    OUT_DIR,
    "osm_roads.gpkg",
)

ROUTE_FEATURES_PARQUET = _run_file(
    OUT_DIR,
    "route_features.parquet",
)

ROUTE_FEATURES_CSV = _run_file(
    OUT_DIR,
    "route_features.csv",
)

PREDICTION_ERRORS_PARQUET = _run_file(
    OUT_DIR,
    "prediction_errors.parquet",
)

PREDICTION_ERRORS_CSV = _run_file(
    OUT_DIR,
    "prediction_errors.csv",
)


# =========================================================
# Phase 3 — residual modelling and profile calibration
# =========================================================

REGRESSION_TABLE_PARQUET = _run_file(
    OUT_DIR,
    "regression_table.parquet",
)

REGRESSION_TABLE_CSV = _run_file(
    OUT_DIR,
    "regression_table.csv",
)

REGRESSION_COEFFICIENTS_CSV = _run_file(
    RESULTS_DIR,
    "regression_coefficients.csv",
)

REGRESSION_SUMMARY_TXT = _run_file(
    RESULTS_DIR,
    "regression_summary.txt",
)

CALIBRATED_LUA = _run_file(
    RUN_DIR,
    "ambulance_nl_calibrated.lua",
)

CALIBRATED_DIFF = _run_file(
    RUN_DIR,
    "calibrated_profile.diff",
)

CALIBRATION_CHANGES_CSV = _run_file(
    RESULTS_DIR,
    "calibration_changes.csv",
)


# =========================================================
# Phase 4 — calibrated prediction and evaluation
# =========================================================

CALIBRATED_PREDICTIONS_PARQUET = _run_file(
    OUT_DIR,
    "calibrated_predictions.parquet",
)

CALIBRATION_EVALUATION_METRICS_CSV = _run_file(
    RESULTS_DIR,
    "calibration_evaluation_metrics.csv",
)

CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET = _run_file(
    RESULTS_DIR,
    "calibration_evaluation_trip_errors.parquet",
)

CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV = _run_file(
    RESULTS_DIR,
    "calibration_evaluation_route_diagnostics.csv",
)

RESULTS_REPORT_PDF = _run_file(
    REPORT_DIR,
    "results_report.pdf",
)

RESULTS_REPORT_MD = _run_file(
    REPORT_DIR,
    "results_report.md",
)

RESULTS_REPORT_TXT = _run_file(
    REPORT_DIR,
    "results_report.txt",
)
