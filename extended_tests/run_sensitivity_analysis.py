"""
run_sensitivity_analysis.py

Run selected sensitivity analyses for the active main calibration run.

This script separates two types of sensitivity checks:

1. Full reruns
   These rerun the full calibration pipeline because the parameter can change
   realised travel-time construction or route reconstruction:
       - arrival radius: 100 m, 150 m, 200 m
       - map-matching confidence threshold: 0.70, 0.80, 0.90

2. Diagnostic-only checks
   These are recorded as diagnostic settings only. They are not treated as full
   calibration reruns unless the wider pipeline explicitly implements them:
       - GPS continuity threshold
       - minimum observations

Outputs:
    runs/RunXXX/extended_tests/sensitivity_analysis/
        status.csv
        sensitivity_summary.csv
        preprocessing_diagnostics.csv
        arrival_radius_100m/
        arrival_radius_200m/
        map_match_confidence_0p70/
        map_match_confidence_0p90/

Baseline settings are summarised from the active main run:
    - arrival_radius_150m_baseline
    - map_match_confidence_0p80_baseline

Required configuration:
    The active main run must be available through ACTIVE_RUN_DIR, and the main
    input must be inferable from one of:
        PIPELINE_INPUT_NAME
        PIPELINE_MAIN_INPUT
        PIPELINE_DATASET
        PIPELINE_MAIN_DATASET

    If not, set for example:
        PIPELINE_INPUT_NAME=BN_2024
        PIPELINE_MAIN_REGION=BN

Important:
    This script passes parameter overrides through environment variables. The
    core methodology scripts/configs.py must read these variables for the reruns
    to actually change behaviour.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import polars as pl

from extended_test_utils import (
    DATA_DIR,
    EXTENDED_ROOT,
    FULL_CALIBRATION_STEPS,
    MAIN_RUN_DIR,
    STANDARD_SUMMARY_COLUMNS,
    StatusRow,
    build_env,
    ensure_run_subdirs,
    input_dir_for,
    log,
    print_header,
    run_step,
    summarise_evaluation_run,
    validate_extended_input,
    write_manifest,
    write_status_csv,
    write_summary,
)


TEST_TYPE = "sensitivity_analysis"
ROOT = EXTENDED_ROOT / TEST_TYPE

SUMMARY_CSV = ROOT / "sensitivity_summary.csv"
STATUS_CSV = ROOT / "status.csv"
PREPROCESSING_DIAGNOSTICS_CSV = ROOT / "preprocessing_diagnostics.csv"


# =========================================================
# Sensitivity specification
# =========================================================

@dataclass(frozen=True)
class MainRunMetadata:
    input_name: str
    region: str


@dataclass(frozen=True)
class SensitivityVariant:
    dimension: str
    value: str
    label: str
    run_dir_name: str
    env_overrides: dict[str, str]
    is_baseline: bool = False


FULL_RERUN_VARIANTS = [
    SensitivityVariant(
        dimension="arrival_radius",
        value="100",
        label="arrival_radius_100m",
        run_dir_name="arrival_radius_100m",
        env_overrides={
            "PIPELINE_ARRIVAL_RADIUS_M": "100",
            "ARRIVAL_RADIUS_M": "100",
        },
    ),
    SensitivityVariant(
        dimension="arrival_radius",
        value="150",
        label="arrival_radius_150m_baseline",
        run_dir_name="arrival_radius_150m_baseline",
        env_overrides={
            "PIPELINE_ARRIVAL_RADIUS_M": "150",
            "ARRIVAL_RADIUS_M": "150",
        },
        is_baseline=True,
    ),
    SensitivityVariant(
        dimension="arrival_radius",
        value="200",
        label="arrival_radius_200m",
        run_dir_name="arrival_radius_200m",
        env_overrides={
            "PIPELINE_ARRIVAL_RADIUS_M": "200",
            "ARRIVAL_RADIUS_M": "200",
        },
    ),
    SensitivityVariant(
        dimension="map_match_confidence",
        value="0.70",
        label="map_match_confidence_0p70",
        run_dir_name="map_match_confidence_0p70",
        env_overrides={
            "PIPELINE_MAP_MATCH_CONFIDENCE_THRESHOLD": "0.70",
            "MAP_MATCH_CONFIDENCE_THRESHOLD": "0.70",
            "OSRM_MATCH_CONFIDENCE_THRESHOLD": "0.70",
        },
    ),
    SensitivityVariant(
        dimension="map_match_confidence",
        value="0.80",
        label="map_match_confidence_0p80_baseline",
        run_dir_name="map_match_confidence_0p80_baseline",
        env_overrides={
            "PIPELINE_MAP_MATCH_CONFIDENCE_THRESHOLD": "0.80",
            "MAP_MATCH_CONFIDENCE_THRESHOLD": "0.80",
            "OSRM_MATCH_CONFIDENCE_THRESHOLD": "0.80",
        },
        is_baseline=True,
    ),
    SensitivityVariant(
        dimension="map_match_confidence",
        value="0.90",
        label="map_match_confidence_0p90",
        run_dir_name="map_match_confidence_0p90",
        env_overrides={
            "PIPELINE_MAP_MATCH_CONFIDENCE_THRESHOLD": "0.90",
            "MAP_MATCH_CONFIDENCE_THRESHOLD": "0.90",
            "OSRM_MATCH_CONFIDENCE_THRESHOLD": "0.90",
        },
    ),
]


DIAGNOSTIC_ONLY_SETTINGS = [
    {
        "diagnostic": "gps_continuity_threshold",
        "main_value": "configured in configs.py",
        "alternative_values": "diagnostic only",
        "rerun_type": "diagnostic_only",
        "message": (
            "GPS continuity is treated as a retention/rejection diagnostic. It is "
            "not reported as a full calibration-performance sensitivity unless "
            "the pipeline explicitly reruns trip reconstruction for this threshold."
        ),
    },
    {
        "diagnostic": "minimum_observations",
        "main_value": "configured in configs.py",
        "alternative_values": "diagnostic only",
        "rerun_type": "diagnostic_only",
        "message": (
            "Minimum-observation thresholds are treated as retention diagnostics. "
            "They are not mixed with full rerun performance metrics."
        ),
    },
]


# =========================================================
# Metadata inference
# =========================================================

def _normalise_input_name(value: str) -> str:
    return str(value).strip().removeprefix("input_")


def _first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return _normalise_input_name(value)
    return None


def _parse_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}

    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue

        key, value = text.split("=", 1)
        values[key.strip()] = value.strip()

    return values


def _metadata_from_manifest_files() -> tuple[str | None, str | None]:
    candidates = [
        MAIN_RUN_DIR / "run_manifest.txt",
        MAIN_RUN_DIR / "pipeline_manifest.txt",
        MAIN_RUN_DIR / "active_run_manifest.txt",
        MAIN_RUN_DIR / "extended_run_manifest.txt",
        MAIN_RUN_DIR / "run_configuration.txt",
    ]

    merged: dict[str, str] = {}
    for path in candidates:
        merged.update(_parse_key_value_file(path))

    input_name = None
    for key in [
        "pipeline_input_name",
        "input_name",
        "dataset",
        "dataset_label",
        "main_input",
    ]:
        if key in merged and merged[key]:
            input_name = _normalise_input_name(merged[key])
            break

    region = None
    for key in [
        "main_region",
        "active_region",
        "region",
        "rav",
    ]:
        if key in merged and merged[key]:
            region = merged[key].strip()
            break

    return input_name, region


def infer_main_run_metadata() -> MainRunMetadata:
    input_name = _first_env([
        "PIPELINE_INPUT_NAME",
        "PIPELINE_MAIN_INPUT",
        "PIPELINE_DATASET",
        "PIPELINE_MAIN_DATASET",
    ])
    region = _first_env([
        "PIPELINE_MAIN_REGION",
        "PIPELINE_ACTIVE_REGION",
        "PIPELINE_REGION",
    ])

    manifest_input, manifest_region = _metadata_from_manifest_files()

    if input_name is None:
        input_name = manifest_input

    if region is None:
        region = manifest_region

    if region is None and input_name is not None and "_" in input_name:
        region = input_name.split("_", 1)[0]

    if input_name is None:
        raise RuntimeError(
            "Could not infer the main calibration input. Set PIPELINE_INPUT_NAME, "
            "for example PIPELINE_INPUT_NAME=BN_2024."
        )

    if region is None:
        raise RuntimeError(
            "Could not infer the main calibration region. Set PIPELINE_MAIN_REGION, "
            "for example PIPELINE_MAIN_REGION=BN."
        )

    return MainRunMetadata(input_name=input_name, region=region)


# =========================================================
# CSV helpers
# =========================================================

def write_preprocessing_diagnostics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "diagnostic",
        "main_value",
        "alternative_values",
        "rerun_type",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(DIAGNOSTIC_ONLY_SETTINGS)


def standardise_summary_schema(df: pl.DataFrame) -> pl.DataFrame:
    out = df

    for col in STANDARD_SUMMARY_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).alias(col))

    return out.select(STANDARD_SUMMARY_COLUMNS)


def add_sensitivity_columns(
    df: pl.DataFrame,
    *,
    variant: SensitivityVariant,
) -> pl.DataFrame:
    out = standardise_summary_schema(df)

    return out.with_columns([
        pl.lit(variant.dimension).alias("sensitivity_dimension"),
        pl.lit(variant.value).alias("sensitivity_value"),
        pl.lit(variant.label).alias("variant"),
    ])


def baseline_summary_for_variant(
    *,
    variant: SensitivityVariant,
    metadata: MainRunMetadata,
) -> pl.DataFrame:
    summary = summarise_evaluation_run(
        run_dir=MAIN_RUN_DIR,
        test_type=TEST_TYPE,
        dataset=metadata.input_name,
        region=metadata.region,
        variant=variant.label,
        status="conducted",
        message="baseline setting summarised from active main run",
    )

    return add_sensitivity_columns(summary, variant=variant)


# =========================================================
# Running variants
# =========================================================

def build_variant_env(
    *,
    run_dir: Path,
    metadata: MainRunMetadata,
    variant: SensitivityVariant,
) -> dict[str, str]:
    env = build_env(
        run_dir,
        input_name=metadata.input_name,
        active_region=metadata.region,
    )

    env["PIPELINE_SENSITIVITY_DIMENSION"] = variant.dimension
    env["PIPELINE_SENSITIVITY_VALUE"] = variant.value
    env["PIPELINE_SENSITIVITY_LABEL"] = variant.label

    for key, value in variant.env_overrides.items():
        env[key] = value

    return env


def run_full_rerun_variant(
    *,
    variant: SensitivityVariant,
    metadata: MainRunMetadata,
    overwrite: bool = True,
) -> tuple[StatusRow, pl.DataFrame]:
    run_dir = ROOT / variant.run_dir_name

    print_header(f"Sensitivity rerun: {variant.label}")

    try:
        validate_extended_input(input_dir_for(metadata.input_name))
        ensure_run_subdirs(run_dir, overwrite=overwrite)

        write_manifest(
            run_dir / "sensitivity_run_manifest.txt",
            {
                "test_type": TEST_TYPE,
                "dataset": metadata.input_name,
                "region": metadata.region,
                "variant": variant.label,
                "sensitivity_dimension": variant.dimension,
                "sensitivity_value": variant.value,
                "main_run_dir": MAIN_RUN_DIR,
                "run_dir": run_dir,
                **variant.env_overrides,
            },
        )

        env = build_variant_env(
            run_dir=run_dir,
            metadata=metadata,
            variant=variant,
        )

        failed_step = ""
        return_code = "0"
        message = "completed"

        for label, script_path in FULL_CALIBRATION_STEPS:
            ok, return_code, script_name = run_step(label, script_path, env)
            if not ok:
                failed_step = label
                message = f"Script failed: {script_name}"
                status = StatusRow(
                    test_type=TEST_TYPE,
                    dataset=metadata.input_name,
                    region=metadata.region,
                    variant=variant.label,
                    run_dir=str(run_dir),
                    status="failed",
                    failed_step=failed_step,
                    return_code=return_code,
                    message=message,
                )
                summary = summarise_evaluation_run(
                    run_dir=run_dir,
                    test_type=TEST_TYPE,
                    dataset=metadata.input_name,
                    region=metadata.region,
                    variant=variant.label,
                    status="failed",
                    message=message,
                )
                return status, add_sensitivity_columns(summary, variant=variant)

        status = StatusRow(
            test_type=TEST_TYPE,
            dataset=metadata.input_name,
            region=metadata.region,
            variant=variant.label,
            run_dir=str(run_dir),
            status="success",
            failed_step="",
            return_code="0",
            message="completed",
        )
        summary = summarise_evaluation_run(
            run_dir=run_dir,
            test_type=TEST_TYPE,
            dataset=metadata.input_name,
            region=metadata.region,
            variant=variant.label,
            status="conducted",
            message="",
        )
        return status, add_sensitivity_columns(summary, variant=variant)

    except Exception as exc:
        status = StatusRow(
            test_type=TEST_TYPE,
            dataset=metadata.input_name,
            region=metadata.region,
            variant=variant.label,
            run_dir=str(run_dir),
            status="failed",
            failed_step="setup_or_paths",
            return_code="",
            message=str(exc),
        )
        summary = summarise_evaluation_run(
            run_dir=run_dir,
            test_type=TEST_TYPE,
            dataset=metadata.input_name,
            region=metadata.region,
            variant=variant.label,
            status="failed",
            message=str(exc),
        )
        return status, add_sensitivity_columns(summary, variant=variant)


# =========================================================
# Validation helpers
# =========================================================

def warn_if_configs_may_not_read_overrides() -> None:
    config_path = None

    for candidate in [
        Path("configs.py"),
        Path("config.py"),
    ]:
        project_candidate = DATA_DIR.parent / candidate
        if project_candidate.exists():
            config_path = project_candidate
            break

    if config_path is None:
        log(
            "[WARNING] Could not find configs.py/config.py. Make sure your pipeline "
            "reads the sensitivity environment variables used by this script."
        )
        return

    text = config_path.read_text(encoding="utf-8", errors="ignore")

    expected_patterns = [
        "PIPELINE_ARRIVAL_RADIUS_M",
        "ARRIVAL_RADIUS_M",
        "PIPELINE_MAP_MATCH_CONFIDENCE_THRESHOLD",
        "MAP_MATCH_CONFIDENCE_THRESHOLD",
        "OSRM_MATCH_CONFIDENCE_THRESHOLD",
    ]

    found_any = any(pattern in text for pattern in expected_patterns)

    if not found_any:
        log(
            "[WARNING] configs.py/config.py does not appear to reference the "
            "sensitivity environment variables. The reruns may execute but still "
            "use the default settings unless configs.py supports these overrides."
        )


# =========================================================
# Main
# =========================================================

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print_header("Sensitivity analysis")

    metadata = infer_main_run_metadata()
    log(f"Main calibration input: {metadata.input_name}")
    log(f"Main calibration region: {metadata.region}")

    input_dir = input_dir_for(metadata.input_name)
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Main input directory not found: {input_dir}. "
            f"Expected input under {DATA_DIR / ('input_' + metadata.input_name)}."
        )

    warn_if_configs_may_not_read_overrides()
    write_preprocessing_diagnostics(PREPROCESSING_DIAGNOSTICS_CSV)

    statuses: list[StatusRow] = []
    summaries: list[pl.DataFrame] = []

    for variant in FULL_RERUN_VARIANTS:
        if variant.is_baseline:
            status = StatusRow(
                test_type=TEST_TYPE,
                dataset=metadata.input_name,
                region=metadata.region,
                variant=variant.label,
                run_dir=str(MAIN_RUN_DIR),
                status="success",
                failed_step="",
                return_code="0",
                message="baseline setting summarised from active main run",
            )
            summary = baseline_summary_for_variant(
                variant=variant,
                metadata=metadata,
            )
        else:
            status, summary = run_full_rerun_variant(
                variant=variant,
                metadata=metadata,
            )

        statuses.append(status)
        summaries.append(summary)

        if status.status == "success":
            log(f"[SUCCESS] {variant.label}")
        else:
            log(f"[FAILED] {variant.label}: {status.message}")

    write_status_csv(STATUS_CSV, statuses)
    write_summary(SUMMARY_CSV, summaries)

    print(f"Saved status:                   {STATUS_CSV}")
    print(f"Saved summary:                  {SUMMARY_CSV}")
    print(f"Saved preprocessing diagnostics: {PREPROCESSING_DIAGNOSTICS_CSV}")


if __name__ == "__main__":
    main()