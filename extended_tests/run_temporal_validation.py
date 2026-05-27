"""
run_temporal_validation.py

Evaluate the existing main calibrated profile on temporal validation datasets.

This script does not rerun Phase 3 calibration. It applies the calibrated Lua
profile from the active main run to later-period data from the same RAV region.

Expected input folder if no explicit validation inputs are provided:
    data/input_<MAIN_REGION>_2025/

For example, if PIPELINE_MAIN_REGION=BN:
    data/input_BN_2025/

Each folder must contain:
    dispatch_registers.parquet
    gps_logs.parquet or gps_logs_raw_parquet/*.parquet
    exactly one .osm.pbf file

Outputs:
    runs/RunXXX/extended_tests/temporal_validation/
        status.csv
        temporal_validation_summary.csv
        <DATASET_LABEL>/
            outputs/
            results/
            diagnostics/
            report/

Configuration:
    Preferred:
        PIPELINE_TEMPORAL_VALIDATION_INPUTS=BN_2025

    Fallback:
        PIPELINE_MAIN_REGION=BN
        or
        PIPELINE_ACTIVE_REGION=BN
"""

from __future__ import annotations

import os

from extended_test_utils import (
    DATA_DIR,
    EXTENDED_ROOT,
    ExtendedDataset,
    ensure_extended_root,
    log,
    print_header,
    run_evaluation_dataset,
    temporal_validation_datasets,
    write_status_csv,
    write_summary,
)


TEST_TYPE = "temporal_validation"

SUMMARY_CSV = EXTENDED_ROOT / TEST_TYPE / "temporal_validation_summary.csv"
STATUS_CSV = EXTENDED_ROOT / TEST_TYPE / "status.csv"


def main_region_from_env() -> str | None:
    """
    Return the main calibration region, if supplied.

    This is used only when PIPELINE_TEMPORAL_VALIDATION_INPUTS is not set.
    """
    for name in ["PIPELINE_MAIN_REGION", "PIPELINE_ACTIVE_REGION"]:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    return None


def build_datasets() -> list[ExtendedDataset]:
    """
    Build temporal-validation datasets.

    If PIPELINE_TEMPORAL_VALIDATION_INPUTS is set, use those datasets.
    Otherwise, infer the validation input from the main calibration region:
        data/input_<MAIN_REGION>_2025/
    """
    datasets = temporal_validation_datasets()
    if datasets:
        return datasets

    main_region = main_region_from_env()
    if main_region is None:
        raise RuntimeError(
            "No temporal-validation dataset could be inferred. Set either "
            "PIPELINE_TEMPORAL_VALIDATION_INPUTS, for example 'BN_2025', or set "
            "PIPELINE_MAIN_REGION / PIPELINE_ACTIVE_REGION so the script can look "
            "for data/input_<MAIN_REGION>_2025."
        )

    label = f"{main_region}_2025"

    return [
        ExtendedDataset(
            key=label,
            label=label,
            region=main_region,
            test_type=TEST_TYPE,
            input_dir=DATA_DIR / f"input_{label}",
        )
    ]


def main() -> None:
    ensure_extended_root()
    print_header("Temporal validation")

    datasets = build_datasets()

    statuses = []
    summaries = []

    for dataset in datasets:
        status, summary = run_evaluation_dataset(dataset)
        statuses.append(status)
        summaries.append(summary)

        if status.status == "success":
            log(f"[SUCCESS] {dataset.label}")
        else:
            log(f"[FAILED] {dataset.label}: {status.message}")

    write_status_csv(STATUS_CSV, statuses)
    write_summary(SUMMARY_CSV, summaries)

    print(f"Saved status:  {STATUS_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()