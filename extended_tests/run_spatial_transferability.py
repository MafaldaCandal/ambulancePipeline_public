"""
run_spatial_transferability.py

Evaluate the existing main calibrated profile on transfer regions.

This script does not rerun Phase 3 calibration. It applies the calibrated Lua
profile from the active main run to other RAV regions and compares baseline,
calibrated, and naive-multiplier predictions using the standard evaluation
summary contract.

Expected input folders:
    data/input_BN_2024/
    data/input_BMW_2024/
    data/input_BZO_2024/

Each folder must contain:
    dispatch_registers.parquet
    gps_logs.parquet or gps_logs_raw_parquet/*.parquet
    exactly one .osm.pbf file

Outputs:
    runs/RunXXX/extended_tests/spatial_transferability/
        status.csv
        spatial_transferability_summary.csv
        BN_2024/
        BMW_2024/
        BZO_2024/

If PIPELINE_ACTIVE_REGION or PIPELINE_MAIN_REGION is set, that region is excluded
from the transfer set. This prevents evaluating the calibration region as a
spatial-transfer region.
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
    spatial_transfer_datasets,
    write_status_csv,
    write_summary,
)


TEST_TYPE = "spatial_transferability"

SUMMARY_CSV = EXTENDED_ROOT / TEST_TYPE / "spatial_transferability_summary.csv"
STATUS_CSV = EXTENDED_ROOT / TEST_TYPE / "status.csv"


DEFAULT_TRANSFER_DATASETS = [
    ExtendedDataset(
        key="BN_2024",
        label="BN_2024",
        region="BN",
        test_type=TEST_TYPE,
        input_dir=DATA_DIR / "input_BN_2024",
    ),
    ExtendedDataset(
        key="BMW_2024",
        label="BMW_2024",
        region="BMW",
        test_type=TEST_TYPE,
        input_dir=DATA_DIR / "input_BMW_2024",
    ),
    ExtendedDataset(
        key="BZO_2024",
        label="BZO_2024",
        region="BZO",
        test_type=TEST_TYPE,
        input_dir=DATA_DIR / "input_BZO_2024",
    ),
]


def main_region_from_env() -> str | None:
    """
    Return the region used for the active main calibration run, if supplied.

    The orchestrator may expose this either as PIPELINE_MAIN_REGION or
    PIPELINE_ACTIVE_REGION. If neither is set, the script runs all configured
    transfer datasets and leaves interpretation to the user.
    """
    for name in ["PIPELINE_MAIN_REGION", "PIPELINE_ACTIVE_REGION"]:
        value = os.environ.get(name, "").strip()
        if value:
            return value

    return None


def build_datasets() -> list[ExtendedDataset]:
    """
    Build transfer datasets.

    If PIPELINE_SPATIAL_TRANSFER_INPUTS is set, use the utility-provided datasets.
    Otherwise, use the three Brabant defaults.

    Then exclude the main calibration region when it is known.
    """
    datasets = spatial_transfer_datasets()
    if not datasets:
        datasets = DEFAULT_TRANSFER_DATASETS

    main_region = main_region_from_env()
    if main_region is None:
        return datasets

    kept = [
        dataset for dataset in datasets
        if dataset.region.lower() != main_region.lower()
    ]

    if len(kept) == len(datasets):
        log(
            f"Main region '{main_region}' was not present in the configured "
            "transfer datasets; no dataset was excluded."
        )
    else:
        excluded = sorted(
            dataset.label for dataset in datasets
            if dataset.region.lower() == main_region.lower()
        )
        log(
            "Excluded main calibration region from spatial transferability: "
            + ", ".join(excluded)
        )

    return kept


def main() -> None:
    ensure_extended_root()
    print_header("Spatial transferability")

    datasets = build_datasets()

    if not datasets:
        raise RuntimeError(
            "No spatial-transfer datasets are configured. Set "
            "PIPELINE_SPATIAL_TRANSFER_INPUTS or provide the default "
            "data/input_BN_2024, data/input_BMW_2024, and data/input_BZO_2024 folders."
        )

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