"""
Phase 1 diagnostics: realised travel-time determination.

This script reads the outputs of:
  - p1s1_build_candidate_trips.py
  - p1s2_identify_trip_start_end.py

It does not change any pipeline outputs. It only writes diagnostic CSV files to:
  results/diagnostics/

Diagnostic purpose:
  Check whether realised dispatch-to-arrival travel times were reconstructed
  plausibly from dispatch and GPS data.
"""

from __future__ import annotations

from pathlib import Path
import sys

import polars as pl


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    RESULTS_DIR,
    TRIP_GPS_SEQUENCES_PARQUET,
    TRIP_REJECTION_SUMMARY_PARQUET,
    TRIP_SUMMARY_PARQUET,
)

from configs import (
    ARRIVAL_RADIUS_M,
    CONSISTENT_MOVEMENT_SECONDS,
    MAX_TRIP_GAP_SEC,
    MIN_OBSERVATIONS,
    MOVING_SPEED_THRESHOLD_KMH,
)


DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def ensure_diagnostics_dir() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def pct(count: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round((count / total) * 100, 2)


def write_table(df: pl.DataFrame, filename: str) -> None:
    path = DIAGNOSTICS_DIR / filename
    df.write_csv(path)
    print(f"Saved: {path}")


def describe_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Return n, mean, median, min, selected quantiles, and max for one numeric column.

    Values are explicitly stored as Float64 so Polars does not fail when
    integer counts and floating-point statistics appear in the same column.
    """
    if column not in df.columns:
        return pl.DataFrame(
            {
                "statistic": ["missing_column"],
                "value": [None],
            },
            schema={
                "statistic": pl.String,
                "value": pl.Float64,
            },
            strict=False,
        )

    s = df.get_column(column).drop_nulls()

    if s.is_empty():
        rows = [
            ("n_non_null", 0.0),
            ("mean", None),
            ("median", None),
            ("min", None),
            ("p05", None),
            ("p25", None),
            ("p75", None),
            ("p95", None),
            ("max", None),
        ]
    else:
        rows = [
            ("n_non_null", float(len(s))),
            ("mean", s.mean()),
            ("median", s.median()),
            ("min", s.min()),
            ("p05", s.quantile(0.05)),
            ("p25", s.quantile(0.25)),
            ("p75", s.quantile(0.75)),
            ("p95", s.quantile(0.95)),
            ("max", s.max()),
        ]

    return pl.DataFrame(
        {
            "statistic": [name for name, _ in rows],
            "value": [None if value is None else float(value) for _, value in rows],
        },
        schema={
            "statistic": pl.String,
            "value": pl.Float64,
        },
        strict=False,
    )


# ---------------------------------------------------------------------
# Diagnostic tables
# ---------------------------------------------------------------------

def retention_summary(diagnostics: pl.DataFrame) -> pl.DataFrame:
    total = diagnostics.height
    kept = diagnostics.select(pl.col("kept").sum()).item() if "kept" in diagnostics.columns else 0
    rejected = total - kept

    return pl.DataFrame({
        "metric": ["candidate_trips", "kept_trips", "rejected_trips"],
        "value": [total, kept, rejected],
        "percentage_of_candidates": [100.0, pct(kept, total), pct(rejected, total)],
    })


def rejection_reasons(diagnostics: pl.DataFrame) -> pl.DataFrame:
    total = diagnostics.height

    reasons = [
        ("no_sustained_start", True),
        ("no_arrival", True),
        ("arrival_before_start", True),
        ("too_few_observations", False),
        ("too_large_trip_gap", False),
    ]

    rows = []
    for reason, hard_filter in reasons:
        if reason not in diagnostics.columns:
            continue

        count = diagnostics.select(pl.col(reason).sum()).item()
        rows.append({
            "reason": reason,
            "count": int(count),
            "percentage_of_candidates": pct(count, total),
            "hard_filter": hard_filter,
        })

    return pl.DataFrame(rows)


def arrival_distance_thresholds(trip_gps: pl.DataFrame) -> pl.DataFrame:
    min_distances = (
        trip_gps
        .group_by("trip_id")
        .agg(pl.col("distance_to_incident_m").min().alias("min_distance_to_incident_m"))
    )

    total = min_distances.height
    thresholds = sorted(set([50, 100, 150, 200, 300, 500, 1000, 5000, ARRIVAL_RADIUS_M]))

    rows = []
    for threshold in thresholds:
        count = min_distances.filter(pl.col("min_distance_to_incident_m") <= threshold).height
        rows.append({
            "threshold_m": threshold,
            "trips_within_threshold": count,
            "percentage_of_candidate_trips": pct(count, total),
        })

    return pl.DataFrame(rows)


def realised_travel_time_summary(trip_summary: pl.DataFrame) -> pl.DataFrame:
    if "realised_travel_time_sec" not in trip_summary.columns:
        return pl.DataFrame({"statistic": ["missing_column"], "realised_travel_time_sec": [None]})

    df = trip_summary.with_columns(
        (pl.col("realised_travel_time_sec") / 60).alias("realised_travel_time_min")
    )

    sec = describe_column(df, "realised_travel_time_sec")
    minutes = describe_column(df, "realised_travel_time_min")

    return sec.join(minutes, on="statistic", how="left")


def gps_quality_summary(diagnostics: pl.DataFrame) -> pl.DataFrame:
    """
    Summarise GPS-quality indicators for reconstructed trips.

    The output has one block per available diagnostic variable.
    Missing variables are skipped rather than causing the diagnostics script
    to fail.
    """
    candidate_columns = [
        "candidate_observations",
        "clean_observations",
        "n_observations",
        "max_gap_sec",
        "mean_gap_sec",
        "median_gap_sec",
        "trace_duration_sec",
    ]

    summaries: list[pl.DataFrame] = []

    for column in candidate_columns:
        if column not in diagnostics.columns:
            continue

        summaries.append(
            describe_column(diagnostics, column)
            .with_columns(pl.lit(column).alias("variable"))
            .select(["variable", "statistic", "value"])
        )

    if not summaries:
        return pl.DataFrame(
            {
                "variable": ["none"],
                "statistic": ["no_available_gps_quality_columns"],
                "value": [None],
            },
            schema={
                "variable": pl.String,
                "statistic": pl.String,
                "value": pl.Float64,
            },
            strict=False,
        )

    return pl.concat(summaries, how="vertical")


def coordinate_sanity_checks(trip_gps: pl.DataFrame) -> pl.DataFrame:
    total_rows = trip_gps.height

    min_distance_by_trip = (
        trip_gps
        .group_by("trip_id")
        .agg(pl.col("distance_to_incident_m").min().alias("min_distance_to_incident_m"))
    )

    total_trips = min_distance_by_trip.height

    checks = [
        {
            "check": "gps_rows",
            "count": total_rows,
            "percentage": 100.0,
            "note": "Total GPS rows in candidate trip table",
        },
        {
            "check": "vehicle_lat_is_0",
            "count": trip_gps.filter(pl.col("vehicle_lat") == 0).height,
            "percentage": pct(trip_gps.filter(pl.col("vehicle_lat") == 0).height, total_rows),
            "note": "Likely invalid GPS coordinate",
        },
        {
            "check": "vehicle_lon_is_0",
            "count": trip_gps.filter(pl.col("vehicle_lon") == 0).height,
            "percentage": pct(trip_gps.filter(pl.col("vehicle_lon") == 0).height, total_rows),
            "note": "Likely invalid GPS coordinate",
        },
        {
            "check": "vehicle_coordinate_outside_nl_bbox",
            "count": trip_gps.filter(
                ~(
                    pl.col("vehicle_lat").is_between(50.0, 54.0)
                    & pl.col("vehicle_lon").is_between(3.0, 8.0)
                )
            ).height,
            "percentage": pct(
                trip_gps.filter(
                    ~(
                        pl.col("vehicle_lat").is_between(50.0, 54.0)
                        & pl.col("vehicle_lon").is_between(3.0, 8.0)
                    )
                ).height,
                total_rows,
            ),
            "note": "Broad Netherlands bounding-box check",
        },
        {
            "check": "incident_coordinate_outside_nl_bbox",
            "count": trip_gps.select(["trip_id", "incident_lat", "incident_lon"]).unique().filter(
                ~(
                    pl.col("incident_lat").is_between(50.0, 54.0)
                    & pl.col("incident_lon").is_between(3.0, 8.0)
                )
            ).height,
            "percentage": pct(
                trip_gps.select(["trip_id", "incident_lat", "incident_lon"]).unique().filter(
                    ~(
                        pl.col("incident_lat").is_between(50.0, 54.0)
                        & pl.col("incident_lon").is_between(3.0, 8.0)
                    )
                ).height,
                total_trips,
            ),
            "note": "Broad Netherlands bounding-box check, trip-level",
        },
        {
            "check": "trips_never_within_5km_of_incident",
            "count": min_distance_by_trip.filter(pl.col("min_distance_to_incident_m") > 5000).height,
            "percentage": pct(
                min_distance_by_trip.filter(pl.col("min_distance_to_incident_m") > 5000).height,
                total_trips,
            ),
            "note": "Potential coordinate, linking, cancellation, or window issue",
        },
    ]

    return pl.DataFrame(checks)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    ensure_diagnostics_dir()

    print("Loading Phase 1 outputs...")
    trip_gps = pl.read_parquet(TRIP_GPS_SEQUENCES_PARQUET)
    diagnostics = pl.read_parquet(TRIP_REJECTION_SUMMARY_PARQUET)
    trip_summary = pl.read_parquet(TRIP_SUMMARY_PARQUET)

    print("Writing Phase 1 diagnostic tables...")
    write_table(retention_summary(diagnostics), "phase1_retention_summary.csv")
    write_table(rejection_reasons(diagnostics), "phase1_rejection_reasons.csv")
    write_table(arrival_distance_thresholds(trip_gps), "phase1_arrival_distance_thresholds.csv")
    write_table(realised_travel_time_summary(trip_summary), "phase1_realised_travel_time_summary.csv")
    write_table(gps_quality_summary(diagnostics), "phase1_gps_quality_summary.csv")
    write_table(coordinate_sanity_checks(trip_gps), "phase1_coordinate_sanity_checks.csv")

    print("\nPhase 1 diagnostic summary")
    print("--------------------------")
    print(f"Current ARRIVAL_RADIUS_M: {ARRIVAL_RADIUS_M}")
    print(f"Current CONSISTENT_MOVEMENT_SECONDS: {CONSISTENT_MOVEMENT_SECONDS}")
    print(f"Current MOVING_SPEED_THRESHOLD_KMH: {MOVING_SPEED_THRESHOLD_KMH}")
    print(f"Diagnostic MAX_TRIP_GAP_SEC: {MAX_TRIP_GAP_SEC}")
    print(f"Diagnostic MIN_OBSERVATIONS: {MIN_OBSERVATIONS}")
    print(retention_summary(diagnostics))


if __name__ == "__main__":
    main()