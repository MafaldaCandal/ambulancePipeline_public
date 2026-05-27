"""
p1s2_identify_trip_start_end.py

Phase 1, step 2.

Purpose:
    Convert candidate trip GPS sequences into realised dispatch-to-arrival
    trajectories.

Input:
    runs/RunXXX/outputs/trip_gps_sequences.parquet

Outputs:
    runs/RunXXX/outputs/clean_trajectories.parquet
    runs/RunXXX/outputs/trip_summary.parquet
    runs/RunXXX/outputs/trip_rejection_summary.parquet

Hard exclusions:
    1. no sustained post-dispatch movement
    2. no arrival within incident radius after start
    3. arrival before or equal to start

Diagnostic-only quality checks, such as too few observations or large GPS gaps,
belong in diagnostics/diagnose_phase1.py, not in this methodology step.
"""

import polars as pl

from paths import (
    TRIP_GPS_SEQUENCES_PARQUET,
    CLEAN_TRAJECTORIES_PARQUET,
    TRIP_SUMMARY_PARQUET,
    TRIP_REJECTION_SUMMARY_PARQUET,
)

from configs import (
    ARRIVAL_RADIUS_M,
    CONSISTENT_MOVEMENT_SECONDS,
)


# =========================================================
# Realised trip bounds
# =========================================================

def find_trip_bounds() -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Find realised start and arrival timestamps for each candidate trip.

    Retention logic:
        - keep trips with a valid sustained-movement start;
        - keep trips with an arrival inside the incident radius after start;
        - require arrival after start.

    Returns:
        trip_bounds:
            One row per retained trip with start, arrival, and realised
            travel time.

        rejection_summary:
            One row per candidate trip with hard rejection flags.
    """

    df_raw = (
        pl.read_parquet(TRIP_GPS_SEQUENCES_PARQUET)
        .sort(["trip_id", "timestamp"])
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
    )

    all_trips = (
        df_raw
        .group_by("trip_id")
        .agg([
            pl.col("dispatch_id").first(),
            pl.col("request_id").first(),
            pl.col("vehicle_id").first(),
            pl.col("region_id").first(),
            pl.col("urgency").first(),
            pl.col("dispatch_time").first(),
            pl.len().alias("candidate_observations"),
        ])
    )

    # -----------------------------------------------------
    # Sustained movement start
    # -----------------------------------------------------
    # Start is defined as the first moving timestamp for which there is
    # another moving timestamp at least CONSISTENT_MOVEMENT_SECONDS later.
    # This is intentionally permissive; detailed trajectory quality is
    # assessed later through map-matching and diagnostics.

    moving_points = (
        df_raw
        .filter(pl.col("moving"))
        .select([
            "trip_id",
            "timestamp",
        ])
        .sort(["trip_id", "timestamp"])
        .with_columns([
            pl.col("timestamp").max().over("trip_id").alias("last_moving_timestamp"),
        ])
    )

    start_candidates = (
        moving_points
        .filter(
            (
                pl.col("last_moving_timestamp") - pl.col("timestamp")
            )
            .dt.total_seconds()
            >= CONSISTENT_MOVEMENT_SECONDS
        )
        .group_by("trip_id")
        .agg(
            pl.col("timestamp").min().alias("start_timestamp")
        )
    )

    # -----------------------------------------------------
    # Arrival after start
    # -----------------------------------------------------
    # Arrival is the first timestamp after the detected start at which the
    # vehicle is within ARRIVAL_RADIUS_M of the incident location.

    arrival_candidates = (
        df_raw
        .join(start_candidates, on="trip_id", how="inner")
        .filter(
            (pl.col("timestamp") >= pl.col("start_timestamp"))
            & (pl.col("distance_to_incident_m") <= ARRIVAL_RADIUS_M)
        )
        .group_by("trip_id")
        .agg(
            pl.col("timestamp").min().alias("arrival_timestamp")
        )
    )

    # -----------------------------------------------------
    # Bounds and hard rejection flags
    # -----------------------------------------------------

    rejection_summary = (
        all_trips
        .join(start_candidates, on="trip_id", how="left")
        .join(arrival_candidates, on="trip_id", how="left")
        .with_columns([
            pl.col("start_timestamp")
            .is_null()
            .alias("no_sustained_start"),

            pl.col("arrival_timestamp")
            .is_null()
            .alias("no_arrival"),

            (
                pl.col("start_timestamp").is_not_null()
                & pl.col("arrival_timestamp").is_not_null()
                & (
                    pl.col("arrival_timestamp")
                    <= pl.col("start_timestamp")
                )
            ).alias("arrival_before_start"),
        ])
        .with_columns([
            ~(
                pl.col("no_sustained_start")
                | pl.col("no_arrival")
                | pl.col("arrival_before_start")
            ).alias("kept")
        ])
        .sort("trip_id")
    )

    trip_bounds = (
        rejection_summary
        .filter(pl.col("kept"))
        .select([
            "trip_id",
            "start_timestamp",
            "arrival_timestamp",
        ])
        .with_columns([
            (
                pl.col("arrival_timestamp") - pl.col("start_timestamp")
            )
            .dt.total_seconds()
            .alias("realised_travel_time_sec")
        ])
        .sort("trip_id")
    )

    return trip_bounds, rejection_summary


# =========================================================
# Clean realised trajectories
# =========================================================

def extract_clean_trajectories(
    trip_bounds: pl.DataFrame,
) -> pl.DataFrame:
    """
    Keep only GPS observations between realised start and arrival.
    """

    return (
        pl.read_parquet(TRIP_GPS_SEQUENCES_PARQUET)
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
        .join(trip_bounds, on="trip_id", how="inner")
        .filter(
            (pl.col("timestamp") >= pl.col("start_timestamp"))
            & (pl.col("timestamp") <= pl.col("arrival_timestamp"))
        )
        .sort(["trip_id", "timestamp"])
    )


# =========================================================
# Trip summary
# =========================================================

def build_trip_summary(
    clean_trajectories: pl.DataFrame,
) -> pl.DataFrame:
    """
    Build one-row-per-trip summary from retained trajectories.

    The GPS gap is recomputed inside the realised segment, rather than using
    the candidate-window dt_sec from p1s1.
    """

    clean_with_segment_gaps = (
        clean_trajectories
        .sort(["trip_id", "timestamp"])
        .with_columns([
            (
                pl.col("timestamp")
                - pl.col("timestamp").shift(1).over("trip_id")
            )
            .dt.total_seconds()
            .fill_null(0)
            .alias("segment_dt_sec")
        ])
    )

    return (
        clean_with_segment_gaps
        .group_by("trip_id")
        .agg([
            pl.col("dispatch_id").first(),
            pl.col("request_id").first(),
            pl.col("vehicle_id").first(),
            pl.col("region_id").first(),
            pl.col("urgency").first(),
            pl.col("dispatch_time").first(),
            pl.col("start_timestamp").first(),
            pl.col("arrival_timestamp").first(),
            pl.col("realised_travel_time_sec").first(),
            pl.col("incident_lat").first(),
            pl.col("incident_lon").first(),

            # Retained realised-segment diagnostics.
            pl.len().alias("n_observations"),
            pl.col("segment_dt_sec").max().alias("max_gap_sec"),
        ])
        .sort("trip_id")
    )


# =========================================================
# Console summary
# =========================================================

def print_rejection_summary(
    rejection_summary: pl.DataFrame,
) -> None:
    """
    Print hard retention and rejection summary.
    """

    summary = rejection_summary.select([
        pl.len().alias("candidate_trips"),
        pl.col("kept").sum().alias("kept_trips"),
        (~pl.col("kept")).sum().alias("rejected_trips"),
        pl.col("no_sustained_start").sum().alias("no_sustained_start"),
        pl.col("no_arrival").sum().alias("no_arrival"),
        pl.col("arrival_before_start").sum().alias("arrival_before_start"),
    ])

    print(summary)


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    print("Finding realised trip bounds...")

    trip_bounds, rejection_summary = find_trip_bounds()

    print("Extracting clean realised trajectories...")

    clean_trajectories = extract_clean_trajectories(trip_bounds)
    trip_summary = build_trip_summary(clean_trajectories)

    clean_trajectories.write_parquet(CLEAN_TRAJECTORIES_PARQUET)
    trip_summary.write_parquet(TRIP_SUMMARY_PARQUET)
    rejection_summary.write_parquet(TRIP_REJECTION_SUMMARY_PARQUET)

    print(f"Saved clean trajectories: {CLEAN_TRAJECTORIES_PARQUET}")
    print(f"Saved trip summary: {TRIP_SUMMARY_PARQUET}")
    print(f"Saved trip rejection summary: {TRIP_REJECTION_SUMMARY_PARQUET}")

    print_rejection_summary(rejection_summary)


if __name__ == "__main__":
    main()