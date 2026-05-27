"""
p1s1_build_candidate_trips.py

Phase 1, step 1.

Purpose:
    Link prepared dispatch registers and GPS logs into candidate trip GPS
    sequences.

Inputs:
    data/input_<dataset>/dispatch_registers.parquet
    data/input_<dataset>/gps_logs.parquet
    or data/input_<dataset>/gps_logs_raw_parquet/*.parquet

Output:
    runs/RunXXX/outputs/trip_gps_sequences.parquet

Method:
    For each dispatch, retain GPS observations from the same vehicle within
    a fixed post-dispatch time window. Then compute movement and
    incident-distance features for each candidate trip.

Note:
    This implementation keeps the global LazyFrame join approach:

        GPS logs INNER JOIN dispatches ON vehicle_id
        then filter to dispatch_time <= timestamp <= window_end

    This is methodologically correct and simple, but can become expensive on
    very large empirical datasets because the join first creates GPS-dispatch
    combinations by vehicle before applying the time-window filter.
"""

import polars as pl
from datetime import datetime

from paths import (
    DISPATCH_REGISTERS_PARQUET,
    GPS_LOGS_PARQUET,
    GPS_RAW_DIR,
    GPS_RAW_PATTERN,
    TRIP_GPS_SEQUENCES_PARQUET,
)

from configs import (
    MAX_TRIP_WINDOW_MINUTES,
    EARTH_RADIUS_M,
    MOVING_SPEED_THRESHOLD_KMH,
)



# =========================================================
# Input resolution
# =========================================================

def gps_scan_source() -> str:
    """
    Resolve GPS input for Polars scan_parquet.

    Preferred:
        data/input_<dataset>/gps_logs.parquet

    Fallback:
        data/input_<dataset>/gps_logs_raw_parquet/*.parquet

    This matches the prepared-input contract used by orchestrator.py and
    paths.py.
    """
    if GPS_LOGS_PARQUET.exists():
        return str(GPS_LOGS_PARQUET)

    if GPS_RAW_DIR.exists() and any(GPS_RAW_DIR.glob("*.parquet")):
        return str(GPS_RAW_PATTERN)

    raise FileNotFoundError(
        "Missing GPS input. Expected either:\n"
        f"  - {GPS_LOGS_PARQUET}\n"
        f"  - one or more parquet files in {GPS_RAW_DIR}"
    )


def check_inputs() -> None:
    """
    Validate required prepared inputs.
    """
    if not DISPATCH_REGISTERS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing dispatch table: {DISPATCH_REGISTERS_PARQUET}"
        )

    gps_scan_source()


# =========================================================
# Spatial helper
# =========================================================

def haversine_m(
    lat1: pl.Expr,
    lon1: pl.Expr,
    lat2: pl.Expr,
    lon2: pl.Expr,
) -> pl.Expr:
    """
    Compute great-circle distance in metres between two coordinate pairs.

    Inputs are Polars expressions in decimal degrees.
    """

    lat1_rad = lat1.radians()
    lon1_rad = lon1.radians()
    lat2_rad = lat2.radians()
    lon2_rad = lon2.radians()

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        (dlat / 2).sin() ** 2
        + lat1_rad.cos()
        * lat2_rad.cos()
        * ((dlon / 2).sin() ** 2)
    )

    return 2 * EARTH_RADIUS_M * a.sqrt().arcsin()


# =========================================================
# Candidate trip construction
# =========================================================

def build_trip_gps_sequences() -> pl.LazyFrame:
    """
    Link GPS observations to dispatches using vehicle_id and a fixed
    post-dispatch time window.

    Output grain:
        one row per GPS observation linked to one candidate dispatch trip.

    Output columns:
        trip_id, dispatch_id, request_id, vehicle_id, region_id, urgency,
        dispatch_time, window_end, timestamp, vehicle_lat, vehicle_lon,
        incident_lat, incident_lon, prev_timestamp, prev_vehicle_lat,
        prev_vehicle_lon, dt_sec, displacement_m, speed_kmh,
        distance_to_incident_m, moving, n_observations, max_gap_sec.
    """

    dispatches = (
        pl.scan_parquet(DISPATCH_REGISTERS_PARQUET)
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("vehicle_id").cast(pl.String),
        ])
        .with_columns([
            (
                pl.col("timestamp")
                + pl.duration(minutes=MAX_TRIP_WINDOW_MINUTES)
            ).alias("window_end"),
        ])
        .select([
            "dispatch_id",
            "request_id",
            "vehicle_id",
            "region_id",
            "urgency",
            pl.col("timestamp").alias("dispatch_time"),
            "window_end",
            "incident_lat",
            "incident_lon",
        ])
        .sort(["vehicle_id", "dispatch_time"])
        .with_columns([
            pl.col("dispatch_id").cast(pl.String).alias("trip_id")
        ])
    )

    gps = (
        pl.scan_parquet(gps_scan_source())
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("vehicle_id").cast(pl.String),
        ])
        .select([
            "timestamp",
            "vehicle_id",
            "vehicle_lat",
            "vehicle_lon",
        ])
    )

    candidate_sequences = (
        gps
        .join(dispatches, on="vehicle_id", how="inner")
        .filter(
            (pl.col("timestamp") >= pl.col("dispatch_time"))
            & (pl.col("timestamp") <= pl.col("window_end"))
        )
        .sort(["trip_id", "timestamp"])
        .with_columns([
            pl.col("timestamp")
            .shift(1)
            .over("trip_id")
            .alias("prev_timestamp"),

            pl.col("vehicle_lat")
            .shift(1)
            .over("trip_id")
            .alias("prev_vehicle_lat"),

            pl.col("vehicle_lon")
            .shift(1)
            .over("trip_id")
            .alias("prev_vehicle_lon"),
        ])
        .with_columns([
            (
                pl.col("timestamp") - pl.col("prev_timestamp")
            )
            .dt.total_seconds()
            .alias("dt_sec"),

            haversine_m(
                pl.col("prev_vehicle_lat"),
                pl.col("prev_vehicle_lon"),
                pl.col("vehicle_lat"),
                pl.col("vehicle_lon"),
            ).alias("displacement_m"),

            haversine_m(
                pl.col("vehicle_lat"),
                pl.col("vehicle_lon"),
                pl.col("incident_lat"),
                pl.col("incident_lon"),
            ).alias("distance_to_incident_m"),
        ])
        .with_columns([
            pl.col("dt_sec").fill_null(0),
            pl.col("displacement_m").fill_null(0),
        ])
        .with_columns([
            pl.when(pl.col("dt_sec") > 0)
            .then((pl.col("displacement_m") / pl.col("dt_sec")) * 3.6)
            .otherwise(0)
            .alias("speed_kmh"),
        ])
        .with_columns([
            (
                pl.col("speed_kmh") >= MOVING_SPEED_THRESHOLD_KMH
            ).alias("moving"),

            pl.len()
            .over("trip_id")
            .alias("n_observations"),

            pl.col("dt_sec")
            .max()
            .over("trip_id")
            .alias("max_gap_sec"),
        ])
        .select([
            "trip_id",
            "dispatch_id",
            "request_id",
            "vehicle_id",
            "region_id",
            "urgency",
            "dispatch_time",
            "window_end",
            "timestamp",
            "vehicle_lat",
            "vehicle_lon",
            "incident_lat",
            "incident_lon",
            "prev_timestamp",
            "prev_vehicle_lat",
            "prev_vehicle_lon",
            "dt_sec",
            "displacement_m",
            "speed_kmh",
            "distance_to_incident_m",
            "moving",
            "n_observations",
            "max_gap_sec",
        ])
        .sort(["trip_id", "timestamp"])
    )

    return candidate_sequences


# =========================================================
# Console summary
# =========================================================

def log_step(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def print_input_summary() -> None:
    """
    Print input sizes before the expensive join starts.
    """

    log_step("Reading input summaries...")

    dispatch_summary = (
        pl.scan_parquet(DISPATCH_REGISTERS_PARQUET)
        .select([
            pl.len().alias("n_dispatches"),
            pl.col("vehicle_id").n_unique().alias("n_dispatch_vehicles"),
            pl.col("timestamp").min().alias("min_dispatch_time"),
            pl.col("timestamp").max().alias("max_dispatch_time"),
        ])
        .collect()
    )

    gps_summary = (
        pl.scan_parquet(gps_scan_source())
        .select([
            pl.len().alias("n_gps_rows"),
            pl.col("vehicle_id").n_unique().alias("n_gps_vehicles"),
            pl.col("timestamp").min().alias("min_gps_time"),
            pl.col("timestamp").max().alias("max_gps_time"),
        ])
        .collect()
    )

    log_step("Dispatch input summary:")
    print(dispatch_summary)

    log_step("GPS input summary:")
    print(gps_summary)


def print_output_summary() -> None:
    """
    Print basic output diagnostics after writing the candidate sequences.
    """

    summary = (
        pl.scan_parquet(TRIP_GPS_SEQUENCES_PARQUET)
        .select([
            pl.len().alias("gps_rows"),
            pl.col("trip_id").n_unique().alias("n_trips"),
            pl.col("vehicle_id").n_unique().alias("n_vehicles"),
            pl.col("timestamp").min().alias("min_timestamp"),
            pl.col("timestamp").max().alias("max_timestamp"),
            pl.col("distance_to_incident_m")
            .min()
            .alias("min_distance_to_incident_m"),
        ])
        .collect()
    )

    print(summary)


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    log_step("Starting candidate trip GPS sequence construction.")

    check_inputs()
    print_input_summary()

    log_step("Building lazy candidate-trip query...")
    trips = build_trip_gps_sequences()

    log_step(
        "Writing candidate trip GPS sequences. "
        "This is the expensive step and may be silent for a while..."
    )

    trips.sink_parquet(TRIP_GPS_SEQUENCES_PARQUET)

    log_step(f"Saved candidate trip GPS sequences: {TRIP_GPS_SEQUENCES_PARQUET}")

    log_step("Computing output summary...")
    print_output_summary()

    log_step("Finished candidate trip GPS sequence construction.")


if __name__ == "__main__":
    main()