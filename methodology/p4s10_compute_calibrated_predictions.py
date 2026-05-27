"""
p4s10_generate_calibrated_predictions.py

Phase 4, step 10.

Purpose:
    Generate calibrated OSRM predictions for the active run.

Inputs:
    runs/RunXXX/ambulance_nl_calibrated.lua
    runs/RunXXX/outputs/clean_trajectories.parquet
    runs/RunXXX/outputs/prediction_errors.parquet

Output:
    runs/RunXXX/outputs/calibrated_predictions.parquet

Design:
    This core file evaluates the calibrated profile on the active run only.
    Off-sample validation, transferability, and sensitivity analyses belong
    in extended_tests/.
"""

from __future__ import annotations

import polars as pl

from paths import (
    CALIBRATED_LUA,
    CLEAN_TRAJECTORIES_PARQUET,
    PREDICTION_ERRORS_PARQUET,
    CALIBRATED_PREDICTIONS_PARQUET,
)

from utils.osrm_utils import (
    run_osrm_preprocessing,
    start_osrm_server,
    stop_osrm_server,
    wait_for_osrm,
    query_osrm_match,
)


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    """
    Validate required Phase 4 inputs.
    """
    required_paths = [
        CALIBRATED_LUA,
        CLEAN_TRAJECTORIES_PARQUET,
        PREDICTION_ERRORS_PARQUET,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


# =========================================================
# Baseline data loading
# =========================================================

def load_baseline_errors() -> pl.DataFrame:
    """
    Load baseline prediction errors for trips retained after map-matching.
    """

    baseline_errors = (
        pl.read_parquet(PREDICTION_ERRORS_PARQUET)
        .with_columns([
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
        .select([
            "trip_id",
            "dispatch_time",
            "realised_travel_time_sec",
            "baseline_predicted_time_sec",
            "prediction_error_sec",
            "absolute_error_sec",
            "percentage_error",
            "absolute_percentage_error",
        ])
    )

    if baseline_errors.is_empty():
        raise RuntimeError(
            f"Baseline prediction-error file is empty: {PREDICTION_ERRORS_PARQUET}"
        )

    return baseline_errors


# =========================================================
# Calibrated prediction generation
# =========================================================

def compute_calibrated_predictions() -> pl.DataFrame:
    """
    Re-map-match clean trajectories using the calibrated OSRM profile and
    return trip-level calibrated predictions.
    """

    baseline_errors = load_baseline_errors()
    valid_trip_ids = baseline_errors.select("trip_id")

    gps = (
        pl.read_parquet(CLEAN_TRAJECTORIES_PARQUET)
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
        ])
        .join(valid_trip_ids, on="trip_id", how="inner")
        .sort(["trip_id", "timestamp"])
    )

    if gps.is_empty():
        raise RuntimeError(
            "No clean trajectory rows matched the baseline prediction-error trips."
        )

    rows = []
    rejections = []

    total_trips = 0
    kept_trips = 0

    for trip in gps.partition_by("trip_id"):
        total_trips += 1
        trip_id = trip["trip_id"][0]

        result = query_osrm_match(
            trip,
            include_geometry=False,
        )

        if result["status"] != "ok":
            rejections.append({
                "trip_id": trip_id,
                "rejection_reason": result["status"],
                "error_message": result.get("error_message"),
                "calibrated_match_confidence": result.get("match_confidence"),
                "n_points_original": result.get("n_points_original"),
                "n_points_submitted": result.get("n_points_submitted"),
                "trace_duration_sec": result.get("trace_duration_sec"),
                "trace_max_gap_sec": result.get("trace_max_gap_sec"),
                "request_url_length": result.get("request_url_length"),
            })
            continue

        kept_trips += 1

        rows.append({
            "trip_id": trip_id,
            "calibrated_osrm_time_sec": result["osrm_time_sec"],
            "calibrated_distance_km": result["distance_km"],
            "calibrated_match_confidence": result["match_confidence"],
            "n_points_original": result["n_points_original"],
            "n_points_submitted": result["n_points_submitted"],
            "trace_duration_sec": result["trace_duration_sec"],
            "trace_max_gap_sec": result["trace_max_gap_sec"],
            "request_url_length": result["request_url_length"],
        })

    if not rows:
        rejection_path = CALIBRATED_PREDICTIONS_PARQUET.with_name(
            "calibrated_prediction_rejections.parquet"
        )

        if rejections:
            pl.DataFrame(rejections).write_parquet(rejection_path)

        raise RuntimeError(
            "No calibrated predictions were produced. "
            f"Total submitted trips: {total_trips}. "
            f"Rejections saved to: {rejection_path if rejections else 'not available'}"
        )

    calibrated = pl.DataFrame(rows)

    out = (
        baseline_errors
        .join(calibrated, on="trip_id", how="inner")
        .with_columns([
            (
                pl.col("realised_travel_time_sec")
                - pl.col("calibrated_osrm_time_sec")
            ).alias("calibrated_prediction_error_sec"),
        ])
        .with_columns([
            pl.col("calibrated_prediction_error_sec")
            .abs()
            .alias("calibrated_absolute_error_sec"),
        ])
        .sort("trip_id")
    )

    if out.is_empty():
        raise RuntimeError(
            "Calibrated prediction output is empty after joining baseline and "
            "calibrated predictions."
        )

    if rejections:
        rejection_path = CALIBRATED_PREDICTIONS_PARQUET.with_name(
            "calibrated_prediction_rejections.parquet"
        )
        pl.DataFrame(rejections).write_parquet(rejection_path)
        print(f"Saved calibrated rejection summary: {rejection_path}")

    print(f"Total trips submitted to calibrated OSRM: {total_trips}")
    print(f"Kept calibrated matches: {kept_trips}")
    print(f"Rejected calibrated matches: {len(rejections)}")

    return out


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    check_inputs()

    run_osrm_preprocessing(CALIBRATED_LUA)

    server = start_osrm_server()

    try:
        wait_for_osrm()

        print("Computing calibrated predictions...")
        calibrated_predictions = compute_calibrated_predictions()

        CALIBRATED_PREDICTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        calibrated_predictions.write_parquet(CALIBRATED_PREDICTIONS_PARQUET)

        print(f"Saved: {CALIBRATED_PREDICTIONS_PARQUET}")

    finally:
        stop_osrm_server(server)


if __name__ == "__main__":
    main()