"""
p2s6_compute_prediction_errors.py

Phase 2, step 6.

Purpose:
    Compare realised travel times against baseline OSRM-predicted travel
    times for successfully map-matched trips.

Inputs:
    runs/RunXXX/outputs/trip_summary.parquet
    runs/RunXXX/outputs/routes.parquet

Outputs:
    runs/RunXXX/outputs/prediction_errors.parquet
    runs/RunXXX/outputs/prediction_errors.csv

Output grain:
    One row per trip with both a realised travel time and a matched OSRM
    baseline prediction.

Columns:
    trip_id
    dispatch_time
    realised_travel_time_sec
    baseline_predicted_time_sec
    prediction_error_sec
    absolute_error_sec
    percentage_error
    absolute_percentage_error
"""

import polars as pl

from paths import (
    TRIP_SUMMARY_PARQUET,
    ROUTES_PARQUET,
    PREDICTION_ERRORS_PARQUET,
    PREDICTION_ERRORS_CSV,
)


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    """
    Validate required Phase 2 inputs.
    """
    if not TRIP_SUMMARY_PARQUET.exists():
        raise FileNotFoundError(f"Missing trip summary: {TRIP_SUMMARY_PARQUET}")

    if not ROUTES_PARQUET.exists():
        raise FileNotFoundError(f"Missing matched routes: {ROUTES_PARQUET}")


# =========================================================
# Prediction errors
# =========================================================

def compute_prediction_errors() -> pl.DataFrame:
    """
    Compute signed, absolute, percentage, and absolute-percentage errors.
    """

    check_inputs()

    trip_summary = (
        pl.read_parquet(TRIP_SUMMARY_PARQUET)
        .select([
            "trip_id",
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
            "realised_travel_time_sec",
        ])
    )

    routes = (
        pl.read_parquet(ROUTES_PARQUET)
        .select([
            "trip_id",
            pl.col("osrm_time_sec").alias("baseline_predicted_time_sec"),
        ])
    )

    errors = (
        trip_summary
        .join(routes, on="trip_id", how="inner")
        .with_columns([
            (
                pl.col("realised_travel_time_sec")
                - pl.col("baseline_predicted_time_sec")
            ).alias("prediction_error_sec"),
        ])
        .with_columns([
            pl.col("prediction_error_sec")
            .abs()
            .alias("absolute_error_sec"),

            pl.when(pl.col("realised_travel_time_sec") > 0)
            .then(
                (
                    pl.col("prediction_error_sec")
                    / pl.col("realised_travel_time_sec")
                )
                * 100
            )
            .otherwise(None)
            .alias("percentage_error"),
        ])
        .with_columns([
            pl.col("percentage_error")
            .abs()
            .alias("absolute_percentage_error"),
        ])
        .sort("trip_id")
    )

    print(f"Trips in trip summary: {trip_summary.height}")
    print(f"Matched routes: {routes.height}")
    print(f"Prediction-error rows: {errors.height}")

    if errors.is_empty():
        raise RuntimeError(
            "Prediction-error table is empty. "
            "No trip_id values overlapped between trip_summary.parquet and "
            "routes.parquet."
        )

    return errors


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    print("Computing prediction errors...")

    errors = compute_prediction_errors()

    PREDICTION_ERRORS_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    errors.write_parquet(PREDICTION_ERRORS_PARQUET)
    errors.write_csv(PREDICTION_ERRORS_CSV)

    print(f"Saved: {PREDICTION_ERRORS_PARQUET}")
    print(f"Saved: {PREDICTION_ERRORS_CSV}")


if __name__ == "__main__":
    main()