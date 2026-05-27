"""
p3s7_prepare_regression_dataset.py

Phase 3, step 7.

Purpose:
    Create the modelling-ready regression table used by Phase 3 regression
    scripts.

Inputs:
    runs/RunXXX/outputs/prediction_errors.parquet
    runs/RunXXX/outputs/route_features.parquet
    runs/RunXXX/outputs/trip_gps_sequences.parquet
    runs/RunXXX/outputs/trip_summary.parquet

Outputs:
    runs/RunXXX/outputs/regression_table.parquet
    runs/RunXXX/outputs/regression_table.csv

Design:
    This file prepares the full calibration-ready dataset. It does not apply
    off-sample validation splits. Training/validation splits belong in
    extended_tests/.
"""

import polars as pl

from paths import (
    TRIP_SUMMARY_PARQUET,
    TRIP_GPS_SEQUENCES_PARQUET,
    ROUTE_FEATURES_PARQUET,
    PREDICTION_ERRORS_PARQUET,
    REGRESSION_TABLE_PARQUET,
    REGRESSION_TABLE_CSV,
)

from configs import (
    ROAD_CLASSES,
    MOVING_AT_DISPATCH_WINDOW_SEC,
    IQR_OUTLIER_MULTIPLIER,
)


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    """
    Validate required Phase 3 inputs.
    """
    required_paths = [
        TRIP_SUMMARY_PARQUET,
        TRIP_GPS_SEQUENCES_PARQUET,
        ROUTE_FEATURES_PARQUET,
        PREDICTION_ERRORS_PARQUET,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required input file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


# =========================================================
# Explanatory variables
# =========================================================

def get_explanatory_variables() -> pl.DataFrame:
    """
    Read route features and construct kilometre and proportion variables.

    Proportions are computed using non-unclassified road distance as the
    denominator. This keeps the explanatory proportion model focused on
    interpretable road classes.
    """

    road_cols = [f"km_{cls}" for cls in ROAD_CLASSES]
    denominator_cols = [col for col in road_cols if col != "km_unclassified"]

    route_features = (
        pl.read_parquet(ROUTE_FEATURES_PARQUET)
        .select([
            "trip_id",
            *road_cols,
            "n_turns",
        ])
        .with_columns([
            pl.col("n_turns").fill_null(0).cast(pl.Float64),
            *[
                pl.col(col).fill_null(0.0).cast(pl.Float64)
                for col in road_cols
            ],
        ])
        .with_columns([
            pl.sum_horizontal([
                pl.col(col)
                for col in denominator_cols
            ]).alias("road_km_without_unclassified")
        ])
    )

    prop_exprs = []

    for cls in ROAD_CLASSES:
        if cls == "unclassified":
            continue

        prop_exprs.append(
            pl.when(pl.col("road_km_without_unclassified") > 0)
            .then(pl.col(f"km_{cls}") / pl.col("road_km_without_unclassified"))
            .otherwise(0.0)
            .alias(f"prop_{cls}")
        )

    return route_features.with_columns(prop_exprs)


# =========================================================
# Controls
# =========================================================

def get_controls() -> pl.DataFrame:
    """
    Construct trip-level controls.

    Controls:
        - route distance;
        - whether the vehicle was already moving shortly after dispatch;
        - whether the deployment urgency was A0.
    """

    distance_control = (
        pl.read_parquet(ROUTE_FEATURES_PARQUET)
        .select([
            "trip_id",
            pl.col("distance_km").cast(pl.Float64),
        ])
    )

    moving_at_dispatch = (
        pl.read_parquet(TRIP_GPS_SEQUENCES_PARQUET)
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
        .with_columns([
            (
                (
                    (pl.col("timestamp") >= pl.col("dispatch_time"))
                    & (
                        (pl.col("timestamp") - pl.col("dispatch_time"))
                        .dt.total_seconds()
                        <= MOVING_AT_DISPATCH_WINDOW_SEC
                    )
                    & pl.col("moving")
                )
                .max()
                .over("trip_id")
                .alias("moving_at_dispatch")
            )
        ])
        .group_by("trip_id")
        .agg(pl.col("moving_at_dispatch").first())
    )

    is_a0 = (
        pl.read_parquet(TRIP_SUMMARY_PARQUET)
        .select([
            "trip_id",
            (pl.col("urgency") == "A0").alias("is_A0"),
        ])
    )

    return (
        distance_control
        .join(moving_at_dispatch, on="trip_id", how="left")
        .join(is_a0, on="trip_id", how="left")
        .with_columns([
            pl.col("moving_at_dispatch").fill_null(False).cast(pl.Boolean),
            pl.col("is_A0").fill_null(False).cast(pl.Boolean),
        ])
    )


# =========================================================
# Temporal variables
# =========================================================

def get_temporal_variables() -> pl.DataFrame:
    """
    Construct temporal categories used in regression variants.

    Times are converted to Europe/Amsterdam before categorisation.
    """

    return (
        pl.read_parquet(PREDICTION_ERRORS_PARQUET)
        .select([
            "trip_id",
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
        .with_columns([
            pl.col("dispatch_time")
            .dt.convert_time_zone("Europe/Amsterdam")
            .alias("dispatch_time_local")
        ])
        .with_columns([
            pl.col("dispatch_time_local").dt.month().alias("dispatch_month"),
            pl.col("dispatch_time_local").dt.hour().alias("dispatch_hour"),
        ])
        .with_columns([
            pl.when(pl.col("dispatch_month").is_in([3, 4, 5]))
            .then(pl.lit("spring"))
            .when(pl.col("dispatch_month").is_in([6, 7, 8]))
            .then(pl.lit("summer"))
            .when(pl.col("dispatch_month").is_in([9, 10, 11]))
            .then(pl.lit("autumn"))
            .otherwise(pl.lit("winter"))
            .alias("season"),

            pl.when(
                pl.col("dispatch_hour").is_between(7, 9)
                | pl.col("dispatch_hour").is_between(16, 18)
            )
            .then(pl.lit("peak"))
            .when(pl.col("dispatch_hour").is_between(10, 21))
            .then(pl.lit("day"))
            .otherwise(pl.lit("night"))
            .alias("period_of_day"),
        ])
        .select([
            "trip_id",
            "season",
            "period_of_day",
        ])
    )


# =========================================================
# Regression table
# =========================================================

def add_iqr3_residual_outlier_flags(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add conservative {IQR_OUTLIER_MULTIPLIER}×IQR residual-outlier flags based on prediction_error_sec.

    This does not remove observations. It only marks trips whose baseline
    prediction error lies outside [Q1 - {IQR_OUTLIER_MULTIPLIER}×IQR, Q3 + {IQR_OUTLIER_MULTIPLIER}×IQR].
    """
    if "prediction_error_sec" not in df.columns:
        raise ValueError(
            "Cannot compute IQR outlier flags because prediction_error_sec is missing."
        )

    q1 = df.select(pl.col("prediction_error_sec").quantile(0.25)).item()
    q3 = df.select(pl.col("prediction_error_sec").quantile(0.75)).item()

    if q1 is None or q3 is None:
        return df.with_columns([
            pl.lit(None).cast(pl.Float64).alias("prediction_error_iqr_q1"),
            pl.lit(None).cast(pl.Float64).alias("prediction_error_iqr_q3"),
            pl.lit(None).cast(pl.Float64).alias("prediction_error_iqr"),
            pl.lit(None).cast(pl.Float64).alias("prediction_error_iqr_lower_bound"),
            pl.lit(None).cast(pl.Float64).alias("prediction_error_iqr_upper_bound"),
            pl.lit(False).alias("extreme_prediction_error_iqr3"),
        ])

    iqr = float(q3) - float(q1)
    lower = float(q1) - IQR_OUTLIER_MULTIPLIER * iqr
    upper = float(q3) + IQR_OUTLIER_MULTIPLIER * iqr

    return df.with_columns([
        pl.lit(float(q1)).alias("prediction_error_iqr_q1"),
        pl.lit(float(q3)).alias("prediction_error_iqr_q3"),
        pl.lit(iqr).alias("prediction_error_iqr"),
        pl.lit(lower).alias("prediction_error_iqr_lower_bound"),
        pl.lit(upper).alias("prediction_error_iqr_upper_bound"),
        (
            (pl.col("prediction_error_sec") < lower)
            | (pl.col("prediction_error_sec") > upper)
        ).alias("extreme_prediction_error_iqr3"),
    ])


def create_regression_table() -> pl.DataFrame:
    """
    Create the final modelling-ready regression table.

    The table retains dispatch_time so extended tests can apply temporal
    training/validation splits without changing this core script.
    """

    check_inputs()

    prediction_errors = (
        pl.read_parquet(PREDICTION_ERRORS_PARQUET)
        .with_columns([
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
        .select([
            "trip_id",
            "dispatch_time",
            pl.col("prediction_error_sec").cast(pl.Float64),
        ])
    )

    explanatory_variables = get_explanatory_variables()
    controls = get_controls()
    temporal_variables = get_temporal_variables()

    regression_table = (
        prediction_errors
        .join(explanatory_variables, on="trip_id", how="inner")
        .join(controls, on="trip_id", how="inner")
        .join(temporal_variables, on="trip_id", how="inner")
        .drop_nulls([
            "prediction_error_sec",
            "distance_km",
            "n_turns",
        ])
        .sort("trip_id")
    )

    if regression_table.is_empty():
        raise RuntimeError(
            "Regression table is empty after joining prediction errors, "
            "route features, controls, and temporal variables."
        )

    regression_table = add_iqr3_residual_outlier_flags(regression_table)

    return regression_table


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    print("Creating regression table...")

    regression_table = create_regression_table()

    REGRESSION_TABLE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    REGRESSION_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)

    regression_table.write_parquet(REGRESSION_TABLE_PARQUET)
    regression_table.write_csv(REGRESSION_TABLE_CSV)

    print(f"Saved: {REGRESSION_TABLE_PARQUET}")
    print(f"Saved: {REGRESSION_TABLE_CSV}")
    print(f"Rows: {regression_table.height}")


if __name__ == "__main__":
    main()