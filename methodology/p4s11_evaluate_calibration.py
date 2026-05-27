"""
p4s11_evaluate_calibration.py

Phase 4, step 11.

Purpose:
    Evaluate baseline OSRM, calibrated OSRM, and fixed naive multiplier
    benchmarks for the active run.

Input:
    runs/RunXXX/outputs/calibrated_predictions.parquet

Outputs:
    runs/RunXXX/results/calibration_evaluation_trip_errors.parquet
    runs/RunXXX/results/calibration_evaluation_metrics.csv
    runs/RunXXX/results/calibration_evaluation_route_diagnostics.csv

Design:
    This core file evaluates the active run only. Off-sample validation,
    transferability, and sensitivity analyses belong in extended_tests/.

    Naive multipliers are fixed benchmarks, not re-estimated corrections. This
    avoids selecting a multiplier on the same validation or transfer sample
    when this evaluator is reused by extended tests.
"""

from __future__ import annotations

import polars as pl

from paths import (
    CALIBRATED_PREDICTIONS_PARQUET,
    CALIBRATION_EVALUATION_METRICS_CSV,
    CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET,
    CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV,
)

from configs import (
    LARGE_ERROR_SEC,
    RESPONSE_STANDARD_SEC,
)


NAIVE_MULTIPLIERS = [0.70, 0.80, 0.90]


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    """
    Validate required Phase 4 evaluation input.
    """
    if not CALIBRATED_PREDICTIONS_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing calibrated predictions: {CALIBRATED_PREDICTIONS_PARQUET}"
        )


# =========================================================
# Model-name helpers
# =========================================================

def multiplier_suffix(multiplier: float) -> str:
    """
    Return a stable model-name suffix for a multiplier.

    Example:
        0.70 -> 0p70
    """
    return f"{multiplier:.2f}".replace(".", "p")


def multiplier_model_name(multiplier: float) -> str:
    return f"naive_multiplier_{multiplier_suffix(multiplier)}"


# =========================================================
# Trip-level error table
# =========================================================

def build_trip_error_table(predictions: pl.DataFrame) -> pl.DataFrame:
    """
    Create trip-level baseline and calibrated prediction errors.
    """

    required_cols = [
        "trip_id",
        "realised_travel_time_sec",
        "baseline_predicted_time_sec",
        "calibrated_osrm_time_sec",
        "prediction_error_sec",
        "calibrated_prediction_error_sec",
    ]

    missing = [col for col in required_cols if col not in predictions.columns]

    if missing:
        raise ValueError(
            f"Calibrated predictions are missing required columns: {missing}"
        )

    trip_errors = (
        predictions
        .with_columns([
            pl.col("prediction_error_sec")
            .abs()
            .alias("baseline_absolute_error_sec"),

            pl.col("calibrated_prediction_error_sec")
            .abs()
            .alias("calibrated_absolute_error_sec"),

            (
                (pl.col("realised_travel_time_sec") <= RESPONSE_STANDARD_SEC)
                != (pl.col("baseline_predicted_time_sec") <= RESPONSE_STANDARD_SEC)
            ).alias("baseline_standard_misclassified"),

            (
                (pl.col("realised_travel_time_sec") <= RESPONSE_STANDARD_SEC)
                != (pl.col("calibrated_osrm_time_sec") <= RESPONSE_STANDARD_SEC)
            ).alias("calibrated_standard_misclassified"),
        ])
        .sort("trip_id")
    )

    if trip_errors.is_empty():
        raise RuntimeError("Trip-error table is empty.")

    return trip_errors


# =========================================================
# Naive multiplier benchmarks
# =========================================================

def add_multiplier_errors(trip_errors: pl.DataFrame) -> pl.DataFrame:
    """
    Add fixed naive multiplier predictions and errors.

    These benchmarks rescale baseline OSRM predictions by fixed constants.
    They are not fitted on the evaluation sample.
    """
    out = trip_errors

    for multiplier in NAIVE_MULTIPLIERS:
        suffix = multiplier_suffix(multiplier)

        predicted_col = f"naive_multiplier_{suffix}_predicted_time_sec"
        error_col = f"naive_multiplier_{suffix}_prediction_error_sec"
        absolute_error_col = f"naive_multiplier_{suffix}_absolute_error_sec"
        misclassified_col = f"naive_multiplier_{suffix}_standard_misclassified"

        out = (
            out
            .with_columns([
                (
                    pl.col("baseline_predicted_time_sec") * multiplier
                ).alias(predicted_col),
            ])
            .with_columns([
                (
                    pl.col("realised_travel_time_sec")
                    - pl.col(predicted_col)
                ).alias(error_col),
            ])
            .with_columns([
                pl.col(error_col)
                .abs()
                .alias(absolute_error_col),

                (
                    (pl.col("realised_travel_time_sec") <= RESPONSE_STANDARD_SEC)
                    != (pl.col(predicted_col) <= RESPONSE_STANDARD_SEC)
                ).alias(misclassified_col),
            ])
        )

    return out


# =========================================================
# Metric computation
# =========================================================

def compute_metrics_for_model(
    trip_errors: pl.DataFrame,
    model_name: str,
    predicted_col: str,
    error_col: str,
    absolute_error_col: str,
    standard_misclassified_col: str,
    multiplier: float | None = None,
) -> pl.DataFrame:
    """
    Compute performance metrics for one model.
    """

    return (
        trip_errors
        .select([
            pl.len().alias("n_trips"),
            pl.col(error_col).mean().alias("mean_signed_error_sec"),
            pl.col(error_col).median().alias("median_signed_error_sec"),
            pl.col(absolute_error_col).mean().alias("mae_sec"),
            pl.col(absolute_error_col).median().alias("median_ae_sec"),
            (pl.col(error_col) ** 2).mean().sqrt().alias("rmse_sec"),
            pl.col(absolute_error_col)
            .quantile(0.95, interpolation="nearest")
            .alias("p95_ae_sec"),
            (pl.col(absolute_error_col) > LARGE_ERROR_SEC)
            .mean()
            .alias("share_error_gt_5min"),
            pl.col(standard_misclassified_col)
            .mean()
            .alias("share_15min_misclassified"),
            pl.col(predicted_col).mean().alias("mean_predicted_time_sec"),
            pl.col("realised_travel_time_sec").mean().alias("mean_realised_time_sec"),
        ])
        .with_columns([
            pl.lit(model_name).alias("model"),
            pl.lit(multiplier, dtype=pl.Float64).alias("multiplier"),
        ])
        .select([
            "model",
            "n_trips",
            "mean_signed_error_sec",
            "median_signed_error_sec",
            "mae_sec",
            "median_ae_sec",
            "rmse_sec",
            "p95_ae_sec",
            "share_error_gt_5min",
            "share_15min_misclassified",
            "mean_predicted_time_sec",
            "mean_realised_time_sec",
            "multiplier",
        ])
    )


def compute_all_performance_metrics(trip_errors: pl.DataFrame) -> pl.DataFrame:
    """
    Compute metrics for baseline, calibrated, and fixed multiplier benchmarks.
    """

    model_frames = [
        compute_metrics_for_model(
            trip_errors=trip_errors,
            model_name="baseline_osrm",
            predicted_col="baseline_predicted_time_sec",
            error_col="prediction_error_sec",
            absolute_error_col="baseline_absolute_error_sec",
            standard_misclassified_col="baseline_standard_misclassified",
        ),
        compute_metrics_for_model(
            trip_errors=trip_errors,
            model_name="calibrated_osrm",
            predicted_col="calibrated_osrm_time_sec",
            error_col="calibrated_prediction_error_sec",
            absolute_error_col="calibrated_absolute_error_sec",
            standard_misclassified_col="calibrated_standard_misclassified",
        ),
    ]

    for multiplier in NAIVE_MULTIPLIERS:
        suffix = multiplier_suffix(multiplier)
        model_frames.append(
            compute_metrics_for_model(
                trip_errors=trip_errors,
                model_name=multiplier_model_name(multiplier),
                predicted_col=f"naive_multiplier_{suffix}_predicted_time_sec",
                error_col=f"naive_multiplier_{suffix}_prediction_error_sec",
                absolute_error_col=f"naive_multiplier_{suffix}_absolute_error_sec",
                standard_misclassified_col=(
                    f"naive_multiplier_{suffix}_standard_misclassified"
                ),
                multiplier=multiplier,
            )
        )

    return pl.concat(model_frames, how="vertical")


def build_model_comparison(metrics: pl.DataFrame) -> pl.DataFrame:
    """
    Add changes relative to baseline for calibrated and multiplier models.
    """

    baseline = (
        metrics
        .filter(pl.col("model") == "baseline_osrm")
        .select([
            pl.col("mae_sec").first().alias("baseline_mae_sec"),
            pl.col("rmse_sec").first().alias("baseline_rmse_sec"),
            pl.col("share_error_gt_5min").first().alias("baseline_share_error_gt_5min"),
            pl.col("share_15min_misclassified")
            .first()
            .alias("baseline_share_15min_misclassified"),
        ])
    )

    if baseline.height != 1:
        raise RuntimeError("Could not identify exactly one baseline_osrm metric row.")

    baseline_values = baseline.row(0, named=True)

    return (
        metrics
        .with_columns([
            pl.lit(baseline_values["baseline_mae_sec"]).alias("baseline_mae_sec"),
            pl.lit(baseline_values["baseline_rmse_sec"]).alias("baseline_rmse_sec"),
            pl.lit(baseline_values["baseline_share_error_gt_5min"])
            .alias("baseline_share_error_gt_5min"),
            pl.lit(baseline_values["baseline_share_15min_misclassified"])
            .alias("baseline_share_15min_misclassified"),
        ])
        .with_columns([
            (pl.col("mae_sec") - pl.col("baseline_mae_sec")).alias("mae_change_sec"),
            (
                (pl.col("mae_sec") - pl.col("baseline_mae_sec"))
                / pl.col("baseline_mae_sec")
                * 100
            ).alias("mae_change_percent"),

            (pl.col("rmse_sec") - pl.col("baseline_rmse_sec")).alias("rmse_change_sec"),
            (
                (pl.col("rmse_sec") - pl.col("baseline_rmse_sec"))
                / pl.col("baseline_rmse_sec")
                * 100
            ).alias("rmse_change_percent"),

            (
                pl.col("share_error_gt_5min")
                - pl.col("baseline_share_error_gt_5min")
            ).alias("large_error_share_change"),

            (
                pl.col("share_15min_misclassified")
                - pl.col("baseline_share_15min_misclassified")
            ).alias("standard_misclassification_change"),
        ])
        .sort("model")
    )


# =========================================================
# Route diagnostics
# =========================================================

def compute_route_diagnostics(trip_errors: pl.DataFrame) -> pl.DataFrame:
    """
    Summarise calibrated route/matching diagnostics when available.
    """

    agg_exprs = [
        pl.len().alias("n_trips"),
    ]

    if "calibrated_match_confidence" in trip_errors.columns:
        agg_exprs.extend([
            pl.col("calibrated_match_confidence")
            .mean()
            .alias("mean_calibrated_match_confidence"),
            pl.col("calibrated_match_confidence")
            .median()
            .alias("median_calibrated_match_confidence"),
            pl.col("calibrated_match_confidence")
            .min()
            .alias("min_calibrated_match_confidence"),
        ])

    if "calibrated_distance_km" in trip_errors.columns:
        agg_exprs.extend([
            pl.col("calibrated_distance_km")
            .mean()
            .alias("mean_calibrated_distance_km"),
            pl.col("calibrated_distance_km")
            .median()
            .alias("median_calibrated_distance_km"),
            pl.col("calibrated_distance_km")
            .max()
            .alias("max_calibrated_distance_km"),
        ])

    return trip_errors.select(agg_exprs)


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    check_inputs()

    print("Loading calibrated predictions...")
    predictions = pl.read_parquet(CALIBRATED_PREDICTIONS_PARQUET)

    print("Building trip-level error table...")
    trip_errors = build_trip_error_table(predictions)
    trip_errors = add_multiplier_errors(trip_errors)

    print("Computing performance metrics...")
    metrics = compute_all_performance_metrics(trip_errors)
    model_comparison = build_model_comparison(metrics)

    print("Computing route diagnostics...")
    route_diagnostics = compute_route_diagnostics(trip_errors)

    CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CALIBRATION_EVALUATION_METRICS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trip_errors.write_parquet(CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET)
    model_comparison.write_csv(CALIBRATION_EVALUATION_METRICS_CSV)
    route_diagnostics.write_csv(CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV)

    print(f"Saved: {CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET}")
    print(f"Saved: {CALIBRATION_EVALUATION_METRICS_CSV}")
    print(f"Saved: {CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV}")

    print("\nModel comparison:")
    print(model_comparison)


if __name__ == "__main__":
    main()
