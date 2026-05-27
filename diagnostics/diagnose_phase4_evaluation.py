"""
diagnostics/diagnose_phase4_evaluation.py

Phase 4 diagnostics: calibrated prediction and evaluation.

This script reads the outputs of:
  - p4s10_compute_calibrated_predictions.py
  - p4s11_evaluate_calibration.py

It does not call OSRM and does not modify pipeline outputs. It only writes
diagnostic CSV files to:

    results/diagnostics/

Diagnostic purpose:
  Check whether calibrated prediction coverage, error improvements,
  15-minute-standard classification, calibrated route quality, and naive
  multiplier benchmarks look plausible.
"""

from __future__ import annotations
from typing import Any

from pathlib import Path
import sys

import polars as pl


# =========================================================
# Project import path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from paths import (  # noqa: E402
    RESULTS_DIR,
    PREDICTION_ERRORS_PARQUET,
    CALIBRATED_PREDICTIONS_PARQUET,
    CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET,
    CALIBRATION_EVALUATION_METRICS_CSV,
    CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV,
)

from configs import (  # noqa: E402
    LARGE_ERROR_SEC,
    RESPONSE_STANDARD_SEC,
)


DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"


# =========================================================
# Settings
# =========================================================

FIXED_NAIVE_MULTIPLIERS = [0.70, 0.80, 0.90]

EXTREME_CASES_N = 50


# =========================================================
# Helpers
# =========================================================

def ensure_diagnostics_dir() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)


def pct(count: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round((float(count) / float(total)) * 100, 2)


def write_table(df: pl.DataFrame, filename: str) -> None:
    path = DIAGNOSTICS_DIR / filename
    df.write_csv(path)
    print(f"Saved: {path}")


def as_float_or_none(value: Any) -> float | None:
    """
    Convert numeric diagnostic values to float.

    Returns None for missing values. Raises a clear error for unexpected
    non-numeric values, because metric tables should only contain numeric
    values or None.
    """
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Metric table values must be numeric or None. Got {value!r} "
            f"of type {type(value).__name__}."
        ) from exc


def metric_table(rows: list[tuple[str, Any]]) -> pl.DataFrame:
    """
    Build a metric/value table with Float64 values.
    """
    return pl.DataFrame(
        {
            "metric": [name for name, _ in rows],
            "value": [as_float_or_none(raw_value) for _, raw_value in rows],
        },
        schema={
            "metric": pl.String,
            "value": pl.Float64,
        },
        strict=False,
    )


def check_inputs() -> None:
    required_paths = [
        PREDICTION_ERRORS_PARQUET,
        CALIBRATED_PREDICTIONS_PARQUET,
        CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET,
        CALIBRATION_EVALUATION_METRICS_CSV,
        CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required Phase 4 diagnostic input file(s):\n"
            + "\n".join(f"  - {path}" for path in missing)
        )


def describe_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Return count, mean, median, selected quantiles, min, and max for one
    numeric column.

    Values are explicitly stored as Float64 to avoid Polars failing when
    integer counts and floating-point statistics appear in the same column.
    """
    if column not in df.columns:
        return pl.DataFrame(
            {
                "statistic": ["missing_column"],
                column: [None],
            },
            schema={
                "statistic": pl.String,
                column: pl.Float64,
            },
            strict=False,
        )

    s = df.get_column(column).drop_nulls()

    if s.is_empty():
        rows = [
            ("count", 0.0),
            ("mean", None),
            ("median", None),
            ("p05", None),
            ("p25", None),
            ("p75", None),
            ("p95", None),
            ("min", None),
            ("max", None),
        ]
    else:
        rows = [
            ("count", float(len(s))),
            ("mean", s.mean()),
            ("median", s.median()),
            ("p05", s.quantile(0.05)),
            ("p25", s.quantile(0.25)),
            ("p75", s.quantile(0.75)),
            ("p95", s.quantile(0.95)),
            ("min", s.min()),
            ("max", s.max()),
        ]

    return pl.DataFrame(
        {
            "statistic": [name for name, _ in rows],
            column: [None if value is None else float(value) for _, value in rows],
        },
        schema={
            "statistic": pl.String,
            column: pl.Float64,
        },
        strict=False,
    )


def rmse_expr(error_col: str) -> pl.Expr:
    return (pl.col(error_col) ** 2).mean().sqrt()


# =========================================================
# Core diagnostics
# =========================================================

def prediction_coverage(
    baseline_errors: pl.DataFrame,
    calibrated_predictions: pl.DataFrame,
    trip_errors: pl.DataFrame,
) -> pl.DataFrame:
    baseline_trips = baseline_errors.select(pl.col("trip_id").n_unique()).item()
    calibrated_trips = calibrated_predictions.select(pl.col("trip_id").n_unique()).item()
    evaluated_trips = trip_errors.select(pl.col("trip_id").n_unique()).item()

    return pl.DataFrame({
        "metric": [
            "baseline_prediction_error_trips",
            "calibrated_prediction_trips",
            "phase4_evaluated_trips",
            "lost_baseline_to_calibrated",
            "lost_calibrated_to_evaluation",
        ],
        "value": [
            baseline_trips,
            calibrated_trips,
            evaluated_trips,
            baseline_trips - calibrated_trips,
            calibrated_trips - evaluated_trips,
        ],
        "percentage_of_baseline_trips": [
            100.0,
            pct(calibrated_trips, baseline_trips),
            pct(evaluated_trips, baseline_trips),
            pct(baseline_trips - calibrated_trips, baseline_trips),
            pct(calibrated_trips - evaluated_trips, baseline_trips),
        ],
    })


def error_distribution_summary(trip_errors: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "prediction_error_sec",
        "calibrated_prediction_error_sec",
        "baseline_absolute_error_sec",
        "calibrated_absolute_error_sec",
    ]

    summaries = []

    for column in columns:
        if column in trip_errors.columns:
            summary = describe_column(trip_errors, column).rename({column: "value"})
            summary = summary.with_columns(pl.lit(column).alias("variable"))
            summaries.append(summary.select(["variable", "statistic", "value"]))

    if not summaries:
        return pl.DataFrame(
            {
                "variable": ["none"],
                "statistic": ["missing_error_columns"],
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


def error_improvement_summary(trip_errors: pl.DataFrame) -> pl.DataFrame:
    required = [
        "baseline_absolute_error_sec",
        "calibrated_absolute_error_sec",
        "prediction_error_sec",
        "calibrated_prediction_error_sec",
    ]

    missing = [col for col in required if col not in trip_errors.columns]

    if missing:
        raise ValueError(f"Trip errors missing required columns: {missing}")

    df = trip_errors.with_columns([
        (
            pl.col("calibrated_absolute_error_sec")
            - pl.col("baseline_absolute_error_sec")
        ).alias("absolute_error_change_sec"),

        (
            pl.col("baseline_absolute_error_sec")
            - pl.col("calibrated_absolute_error_sec")
        ).alias("absolute_error_improvement_sec"),

        (
            pl.col("calibrated_absolute_error_sec")
            < pl.col("baseline_absolute_error_sec")
        ).alias("calibration_improved_trip"),

        (
            pl.col("calibrated_absolute_error_sec")
            > pl.col("baseline_absolute_error_sec")
        ).alias("calibration_worsened_trip"),
    ])

    baseline_mae = df.select(pl.col("baseline_absolute_error_sec").mean()).item()
    calibrated_mae = df.select(pl.col("calibrated_absolute_error_sec").mean()).item()

    baseline_rmse = df.select(rmse_expr("prediction_error_sec")).item()
    calibrated_rmse = df.select(rmse_expr("calibrated_prediction_error_sec")).item()

    n = df.height

    improved_n = df.filter(pl.col("calibration_improved_trip")).height
    worsened_n = df.filter(pl.col("calibration_worsened_trip")).height
    unchanged_n = df.filter(
        pl.col("calibrated_absolute_error_sec")
        == pl.col("baseline_absolute_error_sec")
    ).height

    return metric_table([
        ("n_trips", n),
        ("baseline_mae_sec", baseline_mae),
        ("calibrated_mae_sec", calibrated_mae),
        ("mae_change_sec", calibrated_mae - baseline_mae),
        (
            "mae_change_percent",
            (
                ((calibrated_mae - baseline_mae) / baseline_mae) * 100
                if baseline_mae and baseline_mae > 0
                else None
            ),
        ),
        ("baseline_rmse_sec", baseline_rmse),
        ("calibrated_rmse_sec", calibrated_rmse),
        ("rmse_change_sec", calibrated_rmse - baseline_rmse),
        (
            "rmse_change_percent",
            (
                ((calibrated_rmse - baseline_rmse) / baseline_rmse) * 100
                if baseline_rmse and baseline_rmse > 0
                else None
            ),
        ),
        ("trips_improved", improved_n),
        ("trips_worsened", worsened_n),
        ("trips_unchanged", unchanged_n),
        ("share_trips_improved", pct(improved_n, n)),
        ("share_trips_worsened", pct(worsened_n, n)),
    ])


def standard_misclassification_summary(trip_errors: pl.DataFrame) -> pl.DataFrame:
    required = [
        "realised_travel_time_sec",
        "baseline_predicted_time_sec",
        "calibrated_osrm_time_sec",
    ]

    missing = [col for col in required if col not in trip_errors.columns]

    if missing:
        raise ValueError(f"Trip errors missing required columns: {missing}")

    df = trip_errors.with_columns([
        (pl.col("realised_travel_time_sec") <= RESPONSE_STANDARD_SEC)
        .alias("realised_within_standard"),

        (pl.col("baseline_predicted_time_sec") <= RESPONSE_STANDARD_SEC)
        .alias("baseline_within_standard"),

        (pl.col("calibrated_osrm_time_sec") <= RESPONSE_STANDARD_SEC)
        .alias("calibrated_within_standard"),
    ]).with_columns([
        (
            pl.col("realised_within_standard")
            != pl.col("baseline_within_standard")
        ).alias("baseline_misclassified"),

        (
            pl.col("realised_within_standard")
            != pl.col("calibrated_within_standard")
        ).alias("calibrated_misclassified"),
    ])

    n = df.height

    baseline_misclassified = df.filter(pl.col("baseline_misclassified")).height
    calibrated_misclassified = df.filter(pl.col("calibrated_misclassified")).height

    improved = df.filter(
        pl.col("baseline_misclassified")
        & ~pl.col("calibrated_misclassified")
    ).height

    worsened = df.filter(
        ~pl.col("baseline_misclassified")
        & pl.col("calibrated_misclassified")
    ).height

    return metric_table([
        ("n_trips", n),
        ("baseline_misclassified_trips", baseline_misclassified),
        ("calibrated_misclassified_trips", calibrated_misclassified),
        ("misclassification_change_trips", calibrated_misclassified - baseline_misclassified),
        ("baseline_misclassification_share", pct(baseline_misclassified, n)),
        ("calibrated_misclassification_share", pct(calibrated_misclassified, n)),
        (
            "misclassification_change_percentage_points",
            pct(calibrated_misclassified, n) - pct(baseline_misclassified, n),
        ),
        ("trips_fixed_by_calibration", improved),
        ("trips_broken_by_calibration", worsened),
    ])


def calibrated_route_quality(
    calibrated_predictions: pl.DataFrame,
    route_diagnostics: pl.DataFrame,
) -> pl.DataFrame:
    columns = [
        "calibrated_distance_km",
        "calibrated_match_confidence",
        "n_points_original",
        "n_points_submitted",
        "trace_duration_sec",
        "trace_max_gap_sec",
        "request_url_length",
    ]

    summaries = []

    for column in columns:
        if column in calibrated_predictions.columns:
            summary = describe_column(calibrated_predictions, column).rename({column: "value"})
            summary = summary.with_columns(pl.lit(column).alias("variable"))
            summaries.append(summary.select(["variable", "statistic", "value"]))

    if route_diagnostics is not None and not route_diagnostics.is_empty():
        for column in route_diagnostics.columns:
            if column == "n_trips":
                continue

            if route_diagnostics[column].dtype.is_numeric():
                value = route_diagnostics.select(pl.col(column).first()).item()
                summaries.append(
                    pl.DataFrame(
                        {
                            "variable": [f"route_diagnostics.{column}"],
                            "statistic": ["reported_value"],
                            "value": [None if value is None else float(value)],
                        },
                        schema={
                            "variable": pl.String,
                            "statistic": pl.String,
                            "value": pl.Float64,
                        },
                        strict=False,
                    )
                )

    if not summaries:
        return pl.DataFrame(
            {
                "variable": ["none"],
                "statistic": ["missing_route_quality_columns"],
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


def extreme_error_cases(
    trip_errors: pl.DataFrame,
    n: int = EXTREME_CASES_N,
) -> pl.DataFrame:
    required = [
        "trip_id",
        "realised_travel_time_sec",
        "baseline_predicted_time_sec",
        "calibrated_osrm_time_sec",
        "baseline_absolute_error_sec",
        "calibrated_absolute_error_sec",
    ]

    missing = [col for col in required if col not in trip_errors.columns]

    if missing:
        raise ValueError(f"Trip errors missing required columns: {missing}")

    return (
        trip_errors
        .with_columns([
            (
                pl.col("calibrated_absolute_error_sec")
                - pl.col("baseline_absolute_error_sec")
            ).alias("calibration_worsening_sec"),

            (
                pl.col("baseline_absolute_error_sec")
                - pl.col("calibrated_absolute_error_sec")
            ).alias("calibration_improvement_sec"),
        ])
        .select([
            "trip_id",
            "realised_travel_time_sec",
            "baseline_predicted_time_sec",
            "calibrated_osrm_time_sec",
            "baseline_absolute_error_sec",
            "calibrated_absolute_error_sec",
            "calibration_worsening_sec",
            "calibration_improvement_sec",
            *[
                col
                for col in [
                    "calibrated_match_confidence",
                    "calibrated_distance_km",
                    "trace_max_gap_sec",
                    "n_points_submitted",
                ]
                if col in trip_errors.columns
            ],
        ])
        .sort("calibration_worsening_sec", descending=True)
        .head(n)
    )


# =========================================================
# Fixed naive multiplier diagnostics
# =========================================================

def evaluate_multiplier(
    trip_errors: pl.DataFrame,
    multiplier: float,
) -> dict:
    """
    Evaluate one fixed naive multiplier.

    These are diagnostic benchmark calculations only. They mirror the fixed
    0.70, 0.80, and 0.90 multiplier benchmarks used in p4s11 and do not select
    an optimal multiplier on the evaluation sample.
    """
    df = (
        trip_errors
        .with_columns([
            (
                pl.col("baseline_predicted_time_sec") * multiplier
            ).alias("multiplier_predicted_time_sec"),
        ])
        .with_columns([
            (
                pl.col("realised_travel_time_sec")
                - pl.col("multiplier_predicted_time_sec")
            ).alias("multiplier_prediction_error_sec"),
        ])
        .with_columns([
            pl.col("multiplier_prediction_error_sec")
            .abs()
            .alias("multiplier_absolute_error_sec"),

            (
                (pl.col("realised_travel_time_sec") <= RESPONSE_STANDARD_SEC)
                != (pl.col("multiplier_predicted_time_sec") <= RESPONSE_STANDARD_SEC)
            ).alias("multiplier_standard_misclassified"),
        ])
    )

    n = df.height

    return {
        "model": f"naive_multiplier_{str(multiplier).replace('.', 'p')}",
        "multiplier": multiplier,
        "n_trips": n,
        "mean_signed_error_sec": df.select(
            pl.col("multiplier_prediction_error_sec").mean()
        ).item(),
        "median_signed_error_sec": df.select(
            pl.col("multiplier_prediction_error_sec").median()
        ).item(),
        "mae_sec": df.select(
            pl.col("multiplier_absolute_error_sec").mean()
        ).item(),
        "median_ae_sec": df.select(
            pl.col("multiplier_absolute_error_sec").median()
        ).item(),
        "rmse_sec": df.select(
            rmse_expr("multiplier_prediction_error_sec")
        ).item(),
        "share_error_gt_5min": df.select(
            (pl.col("multiplier_absolute_error_sec") > LARGE_ERROR_SEC)
            .mean()
        ).item(),
        "share_15min_misclassified": df.select(
            pl.col("multiplier_standard_misclassified").mean()
        ).item(),
    }


def fixed_naive_multiplier_benchmarks(trip_errors: pl.DataFrame) -> pl.DataFrame:
    rows = [
        evaluate_multiplier(trip_errors, multiplier)
        for multiplier in FIXED_NAIVE_MULTIPLIERS
    ]

    return pl.DataFrame(rows).sort("multiplier")


def compare_calibrated_to_fixed_multipliers(
    trip_errors: pl.DataFrame,
    fixed_multiplier_metrics: pl.DataFrame,
) -> pl.DataFrame:
    calibrated = {
        "model": "calibrated_osrm",
        "multiplier": None,
        "n_trips": trip_errors.height,
        "mean_signed_error_sec": trip_errors.select(
            pl.col("calibrated_prediction_error_sec").mean()
        ).item(),
        "median_signed_error_sec": trip_errors.select(
            pl.col("calibrated_prediction_error_sec").median()
        ).item(),
        "mae_sec": trip_errors.select(
            pl.col("calibrated_absolute_error_sec").mean()
        ).item(),
        "median_ae_sec": trip_errors.select(
            pl.col("calibrated_absolute_error_sec").median()
        ).item(),
        "rmse_sec": trip_errors.select(
            rmse_expr("calibrated_prediction_error_sec")
        ).item(),
        "share_error_gt_5min": trip_errors.select(
            (pl.col("calibrated_absolute_error_sec") > LARGE_ERROR_SEC).mean()
        ).item(),
        "share_15min_misclassified": trip_errors.select(
            pl.col("calibrated_standard_misclassified").mean()
            if "calibrated_standard_misclassified" in trip_errors.columns
            else (
                (pl.col("realised_travel_time_sec") <= RESPONSE_STANDARD_SEC)
                != (pl.col("calibrated_osrm_time_sec") <= RESPONSE_STANDARD_SEC)
            ).mean()
        ).item(),
    }

    out = pl.concat(
        [
            pl.DataFrame([calibrated]),
            fixed_multiplier_metrics,
        ],
        how="diagonal_relaxed",
    )

    baseline_mae = calibrated["mae_sec"]
    baseline_rmse = calibrated["rmse_sec"]

    return (
        out
        .with_columns([
            (
                pl.col("mae_sec") - pl.lit(baseline_mae)
            ).alias("mae_difference_vs_calibrated_sec"),
            (
                pl.col("rmse_sec") - pl.lit(baseline_rmse)
            ).alias("rmse_difference_vs_calibrated_sec"),
        ])
    )


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    ensure_diagnostics_dir()
    check_inputs()

    print("Loading Phase 4 outputs...")

    baseline_errors = pl.read_parquet(PREDICTION_ERRORS_PARQUET)
    calibrated_predictions = pl.read_parquet(CALIBRATED_PREDICTIONS_PARQUET)
    trip_errors = pl.read_parquet(CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET)
    metrics = pl.read_csv(CALIBRATION_EVALUATION_METRICS_CSV)
    route_diagnostics = pl.read_csv(CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV)

    print("Writing Phase 4 diagnostic tables...")

    write_table(
        prediction_coverage(
            baseline_errors=baseline_errors,
            calibrated_predictions=calibrated_predictions,
            trip_errors=trip_errors,
        ),
        "phase4_prediction_coverage.csv",
    )

    write_table(
        error_distribution_summary(trip_errors),
        "phase4_error_distribution_summary.csv",
    )

    write_table(
        error_improvement_summary(trip_errors),
        "phase4_error_improvement_summary.csv",
    )

    write_table(
        standard_misclassification_summary(trip_errors),
        "phase4_standard_misclassification_summary.csv",
    )

    write_table(
        calibrated_route_quality(
            calibrated_predictions=calibrated_predictions,
            route_diagnostics=route_diagnostics,
        ),
        "phase4_calibrated_route_quality.csv",
    )

    write_table(
        extreme_error_cases(trip_errors),
        "phase4_extreme_error_cases.csv",
    )

    fixed_multiplier_metrics = fixed_naive_multiplier_benchmarks(trip_errors)

    write_table(
        fixed_multiplier_metrics,
        "phase4_naive_multiplier_fixed_benchmarks.csv",
    )

    write_table(
        compare_calibrated_to_fixed_multipliers(
            trip_errors=trip_errors,
            fixed_multiplier_metrics=fixed_multiplier_metrics,
        ),
        "phase4_calibrated_vs_fixed_multipliers.csv",
    )

    print("\nPhase 4 diagnostic summary")
    print("--------------------------")
    print("Core calibration metrics:")
    print(metrics)

    print("\nFixed naive multiplier benchmarks:")
    print(fixed_multiplier_metrics)

    print("\nCalibrated OSRM vs fixed naive multiplier benchmarks:")
    print(compare_calibrated_to_fixed_multipliers(
        trip_errors=trip_errors,
        fixed_multiplier_metrics=fixed_multiplier_metrics,
    ))


if __name__ == "__main__":
    main()