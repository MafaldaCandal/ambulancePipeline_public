"""
Phase 2 diagnostics: route matching and prediction.

Reads Phase 2 outputs and writes diagnostic CSV files to:
    runs/RunXXX/results/diagnostics/

This script does not call OSRM and does not change pipeline outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

import polars as pl
from typing import Any


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import (  # noqa: E402
    CLEAN_TRAJECTORIES_PARQUET,
    RESULTS_DIR,
    ROUTES_PARQUET,
    ROUTE_FEATURES_PARQUET,
    ROUTE_REJECTION_SUMMARY_PARQUET,
    PREDICTION_ERRORS_PARQUET,
)

from configs import (  # noqa: E402
    LARGE_ERROR_SEC,
    MATCH_CONFIDENCE_THRESHOLD,
    ROAD_CLASSES,
)


DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def pct(count: int | float, total: int | float) -> float:
    if total == 0:
        return 0.0
    return round(float(count) / float(total) * 100.0, 2)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def write_table(df: pl.DataFrame, filename: str) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAGNOSTICS_DIR / filename
    df.write_csv(path)
    print(f"Saved: {path}")


def metric_table(rows: list[tuple[str, object]]) -> pl.DataFrame:
    """
    Build a simple metric/value table.

    All values are stored as Float64 to avoid Polars inferring Int64 from a
    count and then failing when the next value is decimal.
    """

    return pl.DataFrame(
        {
            "metric": [name for name, _ in rows],
            "value": [as_float(value) for _, value in rows],
        },
        schema={
            "metric": pl.String,
            "value": pl.Float64,
        },
        strict=False,
    )


def numeric_summary(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Return n, mean, median, min, selected quantiles, and max for one column.
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
            ("n_non_null", 0),
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
            ("n_non_null", len(s)),
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
            "value": [as_float(value) for _, value in rows],
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

def route_matching_summary(
    clean: pl.DataFrame,
    routes: pl.DataFrame,
    rejections: pl.DataFrame,
) -> pl.DataFrame:
    clean_trips = int(clean.select(pl.col("trip_id").n_unique()).item())
    matched_routes = int(routes.height)
    rejected_routes = int(rejections.height)

    return pl.DataFrame(
        {
            "metric": [
                "clean_trips_submitted_to_matching",
                "successfully_matched_routes",
                "rejected_routes",
            ],
            "value": [
                float(clean_trips),
                float(matched_routes),
                float(rejected_routes),
            ],
            "percentage_of_clean_trips": [
                100.0,
                pct(matched_routes, clean_trips),
                pct(rejected_routes, clean_trips),
            ],
        },
        schema={
            "metric": pl.String,
            "value": pl.Float64,
            "percentage_of_clean_trips": pl.Float64,
        },
    )


def route_rejection_reasons(
    rejections: pl.DataFrame,
    n_submitted: int,
) -> pl.DataFrame:
    """
    Summarise rejected OSRM Match traces.

    Newer route_rejection_summary files include human-readable columns. Older
    files only include rejection_reason, so this function falls back safely.
    """
    if rejections.height == 0 or "rejection_reason" not in rejections.columns:
        return pl.DataFrame(
            {
                "rejection_category": ["none"],
                "rejection_reason": ["none"],
                "rejection_summary": ["No rejected routes."],
                "count": [0],
                "percentage_of_submitted": [0.0],
                "recommended_check": ["No action needed."],
            },
            schema={
                "rejection_category": pl.String,
                "rejection_reason": pl.String,
                "rejection_summary": pl.String,
                "count": pl.Int64,
                "percentage_of_submitted": pl.Float64,
                "recommended_check": pl.String,
            },
        )

    group_cols = ["rejection_reason"]

    if "rejection_category" in rejections.columns:
        group_cols.insert(0, "rejection_category")

    if "rejection_summary" in rejections.columns:
        group_cols.append("rejection_summary")

    if "recommended_check" in rejections.columns:
        group_cols.append("recommended_check")

    return (
        rejections
        .group_by(group_cols)
        .agg([
            pl.len().alias("count"),
            pl.col("match_confidence").mean().alias("mean_match_confidence")
            if "match_confidence" in rejections.columns
            else pl.lit(None).cast(pl.Float64).alias("mean_match_confidence"),
            pl.col("trace_max_gap_sec").mean().alias("mean_trace_max_gap_sec")
            if "trace_max_gap_sec" in rejections.columns
            else pl.lit(None).cast(pl.Float64).alias("mean_trace_max_gap_sec"),
            pl.col("request_url_length").mean().alias("mean_request_url_length")
            if "request_url_length" in rejections.columns
            else pl.lit(None).cast(pl.Float64).alias("mean_request_url_length"),
        ])
        .with_columns(
            (pl.col("count") / n_submitted * 100)
            .round(2)
            .alias("percentage_of_submitted")
        )
        .sort("count", descending=True)
    )


def route_rejection_categories(
    rejections: pl.DataFrame,
    n_submitted: int,
) -> pl.DataFrame:
    """
    Higher-level rejection summary, grouped only by human-readable category.
    """
    if rejections.height == 0:
        return pl.DataFrame({
            "rejection_category": ["none"],
            "rejection_summary": ["No rejected routes."],
            "count": [0],
            "percentage_of_submitted": [0.0],
        })

    if "rejection_category" not in rejections.columns:
        return (
            rejections
            .group_by("rejection_reason")
            .agg(pl.len().alias("count"))
            .rename({"rejection_reason": "rejection_category"})
            .with_columns([
                pl.col("rejection_category").alias("rejection_summary"),
                (pl.col("count") / n_submitted * 100)
                .round(2)
                .alias("percentage_of_submitted"),
            ])
            .select([
                "rejection_category",
                "rejection_summary",
                "count",
                "percentage_of_submitted",
            ])
            .sort("count", descending=True)
        )

    if "rejection_summary" in rejections.columns:
        return (
            rejections
            .group_by(["rejection_category", "rejection_summary"])
            .agg(pl.len().alias("count"))
            .with_columns(
                (pl.col("count") / n_submitted * 100)
                .round(2)
                .alias("percentage_of_submitted")
            )
            .sort("count", descending=True)
        )

    return (
        rejections
        .group_by("rejection_category")
        .agg(pl.len().alias("count"))
        .with_columns([
            pl.col("rejection_category").alias("rejection_summary"),
            (pl.col("count") / n_submitted * 100)
            .round(2)
            .alias("percentage_of_submitted"),
        ])
        .select([
            "rejection_category",
            "rejection_summary",
            "count",
            "percentage_of_submitted",
        ])
        .sort("count", descending=True)
    )


def match_confidence_summary(
    routes: pl.DataFrame,
    rejections: pl.DataFrame,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []

    if "match_confidence" in routes.columns:
        frames.append(
            routes
            .select("trip_id", "match_confidence")
            .with_columns(pl.lit("successful_routes").alias("source"))
        )

    if "match_confidence" in rejections.columns:
        frames.append(
            rejections
            .filter(pl.col("match_confidence").is_not_null())
            .select("trip_id", "match_confidence")
            .with_columns(pl.lit("rejected_routes_with_confidence").alias("source"))
        )

    if not frames:
        return pl.DataFrame(
            {
                "source": ["missing_match_confidence"],
                "statistic": ["missing_column"],
                "value": [None],
            },
            schema={
                "source": pl.String,
                "statistic": pl.String,
                "value": pl.Float64,
            },
            strict=False,
        )

    combined = pl.concat(frames, how="vertical")

    output_frames: list[pl.DataFrame] = []

    for source in sorted(combined.get_column("source").unique().to_list()):
        source_df = combined.filter(pl.col("source") == source)
        output_frames.append(
            numeric_summary(source_df, "match_confidence")
            .with_columns(pl.lit(source).alias("source"))
            .select(["source", "statistic", "value"])
        )

    threshold_rows: list[dict[str, object]] = []
    thresholds = sorted({0.70, 0.80, 0.90, float(MATCH_CONFIDENCE_THRESHOLD)})

    for threshold in thresholds:
        count = combined.filter(pl.col("match_confidence") >= threshold).height
        threshold_rows.append({
            "source": "all_routes_with_confidence",
            "statistic": f"count_ge_{threshold:.2f}",
            "value": float(count),
        })
        threshold_rows.append({
            "source": "all_routes_with_confidence",
            "statistic": f"percentage_ge_{threshold:.2f}",
            "value": pct(count, combined.height),
        })

    output_frames.append(
        pl.DataFrame(
            threshold_rows,
            schema={
                "source": pl.String,
                "statistic": pl.String,
                "value": pl.Float64,
            },
        )
    )

    return pl.concat(output_frames, how="vertical")


def trace_quality_summary(routes: pl.DataFrame) -> pl.DataFrame:
    columns = [
        "n_points_original",
        "n_points_submitted",
        "trace_duration_sec",
        "trace_max_gap_sec",
        "request_url_length",
    ]

    summaries: list[pl.DataFrame] = []

    for column in columns:
        if column in routes.columns:
            summaries.append(
                numeric_summary(routes, column)
                .with_columns(pl.lit(column).alias("variable"))
                .select(["variable", "statistic", "value"])
            )

    if not summaries:
        return pl.DataFrame(
            {
                "variable": ["none"],
                "statistic": ["missing_trace_columns"],
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


def route_feature_summary(route_features: pl.DataFrame) -> pl.DataFrame:
    n_trips = route_features.height

    road_cols = [
        f"km_{road_class}"
        for road_class in ROAD_CLASSES
        if f"km_{road_class}" in route_features.columns
    ]

    if road_cols:
        total_road_km = float(
            route_features
            .select(pl.sum_horizontal([pl.col(c) for c in road_cols]).sum())
            .item()
        )
    else:
        total_road_km = 0.0

    rows: list[dict[str, object]] = []

    for road_class in ROAD_CLASSES:
        col = f"km_{road_class}"

        if col not in route_features.columns:
            rows.append({
                "road_class": road_class,
                "total_km": 0.0,
                "mean_km_per_trip": 0.0,
                "share_of_total_road_km": 0.0,
                "trips_with_class": 0,
                "percentage_of_trips": 0.0,
            })
            continue

        total_km = float(route_features.select(pl.col(col).sum()).item() or 0.0)
        trips_with_class = int(route_features.filter(pl.col(col) > 0).height)

        rows.append({
            "road_class": road_class,
            "total_km": round(total_km, 4),
            "mean_km_per_trip": round(total_km / n_trips, 4) if n_trips else 0.0,
            "share_of_total_road_km": pct(total_km, total_road_km),
            "trips_with_class": trips_with_class,
            "percentage_of_trips": pct(trips_with_class, n_trips),
        })

    return pl.DataFrame(
        rows,
        schema={
            "road_class": pl.String,
            "total_km": pl.Float64,
            "mean_km_per_trip": pl.Float64,
            "share_of_total_road_km": pl.Float64,
            "trips_with_class": pl.Int64,
            "percentage_of_trips": pl.Float64,
        },
    )


def baseline_error_summary(errors: pl.DataFrame) -> pl.DataFrame:
    if errors.height == 0:
        return metric_table([("n_trips", 0)])

    n = errors.height

    rmse = (
        errors
        .select((pl.col("prediction_error_sec") ** 2).mean().sqrt())
        .item()
    )

    rows: list[tuple[str, object]] = [
        ("n_trips", n),
        (
            "mean_signed_error_sec",
            errors.select(pl.col("prediction_error_sec").mean()).item(),
        ),
        (
            "median_signed_error_sec",
            errors.select(pl.col("prediction_error_sec").median()).item(),
        ),
        (
            "MAE_sec",
            errors.select(pl.col("absolute_error_sec").mean()).item(),
        ),
        (
            "median_AE_sec",
            errors.select(pl.col("absolute_error_sec").median()).item(),
        ),
        ("RMSE_sec", rmse),
        (
            "mean_percentage_error",
            errors.select(pl.col("percentage_error").mean()).item()
            if "percentage_error" in errors.columns
            else None,
        ),
        (
            "median_absolute_percentage_error",
            errors.select(pl.col("absolute_percentage_error").median()).item()
            if "absolute_percentage_error" in errors.columns
            else None,
        ),
        (
            f"share_abs_error_gt_{int(LARGE_ERROR_SEC)}_sec",
            pct(errors.filter(pl.col("absolute_error_sec") > LARGE_ERROR_SEC).height, n),
        ),
        (
            "share_abs_error_gt_600_sec",
            pct(errors.filter(pl.col("absolute_error_sec") > 600).height, n),
        ),
    ]

    return metric_table(rows)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    print("Loading Phase 2 outputs...")

    clean = pl.read_parquet(CLEAN_TRAJECTORIES_PARQUET)
    routes = pl.read_parquet(ROUTES_PARQUET)
    rejections = pl.read_parquet(ROUTE_REJECTION_SUMMARY_PARQUET)
    route_features = pl.read_parquet(ROUTE_FEATURES_PARQUET)
    errors = pl.read_parquet(PREDICTION_ERRORS_PARQUET)

    clean_trips = int(clean.select(pl.col("trip_id").n_unique()).item())

    print("Writing Phase 2 diagnostic tables...")

    write_table(
        route_matching_summary(clean, routes, rejections),
        "phase2_route_matching_summary.csv",
    )
    write_table(
        route_rejection_categories(rejections, clean_trips),
        "phase2_route_rejection_categories.csv",
    )
    write_table(
        route_rejection_reasons(rejections, clean_trips),
        "phase2_route_rejection_reasons.csv",
    )
    write_table(
        match_confidence_summary(routes, rejections),
        "phase2_match_confidence_summary.csv",
    )
    write_table(
        trace_quality_summary(routes),
        "phase2_trace_quality_summary.csv",
    )
    write_table(
        route_feature_summary(route_features),
        "phase2_route_feature_summary.csv",
    )
    write_table(
        baseline_error_summary(errors),
        "phase2_baseline_error_summary.csv",
    )

    print("\nPhase 2 diagnostic summary")
    print("--------------------------")
    print(f"Current MATCH_CONFIDENCE_THRESHOLD: {MATCH_CONFIDENCE_THRESHOLD}")
    print(route_matching_summary(clean, routes, rejections))


if __name__ == "__main__":
    main()
