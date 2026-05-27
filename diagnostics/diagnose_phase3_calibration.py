"""
Phase 3 diagnostics: residual analysis and OSRM calibration.

This script reads the outputs of:
  - p3s7_prepare_regression_dataset.py
  - p3s8_run_regression.py
  - p3s9_create_calibrated_profile.py

It does not run regressions and does not modify Lua profiles. It writes
simple diagnostic CSV files to:
  results/diagnostics/

Diagnostic purpose:
  Check whether the regression dataset, coefficient estimates, and translated
  OSRM parameter changes look credible before predictive evaluation.
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
    PREDICTION_ERRORS_PARQUET,
    ROUTE_FEATURES_PARQUET,
    REGRESSION_TABLE_PARQUET,
    REGRESSION_COEFFICIENTS_CSV,
    CALIBRATION_CHANGES_CSV,
)

from configs import (
    MAX_CALIBRATED_SPEED_KMH,
    MIN_CALIBRATED_SPEED_KMH,
    ROAD_CLASSES,
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
    Return count, mean, median, selected quantiles, min, and max for one
    numeric column.

    Values are explicitly stored as Float64 so Polars does not fail when
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


# ---------------------------------------------------------------------
# Diagnostic tables
# ---------------------------------------------------------------------

def regression_sample_summary(pred_errors: pl.DataFrame, route_features: pl.DataFrame, regression: pl.DataFrame) -> pl.DataFrame:
    prediction_rows = pred_errors.height
    route_feature_rows = route_features.height
    regression_rows = regression.height

    return pl.DataFrame({
        "metric": [
            "prediction_error_rows",
            "route_feature_rows",
            "regression_table_rows",
            "unique_regression_trips",
            "rows_lost_prediction_to_regression",
            "rows_lost_route_features_to_regression",
        ],
        "value": [
            prediction_rows,
            route_feature_rows,
            regression_rows,
            regression.select(pl.col("trip_id").n_unique()).item() if "trip_id" in regression.columns else regression_rows,
            prediction_rows - regression_rows,
            route_feature_rows - regression_rows,
        ],
    })


def variable_missingness(regression: pl.DataFrame) -> pl.DataFrame:
    total = regression.height

    rows = []
    for column in regression.columns:
        missing = regression.select(pl.col(column).is_null().sum()).item()
        rows.append({
            "variable": column,
            "missing_count": int(missing),
            "missing_percentage": pct(missing, total),
        })

    return pl.DataFrame(rows).sort("missing_percentage", descending=True)


def regression_variable_summary(regression: pl.DataFrame) -> pl.DataFrame:
    candidate_columns = [
        "prediction_error_sec",
        "distance_km",
        "n_turns",
        *[f"km_{cls}" for cls in ROAD_CLASSES],
        *[f"prop_{cls}" for cls in ROAD_CLASSES if cls != "unclassified"],
    ]

    summaries = []
    for column in candidate_columns:
        if column in regression.columns:
            summary = describe_column(regression, column).rename({column: "value"})
            summary = summary.with_columns(pl.lit(column).alias("variable"))
            summaries.append(summary.select(["variable", "statistic", "value"]))

    if not summaries:
        return pl.DataFrame({"variable": ["none"], "statistic": ["no_numeric_variables_found"], "value": [None]})

    return pl.concat(summaries)


def road_class_sparsity(regression: pl.DataFrame) -> pl.DataFrame:
    n_trips = regression.height
    rows = []

    for cls in ROAD_CLASSES:
        col = f"km_{cls}"
        if col not in regression.columns:
            rows.append({
                "road_class": cls,
                "trips_with_positive_km": 0,
                "percentage_of_trips": 0.0,
                "total_km": 0.0,
                "mean_km_when_present": 0.0,
            })
            continue

        present = regression.filter(pl.col(col) > 0)
        total_km = regression.select(pl.col(col).sum()).item()
        mean_when_present = present.select(pl.col(col).mean()).item() if present.height else 0.0

        rows.append({
            "road_class": cls,
            "trips_with_positive_km": present.height,
            "percentage_of_trips": pct(present.height, n_trips),
            "total_km": round(float(total_km), 4),
            "mean_km_when_present": round(float(mean_when_present), 4),
        })

    return pl.DataFrame(rows)


def coefficient_diagnostics(coefs: pl.DataFrame, sparsity: pl.DataFrame) -> pl.DataFrame:
    if coefs.height == 0:
        return pl.DataFrame({"variable": ["none"], "interpretation_flag": ["no_coefficients_found"]})

    sparsity_lookup = {
        f"km_{row['road_class']}": row["percentage_of_trips"]
        for row in sparsity.iter_rows(named=True)
    }

    rows = []
    for row in coefs.iter_rows(named=True):
        variable = row.get("variable")
        coefficient = row.get("coefficient")
        p_value = row.get("p_value")
        conf_low = row.get("conf_low")
        conf_high = row.get("conf_high")

        flags = []

        if variable in sparsity_lookup and sparsity_lookup[variable] < 5:
            flags.append("sparse_road_class")

        if coefficient is not None and abs(float(coefficient)) > 120:
            flags.append("large_magnitude")

        if conf_low is not None and conf_high is not None:
            if float(conf_low) < 0 < float(conf_high):
                flags.append("wide_or_sign_uncertain_interval")

        if p_value is not None and float(p_value) > 0.10:
            flags.append("not_statistically_strong")

        if not flags:
            flags.append("plausible")

        rows.append({
            **row,
            "interpretation_flag": ";".join(flags),
        })

    return pl.DataFrame(rows)


def calibration_change_diagnostics(changes: pl.DataFrame) -> pl.DataFrame:
    if changes.height == 0:
        return pl.DataFrame({"parameter_type": ["none"], "diagnostic": ["no_calibration_changes_found"]})

    out = changes

    if {"old_speed_kmh", "new_speed_kmh"}.issubset(set(out.columns)):
        out = out.with_columns([
            pl.when(pl.col("old_speed_kmh").is_not_null() & (pl.col("old_speed_kmh") != 0))
            .then(((pl.col("new_speed_kmh") - pl.col("old_speed_kmh")) / pl.col("old_speed_kmh")) * 100)
            .otherwise(None)
            .round(2)
            .alias("speed_change_percent"),
        ])

    return out


def parameter_warnings(regression: pl.DataFrame, coefs: pl.DataFrame, changes: pl.DataFrame, sparsity: pl.DataFrame) -> pl.DataFrame:
    warnings = []

    # Sparse road classes used in the regression.
    for row in sparsity.iter_rows(named=True):
        if row["percentage_of_trips"] < 5:
            warnings.append({
                "warning_type": "sparse_road_class",
                "variable": f"km_{row['road_class']}",
                "value": row["percentage_of_trips"],
                "message": "Road class appears in fewer than 5% of regression trips.",
            })

    # Large coefficient magnitudes.
    if coefs.height > 0 and "coefficient" in coefs.columns:
        for row in coefs.iter_rows(named=True):
            coef = row.get("coefficient")
            if coef is not None and abs(float(coef)) > 120:
                warnings.append({
                    "warning_type": "large_coefficient_magnitude",
                    "variable": row.get("variable"),
                    "value": float(coef),
                    "message": "Coefficient magnitude exceeds 120 seconds per unit; inspect before interpreting.",
                })

    # Speed caps and large speed changes.
    if changes.height > 0 and {"variable", "new_speed_kmh"}.issubset(set(changes.columns)):
        for row in changes.iter_rows(named=True):
            new_speed = row.get("new_speed_kmh")
            if new_speed is None:
                continue

            if float(new_speed) <= MIN_CALIBRATED_SPEED_KMH:
                warnings.append({
                    "warning_type": "speed_capped_min",
                    "variable": row.get("variable"),
                    "value": float(new_speed),
                    "message": "Calibrated speed reached the configured minimum speed.",
                })

            if float(new_speed) >= MAX_CALIBRATED_SPEED_KMH:
                warnings.append({
                    "warning_type": "speed_capped_max",
                    "variable": row.get("variable"),
                    "value": float(new_speed),
                    "message": "Calibrated speed reached the configured maximum speed.",
                })

            old_speed = row.get("old_speed_kmh")
            if old_speed not in (None, 0):
                change_pct = ((float(new_speed) - float(old_speed)) / float(old_speed)) * 100
                if abs(change_pct) > 50:
                    warnings.append({
                        "warning_type": "large_speed_change",
                        "variable": row.get("variable"),
                        "value": round(change_pct, 2),
                        "message": "Calibrated speed changed by more than 50% relative to baseline.",
                    })

    if not warnings:
        warnings.append({
            "warning_type": "none",
            "variable": None,
            "value": None,
            "message": "No simple parameter warnings triggered.",
        })

    return pl.DataFrame(warnings)


# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main() -> None:
    ensure_diagnostics_dir()

    print("Loading Phase 3 outputs...")
    pred_errors = pl.read_parquet(PREDICTION_ERRORS_PARQUET)
    route_features = pl.read_parquet(ROUTE_FEATURES_PARQUET)
    regression = pl.read_parquet(REGRESSION_TABLE_PARQUET)
    coefs = pl.read_csv(REGRESSION_COEFFICIENTS_CSV)
    changes = pl.read_csv(CALIBRATION_CHANGES_CSV)

    sparsity = road_class_sparsity(regression)

    print("Writing Phase 3 diagnostic tables...")
    write_table(regression_sample_summary(pred_errors, route_features, regression), "phase3_regression_sample_summary.csv")
    write_table(variable_missingness(regression), "phase3_variable_missingness.csv")
    write_table(regression_variable_summary(regression), "phase3_regression_variable_summary.csv")
    write_table(sparsity, "phase3_road_class_sparsity.csv")
    write_table(coefficient_diagnostics(coefs, sparsity), "phase3_coefficient_diagnostics.csv")
    write_table(calibration_change_diagnostics(changes), "phase3_calibration_change_diagnostics.csv")
    write_table(parameter_warnings(regression, coefs, changes, sparsity), "phase3_parameter_warnings.csv")

    print("\nPhase 3 diagnostic summary")
    print("--------------------------")
    print(regression_sample_summary(pred_errors, route_features, regression))
    print("\nParameter warnings:")
    print(parameter_warnings(regression, coefs, changes, sparsity))


if __name__ == "__main__":
    main()