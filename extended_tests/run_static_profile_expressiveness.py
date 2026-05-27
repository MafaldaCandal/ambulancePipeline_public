"""
run_static_profile_expressiveness.py

Assess whether one static OSRM profile leaves time-of-day-specific residual error
within the active main calibration sample.

Outputs:
    runs/RunXXX/extended_tests/static_profile_expressiveness/
        context_augmented_regression.csv
        peak_time_regression.csv
        peak_hour_regression.csv                 # compatibility copy
        static_profile_expressiveness_summary.csv
        status.csv
        peak_time_calibrated_profile/

Design:
    - Uses the active main run as the calibration sample.
    - Fits a pooled context-augmented residual regression with road-class/turn
      terms plus time-of-day dummies for peak time and night.
    - Fits the main road-class/turn regression on peak-time trips only.
    - Evaluates baseline, main calibrated, and context-adjusted predictions on
      the peak-time subset.
    - Attempts an optional peak-time calibrated profile by writing peak-time
      coefficients to a temporary run and calling p3s9, p4s10, and p4s11.

This script does not assume ZHZ. Dataset and region are inferred from:
    1. PIPELINE_INPUT_NAME / PIPELINE_MAIN_INPUT / PIPELINE_DATASET
    2. PIPELINE_MAIN_REGION / PIPELINE_ACTIVE_REGION
    3. columns in regression_table.parquet, if available
    4. safe fallback labels if metadata is unavailable
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from extended_test_utils import (
    EXTENDED_ROOT,
    MAIN_RUN_DIR,
    PROFILE_FROM_COEFFICIENTS_STEPS,
    STANDARD_SUMMARY_COLUMNS,
    StatusRow,
    copy_main_access_profile,
    ensure_run_subdirs,
    print_header,
    run_steps,
    summarise_evaluation_run,
    write_status_csv,
)


TEST_TYPE = "static_profile_expressiveness"
ROOT = EXTENDED_ROOT / TEST_TYPE

STATUS_CSV = ROOT / "status.csv"
CONTEXT_REGRESSION_CSV = ROOT / "context_augmented_regression.csv"
PEAK_TIME_REGRESSION_CSV = ROOT / "peak_time_regression.csv"
PEAK_HOUR_REGRESSION_CSV = ROOT / "peak_hour_regression.csv"  # compatibility name
SUMMARY_CSV = ROOT / "static_profile_expressiveness_summary.csv"
PEAK_TIME_PROFILE_RUN = ROOT / "peak_time_calibrated_profile"

ROAD_TERMS = [
    "km_motorway",
    "km_trunk",
    "km_primary",
    "km_secondary",
    "km_tertiary",
    "km_residential",
    "km_unclassified",
    "km_service",
    "km_living_street",
]


@dataclass(frozen=True)
class AnalysisMetadata:
    dataset: str
    region: str


# =========================================================
# Metadata
# =========================================================

def _env_first(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value.removeprefix("input_")
    return None


def _single_text_value(df: pl.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col not in df.columns:
            continue

        values = [
            str(value)
            for value in df[col].drop_nulls().unique().to_list()
            if str(value).strip()
        ]

        if len(values) == 1:
            return values[0]

    return None


def infer_analysis_metadata(regression_table: pl.DataFrame) -> AnalysisMetadata:
    dataset = _env_first([
        "PIPELINE_INPUT_NAME",
        "PIPELINE_MAIN_INPUT",
        "PIPELINE_DATASET",
        "PIPELINE_MAIN_DATASET",
    ])

    region = _env_first([
        "PIPELINE_MAIN_REGION",
        "PIPELINE_ACTIVE_REGION",
        "PIPELINE_REGION",
    ])

    if dataset is None:
        dataset = _single_text_value(
            regression_table,
            ["dataset", "input_name", "pipeline_input_name", "source_dataset"],
        )

    if region is None:
        region = _single_text_value(
            regression_table,
            ["region", "rav", "rav_region", "region_id"],
        )

    if region is None and dataset is not None and "_" in dataset:
        region = dataset.split("_", 1)[0]

    if dataset is None and region is not None:
        dataset = f"{region}_active_run"

    if dataset is None:
        dataset = "active_main_run"

    if region is None:
        region = "active_main_region"

    return AnalysisMetadata(dataset=dataset, region=region)


# =========================================================
# Basic helpers
# =========================================================

def first_existing_col(df: pl.DataFrame, candidates: Sequence[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def require_statsmodels() -> None:
    try:
        import statsmodels.api as sm  # noqa: F401
    except Exception as exc:
        raise RuntimeError("statsmodels is required for this extended test.") from exc


def standardise_summary_schema(df: pl.DataFrame) -> pl.DataFrame:
    out = df

    for col in STANDARD_SUMMARY_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).alias(col))

    return out.select(STANDARD_SUMMARY_COLUMNS)


# =========================================================
# Input loading
# =========================================================

def load_main_tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    reg_path = MAIN_RUN_DIR / "outputs" / "regression_table.parquet"
    errors_path = MAIN_RUN_DIR / "results" / "calibration_evaluation_trip_errors.parquet"

    if not reg_path.exists():
        raise FileNotFoundError(f"Missing regression table: {reg_path}")
    if not errors_path.exists():
        raise FileNotFoundError(f"Missing trip-error table: {errors_path}")

    return pl.read_parquet(reg_path), pl.read_parquet(errors_path)


def add_time_of_day_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add time-of-day context variables.

    Omitted category:
        ordinary daytime outside peak time and night.
    """
    if "dispatch_time" not in df.columns:
        raise ValueError("regression_table.parquet must contain dispatch_time.")

    dtype = df.schema["dispatch_time"]

    if dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("dispatch_time")
            .str.to_datetime(strict=False)
            .alias("dispatch_time")
        )
    else:
        df = df.with_columns(
            pl.col("dispatch_time")
            .cast(pl.Datetime("us"), strict=False)
            .alias("dispatch_time")
        )

    hour = pl.col("dispatch_time").dt.hour()

    is_peak = ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))
    is_night = (hour < 7) | (hour >= 22)

    return (
        df
        .with_columns([
            pl.when(is_peak)
            .then(pl.lit("peak"))
            .when(is_night)
            .then(pl.lit("night"))
            .otherwise(pl.lit("day"))
            .alias("time_period"),
        ])
        .with_columns([
            (pl.col("time_period") == "peak").cast(pl.Int8).alias("ctx_peak"),
            (pl.col("time_period") == "night").cast(pl.Int8).alias("ctx_night"),
            (pl.col("time_period") == "peak").alias("is_peak_time"),
        ])
    )


# =========================================================
# Regression
# =========================================================

def available_predictors(df: pl.DataFrame, *, include_context: bool) -> list[str]:
    predictors = [term for term in ROAD_TERMS if term in df.columns]

    if "n_turns" in df.columns:
        predictors.append("n_turns")
    elif "turns" in df.columns:
        predictors.append("turns")

    if include_context:
        predictors.extend([
            term for term in ["ctx_peak", "ctx_night"]
            if term in df.columns
        ])

    if not predictors:
        raise ValueError("No usable regression predictors found.")

    return predictors


def fit_ols_hc3(
    df: pl.DataFrame,
    *,
    predictors: list[str],
    outcome: str = "prediction_error_sec",
) -> tuple[pl.DataFrame, object, list[str]]:
    require_statsmodels()
    import statsmodels.api as sm

    missing = [col for col in [outcome] + predictors if col not in df.columns]
    if missing:
        raise ValueError(f"Missing regression columns: {missing}")

    work = df.select([outcome] + predictors).drop_nulls()

    if work.height <= len(predictors) + 5:
        raise ValueError(
            f"Too few rows for regression: n={work.height}, p={len(predictors)}"
        )

    pdf = work.to_pandas()
    y = pdf[outcome].astype(float)
    X = sm.add_constant(pdf[predictors].astype(float), has_constant="add")

    model = sm.OLS(y, X).fit(cov_type="HC3")
    conf = model.conf_int()

    rows = []
    for name in model.params.index:
        variable = "Intercept" if name == "const" else str(name)
        rows.append({
            "variable": variable,
            "coefficient": float(model.params[name]),
            "std_error_HC3": float(model.bse[name]),
            "p_value": float(model.pvalues[name]),
            "conf_low": float(conf.loc[name, 0]),
            "conf_high": float(conf.loc[name, 1]),
            "n": int(model.nobs),
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
        })

    return pl.DataFrame(rows), model, predictors


# =========================================================
# Metric computation
# =========================================================

def compute_metrics(
    df: pl.DataFrame,
    *,
    predicted_col: str,
    model_name: str,
    metadata: AnalysisMetadata,
    variant: str,
) -> dict:
    realised_col = first_existing_col(
        df,
        [
            "realised_travel_time_sec",
            "realized_travel_time_sec",
            "observed_travel_time_sec",
            "actual_travel_time_sec",
        ],
    )

    if realised_col is None:
        raise ValueError("Could not find realised/observed travel-time column.")

    if predicted_col not in df.columns:
        raise ValueError(f"Missing predicted-time column: {predicted_col}")

    out = (
        df
        .with_columns([
            (pl.col(realised_col) - pl.col(predicted_col)).alias("_err"),
            (pl.col(realised_col) - pl.col(predicted_col)).abs().alias("_ae"),
            ((pl.col(realised_col) <= 900) != (pl.col(predicted_col) <= 900))
            .alias("_miscl"),
        ])
        .select([
            pl.lit(TEST_TYPE).alias("test_type"),
            pl.lit(metadata.dataset).alias("dataset"),
            pl.lit(metadata.region).alias("region"),
            pl.lit(variant).alias("variant"),
            pl.lit(model_name).alias("model"),
            pl.len().alias("n_trips"),
            pl.col("_err").mean().alias("mean_signed_error_sec"),
            pl.col("_err").median().alias("median_signed_error_sec"),
            pl.col("_ae").mean().alias("mae_sec"),
            pl.col("_ae").median().alias("median_ae_sec"),
            (pl.col("_err") ** 2).mean().sqrt().alias("rmse_sec"),
            pl.col("_ae").quantile(0.95, interpolation="nearest").alias("p95_ae_sec"),
            (pl.col("_ae") > 300).mean().alias("share_error_gt_5min"),
            pl.col("_miscl").mean().alias("share_15min_misclassified"),
            pl.lit(None).alias("mae_change_sec"),
            pl.lit(None).alias("mae_change_percent"),
            pl.lit(None).alias("rmse_change_sec"),
            pl.lit(None).alias("rmse_change_percent"),
            pl.lit(None).alias("large_error_share_change"),
            pl.lit(None).alias("standard_misclassification_change"),
            pl.lit(None).alias("multiplier"),
            pl.lit("conducted").alias("status"),
            pl.lit("").alias("message"),
            pl.lit(str(MAIN_RUN_DIR)).alias("run_dir"),
        ])
    )

    return out.row(0, named=True)


def add_changes_relative_to_baseline(summary: pl.DataFrame) -> pl.DataFrame:
    if summary.is_empty() or "model" not in summary.columns:
        return summary

    baseline = summary.filter(pl.col("model") == "baseline_osrm")
    if baseline.height != 1:
        return summary

    base = baseline.row(0, named=True)
    base_mae = base.get("mae_sec")
    base_rmse = base.get("rmse_sec")
    base_large = base.get("share_error_gt_5min")
    base_miscl = base.get("share_15min_misclassified")

    rows = []

    for row in summary.iter_rows(named=True):
        out = dict(row)

        if row.get("model") != "baseline_osrm":
            if base_mae not in (None, 0) and row.get("mae_sec") is not None:
                out["mae_change_sec"] = row["mae_sec"] - base_mae
                out["mae_change_percent"] = 100 * (row["mae_sec"] - base_mae) / base_mae

            if base_rmse not in (None, 0) and row.get("rmse_sec") is not None:
                out["rmse_change_sec"] = row["rmse_sec"] - base_rmse
                out["rmse_change_percent"] = 100 * (row["rmse_sec"] - base_rmse) / base_rmse

            if base_large is not None and row.get("share_error_gt_5min") is not None:
                out["large_error_share_change"] = row["share_error_gt_5min"] - base_large

            if base_miscl is not None and row.get("share_15min_misclassified") is not None:
                out["standard_misclassification_change"] = (
                    row["share_15min_misclassified"] - base_miscl
                )

        rows.append(out)

    return standardise_summary_schema(pl.DataFrame(rows))


def build_context_adjusted_summary(
    reg_context: pl.DataFrame,
    trip_errors: pl.DataFrame,
    model,
    predictors: list[str],
    metadata: AnalysisMetadata,
) -> pl.DataFrame:
    import statsmodels.api as sm

    peak = reg_context.filter(pl.col("is_peak_time"))

    if peak.is_empty():
        raise ValueError("No peak-time trips found.")

    if "trip_id" not in peak.columns or "trip_id" not in trip_errors.columns:
        raise ValueError("Both regression and trip-error tables must contain trip_id.")

    ids = peak.select("trip_id").unique()
    errors = trip_errors.join(ids, on="trip_id", how="inner")

    if errors.is_empty():
        raise ValueError("No peak-time trips matched the trip-error table.")

    baseline_col = first_existing_col(
        errors,
        [
            "baseline_predicted_time_sec",
            "baseline_predicted_travel_time_sec",
            "predicted_travel_time_sec",
            "osrm_predicted_time_sec",
            "osrm_duration_sec",
        ],
    )
    calibrated_col = first_existing_col(
        errors,
        [
            "calibrated_osrm_time_sec",
            "calibrated_predicted_time_sec",
            "calibrated_predicted_travel_time_sec",
        ],
    )

    if baseline_col is None:
        raise ValueError("Could not find baseline predicted-time column.")
    if calibrated_col is None:
        raise ValueError("Could not find calibrated predicted-time column.")

    pdf = peak.select(["trip_id"] + predictors).to_pandas()
    X = sm.add_constant(pdf[predictors].astype(float), has_constant="add")
    pdf["context_fitted_residual_sec"] = model.predict(X)

    fitted = pl.from_pandas(pdf[["trip_id", "context_fitted_residual_sec"]])

    errors = (
        errors
        .join(fitted, on="trip_id", how="inner")
        .with_columns(
            (pl.col(baseline_col) + pl.col("context_fitted_residual_sec"))
            .alias("context_adjusted_predicted_time_sec")
        )
    )

    summary = pl.DataFrame([
        compute_metrics(
            errors,
            predicted_col=baseline_col,
            model_name="baseline_osrm",
            metadata=metadata,
            variant="peak_time",
        ),
        compute_metrics(
            errors,
            predicted_col=calibrated_col,
            model_name="main_calibrated_profile",
            metadata=metadata,
            variant="peak_time",
        ),
        compute_metrics(
            errors,
            predicted_col="context_adjusted_predicted_time_sec",
            model_name="context_adjusted_correction",
            metadata=metadata,
            variant="peak_time",
        ),
    ])

    return add_changes_relative_to_baseline(summary)


# =========================================================
# Optional peak-time profile
# =========================================================

def p3s9_coefficient_table(peak_regression: pl.DataFrame) -> pl.DataFrame:
    allowed = ROAD_TERMS + ["n_turns"]

    df = peak_regression

    if "variable" not in df.columns:
        raise ValueError("Peak-time regression table must contain a variable column.")

    if "turns" in df["variable"].to_list():
        df = df.with_columns(
            pl.when(pl.col("variable") == "turns")
            .then(pl.lit("n_turns"))
            .otherwise(pl.col("variable"))
            .alias("variable")
        )

    return df.filter(pl.col("variable").is_in(allowed))


def _filter_and_copy_by_trip_id(src: Path, dst: Path, ids: pl.DataFrame) -> None:
    if not src.exists():
        return

    df = pl.read_parquet(src)

    if "trip_id" not in df.columns:
        shutil.copy2(src, dst)
        return

    filtered = df.join(ids, on="trip_id", how="inner")

    if filtered.is_empty():
        raise ValueError(f"Filtered peak-time file would be empty: {src}")

    filtered.write_parquet(dst)


def prepare_peak_time_profile_run(
    reg_context: pl.DataFrame,
    peak_regression: pl.DataFrame,
) -> None:
    ensure_run_subdirs(PEAK_TIME_PROFILE_RUN, overwrite=True)
    copy_main_access_profile(PEAK_TIME_PROFILE_RUN)

    ids = reg_context.filter(pl.col("is_peak_time")).select("trip_id").unique()

    required_files = [
        ("outputs", "clean_trajectories.parquet"),
        ("outputs", "prediction_errors.parquet"),
    ]

    optional_files = [
        ("outputs", "route_features.parquet"),
        ("outputs", "regression_table.parquet"),
        ("results", "calibration_evaluation_trip_errors.parquet"),
    ]

    for folder, filename in required_files:
        src = MAIN_RUN_DIR / folder / filename
        dst = PEAK_TIME_PROFILE_RUN / folder / filename

        if not src.exists():
            raise FileNotFoundError(f"Missing required file: {src}")

        _filter_and_copy_by_trip_id(src, dst, ids)

    for folder, filename in optional_files:
        src = MAIN_RUN_DIR / folder / filename
        dst = PEAK_TIME_PROFILE_RUN / folder / filename
        _filter_and_copy_by_trip_id(src, dst, ids)

    coef = p3s9_coefficient_table(peak_regression)

    if coef.is_empty():
        raise ValueError("Peak-time regression produced no coefficients usable by p3s9.")

    coef.write_csv(PEAK_TIME_PROFILE_RUN / "results" / "regression_coefficients.csv")


# =========================================================
# Failure summary
# =========================================================

def failed_summary(message: str, metadata: AnalysisMetadata | None = None) -> pl.DataFrame:
    dataset = metadata.dataset if metadata else "active_main_run"
    region = metadata.region if metadata else "active_main_region"

    return standardise_summary_schema(pl.DataFrame([{
        "test_type": TEST_TYPE,
        "dataset": dataset,
        "region": region,
        "variant": "static_profile_expressiveness",
        "model": "",
        "n_trips": None,
        "mean_signed_error_sec": None,
        "median_signed_error_sec": None,
        "mae_sec": None,
        "median_ae_sec": None,
        "rmse_sec": None,
        "p95_ae_sec": None,
        "share_error_gt_5min": None,
        "share_15min_misclassified": None,
        "mae_change_sec": None,
        "mae_change_percent": None,
        "rmse_change_sec": None,
        "rmse_change_percent": None,
        "large_error_share_change": None,
        "standard_misclassification_change": None,
        "multiplier": None,
        "status": "failed",
        "message": message,
        "run_dir": str(ROOT),
    }]))


# =========================================================
# Main
# =========================================================

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print_header("Static-profile expressiveness: time-of-day")

    statuses: list[StatusRow] = []
    metadata: AnalysisMetadata | None = None

    try:
        reg, trip_errors = load_main_tables()
        metadata = infer_analysis_metadata(reg)

        reg = add_time_of_day_columns(reg)

        context_predictors = available_predictors(reg, include_context=True)
        context_reg, context_model, context_predictors = fit_ols_hc3(
            reg,
            predictors=context_predictors,
        )
        context_reg.write_csv(CONTEXT_REGRESSION_CSV)

        peak = reg.filter(pl.col("is_peak_time"))
        main_predictors = available_predictors(peak, include_context=False)
        peak_reg, _peak_model, _peak_predictors = fit_ols_hc3(
            peak,
            predictors=main_predictors,
        )
        peak_reg.write_csv(PEAK_TIME_REGRESSION_CSV)
        peak_reg.write_csv(PEAK_HOUR_REGRESSION_CSV)  # compatibility copy

        summary = build_context_adjusted_summary(
            reg,
            trip_errors,
            context_model,
            context_predictors,
            metadata,
        )

        statuses.append(StatusRow(
            test_type=TEST_TYPE,
            dataset=metadata.dataset,
            region=metadata.region,
            variant="context_augmented_and_peak_time_regressions",
            run_dir=str(ROOT),
            status="success",
            failed_step="",
            return_code="0",
            message="completed",
        ))

        try:
            prepare_peak_time_profile_run(reg, peak_reg)

            ok, failed_step, return_code, message = run_steps(
                PROFILE_FROM_COEFFICIENTS_STEPS,
                run_dir=PEAK_TIME_PROFILE_RUN,
                input_name=metadata.dataset,
                active_region=metadata.region,
            )

            if ok:
                peak_profile = (
                    summarise_evaluation_run(
                        run_dir=PEAK_TIME_PROFILE_RUN,
                        test_type=TEST_TYPE,
                        dataset=metadata.dataset,
                        region=metadata.region,
                        variant="peak_time_profile",
                        status="conducted",
                        message="",
                    )
                    .filter(pl.col("model") == "calibrated_osrm")
                    .with_columns(
                        pl.lit("peak_time_calibrated_profile").alias("model")
                    )
                )

                summary = pl.concat(
                    [summary, standardise_summary_schema(peak_profile)],
                    how="diagonal_relaxed",
                )
                summary = add_changes_relative_to_baseline(summary)

                statuses.append(StatusRow(
                    test_type=TEST_TYPE,
                    dataset=metadata.dataset,
                    region=metadata.region,
                    variant="peak_time_profile",
                    run_dir=str(PEAK_TIME_PROFILE_RUN),
                    status="success",
                    failed_step="",
                    return_code="0",
                    message="completed",
                ))

            else:
                statuses.append(StatusRow(
                    test_type=TEST_TYPE,
                    dataset=metadata.dataset,
                    region=metadata.region,
                    variant="peak_time_profile",
                    run_dir=str(PEAK_TIME_PROFILE_RUN),
                    status="failed",
                    failed_step=failed_step,
                    return_code=return_code,
                    message=message,
                ))

        except Exception as exc:
            statuses.append(StatusRow(
                test_type=TEST_TYPE,
                dataset=metadata.dataset,
                region=metadata.region,
                variant="peak_time_profile",
                run_dir=str(PEAK_TIME_PROFILE_RUN),
                status="failed",
                failed_step="setup_or_profile_generation",
                return_code="",
                message=str(exc),
            ))

        standardise_summary_schema(summary).write_csv(SUMMARY_CSV)

    except Exception as exc:
        if metadata is None:
            metadata = AnalysisMetadata(
                dataset="active_main_run",
                region="active_main_region",
            )

        statuses.append(StatusRow(
            test_type=TEST_TYPE,
            dataset=metadata.dataset,
            region=metadata.region,
            variant="static_profile_expressiveness",
            run_dir=str(ROOT),
            status="failed",
            failed_step="setup_or_regression",
            return_code="",
            message=str(exc),
        ))

        failed_summary(str(exc), metadata).write_csv(SUMMARY_CSV)

    write_status_csv(STATUS_CSV, statuses)

    print(f"Saved status:  {STATUS_CSV}")
    print(f"Saved summary: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()