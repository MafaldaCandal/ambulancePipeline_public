"""
p3s8_run_regression.py

Phase 3, step 8.

Purpose:
    Fit OLS residual calibration regressions using the modelling-ready
    regression table.

Inputs:
    runs/RunXXX/outputs/regression_table.parquet

Outputs:
    runs/RunXXX/results/regression_coefficients.csv
    runs/RunXXX/results/regression_summary.txt

Design:
    This module is reusable. The main pipeline runs the default calibration
    regression. Extended tests and variants can import run_regression().
"""

import polars as pl
import statsmodels.formula.api as smf

from paths import (
    REGRESSION_TABLE_PARQUET,
    REGRESSION_COEFFICIENTS_CSV,
    REGRESSION_SUMMARY_TXT,
)

from configs import EXCLUDE_IQR3_PREDICTION_ERROR_OUTLIERS

DEFAULT_EXPLANATORY_VARIABLES = [
    "km_motorway",
    "km_trunk",
    "km_primary",
    "km_secondary",
    "km_tertiary",
    "km_residential",
    "km_unclassified",
    "km_service",
    "km_living_street",
    "n_turns",
]


# =========================================================
# Formula construction
# =========================================================

def build_formula(
    explanatory_variables: list[str] | None = None,
    include_is_A0: bool = False,
    include_moving_at_dispatch: bool = False,
    include_distance_km: bool = False,
    include_season: bool = False,
    include_period_of_day: bool = False,
) -> str:
    """
    Build a statsmodels formula.

    Reference categories:
        - season: spring
        - period_of_day: day
    """

    if explanatory_variables is None:
        explanatory_variables = DEFAULT_EXPLANATORY_VARIABLES

    controls = []

    if include_distance_km:
        controls.append("distance_km")

    if include_moving_at_dispatch:
        controls.append("C(moving_at_dispatch)")

    if include_is_A0:
        controls.append("C(is_A0)")

    if include_season:
        controls.append("C(season, Treatment(reference='spring'))")

    if include_period_of_day:
        controls.append("C(period_of_day, Treatment(reference='day'))")

    rhs = explanatory_variables + controls

    if not rhs:
        raise ValueError(
            "At least one explanatory variable or control must be included."
        )

    return "prediction_error_sec ~ " + " + ".join(rhs)


# =========================================================
# Regression helpers
# =========================================================

def prepare_regression_data(regression_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Drop nulls before statsmodels fitting.
    """
    return regression_pl.drop_nulls()


def fit_regression(regression_pl: pl.DataFrame, formula: str):
    """
    Fit OLS regression with HC3 robust standard errors.
    """

    regression_pd = prepare_regression_data(regression_pl).to_pandas()

    if regression_pd.empty:
        raise ValueError("Regression dataset is empty after dropping nulls.")

    categorical_cols = [
        "is_A0",
        "moving_at_dispatch",
        "season",
        "period_of_day",
    ]

    for col in categorical_cols:
        if col in regression_pd.columns:
            regression_pd[col] = regression_pd[col].astype("category")

    return smf.ols(
        formula=formula,
        data=regression_pd,
    ).fit(cov_type="HC3")


def build_coefficient_table(model) -> pl.DataFrame:
    """
    Extract coefficient estimates and HC3 robust uncertainty measures.
    """

    conf_int = model.conf_int()

    return pl.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "std_error_HC3": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
        "conf_low": conf_int[0].values,
        "conf_high": conf_int[1].values,
    })


def save_regression_outputs(model, coef_table: pl.DataFrame) -> None:
    """
    Save main regression outputs to disk.
    """

    REGRESSION_COEFFICIENTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    REGRESSION_SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)

    coef_table.write_csv(REGRESSION_COEFFICIENTS_CSV)

    with open(REGRESSION_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())


def exclude_iqr3_outliers(regression_pl: pl.DataFrame) -> pl.DataFrame:
    """
    Exclude observations flagged as extreme 3×IQR prediction-error outliers.

    This is intended for robustness checks, not for the main calibration model.
    """
    flag_col = "extreme_prediction_error_iqr3"

    if flag_col not in regression_pl.columns:
        raise ValueError(
            f"Cannot exclude IQR outliers because {flag_col!r} is missing. "
            "Rerun p3s7_prepare_regression_dataset.py first."
        )

    before = regression_pl.height
    filtered = regression_pl.filter(~pl.col(flag_col))
    after = filtered.height

    print(
        f"Excluded {before - after} IQR3 residual outlier trip(s) "
        f"from regression sample ({before} -> {after})."
    )

    return filtered

# =========================================================
# Reusable runner
# =========================================================

def run_regression(
    explanatory_variables: list[str] | None = None,
    include_is_A0: bool = False,
    include_moving_at_dispatch: bool = False,
    include_distance_km: bool = False,
    include_season: bool = False,
    include_period_of_day: bool = False,
    filter_expr: pl.Expr | None = None,
    save_outputs: bool = True,
    exclude_extreme_prediction_error_iqr3: bool = False,
):
    """
    Load the regression table, optionally filter it, fit a regression,
    optionally save outputs, and return the fitted model and coefficient table.
    """

    if not REGRESSION_TABLE_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing regression table: {REGRESSION_TABLE_PARQUET}"
        )

    print("Loading regression table...")

    regression_pl = pl.read_parquet(REGRESSION_TABLE_PARQUET)

    if filter_expr is not None:
        regression_pl = regression_pl.filter(filter_expr)

    if exclude_extreme_prediction_error_iqr3:
        regression_pl = exclude_iqr3_outliers(regression_pl)

    if regression_pl.is_empty():
        raise ValueError("Regression table is empty before model fitting.")

    formula = build_formula(
        explanatory_variables=explanatory_variables,
        include_is_A0=include_is_A0,
        include_moving_at_dispatch=include_moving_at_dispatch,
        include_distance_km=include_distance_km,
        include_season=include_season,
        include_period_of_day=include_period_of_day,
    )

    print("Fitting regression...")
    print(formula)

    model = fit_regression(regression_pl, formula)
    coef_table = build_coefficient_table(model)

    if save_outputs:
        save_regression_outputs(model, coef_table)
        print(f"Saved: {REGRESSION_COEFFICIENTS_CSV}")
        print(f"Saved: {REGRESSION_SUMMARY_TXT}")

    return model, coef_table


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    """
    Run the main calibration regression.
    """

    model, _ = run_regression(
        explanatory_variables=None,
        include_is_A0=False,
        include_moving_at_dispatch=False,
        include_distance_km=False,
        include_season=False,
        include_period_of_day=False,
        exclude_extreme_prediction_error_iqr3=EXCLUDE_IQR3_PREDICTION_ERROR_OUTLIERS,
        save_outputs=True,
    )

    print("Regression completed.")
    print(f"Observations: {int(model.nobs)}")
    print(f"Adjusted R²: {model.rsquared_adj:.4f}")


if __name__ == "__main__":
    main()