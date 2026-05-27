"""
p4s12_generate_results_report.py

Phase 4, step 12.

Purpose:
    Generate a compact PDF report from existing pipeline outputs.

Inputs:
    Phase 1:
        runs/RunXXX/outputs/trip_rejection_summary.parquet

    Phase 2:
        runs/RunXXX/outputs/route_rejection_summary.parquet
        runs/RunXXX/outputs/route_features.parquet

    Phase 3:
        runs/RunXXX/results/regression_coefficients.csv
        runs/RunXXX/results/calibration_changes.csv

    Phase 4:
        runs/RunXXX/results/calibration_evaluation_trip_errors.parquet
        runs/RunXXX/results/calibration_evaluation_metrics.csv
        runs/RunXXX/results/calibration_evaluation_route_diagnostics.csv

Output:
    runs/RunXXX/report/results_report.pdf

Design:
    This report describes the active core pipeline run only. Off-sample
    validation, transferability, and sensitivity reports belong in
    extended_tests/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import polars as pl

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
)

from paths import (
    REPORT_DIR,
    TRIP_REJECTION_SUMMARY_PARQUET,
    ROUTE_REJECTION_SUMMARY_PARQUET,
    ROUTE_FEATURES_PARQUET,
    REGRESSION_COEFFICIENTS_CSV,
    CALIBRATION_CHANGES_CSV,
    CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET,
    CALIBRATION_EVALUATION_METRICS_CSV,
    CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV,
)


# =========================================================
# Report settings
# =========================================================

REPORT_PDF = REPORT_DIR / "results_report.pdf"
FIGURE_DIR = REPORT_DIR / "figures"

MAX_TABLE_ROWS = 18
MAX_COEFFICIENTS_TO_DISPLAY = 12

ENABLE_FIGURES = True
ENABLE_ERROR_DISTRIBUTION_FIGURE = True
ENABLE_MODEL_COMPARISON_FIGURE = True
ENABLE_CALIBRATION_CHANGE_FIGURE = True


# =========================================================
# Data container
# =========================================================

@dataclass
class LoadedData:
    trip_rejections: pl.DataFrame | None
    route_rejections: pl.DataFrame | None
    route_features: pl.DataFrame | None
    regression_coefficients: pl.DataFrame | None
    calibration_changes: pl.DataFrame | None
    trip_errors: pl.DataFrame | None
    calibration_metrics: pl.DataFrame | None
    route_diagnostics: pl.DataFrame | None


# =========================================================
# Small utilities
# =========================================================

def path_exists(path: Path) -> bool:
    return path is not None and path.exists()


def read_parquet_if_exists(path: Path) -> pl.DataFrame | None:
    if not path_exists(path):
        print(f"Missing optional file: {path}")
        return None

    return pl.read_parquet(path)


def read_csv_if_exists(path: Path) -> pl.DataFrame | None:
    if not path_exists(path):
        print(f"Missing optional file: {path}")
        return None

    return pl.read_csv(path)


def normalise_metric_columns(df: pl.DataFrame | None) -> pl.DataFrame | None:
    """
    Accept both older and newer metric-column names.
    """
    if df is None:
        return None

    rename_map = {
        "share_large_error": "share_error_gt_5min",
        "share_standard_misclassified": "share_15min_misclassified",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename({old: new})

    return df


def safe_float(value) -> float | None:
    if value is None:
        return None

    try:
        out = float(value)
    except Exception:
        return None

    if math.isnan(out) or math.isinf(out):
        return None

    return out


def fmt(value, digits: int = 1, percent: bool = False) -> str:
    if value is None:
        return "-"

    if isinstance(value, bool):
        return "yes" if value else "no"

    if isinstance(value, str):
        return value

    x = safe_float(value)

    if x is None:
        return str(value)

    if percent:
        return f"{100 * x:.{digits}f}%"

    if abs(x) >= 1000:
        return f"{x:,.0f}"

    return f"{x:.{digits}f}"


def short_label(value: str | None, width: int = 28) -> str:
    if value is None:
        return "-"

    text = str(value)
    return text if len(text) <= width else text[: width - 1] + "..."


def ensure_dirs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def load_all_data() -> LoadedData:
    return LoadedData(
        trip_rejections=read_parquet_if_exists(TRIP_REJECTION_SUMMARY_PARQUET),
        route_rejections=read_parquet_if_exists(ROUTE_REJECTION_SUMMARY_PARQUET),
        route_features=read_parquet_if_exists(ROUTE_FEATURES_PARQUET),
        regression_coefficients=read_csv_if_exists(REGRESSION_COEFFICIENTS_CSV),
        calibration_changes=read_csv_if_exists(CALIBRATION_CHANGES_CSV),
        trip_errors=read_parquet_if_exists(CALIBRATION_EVALUATION_TRIP_ERRORS_PARQUET),
        calibration_metrics=normalise_metric_columns(read_csv_if_exists(CALIBRATION_EVALUATION_METRICS_CSV)),
        route_diagnostics=read_csv_if_exists(
            CALIBRATION_EVALUATION_ROUTE_DIAGNOSTICS_CSV
        ),
    )


# =========================================================
# Tables
# =========================================================

def build_executive_summary(data: LoadedData) -> list[list[str]]:
    rows = [["Item", "Value"]]

    if data.trip_errors is not None and not data.trip_errors.is_empty():
        rows.append(["Evaluation trips", fmt(data.trip_errors.height, 0)])

    if data.calibration_metrics is not None and not data.calibration_metrics.is_empty():
        metrics = data.calibration_metrics

        if {"model", "mae_sec"}.issubset(set(metrics.columns)):
            baseline = metrics.filter(pl.col("model") == "baseline_osrm")
            calibrated = metrics.filter(pl.col("model") == "calibrated_osrm")

            if not baseline.is_empty():
                rows.append([
                    "Baseline MAE",
                    fmt(baseline.select(pl.col("mae_sec").first()).item(), 1) + " sec",
                ])

            if not calibrated.is_empty():
                rows.append([
                    "Calibrated MAE",
                    fmt(calibrated.select(pl.col("mae_sec").first()).item(), 1) + " sec",
                ])

            if not baseline.is_empty() and not calibrated.is_empty():
                old = baseline.select(pl.col("mae_sec").first()).item()
                new = calibrated.select(pl.col("mae_sec").first()).item()

                if old and old > 0:
                    rows.append([
                        "Calibrated MAE change",
                        fmt((new - old) / old, 1, percent=True),
                    ])

    if data.calibration_changes is not None:
        rows.append([
            "Applied calibration changes",
            fmt(data.calibration_changes.height, 0),
        ])

    if len(rows) == 1:
        rows.append([
            "Status",
            "No Phase 4 results found. Run p4s10 and p4s11 first.",
        ])

    return rows


def build_retention_table(data: LoadedData) -> list[list[str]]:
    rows = [["Stage", "Trips", "Share / note"]]

    if data.trip_rejections is not None and "kept" in data.trip_rejections.columns:
        tr = data.trip_rejections
        candidate = tr.height
        kept = tr.filter(pl.col("kept")).height
        rejected = candidate - kept

        rows.append([
            "Candidate trips",
            fmt(candidate, 0),
            "GPS-linked dispatch windows",
        ])
        rows.append([
            "Valid realised trips",
            fmt(kept, 0),
            f"{fmt(kept / candidate if candidate else None, 1, True)} of candidates",
        ])
        rows.append([
            "Rejected before OSRM",
            fmt(rejected, 0),
            "No valid start/arrival interval",
        ])

    if data.route_rejections is not None:
        rows.append([
            "Rejected at map-matching",
            fmt(data.route_rejections.height, 0),
            "OSRM failure or low confidence",
        ])

    if data.route_features is not None:
        rows.append([
            "Route-feature sample",
            fmt(data.route_features.height, 0),
            "Matched routes with road-class features",
        ])

    if data.regression_coefficients is not None:
        rows.append([
            "Regression coefficients",
            fmt(data.regression_coefficients.height, 0),
            "Estimated terms in coefficient table",
        ])

    if data.trip_errors is not None:
        rows.append([
            "Evaluation trip rows",
            fmt(data.trip_errors.height, 0),
            "Trips with baseline and calibrated prediction",
        ])

    if len(rows) == 1:
        rows.append(["No retention inputs found", "-", "Run earlier phases first"])

    return rows


def build_trip_rejection_table(
    trip_rejections: pl.DataFrame | None,
) -> list[list[str]] | None:
    if trip_rejections is None:
        return None

    cols = [
        "no_sustained_start",
        "no_arrival",
        "arrival_before_start",
    ]

    cols = [c for c in cols if c in trip_rejections.columns]

    if not cols:
        return None

    n = trip_rejections.height
    rows = [["Reason", "Trips", "Share of candidates", "Role"]]

    for col in cols:
        count = trip_rejections.select(pl.col(col).sum()).item()
        rows.append([
            col,
            fmt(count, 0),
            fmt(count / n if n else None, 1, True),
            "hard exclusion",
        ])

    return rows


def build_route_rejection_table(
    route_rejections: pl.DataFrame | None,
) -> list[list[str]] | None:
    if route_rejections is None or route_rejections.is_empty():
        return None

    if "rejection_reason" not in route_rejections.columns:
        return None

    total = route_rejections.height

    if "rejection_category" in route_rejections.columns:
        group_cols = ["rejection_category"]
        if "rejection_summary" in route_rejections.columns:
            group_cols.append("rejection_summary")

        grouped = (
            route_rejections
            .group_by(group_cols)
            .agg(pl.len().alias("trips"))
            .sort("trips", descending=True)
        )

        rows = [["Map-match rejection category", "Summary", "Trips", "Share"]]

        for row in grouped.iter_rows(named=True):
            rows.append([
                short_label(row.get("rejection_category"), 30),
                short_label(row.get("rejection_summary"), 45),
                fmt(row.get("trips"), 0),
                fmt(row.get("trips") / total if total else None, 1, True),
            ])

        return rows

    grouped = (
        route_rejections
        .group_by("rejection_reason")
        .agg(pl.len().alias("trips"))
        .sort("trips", descending=True)
    )

    rows = [["Map-match rejection reason", "Trips", "Share of route rejections"]]

    for row in grouped.iter_rows(named=True):
        rows.append([
            str(row["rejection_reason"]),
            fmt(row["trips"], 0),
            fmt(row["trips"] / total if total else None, 1, True),
        ])

    return rows



def build_performance_table(
    metrics: pl.DataFrame | None,
) -> list[list[str]] | None:
    if metrics is None or metrics.is_empty():
        return None

    keep_cols = [
        "model",
        "n_trips",
        "mean_signed_error_sec",
        "mae_sec",
        "median_ae_sec",
        "rmse_sec",
        "share_error_gt_5min",
        "share_15min_misclassified",
        "mae_change_sec",
        "mae_change_percent",
        "rmse_change_sec",
        "rmse_change_percent",
        "standard_misclassification_change",
        "multiplier",
    ]

    keep_cols = [c for c in keep_cols if c in metrics.columns]

    display = metrics.select(keep_cols).sort("model")

    rows = [[
        "Model",
        "N",
        "Bias",
        "MAE",
        "Med AE",
        "RMSE",
        ">5min",
        "15min miscl.",
        "MAE Δ",
        "MAE Δ%",
        "RMSE Δ",
        "Mult.",
    ]]

    for row in display.iter_rows(named=True):
        rows.append([
            short_label(row.get("model"), 22),
            fmt(row.get("n_trips"), 0),
            fmt(row.get("mean_signed_error_sec"), 1),
            fmt(row.get("mae_sec"), 1),
            fmt(row.get("median_ae_sec"), 1),
            fmt(row.get("rmse_sec"), 1),
            fmt(row.get("share_error_gt_5min"), 1, True),
            fmt(row.get("share_15min_misclassified"), 1, True),
            fmt(row.get("mae_change_sec"), 1),
            fmt(
                (safe_float(row.get("mae_change_percent")) or 0) / 100
                if row.get("mae_change_percent") is not None
                else None,
                1,
                True,
            ),
            fmt(row.get("rmse_change_sec"), 1),
            fmt(row.get("multiplier"), 2),
        ])

    return rows


def build_regression_table(
    coef: pl.DataFrame | None,
) -> list[list[str]] | None:
    if coef is None or coef.is_empty():
        return None

    needed = [
        "variable",
        "coefficient",
        "std_error_HC3",
        "p_value",
        "conf_low",
        "conf_high",
    ]

    cols = [c for c in needed if c in coef.columns]

    if "variable" not in cols or "coefficient" not in cols:
        return None

    df = coef.filter(pl.col("variable") != "Intercept")

    if "coefficient" in df.columns:
        df = (
            df
            .with_columns(pl.col("coefficient").abs().alias("abs_coef"))
            .sort("abs_coef", descending=True)
            .head(MAX_COEFFICIENTS_TO_DISPLAY)
        )

    rows = [
        ["Variable", "Coef.", "HC3 SE", "p-value", "CI low", "CI high"],
        ["(Intercept excluded. Sorted by absolute coefficient size.)", "", "", "", "", ""],
    ]

    for row in df.iter_rows(named=True):
        rows.append([
            short_label(row.get("variable"), 28),
            fmt(row.get("coefficient"), 2),
            fmt(row.get("std_error_HC3"), 2),
            fmt(row.get("p_value"), 3),
            fmt(row.get("conf_low"), 2),
            fmt(row.get("conf_high"), 2),
        ])

    return rows


def build_calibration_changes_table(
    changes: pl.DataFrame | None,
) -> list[list[str]] | None:
    if changes is None or changes.is_empty():
        return None

    rows = [["Parameter", "Road class", "Coefficient", "Old value", "New value", "Capped"]]

    for row in changes.iter_rows(named=True):
        parameter_type = row.get("parameter_type")

        if parameter_type == "road_speed":
            old_value = f"{fmt(row.get('old_speed_kmh'), 1)} km/h"
            new_value = f"{fmt(row.get('new_speed_kmh'), 1)} km/h"
            coef = f"{fmt(row.get('coefficient_sec_per_km'), 2)} sec/km"
        elif parameter_type == "turn_penalty":
            old_value = f"{fmt(row.get('old_turn_penalty_sec'), 1)} sec"
            new_value = f"{fmt(row.get('new_turn_penalty_sec'), 1)} sec"
            coef = f"{fmt(row.get('coefficient_sec_per_turn'), 2)} sec/turn"
        else:
            old_value = "-"
            new_value = "-"
            coef = "-"

        rows.append([
            short_label(parameter_type, 18),
            short_label(row.get("road_class"), 18),
            coef,
            old_value,
            new_value,
            fmt(row.get("was_capped")),
        ])

    return rows


def build_route_diagnostics_table(
    route_diagnostics: pl.DataFrame | None,
) -> list[list[str]] | None:
    if route_diagnostics is None or route_diagnostics.is_empty():
        return None

    rows = [[
        "N",
        "Mean conf.",
        "Median conf.",
        "Min conf.",
        "Mean dist. km",
        "Median dist. km",
        "Max dist. km",
    ]]

    for row in route_diagnostics.iter_rows(named=True):
        rows.append([
            fmt(row.get("n_trips"), 0),
            fmt(row.get("mean_calibrated_match_confidence"), 3),
            fmt(row.get("median_calibrated_match_confidence"), 3),
            fmt(row.get("min_calibrated_match_confidence"), 3),
            fmt(row.get("mean_calibrated_distance_km"), 2),
            fmt(row.get("median_calibrated_distance_km"), 2),
            fmt(row.get("max_calibrated_distance_km"), 2),
        ])

    return rows


def build_km_unknown_table(
    route_features: pl.DataFrame | None,
) -> list[list[str]] | None:
    if route_features is None or route_features.is_empty():
        return None

    if "km_unknown" not in route_features.columns:
        return None

    df = route_features

    total_unknown = df.select(pl.col("km_unknown").sum()).item()
    mean_unknown = df.select(pl.col("km_unknown").mean()).item()
    share_any = df.select((pl.col("km_unknown") > 0).mean()).item()

    total_distance = None

    if "distance_km" in df.columns:
        total_distance = df.select(pl.col("distance_km").sum()).item()

    unknown_share_of_distance = None

    if total_distance and total_distance > 0:
        unknown_share_of_distance = total_unknown / total_distance

    return [
        ["Diagnostic", "Value"],
        ["Total unknown route km", fmt(total_unknown, 2)],
        ["Mean unknown km per trip", fmt(mean_unknown, 3)],
        ["Trips with any unknown km", fmt(share_any, 1, True)],
        ["Unknown share of total route km", fmt(unknown_share_of_distance, 2, True)],
    ]


# =========================================================
# Figures
# =========================================================

def save_model_comparison_figure(metrics: pl.DataFrame | None) -> Path | None:
    if metrics is None or metrics.is_empty():
        return None

    if not {"model", "mae_sec"}.issubset(set(metrics.columns)):
        return None

    df = metrics.select(["model", "mae_sec"]).to_pandas()

    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(df["model"].astype(str).tolist(), df["mae_sec"].astype(float).tolist())
    ax.set_ylabel("MAE (seconds)")
    ax.set_title("MAE by model")
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    path = FIGURE_DIR / "mae_by_model.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def save_error_distribution_figure(trip_errors: pl.DataFrame | None) -> Path | None:
    if trip_errors is None or trip_errors.is_empty():
        return None

    required = {
        "prediction_error_sec",
        "calibrated_prediction_error_sec",
    }

    if not required.issubset(set(trip_errors.columns)):
        return None

    if trip_errors.height < 20:
        return None

    base = (
        trip_errors
        .select("prediction_error_sec")
        .to_series()
        .drop_nulls()
        .to_list()
    )

    calib = (
        trip_errors
        .select("calibrated_prediction_error_sec")
        .to_series()
        .drop_nulls()
        .to_list()
    )

    if not base or not calib:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.hist(base, bins=30, alpha=0.6, label="baseline")
    ax.hist(calib, bins=30, alpha=0.6, label="calibrated")
    ax.axvline(0, linewidth=1)
    ax.set_title("Signed prediction error distribution")
    ax.set_xlabel("realised - predicted time (seconds)")
    ax.set_ylabel("Trips")
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = FIGURE_DIR / "error_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def save_calibration_change_figure(changes: pl.DataFrame | None) -> Path | None:
    if changes is None or changes.is_empty():
        return None

    required = {
        "parameter_type",
        "road_class",
        "old_speed_kmh",
        "new_speed_kmh",
    }

    if not required.issubset(set(changes.columns)):
        return None

    road = changes.filter(pl.col("parameter_type") == "road_speed")

    if road.is_empty():
        return None

    road = (
        road
        .with_columns(
            (pl.col("new_speed_kmh") - pl.col("old_speed_kmh"))
            .alias("speed_change_kmh")
        )
        .sort("speed_change_kmh")
    )

    labels = [short_label(x, 20) for x in road["road_class"].to_list()]
    values = road["speed_change_kmh"].to_list()

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.barh(labels, values)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("New speed - old speed (km/h)")
    ax.set_title("Applied road-speed calibration changes")
    fig.tight_layout()

    path = FIGURE_DIR / "calibration_speed_changes.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def build_figures(data: LoadedData) -> dict[str, Path | None]:
    if not ENABLE_FIGURES:
        return {
            "model_comparison": None,
            "error_distribution": None,
            "calibration_changes": None,
        }

    return {
        "model_comparison": (
            save_model_comparison_figure(data.calibration_metrics)
            if ENABLE_MODEL_COMPARISON_FIGURE
            else None
        ),
        "error_distribution": (
            save_error_distribution_figure(data.trip_errors)
            if ENABLE_ERROR_DISTRIBUTION_FIGURE
            else None
        ),
        "calibration_changes": (
            save_calibration_change_figure(data.calibration_changes)
            if ENABLE_CALIBRATION_CHANGE_FIGURE
            else None
        ),
    }


# =========================================================
# PDF helpers
# =========================================================

def make_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        alignment=TA_LEFT,
    ))

    styles.add(ParagraphStyle(
        name="Note",
        parent=styles["BodyText"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#555555"),
        alignment=TA_LEFT,
    ))

    styles["Title"].fontSize = 18
    styles["Heading1"].fontSize = 13
    styles["Heading2"].fontSize = 11

    return styles


def paragraph(text: str, style) -> Paragraph:
    clean = (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return Paragraph(clean, style)


def section(title: str, styles) -> list:
    return [
        Spacer(1, 0.2 * cm),
        Paragraph(title, styles["Heading1"]),
        Spacer(1, 0.12 * cm),
    ]


def table_from_rows(
    rows: list[list[str]],
    col_widths: list[float] | None = None,
    font_size: int = 7,
) -> Table:
    wrapped = []

    for row in rows:
        wrapped.append([
            Paragraph(
                str(cell).replace("&", "&amp;"),
                ParagraphStyle(
                    name="TableCell",
                    fontSize=font_size,
                    leading=font_size + 2,
                    alignment=TA_LEFT,
                ),
            )
            for cell in row
        ])

    table = Table(
        wrapped,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BDBDBD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    return table


def add_table(
    story: list,
    title: str,
    rows: list[list[str]] | None,
    styles,
    col_widths: list[float] | None = None,
) -> None:
    if not rows:
        return

    if len(rows) > MAX_TABLE_ROWS + 1:
        n_cols = len(rows[0])
        rows = rows[:MAX_TABLE_ROWS + 1] + [
            ["..."] + ["truncated in PDF report"] + [""] * max(0, n_cols - 2)
        ]

    story.append(Paragraph(title, styles["Heading2"]))
    story.append(table_from_rows(rows, col_widths=col_widths))
    story.append(Spacer(1, 0.25 * cm))


def add_image(
    story: list,
    path: Path | None,
    title: str,
    styles,
    width_cm: float = 16.0,
) -> None:
    if path is None or not path.exists():
        return

    story.append(KeepTogether([
        Paragraph(title, styles["Heading2"]),
        Image(
            str(path),
            width=width_cm * cm,
            height=(width_cm * 0.55) * cm,
        ),
        Spacer(1, 0.2 * cm),
    ]))


# =========================================================
# Report assembly
# =========================================================

def build_story(
    data: LoadedData,
    figure_paths: dict[str, Path | None],
) -> list:
    styles = make_styles()
    story = []

    story.append(
        Paragraph(
            "Ambulance OSRM Calibration - Results Report",
            styles["Title"],
        )
    )

    story.append(Spacer(1, 0.25 * cm))

    story.append(paragraph(
        "Automatically generated from the active pipeline run. "
        "Tables are the primary evidence; figures are limited to compact "
        "diagnostic summaries.",
        styles["Note"],
    ))

    story.append(Spacer(1, 0.35 * cm))

    story.extend(section("1. Executive summary", styles))
    add_table(
        story,
        "Run-level summary",
        build_executive_summary(data),
        styles,
        col_widths=[7 * cm, 8 * cm],
    )

    story.extend(section("2. Retention and rejection diagnostics", styles))
    add_table(
        story,
        "Pipeline retention overview",
        build_retention_table(data),
        styles,
        col_widths=[5 * cm, 3 * cm, 8 * cm],
    )
    add_table(
        story,
        "Trip reconstruction rejection reasons",
        build_trip_rejection_table(data.trip_rejections),
        styles,
        col_widths=[5.5 * cm, 2.5 * cm, 3.5 * cm, 4 * cm],
    )
    add_table(
        story,
        "Map-matching rejection reasons",
        build_route_rejection_table(data.route_rejections),
        styles,
        col_widths=[7 * cm, 3 * cm, 5 * cm],
    )

    story.extend(section("3. Calibration performance", styles))
    add_table(
        story,
        "Baseline, calibrated, and multiplier comparison",
        build_performance_table(data.calibration_metrics),
        styles,
    )
    add_image(
        story,
        figure_paths.get("model_comparison"),
        "Figure: MAE by model",
        styles,
    )
    add_image(
        story,
        figure_paths.get("error_distribution"),
        "Figure: signed prediction error distribution",
        styles,
    )

    story.append(PageBreak())
    story.extend(section("4. Regression and calibration interpretation", styles))
    add_table(
        story,
        "Largest regression coefficients",
        build_regression_table(data.regression_coefficients),
        styles,
    )
    add_table(
        story,
        "Applied OSRM profile changes",
        build_calibration_changes_table(data.calibration_changes),
        styles,
    )
    add_image(
        story,
        figure_paths.get("calibration_changes"),
        "Figure: road-speed parameter changes",
        styles,
    )

    story.extend(section("5. Route diagnostics", styles))
    add_table(
        story,
        "Calibrated route diagnostics",
        build_route_diagnostics_table(data.route_diagnostics),
        styles,
    )
    add_table(
        story,
        "Unknown road-class assignment diagnostic",
        build_km_unknown_table(data.route_features),
        styles,
        col_widths=[8 * cm, 5 * cm],
    )

    story.extend(section("6. Interpretation guardrails", styles))

    guardrails = [
        "Calibration-sample performance is a fit check, not independent validation.",
        "Off-sample validation, transferability, and sensitivity analyses are not part of this core report.",
        "The multiplier benchmark is useful as a simple comparator, but it is not an interpretable routing-engine calibration.",
        "Unknown road-class distance should be inspected when km_unknown is non-negligible.",
    ]

    for text in guardrails:
        story.append(paragraph(f"- {text}", styles["BodyText"]))

    return story


def build_pdf(data: LoadedData) -> None:
    ensure_dirs()

    figure_paths = build_figures(data)
    story = build_story(data, figure_paths)

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
        title="Ambulance OSRM Calibration Results Report",
    )

    doc.build(story)

    print(f"Saved: {REPORT_PDF}")


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    ensure_dirs()
    data = load_all_data()
    build_pdf(data)


if __name__ == "__main__":
    main()