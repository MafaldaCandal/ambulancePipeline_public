"""
generate_extended_tests_report.py

Generate a compact PDF and combined CSV from extended-test summaries.

Inputs:
    runs/RunXXX/extended_tests/temporal_validation/temporal_validation_summary.csv
    runs/RunXXX/extended_tests/spatial_transferability/spatial_transferability_summary.csv
    runs/RunXXX/extended_tests/static_profile_expressiveness/static_profile_expressiveness_summary.csv
    runs/RunXXX/extended_tests/sensitivity_analysis/sensitivity_summary.csv
    runs/RunXXX/extended_tests/sensitivity_analysis/preprocessing_diagnostics.csv

Outputs:
    runs/RunXXX/extended_tests/extended_tests_summary.csv
    runs/RunXXX/report/extended_tests_report.pdf
    runs/RunXXX/report/extended_tests_figures/*.png
"""

from __future__ import annotations

from pathlib import Path
import html
import math

import matplotlib.pyplot as plt
import polars as pl

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from extended_test_utils import EXTENDED_ROOT, MAIN_RUN_DIR


# =========================================================
# Paths
# =========================================================

INPUT_SUMMARIES = [
    EXTENDED_ROOT / "temporal_validation" / "temporal_validation_summary.csv",
    EXTENDED_ROOT / "spatial_transferability" / "spatial_transferability_summary.csv",
    EXTENDED_ROOT / "static_profile_expressiveness" / "static_profile_expressiveness_summary.csv",
    EXTENDED_ROOT / "sensitivity_analysis" / "sensitivity_summary.csv",
]

PREPROCESSING_DIAGNOSTICS_CSV = (
    EXTENDED_ROOT / "sensitivity_analysis" / "preprocessing_diagnostics.csv"
)

COMBINED_SUMMARY_CSV = EXTENDED_ROOT / "extended_tests_summary.csv"
REPORT_PDF = MAIN_RUN_DIR / "report" / "extended_tests_report.pdf"
FIGURE_DIR = MAIN_RUN_DIR / "report" / "extended_tests_figures"


# =========================================================
# Formatting
# =========================================================

MODEL_LABELS = {
    "baseline_osrm": "Baseline OSRM",
    "calibrated_osrm": "Calibrated OSRM",
    "main_calibrated_profile": "Main calibrated profile",
    "best_naive_multiplier": "Best naive multiplier",
    "naive_multiplier": "Naive multiplier",
    "context_adjusted_correction": "Context-adjusted correction",
    "peak_time_calibrated_profile": "Peak-time calibrated profile",
}

TEST_LABELS = {
    "temporal_validation": "Temporal validation",
    "spatial_transferability": "Spatial transferability",
    "static_profile_expressiveness": "Static-profile expressiveness",
    "sensitivity_analysis": "Sensitivity analysis",
}


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
    x = safe_float(value)

    if x is None:
        if value is None:
            return "-"
        text = str(value)
        return text if text else "-"

    if percent:
        return f"{100 * x:.{digits}f}%"

    if abs(x) >= 1000:
        return f"{x:,.0f}"

    return f"{x:.{digits}f}"


def fmt_percent_points(value, digits: int = 1) -> str:
    x = safe_float(value)
    if x is None:
        return "-"
    return f"{100 * x:+.{digits}f} pp"


def fmt_change_percent(value, digits: int = 1) -> str:
    x = safe_float(value)
    if x is None:
        return "-"
    return f"{x:+.{digits}f}%"


def display_model(value) -> str:
    text = str(value or "-")
    return MODEL_LABELS.get(text, text.replace("_", " ").title())


def display_test_type(value) -> str:
    text = str(value or "-")
    return TEST_LABELS.get(text, text.replace("_", " ").title())


def display_variant(value) -> str:
    text = str(value or "-")
    return text.replace("_", " ")


def short_text(value, width: int = 80) -> str:
    text = str(value or "-")
    return text if len(text) <= width else text[: width - 1] + "…"


def paragraph(text: str, style) -> Paragraph:
    return Paragraph(html.escape(str(text)), style)


# =========================================================
# Data loading
# =========================================================

def read_summaries() -> pl.DataFrame:
    frames = []

    for path in INPUT_SUMMARIES:
        if path.exists():
            frames.append(pl.read_csv(path))
        else:
            print(f"Missing optional summary: {path}")

    if not frames:
        raise FileNotFoundError("No extended-test summary CSVs were found.")

    return pl.concat(frames, how="diagonal_relaxed")


def read_preprocessing_diagnostics() -> pl.DataFrame | None:
    if not PREPROCESSING_DIAGNOSTICS_CSV.exists():
        return None
    return pl.read_csv(PREPROCESSING_DIAGNOSTICS_CSV)


# =========================================================
# ReportLab styles and tables
# =========================================================

def make_styles():
    styles = getSampleStyleSheet()
    styles["Title"].fontSize = 18
    styles["Heading1"].fontSize = 13
    styles["Heading2"].fontSize = 11

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

    return styles


def table_from_rows(rows: list[list[str]], font_size: int = 7) -> Table:
    cell_style = ParagraphStyle(
        name="TableCell",
        fontSize=font_size,
        leading=font_size + 2,
        alignment=TA_LEFT,
    )

    wrapped = [
        [Paragraph(html.escape(str(cell)), cell_style) for cell in row]
        for row in rows
    ]

    table = Table(wrapped, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BDBDBD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    return table


def add_table(story: list, title: str, rows: list[list[str]], styles) -> None:
    if not rows or len(rows) <= 1:
        return

    story.append(Paragraph(title, styles["Heading2"]))
    story.append(table_from_rows(rows))
    story.append(Spacer(1, 0.25 * cm))


def add_image(story: list, path: Path | None, title: str, styles) -> None:
    if path is None or not path.exists():
        return

    story.append(KeepTogether([
        Paragraph(title, styles["Heading2"]),
        Image(str(path), width=16 * cm, height=8.8 * cm),
        Spacer(1, 0.2 * cm),
    ]))


# =========================================================
# Table rows
# =========================================================

def executive_rows(df: pl.DataFrame) -> list[list[str]]:
    rows = [["Item", "Value"]]

    rows.append(["Combined summary rows", fmt(df.height, 0)])

    if "status" in df.columns:
        rows.append([
            "Conducted rows",
            fmt(df.filter(pl.col("status") == "conducted").height, 0),
        ])
        rows.append([
            "Failed/skipped/missing rows",
            fmt(df.filter(pl.col("status") != "conducted").height, 0),
        ])

    for test_type in [
        "temporal_validation",
        "spatial_transferability",
        "static_profile_expressiveness",
        "sensitivity_analysis",
    ]:
        sub = (
            df.filter(pl.col("test_type") == test_type)
            if "test_type" in df.columns
            else pl.DataFrame()
        )
        rows.append([display_test_type(test_type), fmt(sub.height, 0)])

    return rows


def performance_rows(df: pl.DataFrame, test_type: str) -> list[list[str]]:
    if "test_type" not in df.columns:
        return []

    sub = df.filter(pl.col("test_type") == test_type)
    if sub.is_empty():
        return []

    keep_models = [
        "baseline_osrm",
        "calibrated_osrm",
        "main_calibrated_profile",
        "best_naive_multiplier",
        "naive_multiplier",
    ]

    if "model" in sub.columns:
        sub = sub.filter(pl.col("model").is_in(keep_models))

    if sub.is_empty():
        return []

    rows = [[
        "Dataset",
        "Region",
        "Variant",
        "Model",
        "N",
        "MAE",
        "Median AE",
        "RMSE",
        ">5 min",
        "15-min miscl.",
        "MAE Δ%",
        "Status",
    ]]

    for row in sub.iter_rows(named=True):
        rows.append([
            str(row.get("dataset") or "-"),
            str(row.get("region") or "-"),
            display_variant(row.get("variant")),
            display_model(row.get("model")),
            fmt(row.get("n_trips"), 0),
            fmt(row.get("mae_sec"), 1),
            fmt(row.get("median_ae_sec"), 1),
            fmt(row.get("rmse_sec"), 1),
            fmt(row.get("share_error_gt_5min"), 1, True),
            fmt(row.get("share_15min_misclassified"), 1, True),
            fmt_change_percent(row.get("mae_change_percent")),
            str(row.get("status") or "-"),
        ])

    return rows


def static_rows(df: pl.DataFrame) -> list[list[str]]:
    if "test_type" not in df.columns:
        return []

    sub = df.filter(pl.col("test_type") == "static_profile_expressiveness")
    if sub.is_empty():
        return []

    rows = [[
        "Dataset",
        "Region",
        "Variant",
        "Model",
        "N",
        "MAE",
        "Median AE",
        "RMSE",
        ">5 min",
        "15-min miscl.",
        "MAE Δ%",
        "Status",
    ]]

    for row in sub.iter_rows(named=True):
        rows.append([
            str(row.get("dataset") or "-"),
            str(row.get("region") or "-"),
            display_variant(row.get("variant")),
            display_model(row.get("model")),
            fmt(row.get("n_trips"), 0),
            fmt(row.get("mae_sec"), 1),
            fmt(row.get("median_ae_sec"), 1),
            fmt(row.get("rmse_sec"), 1),
            fmt(row.get("share_error_gt_5min"), 1, True),
            fmt(row.get("share_15min_misclassified"), 1, True),
            fmt_change_percent(row.get("mae_change_percent")),
            str(row.get("status") or "-"),
        ])

    return rows


def sensitivity_rows(df: pl.DataFrame) -> list[list[str]]:
    if "test_type" not in df.columns:
        return []

    sub = df.filter(pl.col("test_type") == "sensitivity_analysis")
    if sub.is_empty():
        return []

    if "model" in sub.columns:
        sub = sub.filter(pl.col("model").is_in([
            "calibrated_osrm",
            "main_calibrated_profile",
        ]))

    rows = [[
        "Dimension",
        "Value",
        "Variant",
        "Model",
        "N",
        "MAE",
        "Median AE",
        "RMSE",
        ">5 min",
        "15-min miscl.",
        "Status",
    ]]

    for row in sub.iter_rows(named=True):
        rows.append([
            str(row.get("sensitivity_dimension") or "-"),
            str(row.get("sensitivity_value") or "-"),
            display_variant(row.get("variant")),
            display_model(row.get("model")),
            fmt(row.get("n_trips"), 0),
            fmt(row.get("mae_sec"), 1),
            fmt(row.get("median_ae_sec"), 1),
            fmt(row.get("rmse_sec"), 1),
            fmt(row.get("share_error_gt_5min"), 1, True),
            fmt(row.get("share_15min_misclassified"), 1, True),
            str(row.get("status") or "-"),
        ])

    return rows


def preprocessing_diagnostic_rows(df: pl.DataFrame | None) -> list[list[str]]:
    if df is None or df.is_empty():
        return []

    rows = [[
        "Diagnostic",
        "Main value",
        "Alternative values",
        "Rerun type",
        "Message",
    ]]

    for row in df.iter_rows(named=True):
        rows.append([
            str(row.get("diagnostic") or "-"),
            str(row.get("main_value") or "-"),
            str(row.get("alternative_values") or "-"),
            str(row.get("rerun_type") or "-"),
            short_text(row.get("message"), 95),
        ])

    return rows


# =========================================================
# Figures
# =========================================================

def save_transferability_figure(df: pl.DataFrame) -> Path | None:
    required = {"test_type", "region", "model", "mae_sec"}
    if not required.issubset(set(df.columns)):
        return None

    sub = (
        df
        .filter(pl.col("test_type") == "spatial_transferability")
        .filter(pl.col("model").is_in(["baseline_osrm", "calibrated_osrm"]))
    )

    if sub.is_empty():
        return None

    pivot = sub.pivot(
        values="mae_sec",
        index="region",
        columns="model",  # type: ignore[call-arg]
        aggregate_function="first",
    )  # type: ignore[call-arg]

    if not {"baseline_osrm", "calibrated_osrm"}.issubset(set(pivot.columns)):
        return None

    pivot = (
        pivot
        .with_columns(
            (
                (pl.col("calibrated_osrm") - pl.col("baseline_osrm"))
                / pl.col("baseline_osrm")
                * 100
            ).alias("mae_change_percent")
        )
        .sort("region")
    )

    labels = [str(x) for x in pivot["region"].to_list()]
    values = [float(x) for x in pivot["mae_change_percent"].to_list()]

    if not values:
        return None

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "transferability_mae_change.png"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("MAE change vs baseline (%)")
    ax.set_title("Spatial transferability: calibrated profile vs baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


def save_sensitivity_figure(df: pl.DataFrame) -> Path | None:
    required = {"test_type", "variant", "model", "mae_sec"}
    if not required.issubset(set(df.columns)):
        return None

    sub = (
        df
        .filter(pl.col("test_type") == "sensitivity_analysis")
        .filter(pl.col("model").is_in(["calibrated_osrm", "main_calibrated_profile"]))
    )

    if sub.is_empty():
        return None

    if "sensitivity_dimension" in sub.columns:
        sub = sub.with_columns(
            (
                pl.col("sensitivity_dimension").cast(pl.Utf8)
                + ": "
                + pl.col("variant").cast(pl.Utf8)
            ).alias("_label")
        )
        label_col = "_label"
    else:
        label_col = "variant"

    labels = [display_variant(x) for x in sub[label_col].to_list()]
    values = [float(x) for x in sub["mae_sec"].to_list() if safe_float(x) is not None]

    labels = labels[: len(values)]

    if not values:
        return None

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / "sensitivity_calibrated_mae.png"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values)
    ax.set_ylabel("Calibrated MAE (seconds)")
    ax.set_title("Sensitivity analysis: calibrated MAE")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


# =========================================================
# Story
# =========================================================

def build_story(
    df: pl.DataFrame,
    figures: dict[str, Path | None],
    preprocessing_diagnostics: pl.DataFrame | None,
) -> list:
    styles = make_styles()
    story = []

    story.append(Paragraph("Extended Tests Report", styles["Title"]))
    story.append(Spacer(1, 0.25 * cm))
    story.append(paragraph(
        "Automatically generated from optional extended-test outputs. "
        "These tests are separate from the core four-phase methodology pipeline.",
        styles["Note"],
    ))

    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("1. Executive summary", styles["Heading1"]))
    add_table(story, "Extended-test output overview", executive_rows(df), styles)

    story.append(Paragraph("2. Temporal validation", styles["Heading1"]))
    add_table(
        story,
        "Temporal validation performance",
        performance_rows(df, "temporal_validation"),
        styles,
    )

    story.append(Paragraph("3. Spatial transferability", styles["Heading1"]))
    add_table(
        story,
        "Transferability performance",
        performance_rows(df, "spatial_transferability"),
        styles,
    )
    add_image(
        story,
        figures.get("transferability"),
        "Figure: transferability MAE change",
        styles,
    )

    story.append(PageBreak())

    story.append(Paragraph("4. Static-profile expressiveness", styles["Heading1"]))
    add_table(
        story,
        "Peak-time and context-adjusted comparison",
        static_rows(df),
        styles,
    )

    story.append(Paragraph("5. Sensitivity analysis", styles["Heading1"]))
    add_table(
        story,
        "Full sensitivity reruns, calibrated model rows",
        sensitivity_rows(df),
        styles,
    )
    add_image(
        story,
        figures.get("sensitivity"),
        "Figure: sensitivity calibrated MAE",
        styles,
    )

    add_table(
        story,
        "Diagnostic-only preprocessing checks",
        preprocessing_diagnostic_rows(preprocessing_diagnostics),
        styles,
    )

    return story


# =========================================================
# Main
# =========================================================

def main() -> None:
    df = read_summaries()
    preprocessing_diagnostics = read_preprocessing_diagnostics()

    COMBINED_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(COMBINED_SUMMARY_CSV)

    figures = {
        "transferability": save_transferability_figure(df),
        "sensitivity": save_sensitivity_figure(df),
    }

    REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
        title="Extended Tests Report",
    )

    doc.build(build_story(df, figures, preprocessing_diagnostics))

    print(f"Saved combined summary: {COMBINED_SUMMARY_CSV}")
    print(f"Saved PDF report:      {REPORT_PDF}")

    if preprocessing_diagnostics is not None:
        print(f"Included preprocessing diagnostics: {PREPROCESSING_DIAGNOSTICS_CSV}")


if __name__ == "__main__":
    main()