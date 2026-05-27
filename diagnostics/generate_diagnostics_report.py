"""
diagnostics/generate_diagnostics_report.py

Generate a compact PDF/Markdown diagnostics report from the diagnostic CSVs
created by the four phase-level diagnostic scripts.

Inputs:
    runs/RunXXX/results/diagnostics/*.csv
    runs/RunXXX/diagnostics/*.csv

Outputs:
    runs/RunXXX/report/diagnostics_report.pdf
    runs/RunXXX/report/diagnostics_report.md

Design:
    - Does not call OSRM.
    - Does not rerun methodology scripts.
    - Does not modify pipeline outputs.
    - Only reads existing diagnostic CSVs and packages them into a report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Any
import sys

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
)


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paths import REPORT_DIR, RESULTS_DIR, RUN_DIR  # noqa: E402

# =========================================================
# Settings
# =========================================================

REPORT_PDF = REPORT_DIR / "diagnostics_report.pdf"
REPORT_MD = REPORT_DIR / "diagnostics_report.md"

RESULTS_DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
RUN_DIAGNOSTICS_DIR = RUN_DIR / "diagnostics"

MAX_TABLE_ROWS = 22
MAX_TABLE_COLS = 8
MAX_CELL_CHARS = 52


# =========================================================
# Diagnostic report structure
# =========================================================

@dataclass(frozen=True)
class DiagnosticFile:
    phase: str
    filename: str
    title: str
    note: str


DIAGNOSTIC_FILES: list[DiagnosticFile] = [
    DiagnosticFile(
        "Phase 1 — realised travel-time reconstruction",
        "phase1_retention_summary.csv",
        "Trip reconstruction retention",
        "Counts candidate trips, kept trips, and rejected trips after start/arrival reconstruction.",
    ),
    DiagnosticFile(
        "Phase 1 — realised travel-time reconstruction",
        "phase1_rejection_reasons.csv",
        "Trip reconstruction rejection reasons",
        "Separates hard reconstruction failures from diagnostic quality indicators.",
    ),
    DiagnosticFile(
        "Phase 1 — realised travel-time reconstruction",
        "phase1_arrival_distance_thresholds.csv",
        "Arrival-distance threshold diagnostics",
        "Shows how many trips would reach alternative incident-radius thresholds.",
    ),
    DiagnosticFile(
        "Phase 1 — realised travel-time reconstruction",
        "phase1_realised_travel_time_summary.csv",
        "Realised travel-time distribution",
        "Summarises the reconstructed dispatch-to-arrival travel-time distribution.",
    ),
    DiagnosticFile(
        "Phase 1 — realised travel-time reconstruction",
        "phase1_gps_quality_summary.csv",
        "GPS quality diagnostics",
        "Summarises observation counts and gap-related indicators on reconstructed trips.",
    ),
    DiagnosticFile(
        "Phase 1 — realised travel-time reconstruction",
        "phase1_coordinate_sanity_checks.csv",
        "Coordinate sanity checks",
        "Flags implausible GPS or incident coordinates and trips far from the incident.",
    ),

    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_route_matching_summary.csv",
        "Route-matching retention",
        "Compares clean trajectories submitted to OSRM Match with successful and rejected routes.",
    ),
    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_route_rejection_categories.csv",
        "Route-matching rejection categories",
        "Groups rejected OSRM Match traces by human-readable rejection category and explanation.",
    ),
    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_route_rejection_reasons.csv",
        "Route-matching rejection reasons",
        "Summarises raw OSRM Match statuses together with clearer rejection metadata where available.",
    ),
    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_match_confidence_summary.csv",
        "Map-matching confidence",
        "Shows confidence distributions and retention under alternative confidence thresholds.",
    ),
    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_trace_quality_summary.csv",
        "Trace quality submitted to OSRM",
        "Summarises submitted points, trace duration, trace gaps, and request sizes.",
    ),
    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_route_feature_summary.csv",
        "Route-feature construction",
        "Reports total and per-trip kilometres by road class.",
    ),
    DiagnosticFile(
        "Phase 2 — route matching and baseline prediction",
        "phase2_baseline_error_summary.csv",
        "Baseline prediction-error diagnostics",
        "Summarises uncalibrated OSRM signed and absolute prediction errors.",
    ),

    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_regression_sample_summary.csv",
        "Regression sample construction",
        "Checks row counts from prediction errors and route features into the regression table.",
    ),
    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_variable_missingness.csv",
        "Regression variable missingness",
        "Reports missing values across the regression dataset.",
    ),
    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_regression_variable_summary.csv",
        "Regression variable distributions",
        "Summarises the dependent variable and route-feature regressors.",
    ),
    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_road_class_sparsity.csv",
        "Road-class sparsity",
        "Shows how frequently each road class appears in the regression sample.",
    ),
    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_coefficient_diagnostics.csv",
        "Coefficient diagnostics",
        "Flags sparse, large, weak, or sign-uncertain coefficient estimates.",
    ),
    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_calibration_change_diagnostics.csv",
        "Calibration-change diagnostics",
        "Summarises translated OSRM speed and turn-penalty changes.",
    ),
    DiagnosticFile(
        "Phase 3 — residual modelling and profile calibration",
        "phase3_parameter_warnings.csv",
        "Parameter warnings",
        "Reports speed caps, large speed changes, sparse variables, and large coefficient magnitudes.",
    ),

    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_prediction_coverage.csv",
        "Calibrated prediction coverage",
        "Checks whether trips are lost between baseline prediction, calibrated prediction, and evaluation.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_error_distribution_summary.csv",
        "Error-distribution comparison",
        "Compares baseline and calibrated signed/absolute prediction-error distributions.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_error_improvement_summary.csv",
        "Error improvement",
        "Reports MAE/RMSE changes and trip-level improvement/worsening counts.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_standard_misclassification_summary.csv",
        "15-minute-standard misclassification",
        "Checks whether calibration fixes or introduces 15-minute-standard classification errors.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_calibrated_route_quality.csv",
        "Calibrated route quality",
        "Summarises calibrated OSRM match confidence, distances, and trace-quality indicators.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_extreme_error_cases.csv",
        "Extreme error cases",
        "Lists the trips where calibration worsened absolute error the most.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_naive_multiplier_fixed_benchmarks.csv",
        "Fixed naive multiplier benchmarks",
        "Reports performance for the fixed 0.70, 0.80, and 0.90 multiplier corrections.",
    ),
    DiagnosticFile(
        "Phase 4 — calibrated prediction and evaluation",
        "phase4_calibrated_vs_fixed_multipliers.csv",
        "Calibrated profile versus fixed multiplier benchmarks",
        "Compares calibrated OSRM against the fixed simple multiplier alternatives.",
    ),
]


# =========================================================
# Helpers
# =========================================================

def expected_input_dirs() -> list[Path]:
    dirs: list[Path] = []

    for candidate in [RESULTS_DIAGNOSTICS_DIR, RUN_DIAGNOSTICS_DIR]:
        if candidate is None:
            continue

        candidate = Path(candidate)

        if candidate not in dirs:
            dirs.append(candidate)

    return dirs


def ordered_input_dirs() -> list[Path]:
    return [
        path
        for path in expected_input_dirs()
        if path.exists() and path.is_dir()
    ]


def find_diagnostic_file(filename: str) -> Path | None:
    for directory in ordered_input_dirs():
        path = directory / filename
        if path.exists():
            return path
    return None


def safe_read_csv(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Could not read diagnostic CSV {path}: {exc}") from exc


def to_finite_float(value: Any) -> float | None:
    """
    Convert a value to a finite float if possible.

    Returns None for missing values, booleans, non-numeric strings,
    NaN, and infinity.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(x) or math.isinf(x):
        return None

    return x


def fmt(value: Any, digits: int = 3) -> str:
    """
    Format values safely for Markdown/PDF tables.
    """
    if value is None:
        return "-"

    if isinstance(value, bool):
        return "true" if value else "false"

    x = to_finite_float(value)

    if x is not None:
        if abs(x) >= 1000:
            return f"{x:,.0f}"
        if x.is_integer():
            return str(int(x))
        return f"{x:.{digits}f}"

    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def shrink_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    if df.width > MAX_TABLE_COLS:
        df = df.select(df.columns[:MAX_TABLE_COLS])

    if df.height > MAX_TABLE_ROWS:
        df = df.head(MAX_TABLE_ROWS)

    return df


def table_rows_from_df(df: pl.DataFrame) -> list[list[str]]:
    df_small = shrink_dataframe(df)
    rows = [[fmt(col) for col in df_small.columns]]

    for row in df_small.iter_rows(named=True):
        rows.append([fmt(row.get(col)) for col in df_small.columns])

    if df.height > MAX_TABLE_ROWS or df.width > MAX_TABLE_COLS:
        rows.append([
            "…",
            f"truncated in PDF; source has {df.height} rows and {df.width} columns",
            *["" for _ in range(max(0, len(rows[0]) - 2))],
        ])

    return rows


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

    styles.add(ParagraphStyle(
        name="TableCell",
        parent=styles["BodyText"],
        fontSize=6.6,
        leading=8,
        alignment=TA_LEFT,
    ))

    styles["Title"].fontSize = 18
    styles["Heading1"].fontSize = 13
    styles["Heading2"].fontSize = 10

    return styles


def escape_text(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def para(text: object, style) -> Paragraph:
    return Paragraph(escape_text(text), style)


def build_table(rows: list[list[str]]) -> Table:
    styles = make_styles()
    wrapped = [
        [para(cell, styles["TableCell"]) for cell in row]
        for row in rows
    ]

    n_cols = len(rows[0]) if rows else 1
    available_width = 18.0 * cm
    col_width = available_width / max(n_cols, 1)

    table = Table(
        wrapped,
        colWidths=[col_width] * n_cols,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAEAEA")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BDBDBD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))

    return table


# =========================================================
# Loading and summary
# =========================================================

@dataclass
class LoadedDiagnostic:
    spec: DiagnosticFile
    path: Path | None
    dataframe: pl.DataFrame | None
    error: str | None = None


def load_diagnostics() -> list[LoadedDiagnostic]:
    loaded: list[LoadedDiagnostic] = []

    for spec in DIAGNOSTIC_FILES:
        path = find_diagnostic_file(spec.filename)

        if path is None:
            loaded.append(LoadedDiagnostic(spec=spec, path=None, dataframe=None))
            continue

        try:
            df = safe_read_csv(path)
            loaded.append(LoadedDiagnostic(spec=spec, path=path, dataframe=df))
        except Exception as exc:
            loaded.append(
                LoadedDiagnostic(
                    spec=spec,
                    path=path,
                    dataframe=None,
                    error=str(exc),
                )
            )

    return loaded


def build_inventory(loaded: list[LoadedDiagnostic]) -> pl.DataFrame:
    rows = []

    for item in loaded:
        rows.append({
            "phase": item.spec.phase,
            "file": item.spec.filename,
            "status": (
                "missing" if item.path is None
                else "read_error" if item.error
                else "available"
            ),
            "rows": item.dataframe.height if item.dataframe is not None else None,
            "columns": item.dataframe.width if item.dataframe is not None else None,
            "path": str(item.path) if item.path else "",
            "message": item.error or "",
        })

    return pl.DataFrame(rows)


def phase_counts(inventory: pl.DataFrame) -> pl.DataFrame:
    return (
        inventory
        .group_by("phase", "status")
        .agg(pl.len().alias("n_files"))
        .sort(["phase", "status"])
    )


# =========================================================
# Markdown output
# =========================================================

def dataframe_to_markdown(df: pl.DataFrame, max_rows: int = MAX_TABLE_ROWS) -> str:
    df_small = df.head(max_rows)
    columns = df_small.columns

    if not columns:
        return ""

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")

    for row in df_small.iter_rows(named=True):
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")

    if df.height > max_rows:
        lines.append(f"\n_Table truncated in Markdown. Source has {df.height} rows._")

    return "\n".join(lines)


def write_markdown_report(loaded: list[LoadedDiagnostic]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(loaded)
    counts = phase_counts(inventory)

    lines: list[str] = []
    lines.append("# Diagnostics Report")
    lines.append("")
    lines.append(
        "This report packages the diagnostic CSV files produced by the "
        "phase-level diagnostic scripts. It does not rerun the pipeline and "
        "does not modify outputs."
    )
    lines.append("")

    lines.append("## Diagnostic file inventory")
    lines.append("")
    lines.append(
        dataframe_to_markdown(
            inventory.select(["phase", "file", "status", "rows", "columns"]),
            max_rows=100,
        )
    )
    lines.append("")

    lines.append("## Availability by phase")
    lines.append("")
    lines.append(dataframe_to_markdown(counts, max_rows=100))
    lines.append("")

    current_phase = None

    for item in loaded:
        if item.spec.phase != current_phase:
            current_phase = item.spec.phase
            lines.append(f"## {current_phase}")
            lines.append("")

        lines.append(f"### {item.spec.title}")
        lines.append("")
        lines.append(item.spec.note)
        lines.append("")

        if item.path is None:
            lines.append(f"Missing diagnostic file: `{item.spec.filename}`")
            lines.append("")
            continue

        if item.error:
            lines.append(f"Could not read `{item.path}`: {item.error}")
            lines.append("")
            continue

        assert item.dataframe is not None
        lines.append(f"Source: `{item.path}`")
        lines.append("")
        lines.append(
            dataframe_to_markdown(
                shrink_dataframe(item.dataframe),
                max_rows=MAX_TABLE_ROWS,
            )
        )
        lines.append("")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {REPORT_MD}")


# =========================================================
# PDF output
# =========================================================

def add_table(
    story: list,
    title: str,
    df: pl.DataFrame | None,
    styles,
    note: str | None = None,
) -> None:
    story.append(Paragraph(escape_text(title), styles["Heading2"]))

    if note:
        story.append(para(note, styles["Note"]))
        story.append(Spacer(1, 0.08 * cm))

    if df is None:
        story.append(para("No table available.", styles["Note"]))
        story.append(Spacer(1, 0.18 * cm))
        return

    rows = table_rows_from_df(df)
    story.append(build_table(rows))
    story.append(Spacer(1, 0.25 * cm))


def build_pdf_report(loaded: list[LoadedDiagnostic]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    styles = make_styles()
    inventory = build_inventory(loaded)
    counts = phase_counts(inventory)

    story: list = []

    story.append(
        Paragraph(
            "Ambulance OSRM Calibration — Diagnostics Report",
            styles["Title"],
        )
    )
    story.append(Spacer(1, 0.25 * cm))
    story.append(para(
        "Automatically generated from existing phase-level diagnostic CSV files. "
        "This report is an audit artefact: it does not rerun OSRM, recompute "
        "predictions, or modify methodology outputs.",
        styles["Note"],
    ))
    story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("1. Diagnostic file inventory", styles["Heading1"]))

    add_table(
        story,
        "Availability by phase",
        counts,
        styles,
    )

    add_table(
        story,
        "Expected diagnostic files",
        inventory.select(["phase", "file", "status", "rows", "columns"]),
        styles,
    )

    current_phase = None
    phase_index = 1

    for item in loaded:
        if item.spec.phase != current_phase:
            current_phase = item.spec.phase
            phase_index += 1
            story.append(PageBreak())
            story.append(
                Paragraph(
                    f"{phase_index}. {current_phase}",
                    styles["Heading1"],
                )
            )
            story.append(Spacer(1, 0.15 * cm))

        if item.path is None:
            story.append(Paragraph(escape_text(item.spec.title), styles["Heading2"]))
            story.append(
                para(
                    f"Missing diagnostic file: {item.spec.filename}",
                    styles["Note"],
                )
            )
            story.append(Spacer(1, 0.2 * cm))
            continue

        if item.error:
            story.append(Paragraph(escape_text(item.spec.title), styles["Heading2"]))
            story.append(
                para(
                    f"Could not read {item.path}: {item.error}",
                    styles["Note"],
                )
            )
            story.append(Spacer(1, 0.2 * cm))
            continue

        assert item.dataframe is not None
        add_table(
            story,
            item.spec.title,
            item.dataframe,
            styles,
            note=f"{item.spec.note} Source: {item.path.name}",
        )

    doc = SimpleDocTemplate(
        str(REPORT_PDF),
        pagesize=A4,
        rightMargin=1.0 * cm,
        leftMargin=1.0 * cm,
        topMargin=1.1 * cm,
        bottomMargin=1.1 * cm,
        title="Ambulance OSRM Calibration Diagnostics Report",
    )

    doc.build(story)
    print(f"Saved: {REPORT_PDF}")


# =========================================================
# Main
# =========================================================

def main() -> None:
    print("Loading diagnostic CSV files...")

    print("Expected search locations:")
    for directory in expected_input_dirs():
        exists = "exists" if directory.exists() else "missing"
        print(f"  - {directory} [{exists}]")

    input_dirs = ordered_input_dirs()

    if not input_dirs:
        expected = "\n".join(f"  - {path}" for path in expected_input_dirs())
        raise FileNotFoundError(
            "No diagnostics directory was found for the active run.\n\n"
            f"Expected one of:\n{expected}\n\n"
            "Run the phase-level diagnostic scripts first, or set ACTIVE_RUN_DIR "
            "to the run that already contains diagnostic CSVs."
        )

    print("Using search locations:")
    for directory in input_dirs:
        print(f"  - {directory}")

    loaded = load_diagnostics()
    inventory = build_inventory(loaded)

    available = inventory.filter(pl.col("status") == "available").height
    missing = inventory.filter(pl.col("status") == "missing").height
    read_errors = inventory.filter(pl.col("status") == "read_error").height

    if available == 0:
        raise FileNotFoundError(
            "No diagnostic CSV files were found in the diagnostics directory "
            "for the active run. Run the phase-level diagnostic scripts first."
        )

    print(
        f"Diagnostic inventory: {available} available, "
        f"{missing} missing, {read_errors} read errors."
    )

    write_markdown_report(loaded)
    build_pdf_report(loaded)


if __name__ == "__main__":
    main()