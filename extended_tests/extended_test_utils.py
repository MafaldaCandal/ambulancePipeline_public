"""
extended_test_utils.py

Shared utilities for optional extended tests.

This module is not a methodology phase. It supports extended_tests/ scripts by:
    - resolving the active main run from ACTIVE_RUN_DIR,
    - reading inferred extended-test datasets from orchestrator environment vars,
    - creating isolated extended-test run folders,
    - running methodology scripts as subprocesses with explicit environment vars,
    - summarising Phase 4 evaluation metrics with a stable column contract.

Important design choices:
    - Extended-test inputs are expected under data/input_<NAME>/.
    - Extended tests reuse the calibrated profile from the active main run.
    - Evaluation-only tests do not rerun Phase 3 calibration.
    - The Phase 4 calibrated-prediction script is resolved robustly because older
      project versions used slightly different names.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import polars as pl


# =========================================================
# Project paths and environment
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
METHODOLOGY_DIR = PROJECT_ROOT / "methodology"


def _split_env_list(name: str) -> list[str]:
    value = os.environ.get(name, "").strip()
    if not value:
        return []
    return [
        item.strip().removeprefix("input_")
        for item in value.split(";")
        if item.strip()
    ]


TEMPORAL_VALIDATION_INPUTS = _split_env_list("PIPELINE_TEMPORAL_VALIDATION_INPUTS")
SPATIAL_TRANSFER_INPUTS = _split_env_list("PIPELINE_SPATIAL_TRANSFER_INPUTS")


def latest_run_dir() -> Path:
    if not RUNS_DIR.exists():
        raise RuntimeError(f"Runs directory does not exist: {RUNS_DIR}")

    runs = [
        p for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("Run") and p.name[3:].isdigit()
    ]
    if not runs:
        raise RuntimeError("No runs/RunXXX folder exists and ACTIVE_RUN_DIR is not set.")

    return max(runs, key=lambda p: int(p.name[3:]))


def get_main_run_dir() -> Path:
    active = os.environ.get("ACTIVE_RUN_DIR")
    return Path(active).resolve() if active else latest_run_dir().resolve()


MAIN_RUN_DIR = get_main_run_dir()
EXTENDED_ROOT = MAIN_RUN_DIR / "extended_tests"


# =========================================================
# Methodology script resolution
# =========================================================

def methodology_script(*candidate_names: str) -> Path:
    """
    Return the first existing methodology script from candidate_names.

    This prevents fragile failures when a script was renamed, e.g.
    p4s10_generate_calibrated_predictions.py versus
    p4s10_compute_calibrated_predictions.py.
    """
    checked: list[Path] = []

    for name in candidate_names:
        path = METHODOLOGY_DIR / name
        checked.append(path)
        if path.exists():
            return path

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "None of the expected methodology scripts were found:\n"
        f"{checked_text}"
    )


P1S1_BUILD_CANDIDATE_TRIPS = methodology_script("p1s1_build_candidate_trips.py")
P1S2_IDENTIFY_TRIP_START_END = methodology_script("p1s2_identify_trip_start_end.py")
P2S3_ADJUST_OSRM_ACCESSIBILITY = methodology_script("p2s3_adjust_osrm_accessibility.py")
P2S4_MAP_MATCH_ROUTES = methodology_script("p2s4_map_match_routes.py")
P2S5_EXTRACT_ROUTE_FEATURES = methodology_script("p2s5_extract_route_features.py")
P2S6_COMPUTE_PREDICTION_ERRORS = methodology_script("p2s6_compute_prediction_errors.py")
P3S7_PREPARE_REGRESSION_DATASET = methodology_script("p3s7_prepare_regression_dataset.py")
P3S8_RUN_REGRESSION = methodology_script("p3s8_run_regression.py")
P3S9_CREATE_CALIBRATED_PROFILE = methodology_script("p3s9_create_calibrated_profile.py")
P4S10_CALIBRATED_PREDICTIONS = methodology_script(
    "p4s10_generate_calibrated_predictions.py",
    "p4s10_compute_calibrated_predictions.py",
)
P4S11_EVALUATE_CALIBRATION = methodology_script("p4s11_evaluate_calibration.py")


# =========================================================
# Methodology script sequences
# =========================================================

EVALUATION_ONLY_STEPS: list[tuple[str, Path]] = [
    ("Build candidate trips", P1S1_BUILD_CANDIDATE_TRIPS),
    ("Identify trip start and arrival", P1S2_IDENTIFY_TRIP_START_END),
    ("Adjust OSRM accessibility profile", P2S3_ADJUST_OSRM_ACCESSIBILITY),
    ("Map-match routes", P2S4_MAP_MATCH_ROUTES),
    ("Extract route features", P2S5_EXTRACT_ROUTE_FEATURES),
    ("Compute prediction errors", P2S6_COMPUTE_PREDICTION_ERRORS),
    ("Generate calibrated predictions", P4S10_CALIBRATED_PREDICTIONS),
    ("Evaluate calibration", P4S11_EVALUATE_CALIBRATION),
]

FULL_CALIBRATION_STEPS: list[tuple[str, Path]] = [
    ("Build candidate trips", P1S1_BUILD_CANDIDATE_TRIPS),
    ("Identify trip start and arrival", P1S2_IDENTIFY_TRIP_START_END),
    ("Adjust OSRM accessibility profile", P2S3_ADJUST_OSRM_ACCESSIBILITY),
    ("Map-match routes", P2S4_MAP_MATCH_ROUTES),
    ("Extract route features", P2S5_EXTRACT_ROUTE_FEATURES),
    ("Compute prediction errors", P2S6_COMPUTE_PREDICTION_ERRORS),
    ("Prepare regression dataset", P3S7_PREPARE_REGRESSION_DATASET),
    ("Run regression", P3S8_RUN_REGRESSION),
    ("Create calibrated profile", P3S9_CREATE_CALIBRATED_PROFILE),
    ("Generate calibrated predictions", P4S10_CALIBRATED_PREDICTIONS),
    ("Evaluate calibration", P4S11_EVALUATE_CALIBRATION),
]

PROFILE_FROM_COEFFICIENTS_STEPS: list[tuple[str, Path]] = [
    ("Create calibrated profile", P3S9_CREATE_CALIBRATED_PROFILE),
    ("Generate calibrated predictions", P4S10_CALIBRATED_PREDICTIONS),
    ("Evaluate calibration", P4S11_EVALUATE_CALIBRATION),
]


# =========================================================
# Data classes
# =========================================================

@dataclass(frozen=True)
class ExtendedDataset:
    key: str
    label: str
    region: str
    test_type: str
    input_dir: Path

    @property
    def run_dir(self) -> Path:
        return EXTENDED_ROOT / self.test_type / self.label

    @property
    def pipeline_input_name(self) -> str:
        """
        Name passed to PIPELINE_INPUT_NAME.

        Use label rather than key because labels correspond to data/input_<LABEL>
        folders such as input_ZHZ_2025 or input_BN_2024. Keys may include test
        prefixes such as validation_ or transfer_.
        """
        return normalise_input_name(self.label)


@dataclass(frozen=True)
class StatusRow:
    test_type: str
    dataset: str
    region: str
    variant: str
    run_dir: str
    status: str
    failed_step: str
    return_code: str
    message: str


# =========================================================
# Dataset helpers
# =========================================================

def normalise_input_name(value: str) -> str:
    return str(value).strip().removeprefix("input_")


def input_dir_for(input_name: str) -> Path:
    return DATA_DIR / f"input_{normalise_input_name(input_name)}"


def infer_region(input_name: str) -> str:
    name = normalise_input_name(input_name)

    if name.lower() == "synthetic":
        return "synthetic"

    return name.split("_", 1)[0]


def build_extended_datasets(
    input_names: Iterable[str],
    *,
    test_type: str,
) -> list[ExtendedDataset]:
    datasets: list[ExtendedDataset] = []

    for name in input_names:
        label = normalise_input_name(name)
        datasets.append(
            ExtendedDataset(
                key=label,
                label=label,
                region=infer_region(label),
                test_type=test_type,
                input_dir=input_dir_for(label),
            )
        )

    return datasets


def temporal_validation_datasets() -> list[ExtendedDataset]:
    return build_extended_datasets(
        TEMPORAL_VALIDATION_INPUTS,
        test_type="temporal_validation",
    )


def spatial_transfer_datasets() -> list[ExtendedDataset]:
    return build_extended_datasets(
        SPATIAL_TRANSFER_INPUTS,
        test_type="spatial_transferability",
    )


# =========================================================
# Logging
# =========================================================

def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def print_header(title: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


# =========================================================
# File checks and setup
# =========================================================

def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def find_single_pbf(input_dir: Path) -> Path:
    pbf_files = sorted(input_dir.glob("*.osm.pbf"))

    if len(pbf_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .osm.pbf file in {input_dir}, "
            f"found {len(pbf_files)}."
        )

    return pbf_files[0]


def validate_extended_input(input_dir: Path) -> Path:
    require_dir(input_dir, "extended input directory")
    require_file(input_dir / "dispatch_registers.parquet", "dispatch parquet")

    gps_single = input_dir / "gps_logs.parquet"
    gps_chunks = input_dir / "gps_logs_raw_parquet"

    if not gps_single.exists() and not (
        gps_chunks.exists() and any(gps_chunks.glob("*.parquet"))
    ):
        raise FileNotFoundError(
            "Missing GPS input. Expected either "
            f"{gps_single} or one or more parquet files in {gps_chunks}."
        )

    return find_single_pbf(input_dir)


def ensure_run_subdirs(run_dir: Path, overwrite: bool = True) -> None:
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)

    for name in ["outputs", "results", "diagnostics", "report"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)


def ensure_extended_root() -> None:
    EXTENDED_ROOT.mkdir(parents=True, exist_ok=True)


def copy_main_calibrated_profile(target_run_dir: Path) -> None:
    source = MAIN_RUN_DIR / "ambulance_nl_calibrated.lua"
    require_file(source, "main calibrated Lua profile")

    shutil.copy2(source, target_run_dir / "ambulance_nl_calibrated.lua")

    diff = MAIN_RUN_DIR / "calibrated_profile.diff"
    if diff.exists():
        shutil.copy2(diff, target_run_dir / "calibrated_profile.diff")


def copy_main_access_profile(target_run_dir: Path) -> None:
    source = MAIN_RUN_DIR / "ambulance_nl.lua"
    require_file(source, "main ambulance-access Lua profile")

    shutil.copy2(source, target_run_dir / "ambulance_nl.lua")


# =========================================================
# Subprocess execution
# =========================================================

def build_env(
    run_dir: Path,
    *,
    input_name: str | None = None,
    active_region: str | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ACTIVE_RUN_DIR"] = str(run_dir.resolve())

    if input_name is not None:
        env["PIPELINE_INPUT_NAME"] = normalise_input_name(input_name)

    if active_region is not None:
        env["PIPELINE_ACTIVE_REGION"] = active_region

    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing
        else str(PROJECT_ROOT) + os.pathsep + existing
    )

    return env


def run_step(
    label: str,
    script_path: Path,
    env: dict[str, str],
) -> tuple[bool, str, str]:
    if not script_path.exists():
        raise FileNotFoundError(f"Methodology script not found: {script_path}")

    print_header(f"Running: {label}\nScript: {script_path.relative_to(PROJECT_ROOT)}")

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
    )

    return completed.returncode == 0, str(completed.returncode), script_path.name


def run_steps(
    steps: Iterable[tuple[str, Path]],
    *,
    run_dir: Path,
    input_name: str | None = None,
    active_region: str | None = None,
) -> tuple[bool, str, str, str]:
    env = build_env(
        run_dir,
        input_name=input_name,
        active_region=active_region,
    )

    for label, script_path in steps:
        ok, return_code, script_name = run_step(label, script_path, env)

        if not ok:
            return False, label, return_code, f"Script failed: {script_name}"

    return True, "", "0", "completed"


# =========================================================
# Standard CSV writing
# =========================================================

def write_status_csv(path: Path, rows: list[StatusRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "test_type",
        "dataset",
        "region",
        "variant",
        "run_dir",
        "status",
        "failed_step",
        "return_code",
        "message",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row.__dict__)


def write_manifest(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================
# Metrics summarisation
# =========================================================

STANDARD_SUMMARY_COLUMNS = [
    "test_type",
    "dataset",
    "region",
    "variant",
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
    "mae_change_sec",
    "mae_change_percent",
    "rmse_change_sec",
    "rmse_change_percent",
    "large_error_share_change",
    "standard_misclassification_change",
    "multiplier",
    "status",
    "message",
    "run_dir",
]


def read_csv_if_exists(path: Path) -> pl.DataFrame | None:
    return pl.read_csv(path) if path.exists() else None


def read_parquet_if_exists(path: Path) -> pl.DataFrame | None:
    return pl.read_parquet(path) if path.exists() else None


def _first_existing_col(df: pl.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _safe_float(value: object) -> float | None:
    if value is None:
        return None

    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None

    if out != out or out in (float("inf"), float("-inf")):
        return None

    return out


def model_error_column_candidates(model: str) -> tuple[list[str], list[str]]:
    """
    Return candidate absolute-error and signed-error columns for a model.

    The p4 evaluator has had slightly different column names across versions.
    This function keeps extended-test summaries robust to those small changes.
    """
    if model == "baseline_osrm":
        return (
            [
                "baseline_absolute_error_sec",
                "baseline_abs_error_sec",
                "absolute_error_sec",
            ],
            [
                "baseline_prediction_error_sec",
                "baseline_error_sec",
                "prediction_error_sec",
            ],
        )

    if model in {"calibrated_osrm", "main_calibrated_profile"}:
        return (
            [
                "calibrated_absolute_error_sec",
                "calibrated_abs_error_sec",
            ],
            [
                "calibrated_prediction_error_sec",
                "calibrated_error_sec",
            ],
        )

    if model in {"best_naive_multiplier", "naive_multiplier"}:
        return (
            [
                "multiplier_absolute_error_sec",
                "best_multiplier_absolute_error_sec",
                "naive_multiplier_absolute_error_sec",
            ],
            [
                "multiplier_prediction_error_sec",
                "best_multiplier_prediction_error_sec",
                "naive_multiplier_error_sec",
            ],
        )

    return ([], [])


def p95_for_model(trip_errors: pl.DataFrame | None, model: str) -> float | None:
    if trip_errors is None or trip_errors.is_empty():
        return None

    absolute_candidates, signed_candidates = model_error_column_candidates(model)

    absolute_col = _first_existing_col(trip_errors, absolute_candidates)
    if absolute_col is not None:
        return trip_errors.select(
            pl.col(absolute_col).quantile(0.95, interpolation="nearest")
        ).item()

    signed_col = _first_existing_col(trip_errors, signed_candidates)
    if signed_col is not None:
        return trip_errors.select(
            pl.col(signed_col).abs().quantile(0.95, interpolation="nearest")
        ).item()

    return None


def _rename_metric_columns(metrics: pl.DataFrame) -> pl.DataFrame:
    rename_map = {
        "share_large_error": "share_error_gt_5min",
        "share_standard_misclassified": "share_15min_misclassified",
        "mean_error_sec": "mean_signed_error_sec",
        "median_error_sec": "median_signed_error_sec",
    }

    out = metrics

    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out = out.rename({old: new})

    return out


def _get_row_metric(row: dict[str, object], key: str) -> float | None:
    return _safe_float(row.get(key))


def _add_missing_changes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = next(
        (row for row in rows if str(row.get("model")) == "baseline_osrm"),
        None,
    )

    if baseline is None:
        return rows

    base_mae = _get_row_metric(baseline, "mae_sec")
    base_rmse = _get_row_metric(baseline, "rmse_sec")
    base_large = _get_row_metric(baseline, "share_error_gt_5min")
    base_miscl = _get_row_metric(baseline, "share_15min_misclassified")

    updated: list[dict[str, object]] = []

    for row in rows:
        out = dict(row)

        if str(out.get("model")) != "baseline_osrm":
            mae = _get_row_metric(out, "mae_sec")
            rmse = _get_row_metric(out, "rmse_sec")
            large = _get_row_metric(out, "share_error_gt_5min")
            miscl = _get_row_metric(out, "share_15min_misclassified")

            if out.get("mae_change_sec") is None and base_mae not in (None, 0) and mae is not None:
                out["mae_change_sec"] = mae - base_mae
                out["mae_change_percent"] = 100 * (mae - base_mae) / base_mae

            if out.get("rmse_change_sec") is None and base_rmse not in (None, 0) and rmse is not None:
                out["rmse_change_sec"] = rmse - base_rmse
                out["rmse_change_percent"] = 100 * (rmse - base_rmse) / base_rmse

            if out.get("large_error_share_change") is None and base_large is not None and large is not None:
                out["large_error_share_change"] = large - base_large

            if out.get("standard_misclassification_change") is None and base_miscl is not None and miscl is not None:
                out["standard_misclassification_change"] = miscl - base_miscl

        updated.append(out)

    return updated


def empty_summary_row(
    *,
    run_dir: Path,
    test_type: str,
    dataset: str,
    region: str,
    variant: str,
    status: str,
    message: str,
) -> pl.DataFrame:
    return pl.DataFrame([{
        "test_type": test_type,
        "dataset": dataset,
        "region": region,
        "variant": variant,
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
        "status": status,
        "message": message,
        "run_dir": str(run_dir),
    }]).select(STANDARD_SUMMARY_COLUMNS)


def summarise_evaluation_run(
    *,
    run_dir: Path,
    test_type: str,
    dataset: str,
    region: str,
    variant: str,
    status: str = "conducted",
    message: str = "",
) -> pl.DataFrame:
    metrics_path = run_dir / "results" / "calibration_evaluation_metrics.csv"
    trip_errors_path = run_dir / "results" / "calibration_evaluation_trip_errors.parquet"

    if not metrics_path.exists():
        summary_status = "missing_metrics" if status == "conducted" else status
        summary_message = message or f"Missing metrics file: {metrics_path}"

        return empty_summary_row(
            run_dir=run_dir,
            test_type=test_type,
            dataset=dataset,
            region=region,
            variant=variant,
            status=summary_status,
            message=summary_message,
        )

    metrics = _rename_metric_columns(pl.read_csv(metrics_path))
    trip_errors = read_parquet_if_exists(trip_errors_path)

    rows: list[dict[str, object]] = []

    for row in metrics.iter_rows(named=True):
        model = str(row.get("model", ""))

        rows.append({
            "test_type": test_type,
            "dataset": dataset,
            "region": region,
            "variant": variant,
            "model": model,
            "n_trips": row.get("n_trips"),
            "mean_signed_error_sec": row.get("mean_signed_error_sec"),
            "median_signed_error_sec": row.get("median_signed_error_sec"),
            "mae_sec": row.get("mae_sec"),
            "median_ae_sec": row.get("median_ae_sec"),
            "rmse_sec": row.get("rmse_sec"),
            "p95_ae_sec": p95_for_model(trip_errors, model),
            "share_error_gt_5min": row.get("share_error_gt_5min"),
            "share_15min_misclassified": row.get("share_15min_misclassified"),
            "mae_change_sec": row.get("mae_change_sec"),
            "mae_change_percent": row.get("mae_change_percent"),
            "rmse_change_sec": row.get("rmse_change_sec"),
            "rmse_change_percent": row.get("rmse_change_percent"),
            "large_error_share_change": row.get("large_error_share_change"),
            "standard_misclassification_change": row.get("standard_misclassification_change"),
            "multiplier": row.get("multiplier"),
            "status": status,
            "message": message,
            "run_dir": str(run_dir),
        })

    rows = _add_missing_changes(rows)

    return pl.DataFrame(rows).select(STANDARD_SUMMARY_COLUMNS)


def write_summary(path: Path, frames: list[pl.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        pl.DataFrame({col: [] for col in STANDARD_SUMMARY_COLUMNS}).write_csv(path)
        return

    pl.concat(frames, how="diagonal_relaxed").select(STANDARD_SUMMARY_COLUMNS).write_csv(path)


# =========================================================
# Dataset evaluation helper
# =========================================================

def run_evaluation_dataset(
    dataset: ExtendedDataset,
    *,
    overwrite: bool = True,
) -> tuple[StatusRow, pl.DataFrame]:
    print_header(f"{dataset.test_type}: {dataset.label}")

    try:
        pbf_path = validate_extended_input(dataset.input_dir)
        ensure_run_subdirs(dataset.run_dir, overwrite=overwrite)
        copy_main_calibrated_profile(dataset.run_dir)

        write_manifest(
            dataset.run_dir / "extended_run_manifest.txt",
            {
                "test_type": dataset.test_type,
                "dataset_key": dataset.key,
                "dataset_label": dataset.label,
                "region": dataset.region,
                "pipeline_input_name": dataset.pipeline_input_name,
                "input_dir": dataset.input_dir.resolve(),
                "osm_pbf": pbf_path.resolve(),
                "main_run_dir": MAIN_RUN_DIR,
                "extended_run_dir": dataset.run_dir,
                "phase3_regression_calibration": "skipped",
                "calibrated_profile_source": MAIN_RUN_DIR / "ambulance_nl_calibrated.lua",
            },
        )

        ok, failed_step, return_code, message = run_steps(
            EVALUATION_ONLY_STEPS,
            run_dir=dataset.run_dir,
            input_name=dataset.pipeline_input_name,
            active_region=dataset.region,
        )

        if not ok:
            status_row = StatusRow(
                test_type=dataset.test_type,
                dataset=dataset.label,
                region=dataset.region,
                variant="main_calibrated_profile",
                run_dir=str(dataset.run_dir),
                status="failed",
                failed_step=failed_step,
                return_code=return_code,
                message=message,
            )
            summary = summarise_evaluation_run(
                run_dir=dataset.run_dir,
                test_type=dataset.test_type,
                dataset=dataset.label,
                region=dataset.region,
                variant="main_calibrated_profile",
                status="failed",
                message=message,
            )
            return status_row, summary

        status_row = StatusRow(
            test_type=dataset.test_type,
            dataset=dataset.label,
            region=dataset.region,
            variant="main_calibrated_profile",
            run_dir=str(dataset.run_dir),
            status="success",
            failed_step="",
            return_code="0",
            message="completed",
        )
        summary = summarise_evaluation_run(
            run_dir=dataset.run_dir,
            test_type=dataset.test_type,
            dataset=dataset.label,
            region=dataset.region,
            variant="main_calibrated_profile",
            status="conducted",
            message="",
        )
        return status_row, summary

    except Exception as exc:
        status_row = StatusRow(
            test_type=dataset.test_type,
            dataset=dataset.label,
            region=dataset.region,
            variant="main_calibrated_profile",
            run_dir=str(dataset.run_dir),
            status="failed",
            failed_step="setup_or_paths",
            return_code="",
            message=str(exc),
        )
        summary = summarise_evaluation_run(
            run_dir=dataset.run_dir,
            test_type=dataset.test_type,
            dataset=dataset.label,
            region=dataset.region,
            variant="main_calibrated_profile",
            status="failed",
            message=str(exc),
        )
        return status_row, summary