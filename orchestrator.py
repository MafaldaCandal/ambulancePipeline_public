"""
orchestrator.py

Public/reproducible analytical pipeline.

Default behaviour:
    Run the core methodology pipeline on non-confidential synthetic data:

        data/input_synthetic/

This makes the repository runnable from a clean checkout, provided the synthetic
input folder contains the expected prepared files.

Real-data runs are explicit:

        python orchestrator.py --input BN_2024
        python orchestrator.py --input BZO_2024
        python orchestrator.py --input BMW_2024

Raw-data parsing and synthetic-data generation are intentionally outside the core
methodology pipeline. This orchestrator starts from prepared input tables.

Core methodology phases:
    Phase 1: realised travel-time reconstruction
    Phase 2: OSRM baseline prediction and route-feature extraction
    Phase 3: residual modelling and calibrated profile creation
    Phase 4: calibrated prediction and evaluation

Optional:
    diagnostics/
    extended_tests/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import csv
import time
from datetime import datetime
from pathlib import Path

import configs


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_INPUT_NAME = "synthetic"


# =========================================================
# CLI and input selection
# =========================================================

def normalise_input_name(value: str | None) -> str:
    """
    Convert user-facing dataset names to the canonical name used by paths.py.

    Accepted examples:
        synthetic       -> synthetic
        input_synthetic -> synthetic
        BN_2024         -> BN_2024
        input_BN_2024   -> BN_2024
    """
    if value is None or not str(value).strip():
        return DEFAULT_INPUT_NAME

    name = str(value).strip()
    if name.startswith("input_"):
        name = name.removeprefix("input_")

    return name


def input_dir_for(input_name: str) -> Path:
    """
    Return the prepared-input folder for a canonical input name.
    """
    name = normalise_input_name(input_name)
    return DATA_DIR / "input_synthetic" if name == "synthetic" else DATA_DIR / f"input_{name}"


def infer_active_region(input_name: str) -> str:
    """
    Infer the routing region key used by configs.OSM_PBF_FILES.

    For real datasets, the convention is REGION_YEAR, e.g. BN_2024.
    For synthetic data, the key is synthetic.
    """
    name = normalise_input_name(input_name)

    if name.lower() == "synthetic":
        return "synthetic"

    return name.split("_", 1)[0]


def infer_active_year(input_name: str, default: int = 2024) -> int:
    """
    Infer the active year from REGION_YEAR style input names.
    """
    name = normalise_input_name(input_name)
    parts = name.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])

    return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ambulance routing calibration methodology pipeline. "
            "By default, this uses data/input_synthetic for one-command "
            "reproducibility."
        )
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT_NAME,
        help=(
            "Prepared input dataset to use. Default: synthetic. Examples: "
            "synthetic, BN_2024, BZO_2024, BMW_2024. "
            "The corresponding folder must exist under data/input_<name>."
        ),
    )

    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run diagnostic scripts after the core methodology pipeline.",
    )

    parser.add_argument(
        "--extended-tests",
        action="store_true",
        help="Run extended validation, transferability, expressiveness, and sensitivity tests.",
    )

    parser.add_argument(
        "--no-config-optionals",
        action="store_true",
        help=(
            "Ignore RUN_DIAGNOSTICS and RUN_EXTENDED_TESTS from configs.py. "
            "Only CLI flags will control optional stages."
        ),
    )

    return parser.parse_args()


# =========================================================
# Run management
# =========================================================

def get_next_run_dir() -> Path:
    """
    Create the next available runs/RunXXX directory.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    existing = [
        p for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("Run") and p.name[3:].isdigit()
    ]

    next_idx = 1 if not existing else max(int(p.name[3:]) for p in existing) + 1

    run_dir = RUNS_DIR / f"Run{next_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Main run folders.
    (run_dir / "outputs").mkdir(exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)
    (run_dir / "report").mkdir(exist_ok=True)

    return run_dir


def build_env(run_dir: Path, *, input_name: str, active_region: str) -> dict[str, str]:
    """
    Build subprocess environment.

    ACTIVE_RUN_DIR is used by paths.py.
    PIPELINE_INPUT_NAME and PIPELINE_ACTIVE_REGION make the dataset choice
    explicit for paths.py and all subprocesses.

    PIPELINE_MAIN_* values are duplicated intentionally so extended-test scripts
    can distinguish the main calibration run from temporary transfer/validation
    datasets.

    PYTHONPATH is needed because scripts live in subfolders but still import
    root-level modules such as configs.py and paths.py.
    """
    normalised_input = normalise_input_name(input_name)
    active_year = infer_active_year(normalised_input)

    env = os.environ.copy()
    env["ACTIVE_RUN_DIR"] = str(run_dir.resolve())

    env["PIPELINE_INPUT_NAME"] = normalised_input
    env["PIPELINE_ACTIVE_REGION"] = active_region
    env["PIPELINE_ACTIVE_YEAR"] = str(active_year)

    env["PIPELINE_MAIN_INPUT"] = normalised_input
    env["PIPELINE_MAIN_DATASET"] = normalised_input
    env["PIPELINE_MAIN_REGION"] = active_region
    env["PIPELINE_MAIN_YEAR"] = str(active_year)

    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not existing_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
    )

    return env


def save_run_configuration_snapshot(run_dir: Path, env: dict[str, str]) -> None:
    """
    Save the configuration used for this run.

    This stores:
        - configs_used.py: exact source file present at runtime
        - resolved_configs.json: uppercase config values after environment
          variables have been applied
        - pipeline_environment.json: PIPELINE_* and ACTIVE_RUN_DIR variables
    """
    configs_source = PROJECT_ROOT / "configs.py"
    configs_copy = run_dir / "configs_used.py"

    if not configs_source.exists():
        raise FileNotFoundError(f"Could not find configs.py at {configs_source}")

    shutil.copy2(configs_source, configs_copy)

    resolved_config_path = run_dir / "resolved_configs.json"
    pipeline_env_path = run_dir / "pipeline_environment.json"

    snapshot_code = f"""
import json
from pathlib import Path
import configs

out = {{}}

for name in dir(configs):
    if not name.isupper():
        continue

    value = getattr(configs, name)

    try:
        json.dumps(value)
        out[name] = value
    except TypeError:
        out[name] = repr(value)

Path({json.dumps(str(resolved_config_path))}).write_text(
    json.dumps(out, indent=2, sort_keys=True),
    encoding="utf-8",
)
"""

    subprocess.run(
        [sys.executable, "-c", snapshot_code],
        check=True,
        cwd=PROJECT_ROOT,
        env=env,
    )

    pipeline_env = {
        key: value
        for key, value in env.items()
        if key.startswith("PIPELINE_") or key == "ACTIVE_RUN_DIR"
    }

    pipeline_env_path.write_text(
        json.dumps(pipeline_env, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\nSaved run configuration snapshot:")
    print(f"  {configs_copy}")
    print(f"  {resolved_config_path}")
    print(f"  {pipeline_env_path}")


# =========================================================
# Input validation
# =========================================================

def resolve_pbf_for_input(*, input_name: str, active_region: str) -> Path:
    """
    Resolve the OSM PBF for a prepared input folder.

    First try the configured filename in configs.OSM_PBF_FILES. If the configured
    file is absent, fall back to exactly one *.osm.pbf in the input folder.
    """
    active_input_dir = input_dir_for(input_name)
    pbf_files = sorted(active_input_dir.glob("*.osm.pbf"))

    pbf_name = getattr(configs, "OSM_PBF_FILES", {}).get(active_region)
    if pbf_name:
        configured_path = active_input_dir / pbf_name
        if configured_path.exists():
            return configured_path

    if len(pbf_files) == 1:
        return pbf_files[0]

    if pbf_name:
        raise FileNotFoundError(
            f"Missing configured OSM PBF file: {active_input_dir / pbf_name}\n"
            f"Fallback requires exactly one *.osm.pbf in {active_input_dir}, "
            f"but found {len(pbf_files)}."
        )

    raise FileNotFoundError(
        f"configs.OSM_PBF_FILES has no entry for active region {active_region!r}, "
        f"and fallback requires exactly one *.osm.pbf in {active_input_dir}; "
        f"found {len(pbf_files)}."
    )


def validate_prepared_inputs(*, input_name: str, active_region: str) -> None:
    """
    Check that the analytical pipeline can start from prepared input files.

    Required:
        data/input_<name>/dispatch_registers.parquet

    GPS can be provided either as:
        data/input_<name>/gps_logs.parquet

    or as chunked parquet files:
        data/input_<name>/gps_logs_raw_parquet/*.parquet
    """
    active_input_dir = input_dir_for(input_name)

    dispatch_path = active_input_dir / "dispatch_registers.parquet"
    gps_single_path = active_input_dir / "gps_logs.parquet"
    gps_chunk_dir = active_input_dir / "gps_logs_raw_parquet"

    missing_messages: list[str] = []

    if not active_input_dir.exists():
        missing_messages.append(f"Prepared input directory does not exist: {active_input_dir}")

    if not dispatch_path.exists():
        missing_messages.append(f"Missing dispatch table: {dispatch_path}")

    has_gps_single = gps_single_path.exists()
    has_gps_chunks = gps_chunk_dir.exists() and any(gps_chunk_dir.glob("*.parquet"))

    if not has_gps_single and not has_gps_chunks:
        missing_messages.append(
            "Missing GPS input. Expected either:\n"
            f"  - {gps_single_path}\n"
            f"  - one or more parquet files in {gps_chunk_dir}"
        )

    if active_input_dir.exists():
        try:
            resolve_pbf_for_input(input_name=input_name, active_region=active_region)
        except FileNotFoundError as exc:
            missing_messages.append(str(exc))

    if missing_messages:
        msg = [
            "The core analytical pipeline starts from prepared inputs, but the selected input is incomplete.",
            "",
            f"Selected input:  {normalise_input_name(input_name)}",
            f"Input folder:    {active_input_dir}",
            f"Active region:   {active_region}",
            "",
            "Problem(s):",
        ]
        msg.extend(f"  - {line}" for line in missing_messages)
        msg.extend([
            "",
            "For one-command reproducibility, make sure data/input_synthetic contains the synthetic prepared inputs.",
            "For real-data runs, pass an explicit dataset, e.g.:",
            "  python orchestrator.py --input BN_2024",
        ])

        raise FileNotFoundError("\n".join(msg))


# =========================================================
# Subprocess execution
# =========================================================

def run_step(label: str, script: str, env: dict[str, str]) -> None:
    """
    Run one pipeline script with the current Python interpreter and record timing.
    """
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        raise FileNotFoundError(f"Pipeline script not found: {script_path}")

    run_dir = Path(env["ACTIVE_RUN_DIR"])
    timing_path = run_dir / "results" / "run_step_timings.csv"
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    step_group = script.split("/", 1)[0] if "/" in script else "root"

    print("\n" + "=" * 80)
    print(f"Running: {label}")
    print(f"Script:  {script}")
    print("=" * 80)

    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        cwd=PROJECT_ROOT,
        env=env,
    )

    elapsed_sec = round(time.perf_counter() - start, 3)
    finished_at = datetime.now().isoformat(timespec="seconds")
    status = "success" if completed.returncode == 0 else "failed"

    file_exists = timing_path.exists()

    with timing_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "started_at",
                "finished_at",
                "step_group",
                "label",
                "script",
                "status",
                "return_code",
                "elapsed_sec",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "started_at": started_at,
            "finished_at": finished_at,
            "step_group": step_group,
            "label": label,
            "script": script,
            "status": status,
            "return_code": completed.returncode,
            "elapsed_sec": elapsed_sec,
        })

    print(f"Step time: {elapsed_sec:.1f} sec")
    print(f"Timing log: {timing_path}")

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
        )


def run_optional_step(label: str, script: str, env: dict[str, str]) -> None:
    """
    Run an optional script only if it exists.
    """
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        print(f"\nSkipping optional step because script was not found: {script}")
        return

    run_step(label, script, env)


# =========================================================
# Pipeline definitions
# =========================================================

CORE_STEPS = [
    # Phase 1: realised travel-time reconstruction
    ("Build candidate trips", "methodology/p1s1_build_candidate_trips.py"),
    ("Identify trip start and arrival", "methodology/p1s2_identify_trip_start_end.py"),

    # Phase 2: baseline OSRM predictions and route features
    ("Adjust OSRM accessibility profile", "methodology/p2s3_adjust_osrm_accessibility.py"),
    ("Map-match routes", "methodology/p2s4_map_match_routes.py"),
    ("Extract route features", "methodology/p2s5_extract_route_features.py"),
    ("Compute prediction errors", "methodology/p2s6_compute_prediction_errors.py"),

    # Phase 3: residual modelling and calibrated profile
    ("Prepare regression dataset", "methodology/p3s7_prepare_regression_dataset.py"),
    ("Run regression", "methodology/p3s8_run_regression.py"),
    ("Create calibrated profile", "methodology/p3s9_create_calibrated_profile.py"),

    # Phase 4: calibrated prediction and evaluation
    ("Generate calibrated predictions", "methodology/p4s10_compute_calibrated_predictions.py"),
    ("Evaluate calibration", "methodology/p4s11_evaluate_calibration.py"),
    ("Generate core results report", "methodology/p4s12_generate_results_report.py"),
]


DIAGNOSTIC_STEPS = [
    ("Diagnose Phase 1", "diagnostics/diagnose_phase1_realised.py"),
    ("Diagnose Phase 2", "diagnostics/diagnose_phase2_prediction.py"),
    ("Diagnose Phase 3", "diagnostics/diagnose_phase3_calibration.py"),
    ("Diagnose Phase 4", "diagnostics/diagnose_phase4_evaluation.py"),
    ("Generate diagnostic report", "diagnostics/generate_diagnostics_report.py"),
]


EXTENDED_TEST_STEPS = [
    ("Run temporal validation", "extended_tests/run_temporal_validation.py"),
    ("Run spatial transferability", "extended_tests/run_spatial_transferability.py"),
    ("Run expressiveness tests", "extended_tests/run_static_profile_expressiveness.py"),
    ("Run sensitivity analysis", "extended_tests/run_sensitivity_analysis.py"),
    ("Generate extended tests report", "extended_tests/generate_extended_tests_report.py"),
]


# =========================================================
# Main
# =========================================================

def main() -> None:
    args = parse_args()

    input_name = normalise_input_name(args.input)
    active_region = infer_active_region(input_name)

    validate_prepared_inputs(input_name=input_name, active_region=active_region)

    run_dir = get_next_run_dir()
    env = build_env(run_dir, input_name=input_name, active_region=active_region)

    run_diagnostics = args.diagnostics
    run_extended_tests = args.extended_tests

    if not args.no_config_optionals:
        run_diagnostics = run_diagnostics or getattr(configs, "RUN_DIAGNOSTICS", False)
        run_extended_tests = run_extended_tests or getattr(configs, "RUN_EXTENDED_TESTS", False)

    env["PIPELINE_RUN_DIAGNOSTICS"] = "1" if run_diagnostics else "0"
    env["PIPELINE_RUN_EXTENDED_TESTS"] = "1" if run_extended_tests else "0"

    save_run_configuration_snapshot(run_dir, env)

    print("\nCreated analytical pipeline run:")
    print(run_dir)
    print("\nSelected input:")
    print(f"  name:   {input_name}")
    print(f"  folder: {input_dir_for(input_name)}")
    print(f"  region: {active_region}")

    for label, script in CORE_STEPS:
        run_step(label, script, env)

    if run_diagnostics:
        print("\nRunning optional diagnostics.")
        for label, script in DIAGNOSTIC_STEPS:
            run_optional_step(label, script, env)

    if run_extended_tests:
        print("\nRunning optional extended tests.")
        for label, script in EXTENDED_TEST_STEPS:
            run_optional_step(label, script, env)

    print("\nAnalytical pipeline complete.")
    print("Run directory:")
    print(run_dir)


if __name__ == "__main__":
    main()