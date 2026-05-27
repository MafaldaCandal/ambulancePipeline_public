"""
utils/osrm_utils.py

OSRM integration utilities.

Purpose:
    Centralise OSRM/Docker interaction used by the pipeline:
    - OSRM graph preprocessing
    - OSRM server lifecycle
    - OSRM Route API queries
    - OSRM Match API queries
    - trace simplification and match diagnostics

Design:
    Core methodology scripts can use the defaults, which are resolved from
    paths.py. Input-preparation or synthetic scripts can pass explicit
    osm_pbf/osrm_file paths without depending on the active run.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import time
from typing import Any

import polars as pl
import requests
from shapely.geometry import LineString

from configs import (
    ACTIVE_REGION,
    OSRM_URL,
    OSRM_DOCKER_IMAGE,
    OSRM_CONTAINER_NAME,
    OSRM_TEST_ROUTES,
    MATCH_CONFIDENCE_THRESHOLD,
    MATCH_SAMPLE_INTERVAL_SEC,
    MATCH_MAX_POINTS,
    MATCH_REQUEST_TIMEOUT_SEC,
    MATCH_MIN_POINTS,
)


# =========================================================
# Project / default path resolution
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_default_osm_pbf() -> Path:
    """
    Resolve the active OSM PBF from paths.py.

    Imported lazily so utilities can be used by input-preparation scripts
    before an active run exists.
    """
    from paths import OSM_PBF

    return OSM_PBF


def get_default_osrm_file() -> Path:
    """
    Resolve the active OSRM base file from paths.py.

    Imported lazily so utilities can be used by input-preparation scripts
    before an active run exists.
    """
    from paths import OSRM_FILE

    return OSRM_FILE


def osrm_file_from_pbf(pbf_path: Path) -> Path:
    """
    Convert an .osm.pbf path to the corresponding .osrm base path.
    """
    if pbf_path.name.endswith(".osm.pbf"):
        return pbf_path.with_name(
            pbf_path.name.removesuffix(".osm.pbf") + ".osrm"
        )

    return pbf_path.with_suffix(".osrm")


# =========================================================
# Docker / command utilities
# =========================================================

def docker_path(path: Path) -> str:
    """
    Convert a project-local host path to the corresponding Docker path.

    The project root is mounted inside Docker as /data.
    """
    resolved = Path(path).resolve()

    try:
        rel = resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Path is outside PROJECT_ROOT and cannot be mounted through "
            f"the standard Docker volume: {resolved}"
        ) from exc

    return f"/data/{rel.as_posix()}"


def run_command(cmd: list[str]) -> None:
    """
    Run a command and raise if it fails.
    """
    print("\nRunning:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def stop_container() -> None:
    """
    Remove the configured OSRM Docker container if it exists.
    """
    subprocess.run(
        ["docker", "rm", "-f", OSRM_CONTAINER_NAME],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def remove_existing_osrm_outputs(osrm_file: Path | None = None) -> None:
    """
    Remove existing OSRM graph files for the configured OSRM base file.
    """
    if osrm_file is None:
        osrm_file = get_default_osrm_file()

    for path in osrm_file.parent.glob(f"{osrm_file.name}*"):
        if path.is_file():
            path.unlink()


def extract_default_car_lua(output_path: Path) -> None:
    """
    Extract OSRM's default car.lua from the Docker image.

    Tries the common Docker image path first and the local-install path second.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_paths = [
        "/opt/car.lua",
        "/usr/local/share/osrm/profiles/car.lua",
    ]

    errors = []

    for profile_path in candidate_paths:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "cat",
            OSRM_DOCKER_IMAGE,
            profile_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout.strip():
            output_path.write_text(result.stdout, encoding="utf-8")
            print(f"Saved default OSRM profile: {output_path}")
            print(f"Extracted from Docker path: {profile_path}")
            return

        errors.append(
            f"{profile_path}: returncode={result.returncode}; "
            f"stderr={result.stderr.strip()}"
        )

    raise RuntimeError(
        "Could not extract car.lua from the OSRM Docker image. Tried:\n"
        + "\n".join(f"  - {err}" for err in errors)
    )


# =========================================================
# OSRM graph preprocessing
# =========================================================

def run_osrm_preprocessing(
    lua_profile: Path,
    osm_pbf: Path | None = None,
    osrm_file: Path | None = None,
) -> None:
    """
    Build OSRM graph files from an OSM PBF and Lua profile.

    Defaults:
        Uses the active OSM extract from paths.py.

    Explicit paths:
        Synthetic/input-preparation scripts can pass regional PBF and OSRM
        paths directly.
    """
    if osm_pbf is None:
        osm_pbf = get_default_osm_pbf()

    if osrm_file is None:
        osrm_file = osrm_file_from_pbf(osm_pbf)

    if not osm_pbf.exists():
        raise FileNotFoundError(f"Missing OSM PBF file: {osm_pbf}")

    if not lua_profile.exists():
        raise FileNotFoundError(f"Missing OSRM Lua profile: {lua_profile}")

    volume = f"{PROJECT_ROOT}:/data"

    stop_container()
    remove_existing_osrm_outputs(osrm_file)

    run_command([
        "docker",
        "run",
        "--rm",
        "-v",
        volume,
        OSRM_DOCKER_IMAGE,
        "osrm-extract",
        "-p",
        docker_path(lua_profile),
        docker_path(osm_pbf),
    ])

    run_command([
        "docker",
        "run",
        "--rm",
        "-v",
        volume,
        OSRM_DOCKER_IMAGE,
        "osrm-contract",
        docker_path(osrm_file),
    ])


# =========================================================
# OSRM server lifecycle
# =========================================================

def start_osrm_server(
    osrm_file: Path | None = None,
) -> subprocess.Popen:
    """
    Start an OSRM HTTP server from an OSRM graph file.
    """
    if osrm_file is None:
        osrm_file = get_default_osrm_file()

    if not osrm_file.exists():
        raise FileNotFoundError(
            f"Missing OSRM graph file: {osrm_file}. "
            "Run OSRM preprocessing first."
        )

    stop_container()

    print("Starting OSRM server...")

    return subprocess.Popen([
        "docker",
        "run",
        "--rm",
        "--name",
        OSRM_CONTAINER_NAME,
        "-p",
        "5000:5000",
        "-v",
        f"{PROJECT_ROOT}:/data",
        OSRM_DOCKER_IMAGE,
        "osrm-routed",
        docker_path(osrm_file),
    ])


def stop_osrm_server(server: subprocess.Popen | None = None) -> None:
    """
    Stop the OSRM server process and remove its Docker container.
    """
    print("Stopping OSRM server...")

    if server is not None:
        server.terminate()

        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    stop_container()


def wait_for_osrm(
    max_attempts: int = 30,
    timeout_sec: int = 2,
    region: str | None = None,
) -> None:
    """
    Wait until OSRM responds to a small test route query.

    If region is provided, that region key is used. Otherwise ACTIVE_REGION
    from configs.py is used.
    """
    active_region = region or ACTIVE_REGION

    if active_region not in OSRM_TEST_ROUTES:
        raise KeyError(
            f"No OSRM test route configured for region={active_region!r}. "
            f"Available regions: {sorted(OSRM_TEST_ROUTES)}"
        )

    (lon1, lat1), (lon2, lat2) = OSRM_TEST_ROUTES[active_region]

    test_url = (
        f"{OSRM_URL}/route/v1/driving/"
        f"{lon1},{lat1};{lon2},{lat2}?overview=false"
    )

    print("Waiting for OSRM server...")

    for _ in range(max_attempts):
        try:
            response = requests.get(test_url, timeout=timeout_sec)

            if response.status_code in {200, 400}:
                print("OSRM server is ready.")
                return

        except requests.RequestException:
            pass

        time.sleep(1)

    raise RuntimeError(
        "OSRM server did not become ready. "
        f"Test URL was: {test_url}"
    )


# =========================================================
# OSRM Route API
# =========================================================

def query_osrm_route(
    origin_lon: float,
    origin_lat: float,
    dest_lon: float,
    dest_lat: float,
    overview: str = "full",
    geometries: str = "geojson",
    steps: bool = True,
    timeout_sec: int = 10,
) -> dict[str, Any] | None:
    """
    Query OSRM's Route endpoint for one origin-destination pair.

    Returns None if the request fails or OSRM does not return an OK route.
    """
    coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"

    url = (
        f"{OSRM_URL}/route/v1/driving/{coords}"
        f"?overview={overview}&geometries={geometries}"
        f"&steps={str(steps).lower()}"
    )

    try:
        response = requests.get(url, timeout=timeout_sec)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None
    except ValueError:
        return None

    if data.get("code") != "Ok" or not data.get("routes"):
        return None

    route = data["routes"][0]

    out: dict[str, Any] = {
        "duration": route["duration"],
        "distance": route["distance"],
    }

    if "geometry" in route:
        out["geometry"] = route["geometry"]

    if steps and "legs" in route:
        out["n_turns"] = estimate_turn_count(route)

    return out


# =========================================================
# OSRM Match API helpers
# =========================================================

def simplify_trace(
    points: pl.DataFrame,
    timestamp_col: str = "timestamp",
    lon_col: str = "vehicle_lon",
    lat_col: str = "vehicle_lat",
    min_points: int = MATCH_MIN_POINTS,
    sample_interval_sec: int = MATCH_SAMPLE_INTERVAL_SEC,
    max_points: int = MATCH_MAX_POINTS,
    turn_angle_threshold_deg: float = 35.0,
) -> pl.DataFrame:
    """
    Simplify a GPS trace before OSRM Match.

    Strategy:
        1. Sort by timestamp.
        2. Remove null coordinates and exact duplicate coordinate points.
        3. Always keep first and last points.
        4. Keep points spaced by at least sample_interval_sec.
        5. Additionally preserve sharp-turn points.
        6. If still above max_points, downsample while preserving endpoints.

    This is intentionally simple. It improves over pure time thinning by
    retaining geometry that is important for map matching.
    """

    required = [timestamp_col, lon_col, lat_col]
    missing = [col for col in required if col not in points.columns]
    if missing:
        raise ValueError(f"Trace is missing required column(s): {missing}")

    points = (
        points
        .sort(timestamp_col)
        .filter(
            pl.col(timestamp_col).is_not_null()
            & pl.col(lon_col).is_not_null()
            & pl.col(lat_col).is_not_null()
        )
        .unique(subset=[timestamp_col, lon_col, lat_col], keep="first")
        .sort(timestamp_col)
        .with_row_index("_original_idx")
    )

    if points.height <= min_points:
        return points.drop("_original_idx")

    if max_points < min_points:
        raise ValueError(
            f"max_points={max_points} is smaller than min_points={min_points}."
        )

    timestamps = points[timestamp_col].to_list()
    lons = points[lon_col].to_list()
    lats = points[lat_col].to_list()

    keep: set[int] = {0, points.height - 1}

    # -----------------------------------------------------
    # Time-based retention
    # -----------------------------------------------------
    last_kept_time = timestamps[0]

    for i in range(1, points.height - 1):
        dt = (timestamps[i] - last_kept_time).total_seconds()

        if dt >= sample_interval_sec:
            keep.add(i)
            last_kept_time = timestamps[i]

    # -----------------------------------------------------
    # Geometry-based retention: preserve sharp turns
    # -----------------------------------------------------
    def angle_deg(i: int) -> float | None:
        ax = lons[i] - lons[i - 1]
        ay = lats[i] - lats[i - 1]
        bx = lons[i + 1] - lons[i]
        by = lats[i + 1] - lats[i]

        norm_a = (ax * ax + ay * ay) ** 0.5
        norm_b = (bx * bx + by * by) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return None

        cos_theta = (ax * bx + ay * by) / (norm_a * norm_b)
        cos_theta = max(-1.0, min(1.0, cos_theta))

        import math
        return math.degrees(math.acos(cos_theta))

    for i in range(1, points.height - 1):
        angle = angle_deg(i)
        if angle is not None and angle >= turn_angle_threshold_deg:
            keep.add(i)

    keep_indices = sorted(keep)

    # -----------------------------------------------------
    # Hard cap for OSRM Match request size
    # -----------------------------------------------------
    if len(keep_indices) > max_points:
        # Keep endpoints, then sample evenly from internal retained points.
        internal = keep_indices[1:-1]
        slots = max_points - 2

        if slots <= 0:
            keep_indices = [0, points.height - 1]
        elif len(internal) <= slots:
            keep_indices = [0, *internal, points.height - 1]
        else:
            selected_internal = [
                internal[round(i * (len(internal) - 1) / (slots - 1))]
                for i in range(slots)
            ]
            keep_indices = sorted({0, *selected_internal, points.height - 1})

    return (
        points
        .filter(pl.col("_original_idx").is_in(keep_indices))
        .drop("_original_idx")
        .sort(timestamp_col)
    )
    

def build_match_url(
    points: pl.DataFrame,
    lon_col: str = "vehicle_lon",
    lat_col: str = "vehicle_lat",
) -> str:
    """
    Build the OSRM Match endpoint URL for a sequence of lon/lat points.
    """
    coords = ";".join(
        f"{lon},{lat}"
        for lon, lat in zip(points[lon_col].to_list(), points[lat_col].to_list())
    )

    return f"{OSRM_URL}/match/v1/driving/{coords}"


def trace_diagnostics(
    original_points: pl.DataFrame,
    submitted_points: pl.DataFrame,
    url: str,
    timestamp_col: str = "timestamp",
) -> dict[str, Any]:
    """
    Return basic diagnostics for a trace submitted to OSRM Match.
    """
    original_points = original_points.sort(timestamp_col)

    duration_sec = (
        original_points[timestamp_col][-1]
        - original_points[timestamp_col][0]
    ).total_seconds()

    max_gap_sec = (
        original_points
        .with_columns(
            pl.col(timestamp_col)
            .diff()
            .dt.total_seconds()
            .alias("dt_sec")
        )
        .select(pl.col("dt_sec").max())
        .item()
    )

    return {
        "n_points_original": original_points.height,
        "n_points_submitted": submitted_points.height,
        "trace_duration_sec": duration_sec,
        "trace_max_gap_sec": max_gap_sec,
        "request_url_length": len(url),
    }


# =========================================================
# OSRM Match rejection explanation
# =========================================================

def explain_match_rejection(result: dict[str, Any]) -> dict[str, str]:
    """
    Convert a raw OSRM Match result into human-readable rejection metadata.

    The raw ``status`` remains the machine-readable reason. The returned fields
    are intended for route_rejection_summary.parquet and diagnostics reports.
    """
    status = str(result.get("status") or "unknown")
    message = str(result.get("error_message") or "").lower()
    url_len = result.get("request_url_length")

    if status == "low_confidence":
        threshold = result.get("match_confidence_threshold", MATCH_CONFIDENCE_THRESHOLD)
        return {
            "rejection_category": "low_confidence_match",
            "rejection_summary": "Matched by OSRM, but below the confidence threshold.",
            "rejection_explanation": (
                "OSRM returned a route match, but its confidence score was below "
                f"the configured threshold ({threshold})."
            ),
            "recommended_check": (
                "Inspect match_confidence, GPS gaps, trace simplification, and "
                "whether the access profile or OSM extract is too restrictive."
            ),
        }

    if status == "request_timeout":
        return {
            "rejection_category": "osrm_timeout",
            "rejection_summary": "The OSRM Match request timed out.",
            "rejection_explanation": (
                "The HTTP request to OSRM did not complete within the configured "
                "timeout. This is an infrastructure/performance failure, not a "
                "low-quality match."
            ),
            "recommended_check": (
                "Increase MATCH_REQUEST_TIMEOUT_SEC, reduce submitted points, or "
                "check Docker/OSRM server performance."
            ),
        }

    if status == "request_too_long" or "http 414" in message or (url_len is not None and url_len > 8000):
        return {
            "rejection_category": "request_too_long",
            "rejection_summary": "The OSRM Match URL was too long.",
            "rejection_explanation": (
                "The submitted trace likely produced a URL that exceeded what the "
                "server or client would accept."
            ),
            "recommended_check": (
                "Lower MATCH_MAX_POINTS or increase MATCH_SAMPLE_INTERVAL_SEC."
            ),
        }

    if status == "request_connection_error":
        return {
            "rejection_category": "osrm_connection_failure",
            "rejection_summary": "The OSRM server connection failed.",
            "rejection_explanation": (
                "The request could not connect to the OSRM server, or the server "
                "closed the connection."
            ),
            "recommended_check": (
                "Check whether the Docker container is still running and inspect "
                "OSRM server logs."
            ),
        }

    if status in {"request_http_error", "response_parse_error", "request_failed"}:
        return {
            "rejection_category": "osrm_request_failed",
            "rejection_summary": "The OSRM Match request failed before a usable match was returned.",
            "rejection_explanation": (
                "The request failed because of an HTTP error, response parsing "
                "error, or another non-specific request problem."
            ),
            "recommended_check": "Inspect error_message directly.",
        }

    if status == "no_segment":
        return {
            "rejection_category": "no_nearby_routable_segment",
            "rejection_summary": "OSRM could not snap the trace to nearby routable road segments.",
            "rejection_explanation": (
                "The GPS points were not close enough to routable segments in the "
                "active OSRM network."
            ),
            "recommended_check": (
                "Check whether points fall outside the OSM extract or on roads "
                "excluded by the Lua profile."
            ),
        }

    if status == "no_match":
        return {
            "rejection_category": "no_coherent_route_match",
            "rejection_summary": "OSRM could not construct a coherent route through the trace.",
            "rejection_explanation": (
                "Points may be near roads, but the sequence could not be matched "
                "to a plausible continuous path."
            ),
            "recommended_check": (
                "Inspect GPS gaps, jumps, trip start/end detection, and whether "
                "the trace contains unrelated movement."
            ),
        }

    if status == "too_few_points_for_match":
        return {
            "rejection_category": "too_few_points",
            "rejection_summary": "Too few GPS points were available for OSRM Match.",
            "rejection_explanation": (
                "After trace simplification, the trace had fewer points than "
                "MATCH_MIN_POINTS."
            ),
            "recommended_check": (
                "Check GPS sampling density, MATCH_MIN_POINTS, and trace simplification settings."
            ),
        }

    if status == "osrm_error":
        return {
            "rejection_category": "unexpected_osrm_response",
            "rejection_summary": "OSRM returned an unexpected error response.",
            "rejection_explanation": (
                "The OSRM response was valid enough to parse, but did not match "
                "the expected successful or known failure cases."
            ),
            "recommended_check": "Inspect the raw error_message and OSRM logs.",
        }

    return {
        "rejection_category": "other_osrm_rejection",
        "rejection_summary": f"Unhandled OSRM Match status: {status}",
        "rejection_explanation": f"No specific explanation is defined for status={status!r}.",
        "recommended_check": "Inspect raw status, error_message, and trace diagnostics.",
    }


def estimate_turn_count(matching_or_route: dict[str, Any]) -> int:
    """
    Estimate turn count from OSRM step manoeuvres.
    """
    turn_types = {
        "turn",
        "new name",
        "merge",
        "fork",
        "end of road",
    }

    count = 0

    for leg in matching_or_route.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            if maneuver.get("type") in turn_types:
                count += 1

    return count


# =========================================================
# OSRM Match API
# =========================================================

def query_osrm_match(
    points: pl.DataFrame,
    confidence_threshold: float = MATCH_CONFIDENCE_THRESHOLD,
    timestamp_col: str = "timestamp",
    lon_col: str = "vehicle_lon",
    lat_col: str = "vehicle_lat",
    min_points: int = MATCH_MIN_POINTS,
    sample_interval_sec: int = MATCH_SAMPLE_INTERVAL_SEC,
    max_points: int = MATCH_MAX_POINTS,
    request_timeout_sec: int = MATCH_REQUEST_TIMEOUT_SEC,
    include_geometry: bool = True,
) -> dict[str, Any]:
    """
    Query OSRM Match for one GPS trace and return a structured result.
    """
    original_points = points.sort(timestamp_col)

    match_points = simplify_trace(
        original_points,
        timestamp_col=timestamp_col,
        min_points=min_points,
        sample_interval_sec=sample_interval_sec,
        max_points=max_points,
    )

    if match_points.height < min_points:
        return {
            "status": "too_few_points_for_match",
            "error_message": f"Only {match_points.height} points after simplification",
            "match_confidence": None,
            "n_points_original": original_points.height,
            "n_points_submitted": match_points.height,
            "trace_duration_sec": None,
            "trace_max_gap_sec": None,
            "request_url_length": None,
        }

    url = build_match_url(match_points, lon_col=lon_col, lat_col=lat_col)

    diagnostics = trace_diagnostics(
        original_points,
        match_points,
        url,
        timestamp_col=timestamp_col,
    )

    params = {
        "geometries": "geojson",
        "overview": "full",
        "annotations": "true",
        "steps": "true",
    }

    try:
        response = requests.get(url, params=params, timeout=request_timeout_sec)
    except requests.exceptions.Timeout as exc:
        return {
            "status": "request_timeout",
            "error_message": str(exc),
            "match_confidence": None,
            **diagnostics,
        }
    except requests.exceptions.ConnectionError as exc:
        return {
            "status": "request_connection_error",
            "error_message": str(exc),
            "match_confidence": None,
            **diagnostics,
        }
    except requests.exceptions.RequestException as exc:
        return {
            "status": "request_failed",
            "error_message": str(exc),
            "match_confidence": None,
            **diagnostics,
        }

    if response.status_code != 200:
        status = "request_too_long" if response.status_code == 414 else "request_http_error"
        return {
            "status": status,
            "error_message": f"HTTP {response.status_code}: {response.text[:500]}",
            "match_confidence": None,
            **diagnostics,
        }

    try:
        data = response.json()
    except ValueError as exc:
        return {
            "status": "response_parse_error",
            "error_message": f"Could not parse OSRM JSON response: {exc}",
            "match_confidence": None,
            **diagnostics,
        }

    code = data.get("code")

    if code == "NoSegment":
        return {
            "status": "no_segment",
            "error_message": data.get("message"),
            "match_confidence": None,
            **diagnostics,
        }

    if code == "NoMatch":
        return {
            "status": "no_match",
            "error_message": data.get("message"),
            "match_confidence": None,
            **diagnostics,
        }

    if code != "Ok":
        return {
            "status": "osrm_error",
            "error_message": f"OSRM code={code!r}; message={data.get('message')!r}",
            "match_confidence": None,
            **diagnostics,
        }

    if not data.get("matchings"):
        return {
            "status": "no_match",
            "error_message": "OSRM returned Ok but no matchings.",
            "match_confidence": None,
            **diagnostics,
        }

    matching = data["matchings"][0]
    confidence = matching.get("confidence", 0.0)

    if confidence < confidence_threshold:
        return {
            "status": "low_confidence",
            "error_message": None,
            "match_confidence": confidence,
            "match_confidence_threshold": confidence_threshold,
            **diagnostics,
        }

    out: dict[str, Any] = {
        "status": "ok",
        "osrm_time_sec": matching["duration"],
        "distance_m": matching["distance"],
        "distance_km": matching["distance"] / 1000,
        "n_turns": estimate_turn_count(matching),
        "match_confidence": confidence,
        **diagnostics,
    }

    if include_geometry:
        try:
            out["geometry"] = LineString(matching["geometry"]["coordinates"])
        except KeyError as exc:
            return {
                "status": "osrm_error",
                "error_message": f"OSRM matching did not include expected geometry: {exc}",
                "match_confidence": confidence,
                **diagnostics,
            }

    return out


def osrm_match_trace(points: pl.DataFrame) -> dict[str, Any]:
    """
    Backwards-compatible alias.
    """
    return query_osrm_match(points)