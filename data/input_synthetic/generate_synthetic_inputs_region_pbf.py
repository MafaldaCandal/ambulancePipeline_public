"""
Generate one public synthetic input dataset for the ambulance routing pipeline.

Purpose:
    Create non-confidential, pipeline-ready inputs for the default command:

        python orchestrator.py

Expected folder:
    data/input_synthetic/

Required before running:
    Place exactly one regional .osm.pbf file in data/input_synthetic/.
    The file can be named anything, for example:
        zuid-holland-260512.osm.pbf

Outputs:
    data/input_synthetic/dispatch_registers.parquet
    data/input_synthetic/gps_logs.parquet
    data/input_synthetic/synthetic_truth.parquet
    data/input_synthetic/synthetic_routes.geojson

Notes:
    - This script generates one simple synthetic dataset only.
    - It does not create validation or transferability datasets.
    - Extended empirical tests should use real prepared folders such as
      data/input_ZHZ_2024/, data/input_ZHZ_2025/, data/input_BN_2024/, etc.
    - The GPS output is a single parquet file, not chunked.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json
import math
import random
import shutil
import sys

import numpy as np
import polars as pl


# =========================================================
# Project import path
# =========================================================

def find_project_root(start: Path) -> Path:
    """
    Walk upwards until the repository root is found.

    This makes the script safe to keep inside data/input_synthetic/ while still
    being able to import utils.osrm_utils from the project root.
    """
    current = start.resolve()

    for candidate in [current, *current.parents]:
        if (candidate / "orchestrator.py").exists() and (candidate / "utils").is_dir():
            return candidate

    raise RuntimeError(
        "Could not locate project root. Expected to find orchestrator.py and utils/ "
        "above this script."
    )


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.osrm_utils import (  # noqa: E402
    extract_default_car_lua,
    osrm_file_from_pbf,
    query_osrm_route,
    run_osrm_preprocessing,
    start_osrm_server,
    stop_osrm_server,
    wait_for_osrm,
)


# =========================================================
# Paths
# =========================================================

INPUT_DIR = PROJECT_ROOT / "data" / "input_synthetic"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

DISPATCH_REGISTERS_PARQUET = INPUT_DIR / "dispatch_registers.parquet"
GPS_LOGS_PARQUET = INPUT_DIR / "gps_logs.parquet"
SYNTHETIC_TRUTH_PARQUET = INPUT_DIR / "synthetic_truth.parquet"
SYNTHETIC_ROUTES_GEOJSON = INPUT_DIR / "synthetic_routes.geojson"

GENERATION_DIR = INPUT_DIR / "_generation_tmp"
SYNTHETIC_CAR_LUA = GENERATION_DIR / "car.lua"


# =========================================================
# Synthetic generation settings
# =========================================================

SEED = 42

# One public demo dataset.
SYNTHETIC_REGION_ID = "synthetic"
SYNTHETIC_YEAR = 2024
SYNTHETIC_DATASET_ROLE = "public_demo"

# Keep this reasonably small so the public demo is runnable.
N_SYNTHETIC_TRIPS_TARGET = 120
MIN_SYNTHETIC_TRIPS_REQUIRED = 40

N_VEHICLES = 12
A0_PROB = 0.10

# Autumn-only demo window, consistent with the simplified thesis analysis.
START_TIME = datetime(2024, 9, 1, 6, 0, tzinfo=timezone.utc)
DISPATCH_INTERVAL_HOURS = 7
DISPATCH_JITTER_MINUTES = 20

# Route acceptance.
SYNTHETIC_ROUTE_MIN_BASELINE_SEC = 2 * 60
SYNTHETIC_ROUTE_MAX_BASELINE_SEC = 30 * 60
SYNTHETIC_ROUTE_QUERY_MAX_ATTEMPTS = 60

# GPS simulation.
SYNTHETIC_GPS_SAMPLING_INTERVAL_RANGE_SEC = (1, 3)
SYNTHETIC_GPS_DROPOUT_PROB = 0.02
SYNTHETIC_GPS_NOISE_MAX_M = 8.0
SYNTHETIC_MIN_OBSERVATIONS = 30

# Realised ambulance time relative to baseline OSRM.
AMBULANCE_TIME_FACTOR_MEAN = 0.82
AMBULANCE_TIME_FACTOR_SD = 0.08
AMBULANCE_TIME_FACTOR_MIN = 0.55
AMBULANCE_TIME_FACTOR_MAX = 1.20

# Bounding box should lie inside the .osm.pbf placed in data/input_synthetic.
# Default is a Zuid-Holland/ZHZ-like area, matching the screenshot PBF.
SYNTHETIC_BBOX = {
    "min_lon": 4.45,
    "max_lon": 4.95,
    "min_lat": 51.65,
    "max_lat": 52.05,
}

# wait_for_osrm() uses the existing configs.OSRM_TEST_ROUTES keys.
# For the default Zuid-Holland extract, zhz is appropriate.
OSRM_READY_REGION_KEY = "ZHZ"


# =========================================================
# Random state
# =========================================================

rng = np.random.default_rng(SEED)
random.seed(SEED)


# =========================================================
# File helpers
# =========================================================

def find_input_pbf() -> Path:
    """
    Find the single regional .osm.pbf file used to generate synthetic routes.

    The file does not need to be named synthetic.osm.pbf. The cleaned paths.py
    can also work with any single .osm.pbf in data/input_synthetic/.
    """
    pbf_files = sorted(INPUT_DIR.glob("*.osm.pbf"))

    if not pbf_files:
        raise FileNotFoundError(
            f"No .osm.pbf file found in {INPUT_DIR}.\n"
            "Place one regional OSM extract in this folder before running this script."
        )

    if len(pbf_files) == 1:
        return pbf_files[0]

    preferred = INPUT_DIR / "synthetic.osm.pbf"
    if preferred.exists():
        return preferred

    raise FileExistsError(
        f"Expected exactly one .osm.pbf file in {INPUT_DIR}, found {len(pbf_files)}:\n"
        + "\n".join(f"  - {p.name}" for p in pbf_files)
        + "\nKeep only one file, or name the intended one synthetic.osm.pbf."
    )


def remove_previous_outputs() -> None:
    """
    Remove generated synthetic parquet/geojson outputs before regenerating.
    """
    for path in [
        DISPATCH_REGISTERS_PARQUET,
        GPS_LOGS_PARQUET,
        SYNTHETIC_TRUTH_PARQUET,
        SYNTHETIC_ROUTES_GEOJSON,
    ]:
        if path.exists():
            path.unlink()


def clean_generation_dir() -> None:
    if GENERATION_DIR.exists():
        shutil.rmtree(GENERATION_DIR)
    GENERATION_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Spatial helpers
# =========================================================

def sample_point_in_bbox() -> tuple[float, float]:
    lon = rng.uniform(SYNTHETIC_BBOX["min_lon"], SYNTHETIC_BBOX["max_lon"])
    lat = rng.uniform(SYNTHETIC_BBOX["min_lat"], SYNTHETIC_BBOX["max_lat"])
    return float(lon), float(lat)


def haversine_m(
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> float:
    r = 6_371_000.0

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def cumulative_distances(coords: list[list[float]]) -> list[float]:
    distances = [0.0]

    for i in range(1, len(coords)):
        lon1, lat1 = coords[i - 1]
        lon2, lat2 = coords[i]
        distances.append(distances[-1] + haversine_m(lon1, lat1, lon2, lat2))

    return distances


def interpolate_coords(
    coords: list[list[float]],
    fraction: float,
) -> tuple[float, float]:
    """
    Interpolate along a lon/lat route geometry.
    """
    if not coords:
        raise ValueError("Cannot interpolate an empty coordinate sequence.")

    if len(coords) == 1:
        lon, lat = coords[0]
        return float(lon), float(lat)

    fraction = max(0.0, min(1.0, fraction))

    cumulative = cumulative_distances(coords)
    total = cumulative[-1]

    if total <= 0:
        lon, lat = coords[0]
        return float(lon), float(lat)

    target = fraction * total

    for i in range(1, len(cumulative)):
        if cumulative[i] >= target:
            prev_d = cumulative[i - 1]
            next_d = cumulative[i]
            segment_fraction = 0.0 if next_d == prev_d else (target - prev_d) / (next_d - prev_d)

            lon1, lat1 = coords[i - 1]
            lon2, lat2 = coords[i]

            lon = lon1 + segment_fraction * (lon2 - lon1)
            lat = lat1 + segment_fraction * (lat2 - lat1)

            return float(lon), float(lat)

    lon, lat = coords[-1]
    return float(lon), float(lat)


def add_coordinate_noise(
    lon: float,
    lat: float,
    max_noise_m: float = SYNTHETIC_GPS_NOISE_MAX_M,
) -> tuple[float, float]:
    """
    Add small random coordinate noise in metres, approximated in lon/lat.
    """
    noise_m = rng.uniform(0, max_noise_m)
    angle = rng.uniform(0, 2 * math.pi)

    dx = noise_m * math.cos(angle)
    dy = noise_m * math.sin(angle)

    lat_offset = dy / 111_320
    lon_offset = dx / (111_320 * max(math.cos(math.radians(lat)), 0.1))

    return float(lon + lon_offset), float(lat + lat_offset)


# =========================================================
# Synthetic data generation
# =========================================================

def make_dispatch_time(index: int) -> datetime:
    jitter = int(rng.integers(-DISPATCH_JITTER_MINUTES, DISPATCH_JITTER_MINUTES + 1))
    return START_TIME + timedelta(hours=(index - 1) * DISPATCH_INTERVAL_HOURS, minutes=jitter)


def sample_valid_route() -> dict[str, Any] | None:
    """
    Sample random OD pairs until OSRM returns an acceptable route.
    """
    for _ in range(SYNTHETIC_ROUTE_QUERY_MAX_ATTEMPTS):
        origin_lon, origin_lat = sample_point_in_bbox()
        dest_lon, dest_lat = sample_point_in_bbox()

        route = query_osrm_route(
            origin_lon=origin_lon,
            origin_lat=origin_lat,
            dest_lon=dest_lon,
            dest_lat=dest_lat,
        )

        if route is None:
            continue

        duration = float(route["duration"])

        if SYNTHETIC_ROUTE_MIN_BASELINE_SEC <= duration <= SYNTHETIC_ROUTE_MAX_BASELINE_SEC:
            return {
                "origin_lon": origin_lon,
                "origin_lat": origin_lat,
                "dest_lon": dest_lon,
                "dest_lat": dest_lat,
                **route,
            }

    return None


def sample_urgency() -> str:
    return "A0" if rng.random() < A0_PROB else "A1"


def sample_realised_duration_sec(baseline_duration_sec: float) -> float:
    factor = float(
        np.clip(
            rng.normal(AMBULANCE_TIME_FACTOR_MEAN, AMBULANCE_TIME_FACTOR_SD),
            AMBULANCE_TIME_FACTOR_MIN,
            AMBULANCE_TIME_FACTOR_MAX,
        )
    )
    return max(30.0, baseline_duration_sec * factor)


def generate_gps_points(
    *,
    trip_id: str,
    vehicle_id: str,
    dispatch_time: datetime,
    realised_duration_sec: float,
    coords: list[list[float]],
) -> list[dict[str, Any]]:
    """
    Generate synthetic GPS observations along one route geometry.

    The saved GPS table deliberately omits trip_id. The methodology pipeline
    must reconstruct trip-to-GPS linkage from vehicle_id and timestamps.
    """
    rows = []
    elapsed = 0.0

    while elapsed <= realised_duration_sec:
        if rng.random() < SYNTHETIC_GPS_DROPOUT_PROB:
            elapsed += int(
                rng.integers(
                    SYNTHETIC_GPS_SAMPLING_INTERVAL_RANGE_SEC[0],
                    SYNTHETIC_GPS_SAMPLING_INTERVAL_RANGE_SEC[1] + 1,
                )
            )
            continue

        fraction = elapsed / realised_duration_sec
        lon, lat = interpolate_coords(coords, fraction)
        lon, lat = add_coordinate_noise(lon, lat)

        rows.append({
            "timestamp": dispatch_time + timedelta(seconds=float(elapsed)),
            "vehicle_id": vehicle_id,
            "vehicle_lat": lat,
            "vehicle_lon": lon,
            "_trip_id": trip_id,
        })

        elapsed += int(
            rng.integers(
                SYNTHETIC_GPS_SAMPLING_INTERVAL_RANGE_SEC[0],
                SYNTHETIC_GPS_SAMPLING_INTERVAL_RANGE_SEC[1] + 1,
            )
        )

    final_lon, final_lat = interpolate_coords(coords, 1.0)
    final_lon, final_lat = add_coordinate_noise(final_lon, final_lat)

    rows.append({
        "timestamp": dispatch_time + timedelta(seconds=float(realised_duration_sec)),
        "vehicle_id": vehicle_id,
        "vehicle_lat": final_lat,
        "vehicle_lon": final_lon,
        "_trip_id": trip_id,
    })

    if len(rows) < SYNTHETIC_MIN_OBSERVATIONS:
        return []

    return rows


def generate_synthetic_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Generate dispatch, GPS, truth, and route-geometry records.
    """
    dispatch_rows: list[dict] = []
    gps_rows: list[dict] = []
    truth_rows: list[dict] = []
    route_features: list[dict] = []

    print(f"\nGenerating {N_SYNTHETIC_TRIPS_TARGET} synthetic trips...")

    attempts = 0
    max_attempts = N_SYNTHETIC_TRIPS_TARGET * 4

    while len(dispatch_rows) < N_SYNTHETIC_TRIPS_TARGET and attempts < max_attempts:
        attempts += 1
        trip_number = len(dispatch_rows) + 1

        route = sample_valid_route()

        if route is None:
            print(f"  Attempt {attempts}: no valid OSRM route found.")
            continue

        trip_id = f"SYN_{SYNTHETIC_YEAR}_{trip_number:06d}"
        dispatch_id = f"D_{trip_id}"
        request_id = f"R_{trip_id}"
        vehicle_id = f"SYN_VEH_{((trip_number - 1) % N_VEHICLES) + 1:03d}"
        dispatch_time = make_dispatch_time(trip_number)

        coords = route["geometry"]["coordinates"]
        realised_duration_sec = sample_realised_duration_sec(float(route["duration"]))

        gps = generate_gps_points(
            trip_id=trip_id,
            vehicle_id=vehicle_id,
            dispatch_time=dispatch_time,
            realised_duration_sec=realised_duration_sec,
            coords=coords,
        )

        if not gps:
            print(f"  Attempt {attempts}: too few synthetic GPS observations.")
            continue

        urgency = sample_urgency()

        dispatch_rows.append({
            "timestamp": dispatch_time,
            "region_id": SYNTHETIC_REGION_ID,
            "vehicle_id": vehicle_id,
            "request_id": request_id,
            "dispatch_id": dispatch_id,
            "urgency": urgency,
            "incident_lat": float(route["dest_lat"]),
            "incident_lon": float(route["dest_lon"]),
        })

        gps_rows.extend(gps)

        truth_rows.append({
            "trip_id": trip_id,
            "dispatch_id": dispatch_id,
            "request_id": request_id,
            "vehicle_id": vehicle_id,
            "region_id": SYNTHETIC_REGION_ID,
            "dataset_role": SYNTHETIC_DATASET_ROLE,
            "year": SYNTHETIC_YEAR,
            "dispatch_time": dispatch_time,
            "origin_lat": float(route["origin_lat"]),
            "origin_lon": float(route["origin_lon"]),
            "incident_lat": float(route["dest_lat"]),
            "incident_lon": float(route["dest_lon"]),
            "baseline_osrm_time_sec": float(route["duration"]),
            "baseline_distance_m": float(route["distance"]),
            "synthetic_realised_time_sec": float(realised_duration_sec),
            "urgency": urgency,
        })

        route_features.append({
            "type": "Feature",
            "properties": {
                "trip_id": trip_id,
                "region_id": SYNTHETIC_REGION_ID,
                "year": SYNTHETIC_YEAR,
                "dataset_role": SYNTHETIC_DATASET_ROLE,
                "baseline_osrm_time_sec": float(route["duration"]),
                "baseline_distance_m": float(route["distance"]),
                "synthetic_realised_time_sec": float(realised_duration_sec),
            },
            "geometry": route["geometry"],
        })

        if trip_number % 25 == 0:
            print(f"  Generated {trip_number} trips.")

    if len(dispatch_rows) < MIN_SYNTHETIC_TRIPS_REQUIRED:
        raise RuntimeError(
            f"Generated only {len(dispatch_rows)} usable trips after {attempts} attempts. "
            f"Minimum required is {MIN_SYNTHETIC_TRIPS_REQUIRED}. "
            "Check that the bounding box lies inside the provided .osm.pbf."
        )

    print(f"Generated usable synthetic trips: {len(dispatch_rows)}")
    return dispatch_rows, gps_rows, truth_rows, route_features


# =========================================================
# Output writing
# =========================================================

def write_outputs(
    *,
    dispatch_rows: list[dict],
    gps_rows: list[dict],
    truth_rows: list[dict],
    route_features: list[dict],
) -> None:
    """
    Write pipeline-ready synthetic inputs and diagnostic truth files.
    """
    if not dispatch_rows:
        raise RuntimeError("No synthetic dispatch rows were generated.")

    if not gps_rows:
        raise RuntimeError("No synthetic GPS rows were generated.")

    dispatch_df = (
        pl.DataFrame(dispatch_rows)
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("region_id").cast(pl.String),
            pl.col("vehicle_id").cast(pl.String),
            pl.col("request_id").cast(pl.String),
            pl.col("dispatch_id").cast(pl.String),
            pl.col("urgency").cast(pl.String),
            pl.col("incident_lat").cast(pl.Float64),
            pl.col("incident_lon").cast(pl.Float64),
        ])
        .sort(["timestamp", "vehicle_id"])
    )

    gps_df = (
        pl.DataFrame(gps_rows)
        .with_columns([
            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
            pl.col("vehicle_id").cast(pl.String),
            pl.col("vehicle_lat").cast(pl.Float64),
            pl.col("vehicle_lon").cast(pl.Float64),
        ])
        .select([
            "timestamp",
            "vehicle_id",
            "vehicle_lat",
            "vehicle_lon",
        ])
        .sort(["timestamp", "vehicle_id"])
    )

    truth_df = (
        pl.DataFrame(truth_rows)
        .with_columns([
            pl.col("dispatch_time").cast(pl.Datetime("us", "UTC")),
        ])
        .sort(["dispatch_time", "vehicle_id"])
    )

    dispatch_df.write_parquet(DISPATCH_REGISTERS_PARQUET)
    gps_df.write_parquet(GPS_LOGS_PARQUET)
    truth_df.write_parquet(SYNTHETIC_TRUTH_PARQUET)

    geojson = {
        "type": "FeatureCollection",
        "features": route_features,
    }

    SYNTHETIC_ROUTES_GEOJSON.write_text(json.dumps(geojson), encoding="utf-8")

    print(f"\nSaved dispatch registers: {DISPATCH_REGISTERS_PARQUET}")
    print(f"Saved GPS logs:            {GPS_LOGS_PARQUET}")
    print(f"Saved synthetic truth:     {SYNTHETIC_TRUTH_PARQUET}")
    print(f"Saved route GeoJSON:       {SYNTHETIC_ROUTES_GEOJSON}")

    print("\nSynthetic input summary:")
    print(dispatch_df.select([
        pl.len().alias("dispatches"),
        pl.col("vehicle_id").n_unique().alias("vehicles"),
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
    ]))

    print(gps_df.select([
        pl.len().alias("gps_rows"),
        pl.col("vehicle_id").n_unique().alias("vehicles"),
        pl.col("timestamp").min().alias("min_timestamp"),
        pl.col("timestamp").max().alias("max_timestamp"),
    ]))


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    pbf_path = find_input_pbf()
    osrm_file = osrm_file_from_pbf(pbf_path)

    remove_previous_outputs()
    clean_generation_dir()

    print("Synthetic input generation")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Using OSM PBF: {pbf_path}")

    extract_default_car_lua(SYNTHETIC_CAR_LUA)

    run_osrm_preprocessing(
        lua_profile=SYNTHETIC_CAR_LUA,
        osm_pbf=pbf_path,
        osrm_file=osrm_file,
    )

    server = start_osrm_server(osrm_file=osrm_file)

    try:
        wait_for_osrm(region=OSRM_READY_REGION_KEY)

        dispatch_rows, gps_rows, truth_rows, route_features = generate_synthetic_rows()

    finally:
        stop_osrm_server(server)

    write_outputs(
        dispatch_rows=dispatch_rows,
        gps_rows=gps_rows,
        truth_rows=truth_rows,
        route_features=route_features,
    )

    print("\nDone. You can now run:")
    print("  python orchestrator.py")


if __name__ == "__main__":
    main()
