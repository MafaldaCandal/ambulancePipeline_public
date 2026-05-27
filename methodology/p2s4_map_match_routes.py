"""
p2s4_map_match_routes.py

Phase 2, step 4.

Purpose:
    Map-match retained realised GPS trajectories with OSRM Match.

Inputs:
    runs/RunXXX/outputs/clean_trajectories.parquet
    runs/RunXXX/ambulance_nl.lua

Outputs:
    runs/RunXXX/outputs/routes.parquet
    runs/RunXXX/outputs/routes.geojson
    runs/RunXXX/outputs/route_rejection_summary.parquet

Design:
    The raw OSRM Match status is retained, but rejected trips also receive
    human-readable rejection categories, explanations, and recommended checks.
"""

from __future__ import annotations

import json

import geopandas as gpd
import polars as pl

from paths import (
    ACCESS_LUA,
    CLEAN_TRAJECTORIES_PARQUET,
    ROUTES_GEOJSON,
    ROUTES_PARQUET,
    ROUTE_REJECTION_SUMMARY_PARQUET,
)

from utils.osrm_utils import (
    explain_match_rejection,
    query_osrm_match,
    run_osrm_preprocessing,
    start_osrm_server,
    stop_osrm_server,
    wait_for_osrm,
)


ROUTE_SCHEMA = {
    "trip_id": pl.String,
    "osrm_time_sec": pl.Float64,
    "distance_m": pl.Float64,
    "distance_km": pl.Float64,
    "n_turns": pl.Int64,
    "match_confidence": pl.Float64,
    "n_points_original": pl.Int64,
    "n_points_submitted": pl.Int64,
    "trace_duration_sec": pl.Float64,
    "trace_max_gap_sec": pl.Float64,
    "request_url_length": pl.Int64,
}

REJECTION_SCHEMA = {
    "trip_id": pl.String,
    "rejection_reason": pl.String,
    "rejection_category": pl.String,
    "rejection_summary": pl.String,
    "rejection_explanation": pl.String,
    "recommended_check": pl.String,
    "error_message": pl.String,
    "match_confidence": pl.Float64,
    "match_confidence_threshold": pl.Float64,
    "n_points_original": pl.Int64,
    "n_points_submitted": pl.Int64,
    "trace_duration_sec": pl.Float64,
    "trace_max_gap_sec": pl.Float64,
    "request_url_length": pl.Int64,
}

EXPECTED_REJECTION_REASONS = [
    "low_confidence",
    "request_failed",
    "request_timeout",
    "request_connection_error",
    "request_too_long",
    "request_http_error",
    "response_parse_error",
    "no_segment",
    "no_match",
    "too_few_points_for_match",
    "osrm_error",
    "unexpected_osrm_response",
]


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    if not CLEAN_TRAJECTORIES_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing clean trajectories: {CLEAN_TRAJECTORIES_PARQUET}"
        )

    if not ACCESS_LUA.exists():
        raise FileNotFoundError(
            f"Missing ambulance access Lua profile: {ACCESS_LUA}"
        )


# =========================================================
# Output helpers
# =========================================================

def write_empty_routes() -> None:
    pl.DataFrame(schema=ROUTE_SCHEMA).write_parquet(ROUTES_PARQUET)

    ROUTES_GEOJSON.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}),
        encoding="utf-8",
    )


def write_routes(rows: list[dict]) -> None:
    if not rows:
        write_empty_routes()
        return

    routes = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
    routes.drop(columns="geometry").to_parquet(ROUTES_PARQUET)
    routes.to_file(ROUTES_GEOJSON, driver="GeoJSON")


def write_rejections(rejections: list[dict]) -> pl.DataFrame:
    rejections_df = pl.DataFrame(
        rejections,
        schema=REJECTION_SCHEMA,
    )
    rejections_df.write_parquet(ROUTE_REJECTION_SUMMARY_PARQUET)
    return rejections_df


def rejection_reason_count_table(rejections_df: pl.DataFrame) -> pl.DataFrame:
    """
    Count raw OSRM/status rejection reasons, including expected reasons
    with zero rejected trips.
    """
    if rejections_df.height > 0 and "rejection_reason" in rejections_df.columns:
        observed_reasons = set(
            rejections_df.get_column("rejection_reason").to_list()
        )
    else:
        observed_reasons = set()

    all_reasons = sorted(set(EXPECTED_REJECTION_REASONS) | observed_reasons)

    template = pl.DataFrame(
        {"rejection_reason": all_reasons},
        schema={"rejection_reason": pl.String},
    )

    if rejections_df.height == 0:
        counts = pl.DataFrame(
            {
                "rejection_reason": [],
                "n_trips": [],
            },
            schema={
                "rejection_reason": pl.String,
                "n_trips": pl.UInt32,
            },
        )
    else:
        counts = (
            rejections_df
            .group_by("rejection_reason")
            .agg(pl.len().alias("n_trips"))
        )

    return (
        template
        .join(counts, on="rejection_reason", how="left")
        .with_columns(
            pl.col("n_trips")
            .fill_null(0)
            .cast(pl.UInt32)
        )
        .sort("rejection_reason")
    )


# =========================================================
# Clean trajectories -> matched routes
# =========================================================

def build_matched_routes() -> None:
    print("Loading clean trajectories...")
    check_inputs()

    df = pl.read_parquet(CLEAN_TRAJECTORIES_PARQUET).sort(
        ["trip_id", "timestamp"]
    )

    if df.is_empty():
        raise RuntimeError(
            f"Clean trajectories file is empty: {CLEAN_TRAJECTORIES_PARQUET}"
        )

    rows: list[dict] = []
    rejections: list[dict] = []

    total_trips = 0
    kept_trips = 0

    for trip in df.partition_by("trip_id"):
        total_trips += 1
        trip_id = trip["trip_id"][0]

        result = query_osrm_match(trip)

        if result["status"] != "ok":
            explanation = explain_match_rejection(result)

            rejections.append({
                "trip_id": trip_id,
                "rejection_reason": result["status"],
                "rejection_category": explanation["rejection_category"],
                "rejection_summary": explanation["rejection_summary"],
                "rejection_explanation": explanation["rejection_explanation"],
                "recommended_check": explanation["recommended_check"],
                "error_message": result.get("error_message"),
                "match_confidence": result.get("match_confidence"),
                "match_confidence_threshold": result.get("match_confidence_threshold"),
                "n_points_original": result.get("n_points_original"),
                "n_points_submitted": result.get("n_points_submitted"),
                "trace_duration_sec": result.get("trace_duration_sec"),
                "trace_max_gap_sec": result.get("trace_max_gap_sec"),
                "request_url_length": result.get("request_url_length"),
            })
            continue

        kept_trips += 1

        rows.append({
            "trip_id": trip_id,
            "osrm_time_sec": result["osrm_time_sec"],
            "distance_m": result["distance_m"],
            "distance_km": result["distance_km"],
            "n_turns": result["n_turns"],
            "match_confidence": result["match_confidence"],
            "n_points_original": result["n_points_original"],
            "n_points_submitted": result["n_points_submitted"],
            "trace_duration_sec": result["trace_duration_sec"],
            "trace_max_gap_sec": result["trace_max_gap_sec"],
            "request_url_length": result["request_url_length"],
            "geometry": result["geometry"],
        })

    ROUTES_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    write_routes(rows)
    rejections_df = write_rejections(rejections)

    print(f"Saved: {ROUTES_PARQUET}")
    print(f"Saved: {ROUTES_GEOJSON}")
    print(f"Saved: {ROUTE_REJECTION_SUMMARY_PARQUET}")

    print(f"Total clean trips submitted to OSRM: {total_trips}")
    print(f"Kept matched routes: {kept_trips}")
    print(f"Rejected routes: {len(rejections)}")

    print("\nRejected routes by raw OSRM/status reason:")

    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80):
        print(rejection_reason_count_table(rejections_df))


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    check_inputs()
    run_osrm_preprocessing(ACCESS_LUA)

    server = start_osrm_server()

    try:
        wait_for_osrm()
        build_matched_routes()
    finally:
        stop_osrm_server(server)


if __name__ == "__main__":
    main()