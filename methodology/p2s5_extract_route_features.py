"""
p2s5_extract_route_features.py

Phase 2, step 5.

Purpose:
    Convert map-matched realised route geometries into trip-level route
    features used for residual modelling.

Inputs:
    runs/RunXXX/outputs/routes.parquet
    runs/RunXXX/outputs/routes.geojson

Outputs:
    runs/RunXXX/outputs/route_features.parquet
    runs/RunXXX/outputs/route_features.csv

Method:
    1. Load projected OSM road edges through utils.osm_utils.
    2. Load projected matched route geometries.
    3. Densify each route into short segments.
    4. Assign each segment to the nearest OSM road class.
    5. Aggregate segment lengths into km_* road-class variables.
    6. Attach route-level totals such as distance_km and n_turns.

Notes:
    Segments that cannot be assigned to a known OSM road class within the
    configured matching distance are retained as km_unknown for diagnostics.
"""

import geopandas as gpd
import polars as pl
from shapely.geometry import LineString

from paths import (
    ROUTES_GEOJSON,
    ROUTES_PARQUET,
    ROUTE_FEATURES_PARQUET,
    ROUTE_FEATURES_CSV,
)

from configs import (
    PROJECTED_CRS,
    ROAD_CLASSES,
    DENSIFY_INTERVAL_M,
    MAX_ROAD_MATCH_DISTANCE_M,
)

from utils.osm_utils import load_osm_edges


# =========================================================
# Input checks
# =========================================================

def check_inputs() -> None:
    """
    Validate required Phase 2 inputs.
    """
    if not ROUTES_GEOJSON.exists():
        raise FileNotFoundError(f"Missing routes GeoJSON: {ROUTES_GEOJSON}")

    if not ROUTES_PARQUET.exists():
        raise FileNotFoundError(f"Missing routes parquet: {ROUTES_PARQUET}")


# =========================================================
# Route loading
# =========================================================

def load_routes() -> gpd.GeoDataFrame:
    """
    Load matched route geometries and project them for metric operations.
    """

    routes = gpd.read_file(ROUTES_GEOJSON)

    if "trip_id" not in routes.columns:
        raise ValueError("Route GeoJSON must contain a trip_id column.")

    if "geometry" not in routes.columns:
        raise ValueError("Route GeoJSON must contain a geometry column.")

    routes = routes[["trip_id", "geometry"]].copy()
    routes = routes.dropna(subset=["trip_id", "geometry"])

    if routes.empty:
        raise ValueError(f"Route GeoJSON contains no usable routes: {ROUTES_GEOJSON}")

    # OSRM GeoJSON geometries are lon/lat.
    if routes.crs is None:
        routes = routes.set_crs("EPSG:4326")
    else:
        routes = routes.set_crs("EPSG:4326", allow_override=True)

    return routes.to_crs(PROJECTED_CRS)


# =========================================================
# Route segmentation
# =========================================================

def route_to_segments(
    trip_id: str,
    line: LineString,
    interval_m: float = DENSIFY_INTERVAL_M,
) -> list[dict]:
    """
    Convert one route LineString into short consecutive segments.

    Each segment is later assigned to the nearest OSM road class.
    """

    rows = []

    if line is None or line.length == 0:
        return rows

    distances = list(range(0, int(line.length), int(interval_m)))

    if not distances or distances[-1] < line.length:
        distances.append(int(line.length))

    points = [line.interpolate(d) for d in distances]

    for i in range(len(points) - 1):
        segment = LineString([points[i], points[i + 1]])

        if segment.length == 0:
            continue

        rows.append({
            "trip_id": trip_id,
            "segment_id": i,
            "segment_length_m": segment.length,
            "geometry": segment,
        })

    return rows


def create_route_segments(routes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create short route segments for all matched routes.
    """

    rows = []

    for _, row in routes.iterrows():
        rows.extend(
            route_to_segments(
                trip_id=row["trip_id"],
                line=row.geometry,
                interval_m=DENSIFY_INTERVAL_M,
            )
        )

    if not rows:
        return gpd.GeoDataFrame(
            columns=[
                "trip_id",
                "segment_id",
                "segment_length_m",
                "geometry",
            ],
            geometry="geometry",
            crs=PROJECTED_CRS,
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=PROJECTED_CRS)


# =========================================================
# Road-class assignment
# =========================================================

def assign_segments_to_road_class(
    segments: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Assign each route segment to the nearest OSM road class.

    Segments without a nearby OSM road within MAX_ROAD_MATCH_DISTANCE_M are
    labelled as unknown. This is retained downstream as km_unknown.
    """

    if segments.empty:
        return gpd.GeoDataFrame(
            columns=[
                "trip_id",
                "segment_id",
                "segment_length_m",
                "highway",
                "match_distance_m",
                "geometry",
            ],
            geometry="geometry",
            crs=PROJECTED_CRS,
        )

    midpoints = segments.copy()
    midpoints["geometry"] = midpoints.geometry.interpolate(0.5, normalized=True)

    matched = gpd.sjoin_nearest(
        midpoints,
        edges[["highway", "geometry"]],
        how="left",
        max_distance=MAX_ROAD_MATCH_DISTANCE_M,
        distance_col="match_distance_m",
    )

    matched = matched.drop(columns=["index_right"], errors="ignore")
    matched["highway"] = matched["highway"].fillna("unknown")

    return matched


def create_road_class_table(
    matched_segments: gpd.GeoDataFrame,
) -> pl.DataFrame:
    """
    Aggregate segment lengths by road class.

    All configured ROAD_CLASSES are guaranteed to exist in the output.
    km_unknown is also retained to make unassigned route distance visible
    in diagnostics.
    """

    required_cols = [f"km_{cls}" for cls in ROAD_CLASSES] + ["km_unknown"]

    if matched_segments.empty:
        return pl.DataFrame(
            {
                "trip_id": [],
                **{col: [] for col in required_cols},
            }
        )

    required = ["trip_id", "highway", "segment_length_m"]
    missing = [col for col in required if col not in matched_segments.columns]

    if missing:
        raise ValueError(
            f"Matched segment table is missing required columns: {missing}"
        )

    df = pl.from_pandas(
        matched_segments[
            [
                "trip_id",
                "highway",
                "segment_length_m",
            ]
        ]
    )

    known_classes = set(ROAD_CLASSES)

    df = df.with_columns([
        pl.when(pl.col("highway").is_in(list(known_classes)))
        .then(pl.col("highway"))
        .otherwise(pl.lit("unknown"))
        .alias("road_class")
    ])

    grouped = df.group_by(["trip_id", "road_class"]).agg(
        (pl.col("segment_length_m").sum() / 1000).alias("km")
    )

    out = grouped.pivot(
        values="km",
        index="trip_id",
        on="road_class",
        aggregate_function="first",
    )  # type: ignore

    out = out.fill_null(0.0)

    rename_map = {
        col: f"km_{col}"
        for col in out.columns
        if col != "trip_id" and not col.startswith("km_")
    }

    out = out.rename(rename_map)

    for col in required_cols:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(col))

    return out.select(["trip_id", *required_cols])


def attach_route_totals(
    road_class_table: pl.DataFrame,
    routes_summary: pl.DataFrame,
) -> pl.DataFrame:
    """
    Attach route-level totals such as distance_km and n_turns to features.
    """

    required = ["trip_id", "distance_km", "n_turns"]
    missing = [col for col in required if col not in routes_summary.columns]

    if missing:
        raise ValueError(f"routes_summary is missing required columns: {missing}")

    road_cols = [f"km_{cls}" for cls in ROAD_CLASSES] + ["km_unknown"]

    out = (
        routes_summary
        .select(required)
        .join(road_class_table, on="trip_id", how="left")
    )

    for col in road_cols:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(col))

    out = out.with_columns([
        pl.col(col).fill_null(0.0)
        for col in road_cols
    ])

    return (
        out
        .select(["trip_id", "distance_km", "n_turns", *road_cols])
        .sort("trip_id")
    )


# =========================================================
# Route features
# =========================================================

def create_route_features() -> pl.DataFrame:
    """
    Create trip-level route features from matched routes and OSM road classes.
    """

    check_inputs()

    print("Loading OSM edges...")
    edges = load_osm_edges()

    print("Loading route geometries...")
    routes = load_routes()

    print(f"Densifying routes every {DENSIFY_INTERVAL_M} m...")
    segments = create_route_segments(routes)

    if segments.empty:
        raise RuntimeError(
            "No route segments were created from matched route geometries. "
            "Check routes.geojson before continuing."
        )

    print("Assigning route segments to road classes...")
    matched_segments = assign_segments_to_road_class(segments, edges)

    print("Aggregating road-class features...")
    road_table = create_road_class_table(matched_segments)

    print("Loading route-level totals...")
    routes_summary = (
        pl.read_parquet(ROUTES_PARQUET)
        .select([
            "trip_id",
            "distance_km",
            "n_turns",
        ])
    )

    print("Creating route feature table...")
    return attach_route_totals(
        road_class_table=road_table,
        routes_summary=routes_summary,
    )


# =========================================================
# Main execution
# =========================================================

def main() -> None:
    route_features = create_route_features()

    ROUTE_FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    route_features.write_parquet(ROUTE_FEATURES_PARQUET)
    route_features.write_csv(ROUTE_FEATURES_CSV)

    print(f"Saved: {ROUTE_FEATURES_PARQUET}")
    print(f"Saved: {ROUTE_FEATURES_CSV}")


if __name__ == "__main__":
    main()