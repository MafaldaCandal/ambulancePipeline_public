"""
utils/osm_utils.py

OSM/geospatial integration utilities.

Purpose:
    Centralise road-network extraction and loading from the active OSM extract.

Design:
    Phase scripts should call these utilities instead of embedding GDAL,
    ogr2ogr, or road-loading logic directly.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

import geopandas as gpd

from paths import (
    PROJECT_ROOT,
    OSM_PBF,
    OUT_DIR,
    ROADS_GPKG,
)

from configs import (
    PROJECTED_CRS,
    GDAL_DOCKER_IMAGE,
)

from utils.osrm_utils import docker_path


# =========================================================
# OSM road-network extraction
# =========================================================

def create_roads_gpkg_if_needed(
    osm_pbf: Path = OSM_PBF,
    roads_gpkg: Path = ROADS_GPKG,
    overwrite: bool = False,
) -> None:
    """
    Create a GeoPackage of OSM road lines from an .osm.pbf extract.

    Uses ogr2ogr through the configured GDAL Docker image.

    Extracted layer:
        lines where highway IS NOT NULL

    Selected columns:
        osm_id, highway
    """
    if not osm_pbf.exists():
        raise FileNotFoundError(f"Missing OSM PBF file: {osm_pbf}")

    if roads_gpkg.exists() and not overwrite:
        print(f"Using existing road file: {roads_gpkg}")
        return

    print("Creating road GeoPackage with ogr2ogr...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roads_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if roads_gpkg.exists() and overwrite:
        roads_gpkg.unlink()

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{PROJECT_ROOT}:/data",
        GDAL_DOCKER_IMAGE,
        "ogr2ogr",
        "-f", "GPKG",
        docker_path(roads_gpkg),
        docker_path(osm_pbf),
        "lines",
        "-where", "highway IS NOT NULL",
        "-select", "osm_id,highway",
    ]

    subprocess.run(cmd, check=True)


# =========================================================
# OSM road loading / normalisation
# =========================================================

def normalise_highway(value: Any) -> str | None:
    """
    Normalise OSM highway values to a single string.

    Some readers return list-like or comma-separated highway values. The
    feature-extraction phase expects one primary class per edge.
    """
    if isinstance(value, list):
        return str(value[0]) if value else None

    if isinstance(value, str) and "," in value:
        return value.split(",")[0].strip()

    if value is None:
        return None

    return str(value)


def load_osm_edges(
    roads_gpkg: Path = ROADS_GPKG,
    projected_crs: str = PROJECTED_CRS,
    create_if_needed: bool = True,
) -> gpd.GeoDataFrame:
    """
    Load OSM road edges and project them for metric distance operations.

    Returns:
        GeoDataFrame with at least:
            id, highway, geometry

    CRS:
        Returned in configs.PROJECTED_CRS, usually EPSG:28992.
    """
    if create_if_needed:
        create_roads_gpkg_if_needed(roads_gpkg=roads_gpkg)

    if not roads_gpkg.exists():
        raise FileNotFoundError(f"Missing road GeoPackage: {roads_gpkg}")

    edges = gpd.read_file(roads_gpkg)

    if "osm_id" in edges.columns:
        edges = edges.rename(columns={"osm_id": "id"})

    required = ["highway", "geometry"]
    missing = [col for col in required if col not in edges.columns]

    if missing:
        raise ValueError(
            f"Road layer is missing required columns: {missing}. "
            f"Available columns: {list(edges.columns)}"
        )

    keep_cols = [
        col
        for col in ["id", "highway", "geometry"]
        if col in edges.columns
    ]

    edges = edges[keep_cols].copy()
    edges = edges.dropna(subset=["highway", "geometry"])

    if edges.empty:
        raise ValueError(
            f"Road GeoPackage contains no usable highway geometries: {roads_gpkg}"
        )

    if edges.crs is None:
        edges = edges.set_crs("EPSG:4326")

    edges["highway"] = edges["highway"].apply(normalise_highway)
    edges = edges.dropna(subset=["highway"])

    edges = edges.to_crs(projected_crs)

    return edges