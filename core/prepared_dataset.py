from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from utils.constants import DEFAULT_PREDICTOR_FEATURES, RESPONSE_VARIABLES


def prepared_data_root(project_root: Path) -> Path:
    return project_root / "data" / "prepared" / "wgs84_1km"


def mean_model_raster_paths(project_root: Path, response_name: str) -> Dict[str, Path]:
    if response_name not in RESPONSE_VARIABLES:
        raise ValueError("Unsupported response variable: {0}".format(response_name))

    root = prepared_data_root(project_root)
    raster_map = {
        response_name: root / "response_means" / "{0}_mean.tif".format(response_name),
        "tree_diversity": root / "biodiversity" / "tree_diversity.tif",
        "structure_diversity": root / "biodiversity" / "structure_diversity.tif",
        "MAT": root / "climate" / "MAT.tif",
        "MAP": root / "climate" / "MAP.tif",
        "VPD": root / "climate" / "VPD.tif",
        "SM": root / "climate" / "SM.tif",
        "SSRD": root / "climate" / "SSRD.tif",
        "DEM": root / "terrain" / "DEM.tif",
        "slope": root / "terrain" / "slope.tif",
        "aspect": root / "terrain" / "aspect.tif",
    }
    missing = [name for name, path in raster_map.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared mean-model rasters for {0}: {1}".format(
                response_name,
                ", ".join(missing),
            )
        )
    return raster_map


def all_mean_model_raster_paths(project_root: Path) -> Dict[str, Path]:
    root = prepared_data_root(project_root)
    raster_map = {
        "GPP": root / "response_means" / "GPP_mean.tif",
        "LAI": root / "response_means" / "LAI_mean.tif",
        "VOD": root / "response_means" / "VOD_mean.tif",
        "tree_diversity": root / "biodiversity" / "tree_diversity.tif",
        "structure_diversity": root / "biodiversity" / "structure_diversity.tif",
        "MAT": root / "climate" / "MAT.tif",
        "MAP": root / "climate" / "MAP.tif",
        "VPD": root / "climate" / "VPD.tif",
        "SM": root / "climate" / "SM.tif",
        "SSRD": root / "climate" / "SSRD.tif",
        "DEM": root / "terrain" / "DEM.tif",
        "slope": root / "terrain" / "slope.tif",
        "aspect": root / "terrain" / "aspect.tif",
    }
    missing = [name for name, path in raster_map.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared mean-model rasters: {0}".format(", ".join(missing))
        )
    return raster_map


def mean_model_feature_names() -> List[str]:
    return list(DEFAULT_PREDICTOR_FEATURES)
