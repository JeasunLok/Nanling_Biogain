from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import rasterio

from utils.unit_conversions import SECONDS_PER_YEAR, convert_display_array


def prepared_raster_paths(project_root: Path) -> Dict[str, Path]:
    prepared_root = project_root / "data" / "prepared" / "wgs84_1km"
    return {
        "MAT": prepared_root / "climate" / "MAT.tif",
        "MAP": prepared_root / "climate" / "MAP.tif",
        "VPD": prepared_root / "climate" / "VPD.tif",
        "SSRD": prepared_root / "climate" / "SSRD.tif",
        "aspect": prepared_root / "terrain" / "aspect.tif",
    }


def needs_conversion(variable_name: str, array: np.ndarray, nodata: float) -> bool:
    valid_mask = np.isfinite(array)
    valid_mask &= ~np.isclose(array, nodata)
    if not np.any(valid_mask):
        return False

    values = array[valid_mask]
    if variable_name == "MAT":
        return float(np.nanmean(values)) > 200.0
    if variable_name == "MAP":
        return float(np.nanmax(values)) < 20.0
    if variable_name == "VPD":
        return float(np.nanmax(values)) < 2.0
    if variable_name == "SSRD":
        return float(np.nanmean(values)) > 1.0e6
    if variable_name == "aspect":
        return float(np.nanmin(values)) < 0.0 or float(np.nanmax(values)) > 360.0
    return False


def rewrite_raster(path: Path, variable_name: str) -> str:
    with rasterio.open(path) as dataset:
        profile = dataset.profile.copy()
        nodata = dataset.nodata
        if nodata is None:
            raise ValueError("Prepared raster is missing nodata: {0}".format(path))
        array = dataset.read(1).astype(np.float32)

    if not needs_conversion(variable_name, array, float(nodata)):
        return "skipped"

    converted = convert_display_array(variable_name, array, float(nodata))
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(converted.astype(np.float32), 1)
    return "converted"


def main() -> int:
    project_root = Path(__file__).resolve().parent
    statuses = {}
    for variable_name, path in prepared_raster_paths(project_root).items():
        statuses[variable_name] = rewrite_raster(path, variable_name)

    print("Prepared-unit normalization complete.")
    for variable_name, status in statuses.items():
        print("{0}: {1}".format(variable_name, status))
    print("SSRD conversion target:", "W/m-2 using annual seconds", float(SECONDS_PER_YEAR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
