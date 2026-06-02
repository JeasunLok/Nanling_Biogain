from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
import shapefile
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

from utils.unit_conversions import convert_display_array


TARGET_CRS = CRS.from_epsg(4326)
FLOAT_NODATA = -9999.0
GPP_SCALE_FACTOR = 0.1
VOD_YEAR_START = 2001
VOD_YEAR_END = 2022
GPP_INVALID_SOURCE_VALUES = (65534.0, 65535.0)

SPECIES_RICHNESS_SUBDATASET = (
    "netcdf:{path}:Species richness"
)
STRUCTURAL_DIVERSITY_SUBDATASET = (
    "netcdf:{path}:Structural diversity"
)

GPP_PATTERN = re.compile(r"GOSIF_GPP_(\d{4})_Mean\.tif$", re.IGNORECASE)
LAI_PATTERN = re.compile(r"LAI_yearly_(\d{4})\.tif$", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(\d{4})")


@dataclass(frozen=True)
class PreparationPaths:
    project_root: Path
    nanling_shp: Path
    species_richness_nc: Path
    structural_diversity_nc: Path
    gpp_dir: Path
    lai_dir: Path
    vod_dir: Path
    era5_dir: Path
    terrain_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class TargetGrid:
    transform: Affine
    width: int
    height: int
    crs: CRS
    bounds: Tuple[float, float, float, float]
    geometry: List[dict]


def default_preparation_paths(project_root: Path) -> PreparationPaths:
    return PreparationPaths(
        project_root=project_root,
        nanling_shp=project_root / "data" / "Nanling.shp",
        species_richness_nc=Path(r"E:\WorkSpace\R6_Nanling_Software\30177985\Species_richness.nc"),
        structural_diversity_nc=Path(r"E:\WorkSpace\R6_Nanling_Software\30177985\Structural_diversity.nc"),
        gpp_dir=Path(r"E:\Data\GOSIF_GPP\GPP_Annual"),
        lai_dir=Path(r"I:\data\LAI_Reprocessed_MODIS\annual"),
        vod_dir=Path(r"I:\data\VOD\VODCA_CXKu\VODCA_CXKu\daily_images_VODCA_CXKu"),
        era5_dir=Path(r"E:\Data\ERA5\Data"),
        terrain_dir=Path(r"E:\Data\ETOPO_2022\ETOPD_2022_surface_tif005D"),
        output_dir=project_root / "data" / "prepared" / "wgs84_1km",
    )


def prepare_nanling_wgs84_dataset(paths: PreparationPaths) -> Dict[str, object]:
    ensure_directories(paths.output_dir)

    geometry = read_shapefile_geometry(paths.nanling_shp)
    target_grid = build_target_grid(paths.species_richness_nc, geometry)

    manifest = {
        "target_crs": "EPSG:4326",
        "target_resolution": [target_grid.transform.a, abs(target_grid.transform.e)],
        "target_shape": [target_grid.height, target_grid.width],
        "outputs": {},
    }

    biodiversity_dir = paths.output_dir / "biodiversity"
    climate_dir = paths.output_dir / "climate"
    terrain_dir = paths.output_dir / "terrain"
    responses_dir = paths.output_dir / "responses"
    ensure_directories(biodiversity_dir, climate_dir, terrain_dir, responses_dir)

    manifest["outputs"]["biodiversity"] = prepare_biodiversity_layers(
        paths=paths,
        target_grid=target_grid,
        output_dir=biodiversity_dir,
    )
    manifest["outputs"]["climate"] = prepare_climate_layers(
        paths=paths,
        target_grid=target_grid,
        output_dir=climate_dir,
    )
    manifest["outputs"]["terrain"] = prepare_terrain_layers(
        paths=paths,
        target_grid=target_grid,
        output_dir=terrain_dir,
    )
    manifest["outputs"]["responses"] = prepare_response_layers(
        paths=paths,
        target_grid=target_grid,
        output_dir=responses_dir,
    )

    manifest_path = paths.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def prepare_biodiversity_layers(
    paths: PreparationPaths,
    target_grid: TargetGrid,
    output_dir: Path,
) -> Dict[str, str]:
    species_path = output_dir / "tree_diversity.tif"
    structural_path = output_dir / "structure_diversity.tif"

    crop_netcdf_subdataset(
        variable_name="tree_diversity",
        subdataset_path=SPECIES_RICHNESS_SUBDATASET.format(path=str(paths.species_richness_nc)),
        target_grid=target_grid,
        output_path=species_path,
    )
    crop_netcdf_subdataset(
        variable_name="structure_diversity",
        subdataset_path=STRUCTURAL_DIVERSITY_SUBDATASET.format(
            path=str(paths.structural_diversity_nc)
        ),
        target_grid=target_grid,
        output_path=structural_path,
    )
    return {
        "tree_diversity": str(species_path),
        "structure_diversity": str(structural_path),
    }


def prepare_climate_layers(
    paths: PreparationPaths,
    target_grid: TargetGrid,
    output_dir: Path,
) -> Dict[str, str]:
    variable_map = {
        "MAT": paths.era5_dir / "temperature_2m_2002_2022_mean.tif",
        "MAP": paths.era5_dir / "total_precipitation_2002_2022_mean.tif",
        "VPD": paths.era5_dir / "vapor_pressure_deficit_2002_2022_mean.tif",
        "SM": paths.era5_dir / "soil_water_content_3layers_2002_2022_mean.tif",
        "SSRD": paths.era5_dir / "surface_solar_radiation_downward_2002_2022_mean.tif",
    }
    return prepare_static_layers(variable_map, target_grid, output_dir)


def prepare_terrain_layers(
    paths: PreparationPaths,
    target_grid: TargetGrid,
    output_dir: Path,
) -> Dict[str, str]:
    variable_map = {
        "DEM": paths.terrain_dir / "ETOPD_2022_surface_DEM_tif005D.tif",
        "slope": paths.terrain_dir / "ETOPD_2022_surface_Slope_tif005D.tif",
        "aspect": paths.terrain_dir / "ETOPD_2022_surface_Aspect_tif005D.tif",
    }
    return prepare_static_layers(variable_map, target_grid, output_dir)


def prepare_response_layers(
    paths: PreparationPaths,
    target_grid: TargetGrid,
    output_dir: Path,
) -> Dict[str, object]:
    gpp_outputs = prepare_gpp_layers(paths.gpp_dir, target_grid, output_dir / "GPP")
    lai_outputs = prepare_lai_layers(paths.lai_dir, target_grid, output_dir / "LAI")
    vod_outputs = prepare_vod_layers(paths.vod_dir, target_grid, output_dir / "VOD")
    return {
        "GPP": gpp_outputs,
        "LAI": lai_outputs,
        "VOD": vod_outputs,
    }


def prepare_static_layers(
    variable_map: Dict[str, Path],
    target_grid: TargetGrid,
    output_dir: Path,
) -> Dict[str, str]:
    ensure_directories(output_dir)
    outputs: Dict[str, str] = {}
    for variable_name, source_path in variable_map.items():
        output_path = output_dir / "{0}.tif".format(variable_name)
        warp_source_to_target(
            variable_name=variable_name,
            source_path=source_path,
            target_grid=target_grid,
            output_path=output_path,
            scale_factor=1.0,
            invalid_source_values=None,
        )
        outputs[variable_name] = str(output_path)
    return outputs


def prepare_gpp_layers(source_dir: Path, target_grid: TargetGrid, output_dir: Path) -> Dict[str, str]:
    ensure_directories(output_dir)
    outputs: Dict[str, str] = {}
    for tif_path in sorted(source_dir.glob("GOSIF_GPP_*_Mean.tif")):
        match = GPP_PATTERN.search(tif_path.name)
        if match is None:
            continue
        year = match.group(1)
        output_path = output_dir / "GPP_{0}.tif".format(year)
        warp_source_to_target(
            variable_name="GPP",
            source_path=tif_path,
            target_grid=target_grid,
            output_path=output_path,
            scale_factor=GPP_SCALE_FACTOR,
            invalid_source_values=GPP_INVALID_SOURCE_VALUES,
        )
        outputs[year] = str(output_path)
    return outputs


def prepare_lai_layers(source_dir: Path, target_grid: TargetGrid, output_dir: Path) -> Dict[str, str]:
    ensure_directories(output_dir)
    outputs: Dict[str, str] = {}
    for tif_path in sorted(source_dir.glob("LAI_yearly_*.tif")):
        match = LAI_PATTERN.search(tif_path.name)
        if match is None:
            continue
        year = match.group(1)
        output_path = output_dir / "LAI_{0}.tif".format(year)
        warp_source_to_target(
            variable_name="LAI",
            source_path=tif_path,
            target_grid=target_grid,
            output_path=output_path,
            scale_factor=1.0,
            invalid_source_values=None,
        )
        outputs[year] = str(output_path)
    return outputs


def prepare_vod_layers(source_dir: Path, target_grid: TargetGrid, output_dir: Path) -> Dict[str, str]:
    ensure_directories(output_dir)
    outputs: Dict[str, str] = {}
    for year_dir in sorted(source_dir.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)
        if year < VOD_YEAR_START or year > VOD_YEAR_END:
            continue
        output_path = output_dir / "VOD_{0}.tif".format(year_dir.name)
        if not output_path.exists():
            annual_array, transform, crs = aggregate_vod_year(year_dir)
            write_reprojected_array(
                variable_name="VOD",
                array=annual_array,
                source_transform=transform,
                source_crs=crs,
                source_nodata=FLOAT_NODATA,
                target_grid=target_grid,
                output_path=output_path,
                scale_factor=1.0,
            )
        outputs[year_dir.name] = str(output_path)
    return outputs


def read_shapefile_geometry(shapefile_path: Path) -> List[dict]:
    reader = shapefile.Reader(str(shapefile_path))
    geometries = [shape_rec.shape.__geo_interface__ for shape_rec in reader.iterShapeRecords()]
    if not geometries:
        raise ValueError("No shapes found in {0}".format(shapefile_path))
    return geometries


def build_target_grid(species_richness_nc: Path, geometry: List[dict]) -> TargetGrid:
    subdataset_path = SPECIES_RICHNESS_SUBDATASET.format(path=str(species_richness_nc))
    with rasterio.open(subdataset_path) as dataset:
        source_bounds = dataset.bounds
        source_transform = dataset.transform

    geom_bounds = bounds_from_geometries(geometry)
    clipped_bounds = (
        max(source_bounds.left, geom_bounds[0]),
        max(source_bounds.bottom, geom_bounds[1]),
        min(source_bounds.right, geom_bounds[2]),
        min(source_bounds.top, geom_bounds[3]),
    )
    if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
        raise ValueError("Nanling boundary does not intersect the biodiversity grid.")

    window = window_from_bounds(clipped_bounds, source_transform)
    transform = rasterio.windows.transform(window, source_transform)
    width = int(window.width)
    height = int(window.height)
    bounds = rasterio.windows.bounds(window, source_transform)
    return TargetGrid(
        transform=transform,
        width=width,
        height=height,
        crs=TARGET_CRS,
        bounds=bounds,
        geometry=geometry,
    )


def bounds_from_geometries(geometry: List[dict]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for geom in geometry:
        collect_coordinates(geom["coordinates"], xs, ys)
    return (min(xs), min(ys), max(xs), max(ys))


def collect_coordinates(coords: object, xs: List[float], ys: List[float]) -> None:
    if isinstance(coords, (tuple, list)) and coords:
        first = coords[0]
        if isinstance(first, (float, int)):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            return
        for part in coords:
            collect_coordinates(part, xs, ys)


def window_from_bounds(bounds: Tuple[float, float, float, float], transform: Affine) -> Window:
    epsilon = 1e-9
    col_off = max(0, int(math.floor(((bounds[0] - transform.c) / transform.a) + epsilon)))
    row_off = max(
        0,
        int(math.floor(((transform.f - bounds[3]) / abs(transform.e)) + epsilon)),
    )
    col_end = int(math.ceil(((bounds[2] - transform.c) / transform.a) - epsilon))
    row_end = int(math.ceil(((transform.f - bounds[1]) / abs(transform.e)) - epsilon))
    return Window(
        col_off=col_off,
        row_off=row_off,
        width=max(1, col_end - col_off),
        height=max(1, row_end - row_off),
    )


def crop_netcdf_subdataset(
    variable_name: str,
    subdataset_path: str,
    target_grid: TargetGrid,
    output_path: Path,
) -> None:
    if output_path.exists():
        return
    with rasterio.open(subdataset_path) as dataset:
        source_crs = dataset.crs or TARGET_CRS
        window = window_from_bounds(target_grid.bounds, dataset.transform)
        data = dataset.read(1, window=window).astype(np.float32)
        transform = rasterio.windows.transform(window, dataset.transform)
        nodata = dataset.nodata if dataset.nodata is not None else FLOAT_NODATA

    mask = geometry_mask(
        target_grid.geometry,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        invert=True,
    )
    data = np.where(mask, data, nodata).astype(np.float32)
    if transform != target_grid.transform:
        write_reprojected_array(
            variable_name=variable_name,
            array=data,
            source_transform=transform,
            source_crs=source_crs,
            source_nodata=nodata,
            target_grid=target_grid,
            output_path=output_path,
            scale_factor=1.0,
        )
        return

    data = convert_display_array(variable_name, data, nodata)
    write_array(
        output_path=output_path,
        array=data,
        transform=target_grid.transform,
        crs=target_grid.crs,
        nodata=nodata,
    )


def warp_source_to_target(
    variable_name: str,
    source_path: Path,
    target_grid: TargetGrid,
    output_path: Path,
    scale_factor: float,
    invalid_source_values: Optional[Tuple[float, ...]],
) -> None:
    if output_path.exists():
        return
    with rasterio.open(source_path) as dataset:
        array = dataset.read(1).astype(np.float32)
        source_transform = dataset.transform
        source_crs = dataset.crs or TARGET_CRS
        source_nodata = dataset.nodata if dataset.nodata is not None else FLOAT_NODATA

    write_reprojected_array(
        variable_name=variable_name,
        array=array,
        source_transform=source_transform,
        source_crs=source_crs,
        source_nodata=source_nodata,
        target_grid=target_grid,
        output_path=output_path,
        scale_factor=scale_factor,
        invalid_source_values=invalid_source_values,
    )


def write_reprojected_array(
    variable_name: str,
    array: np.ndarray,
    source_transform: Affine,
    source_crs: CRS,
    source_nodata: float,
    target_grid: TargetGrid,
    output_path: Path,
    scale_factor: float,
    invalid_source_values: Optional[Tuple[float, ...]] = None,
) -> None:
    source_array = array.copy()
    invalid_mask = np.zeros(source_array.shape, dtype=bool)
    if source_nodata is not None:
        invalid_mask |= source_array == source_nodata
    if invalid_source_values:
        for invalid_value in invalid_source_values:
            invalid_mask |= np.isclose(source_array, invalid_value)
    source_array[invalid_mask] = np.float32(FLOAT_NODATA)

    destination = np.full((target_grid.height, target_grid.width), FLOAT_NODATA, dtype=np.float32)
    reproject(
        source=source_array,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=FLOAT_NODATA,
        dst_transform=target_grid.transform,
        dst_crs=target_grid.crs,
        dst_nodata=FLOAT_NODATA,
        resampling=Resampling.bilinear,
    )
    if scale_factor != 1.0:
        valid_mask = destination != FLOAT_NODATA
        destination[valid_mask] = destination[valid_mask] * np.float32(scale_factor)

    mask = geometry_mask(
        target_grid.geometry,
        out_shape=(target_grid.height, target_grid.width),
        transform=target_grid.transform,
        invert=True,
    )
    destination = np.where(mask, destination, FLOAT_NODATA).astype(np.float32)
    destination = convert_display_array(variable_name, destination, FLOAT_NODATA)
    write_array(output_path, destination, target_grid.transform, target_grid.crs, FLOAT_NODATA)


def write_array(
    output_path: Path,
    array: np.ndarray,
    transform: Affine,
    crs: CRS,
    nodata: float,
) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 1,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(array.astype(np.float32), 1)


def aggregate_vod_year(year_dir: Path) -> Tuple[np.ndarray, Affine, CRS]:
    daily_files = sorted(year_dir.glob("*.nc"))
    if not daily_files:
        raise ValueError("No daily VOD files found in {0}".format(year_dir))

    total_sum: Optional[np.ndarray] = None
    total_count: Optional[np.ndarray] = None
    source_transform: Optional[Affine] = None
    source_crs: Optional[CRS] = None

    for daily_path in daily_files:
        with rasterio.open(daily_path) as dataset:
            daily = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata if dataset.nodata is not None else FLOAT_NODATA
            valid_mask = np.isfinite(daily) & (daily != nodata)

            if total_sum is None:
                total_sum = np.zeros(daily.shape, dtype=np.float64)
                total_count = np.zeros(daily.shape, dtype=np.uint16)
                source_transform = dataset.transform
                source_crs = dataset.crs or TARGET_CRS

            total_sum[valid_mask] += daily[valid_mask]
            total_count[valid_mask] += 1

    assert total_sum is not None
    assert total_count is not None
    assert source_transform is not None
    assert source_crs is not None

    annual_mean = np.full(total_sum.shape, FLOAT_NODATA, dtype=np.float32)
    valid_pixels = total_count > 0
    annual_mean[valid_pixels] = (total_sum[valid_pixels] / total_count[valid_pixels]).astype(
        np.float32
    )
    return annual_mean, source_transform, source_crs


def ensure_directories(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
