from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject

from utils.constants import CANONICAL_VARIABLES, VARIABLE_ALIAS_MAP


@dataclass(frozen=True)
class RasterRecord:
    variable_name: str
    path: Path
    crs: str
    transform: Affine
    resolution: tuple[float, float]
    shape: tuple[int, int]
    nodata: Optional[float]
    dtype: str


class RasterRegistry:
    """Registry for aligned single-band input rasters."""

    def __init__(self) -> None:
        self.records: list[RasterRecord] = []

    def register(self, path: Path, variable_name_override: Optional[str] = None) -> RasterRecord:
        if not path.exists():
            raise FileNotFoundError(path)

        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise ValueError(f"{path.name} must be a single-band raster.")

            base_name = variable_name_override or normalize_variable_name(path.stem)
            variable_name = self._ensure_unique_name(base_name)
            record = RasterRecord(
                variable_name=variable_name,
                path=path,
                crs=str(dataset.crs),
                transform=dataset.transform,
                resolution=dataset.res,
                shape=(dataset.height, dataset.width),
                nodata=dataset.nodata,
                dtype=str(dataset.dtypes[0]),
            )

        self.records.append(record)
        return record

    def register_or_replace(
        self,
        path: Path,
        variable_name_override: Optional[str] = None,
    ) -> RasterRecord:
        base_name = variable_name_override or normalize_variable_name(path.stem)
        existing_index = self._find_record_index(base_name)
        if existing_index is not None:
            del self.records[existing_index]
        return self.register(path, variable_name_override=base_name)

    def clear(self) -> None:
        self.records = []

    def validate_alignment(self) -> str:
        if not self.records:
            raise ValueError("No rasters loaded.")

        reference = self.records[0]
        for record in self.records[1:]:
            if record.crs != reference.crs:
                raise ValueError(f"CRS mismatch: {record.variable_name} vs {reference.variable_name}")
            if record.transform != reference.transform:
                raise ValueError(
                    f"Transform mismatch: {record.variable_name} vs {reference.variable_name}"
                )
            if record.resolution != reference.resolution:
                raise ValueError(
                    f"Resolution mismatch: {record.variable_name} vs {reference.variable_name}"
                )
            if record.shape != reference.shape:
                raise ValueError(
                    f"Shape mismatch: {record.variable_name} vs {reference.variable_name}"
                )

        return (
            f"Alignment OK for {len(self.records)} raster(s). "
            f"Reference grid: crs={reference.crs}, shape={reference.shape}, "
            f"resolution={reference.resolution}"
        )

    def has_variable(self, variable_name: str) -> bool:
        return any(record.variable_name == variable_name for record in self.records)

    def preferred_reference_name(self) -> str:
        for candidate in ["GPP", "LAI", "VOD"]:
            if self.has_variable(candidate):
                return candidate
        raise ValueError("At least one of GPP, LAI, or VOD must be loaded as alignment reference.")

    def ensure_aligned(self, aligned_dir: Path, progress_callback=None) -> str:
        if not self.records:
            raise ValueError("No rasters loaded.")
        try:
            return self.validate_alignment()
        except ValueError:
            reference_name = self.preferred_reference_name()
            self.align_to_reference(reference_name, aligned_dir, progress_callback=progress_callback)
            return self.validate_alignment()

    def align_to_reference(self, reference_name: str, aligned_dir: Path, progress_callback=None) -> None:
        reference = self.get_record(reference_name)
        aligned_dir.mkdir(parents=True, exist_ok=True)
        updated_records: list[RasterRecord] = []
        total_records = max(1, len(self.records))
        for index, record in enumerate(self.records, start=1):
            if self._is_aligned_to_reference(record, reference):
                updated_records.append(record)
                if progress_callback is not None:
                    progress_callback(
                        int(index * 100 / total_records),
                        "Checked {0}".format(record.variable_name),
                    )
                continue
            output_path = aligned_dir / "{0}_aligned.tif".format(record.variable_name)
            self._write_aligned_raster(record, reference, output_path)
            updated_records.append(self._build_record(output_path, record.variable_name))
            if progress_callback is not None:
                progress_callback(
                    int(index * 100 / total_records),
                    "Aligned {0}".format(record.variable_name),
                )
        self.records = updated_records

    def get_record(self, variable_name: str) -> RasterRecord:
        for record in self.records:
            if record.variable_name == variable_name:
                return record
        raise KeyError(variable_name)

    def read_band(self, variable_name: str) -> np.ndarray:
        record = self.get_record(variable_name)
        with rasterio.open(record.path) as dataset:
            return dataset.read(1)

    def read_preview_subset(
        self,
        variable_name: str,
        geometry: Iterable[dict],
        bounds: Tuple[float, float, float, float],
    ) -> tuple[np.ndarray, Affine, Optional[float]]:
        record = self.get_record(variable_name)
        if record.crs not in {"EPSG:4326", "OGC:CRS84"}:
            raise ValueError(
                "Preview currently requires WGS84 rasters. "
                "Raster CRS was {0}.".format(record.crs)
            )

        raster_bounds = raster_bounds_from_record(record)
        clipped_bounds = (
            max(bounds[0], raster_bounds[0]),
            max(bounds[1], raster_bounds[1]),
            min(bounds[2], raster_bounds[2]),
            min(bounds[3], raster_bounds[3]),
        )
        if clipped_bounds[0] >= clipped_bounds[2] or clipped_bounds[1] >= clipped_bounds[3]:
            raise ValueError("Raster does not overlap the Nanling boundary.")

        with rasterio.open(record.path) as dataset:
            window = rasterio.windows.from_bounds(
                *clipped_bounds,
                transform=dataset.transform,
            ).round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                raise ValueError("Raster does not overlap the Nanling boundary.")
            array = dataset.read(1, window=window)
            transform = rasterio.windows.transform(window, dataset.transform)
            nodata = dataset.nodata

        mask = geometry_mask(
            list(geometry),
            out_shape=array.shape,
            transform=transform,
            invert=True,
        )
        masked_array = array.astype(np.float32, copy=False)
        fill_value = np.nan if nodata is None else nodata
        masked_array = np.where(mask, masked_array, fill_value)
        valid_mask = np.isfinite(masked_array)
        if nodata is not None:
            valid_mask &= ~np.isclose(masked_array, nodata)
        if not np.any(valid_mask):
            raise ValueError("Raster has no valid values inside the Nanling boundary.")
        return masked_array, transform, nodata

    def load_arrays(self) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        for record in self.records:
            arrays[record.variable_name] = self.read_band(record.variable_name)
        return arrays

    def nodata_map(self) -> Dict[str, Optional[float]]:
        return {record.variable_name: record.nodata for record in self.records}

    def export_reference_profile(self) -> dict:
        if not self.records:
            raise ValueError("No rasters loaded.")

        reference = self.get_record(self.preferred_reference_name())
        with rasterio.open(reference.path) as dataset:
            profile = dataset.profile.copy()

        profile.update(count=1)
        return profile

    def _ensure_unique_name(self, variable_name: str) -> str:
        existing_names = {record.variable_name for record in self.records}
        if variable_name not in existing_names:
            return variable_name

        suffix = 2
        while True:
            candidate = f"{variable_name}_{suffix}"
            if candidate not in existing_names:
                return candidate
            suffix += 1

    def _find_record_index(self, variable_name: str) -> Optional[int]:
        for index, record in enumerate(self.records):
            if record.variable_name == variable_name:
                return index
        return None

    def _is_aligned_to_reference(self, record: RasterRecord, reference: RasterRecord) -> bool:
        return (
            record.crs == reference.crs
            and record.transform == reference.transform
            and record.resolution == reference.resolution
            and record.shape == reference.shape
        )

    def _write_aligned_raster(
        self,
        record: RasterRecord,
        reference: RasterRecord,
        output_path: Path,
    ) -> None:
        with rasterio.open(record.path) as source_dataset:
            source_array = source_dataset.read(1)
            source_nodata = source_dataset.nodata
            destination = np.full(reference.shape, source_nodata if source_nodata is not None else -9999.0, dtype=np.float32)
            reproject(
                source=source_array,
                destination=destination,
                src_transform=source_dataset.transform,
                src_crs=source_dataset.crs,
                src_nodata=source_nodata,
                dst_transform=reference.transform,
                dst_crs=reference.crs,
                dst_nodata=source_nodata if source_nodata is not None else -9999.0,
                resampling=Resampling.bilinear,
            )
            profile = source_dataset.profile.copy()

        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            width=reference.shape[1],
            height=reference.shape[0],
            transform=reference.transform,
            crs=reference.crs,
            compress="lzw",
            nodata=source_nodata if source_nodata is not None else -9999.0,
        )
        with rasterio.open(output_path, "w", **profile) as dataset:
            dataset.write(destination.astype(np.float32), 1)

    def _build_record(self, path: Path, variable_name: str) -> RasterRecord:
        with rasterio.open(path) as dataset:
            return RasterRecord(
                variable_name=variable_name,
                path=path,
                crs=str(dataset.crs),
                transform=dataset.transform,
                resolution=dataset.res,
                shape=(dataset.height, dataset.width),
                nodata=dataset.nodata,
                dtype=str(dataset.dtypes[0]),
            )


def normalize_variable_name(name: str) -> str:
    mapped = VARIABLE_ALIAS_MAP.get(name, name)
    if mapped not in CANONICAL_VARIABLES:
        return mapped
    return mapped


def raster_bounds_from_record(record: RasterRecord) -> tuple[float, float, float, float]:
    left = record.transform.c
    top = record.transform.f
    right = left + record.transform.a * record.shape[1]
    bottom = top + record.transform.e * record.shape[0]
    return (
        min(left, right),
        min(bottom, top),
        max(left, right),
        max(bottom, top),
    )
