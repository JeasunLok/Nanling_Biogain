from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio


YEAR_PATTERN = re.compile(r"_(\d{4})\.tif$", re.IGNORECASE)


def compute_multi_year_mean(
    input_dir: Path,
    output_path: Path,
) -> Dict[str, object]:
    tif_paths = sorted(input_dir.glob("*.tif"))
    if not tif_paths:
        raise ValueError("No TIFF files found in {0}".format(input_dir))

    total_sum: Optional[np.ndarray] = None
    total_count: Optional[np.ndarray] = None
    profile: Optional[dict] = None
    years: List[str] = []

    for tif_path in tif_paths:
        year = parse_year_from_name(tif_path.name)
        if year is not None:
            years.append(year)

        with rasterio.open(tif_path) as dataset:
            array = dataset.read(1).astype(np.float32)
            nodata = dataset.nodata

            if profile is None:
                profile = dataset.profile.copy()
                total_sum = np.zeros(array.shape, dtype=np.float64)
                total_count = np.zeros(array.shape, dtype=np.uint16)

            valid_mask = np.isfinite(array)
            if nodata is not None:
                valid_mask &= array != nodata

            assert total_sum is not None
            assert total_count is not None
            total_sum[valid_mask] += array[valid_mask]
            total_count[valid_mask] += 1

    assert profile is not None
    assert total_sum is not None
    assert total_count is not None

    mean_array = np.full(total_sum.shape, np.float32(-9999.0), dtype=np.float32)
    valid_pixels = total_count > 0
    mean_array[valid_pixels] = (total_sum[valid_pixels] / total_count[valid_pixels]).astype(
        np.float32
    )

    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dataset:
        dataset.write(mean_array, 1)

    return {
        "output_path": str(output_path),
        "source_count": len(tif_paths),
        "years": years,
    }


def parse_year_from_name(filename: str) -> Optional[str]:
    match = YEAR_PATTERN.search(filename)
    if match is None:
        return None
    return match.group(1)


def write_mean_manifest(output_dir: Path, manifest: Dict[str, object]) -> Path:
    output_path = output_dir / "multi_year_mean_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path
