from __future__ import annotations

from typing import Optional

import numpy as np

SECONDS_PER_YEAR = np.float32(365.25 * 24 * 3600)


def convert_display_array(
    variable_name: str,
    array: np.ndarray,
    nodata: Optional[float],
) -> np.ndarray:
    converted = array.astype(np.float32, copy=True)
    valid_mask = np.isfinite(converted)
    if nodata is not None:
        valid_mask &= ~np.isclose(converted, nodata)

    if not np.any(valid_mask):
        return converted

    if variable_name == "MAT":
        converted[valid_mask] = converted[valid_mask] - np.float32(273.15)
    elif variable_name == "MAP":
        converted[valid_mask] = converted[valid_mask] * np.float32(1000.0)
    elif variable_name == "VPD":
        converted[valid_mask] = converted[valid_mask] * np.float32(10.0)
    elif variable_name == "SSRD":
        converted[valid_mask] = converted[valid_mask] / SECONDS_PER_YEAR
    elif variable_name == "aspect":
        converted[valid_mask] = np.mod(converted[valid_mask], np.float32(360.0))

    return converted
