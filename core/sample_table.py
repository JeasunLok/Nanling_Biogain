from __future__ import annotations

from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd


def build_sample_table(
    arrays: Mapping[str, np.ndarray],
    nodata_map: Mapping[str, Optional[Union[float, int]]],
) -> pd.DataFrame:
    """Build a pixel-level table using a shared valid-data mask."""

    if not arrays:
        raise ValueError("No arrays provided.")

    first_shape = next(iter(arrays.values())).shape
    for name, array in arrays.items():
        if array.shape != first_shape:
            raise ValueError(f"Shape mismatch for {name}: expected {first_shape}, got {array.shape}")
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D single-band array.")

    valid_mask = np.ones(first_shape, dtype=bool)
    for name, array in arrays.items():
        nodata = nodata_map.get(name)
        valid_mask &= np.isfinite(array)
        if nodata is not None:
            valid_mask &= array != nodata

    if not np.any(valid_mask):
        raise ValueError("No valid pixels remain after nodata filtering.")

    data = {
        name: np.asarray(array[valid_mask], dtype=float)
        for name, array in arrays.items()
    }
    return pd.DataFrame(data)
