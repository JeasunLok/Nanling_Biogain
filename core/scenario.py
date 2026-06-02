from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Dict, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestRegressor

from core.cancellation import OperationCancelledError

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_prediction_table(
    arrays: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    nodata_map: Dict[str, Optional[float]],
) -> tuple[pd.DataFrame, np.ndarray]:
    if not feature_names:
        raise ValueError("No feature names provided.")

    first_shape = next(iter(arrays.values())).shape
    valid_mask = np.ones(first_shape, dtype=bool)
    for name in feature_names:
        if name not in arrays:
            raise KeyError("Missing feature raster: {0}".format(name))
        valid_mask &= np.isfinite(arrays[name])
        nodata = nodata_map.get(name)
        if nodata is not None:
            valid_mask &= arrays[name] != nodata

    if not np.any(valid_mask):
        raise ValueError("No valid pixels remain for prediction.")

    prediction_table = pd.DataFrame(
        {name: np.asarray(arrays[name][valid_mask], dtype=float) for name in feature_names}
    )
    return prediction_table, valid_mask


def run_plus_one_scenario(
    model: RandomForestRegressor,
    arrays: Dict[str, np.ndarray],
    feature_names: Sequence[str],
    nodata_map: Dict[str, Optional[float]],
    adjusted_feature: str = "tree_diversity",
    delta: float = 1.0,
    progress_callback=None,
    cancel_event: Optional[Event] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_cancel(cancel_event)
    _emit_progress(progress_callback, 10, "Building prediction table")
    prediction_table, valid_mask = build_prediction_table(arrays, feature_names, nodata_map)
    _check_cancel(cancel_event)
    _emit_progress(progress_callback, 35, "Running baseline prediction")
    baseline_prediction = model.predict(prediction_table)

    scenario_table = prediction_table.copy()
    if adjusted_feature not in scenario_table.columns:
        raise KeyError("Missing scenario feature: {0}".format(adjusted_feature))
    scenario_table.loc[:, adjusted_feature] = scenario_table.loc[:, adjusted_feature] + delta
    _check_cancel(cancel_event)
    _emit_progress(progress_callback, 65, "Running biodiversity gain prediction")
    scenario_prediction = model.predict(scenario_table)

    _check_cancel(cancel_event)
    _emit_progress(progress_callback, 85, "Building gain rasters")
    delta_raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    delta_raster[valid_mask] = scenario_prediction - baseline_prediction
    percent_raster = np.full(valid_mask.shape, np.nan, dtype=np.float32)
    nonzero_mask = np.abs(baseline_prediction) > 1e-12
    percent_values = np.zeros_like(baseline_prediction, dtype=np.float32)
    percent_values[nonzero_mask] = (
        (scenario_prediction[nonzero_mask] - baseline_prediction[nonzero_mask])
        / baseline_prediction[nonzero_mask]
        * 100.0
    ).astype(np.float32)
    percent_raster[valid_mask] = percent_values
    _emit_progress(progress_callback, 100, "Biodiversity gain evaluation complete")
    return delta_raster, percent_raster, valid_mask


def export_prediction_raster(output_path: Path, profile: dict, array: np.ndarray) -> Path:
    write_profile = profile.copy()
    write_profile.update(dtype="float32", nodata=np.nan)

    with rasterio.open(output_path, "w", **write_profile) as dataset:
        dataset.write(array.astype(np.float32), 1)
    return output_path


def export_preview_png(output_path: Path, array: np.ndarray, title: str) -> Path:
    masked = np.ma.masked_invalid(array)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    image = ax.imshow(masked, cmap="viridis")
    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _emit_progress(progress_callback, percent: int, message: str) -> None:
    if progress_callback is not None:
        progress_callback(percent, message)


def _check_cancel(cancel_event: Optional[Event]) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelledError("Operation cancelled.")
