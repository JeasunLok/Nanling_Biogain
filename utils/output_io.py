from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from core.modeling import ModelRunResult
from utils.paths import ensure_output_dir


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(data: Dict[str, Any], filename: str) -> Path:
    output_dir = ensure_output_dir()
    output_path = output_dir / filename
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    return output_path


def export_sample_table_summary(
    sample_table: pd.DataFrame,
    raster_names: list[str],
    label: str = "sample_table_summary",
) -> Path:
    summary = {
        "timestamp": timestamp_tag(),
        "row_count": int(len(sample_table)),
        "column_count": int(len(sample_table.columns)),
        "columns": list(sample_table.columns),
        "source_rasters": raster_names,
    }
    return write_json(summary, "{0}.json".format(label))


def export_model_run_result(
    result: ModelRunResult,
    response_name: str,
    random_state: int,
    test_size: float,
) -> Path:
    payload = {
        "timestamp": timestamp_tag(),
        "response_name": response_name,
        "feature_names": result.feature_names,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "metrics": {
            "r2": result.r2,
            "rmse": result.rmse,
            "mae": result.mae,
        },
        "feature_importance": result.feature_importance,
        "split_config": {
            "random_state": random_state,
            "test_size_fraction": test_size,
        },
    }
    return write_json(payload, "{0}_model_run_summary.json".format(response_name))


def export_feature_importance_csv(result: ModelRunResult, response_name: str) -> Path:
    rows = [
        {"feature_name": name, "importance": importance}
        for name, importance in result.feature_importance.items()
    ]
    frame = pd.DataFrame(rows).sort_values("importance", ascending=False)
    output_dir = ensure_output_dir()
    output_path = output_dir / "{0}_feature_importance.csv".format(response_name)
    frame.to_csv(output_path, index=False)
    return output_path


def export_scenario_summary(
    adjusted_feature: str,
    delta: float,
    response_name: str,
    absolute_output_raster: Path,
    percent_output_raster: Path,
    mean_gain: float,
    mean_percent_gain: float,
    valid_pixel_count: int,
) -> Path:
    payload = {
        "timestamp": timestamp_tag(),
        "response_name": response_name,
        "scenario": {
            "adjusted_feature": adjusted_feature,
            "delta": delta,
        },
        "absolute_output_raster": str(absolute_output_raster),
        "percent_output_raster": str(percent_output_raster),
        "mean_gain": mean_gain,
        "mean_percent_gain": mean_percent_gain,
        "valid_pixel_count": valid_pixel_count,
    }
    return write_json(
        payload,
        "{0}_{1}_scenario_summary.json".format(response_name, adjusted_feature),
    )
