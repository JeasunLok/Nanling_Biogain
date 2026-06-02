from __future__ import annotations

import json
from pathlib import Path

from core.modeling import train_linear_regression, train_ridge_regression
from core.prepared_dataset import mean_model_feature_names, mean_model_raster_paths
from core.raster_registry import RasterRegistry
from core.sample_table import build_sample_table
from utils.constants import RESPONSE_VARIABLES
from utils.paths import ensure_output_dir


def main() -> int:
    project_root = Path(__file__).resolve().parent
    output_dir = ensure_output_dir() / "linear_mean_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {}
    for response_name in RESPONSE_VARIABLES:
        registry = RasterRegistry()
        raster_map = mean_model_raster_paths(project_root, response_name)
        for variable_name, path in raster_map.items():
            registry.register(path, variable_name_override=variable_name)

        registry.validate_alignment()
        sample_table = build_sample_table(registry.load_arrays(), registry.nodata_map())
        feature_names = mean_model_feature_names()

        _, ols_result = train_linear_regression(
            sample_table=sample_table,
            response_name=response_name,
            feature_names=feature_names,
            test_size=0.2,
            random_state=42,
        )
        _, ridge_result = train_ridge_regression(
            sample_table=sample_table,
            response_name=response_name,
            feature_names=feature_names,
            test_size=0.2,
            random_state=42,
            alpha=1.0,
        )

        response_result = {
            "sample_table_rows": int(len(sample_table)),
            "OLS": serialize_linear_result(ols_result),
            "Ridge": serialize_linear_result(ridge_result),
        }
        run_manifest[response_name] = response_result

        response_path = output_dir / "{0}_linear_models.json".format(response_name)
        response_path.write_text(
            json.dumps(response_result, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    manifest_path = output_dir / "linear_mean_model_runs.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print("Linear mean-model runs written to:", manifest_path)
    return 0


def serialize_linear_result(result: object) -> dict:
    return {
        "model_name": result.model_name,
        "response_name": result.response_name,
        "feature_names": result.feature_names,
        "train_size": result.train_size,
        "test_size": result.test_size,
        "metrics": {
            "r2": result.r2,
            "rmse": result.rmse,
            "mae": result.mae,
        },
        "intercept": result.intercept,
        "coefficients": result.coefficients,
    }


if __name__ == "__main__":
    raise SystemExit(main())
