from __future__ import annotations

import json
from pathlib import Path

from core.modeling import train_random_forest
from core.prepared_dataset import mean_model_feature_names, mean_model_raster_paths
from core.raster_registry import RasterRegistry
from core.sample_table import build_sample_table
from utils.constants import RESPONSE_VARIABLES
from utils.output_io import (
    export_feature_importance_csv,
    export_model_run_result,
    export_sample_table_summary,
)
from utils.paths import ensure_output_dir


def main() -> int:
    project_root = Path(__file__).resolve().parent
    output_dir = ensure_output_dir() / "mean_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {}
    for response_name in RESPONSE_VARIABLES:
        registry = RasterRegistry()
        raster_map = mean_model_raster_paths(project_root, response_name)
        for variable_name, path in raster_map.items():
            registry.register(path, variable_name_override=variable_name)

        registry.validate_alignment()
        arrays = registry.load_arrays()
        nodata_map = registry.nodata_map()
        sample_table = build_sample_table(arrays, nodata_map)
        sample_summary_path = export_sample_table_summary(
            sample_table,
            [record.variable_name for record in registry.records],
            label="{0}_sample_table_summary".format(response_name),
        )

        model, result = train_random_forest(
            sample_table=sample_table,
            response_name=response_name,
            feature_names=mean_model_feature_names(),
            test_size=0.2,
            random_state=42,
        )
        model_summary_path = export_model_run_result(
            result=result,
            response_name=response_name,
            random_state=42,
            test_size=0.2,
        )
        importance_path = export_feature_importance_csv(result, response_name)

        run_manifest[response_name] = {
            "sample_table_rows": int(len(sample_table)),
            "sample_table_summary": str(sample_summary_path),
            "model_summary": str(model_summary_path),
            "feature_importance_csv": str(importance_path),
            "metrics": {
                "r2": result.r2,
                "rmse": result.rmse,
                "mae": result.mae,
            },
            "feature_names": result.feature_names,
        }

    manifest_path = output_dir / "mean_model_runs.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print("Mean-model runs written to:", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
