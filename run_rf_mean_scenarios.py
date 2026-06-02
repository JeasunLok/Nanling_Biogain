from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from core.modeling import train_random_forest
from core.prepared_dataset import mean_model_feature_names, mean_model_raster_paths
from core.raster_registry import RasterRegistry
from core.sample_table import build_sample_table
from core.scenario import export_prediction_raster, export_preview_png, run_plus_one_scenario
from utils.constants import RESPONSE_VARIABLES
from utils.output_io import export_scenario_summary
from utils.paths import ensure_output_dir


SCENARIO_FEATURES = ["tree_diversity", "structure_diversity"]


def main() -> int:
    project_root = Path(__file__).resolve().parent
    output_dir = ensure_output_dir() / "rf_mean_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_manifest = {}
    summary_rows = []
    for response_name in RESPONSE_VARIABLES:
        registry = RasterRegistry()
        raster_map = mean_model_raster_paths(project_root, response_name)
        for variable_name, path in raster_map.items():
            registry.register(path, variable_name_override=variable_name)

        registry.validate_alignment()
        arrays = registry.load_arrays()
        nodata_map = registry.nodata_map()
        sample_table = build_sample_table(arrays, nodata_map)
        model, model_result = train_random_forest(
            sample_table=sample_table,
            response_name=response_name,
            feature_names=mean_model_feature_names(),
            test_size=0.2,
            random_state=42,
        )
        model_path = output_dir / "{0}_rf_baseline.joblib".format(response_name)
        joblib.dump(model, model_path)

        response_payload = {
            "model_path": str(model_path),
            "metrics": {
                "r2": model_result.r2,
                "rmse": model_result.rmse,
                "mae": model_result.mae,
            },
            "scenarios": {},
        }

        for scenario_feature in SCENARIO_FEATURES:
            absolute_raster, percent_raster, valid_mask = run_plus_one_scenario(
                model=model,
                arrays=arrays,
                feature_names=model_result.feature_names,
                nodata_map=nodata_map,
                adjusted_feature=scenario_feature,
                delta=1.0,
            )
            absolute_path = export_prediction_raster(
                output_dir / "{0}_{1}_plus_1_absolute.tif".format(response_name, scenario_feature),
                registry.export_reference_profile(),
                absolute_raster,
            )
            absolute_png_path = export_preview_png(
                output_dir / "{0}_{1}_plus_1_absolute.png".format(response_name, scenario_feature),
                absolute_raster,
                "{0} {1} +1 Absolute Gain".format(response_name, scenario_feature),
            )
            percent_path = export_prediction_raster(
                output_dir / "{0}_{1}_plus_1_percent.tif".format(response_name, scenario_feature),
                registry.export_reference_profile(),
                percent_raster,
            )
            percent_png_path = export_preview_png(
                output_dir / "{0}_{1}_plus_1_percent.png".format(response_name, scenario_feature),
                percent_raster,
                "{0} {1} +1 Percent Gain".format(response_name, scenario_feature),
            )
            mean_gain = float(absolute_raster[valid_mask].mean())
            mean_percent_gain = float(percent_raster[valid_mask].mean())
            summary_path = export_scenario_summary(
                adjusted_feature=scenario_feature,
                delta=1.0,
                response_name="{0}_{1}".format(response_name, scenario_feature),
                absolute_output_raster=absolute_path,
                percent_output_raster=percent_path,
                mean_gain=mean_gain,
                mean_percent_gain=mean_percent_gain,
                valid_pixel_count=int(valid_mask.sum()),
            )
            response_payload["scenarios"][scenario_feature] = {
                "absolute_raster": str(absolute_path),
                "absolute_png": str(absolute_png_path),
                "percent_raster": str(percent_path),
                "percent_png": str(percent_png_path),
                "mean_gain": mean_gain,
                "mean_percent_gain": mean_percent_gain,
                "summary_path": str(summary_path),
            }
            summary_rows.append(
                {
                    "response_name": response_name,
                    "scenario_feature": scenario_feature,
                    "model_r2": model_result.r2,
                    "model_rmse": model_result.rmse,
                    "model_mae": model_result.mae,
                    "mean_gain": mean_gain,
                    "mean_percent_gain": mean_percent_gain,
                    "absolute_raster": str(absolute_path),
                    "absolute_png": str(absolute_png_path),
                    "percent_raster": str(percent_path),
                    "percent_png": str(percent_png_path),
                }
            )

        run_manifest[response_name] = response_payload

    summary_frame = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "rf_mean_scenarios_summary.csv"
    summary_xlsx_path = output_dir / "rf_mean_scenarios_summary.xlsx"
    summary_frame.to_csv(summary_csv_path, index=False)
    summary_frame.to_excel(summary_xlsx_path, index=False)

    manifest_path = output_dir / "rf_mean_scenarios.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print("RF mean scenarios written to:", manifest_path)
    print("Scenario summary CSV:", summary_csv_path)
    print("Scenario summary XLSX:", summary_xlsx_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
