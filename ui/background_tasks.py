from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, Signal

from core.cancellation import OperationCancelledError
from core.modeling import train_random_forest_incremental
from core.scenario import export_prediction_raster, run_plus_one_scenario
from utils.output_io import (
    export_feature_importance_csv,
    export_model_run_result,
    export_scenario_summary,
)


class TrainModelWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        sample_table,
        response_name: str,
        feature_names,
        test_size: float,
        random_state: int,
    ) -> None:
        super().__init__()
        self._sample_table = sample_table
        self._response_name = response_name
        self._feature_names = list(feature_names)
        self._test_size = test_size
        self._random_state = random_state
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            model, result = train_random_forest_incremental(
                sample_table=self._sample_table,
                response_name=self._response_name,
                feature_names=self._feature_names,
                test_size=self._test_size,
                random_state=self._random_state,
                progress_callback=self._emit_progress,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return
            summary_path = export_model_run_result(
                result,
                response_name=self._response_name,
                random_state=self._random_state,
                test_size=self._test_size,
            )
            importance_path = export_feature_importance_csv(result, self._response_name)
            self.finished.emit(
                {
                    "model": model,
                    "result": result,
                    "summary_path": summary_path,
                    "importance_path": importance_path,
                }
            )
        except OperationCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(self, value: int, message: str) -> None:
        self.progress.emit(value, message)


class ScenarioWorker(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        model,
        arrays,
        feature_names,
        nodata_map,
        adjusted_feature: str,
        delta: float,
        response_name: str,
        reference_profile: dict,
        output_dir,
    ) -> None:
        super().__init__()
        self._model = model
        self._arrays = arrays
        self._feature_names = list(feature_names)
        self._nodata_map = nodata_map
        self._adjusted_feature = adjusted_feature
        self._delta = delta
        self._response_name = response_name
        self._reference_profile = reference_profile
        self._output_dir = output_dir
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            delta_raster, percent_raster, valid_mask = run_plus_one_scenario(
                model=self._model,
                arrays=self._arrays,
                feature_names=self._feature_names,
                nodata_map=self._nodata_map,
                adjusted_feature=self._adjusted_feature,
                delta=self._delta,
                progress_callback=self._emit_progress,
                cancel_event=self._cancel_event,
            )
            if self._cancel_event.is_set():
                self.cancelled.emit()
                return
            self.progress.emit(92, "Exporting gain rasters")
            absolute_output_path = export_prediction_raster(
                self._output_dir
                / "prediction_{0}_{1}_plus_1_absolute.tif".format(
                    self._response_name,
                    self._adjusted_feature,
                ),
                self._reference_profile,
                delta_raster,
            )
            percent_output_path = export_prediction_raster(
                self._output_dir
                / "prediction_{0}_{1}_plus_1_percent.tif".format(
                    self._response_name,
                    self._adjusted_feature,
                ),
                self._reference_profile,
                percent_raster,
            )
            mean_gain = float(delta_raster[valid_mask].mean())
            mean_percent_gain = float(percent_raster[valid_mask].mean())
            summary_path = export_scenario_summary(
                adjusted_feature=self._adjusted_feature,
                delta=self._delta,
                response_name=self._response_name,
                absolute_output_raster=absolute_output_path,
                percent_output_raster=percent_output_path,
                mean_gain=mean_gain,
                mean_percent_gain=mean_percent_gain,
                valid_pixel_count=int(valid_mask.sum()),
            )
            self.progress.emit(100, "Biodiversity gain evaluation complete")
            self.finished.emit(
                {
                    "delta_raster": delta_raster,
                    "percent_raster": percent_raster,
                    "valid_mask": valid_mask,
                    "absolute_output_path": absolute_output_path,
                    "percent_output_path": percent_output_path,
                    "summary_path": summary_path,
                    "mean_gain": mean_gain,
                    "mean_percent_gain": mean_percent_gain,
                    "scenario_feature": self._adjusted_feature,
                    "response_name": self._response_name,
                }
            )
        except OperationCancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))

    def _emit_progress(self, value: int, message: str) -> None:
        self.progress.emit(value, message)
