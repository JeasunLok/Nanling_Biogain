from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from rasterio.features import geometry_mask

from core.prepared_dataset import prepared_data_root
from ui.background_tasks import ScenarioWorker
from ui.raster_viewer import RasterViewer
from ui.translations import translate
from utils.constants import DISPLAY_UNITS, PREVIEW_STYLE, RESPONSE_VARIABLES
from utils.paths import ensure_output_dir


class ScenarioWindow(QDialog):
    NANLING_2025_CARBON_SINK_TONNES = 210400.0
    CONSERVATIVE_NEP_GPP_RATIO = 0.12

    def __init__(
        self,
        language: str,
        trained_models: dict,
        model_results: dict,
        registry,
        nanling_geometry,
        nanling_bounds,
        sample_tables: dict,
        on_log_message,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.resize(1200, 820)
        self.setModal(False)
        self._language = language
        self._trained_models = trained_models
        self._model_results = model_results
        self._registry = registry
        self._nanling_geometry = nanling_geometry
        self._nanling_bounds = nanling_bounds
        self._sample_tables = sample_tables
        self._on_log_message = on_log_message
        self._worker_thread = None
        self._worker = None
        self._scenario_results = {}
        self._predictor_inputs = {}
        self._predictor_row_labels = {}

        self._response_selector = QComboBox()
        self._response_selector.addItems(RESPONSE_VARIABLES)
        self._feature_selector = QComboBox()
        self._preview_selector = QComboBox()
        self._run_button = QPushButton()
        self._cancel_button = QPushButton()
        self._export_report_button = QPushButton()
        self._predict_button = QPushButton()
        self._cancel_button.setEnabled(False)
        self._progress_label = QLabel()
        self._summary_box = QTextEdit()
        self._summary_box.setReadOnly(True)
        self._viewer = RasterViewer()
        self._prediction_result_label = QLabel()
        self._prediction_result_label.setWordWrap(True)
        self._prediction_result_label.setMaximumWidth(340)

        self._build_ui()
        self._apply_language()
        self._seed_predictor_inputs()
        self._response_selector.currentIndexChanged.connect(self._handle_response_changed)
        self._feature_selector.currentIndexChanged.connect(self._handle_feature_changed)
        self._preview_selector.currentIndexChanged.connect(self._refresh_preview)
        self._run_button.clicked.connect(self._run_assessment)
        self._cancel_button.clicked.connect(self._cancel_task)
        self._predict_button.clicked.connect(self._predict_gain_from_inputs)
        self._export_report_button.clicked.connect(self._export_report)
        self._handle_response_changed()

    def set_language(self, language: str) -> None:
        self._language = language
        self._apply_language()
        self._refresh_preview()

    def _build_ui(self) -> None:
        self._report_box = QGroupBox()
        report_layout = QVBoxLayout(self._report_box)
        report_layout.addWidget(self._summary_box)

        form_panel = QWidget()
        form_layout = QFormLayout(form_panel)
        self._response_label = QLabel()
        self._feature_label = QLabel()
        self._preview_layer_label = QLabel()
        form_layout.addRow(self._response_label, self._response_selector)
        form_layout.addRow(self._feature_label, self._feature_selector)
        form_layout.addRow(self._preview_layer_label, self._preview_selector)

        button_row = QHBoxLayout()
        button_row.addWidget(self._run_button)
        button_row.addWidget(self._cancel_button)
        button_row.addWidget(self._export_report_button)
        button_row.addStretch(1)

        predictor_group = QGroupBox()
        predictor_layout = QFormLayout(predictor_group)
        for feature_name in [
            "tree_diversity",
            "structure_diversity",
            "MAT",
            "MAP",
            "VPD",
            "SM",
            "SSRD",
            "DEM",
            "slope",
            "aspect",
        ]:
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setRange(-1000000.0, 1000000.0)
            spin_box.setSingleStep(0.1)
            self._predictor_inputs[feature_name] = spin_box
            row_label = QLabel()
            self._predictor_row_labels[feature_name] = row_label
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(spin_box)
            unit_label = QLabel(self._unit_for_predictor(feature_name))
            unit_label.setMinimumWidth(72)
            row_layout.addWidget(unit_label)
            predictor_layout.addRow(row_label, row_widget)

        predictor_button_row = QHBoxLayout()
        predictor_button_row.addWidget(self._predict_button)
        predictor_button_row.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(form_panel)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self._progress_label)
        left_layout.addWidget(predictor_group)
        left_layout.addLayout(predictor_button_row)
        left_layout.addWidget(self._prediction_result_label)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._viewer)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(left_panel, 0)
        bottom_layout.addWidget(right_panel, 1)

        root = QVBoxLayout(self)
        root.addWidget(self._report_box, 0)
        root.addLayout(bottom_layout, 1)

    def _apply_language(self) -> None:
        current_feature = self._feature_selector.currentData()
        self._feature_selector.blockSignals(True)
        self._feature_selector.clear()
        self._feature_selector.addItem(self._feature_label_text("tree_diversity"), "tree_diversity")
        self._feature_selector.addItem(
            self._feature_label_text("structure_diversity"),
            "structure_diversity",
        )
        if current_feature is not None:
            feature_index = self._feature_selector.findData(current_feature)
            if feature_index >= 0:
                self._feature_selector.setCurrentIndex(feature_index)
        self._feature_selector.blockSignals(False)
        self.setWindowTitle(self._tr("scenario_window_title"))
        self._report_box.setTitle(self._tr("report_ready"))
        self._response_label.setText(self._tr("response_variable"))
        self._feature_label.setText(self._tr("scenario_feature"))
        self._preview_layer_label.setText(self._tr("preview_layer"))
        self._run_button.setText(self._tr("run_assessment"))
        self._cancel_button.setText(self._tr("cancel"))
        self._export_report_button.setText(self._tr("export_report"))
        self._predict_button.setText(self._tr("predict_gain"))
        for feature_name, label in self._predictor_row_labels.items():
            label.setText(self._tr("var_{0}".format(feature_name)))
        self._populate_preview_selector()

    def _tr(self, key: str) -> str:
        return translate(self._language, key)

    def _current_response(self) -> str:
        return self._response_selector.currentText()

    def _current_feature(self) -> str:
        feature_name = self._feature_selector.currentData()
        return feature_name if feature_name is not None else "tree_diversity"

    def _seed_predictor_inputs(self) -> None:
        response_name = self._current_response()
        sample_table = self._sample_tables.get(response_name)
        defaults = {
            "tree_diversity": 13.0,
            "structure_diversity": 0.4,
            "MAT": 20.0,
            "MAP": 1600.0,
            "VPD": 5.0,
            "SM": 0.5,
            "SSRD": 160.0,
            "DEM": 600.0,
            "slope": 2.0,
            "aspect": 200.0,
        }
        for feature_name, widget in self._predictor_inputs.items():
            if feature_name in defaults:
                widget.setValue(float(defaults[feature_name]))
            elif sample_table is not None and feature_name in sample_table.columns:
                widget.setValue(float(sample_table[feature_name].mean()))

    def _handle_response_changed(self) -> None:
        response_name = self._current_response()
        sample_table = self._sample_tables.get(response_name)
        if sample_table is not None:
            for feature_name, widget in self._predictor_inputs.items():
                if feature_name in {"tree_diversity", "structure_diversity", "MAT", "MAP", "SM", "SSRD", "DEM", "slope", "aspect"}:
                    continue
                if feature_name in sample_table.columns:
                    widget.setValue(float(sample_table[feature_name].mean()))
        trained = response_name in self._trained_models
        self._run_button.setEnabled(trained)
        self._predict_button.setEnabled(trained)
        self._export_report_button.setEnabled(self._scenario_result_key() in self._scenario_results)
        if trained:
            result = self._model_results[response_name]
            self._summary_box.setPlainText(
                "{0}\nR2: {1:.4f}\nRMSE: {2:.4f}\nMAE: {3:.4f}".format(
                    self._tr("response_label_line").format(
                        self._tr("var_{0}".format(response_name))
                    ),
                    result.r2,
                    result.rmse,
                    result.mae,
                )
            )
        else:
            self._summary_box.setPlainText(self._tr("assessment_requires_training"))
        self._prediction_result_label.clear()
        self._populate_preview_selector()
        self._refresh_preview()

    def _handle_feature_changed(self) -> None:
        self._export_report_button.setEnabled(self._scenario_result_key() in self._scenario_results)
        self._populate_preview_selector()
        self._refresh_preview()

    def _run_assessment(self) -> None:
        response_name = self._current_response()
        if response_name not in self._trained_models:
            self._show_error(self._tr("assessment_requires_training"))
            return
        if self._worker_thread is not None:
            return

        arrays = self._registry.load_arrays()
        nodata_map = self._registry.nodata_map()
        self._progress_label.setText(self._tr("scenario_in_progress"))
        self._run_button.setEnabled(False)
        self._cancel_button.setEnabled(True)

        self._worker = ScenarioWorker(
            model=self._trained_models[response_name],
            arrays=arrays,
            feature_names=self._model_results[response_name].feature_names,
            nodata_map=nodata_map,
            adjusted_feature=self._current_feature(),
            delta=1.0,
            response_name=response_name,
            reference_profile=self._registry.export_reference_profile(),
            output_dir=ensure_output_dir(),
        )
        self._worker_thread = QThread(self)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._handle_progress)
        self._worker.finished.connect(self._handle_finished)
        self._worker.failed.connect(self._handle_failed)
        self._worker.cancelled.connect(self._handle_cancelled)
        self._worker.finished.connect(self._cleanup_worker)
        self._worker.failed.connect(self._cleanup_worker)
        self._worker.cancelled.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _cancel_task(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self._progress_label.setText(self._tr("scenario_cancelled"))

    def _handle_progress(self, value: int, message: str) -> None:
        self._progress_label.setText("{0}% | {1}".format(value, message))

    def _handle_finished(self, payload: dict) -> None:
        self._scenario_results[self._scenario_result_key(payload["response_name"], payload["scenario_feature"])] = payload
        self._summary_box.setPlainText(self._build_report_text(payload))
        percent_index = self._preview_selector.findData("percent_gain")
        if percent_index >= 0:
            self._preview_selector.setCurrentIndex(percent_index)
        self._prediction_result_label.clear()
        self._progress_label.setText(self._tr("scenario_complete_title"))
        self._export_report_button.setEnabled(True)
        self._populate_preview_selector()
        self._on_log_message(
            "{0}: {1} | {2}={3:.4f} | {4}={5:.4f}%".format(
                self._tr("scenario_exported"),
                payload["absolute_output_path"],
                self._tr("mean_gain"),
                payload["mean_gain"],
                self._tr("mean_percent_gain"),
                payload["mean_percent_gain"],
            )
        )

    def _handle_failed(self, message: str) -> None:
        self._show_error(message)
        self._progress_label.setText(message)

    def _handle_cancelled(self) -> None:
        self._progress_label.setText(self._tr("scenario_cancelled"))
        self._on_log_message(self._tr("scenario_cancelled"))

    def _cleanup_worker(self) -> None:
        self._cancel_button.setEnabled(False)
        self._run_button.setEnabled(self._current_response() in self._trained_models)
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread = None
        self._worker = None

    def _refresh_preview(self) -> None:
        preview_mode = self._preview_selector.currentData()
        response_name = self._current_response()
        if preview_mode == "response_mean":
            try:
                array, transform, nodata = self._read_prepared_response_preview(response_name)
            except Exception as exc:  # pragma: no cover - UI path
                self._summary_box.setPlainText(str(exc))
                return
            preview_style = PREVIEW_STYLE.get(response_name, {})
            self._viewer.show_single_band(
                array,
                self._tr("var_{0}".format(response_name)),
                nodata=nodata,
                transform=transform,
                overlay_geometries=self._nanling_geometry,
                unit_label=DISPLAY_UNITS.get(response_name, ""),
                cmap_name=preview_style.get("cmap"),
                fixed_range=preview_style.get("fixed_range"),
            )
            return
        scenario_result = self._scenario_results.get(self._scenario_result_key())
        if scenario_result is None:
            return
        if preview_mode == "absolute_gain":
            self._viewer.show_single_band(
                scenario_result["delta_raster"],
                self._gain_preview_title(response_name, scenario_result["scenario_feature"], "absolute"),
                nodata=np.nan,
                transform=self._registry.records[0].transform,
                overlay_geometries=self._nanling_geometry,
                unit_label=DISPLAY_UNITS.get(response_name, ""),
                cmap_name="magma",
                signed_mode="absolute_gain",
                percentile_range=(25.0, 75.0),
            )
        elif preview_mode == "percent_gain":
            self._viewer.show_single_band(
                scenario_result["percent_raster"],
                self._gain_preview_title(response_name, scenario_result["scenario_feature"], "percent"),
                nodata=np.nan,
                transform=self._registry.records[0].transform,
                overlay_geometries=self._nanling_geometry,
                unit_label="%",
                cmap_name="cividis",
                signed_mode="percent_gain",
                percentile_range=(25.0, 75.0),
            )

    def _read_prepared_response_preview(self, response_name: str):
        response_path = (
            prepared_data_root(Path(__file__).resolve().parent.parent)
            / "response_means"
            / "{0}_mean.tif".format(response_name)
        )
        with rasterio.open(response_path) as dataset:
            bounds = dataset.bounds
            clipped_bounds = (
                max(self._nanling_bounds[0], bounds.left),
                max(self._nanling_bounds[1], bounds.bottom),
                min(self._nanling_bounds[2], bounds.right),
                min(self._nanling_bounds[3], bounds.top),
            )
            window = rasterio.windows.from_bounds(
                *clipped_bounds,
                transform=dataset.transform,
            ).round_offsets().round_lengths()
            array = dataset.read(1, window=window).astype(np.float32)
            transform = rasterio.windows.transform(window, dataset.transform)
            nodata = dataset.nodata

        mask = geometry_mask(
            list(self._nanling_geometry),
            out_shape=array.shape,
            transform=transform,
            invert=True,
        )
        fill_value = np.nan if nodata is None else nodata
        array = np.where(mask, array, fill_value).astype(np.float32)
        return array, transform, nodata

    def _predict_gain_from_inputs(self) -> None:
        response_name = self._current_response()
        if response_name not in self._trained_models:
            self._show_error(self._tr("assessment_requires_training"))
            return
        model = self._trained_models[response_name]
        feature_names = self._model_results[response_name].feature_names
        row = {name: self._predictor_inputs[name].value() for name in feature_names}
        prediction_table = pd.DataFrame([row], columns=feature_names)
        baseline_prediction = float(model.predict(prediction_table)[0])
        scenario_table = prediction_table.copy()
        feature_name = self._current_feature()
        scenario_table.loc[:, feature_name] = scenario_table.loc[:, feature_name] + 1.0
        scenario_prediction = float(model.predict(scenario_table)[0])
        gain = scenario_prediction - baseline_prediction
        percent_gain = 0.0
        if abs(baseline_prediction) > 1e-12:
            percent_gain = (gain / baseline_prediction) * 100.0
        response_unit = self._response_unit_suffix(response_name)
        feature_label = self._feature_label_text(feature_name)
        sentence_key = (
            "predict_gain_sentence_positive" if gain >= 0 else "predict_gain_sentence_negative"
        )
        self._prediction_result_label.setText(
            "<b>{0}</b>".format(
                self._tr(sentence_key).format(
                    feature_label,
                    self._tr("var_{0}".format(response_name)),
                    abs(gain),
                    response_unit,
                    abs(percent_gain),
                )
            )
        )

    def _build_report_text(self, payload: dict) -> str:
        result = self._model_results[payload["response_name"]]
        response_name = payload["response_name"]
        response_label = self._tr("var_{0}".format(response_name))
        feature_label = self._feature_label_text(payload["scenario_feature"])
        mean_gain = float(payload["mean_gain"])
        mean_percent_gain = float(payload["mean_percent_gain"])
        gain_key = "absolute_gain_sentence_positive" if mean_gain >= 0 else "absolute_gain_sentence_negative"
        percent_key = "percent_gain_sentence_positive" if mean_percent_gain >= 0 else "percent_gain_sentence_negative"
        lines = [
            self._tr("scenario_window_title"),
            self._tr("response_label_line").format(response_label),
            self._tr("scenario_feature_line").format(feature_label),
            "R2: {0:.4f}".format(result.r2),
            "RMSE: {0:.4f}".format(result.rmse),
            "MAE: {0:.4f}".format(result.mae),
            self._tr(gain_key).format(
                feature_label,
                response_label,
                abs(mean_gain),
                self._response_unit_suffix(response_name),
            ),
            self._tr(percent_key).format(
                feature_label,
                response_label,
                abs(mean_percent_gain),
            ),
        ]
        if response_name == "GPP":
            total_tonnes = self._estimate_total_gpp_gain_tonnes(
                payload["delta_raster"],
                payload.get("valid_mask"),
            )
            total_key = "gpp_total_sentence_positive" if total_tonnes >= 0 else "gpp_total_sentence_negative"
            lines.append(self._tr(total_key).format(feature_label, abs(total_tonnes)))
            lines.extend(self._build_carbon_sink_lines(total_tonnes))
        else:
            lines.append(
                "{0}: {1:.4f}".format(
                    self._tr("mean_gain"),
                    mean_gain,
                )
            )
            lines.append(
                "{0}: {1:.4f}%".format(
                    self._tr("mean_percent_gain"),
                    mean_percent_gain,
                )
            )
        lines.extend(
            [
                "{0}: {1}".format(
                    self._tr("absolute_gain_raster"),
                    payload["absolute_output_path"],
                ),
                "{0}: {1}".format(
                    self._tr("percent_gain_raster"),
                    payload["percent_output_path"],
                ),
            ]
        )
        return "\n".join(lines)

    def _export_report(self) -> None:
        if self._scenario_result_key() not in self._scenario_results:
            self._show_error(self._tr("report_not_ready"))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            self._tr("save_report_title"),
            "biodiversity_gain_report.txt",
            "Text Files (*.txt)",
        )
        if not path:
            return
        Path(path).write_text(self._summary_box.toPlainText(), encoding="utf-8")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, self._tr("load_failed_title"), message)

    def _unit_for_predictor(self, feature_name: str) -> str:
        return DISPLAY_UNITS.get(feature_name, "")

    def cancel_running_task(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _populate_preview_selector(self) -> None:
        current_preview = self._preview_selector.currentData()
        self._preview_selector.blockSignals(True)
        self._preview_selector.clear()
        self._preview_selector.addItem(self._tr("response_mean_preview"), "response_mean")
        self._preview_selector.addItem(self._tr("absolute_gain_preview"), "absolute_gain")
        self._preview_selector.addItem(self._tr("percent_gain_preview"), "percent_gain")
        if current_preview is not None:
            preview_index = self._preview_selector.findData(current_preview)
            if preview_index >= 0:
                self._preview_selector.setCurrentIndex(preview_index)
        scenario_ready = self._scenario_result_key() in self._scenario_results
        model = self._preview_selector.model()
        if hasattr(model, "item"):
            absolute_item = model.item(1)
            percent_item = model.item(2)
            if absolute_item is not None:
                absolute_item.setEnabled(scenario_ready)
            if percent_item is not None:
                percent_item.setEnabled(scenario_ready)
        if not scenario_ready:
            self._preview_selector.setCurrentIndex(0)
        self._preview_selector.blockSignals(False)

    def _response_unit_suffix(self, response_name: str) -> str:
        unit = DISPLAY_UNITS.get(response_name, "")
        return "" if not unit else " {0}".format(unit)

    def _feature_label_text(self, feature_name: str) -> str:
        return self._tr("feature_{0}".format(feature_name))

    def _estimate_total_gpp_gain_tonnes(
        self,
        delta_raster: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> float:
        transform = self._registry.records[0].transform
        biodiversity_mask = self._biodiversity_valid_mask(delta_raster.shape)
        combined_mask = np.isfinite(delta_raster) & biodiversity_mask
        if valid_mask is not None:
            combined_mask &= valid_mask
        rows, cols = delta_raster.shape
        lon_res = abs(transform.a)
        earth_radius = 6371000.0
        total_tonnes = 0.0
        for row in range(rows):
            row_values = delta_raster[row]
            row_mask = combined_mask[row]
            if not np.any(row_mask):
                continue
            lat_top = transform.f + transform.e * row
            lat_bottom = lat_top + transform.e
            lat1 = np.radians(min(lat_top, lat_bottom))
            lat2 = np.radians(max(lat_top, lat_bottom))
            lon_width = np.radians(lon_res)
            cell_area = (earth_radius ** 2) * lon_width * abs(np.sin(lat2) - np.sin(lat1))
            total_tonnes += float(np.nansum(row_values[row_mask]) * cell_area / 1_000_000.0)
        return total_tonnes

    def _build_carbon_sink_lines(self, total_gpp_tonnes: float) -> list[str]:
        ratio = self.CONSERVATIVE_NEP_GPP_RATIO
        sink_tonnes = total_gpp_tonnes * ratio
        percent_increase = 0.0
        if self.NANLING_2025_CARBON_SINK_TONNES > 0:
            percent_increase = sink_tonnes / self.NANLING_2025_CARBON_SINK_TONNES * 100.0
        return [self._tr("carbon_sink_line").format(abs(sink_tonnes), abs(percent_increase))]

    def _biodiversity_valid_mask(self, shape: tuple[int, int]) -> np.ndarray:
        arrays = self._registry.load_arrays()
        nodata_map = self._registry.nodata_map()
        mask = np.ones(shape, dtype=bool)
        for name in ["tree_diversity", "structure_diversity"]:
            array = arrays.get(name)
            if array is None:
                return np.zeros(shape, dtype=bool)
            mask &= np.isfinite(array)
            nodata = nodata_map.get(name)
            if nodata is not None:
                mask &= ~np.isclose(array, nodata)
        return mask

    def _scenario_result_key(
        self,
        response_name: Optional[str] = None,
        feature_name: Optional[str] = None,
    ) -> tuple[str, str]:
        return (
            response_name or self._current_response(),
            feature_name or self._current_feature(),
        )

    def _gain_preview_title(self, response_name: str, feature_name: str, mode: str) -> str:
        feature_label = self._feature_label_text(feature_name)
        response_label = self._tr("var_{0}".format(response_name))
        if mode == "absolute":
            return "{0} -> {1} | {2}".format(
                feature_label,
                response_label,
                self._tr("absolute_gain_preview"),
            )
        return "{0} -> {1} | {2}".format(
            feature_label,
            response_label,
            self._tr("percent_gain_preview"),
        )
