from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import shapefile
from PySide6.QtCore import QThread
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QDialog,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QListWidgetItem,
)

from core.modeling import ModelRunResult
from core.prepared_dataset import all_mean_model_raster_paths
from core.raster_registry import RasterRegistry
from core.sample_table import build_sample_table
from ui.background_tasks import TrainModelWorker
from ui.load_rasters_dialog import LoadRastersDialog
from ui.manual_dialog import ManualDialog
from ui.raster_viewer import RasterViewer
from ui.scenario_window import ScenarioWindow
from ui.task_progress_dialog import TaskProgressDialog
from ui.translations import translate
from utils.constants import (
    DEFAULT_PREDICTOR_FEATURES,
    DISPLAY_UNITS,
    PREVIEW_STYLE,
    RESPONSE_VARIABLES,
)
from utils.output_io import export_sample_table_summary
from utils.paths import clear_output_dir, resource_path


class MainWindow(QMainWindow):
    MODEL_TEST_SIZE = 0.2
    MODEL_RANDOM_STATE = 42

    def __init__(self) -> None:
        super().__init__()
        self.resize(1320, 820)
        self._language = "zh"
        icon_path = resource_path("assets", "Nanling_Biogain.ico")
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.registry = RasterRegistry()
        self._sample_tables: dict[str, pd.DataFrame] = {}
        self._trained_models = {}
        self._model_results: dict[str, ModelRunResult] = {}
        self._nanling_geometry = self._load_nanling_geometry()
        self._nanling_bounds = self._compute_geometry_bounds(self._nanling_geometry)
        self._train_thread = None
        self._train_worker = None
        self._train_dialog = None
        self._scenario_window = None
        self._pending_train_response: Optional[str] = None

        self._viewer = RasterViewer()
        self._raster_list = QListWidget()
        self._raster_list.currentRowChanged.connect(lambda _row: self._preview_selected_raster())
        self._status_log = QTextEdit()
        self._status_log.setReadOnly(True)

        self._build_ui()
        self._apply_styles()
        self._apply_language()
        self._append_log(self._tr("app_initialized"))

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)

        self._open_button = QPushButton()
        self._open_button.clicked.connect(self._open_rasters)

        self._validate_button = QPushButton()
        self._validate_button.clicked.connect(self._validate_grid)

        self._clear_button = QPushButton()
        self._clear_button.clicked.connect(self._clear_session)

        self._load_mean_button = QPushButton()
        self._load_mean_button.clicked.connect(self._load_mean_dataset)

        self._sample_table_button = QPushButton()
        self._sample_table_button.clicked.connect(self._build_sample_table_for_response)

        self._train_button = QPushButton()
        self._train_button.clicked.connect(self._train_rf_model)

        self._scenario_button = QPushButton()
        self._scenario_button.clicked.connect(self._open_scenario_window)

        self._settings_button = QToolButton()
        self._settings_button.setPopupMode(QToolButton.InstantPopup)
        self._settings_menu = QMenu(self)
        self._manual_action = QAction("", self)
        self._manual_action.triggered.connect(self._open_manual)
        self._settings_menu.addAction(self._manual_action)
        self._language_menu = self._settings_menu.addMenu("")
        self._english_action = QAction("English", self)
        self._english_action.triggered.connect(lambda: self._set_language("en"))
        self._chinese_action = QAction("\u4e2d\u6587", self)
        self._chinese_action.triggered.connect(lambda: self._set_language("zh"))
        self._language_menu.addAction(self._english_action)
        self._language_menu.addAction(self._chinese_action)
        self._settings_button.setMenu(self._settings_menu)

        button_row = QHBoxLayout()
        button_row.addWidget(self._open_button)
        button_row.addWidget(self._validate_button)
        button_row.addWidget(self._clear_button)
        button_row.addWidget(self._load_mean_button)
        button_row.addWidget(self._sample_table_button)
        button_row.addWidget(self._train_button)
        button_row.addWidget(self._scenario_button)
        button_row.addWidget(self._settings_button)
        button_row.addStretch(1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self._loaded_rasters_label = QLabel()
        left_layout.addWidget(self._loaded_rasters_label)
        left_layout.addWidget(self._raster_list)
        self._run_log_label = QLabel()
        left_layout.addWidget(self._run_log_label)
        left_layout.addWidget(self._status_log)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._viewer)

        splitter = QSplitter()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 1000])
        splitter.setStretchFactor(1, 1)

        root_layout = QVBoxLayout(container)
        root_layout.addLayout(button_row)
        root_layout.addWidget(splitter)

    def _open_rasters(self) -> None:
        dialog = LoadRastersDialog(self._language, self._variable_label, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            selected_paths = dialog.selected_paths()
            self._validate_required_paths(selected_paths)
            self._load_selected_rasters(selected_paths)
        except Exception as exc:  # pragma: no cover - UI path
            self._show_error(str(exc))
            return

    def _load_mean_dataset(self) -> None:
        try:
            raster_map = all_mean_model_raster_paths(Path(__file__).resolve().parent.parent)
        except Exception as exc:  # pragma: no cover - UI path
            self._show_error(str(exc))
            return

        self._clear_session()
        for variable_name, path in raster_map.items():
            try:
                self.registry.register_or_replace(path, variable_name_override=variable_name)
            except Exception as exc:  # pragma: no cover - UI path
                self._show_error(self._tr("load_prepared_failed").format(path, exc))
                return
        self._refresh_loaded_rasters()
        self._append_log(self._tr("loaded_mean_dataset_all"))

    def _preview_selected_raster(self) -> None:
        index = self._raster_list.currentRow()
        if index < 0:
            return

        record = self.registry.records[index]
        try:
            array, transform, nodata = self.registry.read_preview_subset(
                record.variable_name,
                self._nanling_geometry,
                self._nanling_bounds,
            )
        except Exception as exc:  # pragma: no cover - UI path
            self._show_error(str(exc))
            return

        preview_style = PREVIEW_STYLE.get(record.variable_name, {})
        self._viewer.show_single_band(
            array,
            self._variable_label(record.variable_name),
            nodata=nodata,
            transform=transform,
            overlay_geometries=self._nanling_geometry,
            unit_label=DISPLAY_UNITS.get(record.variable_name, ""),
            cmap_name=preview_style.get("cmap"),
            fixed_range=preview_style.get("fixed_range"),
        )

    def _validate_grid(self) -> None:
        try:
            report = self._ensure_registry_aligned()
        except Exception as exc:  # pragma: no cover - UI path
            self._show_error(str(exc))
            return
        self._append_log(report)
        QMessageBox.information(self, self._tr("validate_grid"), report)

    def _build_sample_table_for_response(self) -> None:
        response_name = self._prompt_response_variable(self._tr("select_response_build"))
        if response_name is None:
            return
        try:
            alignment_report = self._ensure_registry_aligned()
            arrays = self.registry.load_arrays()
            nodata_map = self.registry.nodata_map()
            if response_name not in arrays:
                raise ValueError(self._tr("response_missing_error").format(self._variable_label(response_name)))
            missing_biodiversity = [
                name
                for name in ["tree_diversity", "structure_diversity"]
                if name not in arrays
            ]
            if missing_biodiversity:
                raise ValueError(
                    self._tr("biodiversity_missing_error").format(
                        ", ".join(self._variable_label(name) for name in missing_biodiversity)
                    )
                )
            subset_names = [response_name] + [
                name for name in DEFAULT_PREDICTOR_FEATURES if name in arrays
            ]
            subset_arrays = {name: arrays[name] for name in subset_names}
            subset_nodata = {name: nodata_map.get(name) for name in subset_names}
            sample_table = build_sample_table(subset_arrays, subset_nodata)
            summary_path = export_sample_table_summary(
                sample_table,
                subset_names,
                label="{0}_sample_table_summary".format(response_name),
            )
        except Exception as exc:  # pragma: no cover - UI path
            self._show_error(str(exc))
            return

        self._sample_tables[response_name] = sample_table
        self._append_log(alignment_report)
        self._append_log(
            self._tr("sample_table_built_log").format(self._variable_label(response_name), len(sample_table))
        )
        self._append_log(self._tr("sample_table_summary_saved_log").format(summary_path))

    def _train_rf_model(self) -> None:
        response_name = self._prompt_response_variable(self._tr("select_response_train"))
        if response_name is None:
            return
        sample_table = self._sample_tables.get(response_name)
        if sample_table is None:
            self._show_error(self._tr("build_first").format(self._variable_label(response_name)))
            return

        try:
            self._ensure_registry_aligned()
            feature_names = self._resolve_feature_names(sample_table, response_name)
        except Exception as exc:  # pragma: no cover - UI path
            self._show_error(str(exc))
            return

        self._pending_train_response = response_name
        self._train_dialog = TaskProgressDialog(
            "{0}: {1}".format(self._tr("training_in_progress"), response_name),
            self._tr("cancel"),
            self,
        )
        self._train_worker = TrainModelWorker(
            sample_table=sample_table,
            response_name=response_name,
            feature_names=feature_names,
            test_size=self.MODEL_TEST_SIZE,
            random_state=self.MODEL_RANDOM_STATE,
        )
        self._train_thread = QThread(self)
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.progress.connect(self._handle_train_progress)
        self._train_worker.finished.connect(self._handle_train_finished)
        self._train_worker.failed.connect(self._handle_train_failed)
        self._train_worker.cancelled.connect(self._handle_train_cancelled)
        self._train_worker.finished.connect(self._cleanup_train_worker)
        self._train_worker.failed.connect(self._cleanup_train_worker)
        self._train_worker.cancelled.connect(self._cleanup_train_worker)
        self._train_dialog.cancel_button.clicked.connect(self._train_worker.cancel)
        self._train_dialog.rejected.connect(self._train_worker.cancel)
        self._train_thread.start()
        self._train_dialog.exec()

    def _open_scenario_window(self) -> None:
        self._scenario_window = ScenarioWindow(
            language=self._language,
            trained_models=self._trained_models,
            model_results=self._model_results,
            registry=self.registry,
            nanling_geometry=self._nanling_geometry,
            nanling_bounds=self._nanling_bounds,
            sample_tables=self._sample_tables,
            on_log_message=self._append_log,
            parent=self,
        )
        self._scenario_window.show()
        self._scenario_window.raise_()
        self._scenario_window.activateWindow()

    def _handle_train_progress(self, value: int, message: str) -> None:
        if self._train_dialog is not None:
            self._train_dialog.update_progress(value, message)

    def _handle_train_finished(self, payload: dict) -> None:
        response_name = payload["result"].response_name
        self._trained_models[response_name] = payload["model"]
        self._model_results[response_name] = payload["result"]
        result = payload["result"]
        self._append_log(
            self._tr("rf_complete_log").format(
                self._variable_label(response_name),
                result.train_size,
                result.test_size,
                result.r2,
                result.rmse,
                result.mae,
            )
        )
        self._append_log(self._tr("feature_importance_log").format(result.feature_importance))
        self._append_log(self._tr("model_summary_saved_log").format(payload["summary_path"]))
        self._append_log(self._tr("feature_importance_saved_log").format(payload["importance_path"]))
        if self._train_dialog is not None:
            self._train_dialog.accept()

    def _handle_train_failed(self, message: str) -> None:
        if self._train_dialog is not None:
            self._train_dialog.reject()
        self._show_error(message)

    def _handle_train_cancelled(self) -> None:
        self._append_log(self._tr("training_cancelled"))
        if self._train_dialog is not None:
            self._train_dialog.reject()

    def _cleanup_train_worker(self) -> None:
        if self._train_thread is not None:
            self._train_thread.quit()
            self._train_thread.wait()
            self._train_thread = None
        self._train_worker = None
        self._train_dialog = None
        self._pending_train_response = None

    def _clear_session(self) -> None:
        if self._train_worker is not None:
            self._train_worker.cancel()
        if self._train_thread is not None:
            self._train_thread.quit()
            self._train_thread.wait()
            self._train_thread = None
        self._train_worker = None
        self._train_dialog = None
        self._pending_train_response = None
        if self._scenario_window is not None:
            self._scenario_window.cancel_running_task()
            self._scenario_window.close()
            self._scenario_window = None
        self.registry.clear()
        self._sample_tables = {}
        self._trained_models = {}
        self._model_results = {}
        clear_output_dir()
        self._raster_list.clear()
        self._status_log.clear()
        self._viewer.clear()
        self._append_log(self._tr("session_cleared"))

    def _resolve_feature_names(self, sample_table: pd.DataFrame, response_name: str) -> list[str]:
        available_features = [
            name
            for name in DEFAULT_PREDICTOR_FEATURES
            if name in sample_table.columns and name != response_name
        ]
        if response_name not in sample_table.columns:
            raise ValueError(self._tr("response_missing_error").format(self._variable_label(response_name)))
        missing_biodiversity = [
            name for name in ["tree_diversity", "structure_diversity"] if name not in sample_table.columns
        ]
        if missing_biodiversity:
            raise ValueError(
                self._tr("biodiversity_missing_error").format(
                    ", ".join(self._variable_label(name) for name in missing_biodiversity)
                )
            )
        if not available_features:
            raise ValueError(self._tr("predictor_missing_error").format(self._variable_label(response_name)))
        return available_features

    def _prompt_response_variable(self, prompt: str) -> Optional[str]:
        value, accepted = QInputDialog.getItem(
            self,
            self._tr("select_response_title"),
            prompt,
            RESPONSE_VARIABLES,
            0,
            False,
        )
        if not accepted or not value:
            return None
        return value

    def _append_log(self, message: str) -> None:
        self._status_log.append(message)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, self._tr("load_failed_title"), message)

    def _add_raster_list_item(self, variable_name: str, filename: str) -> None:
        item = QListWidgetItem(self._format_raster_item_text(variable_name, filename))
        item.setData(256, variable_name)
        item.setData(257, filename)
        self._raster_list.addItem(item)

    def _refresh_loaded_rasters(self) -> None:
        current_variable = None
        current_item = self._raster_list.currentItem()
        if current_item is not None:
            current_variable = current_item.data(256)
        self._raster_list.clear()
        for record in self.registry.records:
            self._add_raster_list_item(record.variable_name, record.path.name)
        if self._raster_list.count():
            if current_variable is None:
                self._raster_list.setCurrentRow(0)
            else:
                for index in range(self._raster_list.count()):
                    item = self._raster_list.item(index)
                    if item.data(256) == current_variable:
                        self._raster_list.setCurrentRow(index)
                        break

    def _refresh_raster_list_labels(self) -> None:
        for index in range(self._raster_list.count()):
            item = self._raster_list.item(index)
            variable_name = item.data(256)
            filename = item.data(257)
            item.setText(self._format_raster_item_text(variable_name, filename))

    def _format_raster_item_text(self, variable_name: str, filename: str) -> str:
        unit = DISPLAY_UNITS.get(variable_name, "")
        unit_suffix = "" if not unit else " [{0}]".format(unit)
        return "{0}{1}\n{2}".format(self._variable_label(variable_name), unit_suffix, filename)

    def _open_manual(self) -> None:
        dialog = ManualDialog(self._language, self)
        dialog.exec()

    def _ensure_registry_aligned(self) -> str:
        try:
            self.registry.validate_alignment()
            self._refresh_loaded_rasters()
            return self._alignment_report_text()
        except ValueError:
            pass

        progress_dialog = TaskProgressDialog(
            self._tr("alignment_in_progress"),
            self._tr("cancel"),
            self,
        )
        progress_dialog.cancel_button.setEnabled(False)
        progress_dialog.show()
        progress_dialog.update_progress(0, self._tr("alignment_in_progress"))
        self.registry.ensure_aligned(
            Path(__file__).resolve().parent.parent / "outputs" / "_aligned_session",
            progress_callback=self._alignment_progress_callback(progress_dialog),
        )
        progress_dialog.accept()
        self._refresh_loaded_rasters()
        return self._alignment_report_text()

    def _alignment_report_text(self) -> str:
        if not self.registry.records:
            return self._tr("no_rasters_loaded")
        reference = self.registry.records[0]
        return self._tr("alignment_ok").format(
            len(self.registry.records),
            reference.crs,
            reference.shape,
            reference.resolution,
        )

    def _alignment_progress_callback(self, progress_dialog: TaskProgressDialog):
        def callback(value: int, message: str) -> None:
            if message.startswith("Checked "):
                variable_name = message[len("Checked "):]
                localized_message = self._tr("alignment_checked").format(self._variable_label(variable_name))
            elif message.startswith("Aligned "):
                variable_name = message[len("Aligned "):]
                localized_message = self._tr("alignment_resampled").format(self._variable_label(variable_name))
            else:
                localized_message = message
            progress_dialog.update_progress(value, localized_message)

        return callback

    def _validate_required_paths(self, selected_paths: dict[str, Path]) -> None:
        has_response = any(name in selected_paths for name in ["GPP", "LAI", "VOD"])
        if not has_response:
            raise ValueError(self._tr("required_response_group_error"))
        missing_biodiversity = [
            name
            for name in ["tree_diversity", "structure_diversity"]
            if name not in selected_paths
        ]
        if missing_biodiversity:
            raise ValueError(self._tr("required_biodiversity_pair_error").format(", ".join(missing_biodiversity)))

    def _load_selected_rasters(self, selected_paths: dict[str, Path]) -> None:
        self._clear_session()
        for variable_name, path in selected_paths.items():
            record = self.registry.register_or_replace(path, variable_name_override=variable_name)
            self._append_log(
                self._tr("loaded_raster_log").format(
                    self._variable_label(record.variable_name),
                    record.shape,
                    record.nodata,
                )
            )
        alignment_report = self._ensure_registry_aligned()
        self._append_log(alignment_report)

    def _load_nanling_geometry(self) -> list[dict]:
        shp_path = Path(__file__).resolve().parent.parent / "data" / "Nanling.shp"
        if not shp_path.exists():
            return []
        reader = shapefile.Reader(str(shp_path))
        return [shape_record.shape.__geo_interface__ for shape_record in reader.iterShapeRecords()]

    def _compute_geometry_bounds(self, geometry: list[dict]) -> tuple[float, float, float, float]:
        xs = []
        ys = []
        for item in geometry:
            self._collect_coords(item.get("coordinates", []), xs, ys)
        if not xs or not ys:
            return (0.0, 0.0, 0.0, 0.0)
        return (min(xs), min(ys), max(xs), max(ys))

    def _collect_coords(self, coords: object, xs: list[float], ys: list[float]) -> None:
        if isinstance(coords, (tuple, list)) and coords:
            first = coords[0]
            if isinstance(first, (float, int)):
                xs.append(float(coords[0]))
                ys.append(float(coords[1]))
                return
            for item in coords:
                self._collect_coords(item, xs, ys)

    def _set_language(self, language: str) -> None:
        self._language = language
        self._apply_language()
        if self._scenario_window is not None:
            self._scenario_window.set_language(language)

    def _apply_language(self) -> None:
        self.setWindowTitle(self._tr("window_title"))
        self._open_button.setText(self._tr("load_rasters"))
        self._validate_button.setText(self._tr("validate_grid"))
        self._clear_button.setText(self._tr("clear_session"))
        self._load_mean_button.setText(self._tr("load_mean_dataset_button"))
        self._sample_table_button.setText(self._tr("build_sample_table"))
        self._train_button.setText(self._tr("train_rf"))
        self._scenario_button.setText(self._tr("biodiversity_gain_assessment"))
        self._settings_button.setText(self._tr("settings_button"))
        self._loaded_rasters_label.setText(self._tr("loaded_rasters"))
        self._run_log_label.setText(self._tr("run_log"))
        self._manual_action.setText(self._tr("manual_action"))
        self._language_menu.setTitle(self._tr("language_menu"))
        self._refresh_raster_list_labels()

    def _tr(self, key: str) -> str:
        return translate(self._language, key)

    def _variable_label(self, variable_name: str) -> str:
        return self._tr("var_{0}".format(variable_name))

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 13px;
            }
            QPushButton, QToolButton, QComboBox, QTextEdit, QListWidget {
                font-size: 13px;
            }
            QLabel {
                font-size: 14px;
            }
            QListWidget {
                font-size: 14px;
                outline: none;
            }
            QListWidget::item {
                padding: 10px 8px;
                margin: 3px 2px;
                border-radius: 6px;
            }
            QListWidget::item:selected {
                background: #dceeff;
                border: 1px solid #4f7cac;
                color: #102a43;
            }
            """
        )
        self._loaded_rasters_label.setStyleSheet("font-size: 16px; font-weight: 600;")
        self._run_log_label.setStyleSheet("font-size: 15px; font-weight: 600;")
