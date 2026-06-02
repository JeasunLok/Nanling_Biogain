from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from utils.constants import DISPLAY_UNITS
from ui.translations import translate


VARIABLE_ORDER = [
    "GPP",
    "LAI",
    "VOD",
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
]


class LoadRastersDialog(QDialog):
    def __init__(self, language: str, variable_label_fn, parent=None) -> None:
        super().__init__(parent)
        self.resize(880, 620)
        self._language = language
        self._variable_label_fn = variable_label_fn
        self._path_inputs = {}

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._grid = QGridLayout(self._content)
        self._scroll.setWidget(self._content)

        self._confirm_button = QPushButton()
        self._cancel_button = QPushButton()
        self._confirm_button.clicked.connect(self.accept)
        self._cancel_button.clicked.connect(self.reject)

        self._build_rows()
        self._build_ui()
        self._apply_language()

    def selected_paths(self) -> dict[str, Path]:
        output = {}
        for variable_name, path_input in self._path_inputs.items():
            text = path_input.text().strip()
            if text:
                output[variable_name] = Path(text)
        return output

    def _build_ui(self) -> None:
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._confirm_button)
        button_row.addWidget(self._cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self._scroll)
        layout.addLayout(button_row)

    def _build_rows(self) -> None:
        self._header_variable = QLabel()
        self._header_path = QLabel()
        self._grid.addWidget(self._header_variable, 0, 0)
        self._grid.addWidget(self._header_path, 0, 1)
        self._grid.addWidget(QLabel(""), 0, 2)
        self._grid.addWidget(QLabel(""), 0, 3)
        for row_index, variable_name in enumerate(VARIABLE_ORDER, start=1):
            label_text = self._variable_label_fn(variable_name)
            unit = DISPLAY_UNITS.get(variable_name, "")
            if unit:
                label_text = "{0} [{1}]".format(label_text, unit)
            self._grid.addWidget(QLabel(label_text), row_index, 0)

            path_input = QLineEdit()
            path_input.setReadOnly(True)
            self._path_inputs[variable_name] = path_input
            self._grid.addWidget(path_input, row_index, 1)

            load_button = QPushButton()
            clear_button = QPushButton()
            load_button.setProperty("translation_key", "load_action")
            clear_button.setProperty("translation_key", "clear_action")
            load_button.clicked.connect(
                lambda _checked=False, name=variable_name: self._choose_file(name)
            )
            clear_button.clicked.connect(
                lambda _checked=False, name=variable_name: self._path_inputs[name].clear()
            )
            self._grid.addWidget(load_button, row_index, 2)
            self._grid.addWidget(clear_button, row_index, 3)

    def _choose_file(self, variable_name: str) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            self.windowTitle(),
            "",
            "GeoTIFF (*.tif *.tiff)",
        )
        if selected_path:
            self._path_inputs[variable_name].setText(selected_path)

    def _apply_language(self) -> None:
        self.setWindowTitle(translate(self._language, "load_rasters_title"))
        self._confirm_button.setText(translate(self._language, "confirm_load"))
        self._cancel_button.setText(translate(self._language, "cancel"))
        self._header_variable.setText(translate(self._language, "variable_header"))
        self._header_path.setText(translate(self._language, "path_header"))
        for button in self.findChildren(QPushButton):
            key = button.property("translation_key")
            if key:
                button.setText(translate(self._language, key))
