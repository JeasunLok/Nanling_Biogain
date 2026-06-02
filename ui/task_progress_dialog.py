from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QPushButton, QProgressBar, QVBoxLayout


class TaskProgressDialog(QDialog):
    def __init__(self, title: str, cancel_label: str, parent=None) -> None:
        super().__init__(parent)
        self.setModal(True)
        self._message_label = QLabel()
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._cancel_button = QPushButton(cancel_label)

        layout = QVBoxLayout(self)
        layout.addWidget(self._message_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._cancel_button)

        self.setWindowTitle(title)
        self.update_progress(0, "")

    @property
    def cancel_button(self) -> QPushButton:
        return self._cancel_button

    def update_progress(self, value: int, message: str) -> None:
        self._progress_bar.setValue(value)
        self._message_label.setText(message)
