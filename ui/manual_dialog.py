from __future__ import annotations

from PySide6.QtWidgets import QDialog, QPushButton, QTextEdit, QVBoxLayout

from ui.translations import translate


class ManualDialog(QDialog):
    def __init__(self, language: str, parent=None) -> None:
        super().__init__(parent)
        self._language = language
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._close_button = QPushButton()
        self._close_button.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._text)
        layout.addWidget(self._close_button)
        self._apply_language()

    def _apply_language(self) -> None:
        self.setWindowTitle(translate(self._language, "manual_title"))
        self._text.setPlainText(translate(self._language, "manual_text"))
        self._close_button.setText(translate(self._language, "close"))
