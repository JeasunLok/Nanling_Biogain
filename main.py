from __future__ import annotations

import sys
import ctypes

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from utils.paths import resource_path


def main() -> int:
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("JeasunLok.NanlingBioGain")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("NanlingBioGain")
    icon_path = resource_path("assets", "Nanling_Biogain.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
