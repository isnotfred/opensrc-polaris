import sys
from pathlib import Path


def main() -> int:
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("polaris.local.ai.studio")
    except Exception:
        pass

    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication
    from .ui.main_window import MainWindow
    from .ui.common.styles import MASTER_QSS, get_app_icon

    app = QApplication(sys.argv)
    app.setWindowIcon(get_app_icon())
    
    # Establish clean default typography across all widgets & high-DPI displays
    font = QFont("Segoe UI", 9)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    
    app.setStyleSheet(MASTER_QSS)
    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
