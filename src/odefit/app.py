import sys
from PySide6.QtWidgets import QApplication
from .gui.main_window import MainWindow


def run_gui(project_file: str | None = None) -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("odefit")

    window = MainWindow()
    if project_file:
        window.load_project(project_file)

    window.show()
    return app.exec()