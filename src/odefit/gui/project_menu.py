from PySide6.QtGui import QAction
from PySide6.QtWidgets import QFileDialog


def setup_project_menu(main_window, menu_bar):
    file_menu = menu_bar.addMenu("&File")

    new_action = QAction("&New Project", main_window)
    new_action.setShortcut("Ctrl+N")
    new_action.triggered.connect(lambda: main_window.status_bar.showMessage("Started a new project."))
    file_menu.addAction(new_action)

    open_action = QAction("&Open Project...", main_window)
    open_action.setShortcut("Ctrl+O")
    open_action.triggered.connect(lambda: _on_open_project(main_window))
    file_menu.addAction(open_action)

    file_menu.addSeparator()

    save_action = QAction("&Save", main_window)
    save_action.setShortcut("Ctrl+S")
    save_action.triggered.connect(lambda: _on_save_project(main_window))
    file_menu.addAction(save_action)

    file_menu.addSeparator()

    exit_action = QAction("E&xit", main_window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(main_window.close)
    file_menu.addAction(exit_action)


def _on_open_project(main_window):
    filepath, _ = QFileDialog.getOpenFileName(
        main_window, "Open odefit Project", "", "odefit Projects (*.fit *.json);;All Files (*)"
    )
    if filepath:
        main_window.load_project(filepath)


def _on_save_project(main_window):
    if not main_window.current_project_file:
        filepath, _ = QFileDialog.getSaveFileName(
            main_window, "Save odefit Project", "", "odefit Projects (*.fit)"
        )
        if filepath:
            main_window.current_project_file = filepath

    if main_window.current_project_file:
        main_window.status_bar.showMessage(f"Saved project to {main_window.current_project_file}")