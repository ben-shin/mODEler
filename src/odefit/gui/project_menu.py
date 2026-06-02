from PySide6.QtGui import QAction

def setup_project_menu(main_window, menu_bar):
    file_menu = menu_bar.addMenu("&File")

    # --- NEW PROJECT ---
    new_action = QAction("&New Project", main_window)
    new_action.setShortcut("Ctrl+N")
    new_action.triggered.connect(lambda: main_window.status_bar.showMessage("Started a new project."))
    file_menu.addAction(new_action)

    # --- OPEN PROJECT ---
    open_action = QAction("&Open Project...", main_window)
    open_action.setShortcut("Ctrl+O")
    # This correctly points to our new JSON loader!
    open_action.triggered.connect(main_window.load_project)
    file_menu.addAction(open_action)

    file_menu.addSeparator()

    # --- SAVE PROJECT ---
    save_action = QAction("&Save", main_window)
    save_action.setShortcut("Ctrl+S")
    # --- FIXED: This now directly triggers the massive JSON save method! ---
    save_action.triggered.connect(main_window.save_project)
    file_menu.addAction(save_action)

    file_menu.addSeparator()

    # --- EXIT ---
    exit_action = QAction("E&xit", main_window)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.triggered.connect(main_window.close)
    file_menu.addAction(exit_action)