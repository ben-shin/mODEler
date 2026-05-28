import sys
from pathlib import Path

#src path for odefit.gui
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
# ---------------

from PySide6.QtWidgets import QApplication
from odefit.gui import MainWindow


def main():
    # 1. Create the PySide6 Application instance
    app = QApplication(sys.argv)

    # Optional: Force a modern, clean style across all operating systems
    app.setStyle("Fusion")

    # 2. Create and display your Main Window
    window = MainWindow()
    window.show()

    # 3. Start the application's event loop (keeps the window open)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()