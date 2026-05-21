import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar
)

from .model_editor_panel import ModelEditorPanel
from .parameter_table import ParameterTablePanel
from .project_menu import setup_project_menu
from .data_import_panel import DataImportPanel
from .plot_panel import PlotPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("mODEler")
        self.resize(1000, 700)
        self.current_project_file = None

        self._setup_ui()

        self.menu_bar = self.menuBar()
        setup_project_menu(self, self.menu_bar)

    def _setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)

        # 1. Model Tab (Editor)
        self.model_tab = ModelEditorPanel()
        self.tabs.addTab(self.model_tab, "1. Model Editor")

        # 2. Fit Setup Tab (Parameters)
        self.fit_tab = ParameterTablePanel()
        self.tabs.addTab(self.fit_tab, "2. Fit Settings")

        # 3. Experimental Data Import Tab
        self.data_tab = DataImportPanel()
        self.tabs.addTab(self.data_tab, "3. Experimental Data Import")

        # 4. Plotting Experimental Data
        self.plot_tab = PlotPanel()
        self.tabs.addTab(self.plot_tab, "4. Experimental Data Plot")
        self.data_tab.datasets_updated.connect(self.plot_tab.update_datasets)

        # Wire them together! When model is validated, update the parameter table
        self.model_tab.model_validated.connect(self.fit_tab.update_from_model)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    def load_project(self, filepath: str):
        self.current_project_file = filepath
        self.status_bar.showMessage(f"Loaded project: {filepath}")