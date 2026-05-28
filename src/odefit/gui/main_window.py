import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar
)

from .model_editor_panel import ModelEditorPanel
from .parameter_table import ParameterTablePanel
from .project_menu import setup_project_menu
from .data_import_panel import DataImportPanel
from .plot_panel import PlotPanel
from .fit_panel import FitPanel

from odefit.fitting.optimizer import fit_model
from odefit.fitting.fit_settings import FitSettings
import traceback

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

        # 1. Model Input Tab
        self.model_tab = ModelEditorPanel()
        self.tabs.addTab(self.model_tab, "1. Model Editor")

        # 2. Parameter Setup Tab
        self.parameter_tab = ParameterTablePanel()
        self.tabs.addTab(self.parameter_tab, "2. Fit Settings")

        # 3. Experimental Data Import Tab
        self.data_tab = DataImportPanel()
        self.tabs.addTab(self.data_tab, "3. Experimental Data Import")

        # 4. Plotting Experimental Data
        self.plot_tab = PlotPanel()
        self.tabs.addTab(self.plot_tab, "4. Experimental Data Plot")
        self.data_tab.datasets_updated.connect(self.plot_tab.update_datasets)

        # 5. Fit Tab
        self.fit_tab = FitPanel()
        self.tabs.addTab(self.fit_tab, "5. Parameter Fitting")

        # Tell the Fit Tab to listen to the Data Tab (so it knows what files exist)
        self.data_tab.datasets_updated.connect(self.fit_tab.update_datasets)

        # Tell the Main Window to listen to the "Run Fit" button!
        self.fit_tab.run_fit_requested.connect(self._execute_fit)

        # Wire them together! When model is validated, update the parameter table
        # (Fixed to point to self.parameter_tab!)
        self.model_tab.model_validated.connect(self.parameter_tab.update_from_model)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    def load_project(self, filepath: str):
        self.current_project_file = filepath
        self.status_bar.showMessage(f"Loaded project: {filepath}")

    def _execute_fit(self, dataset_name, method):
        self.fit_tab.clear_log()
        self.fit_tab.log_message("Initializing ODE engine...")

        try:
            # 1. Gather all the pieces from your UI tabs
            dataset = self.data_tab.loaded_datasets[dataset_name]
            model_spec = self.model_tab.get_model_spec()
            parameter_specs = self.parameter_tab.get_parameter_specs()
            initial_condition_specs = self.parameter_tab.get_initial_condition_specs()

            # 2. Build the Species Mapping (Map CSV columns to Model Species perfectly)
            species_mapping = {col: col for col in dataset.signal_columns if col in model_spec.species}
            if not species_mapping:
                self.fit_tab.log_message("⚠️ WARNING: Could not automatically map Data Columns to Model Species.")

            # 3. Build Ben's Settings object
            settings = FitSettings(
                species_mapping=species_mapping,
                use_normalized_data=False,
                method=method,
                loss="linear",
                rtol=1e-6,
                atol=1e-9
            )

            # 4. RUN THE MATH!
            self.fit_tab.log_message("Crunching numbers... Please wait.")

            result = fit_model(
                model=model_spec,
                dataset=dataset,
                parameter_specs=parameter_specs,
                initial_condition_specs=initial_condition_specs,
                settings=settings
            )

            # 5. Output Results to the Console
            if result.success:
                self.fit_tab.log_message("\n✅ FIT SUCCESSFUL!")
            else:
                self.fit_tab.log_message(f"\n❌ FIT FAILED: {result.message}")

            self.fit_tab.log_message(f"\nFitted Parameters:\n{result.fitted_parameters}")
            self.fit_tab.log_message(f"\nStatistics:\n{result.statistics}")

        except Exception as e:
            self.fit_tab.log_message(f"\n🔥 CRITICAL ERROR: {str(e)}")
            self.fit_tab.log_message(traceback.format_exc())