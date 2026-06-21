import logging
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QTabWidget, QStatusBar,
    QFileDialog, QMessageBox
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

import json
from odefit.export.json_export import to_jsonable
from odefit.export.bundle_export import export_fit_bundle

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

        # Wire them together! When model is validated, update BOTH tables
        self.model_tab.model_validated.connect(self.parameter_tab.update_from_model)
        self.model_tab.model_validated.connect(self.fit_tab.update_initial_conditions_from_model)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready.")

    def _execute_fit(self, dataset_name, method):
        self.fit_tab.clear_log()
        self.fit_tab.log_message("Initializing ODE engine...")

        try:
            # 1. Gather all the pieces from your UI tabs
            dataset = self.data_tab.loaded_datasets[dataset_name]
            model_spec = self.model_tab.get_model_spec()
            parameter_specs = self.parameter_tab.get_parameter_specs()
            # ROUTED TO FIT_TAB
            initial_condition_specs = self.fit_tab.get_initial_condition_specs()

            # 2. Build the Species Mapping (Map CSV columns to Model Species perfectly)
            species_mapping = {col: col for col in dataset.signal_columns if col in model_spec.species}
            if not species_mapping:
                self.fit_tab.log_message("WARNING: Could not automatically map Data Columns to Model Species.")

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
                self.fit_tab.log_message("\n FIT SUCCESSFUL!")
            else:
                self.fit_tab.log_message(f"\n FIT FAILED: {result.message}")

            self.fit_tab.log_message(f"\nFitted Parameters:\n{result.fitted_parameters}")
            self.fit_tab.log_message(f"\nStatistics:\n{result.statistics}")

        except Exception as e:
            self.fit_tab.log_message(f"\nCRITICAL ERROR: {str(e)}")
            self.fit_tab.log_message(traceback.format_exc())

    def save_project(self):
        """Saves the current UI state to a single JSON file."""
        default_path = "/home/sebastianjoseph/Documents/mODEler/my_experiment.json"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save mODEler Project",
            default_path,
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            # Force the .json extension
            if not file_path.endswith(".json"):
                file_path += ".json"

            print(f"\n--- SAVE DIAGNOSTIC START ---")
            print(f"DEBUG 1: User selected path -> {file_path}")

            try:
                print("DEBUG 2: Gathering parameter specs...")
                param_specs = self.parameter_tab.get_parameter_specs()
                # ROUTED TO FIT_TAB
                ic_specs = self.fit_tab.get_initial_condition_specs()

                print("DEBUG 3: Building project data dictionary...")
                project_data = {
                    "version": "0.1.0-alpha",
                    "model_text": self.model_tab.reaction_editor.toPlainText(),
                    "parameter_specs": to_jsonable(param_specs),
                    "initial_condition_specs": to_jsonable(ic_specs),
                    "data_files": list(self.data_tab.loaded_datasets.keys())
                }

                print("DEBUG 4: Attempting to write file to disk...")
                with open(file_path, 'w') as f:
                    json.dump(project_data, f, indent=4)

                print("DEBUG 5: File successfully written!")
                print("--- SAVE DIAGNOSTIC END ---\n")

                self.current_project_file = file_path
                self.status_bar.showMessage(f"Project saved to: {file_path}")

            except Exception as e:
                print(f"\nCRITICAL ERROR DURING SAVE: {str(e)}")
                import traceback
                traceback.print_exc()  # This prints the exact line number that crashed
                QMessageBox.critical(self, "Save Error", f"Could not save file:\n{str(e)}")

    def load_project(self):
        """Loads a JSON project file back into the UI."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open mODEler Project", "", "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    project_data = json.load(f)

                    # 1. Restore the model text
                    if "model_text" in project_data:
                        self.model_tab.reaction_editor.setPlainText(project_data["model_text"])
                        # Trigger validation to build the generated ODEs automatically
                        self.model_tab.validate_model()

                    # 2. Restore the parameter bounds in the table
                    if "parameter_specs" in project_data:
                        self.parameter_tab.set_parameters_from_save(project_data["parameter_specs"])

                    # 3. Restore the Initial Conditions table
                    if "initial_condition_specs" in project_data:
                        self.fit_tab.set_initial_conditions_from_save(project_data["initial_condition_specs"])

                self.current_project_file = file_path
                self.status_bar.showMessage(f"Project loaded: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Load Error", f"Could not load file:\n{str(e)}")

    def export_fit_results(self, fit_result, dataset_name):
        """Exports the massive results bundle to a chosen folder."""
        if not fit_result or not fit_result.success:
            QMessageBox.warning(self, "Export Failed", "No successful fit to export!")
            return

        # Ask the user to pick an empty folder
        dir_path = QFileDialog.getExistingDirectory(self, "Select Export Directory")

        if dir_path:
            try:
                self.fit_tab.log_message(f"Exporting massive bundle to {dir_path}...")

                # 1. Grab the current dataset and model
                dataset = self.data_tab.loaded_datasets[dataset_name]
                model_spec = self.model_tab.get_model_spec()

                # 2. Rebuild the species mapping dynamically
                mapping = {col: col for col in dataset.signal_columns if col in model_spec.species}

                # 3. Call your backend magic!
                written_files = export_fit_bundle(
                    fit_result=fit_result,
                    model=model_spec,
                    dataset=dataset,
                    output_dir=dir_path,
                    parameter_specs=self.parameter_tab.get_parameter_specs(),
                    initial_condition_specs=self.fit_tab.get_initial_condition_specs(),
                    species_mapping=mapping,  # <-- FIXED: Pass the dynamic mapping here!
                    include_plots=True
                )

                self.fit_tab.log_message(f"Exported {len(written_files)} files successfully!")

            except Exception as e:
                self.fit_tab.log_message(f"Export Failed: {str(e)}")