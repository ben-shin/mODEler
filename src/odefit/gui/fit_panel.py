from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem
)
from PySide6.QtCore import Signal, Qt  # Added Qt
from PySide6.QtGui import QFont

# Ensure this path matches your project structure!
from odefit.fitting.initial_condition_spec import InitialConditionSpec


class FitPanel(QWidget):
    # Run the fitting protocol
    run_fit_requested = Signal(str, str)

    def __init__(self):
        super().__init__()
        self.datasets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # --- Top Section: Controls ---
        control_group = QGroupBox("1. Fit Settings")
        control_layout = QHBoxLayout()

        # Target Dataset row
        dataset_row = QHBoxLayout()
        dataset_row.addWidget(QLabel("Target Dataset:"))
        self.combo_dataset = QComboBox()
        dataset_row.addWidget(self.combo_dataset)
        dataset_row.addStretch()
        control_layout.addLayout(dataset_row)

        # Optimizer row
        optimizer_row = QHBoxLayout()
        optimizer_row.addWidget(QLabel("Optimizer:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["trf", "lm", "dogbox"])  # Scipy's standard least-squares methods
        optimizer_row.addWidget(self.combo_method)
        optimizer_row.addStretch()
        control_layout.addLayout(optimizer_row)

        self.btn_run = QPushButton("RUN GLOBAL FIT")
        self.btn_run.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_run.setEnabled(False)  # Disabled until data is loaded
        control_layout.addWidget(self.btn_run)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # --- Middle Section: Initial Conditions ---
        self.ic_group = QGroupBox("2. State Variable Initial Conditions")
        ic_layout = QVBoxLayout()

        self.ic_table = QTableWidget(0, 5)
        self.ic_table.setHorizontalHeaderLabels([
            "State Variable", "Initial Value", "Lower Bound", "Upper Bound", "Fit?"
        ])

        ic_layout.addWidget(self.ic_table)
        self.ic_group.setLayout(ic_layout)
        layout.addWidget(self.ic_group)

        # --- Bottom Section: Output Console ---
        log_group = QGroupBox("3. Output")
        log_layout = QVBoxLayout()

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Courier", 10))  # Monospace font for aligned numbers
        self.console.setStyleSheet("background-color: #1e1e1e; color: #4caf50;")
        log_layout.addWidget(self.console)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

    def update_datasets(self, datasets_dict):
        """Catches the updated workspace from the Data Panel."""
        self.datasets = datasets_dict
        self.combo_dataset.clear()

        if self.datasets:
            self.combo_dataset.addItems(list(self.datasets.keys()))
            self.btn_run.setEnabled(True)
        else:
            self.btn_run.setEnabled(False)

    def update_initial_conditions_from_model(self, model_spec):
        """Populates the Initial Conditions table from the loaded model."""
        species_list = model_spec.species

        self.ic_table.setRowCount(len(species_list))

        for row, species_name in enumerate(species_list):
            # Column 0: Species Name (Read-only)
            name_item = QTableWidgetItem(species_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ic_table.setItem(row, 0, name_item)

            # Columns 1-3: Guess, Lower, and Upper Bounds
            self.ic_table.setItem(row, 1, QTableWidgetItem("0.0"))  # Default starting value
            self.ic_table.setItem(row, 2, QTableWidgetItem("0.0"))  # Default lower bound
            self.ic_table.setItem(row, 3, QTableWidgetItem("1e6"))  # Default upper bound

            # Column 4: The "Fit?" Checkbox
            chk_item = QTableWidgetItem()
            new_flags = (chk_item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
            chk_item.setFlags(new_flags)

            # Default to UNCHECKED for Initial Conditions (meaning fixed = True)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.ic_table.setItem(row, 4, chk_item)

    def set_initial_conditions_from_save(self, ic_specs: list[dict]):
        """Populates the Initial Conditions table from a loaded JSON project file."""
        self.ic_table.setRowCount(len(ic_specs))

        for row, ic_data in enumerate(ic_specs):
            # Column 0: Species Name (Read-only).
            # Note: We check for 'species' first to match your backend, fallback to 'name'
            species_name = str(ic_data.get("species", ic_data.get("name", f"Species_{row}")))
            name_item = QTableWidgetItem(species_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.ic_table.setItem(row, 0, name_item)

            # Column 1: Initial Guess
            guess = str(ic_data.get("initial_guess", "0.0"))
            self.ic_table.setItem(row, 1, QTableWidgetItem(guess))

            # Column 2: Lower Bound
            lower = str(ic_data.get("lower_bound", "0.0"))
            self.ic_table.setItem(row, 2, QTableWidgetItem(lower))

            # Column 3: Upper Bound
            upper = str(ic_data.get("upper_bound", "1e6"))
            self.ic_table.setItem(row, 3, QTableWidgetItem(upper))

            # Column 4: The Checkbox
            # Default to True (fixed) if not found, since ICs shouldn't all fit by default
            is_fixed = ic_data.get("fixed", True)

            chk_item = QTableWidgetItem()
            new_flags = (chk_item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
            chk_item.setFlags(new_flags)

            # Translate backend logic to UI: If fixed=True, Uncheck the box!
            chk_item.setCheckState(Qt.CheckState.Unchecked if is_fixed else Qt.CheckState.Checked)
            self.ic_table.setItem(row, 4, chk_item)

    def get_initial_condition_specs(self) -> list[InitialConditionSpec]:
        """Reads the UI table and builds the specs for the Fit Engine."""
        specs = []
        for row in range(self.ic_table.rowCount()):
            name_item = self.ic_table.item(row, 0)
            guess_item = self.ic_table.item(row, 1)
            lower_item = self.ic_table.item(row, 2)
            upper_item = self.ic_table.item(row, 3)
            chk_item = self.ic_table.item(row, 4)

            if not all([name_item, guess_item, lower_item, upper_item, chk_item]):
                continue

            species_name = name_item.text()

            try:
                guess = float(guess_item.text())
                lower = float(lower_item.text())
                upper = float(upper_item.text())
            except ValueError:
                self.log_message(f"Warning: Invalid number in IC table for {species_name}. Defaulting to 0.0.")
                guess, lower, upper = 0.0, 0.0, 1e6

            # Extract the checkbox state (Checked = Fit, Unchecked = Fixed)
            is_fitted = chk_item.checkState() == Qt.CheckState.Checked
            is_fixed = not is_fitted

            specs.append(
                InitialConditionSpec(
                    species=species_name,  # <-- FIXED: correctly uses 'species'
                    initial_guess=guess,
                    lower_bound=lower,
                    upper_bound=upper,
                    fixed=is_fixed,
                    fixed_value=guess if is_fixed else None
                )
            )

        return specs

    def _on_run_clicked(self):
        dataset_name = self.combo_dataset.currentText()
        method = self.combo_method.currentText()

        if not dataset_name:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return

        self.log_message(f"Preparing to fit '{dataset_name}' using '{method}' optimizer...")
        self.run_fit_requested.emit(dataset_name, method)

    def log_message(self, text):
        """Appends text to the console display."""
        self.console.append(text)

    def clear_log(self):
        self.console.clear()