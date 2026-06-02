"""Table of parameters will generate upon model validation"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel
)
from PySide6.QtCore import Slot, Qt
from odefit.model.model_spec import ModelSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.initial_condition_spec import InitialConditionSpec


class ParameterTablePanel(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        label = QLabel("Parameter Configuration:")
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Parameter", "Initial Guess", "Lower Bound", "Upper Bound"])

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(1, 4):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)


        layout.addWidget(self.table)

    @Slot(ModelSpec)
    def update_from_model(self, model_spec: ModelSpec):
        parameters = model_spec.parameters
        self.table.setRowCount(len(parameters))

        for row, param_name in enumerate(parameters):
            name_item = QTableWidgetItem(param_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem("1.0"))
            self.table.setItem(row, 2, QTableWidgetItem("1e-6"))
            self.table.setItem(row, 3, QTableWidgetItem("1e6"))

    def get_parameter_specs(self) -> list[ParameterSpec]:
        """Passes the kinetic parameters from the UI table to the Fit Engine."""
        specs = []
        for row in range(self.table.rowCount()):
            # 1. Safely grab the items first
            name_item = self.table.item(row, 0)
            guess_item = self.table.item(row, 1)
            lower_item = self.table.item(row, 2)
            upper_item = self.table.item(row, 3)

            # 2. Extract the text safely (with fallback defaults just in case)
            param_name = name_item.text() if name_item is not None else f"Param_{row}"
            guess = float(guess_item.text()) if guess_item is not None else 1.0
            lower = float(lower_item.text()) if lower_item is not None else 1e-6
            upper = float(upper_item.text()) if upper_item is not None else 1e6

            specs.append(
                ParameterSpec(
                    name=param_name,
                    initial_guess=guess,
                    lower_bound=lower,
                    upper_bound=upper
                )
            )
        return specs

    def get_initial_condition_specs(self) -> list[InitialConditionSpec]:
        """Passes the starting concentrations to the Fit Engine."""
        # Since we haven't built an Initial Conditions UI table yet, we return an empty list.
        # Ben's engine will safely default the first species to 1.0 and the rest to 0.0!
        return []

    def set_parameters_from_save(self, parameter_specs: list[dict]):
        """Populates the UI table from a loaded JSON project file."""
        # 1. Resize the table to fit the incoming saved data
        self.table.setRowCount(len(parameter_specs))

        # 2. Loop through the saved dictionaries and fill the rows
        for row, param_data in enumerate(parameter_specs):
            # Column 0: Name (Keep it read-only just like the auto-generator!)
            name_item = QTableWidgetItem(str(param_data.get("name", f"Param_{row}")))
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            # Column 1: Initial Guess
            guess = str(param_data.get("initial_guess", "1.0"))
            self.table.setItem(row, 1, QTableWidgetItem(guess))

            # Column 2: Lower Bound
            lower = str(param_data.get("lower_bound", "1e-6"))
            self.table.setItem(row, 2, QTableWidgetItem(lower))

            # Column 3: Upper Bound
            upper = str(param_data.get("upper_bound", "1e6"))
            self.table.setItem(row, 3, QTableWidgetItem(upper))