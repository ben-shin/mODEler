"""Table of parameters will generate upon model validation"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel
)
from PySide6.QtCore import Slot, Qt
from odefit.model.model_spec import ModelSpec


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
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 4):
            header.setSectionResizeMode(i, QHeaderView.Stretch)

        layout.addWidget(self.table)

    @Slot(ModelSpec)
    def update_from_model(self, model_spec: ModelSpec):
        parameters = model_spec.parameters
        self.table.setRowCount(len(parameters))

        for row, param_name in enumerate(parameters):
            name_item = QTableWidgetItem(param_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem("1.0"))
            self.table.setItem(row, 2, QTableWidgetItem("1e-6"))
            self.table.setItem(row, 3, QTableWidgetItem("1e6"))