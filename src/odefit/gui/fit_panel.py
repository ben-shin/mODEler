from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QTextEdit, QMessageBox, QGroupBox
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont


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

        control_layout.addWidget(QLabel("Target Dataset:"))
        self.combo_dataset = QComboBox()
        control_layout.addWidget(self.combo_dataset)

        control_layout.addWidget(QLabel("Optimizer:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["trf", "lm", "dogbox"])  # Scipy's standard least-squares methods
        control_layout.addWidget(self.combo_method)

        self.btn_run = QPushButton("RUN GLOBAL FIT")
        self.btn_run.setStyleSheet(
            "background-color: #2e7d32; color: white; font-weight: bold; padding: 8px;"
        )
        self.btn_run.clicked.connect(self._on_run_clicked)
        self.btn_run.setEnabled(False)  # Disabled until data is loaded
        control_layout.addWidget(self.btn_run)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # --- Bottom Section: Output Console ---
        log_group = QGroupBox("Output")
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