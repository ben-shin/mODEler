"""Input the model's reactions"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSplitter
from PySide6.QtCore import Qt, Signal

from odefit.model.model_spec import build_model_spec, ModelSpec
from odefit.model.ode_generator import generate_ode_lines


class ModelEditorPanel(QWidget):
    model_validated = Signal(ModelSpec)

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        header_label = QLabel("Define Model Reactions:")
        header_label.setStyleSheet("font-weight: bold; font-size: 14px;")

        self.validate_btn = QPushButton("Validate & Generate ODEs")
        # --- FIXED: Now points to the public method ---
        self.validate_btn.clicked.connect(self.validate_model)

        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(self.validate_btn)
        layout.addLayout(header_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: User input
        self.reaction_editor = QPlainTextEdit()
        self.reaction_editor.setPlaceholderText(
            "# Example:\n"
            "2P1-P2\n"
            "P1+P2-P3\n"
            "P2+P3-P5\n"
            "2P5>P10"
        )
        self.reaction_editor.setStyleSheet("font-family: monospace;")
        splitter.addWidget(self.reaction_editor)

        # Right side: Generated ODE view
        self.ode_preview = QPlainTextEdit()
        self.ode_preview.setReadOnly(True)
        self.ode_preview.setPlaceholderText("Generated ODEs will appear here...")
        self.ode_preview.setStyleSheet("font-family: monospace; background-color: #f5f5f5;")
        splitter.addWidget(self.ode_preview)

        layout.addWidget(splitter)

    def validate_model(self):
        """Validates the model text and triggers the generation of ODEs."""
        raw_text = self.reaction_editor.toPlainText()
        if not raw_text.strip():
            self.ode_preview.setPlainText("Please enter at least one reaction.")
            return

        try:
            model_spec = build_model_spec(raw_text)
            ode_lines = generate_ode_lines(model_spec)

            preview_text = "Model Validation Successful\n\n"
            preview_text += "Species: " + ", ".join(model_spec.species) + "\n"
            preview_text += "Parameters: " + ", ".join(model_spec.parameters) + "\n\n"
            preview_text += "Generated ODEs:\n" + "\n".join(ode_lines)

            self.ode_preview.setStyleSheet("font-family: monospace; color: black; background-color: #f5f5f5;")
            self.ode_preview.setPlainText(preview_text)
            self.model_validated.emit(model_spec)

        except Exception as e:
            self.ode_preview.setStyleSheet("font-family: monospace; color: red; background-color: #ffeeee;")
            self.ode_preview.setPlainText(f"Error:\n\n{str(e)}")

    def get_model_spec(self) -> ModelSpec:
        """Passes the current ODE model to the Fit Engine."""
        raw_text = self.reaction_editor.toPlainText()
        return build_model_spec(raw_text)