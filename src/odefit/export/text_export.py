from pathlib import Path

from odefit.model.model_spec import ModelSpec, build_model_spec
from odefit.model.ode_generator import generate_ode_lines


def read_model_text_file(file_path: str | Path) -> str:
    """
    Read a model definition from a text file.
    """

    path = Path(file_path)

    return path.read_text()


def build_model_spec_from_file(file_path: str | Path) -> ModelSpec:
    """
    Read a model text file and build a ModelSpec.
    """

    model_text = read_model_text_file(file_path)

    return build_model_spec(model_text)


def write_generated_odes(
    model: ModelSpec,
    file_path: str | Path,
) -> None:
    """
    Write generated ODEs to a text file.
    """

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ode_lines = generate_ode_lines(model)

    output_text = "\n".join(ode_lines)

    path.write_text(output_text)
