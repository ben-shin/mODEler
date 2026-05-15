from odefit.export.text_export import (
    build_model_spec_from_file,
    read_model_text_file,
    write_generated_odes,
)
from odefit.model.model_spec import build_model_spec


def test_read_model_text_file(tmp_path):
    model_path = tmp_path / "model.txt"
    model_path.write_text("A-B\nB>C\n")

    text = read_model_text_file(model_path)

    assert text == "A-B\nB>C\n"


def test_build_model_spec_from_file(tmp_path):
    model_path = tmp_path / "model.txt"
    model_path.write_text("A-B\nB>C\n")

    model = build_model_spec_from_file(model_path)

    assert model.species == ["A", "B", "C"]
    assert model.parameters == ["k1f", "k1r", "k2f"]


def test_write_generated_odes(tmp_path):
    model = build_model_spec("A-B")

    output_path = tmp_path / "generated_odes.txt"

    write_generated_odes(
        model=model,
        file_path=output_path,
    )

    output_text = output_path.read_text()

    assert output_text == ("dA/dt = -k1f*A + k1r*B\ndB/dt = k1f*A - k1r*B")
