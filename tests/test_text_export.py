from odefit.export.text_export import (
    build_model_spec_from_file,
    build_model_summary_text,
    build_reaction_table_text,
    format_reaction,
    format_reaction_side,
    write_generated_odes,
    write_model_summary,
    write_reaction_table,
)
from odefit.model.model_spec import build_model_spec


def test_build_model_spec_from_file(tmp_path):
    model_path = tmp_path / "model.txt"

    model_path.write_text("A>B")

    model = build_model_spec_from_file(model_path)

    assert model.raw_text == "A>B"
    assert model.species == ["A", "B"]
    assert model.parameters == ["k1f"]
    assert model.metadata["source_file"] == str(model_path)


def test_build_model_spec_from_file_accepts_name_and_metadata(tmp_path):
    model_path = tmp_path / "model.txt"

    model_path.write_text("A>B")

    model = build_model_spec_from_file(
        file_path=model_path,
        name="first_order_loss",
        metadata={
            "experiment": "NMR",
        },
    )

    assert model.name == "first_order_loss"
    assert model.metadata["experiment"] == "NMR"
    assert model.metadata["source_file"] == str(model_path)


def test_write_generated_odes(tmp_path):
    model = build_model_spec("A>B")

    output_path = tmp_path / "generated_odes.txt"

    written_path = write_generated_odes(
        model=model,
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    text = output_path.read_text()

    assert "dA/dt" in text
    assert "dB/dt" in text
    assert "k1f" in text


def test_format_reaction_side_single_species():
    text = format_reaction_side({"A": 1})

    assert text == "A"


def test_format_reaction_side_with_stoichiometry():
    text = format_reaction_side({"A": 2})

    assert text == "2A"


def test_format_reaction_side_multiple_species():
    text = format_reaction_side(
        {
            "A": 1,
            "B": 1,
        }
    )

    assert text == "A + B"


def test_format_reaction_irreversible():
    model = build_model_spec("A>B")

    text = format_reaction(model.reactions[0])

    assert text == "A -> B"


def test_format_reaction_reversible():
    model = build_model_spec("A-B")

    text = format_reaction(model.reactions[0])

    assert text == "A <-> B"


def test_format_reaction_with_label():
    model = build_model_spec("loss: A>B")

    text = format_reaction(model.reactions[0])

    assert text == "loss: A -> B"


def test_build_reaction_table_text():
    model = build_model_spec(
        """
loss: A>B
binding: A+B<->C
"""
    )

    text = build_reaction_table_text(model)

    assert "Reaction table" in text
    assert "index" in text
    assert "source_line" in text
    assert "label" in text
    assert "loss" in text
    assert "binding" in text
    assert "A -> B" in text
    assert "A + B <-> C" in text
    assert "k1f" in text
    assert "k2f" in text
    assert "k2r" in text


def test_write_reaction_table(tmp_path):
    model = build_model_spec("A>B")

    output_path = tmp_path / "reaction_table.txt"

    written_path = write_reaction_table(
        model=model,
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    text = output_path.read_text()

    assert "Reaction table" in text
    assert "A -> B" in text


def test_build_model_summary_text():
    model = build_model_spec(
        "loss: A>B",
        name="first_order_loss",
        metadata={
            "experiment": "NMR amide",
        },
    )

    text = build_model_summary_text(model)

    assert "Model summary" in text
    assert "Name: first_order_loss" in text
    assert "experiment: NMR amide" in text
    assert "Species" in text
    assert "- A" in text
    assert "- B" in text
    assert "Parameters" in text
    assert "- k1f" in text
    assert "Warnings" in text
    assert "Generated ODEs" in text
    assert "dA/dt" in text
    assert "Reaction table" in text
    assert "loss: A -> B" in text


def test_write_model_summary(tmp_path):
    model = build_model_spec(
        "A>B",
        name="first_order_loss",
    )

    output_path = tmp_path / "model_summary.txt"

    written_path = write_model_summary(
        model=model,
        file_path=output_path,
    )

    assert written_path == output_path
    assert output_path.exists()

    text = output_path.read_text()

    assert "Model summary" in text
    assert "Name: first_order_loss" in text
    assert "Generated ODEs" in text
