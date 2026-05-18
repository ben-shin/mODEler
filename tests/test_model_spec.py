from odefit.model.model_spec import (
    ModelSpec,
    build_model_spec,
    build_model_warnings,
)


def test_build_model_spec_returns_model_spec():
    model = build_model_spec("A>B")

    assert isinstance(model, ModelSpec)


def test_build_model_spec_stores_raw_text():
    text = "A>B"

    model = build_model_spec(text)

    assert model.raw_text == text


def test_build_model_spec_stores_reactions():
    model = build_model_spec("A>B")

    assert len(model.reactions) == 1
    assert model.reactions[0].reactants == {"A": 1}
    assert model.reactions[0].products == {"B": 1}


def test_build_model_spec_stores_species():
    model = build_model_spec("A>B")

    assert model.species == ["A", "B"]


def test_build_model_spec_stores_parameters_for_irreversible_model():
    model = build_model_spec("A>B")

    assert model.parameters == ["k1f"]


def test_build_model_spec_stores_parameters_for_reversible_model():
    model = build_model_spec("A-B")

    assert model.parameters == ["k1f", "k1r"]


def test_build_model_spec_handles_mixed_model():
    model = build_model_spec(
        """
A>B
B-C
"""
    )

    assert model.species == ["A", "B", "C"]
    assert model.parameters == ["k1f", "k2f", "k2r"]


def test_build_model_spec_accepts_name():
    model = build_model_spec(
        "A>B",
        name="first_order_loss",
    )

    assert model.name == "first_order_loss"


def test_build_model_spec_name_defaults_to_none():
    model = build_model_spec("A>B")

    assert model.name is None


def test_build_model_spec_accepts_metadata():
    model = build_model_spec(
        "A>B",
        metadata={
            "experiment": "NMR amide timecourse",
            "temperature": "25C",
        },
    )

    assert model.metadata == {
        "experiment": "NMR amide timecourse",
        "temperature": "25C",
    }


def test_build_model_spec_metadata_defaults_to_empty_dict():
    model = build_model_spec("A>B")

    assert model.metadata == {}


def test_build_model_spec_has_warnings_list():
    model = build_model_spec("A>B")

    assert isinstance(model.warnings, list)


def test_build_model_warnings_detects_source_and_sink_species():
    model = build_model_spec("A>B")

    assert any("only as reactants" in warning for warning in model.warnings)
    assert any("only as products" in warning for warning in model.warnings)


def test_build_model_warnings_detects_only_irreversible_model():
    model = build_model_spec(
        """
A>B
B>C
"""
    )

    assert any("only irreversible" in warning for warning in model.warnings)


def test_build_model_warnings_detects_only_reversible_model():
    model = build_model_spec(
        """
A-B
B-C
"""
    )

    assert any("only reversible" in warning for warning in model.warnings)


def test_build_model_spec_species_uses_natural_sorting():
    model = build_model_spec(
        """
P10>P11
P1>P2
P2>P10
"""
    )

    assert model.species == ["P1", "P2", "P10", "P11"]


def test_build_model_warnings_empty_reaction_list():
    warnings = build_model_warnings(
        reactions=[],
        species=[],
    )

    assert warnings == ["Model contains no reactions"]


def test_build_model_spec_preserves_reaction_labels_and_source_lines():
    model = build_model_spec(
        """
first: A>B
second: B>C
"""
    )

    assert model.reactions[0].label == "first"
    assert model.reactions[1].label == "second"

    assert model.reactions[0].source_line == 2
    assert model.reactions[1].source_line == 3
