from odefit.model.model_spec import ModelSpec, build_model_spec


def test_build_model_spec_from_reversible_model():
    text = """
    2P1-P2
    P1+P2-P3
    """

    model = build_model_spec(text)

    assert isinstance(model, ModelSpec)

    assert len(model.reactions) == 2
    assert model.species == ["P1", "P2", "P3"]
    assert model.parameters == ["k1f", "k1r", "k2f", "k2r"]


def test_build_model_spec_from_irreversible_model():
    text = """
    A>B
    B>C
    """

    model = build_model_spec(text)

    assert len(model.reactions) == 2
    assert model.species == ["A", "B", "C"]
    assert model.parameters == ["k1f", "k2f"]


def test_build_model_spec_from_mixed_model():
    text = """
    A-B
    B>C
    C-D
    """

    model = build_model_spec(text)

    assert len(model.reactions) == 3
    assert model.species == ["A", "B", "C", "D"]
    assert model.parameters == ["k1f", "k1r", "k2f", "k3f", "k3r"]


def test_model_spec_keeps_raw_text():
    text = """
    A-B
    """

    model = build_model_spec(text)

    assert model.raw_text == text
