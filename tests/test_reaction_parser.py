from odefit.model.parser import parse_model_text


def test_parse_model_text():

    text = """
    2P1-P2
    P1+P2-P3
    2P3>P6
    """

    reactions = parse_model_text(text)

    assert len(reactions) == 3

    assert reactions[0].reactants == {"P1": 2}
    assert reactions[0].products == {"P2": 1}
    assert reactions[0].reversible is True
    assert reactions[0].forward_rate == "k1f"
    assert reactions[0].reverse_rate == "k1r"

    assert reactions[1].reactants == {"P1": 1, "P2": 1}
    assert reactions[1].products == {"P3": 1}
    assert reactions[1].reversible is True
    assert reactions[1].forward_rate == "k2f"
    assert reactions[1].reverse_rate == "k2r"

    assert reactions[2].reactants == {"P3": 2}
    assert reactions[2].products == {"P6": 1}
    assert reactions[2].reversible is False
    assert reactions[2].forward_rate == "k3f"
    assert reactions[2].reverse_rate is None
