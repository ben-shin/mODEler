import pytest

from odefit.model.parser import parse_model_text


def test_parse_reversible_reactions():
    text = """
    2P1-P2
    P1+P2-P3
    """

    reactions = parse_model_text(text)

    assert len(reactions) == 2

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


def test_parse_irreversible_reaction():
    reactions = parse_model_text("A>B")

    assert len(reactions) == 1
    assert reactions[0].reactants == {"A": 1}
    assert reactions[0].products == {"B": 1}
    assert reactions[0].reversible is False
    assert reactions[0].forward_rate == "k1f"
    assert reactions[0].reverse_rate is None


def test_parser_ignores_empty_lines():
    text = """


    A-B


    B-C
    """

    reactions = parse_model_text(text)

    assert len(reactions) == 2
    assert reactions[0].forward_rate == "k1f"
    assert reactions[1].forward_rate == "k2f"


def test_parser_supports_spaces():
    reactions = parse_model_text("P1 + P2 - P3")

    assert reactions[0].reactants == {"P1": 1, "P2": 1}
    assert reactions[0].products == {"P3": 1}


def test_parser_ignores_comments():
    text = """
    # This is a comment
    A-B  # reversible transition
    B>C  # irreversible transition
    """

    reactions = parse_model_text(text)

    assert len(reactions) == 2
    assert reactions[0].reversible is True
    assert reactions[1].reversible is False


def test_empty_model_raises_error():
    with pytest.raises(ValueError):
        parse_model_text("""
        # only comments

        """)


def test_missing_arrow_raises_error():
    with pytest.raises(ValueError):
        parse_model_text("A+B")


def test_empty_left_side_raises_error():
    with pytest.raises(ValueError):
        parse_model_text("-B")


def test_empty_right_side_raises_error():
    with pytest.raises(ValueError):
        parse_model_text("A-")


def test_multiple_reversible_arrows_raise_error():
    with pytest.raises(ValueError):
        parse_model_text("A-B-C")


def test_multiple_irreversible_arrows_raise_error():
    with pytest.raises(ValueError):
        parse_model_text("A>B>C")


def test_mixed_arrows_raise_error():
    with pytest.raises(ValueError):
        parse_model_text("A-B>C")


def test_missing_species_after_coefficient_raises_error():
    with pytest.raises(ValueError):
        parse_model_text("2-B")
