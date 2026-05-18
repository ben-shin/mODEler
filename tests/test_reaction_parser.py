import pytest

from odefit.model.parser import parse_model_text, parse_reaction_line


def test_parse_explicit_irreversible_arrow():
    reaction = parse_reaction_line("A->B", reaction_index=1)

    assert reaction.reactants == {"A": 1}
    assert reaction.products == {"B": 1}
    assert reaction.reversible is False
    assert reaction.forward_rate == "k1f"
    assert reaction.reverse_rate is None


def test_parse_explicit_reversible_arrow():
    reaction = parse_reaction_line("A<->B", reaction_index=1)

    assert reaction.reactants == {"A": 1}
    assert reaction.products == {"B": 1}
    assert reaction.reversible is True
    assert reaction.forward_rate == "k1f"
    assert reaction.reverse_rate == "k1r"


def test_parse_products_with_multiple_species():
    reaction = parse_reaction_line("A>B+C", reaction_index=1)

    assert reaction.reactants == {"A": 1}
    assert reaction.products == {"B": 1, "C": 1}
    assert reaction.reversible is False


def test_parse_explicit_arrow_with_spaces():
    reaction = parse_reaction_line("A + B -> C + D", reaction_index=1)

    assert reaction.reactants == {"A": 1, "B": 1}
    assert reaction.products == {"C": 1, "D": 1}
    assert reaction.reversible is False


def test_parse_reversible_explicit_arrow_with_stoichiometry():
    reaction = parse_reaction_line("2A <-> A2", reaction_index=1)

    assert reaction.reactants == {"A": 2}
    assert reaction.products == {"A2": 1}
    assert reaction.reversible is True


def test_parse_reaction_label():
    reaction = parse_reaction_line(
        "dimerization: 2A<->A2",
        reaction_index=1,
        source_line=7,
    )

    assert reaction.label == "dimerization"
    assert reaction.source_line == 7
    assert reaction.reactants == {"A": 2}
    assert reaction.products == {"A2": 1}


def test_parse_model_text_preserves_source_line_numbers():
    model_text = """
# comment
A->B

B->C
"""

    reactions = parse_model_text(model_text)

    assert len(reactions) == 2

    assert reactions[0].source_line == 3
    assert reactions[1].source_line == 5


def test_parse_model_text_reaction_indices_ignore_comments_and_blanks():
    model_text = """
# comment
A->B

B<->C
"""

    reactions = parse_model_text(model_text)

    assert reactions[0].forward_rate == "k1f"
    assert reactions[0].reverse_rate is None

    assert reactions[1].forward_rate == "k2f"
    assert reactions[1].reverse_rate == "k2r"


def test_parser_error_reports_line_number():
    model_text = """
A->B
bad reaction
"""

    with pytest.raises(ValueError, match="Line 3"):
        parse_model_text(model_text)


def test_parser_rejects_multiple_arrows_with_line_number():
    model_text = """
A->B>C
"""

    with pytest.raises(ValueError, match="Line 2"):
        parse_model_text(model_text)


def test_parser_rejects_empty_right_side_with_line_number():
    model_text = """
A->
"""

    with pytest.raises(ValueError, match="Line 2"):
        parse_model_text(model_text)


def test_parser_rejects_empty_left_side_with_line_number():
    model_text = """
->B
"""

    with pytest.raises(ValueError, match="Line 2"):
        parse_model_text(model_text)
