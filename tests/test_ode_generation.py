from odefit.model.model_spec import build_model_spec
from odefit.model.ode_generator import (
    format_ode_expression,
    format_ode_term,
    generate_ode_expressions,
    generate_ode_lines,
    generate_ode_terms,
)


def test_format_ode_term_first_negative_unit_coefficient():
    term = format_ode_term(-1, "k1f*A", is_first_term=True)

    assert term == "-k1f*A"


def test_format_ode_term_first_positive_unit_coefficient():
    term = format_ode_term(1, "k1r*B", is_first_term=True)

    assert term == "k1r*B"


def test_format_ode_term_later_negative_coefficient():
    term = format_ode_term(-2, "k1f*A^2", is_first_term=False)

    assert term == " - 2*k1f*A^2"


def test_format_ode_term_later_positive_coefficient():
    term = format_ode_term(2, "k1r*B", is_first_term=False)

    assert term == " + 2*k1r*B"


def test_format_ode_expression():
    expression = format_ode_expression(
        [
            (-2, "k1f*P1^2"),
            (2, "k1r*P2"),
        ]
    )

    assert expression == "-2*k1f*P1^2 + 2*k1r*P2"


def test_generate_ode_terms_for_reversible_dimerization():
    model = build_model_spec("2P1-P2")

    ode_terms = generate_ode_terms(model.reactions, model.species)

    assert ode_terms["P1"] == [
        (-2, "k1f*P1^2"),
        (2, "k1r*P2"),
    ]

    assert ode_terms["P2"] == [
        (1, "k1f*P1^2"),
        (-1, "k1r*P2"),
    ]


def test_generate_ode_expressions_for_reversible_dimerization():
    model = build_model_spec("2P1-P2")

    odes = generate_ode_expressions(model)

    assert odes["P1"] == "-2*k1f*P1^2 + 2*k1r*P2"
    assert odes["P2"] == "k1f*P1^2 - k1r*P2"


def test_generate_ode_expressions_for_bimolecular_reaction():
    model = build_model_spec("P1+P2-P3")

    odes = generate_ode_expressions(model)

    assert odes["P1"] == "-k1f*P1*P2 + k1r*P3"
    assert odes["P2"] == "-k1f*P1*P2 + k1r*P3"
    assert odes["P3"] == "k1f*P1*P2 - k1r*P3"


def test_generate_ode_expressions_for_irreversible_reaction():
    model = build_model_spec("A>B")

    odes = generate_ode_expressions(model)

    assert odes["A"] == "-k1f*A"
    assert odes["B"] == "k1f*A"


def test_generate_ode_expressions_for_mixed_model():
    model = build_model_spec(
        """
        A-B
        B>C
        """
    )

    odes = generate_ode_expressions(model)

    assert odes["A"] == "-k1f*A + k1r*B"
    assert odes["B"] == "k1f*A - k1r*B - k2f*B"
    assert odes["C"] == "k2f*B"


def test_generate_ode_lines():
    model = build_model_spec("A-B")

    lines = generate_ode_lines(model)

    assert lines == [
        "dA/dt = -k1f*A + k1r*B",
        "dB/dt = k1f*A - k1r*B",
    ]
