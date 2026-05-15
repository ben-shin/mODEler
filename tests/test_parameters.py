import pytest

from odefit.model.parameters import detect_parameters
from odefit.model.parser import parse_model_text
from odefit.model.reaction import Reaction


def test_detect_parameters_from_reversible_model():
    reactions = parse_model_text("""
    A-B
    B-C
    """)

    parameters = detect_parameters(reactions)

    assert parameters == ["k1f", "k1r", "k2f", "k2r"]


def test_detect_parameters_from_irreversible_model():
    reactions = parse_model_text("""
    A>B
    B>C
    """)

    parameters = detect_parameters(reactions)

    assert parameters == ["k1f", "k2f"]


def test_detect_parameters_from_mixed_model():
    reactions = parse_model_text("""
    A-B
    B>C
    C-D
    """)

    parameters = detect_parameters(reactions)

    assert parameters == ["k1f", "k1r", "k2f", "k3f", "k3r"]


def test_reversible_reaction_without_reverse_rate_raises_error():
    reactions = [
        Reaction(
            reactants={"A": 1},
            products={"B": 1},
            reversible=True,
            forward_rate="k1f",
            reverse_rate=None,
        )
    ]

    with pytest.raises(ValueError):
        detect_parameters(reactions)
