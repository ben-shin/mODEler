import pytest

from odefit.model.parser import parse_model_text
from odefit.model.rate_laws import (
    generate_forward_rate,
    generate_mass_action_rate,
    generate_reaction_rates,
    generate_reverse_rate,
)
from odefit.model.reaction import Reaction


def test_generate_mass_action_rate_single_species():
    rate = generate_mass_action_rate({"A": 1}, "k1f")

    assert rate == "k1f*A"


def test_generate_mass_action_rate_dimer():
    rate = generate_mass_action_rate({"P1": 2}, "k1f")

    assert rate == "k1f*P1^2"


def test_generate_mass_action_rate_two_species():
    rate = generate_mass_action_rate({"P1": 1, "P2": 1}, "k2f")

    assert rate == "k2f*P1*P2"


def test_generate_forward_rate():
    reaction = parse_model_text("2P1-P2")[0]

    rate = generate_forward_rate(reaction)

    assert rate == "k1f*P1^2"


def test_generate_reverse_rate():
    reaction = parse_model_text("2P1-P2")[0]

    rate = generate_reverse_rate(reaction)

    assert rate == "k1r*P2"


def test_generate_reverse_rate_for_irreversible_reaction_returns_none():
    reaction = parse_model_text("A>B")[0]

    rate = generate_reverse_rate(reaction)

    assert rate is None


def test_generate_reaction_rates_for_reversible_reaction():
    reaction = parse_model_text("P1+P2-P3")[0]

    forward_rate, reverse_rate = generate_reaction_rates(reaction)

    assert forward_rate == "k1f*P1*P2"
    assert reverse_rate == "k1r*P3"


def test_reversible_reaction_without_reverse_rate_raises_error():
    reaction = Reaction(
        reactants={"A": 1},
        products={"B": 1},
        reversible=True,
        forward_rate="k1f",
        reverse_rate=None,
    )

    with pytest.raises(ValueError):
        generate_reverse_rate(reaction)
