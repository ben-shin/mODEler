from odefit.model.parser import parse_model_text
from odefit.model.species import get_species_from_reactions, natural_sort_key


def test_get_species_from_reversible_reaction():
    reactions = parse_model_text("A-B")

    species = get_species_from_reactions(reactions)

    assert species == ["A", "B"]


def test_get_species_from_irreversible_reaction():
    reactions = parse_model_text("A>B")

    species = get_species_from_reactions(reactions)

    assert species == ["A", "B"]


def test_get_species_from_mixed_model():
    reactions = parse_model_text(
        """
A>B
B-C
"""
    )

    species = get_species_from_reactions(reactions)

    assert species == ["A", "B", "C"]


def test_get_species_from_oligomer_model():
    reactions = parse_model_text(
        """
2A-A2
A2+A-A3
"""
    )

    species = get_species_from_reactions(reactions)

    assert species == ["A", "A2", "A3"]


def test_natural_sort_key_orders_numbers_naturally():
    species = ["P10", "P2", "P1", "P11", "P3"]

    sorted_species = sorted(species, key=natural_sort_key)

    assert sorted_species == ["P1", "P2", "P3", "P10", "P11"]


def test_get_species_from_reactions_uses_natural_sorting():
    reactions = parse_model_text(
        """
P10>P11
P1>P2
P2>P10
"""
    )

    species = get_species_from_reactions(reactions)

    assert species == ["P1", "P2", "P10", "P11"]


def test_natural_sorting_with_mixed_species_prefixes():
    reactions = parse_model_text(
        """
A10>A11
A2>A10
B1>B2
"""
    )

    species = get_species_from_reactions(reactions)

    assert species == ["A2", "A10", "A11", "B1", "B2"]
