from odefit.model.parser import parse_model_text
from odefit.model.species import detect_species


def test_detect_species_from_reversible_model():
    text = """
    2P1-P2
    P1+P2-P3
    """

    reactions = parse_model_text(text)
    species = detect_species(reactions)

    assert species == ["P1", "P2", "P3"]


def test_detect_species_from_irreversible_model():
    text = """
    A>B
    B>C
    """

    reactions = parse_model_text(text)
    species = detect_species(reactions)

    assert species == ["A", "B", "C"]


def test_detect_species_from_mixed_model():
    text = """
    A-B
    B>C
    C+D-E
    """

    reactions = parse_model_text(text)
    species = detect_species(reactions)

    assert species == ["A", "B", "C", "D", "E"]


def test_detect_species_from_oligomer_model():
    text = """
    2P1-P2
    P1+P2-P3
    P1+P3-P4
    2P2-P4
    P1+P4-P5
    P2+P3-P5
    P1+P5-P6
    P2+P4-P6
    2P3-P6
    """

    reactions = parse_model_text(text)
    species = detect_species(reactions)

    assert species == ["P1", "P2", "P3", "P4", "P5", "P6"]
