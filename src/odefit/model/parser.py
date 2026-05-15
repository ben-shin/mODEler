from signal import valid_signals

from odefit.model.reaction import Reaction


def strip_comment(line: str) -> str:
    """Remove anything after # for comments."""
    return line.split("#", 1)[0].strip()


def parse_species_term(term: str) -> tuple[str, int]:
    """
    Parse terms like:
        P1 -> ("P1", 1)
        2P1 -> ("P1", 2)
    """

    term = term.strip().replace(" ", "")

    if not term:
        raise ValueError("Empty species term")

    coefficient_str = ""

    for char in term:
        if char.isdigit():
            coefficient_str += char
        else:
            break

    if coefficient_str:
        coefficient = int(coefficient_str)
        species = term[len(coefficient_str) :]
    else:
        coefficient = 1
        species = term

    if not species:
        raise ValueError(f"Missing species name in term: {term}")

    if coefficient <= 0:
        raise ValueError(f"Stoichiometric coefficient must be positive: {term}")

    return species, coefficient


def parse_reaction_side(side: str) -> dict[str, int]:
    """
    Parse
        P1+P2
        2P1+P3

    into:
        {"P1": 1, "P2": 1}
    """

    side = side.strip()

    if not side:
        raise ValueError("Reaction side cannot be empty")

    species_dict = {}

    terms = side.split("+")

    for term in terms:
        species, coefficient = parse_species_term(term)

        if species in species_dict:
            species_dict[species] += coefficient
        else:
            species_dict[species] = coefficient

    return species_dict


def parse_reaction_line(line: str, reaction_index: int) -> Reaction:
    """
    Parse:
        2P1-P2  reversible
        P1+P2-P3    reversible
        A>B irreversible
    """

    line = strip_comment(line)

    if not line:
        raise ValueError("Cannot parse an empty reaction line")

    has_irreversible_arrow = ">" in line
    has_reversible_arrow = "-" in line

    if has_irreversible_arrow and has_reversible_arrow:
        raise ValueError(f"Reaction cannot contain both '>' and '-': {line}")

    forward_rate = f"k{reaction_index}f"

    if has_irreversible_arrow:
        parts = line.split(">")

        if len(parts) != 2:
            raise ValueError(f"Malformed irreversible reaction: {line}")

        left_side, right_side = parts
        reversible = False
        reverse_rate = None

    elif has_reversible_arrow:
        parts = line.split("-")

        if len(parts) != 2:
            raise ValueError(f"Malformed reversible reaction: {line}")

        left_side, right_side = parts
        reversible = True
        reverse_rate = f"k{reaction_index}r"
    else:
        raise ValueError(f"Could not find reaction arrow in line: {line}")

    reactants = parse_reaction_side(left_side)
    products = parse_reaction_side(right_side)

    return Reaction(
        reactants=reactants,
        products=products,
        reversible=reversible,
        forward_rate=forward_rate,
        reverse_rate=reverse_rate,
    )


def parse_model_text(text: str) -> list[Reaction]:
    """
    Parse a multiline reaction model.
    """

    reactions = []

    for line in text.splitlines():
        clean_line = strip_comment(line)

        if not clean_line:
            continue

        reaction_index = len(reactions) + 1
        reaction = parse_reaction_line(clean_line, reaction_index)

        reactions.append(reaction)

    if not reactions:
        raise ValueError("Model text does not contain any reactions")

    return reactions
