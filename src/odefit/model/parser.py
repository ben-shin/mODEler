from odefit.model.reaction import Reaction


def parse_species_term(term: str) -> tuple[str, int]:
    """
    Parse terms like:
        P1 -> ("P1", 1)
        2P1 -> ("P1", 2)
    """

    term = term.strip()

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

    return species, coefficient


def parse_reaction_side(side: str) -> dict[str, int]:
    """
    Parse
        P1+P2
        2P1+P3

    into:
        {"P1": 1, "P2": 1}
    """

    species_dict = {}

    terms = side.split("+")

    for term in terms:
        species, coefficient = parse_species_term(term)
        species_dict[species] = coefficient

    return species_dict


def parse_reaction_line(line: str, reaction_index: int) -> Reaction:
    """
    Parse:
        2P1-P2  reversible
        P1+P2-P3    reversible
        A>B irreversible
    """

    line = line.strip()

    forward_rate = f"k{reaction_index}f"

    if ">" in line:
        left_side, right_side = line.split(">")
        reversible = False
        reverse_rate = None
    elif "-" in line:
        left_side, right_side = line.split("-")
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

    lines = text.strip().splitlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        reaction = parse_reaction_line(line, i)

        reactions.append(reaction)

    return reactions
