import re

from odefit.model.reaction import Reaction


SUPPORTED_ARROWS = ("<->", "->", ">", "-")


def format_parser_error(
    message: str,
    source_line: int | None = None,
) -> str:
    """
    Format parser errors with optional source line number.
    """

    if source_line is None:
        return message

    return f"Line {source_line}: {message}"


def strip_comment(line: str) -> str:
    """
    Remove comments from a model line.

    Everything after # is ignored.
    """

    return line.split("#", 1)[0].strip()


def parse_optional_label(line: str) -> tuple[str | None, str]:
    """
    Parse optional reaction label.

    Supported syntax:

        label: A->B
        dimerization: 2A<->A2

    Returns:
        label, remaining_reaction_text
    """

    if ":" not in line:
        return None, line

    possible_label, remaining = line.split(":", 1)

    possible_label = possible_label.strip()
    remaining = remaining.strip()

    if not possible_label:
        return None, line

    if not remaining:
        return None, line

    return possible_label, remaining


def find_reaction_arrow(
    line: str,
    source_line: int | None = None,
) -> tuple[str, int]:
    """
    Find exactly one reaction arrow in a line.

    Supported arrows:
        A-B      reversible shorthand
        A>B      irreversible shorthand
        A<->B    explicit reversible
        A->B     explicit irreversible
    """

    found_arrows: list[tuple[str, int]] = []

    index = 0

    while index < len(line):
        if line.startswith("<->", index):
            found_arrows.append(("<->", index))
            index += 3
        elif line.startswith("->", index):
            found_arrows.append(("->", index))
            index += 2
        elif line[index] == ">":
            found_arrows.append((">", index))
            index += 1
        elif line[index] == "-":
            found_arrows.append(("-", index))
            index += 1
        else:
            index += 1

    if not found_arrows:
        raise ValueError(
            format_parser_error(
                f"Could not find reaction arrow in line: {line}",
                source_line,
            )
        )

    if len(found_arrows) > 1:
        arrows = ", ".join(arrow for arrow, _ in found_arrows)
        raise ValueError(
            format_parser_error(
                f"Found multiple reaction arrows ({arrows}) in line: {line}",
                source_line,
            )
        )

    return found_arrows[0]


def parse_species_term(
    term: str,
    source_line: int | None = None,
) -> tuple[str, int]:
    """
    Parse terms like:
        P1   -> ("P1", 1)
        2P1  -> ("P1", 2)
        P10  -> ("P10", 1)
    """

    term = term.strip().replace(" ", "")

    if not term:
        raise ValueError(
            format_parser_error(
                "Empty species term",
                source_line,
            )
        )

    match = re.fullmatch(r"(\d*)([A-Za-z_][A-Za-z0-9_]*)", term)

    if match is None:
        raise ValueError(
            format_parser_error(
                f"Malformed species term: {term}",
                source_line,
            )
        )

    coefficient_text, species = match.groups()

    if coefficient_text:
        coefficient = int(coefficient_text)
    else:
        coefficient = 1

    if coefficient <= 0:
        raise ValueError(
            format_parser_error(
                f"Stoichiometric coefficient must be positive: {term}",
                source_line,
            )
        )

    return species, coefficient


def parse_reaction_side(
    side: str,
    source_line: int | None = None,
) -> dict[str, int]:
    """
    Parse one side of a reaction.

    Examples:
        P1
        2P1
        P1+P2
        2P1+P3
    """

    side = side.strip()

    if not side:
        raise ValueError(
            format_parser_error(
                "Reaction side is empty",
                source_line,
            )
        )

    species_dict: dict[str, int] = {}

    terms = side.split("+")

    for term in terms:
        if not term.strip():
            raise ValueError(
                format_parser_error(
                    f"Empty species term in reaction side: {side}",
                    source_line,
                )
            )

        species, coefficient = parse_species_term(
            term,
            source_line=source_line,
        )

        species_dict[species] = species_dict.get(species, 0) + coefficient

    return species_dict


def parse_reaction_line(
    line: str,
    reaction_index: int,
    source_line: int | None = None,
) -> Reaction:
    """
    Parse one reaction line.

    Supported:
        A-B
        A>B
        A<->B
        A->B
        A>B+C
        label: A->B
    """

    line = strip_comment(line)

    if not line:
        raise ValueError(
            format_parser_error(
                "Reaction line is empty",
                source_line,
            )
        )

    label, line_without_label = parse_optional_label(line)

    arrow, arrow_index = find_reaction_arrow(
        line_without_label,
        source_line=source_line,
    )

    left_side = line_without_label[:arrow_index].strip()
    right_side = line_without_label[arrow_index + len(arrow) :].strip()

    if not left_side:
        raise ValueError(
            format_parser_error(
                f"Reaction left side is empty: {line_without_label}",
                source_line,
            )
        )

    if not right_side:
        raise ValueError(
            format_parser_error(
                f"Reaction right side is empty: {line_without_label}",
                source_line,
            )
        )

    if arrow in {"<->", "-"}:
        reversible = True
        reverse_rate = f"k{reaction_index}r"
    elif arrow in {"->", ">"}:
        reversible = False
        reverse_rate = None
    else:
        raise ValueError(
            format_parser_error(
                f"Unsupported reaction arrow: {arrow}",
                source_line,
            )
        )

    reactants = parse_reaction_side(
        left_side,
        source_line=source_line,
    )

    products = parse_reaction_side(
        right_side,
        source_line=source_line,
    )

    forward_rate = f"k{reaction_index}f"

    return Reaction(
        reactants=reactants,
        products=products,
        reversible=reversible,
        forward_rate=forward_rate,
        reverse_rate=reverse_rate,
        label=label,
        source_line=source_line,
    )


def parse_model_text(text: str) -> list[Reaction]:
    """
    Parse a multiline reaction model.

    Empty lines and comments are ignored.

    Reaction indices count actual reactions only, not blank/comment lines.
    Source line numbers preserve the original text line number.
    """

    reactions: list[Reaction] = []

    lines = text.splitlines()

    for source_line, raw_line in enumerate(lines, start=1):
        stripped_line = strip_comment(raw_line)

        if not stripped_line:
            continue

        reaction_index = len(reactions) + 1

        reaction = parse_reaction_line(
            stripped_line,
            reaction_index=reaction_index,
            source_line=source_line,
        )

        reactions.append(reaction)

    return reactions
