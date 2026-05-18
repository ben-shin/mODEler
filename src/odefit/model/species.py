import re

from odefit.model.reaction import Reaction


def natural_sort_key(text: str) -> list[int | str]:
    """
    Return a natural-sort key.

    Example:
        P2 comes before P10.
    """

    parts = re.split(r"(\d+)", text)

    key: list[int | str] = []

    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)

    return key


def get_species_from_reactions(
    reactions: list[Reaction],
) -> list[str]:
    """
    Get a naturally sorted list of all species in reactions.
    """

    species = set()

    for reaction in reactions:
        species.update(reaction.reactants)
        species.update(reaction.products)

    return sorted(species, key=natural_sort_key)
