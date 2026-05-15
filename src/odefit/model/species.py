from odefit.model.reaction import Reaction


def detect_species(reactions: list[Reaction]) -> list[str]:
    """
    Detect all unique species appearing in a list of reactions

    Species are collected from both reactants and products
    """

    species_set: set[str] = set()

    for reaction in reactions:
        species_set.update(reaction.reactants.keys())
        species_set.update(reaction.products.keys())

    return sorted(species_set)
