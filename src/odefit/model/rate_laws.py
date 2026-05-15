from multiprocessing import Value

from odefit.model.reaction import Reaction


def generate_mass_action_rate(side: dict[str, int], parameter: str) -> str:
    """
    Generate a mass-action rate expression.

    Examples:
        {"A": 1}, "k1f"     -> "k1f*A"
        {"A": 2}, "k1f"     -> "k1f*A^2"
        {"A": 1, "B": 1}, "k2f" -> "k2f*A*B"
    """

    factors = [parameter]

    for species, coefficient in side.items():
        if coefficient == 1:
            factors.append(species)
        else:
            factors.append(f"{species}^{coefficient}")

    return "*".join(factors)


def generate_forward_rate(reaction: Reaction) -> str:
    """
    Generate the forward mass action rate for a reaction
    """

    return generate_mass_action_rate(
        side=reaction.reactants,
        parameter=reaction.forward_rate,
    )


def generate_reverse_rate(reaction: Reaction) -> str | None:
    """
    Generate reverse mass action rate for reversible reaction
    Irreversible reactions do not have a reversible rate
    """

    if not reaction.reversible:
        return None

    if reaction.reverse_rate is None:
        raise ValueError("Reversible reaction is missing a reverse rate")

    return generate_mass_action_rate(
        side=reaction.products,
        parameter=reaction.reverse_rate,
    )


def generate_reaction_rates(reaction: Reaction) -> tuple[str, str | None]:
    """
    Generate forward and reverse rate expressions for one reaction
    """

    forward_rate = generate_forward_rate(reaction)
    reverse_rate = generate_reverse_rate(reaction)

    return forward_rate, reverse_rate
