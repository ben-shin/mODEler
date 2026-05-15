from odefit.model.model_spec import ModelSpec
from odefit.model.rate_laws import generate_forward_rate, generate_reverse_rate
from odefit.model.reaction import Reaction

ODETerm = tuple[int, str]


def generate_ode_terms(
    reactions: list[Reaction],
    species: list[str],
) -> dict[str, list[ODETerm]]:
    """
    Generate ODE terms for each species.

    Each term is represented as:
        (stoichiometric_coefficient, rate_expression)

    Example:
        -2*k1f*P1^2 is stored as:
        (-2, "k1f*P1^2")
    """

    ode_terms: dict[str, list[ODETerm]] = {name: [] for name in species}

    for reaction in reactions:
        forward_rate = generate_forward_rate(reaction)

        # Forward reaction:
        # reactants are consumed
        for reactant, coefficient in reaction.reactants.items():
            ode_terms[reactant].append((-coefficient, forward_rate))

        # products are produced
        for product, coefficient in reaction.products.items():
            ode_terms[product].append((coefficient, forward_rate))

        # Reverse reaction, if reversible
        if reaction.reversible:
            reverse_rate = generate_reverse_rate(reaction)

            if reverse_rate is None:
                raise ValueError("Reversible reaction is missing a reverse rate")

            # products are consumed in the reverse direction
            for product, coefficient in reaction.products.items():
                ode_terms[product].append((-coefficient, reverse_rate))

            # reactants are produced in the reverse direction
            for reactant, coefficient in reaction.reactants.items():
                ode_terms[reactant].append((coefficient, reverse_rate))

    return ode_terms


def format_ode_term(coefficient: int, rate: str, is_first_term: bool) -> str:
    """
    Format one ODE term.

    Examples:
        (-1, "k1f*A") -> "-k1f*A"
        (1, "k1r*B")  -> "+ k1r*B"
        (-2, "k1f*A^2") -> "-2*k1f*A^2"
    """

    if coefficient == 0:
        return ""

    absolute_coefficient = abs(coefficient)

    if absolute_coefficient == 1:
        body = rate
    else:
        body = f"{absolute_coefficient}*{rate}"

    if is_first_term:
        if coefficient < 0:
            return f"-{body}"
        return body

    if coefficient < 0:
        return f" - {body}"

    return f" + {body}"


def format_ode_expression(terms: list[ODETerm]) -> str:
    """
    Format all terms for one ODE expression.
    """

    if not terms:
        return "0"

    formatted_terms = []

    for term_index, (coefficient, rate) in enumerate(terms):
        formatted = format_ode_term(
            coefficient=coefficient,
            rate=rate,
            is_first_term=(term_index == 0),
        )

        if formatted:
            formatted_terms.append(formatted)

    if not formatted_terms:
        return "0"

    return "".join(formatted_terms)


def generate_ode_expressions(model: ModelSpec) -> dict[str, str]:
    """
    Generate formatted ODE expressions for every species in a model.
    """

    ode_terms = generate_ode_terms(
        reactions=model.reactions,
        species=model.species,
    )

    return {
        species_name: format_ode_expression(terms)
        for species_name, terms in ode_terms.items()
    }


def generate_ode_lines(model: ModelSpec) -> list[str]:
    """
    Generate readable ODE lines.

    Example:
        dA/dt = -k1f*A + k1r*B
    """

    ode_expressions = generate_ode_expressions(model)

    return [
        f"d{species_name}/dt = {ode_expressions[species_name]}"
        for species_name in model.species
    ]
