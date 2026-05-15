import numpy as np

from odefit.model.model_spec import ModelSpec
from odefit.model.reaction import Reaction


def calculate_mass_action_rate(
    side: dict[str, int],
    parameter_name: str,
    species_values: dict[str, float],
    parameters: dict[str, float],
) -> float:
    """
    Calculate a numerical mass-action rate.

    Example:
        A + B -> C

        rate = k * A * B

    Example:
        2A -> B

        rate = k * A^2
    """

    if parameter_name not in parameters:
        raise ValueError(f"Missing parameter value for {parameter_name}")

    rate = parameters[parameter_name]

    for species_name, coefficient in side.items():
        concentration = species_values[species_name]
        rate *= concentration**coefficient

    return rate


def apply_forward_reaction(
    dydt: dict[str, float],
    reaction: Reaction,
    rate: float,
) -> None:
    """
    Apply the stoichiometric effect of the forward reaction.
    """

    for species_name, coefficient in reaction.reactants.items():
        dydt[species_name] -= coefficient * rate

    for species_name, coefficient in reaction.products.items():
        dydt[species_name] += coefficient * rate


def apply_reverse_reaction(
    dydt: dict[str, float],
    reaction: Reaction,
    rate: float,
) -> None:
    """
    Apply the stoichiometric effect of the reverse reaction.
    """

    for species_name, coefficient in reaction.products.items():
        dydt[species_name] -= coefficient * rate

    for species_name, coefficient in reaction.reactants.items():
        dydt[species_name] += coefficient * rate


def build_rhs_function(
    model: ModelSpec,
    parameters: dict[str, float],
):
    """
    Build a SciPy-compatible RHS function.

    SciPy expects:
        rhs(t, y) -> dydt
    """

    species = model.species

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        species_values = {
            species_name: y[index] for index, species_name in enumerate(species)
        }

        dydt = {species_name: 0.0 for species_name in species}

        for reaction in model.reactions:
            forward_rate = calculate_mass_action_rate(
                side=reaction.reactants,
                parameter_name=reaction.forward_rate,
                species_values=species_values,
                parameters=parameters,
            )

            apply_forward_reaction(
                dydt=dydt,
                reaction=reaction,
                rate=forward_rate,
            )

            if reaction.reversible:
                if reaction.reverse_rate is None:
                    raise ValueError("Reversible reaction is missing a reverse rate")

                reverse_rate = calculate_mass_action_rate(
                    side=reaction.products,
                    parameter_name=reaction.reverse_rate,
                    species_values=species_values,
                    parameters=parameters,
                )

                apply_reverse_reaction(
                    dydt=dydt,
                    reaction=reaction,
                    rate=reverse_rate,
                )

        return np.array([dydt[species_name] for species_name in species])

    return rhs
