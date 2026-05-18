import numpy as np

from odefit.model.model_spec import ModelSpec
from odefit.model.reaction import Reaction


def calculate_mass_action_rate(
    reaction_side: dict[str, int],
    species_values: dict[str, float],
    rate_constant: float,
) -> float:
    """
    Calculate a mass-action reaction rate.

    Example:
        A + B -> C

        rate = k * A * B

    Example:
        2A -> A2

        rate = k * A^2
    """

    rate = rate_constant

    for species_name, coefficient in reaction_side.items():
        species_value = species_values[species_name]
        rate *= species_value**coefficient

    return rate


def apply_reaction_flux(
    derivatives: dict[str, float],
    reaction_side: dict[str, int],
    flux: float,
    sign: float,
) -> None:
    """
    Apply a reaction flux to derivative values.

    sign = -1 for consumption
    sign = +1 for formation
    """

    for species_name, coefficient in reaction_side.items():
        derivatives[species_name] += sign * coefficient * flux


def build_species_value_dict(
    species: list[str],
    values: np.ndarray,
    clip_negative_concentrations: bool = False,
) -> dict[str, float]:
    """
    Build dictionary mapping species names to current values.

    If clip_negative_concentrations is True, negative values are replaced
    by zero before evaluating mass-action rates.
    """

    species_values: dict[str, float] = {}

    for species_name, value in zip(species, values):
        value = float(value)

        if clip_negative_concentrations and value < 0.0:
            value = 0.0

        species_values[species_name] = value

    return species_values


def validate_parameter_values(
    model: ModelSpec,
    parameters: dict[str, float],
) -> None:
    """
    Validate that all model parameters have values.
    """

    for parameter_name in model.parameters:
        if parameter_name not in parameters:
            raise ValueError(f"Missing parameter value: {parameter_name}")


def validate_initial_conditions(
    model: ModelSpec,
    initial_conditions: dict[str, float],
) -> None:
    """
    Validate that all model species have initial conditions.
    """

    for species_name in model.species:
        if species_name not in initial_conditions:
            raise ValueError(f"Missing initial condition for species: {species_name}")


def calculate_reaction_fluxes(
    reaction: Reaction,
    species_values: dict[str, float],
    parameters: dict[str, float],
) -> tuple[float, float | None]:
    """
    Calculate forward and reverse fluxes for one reaction.
    """

    if reaction.forward_rate not in parameters:
        raise ValueError(f"Missing parameter value: {reaction.forward_rate}")

    forward_flux = calculate_mass_action_rate(
        reaction_side=reaction.reactants,
        species_values=species_values,
        rate_constant=parameters[reaction.forward_rate],
    )

    reverse_flux = None

    if reaction.reversible:
        if reaction.reverse_rate is None:
            raise ValueError("Reversible reaction is missing reverse rate")

        if reaction.reverse_rate not in parameters:
            raise ValueError(f"Missing parameter value: {reaction.reverse_rate}")

        reverse_flux = calculate_mass_action_rate(
            reaction_side=reaction.products,
            species_values=species_values,
            rate_constant=parameters[reaction.reverse_rate],
        )

    return forward_flux, reverse_flux


def build_rhs_function(
    model: ModelSpec,
    parameters: dict[str, float],
    clip_negative_concentrations: bool = False,
):
    """
    Build a scipy-compatible RHS function.

    The returned function has signature:

        rhs(t, y)
    """

    validate_parameter_values(
        model=model,
        parameters=parameters,
    )

    def rhs(_time, values):
        species_values = build_species_value_dict(
            species=model.species,
            values=values,
            clip_negative_concentrations=clip_negative_concentrations,
        )

        derivatives = {species_name: 0.0 for species_name in model.species}

        for reaction in model.reactions:
            forward_flux, reverse_flux = calculate_reaction_fluxes(
                reaction=reaction,
                species_values=species_values,
                parameters=parameters,
            )

            apply_reaction_flux(
                derivatives=derivatives,
                reaction_side=reaction.reactants,
                flux=forward_flux,
                sign=-1.0,
            )

            apply_reaction_flux(
                derivatives=derivatives,
                reaction_side=reaction.products,
                flux=forward_flux,
                sign=1.0,
            )

            if reverse_flux is not None:
                apply_reaction_flux(
                    derivatives=derivatives,
                    reaction_side=reaction.products,
                    flux=reverse_flux,
                    sign=-1.0,
                )

                apply_reaction_flux(
                    derivatives=derivatives,
                    reaction_side=reaction.reactants,
                    flux=reverse_flux,
                    sign=1.0,
                )

        return np.array(
            [derivatives[species_name] for species_name in model.species],
            dtype=float,
        )

    return rhs
