import pandas as pd

from odefit.fitting.initial_condition_spec import InitialConditionSpec


def build_initial_condition_table(
    initial_condition_specs: list[InitialConditionSpec],
    fitted_initial_conditions: dict[str, float],
) -> pd.DataFrame:
    """
    Build a tidy initial-condition table for GUI display and CSV export.

    The table includes:
    - species name
    - initial guess
    - fitted/final value
    - bounds
    - fixed state
    - fixed value
    """

    rows = []

    for initial_condition in initial_condition_specs:
        species = initial_condition.species

        if species not in fitted_initial_conditions:
            raise ValueError(f"Missing fitted initial condition for species: {species}")

        rows.append(
            {
                "species": species,
                "initial_guess": initial_condition.initial_guess,
                "fitted_value": fitted_initial_conditions[species],
                "lower_bound": initial_condition.lower_bound,
                "upper_bound": initial_condition.upper_bound,
                "fixed": initial_condition.fixed,
                "fixed_value": initial_condition.fixed_value,
            }
        )

    return pd.DataFrame(rows)


def build_initial_condition_guess_dict(
    initial_condition_specs: list[InitialConditionSpec],
) -> dict[str, float]:
    """
    Build a dictionary of initial-condition values before fitting.

    Fixed initial conditions use fixed_value if provided.
    Otherwise, initial_guess is used.
    """

    initial_conditions: dict[str, float] = {}

    for initial_condition in initial_condition_specs:
        if initial_condition.fixed and initial_condition.fixed_value is not None:
            initial_conditions[initial_condition.species] = (
                initial_condition.fixed_value
            )
        else:
            initial_conditions[initial_condition.species] = (
                initial_condition.initial_guess
            )

    return initial_conditions
