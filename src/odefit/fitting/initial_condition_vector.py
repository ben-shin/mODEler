import numpy as np

from odefit.fitting.initial_condition_spec import InitialConditionSpec


def make_fixed_initial_condition_specs(
    initial_conditions: dict[str, float],
) -> list[InitialConditionSpec]:
    """
    Convert a simple initial condition dictionary into fixed InitialConditionSpec objects.

    This keeps the old fitting API backward-compatible.
    """

    return [
        InitialConditionSpec(
            species=species,
            initial_guess=float(value),
            fixed=True,
            fixed_value=float(value),
        )
        for species, value in initial_conditions.items()
    ]


def get_fixed_initial_condition_value(
    initial_condition: InitialConditionSpec,
) -> float:
    """
    Return the fixed value for an initial condition.

    If fixed_value is None, initial_guess is used.
    """

    if initial_condition.fixed_value is not None:
        return initial_condition.fixed_value

    return initial_condition.initial_guess


def get_free_initial_condition_specs(
    initial_condition_specs: list[InitialConditionSpec],
) -> list[InitialConditionSpec]:
    """
    Return initial conditions that should be optimized.
    """

    return [
        initial_condition
        for initial_condition in initial_condition_specs
        if not initial_condition.fixed
    ]


def build_initial_condition_vector(
    initial_condition_specs: list[InitialConditionSpec],
) -> np.ndarray:
    """
    Build optimizer vector from free initial-condition guesses.
    """

    free_initial_conditions = get_free_initial_condition_specs(initial_condition_specs)

    return np.array(
        [
            initial_condition.initial_guess
            for initial_condition in free_initial_conditions
        ],
        dtype=float,
    )


def build_initial_condition_bounds(
    initial_condition_specs: list[InitialConditionSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lower and upper bounds for free initial conditions.
    """

    free_initial_conditions = get_free_initial_condition_specs(initial_condition_specs)

    lower_bounds = np.array(
        [
            initial_condition.lower_bound
            for initial_condition in free_initial_conditions
        ],
        dtype=float,
    )

    upper_bounds = np.array(
        [
            initial_condition.upper_bound
            for initial_condition in free_initial_conditions
        ],
        dtype=float,
    )

    return lower_bounds, upper_bounds


def vector_to_initial_condition_dict(
    vector: np.ndarray,
    initial_condition_specs: list[InitialConditionSpec],
) -> dict[str, float]:
    """
    Convert optimizer vector into a full initial-condition dictionary.

    Includes both fitted and fixed initial conditions.
    """

    initial_conditions: dict[str, float] = {}
    vector_index = 0

    for initial_condition in initial_condition_specs:
        if initial_condition.fixed:
            initial_conditions[initial_condition.species] = (
                get_fixed_initial_condition_value(initial_condition)
            )

        else:
            if vector_index >= len(vector):
                raise ValueError("Initial-condition vector is shorter than expected")

            initial_conditions[initial_condition.species] = float(vector[vector_index])
            vector_index += 1

    if vector_index != len(vector):
        raise ValueError("Initial-condition vector is longer than expected")

    return initial_conditions


def build_initial_condition_dict(
    initial_condition_specs: list[InitialConditionSpec],
) -> dict[str, float]:
    """
    Build an initial-condition dictionary from guesses/fixed values.
    """

    initial_condition_vector = build_initial_condition_vector(initial_condition_specs)

    return vector_to_initial_condition_dict(
        vector=initial_condition_vector,
        initial_condition_specs=initial_condition_specs,
    )


def validate_initial_condition_specs(
    initial_condition_specs: list[InitialConditionSpec],
) -> None:
    """
    Validate initial-condition specs before fitting.
    """

    seen_species = set()

    for initial_condition in initial_condition_specs:
        if initial_condition.species in seen_species:
            raise ValueError(
                f"Duplicate initial condition for species: {initial_condition.species}"
            )

        seen_species.add(initial_condition.species)

        if initial_condition.lower_bound > initial_condition.upper_bound:
            raise ValueError(
                "Lower bound exceeds upper bound for initial condition: "
                f"{initial_condition.species}"
            )

        if not (
            initial_condition.lower_bound
            <= initial_condition.initial_guess
            <= initial_condition.upper_bound
        ):
            raise ValueError(
                "Initial-condition guess outside bounds for species: "
                f"{initial_condition.species}"
            )

        if initial_condition.fixed:
            fixed_value = get_fixed_initial_condition_value(initial_condition)

            if not (
                initial_condition.lower_bound
                <= fixed_value
                <= initial_condition.upper_bound
            ):
                raise ValueError(
                    "Fixed initial condition outside bounds for species: "
                    f"{initial_condition.species}"
                )
