import numpy as np

from odefit.fitting.parameter_spec import ParameterSpec


def get_free_parameter_specs(
    parameter_specs: list[ParameterSpec],
) -> list[ParameterSpec]:
    """
    Return parameters that should be optimized.
    """

    return [parameter for parameter in parameter_specs if not parameter.fixed]


def build_initial_vector(
    parameter_specs: list[ParameterSpec],
) -> np.ndarray:
    """
    Build optimizer vector from non-fixed parameter guesses.
    """

    free_parameters = get_free_parameter_specs(parameter_specs)

    return np.array(
        [parameter.initial_guess for parameter in free_parameters],
        dtype=float,
    )


def build_bounds(
    parameter_specs: list[ParameterSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lower and upper bounds for non-fixed parameters.
    """

    free_parameters = get_free_parameter_specs(parameter_specs)

    lower_bounds = np.array(
        [parameter.lower_bound for parameter in free_parameters],
        dtype=float,
    )

    upper_bounds = np.array(
        [parameter.upper_bound for parameter in free_parameters],
        dtype=float,
    )

    return lower_bounds, upper_bounds


def vector_to_parameter_dict(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
) -> dict[str, float]:
    """
    Convert optimizer vector back into a full parameter dictionary.

    Includes both fitted and fixed parameters.
    """

    parameter_values: dict[str, float] = {}
    vector_index = 0

    for parameter in parameter_specs:
        if parameter.fixed:
            if parameter.fixed_value is None:
                raise ValueError(
                    f"Fixed parameter missing fixed value: {parameter.name}"
                )

            parameter_values[parameter.name] = parameter.fixed_value

        else:
            if vector_index >= len(vector):
                raise ValueError("Parameter vector is shorter than expected")

            parameter_values[parameter.name] = float(vector[vector_index])
            vector_index += 1

    if vector_index != len(vector):
        raise ValueError("Parameter vector is longer than expected")

    return parameter_values


def validate_parameter_specs(
    parameter_specs: list[ParameterSpec],
) -> None:
    """
    Validate parameter specs before fitting.
    """

    seen_names = set()

    for parameter in parameter_specs:
        if parameter.name in seen_names:
            raise ValueError(f"Duplicate parameter name: {parameter.name}")

        seen_names.add(parameter.name)

        if parameter.lower_bound > parameter.upper_bound:
            raise ValueError(f"Lower bound exceeds upper bound for {parameter.name}")

        if parameter.fixed:
            if parameter.fixed_value is None:
                raise ValueError(
                    f"Fixed parameter missing fixed value: {parameter.name}"
                )

            if (
                not parameter.lower_bound
                <= parameter.fixed_value
                <= parameter.upper_bound
            ):
                raise ValueError(f"Fixed value outside bounds for {parameter.name}")

        else:
            if (
                not parameter.lower_bound
                <= parameter.initial_guess
                <= parameter.upper_bound
            ):
                raise ValueError(f"Initial guess outside bounds for {parameter.name}")
