import numpy as np

from odefit.fitting.parameter_spec import ParameterSpec


def get_free_parameter_specs(
    parameter_specs: list[ParameterSpec],
) -> list[ParameterSpec]:
    """
    Return parameters that should be optimized.

    Fixed parameters and tied parameters are not independent free parameters.
    """

    return [
        parameter
        for parameter in parameter_specs
        if not parameter.fixed and parameter.tied_to is None
    ]


def build_initial_vector(
    parameter_specs: list[ParameterSpec],
) -> np.ndarray:
    """
    Build optimizer vector from independent non-fixed parameter guesses.
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
    Build lower and upper bounds for independent non-fixed parameters.
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

    Includes:
    - fitted free parameters
    - fixed parameters
    - tied parameters
    """

    spec_by_name = {parameter.name: parameter for parameter in parameter_specs}

    parameter_values: dict[str, float] = {}
    vector_index = 0

    # First assign fixed and independent free parameters.
    for parameter in parameter_specs:
        if parameter.fixed:
            if parameter.fixed_value is None:
                raise ValueError(
                    f"Fixed parameter missing fixed value: {parameter.name}"
                )

            parameter_values[parameter.name] = parameter.fixed_value

        elif parameter.tied_to is None:
            if vector_index >= len(vector):
                raise ValueError("Parameter vector is shorter than expected")

            parameter_values[parameter.name] = float(vector[vector_index])
            vector_index += 1

    if vector_index != len(vector):
        raise ValueError("Parameter vector is longer than expected")

    resolving: set[str] = set()

    def resolve_parameter_value(parameter_name: str) -> float:
        """
        Resolve tied parameters recursively.
        """

        if parameter_name in parameter_values:
            return parameter_values[parameter_name]

        if parameter_name not in spec_by_name:
            raise ValueError(f"Unknown tied parameter target: {parameter_name}")

        if parameter_name in resolving:
            raise ValueError(f"Circular parameter tie involving: {parameter_name}")

        resolving.add(parameter_name)

        parameter = spec_by_name[parameter_name]

        if parameter.tied_to is None:
            raise ValueError(
                f"Parameter has no value and is not tied: {parameter_name}"
            )

        value = resolve_parameter_value(parameter.tied_to)

        resolving.remove(parameter_name)

        parameter_values[parameter_name] = value

        return value

    for parameter in parameter_specs:
        resolve_parameter_value(parameter.name)

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

        if parameter.fixed and parameter.tied_to is not None:
            raise ValueError(
                f"Parameter cannot be both fixed and tied: {parameter.name}"
            )

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

        elif parameter.tied_to is None:
            if (
                not parameter.lower_bound
                <= parameter.initial_guess
                <= parameter.upper_bound
            ):
                raise ValueError(f"Initial guess outside bounds for {parameter.name}")

    spec_by_name = {parameter.name: parameter for parameter in parameter_specs}

    for parameter in parameter_specs:
        if parameter.tied_to is None:
            continue

        if parameter.tied_to == parameter.name:
            raise ValueError(f"Parameter cannot be tied to itself: {parameter.name}")

        if parameter.tied_to not in spec_by_name:
            raise ValueError(
                f"Parameter {parameter.name} is tied to unknown parameter: "
                f"{parameter.tied_to}"
            )

    def visit(parameter_name: str, stack: set[str]) -> None:
        parameter = spec_by_name[parameter_name]

        if parameter.tied_to is None:
            return

        if parameter_name in stack:
            raise ValueError(f"Circular parameter tie involving: {parameter_name}")

        visit(parameter.tied_to, stack | {parameter_name})

    for parameter in parameter_specs:
        visit(parameter.name, set())
