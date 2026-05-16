import numpy as np

from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_vector import (
    build_initial_condition_bounds,
    build_initial_condition_vector,
    vector_to_initial_condition_dict,
)
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import (
    build_bounds,
    build_initial_vector,
    vector_to_parameter_dict,
)


def build_optimization_vector(
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
) -> np.ndarray:
    """
    Build one optimizer vector containing:
    - free kinetic parameters
    - free initial conditions
    """

    parameter_vector = build_initial_vector(parameter_specs)
    initial_condition_vector = build_initial_condition_vector(initial_condition_specs)

    return np.concatenate([parameter_vector, initial_condition_vector])


def build_optimization_bounds(
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lower and upper bounds for the combined optimizer vector.
    """

    parameter_lower, parameter_upper = build_bounds(parameter_specs)
    initial_condition_lower, initial_condition_upper = build_initial_condition_bounds(
        initial_condition_specs
    )

    lower_bounds = np.concatenate([parameter_lower, initial_condition_lower])
    upper_bounds = np.concatenate([parameter_upper, initial_condition_upper])

    return lower_bounds, upper_bounds


def split_optimization_vector(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split the combined optimizer vector into:
    - kinetic parameter vector
    - initial-condition vector
    """

    parameter_vector_length = len(build_initial_vector(parameter_specs))

    parameter_vector = vector[:parameter_vector_length]
    initial_condition_vector = vector[parameter_vector_length:]

    expected_length = parameter_vector_length + len(
        build_initial_condition_vector(initial_condition_specs)
    )

    if len(vector) != expected_length:
        raise ValueError("Optimization vector has unexpected length")

    return parameter_vector, initial_condition_vector


def vector_to_model_inputs(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Convert combined optimizer vector into:
    - parameter dictionary
    - initial-condition dictionary
    """

    parameter_vector, initial_condition_vector = split_optimization_vector(
        vector=vector,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
    )

    parameters = vector_to_parameter_dict(
        vector=parameter_vector,
        parameter_specs=parameter_specs,
    )

    initial_conditions = vector_to_initial_condition_dict(
        vector=initial_condition_vector,
        initial_condition_specs=initial_condition_specs,
    )

    return parameters, initial_conditions
