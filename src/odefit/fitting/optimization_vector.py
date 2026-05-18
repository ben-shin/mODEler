import numpy as np

from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_vector import (
    build_initial_condition_bounds,
    build_initial_condition_vector,
    vector_to_initial_condition_dict,
)
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_vector import (
    build_observable_bounds,
    build_observable_vector,
    vector_to_observable_parameters,
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
    observable_specs: list[ObservableSpec] | None = None,
) -> np.ndarray:
    """
    Build one optimizer vector containing:
    - free kinetic parameters
    - free initial conditions
    - free observable parameters, optional
    """

    if observable_specs is None:
        observable_specs = []

    parameter_vector = build_initial_vector(parameter_specs)
    initial_condition_vector = build_initial_condition_vector(initial_condition_specs)
    observable_vector = build_observable_vector(observable_specs)

    return np.concatenate(
        [parameter_vector, initial_condition_vector, observable_vector]
    )


def build_optimization_bounds(
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observable_specs: list[ObservableSpec] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build lower and upper bounds for the combined optimizer vector.
    """

    if observable_specs is None:
        observable_specs = []

    parameter_lower, parameter_upper = build_bounds(parameter_specs)

    initial_condition_lower, initial_condition_upper = build_initial_condition_bounds(
        initial_condition_specs
    )

    observable_lower, observable_upper = build_observable_bounds(observable_specs)

    lower_bounds = np.concatenate(
        [parameter_lower, initial_condition_lower, observable_lower]
    )

    upper_bounds = np.concatenate(
        [parameter_upper, initial_condition_upper, observable_upper]
    )

    return lower_bounds, upper_bounds


def split_optimization_vector(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible split.

    Splits the combined optimizer vector into:
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


def split_full_optimization_vector(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observable_specs: list[ObservableSpec] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the combined optimizer vector into:
    - kinetic parameter vector
    - initial-condition vector
    - observable parameter vector
    """

    if observable_specs is None:
        observable_specs = []

    parameter_vector_length = len(build_initial_vector(parameter_specs))
    initial_condition_vector_length = len(
        build_initial_condition_vector(initial_condition_specs)
    )
    observable_vector_length = len(build_observable_vector(observable_specs))

    expected_length = (
        parameter_vector_length
        + initial_condition_vector_length
        + observable_vector_length
    )

    if len(vector) != expected_length:
        raise ValueError("Optimization vector has unexpected length")

    parameter_end = parameter_vector_length
    initial_condition_end = parameter_end + initial_condition_vector_length

    parameter_vector = vector[:parameter_end]
    initial_condition_vector = vector[parameter_end:initial_condition_end]
    observable_vector = vector[initial_condition_end:]

    return parameter_vector, initial_condition_vector, observable_vector


def vector_to_model_inputs(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Convert combined optimizer vector into:
    - parameter dictionary
    - initial-condition dictionary

    This preserves the old no-observable API.
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


def vector_to_fit_inputs(
    vector: np.ndarray,
    parameter_specs: list[ParameterSpec],
    initial_condition_specs: list[InitialConditionSpec],
    observable_specs: list[ObservableSpec] | None = None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, float | str]],
]:
    """
    Convert full optimizer vector into:
    - parameter dictionary
    - initial-condition dictionary
    - observable parameter dictionary
    """

    if observable_specs is None:
        observable_specs = []

    (
        parameter_vector,
        initial_condition_vector,
        observable_vector,
    ) = split_full_optimization_vector(
        vector=vector,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
    )

    parameters = vector_to_parameter_dict(
        vector=parameter_vector,
        parameter_specs=parameter_specs,
    )

    initial_conditions = vector_to_initial_condition_dict(
        vector=initial_condition_vector,
        initial_condition_specs=initial_condition_specs,
    )

    observable_parameters = vector_to_observable_parameters(
        vector=observable_vector,
        observable_specs=observable_specs,
    )

    return parameters, initial_conditions, observable_parameters
