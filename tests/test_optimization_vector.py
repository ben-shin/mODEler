import numpy as np
import pytest

from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimization_vector import (
    build_optimization_bounds,
    build_optimization_vector,
    split_optimization_vector,
    vector_to_model_inputs,
)
from odefit.fitting.parameter_spec import ParameterSpec


def test_build_optimization_vector_combines_parameters_and_initial_conditions():
    parameter_specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k1r", initial_guess=0.2, fixed=True, fixed_value=0.2),
        ParameterSpec("k2f", initial_guess=0.3, tied_to="k1f"),
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.4, fixed=False),
    ]

    vector = build_optimization_vector(
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
    )

    assert np.allclose(vector, [0.1, 0.4])


def test_build_optimization_bounds_combines_parameters_and_initial_conditions():
    parameter_specs = [
        ParameterSpec(
            "k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec("k2f", initial_guess=0.3, tied_to="k1f"),
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec(
            "B",
            initial_guess=0.4,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=False,
        ),
    ]

    lower, upper = build_optimization_bounds(
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
    )

    assert np.allclose(lower, [0.0, 0.0])
    assert np.allclose(upper, [10.0, 2.0])


def test_split_optimization_vector():
    parameter_specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k2f", initial_guess=0.2),
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.4, fixed=False),
    ]

    parameter_vector, initial_condition_vector = split_optimization_vector(
        vector=np.array([0.5, 0.6, 1.2]),
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
    )

    assert np.allclose(parameter_vector, [0.5, 0.6])
    assert np.allclose(initial_condition_vector, [1.2])


def test_split_optimization_vector_wrong_length_raises_error():
    parameter_specs = [
        ParameterSpec("k1f", initial_guess=0.1),
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=False),
    ]

    with pytest.raises(ValueError):
        split_optimization_vector(
            vector=np.array([0.5, 1.0, 2.0]),
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
        )


def test_vector_to_model_inputs_with_tied_parameter_and_fitted_initial_condition():
    parameter_specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k2f", initial_guess=0.2, tied_to="k1f"),
        ParameterSpec("k3f", initial_guess=0.3, fixed=True, fixed_value=0.7),
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=False),
        InitialConditionSpec("B", initial_guess=0.0, fixed=True, fixed_value=0.0),
    ]

    parameters, initial_conditions = vector_to_model_inputs(
        vector=np.array([0.5, 2.0]),
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
    )

    assert parameters == {
        "k1f": 0.5,
        "k2f": 0.5,
        "k3f": 0.7,
    }

    assert initial_conditions == {
        "A": 2.0,
        "B": 0.0,
    }
