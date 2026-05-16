import numpy as np
import pytest

from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import (
    build_bounds,
    build_initial_vector,
    validate_parameter_specs,
    vector_to_parameter_dict,
)


def test_build_initial_vector_excludes_fixed_parameters():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k1r", initial_guess=0.2, fixed=True, fixed_value=0.2),
    ]

    vector = build_initial_vector(specs)

    assert np.allclose(vector, [0.1])


def test_build_bounds_excludes_fixed_parameters():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1, lower_bound=0.0, upper_bound=1.0),
        ParameterSpec(
            "k1r",
            initial_guess=0.2,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=True,
            fixed_value=0.2,
        ),
    ]

    lower, upper = build_bounds(specs)

    assert np.allclose(lower, [0.0])
    assert np.allclose(upper, [1.0])


def test_vector_to_parameter_dict_includes_fixed_parameters():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k1r", initial_guess=0.2, fixed=True, fixed_value=0.2),
    ]

    params = vector_to_parameter_dict(
        vector=np.array([0.5]),
        parameter_specs=specs,
    )

    assert params == {
        "k1f": 0.5,
        "k1r": 0.2,
    }


def test_initial_guess_outside_bounds_raises_error():
    specs = [
        ParameterSpec("k1f", initial_guess=2.0, lower_bound=0.0, upper_bound=1.0),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)


def test_duplicate_parameter_names_raise_error():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k1f", initial_guess=0.2),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)
