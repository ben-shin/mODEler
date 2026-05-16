import numpy as np
import pytest

from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_vector import (
    build_bounds,
    build_initial_vector,
    get_free_parameter_specs,
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


def test_build_initial_vector_excludes_tied_parameters():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k2f", initial_guess=0.2, tied_to="k1f"),
    ]

    vector = build_initial_vector(specs)

    assert np.allclose(vector, [0.1])


def test_get_free_parameter_specs_excludes_fixed_and_tied_parameters():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k1r", initial_guess=0.2, fixed=True, fixed_value=0.2),
        ParameterSpec("k2f", initial_guess=0.3, tied_to="k1f"),
    ]

    free_specs = get_free_parameter_specs(specs)

    assert [parameter.name for parameter in free_specs] == ["k1f"]


def test_build_bounds_excludes_fixed_and_tied_parameters():
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
        ParameterSpec(
            "k2f",
            initial_guess=0.3,
            lower_bound=0.0,
            upper_bound=1.0,
            tied_to="k1f",
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


def test_vector_to_parameter_dict_resolves_tied_parameters():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k2f", initial_guess=0.2, tied_to="k1f"),
    ]

    params = vector_to_parameter_dict(
        vector=np.array([0.5]),
        parameter_specs=specs,
    )

    assert params == {
        "k1f": 0.5,
        "k2f": 0.5,
    }


def test_vector_to_parameter_dict_resolves_tied_to_fixed_parameter():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1, fixed=True, fixed_value=0.25),
        ParameterSpec("k2f", initial_guess=0.2, tied_to="k1f"),
    ]

    params = vector_to_parameter_dict(
        vector=np.array([]),
        parameter_specs=specs,
    )

    assert params == {
        "k1f": 0.25,
        "k2f": 0.25,
    }


def test_vector_to_parameter_dict_short_vector_raises_error():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k2f", initial_guess=0.2),
    ]

    with pytest.raises(ValueError):
        vector_to_parameter_dict(
            vector=np.array([0.5]),
            parameter_specs=specs,
        )


def test_vector_to_parameter_dict_long_vector_raises_error():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
    ]

    with pytest.raises(ValueError):
        vector_to_parameter_dict(
            vector=np.array([0.5, 0.6]),
            parameter_specs=specs,
        )


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


def test_fixed_parameter_without_fixed_value_raises_error():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1, fixed=True),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)


def test_fixed_parameter_outside_bounds_raises_error():
    specs = [
        ParameterSpec(
            "k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=True,
            fixed_value=2.0,
        ),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)


def test_parameter_cannot_be_fixed_and_tied():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec(
            "k2f",
            initial_guess=0.2,
            fixed=True,
            fixed_value=0.2,
            tied_to="k1f",
        ),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)


def test_parameter_cannot_be_tied_to_itself():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1, tied_to="k1f"),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)


def test_parameter_tied_to_unknown_parameter_raises_error():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1),
        ParameterSpec("k2f", initial_guess=0.2, tied_to="missing"),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)


def test_circular_parameter_tie_raises_error():
    specs = [
        ParameterSpec("k1f", initial_guess=0.1, tied_to="k2f"),
        ParameterSpec("k2f", initial_guess=0.2, tied_to="k1f"),
    ]

    with pytest.raises(ValueError):
        validate_parameter_specs(specs)
