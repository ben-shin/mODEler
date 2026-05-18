import numpy as np
import pytest

from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_vector import (
    build_initial_condition_bounds,
    build_initial_condition_dict,
    build_initial_condition_vector,
    get_free_initial_condition_specs,
    make_fixed_initial_condition_specs,
    validate_initial_condition_specs,
    vector_to_initial_condition_dict,
)


def test_make_fixed_initial_condition_specs():
    specs = make_fixed_initial_condition_specs(
        {
            "A": 1.0,
            "B": 0.0,
        }
    )

    assert len(specs) == 2
    assert specs[0].species == "A"
    assert specs[0].fixed is True
    assert specs[0].fixed_value == 1.0


def test_get_free_initial_condition_specs():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.1, fixed=False),
    ]

    free_specs = get_free_initial_condition_specs(specs)

    assert [spec.species for spec in free_specs] == ["B"]


def test_build_initial_condition_vector():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.1, fixed=False),
    ]

    vector = build_initial_condition_vector(specs)

    assert np.allclose(vector, [0.1])


def test_build_initial_condition_bounds():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec(
            "B",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=2.0,
            fixed=False,
        ),
    ]

    lower, upper = build_initial_condition_bounds(specs)

    assert np.allclose(lower, [0.0])
    assert np.allclose(upper, [2.0])


def test_vector_to_initial_condition_dict():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.1, fixed=False),
    ]

    initial_conditions = vector_to_initial_condition_dict(
        vector=np.array([0.5]),
        initial_condition_specs=specs,
    )

    assert initial_conditions == {
        "A": 1.0,
        "B": 0.5,
    }


def test_vector_to_initial_condition_dict_uses_fixed_value():
    specs = [
        InitialConditionSpec(
            "A",
            initial_guess=0.5,
            fixed=True,
            fixed_value=1.0,
        ),
    ]

    initial_conditions = vector_to_initial_condition_dict(
        vector=np.array([]),
        initial_condition_specs=specs,
    )

    assert initial_conditions == {"A": 1.0}


def test_build_initial_condition_dict():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.2, fixed=False),
    ]

    initial_conditions = build_initial_condition_dict(specs)

    assert initial_conditions == {
        "A": 1.0,
        "B": 0.2,
    }


def test_short_initial_condition_vector_raises_error():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=False),
        InitialConditionSpec("B", initial_guess=0.0, fixed=False),
    ]

    with pytest.raises(ValueError):
        vector_to_initial_condition_dict(
            vector=np.array([1.0]),
            initial_condition_specs=specs,
        )


def test_long_initial_condition_vector_raises_error():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=False),
    ]

    with pytest.raises(ValueError):
        vector_to_initial_condition_dict(
            vector=np.array([1.0, 2.0]),
            initial_condition_specs=specs,
        )


def test_duplicate_initial_condition_species_raises_error():
    specs = [
        InitialConditionSpec("A", initial_guess=1.0),
        InitialConditionSpec("A", initial_guess=0.5),
    ]

    with pytest.raises(ValueError):
        validate_initial_condition_specs(specs)


def test_initial_condition_guess_outside_bounds_raises_error():
    specs = [
        InitialConditionSpec(
            "A",
            initial_guess=2.0,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=False,
        )
    ]

    with pytest.raises(ValueError):
        validate_initial_condition_specs(specs)


def test_fixed_initial_condition_outside_bounds_raises_error():
    specs = [
        InitialConditionSpec(
            "A",
            initial_guess=0.5,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=True,
            fixed_value=2.0,
        )
    ]

    with pytest.raises(ValueError):
        validate_initial_condition_specs(specs)
