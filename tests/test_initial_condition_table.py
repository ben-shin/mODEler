import pytest

from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.initial_condition_table import (
    build_initial_condition_guess_dict,
    build_initial_condition_table,
)


def test_build_initial_condition_table():
    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=5.0,
            fixed=False,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=5.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]

    fitted_initial_conditions = {
        "A": 2.0,
        "B": 0.0,
    }

    table = build_initial_condition_table(
        initial_condition_specs=initial_condition_specs,
        fitted_initial_conditions=fitted_initial_conditions,
    )

    assert list(table.columns) == [
        "species",
        "initial_guess",
        "fitted_value",
        "lower_bound",
        "upper_bound",
        "fixed",
        "fixed_value",
    ]

    assert list(table["species"]) == ["A", "B"]
    assert list(table["initial_guess"]) == [1.0, 0.0]
    assert list(table["fitted_value"]) == [2.0, 0.0]
    assert list(table["lower_bound"]) == [0.0, 0.0]
    assert list(table["upper_bound"]) == [5.0, 5.0]
    assert list(table["fixed"]) == [False, True]


def test_build_initial_condition_table_missing_fitted_value_raises_error():
    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            fixed=False,
        ),
    ]

    fitted_initial_conditions = {}

    with pytest.raises(ValueError):
        build_initial_condition_table(
            initial_condition_specs=initial_condition_specs,
            fitted_initial_conditions=fitted_initial_conditions,
        )


def test_build_initial_condition_guess_dict():
    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            fixed=False,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.5,
            fixed=True,
            fixed_value=0.0,
        ),
        InitialConditionSpec(
            species="C",
            initial_guess=0.25,
            fixed=True,
            fixed_value=None,
        ),
    ]

    initial_conditions = build_initial_condition_guess_dict(initial_condition_specs)

    assert initial_conditions == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.25,
    }
