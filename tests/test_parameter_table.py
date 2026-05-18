import pandas as pd
import pytest

from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.parameter_table import (
    build_initial_parameter_dict,
    build_parameter_table,
)


def test_build_parameter_table():
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec(
            name="k1r",
            initial_guess=0.2,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.2,
        ),
        ParameterSpec(
            name="k2f",
            initial_guess=0.3,
            lower_bound=0.0,
            upper_bound=10.0,
            tied_to="k1f",
        ),
    ]

    fitted_parameters = {
        "k1f": 0.5,
        "k1r": 0.2,
        "k2f": 0.5,
    }

    table = build_parameter_table(
        parameter_specs=parameter_specs,
        fitted_parameters=fitted_parameters,
    )

    assert list(table.columns) == [
        "parameter",
        "initial_guess",
        "fitted_value",
        "lower_bound",
        "upper_bound",
        "fixed",
        "fixed_value",
        "tied_to",
    ]

    assert list(table["parameter"]) == ["k1f", "k1r", "k2f"]
    assert list(table["initial_guess"]) == [0.1, 0.2, 0.3]
    assert list(table["fitted_value"]) == [0.5, 0.2, 0.5]
    assert list(table["fixed"]) == [False, True, False]
    assert table["tied_to"].iloc[0] is None or pd.isna(table["tied_to"].iloc[0])
    assert table["tied_to"].iloc[1] is None or pd.isna(table["tied_to"].iloc[1])
    assert table["tied_to"].iloc[2] == "k1f"


def test_build_parameter_table_missing_fitted_value_raises_error():
    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    fitted_parameters = {}

    with pytest.raises(ValueError):
        build_parameter_table(
            parameter_specs=parameter_specs,
            fitted_parameters=fitted_parameters,
        )


def test_build_initial_parameter_dict():
    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
        ParameterSpec(
            name="k1r",
            initial_guess=0.2,
            fixed=True,
            fixed_value=0.05,
        ),
        ParameterSpec(
            name="k2f",
            initial_guess=0.3,
            tied_to="k1f",
        ),
    ]

    initial_parameters = build_initial_parameter_dict(parameter_specs)

    assert initial_parameters == {
        "k1f": 0.1,
        "k1r": 0.05,
        "k2f": 0.1,
    }


def test_build_initial_parameter_dict_fixed_without_value_raises_error():
    parameter_specs = [
        ParameterSpec(
            name="k1r",
            initial_guess=0.2,
            fixed=True,
            fixed_value=None,
        ),
    ]

    with pytest.raises(ValueError):
        build_initial_parameter_dict(parameter_specs)
