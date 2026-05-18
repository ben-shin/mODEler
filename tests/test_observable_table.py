import pytest

from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.observable_table import build_observable_table


def test_build_observable_table():
    observable_specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_initial_guess=1.0,
            scale_lower_bound=0.0,
            scale_upper_bound=10.0,
            scale_fixed=False,
            offset_initial_guess=0.0,
            offset_lower_bound=-1.0,
            offset_upper_bound=1.0,
            offset_fixed=False,
        )
    ]

    fitted_observables = {
        "amide": {
            "species": "A",
            "scale": 2.0,
            "offset": 0.1,
        }
    }

    table = build_observable_table(
        observable_specs=observable_specs,
        fitted_observables=fitted_observables,
    )

    assert list(table.columns) == [
        "data_column",
        "species",
        "fitted_species",
        "scale_initial_guess",
        "scale_fitted_value",
        "scale_lower_bound",
        "scale_upper_bound",
        "scale_fixed",
        "scale_fixed_value",
        "offset_initial_guess",
        "offset_fitted_value",
        "offset_lower_bound",
        "offset_upper_bound",
        "offset_fixed",
        "offset_fixed_value",
    ]

    assert list(table["data_column"]) == ["amide"]
    assert list(table["species"]) == ["A"]
    assert list(table["fitted_species"]) == ["A"]
    assert list(table["scale_initial_guess"]) == [1.0]
    assert list(table["scale_fitted_value"]) == [2.0]
    assert list(table["offset_initial_guess"]) == [0.0]
    assert list(table["offset_fitted_value"]) == [0.1]


def test_build_observable_table_missing_fitted_observable_raises_error():
    observable_specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
        )
    ]

    fitted_observables = {}

    with pytest.raises(ValueError):
        build_observable_table(
            observable_specs=observable_specs,
            fitted_observables=fitted_observables,
        )
