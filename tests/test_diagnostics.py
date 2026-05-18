import numpy as np
import pandas as pd
import pytest

from odefit.fitting.diagnostics import (
    build_fit_diagnostics_table,
    build_initial_condition_diagnostics_table,
    build_observable_diagnostics_table,
    build_parameter_diagnostics_table,
    get_diagnostic_warnings,
    is_at_lower_bound,
    is_at_upper_bound,
)
from odefit.fitting.fit_result import FitResult
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.simulation.simulation_result import SimulationResult


def make_simulation_result() -> SimulationResult:
    return SimulationResult(
        timepoints=np.array([0.0, 1.0]),
        species=["A", "B"],
        values=np.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
    )


def make_fit_result() -> FitResult:
    return FitResult(
        success=True,
        message="Converged",
        fitted_parameters={
            "k1f": 0.0,
            "k1r": 0.5,
            "k2f": 0.5,
        },
        initial_parameters={
            "k1f": 0.1,
            "k1r": 0.1,
            "k2f": 0.1,
        },
        residuals=np.array([0.0, 0.1]),
        statistics={
            "rss": 0.01,
            "rmse": 0.05,
            "aic": -10.0,
            "bic": -9.0,
            "n_residuals": 2.0,
            "n_parameters": 1.0,
        },
        simulation_result=make_simulation_result(),
        nfev=5,
        cost=0.01,
        fitted_initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        fitted_observables={
            "amide": {
                "species": "A",
                "scale": 10.0,
                "offset": 0.0,
            }
        },
        initial_observables={
            "amide": {
                "species": "A",
                "scale": 1.0,
                "offset": 0.0,
            }
        },
    )


def test_is_at_lower_bound():
    assert is_at_lower_bound(0.0, 0.0)
    assert is_at_lower_bound(1e-10, 0.0)
    assert not is_at_lower_bound(0.1, 0.0)


def test_is_at_upper_bound():
    assert is_at_upper_bound(10.0, 10.0)
    assert is_at_upper_bound(9.999999999, 10.0)
    assert not is_at_upper_bound(9.0, 10.0)


def test_build_parameter_diagnostics_table_flags_lower_bound():
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    table = build_parameter_diagnostics_table(
        parameter_specs=parameter_specs,
        fitted_parameters={"k1f": 0.0},
    )

    assert list(table["kind"]) == ["parameter"]
    assert list(table["name"]) == ["k1f"]
    assert list(table["at_lower_bound"]) == [True]
    assert list(table["at_upper_bound"]) == [False]
    assert list(table["warning"]) == ["at_lower_bound"]


def test_build_parameter_diagnostics_table_does_not_flag_fixed_parameter():
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.0,
        )
    ]

    table = build_parameter_diagnostics_table(
        parameter_specs=parameter_specs,
        fitted_parameters={"k1f": 0.0},
    )

    assert list(table["optimized"]) == [False]
    assert list(table["warning"]) == [""]


def test_build_parameter_diagnostics_table_does_not_flag_tied_parameter():
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec(
            name="k2f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
            tied_to="k1f",
        ),
    ]

    table = build_parameter_diagnostics_table(
        parameter_specs=parameter_specs,
        fitted_parameters={
            "k1f": 0.5,
            "k2f": 0.5,
        },
    )

    assert list(table["optimized"]) == [True, False]
    assert list(table["warning"]) == ["", ""]


def test_build_initial_condition_diagnostics_table_flags_upper_bound():
    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=False,
        )
    ]

    table = build_initial_condition_diagnostics_table(
        initial_condition_specs=initial_condition_specs,
        fitted_initial_conditions={"A": 1.0},
    )

    assert list(table["kind"]) == ["initial_condition"]
    assert list(table["name"]) == ["A"]
    assert list(table["at_upper_bound"]) == [True]
    assert list(table["warning"]) == ["at_upper_bound"]


def test_build_observable_diagnostics_table_flags_bounds():
    observable_specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_initial_guess=1.0,
            scale_lower_bound=0.0,
            scale_upper_bound=10.0,
            scale_fixed=False,
            offset_initial_guess=0.0,
            offset_lower_bound=0.0,
            offset_upper_bound=1.0,
            offset_fixed=False,
        )
    ]

    table = build_observable_diagnostics_table(
        observable_specs=observable_specs,
        fitted_observables={
            "amide": {
                "species": "A",
                "scale": 10.0,
                "offset": 0.0,
            }
        },
    )

    assert list(table["kind"]) == ["observable_scale", "observable_offset"]
    assert list(table["warning"]) == ["at_upper_bound", "at_lower_bound"]


def test_build_fit_diagnostics_table_combines_sections():
    fit_result = make_fit_result()

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec(
            name="k1r",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.5,
        ),
        ParameterSpec(
            name="k2f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
            tied_to="k1f",
        ),
    ]

    initial_condition_specs = [
        InitialConditionSpec(
            species="A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=False,
        ),
        InitialConditionSpec(
            species="B",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=1.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]

    observable_specs = [
        ObservableSpec(
            data_column="amide",
            species="A",
            scale_initial_guess=1.0,
            scale_lower_bound=0.0,
            scale_upper_bound=10.0,
            scale_fixed=False,
            offset_initial_guess=0.0,
            offset_lower_bound=0.0,
            offset_upper_bound=1.0,
            offset_fixed=False,
        )
    ]

    table = build_fit_diagnostics_table(
        fit_result=fit_result,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        observable_specs=observable_specs,
    )

    assert isinstance(table, pd.DataFrame)

    assert set(table["kind"]) == {
        "parameter",
        "initial_condition",
        "observable_scale",
        "observable_offset",
    }

    assert "at_lower_bound" in table.columns
    assert "at_upper_bound" in table.columns
    assert "warning" in table.columns


def test_get_diagnostic_warnings():
    fit_result = make_fit_result()

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    table = build_fit_diagnostics_table(
        fit_result=fit_result,
        parameter_specs=parameter_specs,
    )

    warnings = get_diagnostic_warnings(table)

    assert len(warnings) == 1
    assert "k1f" in warnings[0]
    assert "at_lower_bound" in warnings[0]


def test_missing_fitted_parameter_raises_error():
    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
        )
    ]

    with pytest.raises(ValueError):
        build_parameter_diagnostics_table(
            parameter_specs=parameter_specs,
            fitted_parameters={},
        )
