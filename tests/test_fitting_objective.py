import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.objective import objective_function
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_objective_function_returns_near_zero_residuals_for_true_parameter():
    model = build_model_spec("A>B")

    true_k = 0.5
    timepoints = np.linspace(0.0, 5.0, 12)

    a_values = np.exp(-true_k * timepoints)
    b_values = 1.0 - a_values

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": timepoints,
                "A": a_values,
                "B": b_values,
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True, fixed_value=1.0),
        InitialConditionSpec("B", initial_guess=0.0, fixed=True, fixed_value=0.0),
    ]

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        },
        rtol=1e-9,
        atol=1e-11,
    )

    residuals = objective_function(
        optimization_vector=np.array([true_k]),
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    assert np.max(np.abs(residuals)) < 1e-5


def test_objective_function_with_fitted_initial_condition():
    model = build_model_spec("A>B")

    true_k = 0.5
    true_a0 = 2.0

    timepoints = np.linspace(0.0, 5.0, 12)

    a_values = true_a0 * np.exp(-true_k * timepoints)
    b_values = true_a0 - a_values

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": timepoints,
                "A": a_values,
                "B": b_values,
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        )
    ]

    initial_condition_specs = [
        InitialConditionSpec(
            "A",
            initial_guess=1.0,
            lower_bound=0.0,
            upper_bound=5.0,
            fixed=False,
        ),
        InitialConditionSpec("B", initial_guess=0.0, fixed=True, fixed_value=0.0),
    ]

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        },
        rtol=1e-9,
        atol=1e-11,
    )

    residuals = objective_function(
        optimization_vector=np.array([true_k, true_a0]),
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    assert np.max(np.abs(residuals)) < 1e-5


def test_objective_function_bad_parameter_vector_length_raises_error():
    model = build_model_spec("A>B")

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "A": [1.0, 0.5],
                "B": [0.0, 0.5],
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec("k1f", initial_guess=0.1),
    ]

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.0, fixed=True),
    ]

    settings = FitSettings(species_mapping={"A": "A"})

    with pytest.raises(ValueError):
        objective_function(
            optimization_vector=np.array([0.5, 1.0]),
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_condition_specs=initial_condition_specs,
            settings=settings,
        )
