import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def test_fit_irreversible_decay_synthetic_data():
    model = build_model_spec("A>B")

    true_k = 0.5

    timepoints = np.linspace(0.0, 5.0, 20)
    a_values = np.exp(-true_k * timepoints)
    b_values = 1.0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    dataset = Dataset(
        raw_dataframe=dataframe,
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

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        }
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
        settings=settings,
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.statistics["rmse"] < 1e-3


def test_fit_with_fixed_reverse_parameter():
    model = build_model_spec("A-B")

    true_k = 0.5

    timepoints = np.linspace(0.0, 5.0, 30)
    a_values = np.exp(-true_k * timepoints)
    b_values = 1.0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        ParameterSpec(
            name="k1r",
            initial_guess=0.0,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.0,
        ),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        }
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
        settings=settings,
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.fitted_parameters["k1r"] == 0.0


def test_fit_all_parameters_fixed_raises_error():
    model = build_model_spec("A>B")

    timepoints = np.linspace(0.0, 5.0, 20)
    a_values = np.exp(-0.5 * timepoints)
    b_values = 1.0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    dataset = Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.5,
            lower_bound=0.0,
            upper_bound=10.0,
            fixed=True,
            fixed_value=0.5,
        )
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        }
    )

    with pytest.raises(ValueError):
        fit_model(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )

        def test_fit_tied_parameters_synthetic_data():
            model = build_model_spec(
                """
                A>B
                C>D
                """
            )

            true_k = 0.5

            timepoints = np.linspace(0.0, 5.0, 30)

            a_values = np.exp(-true_k * timepoints)
            b_values = 1.0 - a_values

            c_values = 2.0 * np.exp(-true_k * timepoints)
            d_values = 2.0 - c_values

            dataframe = pd.DataFrame(
                {
                    "time": timepoints,
                    "A": a_values,
                    "B": b_values,
                    "C": c_values,
                    "D": d_values,
                }
            )

            dataset = Dataset(
                raw_dataframe=dataframe,
                time_column="time",
                signal_columns=["A", "B", "C", "D"],
            )

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

            initial_conditions = {
                "A": 1.0,
                "B": 0.0,
                "C": 2.0,
                "D": 0.0,
            }

            settings = FitSettings(
                species_mapping={
                    "A": "A",
                    "B": "B",
                    "C": "C",
                    "D": "D",
                }
            )

            result = fit_model(
                model=model,
                dataset=dataset,
                parameter_specs=parameter_specs,
                initial_conditions=initial_conditions,
                settings=settings,
            )

            assert result.success
            assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
            assert result.fitted_parameters["k2f"] == pytest.approx(true_k, rel=1e-2)
            assert result.fitted_parameters["k1f"] == result.fitted_parameters["k2f"]


from odefit.fitting.initial_condition_spec import InitialConditionSpec


def test_fit_initial_condition_synthetic_data():
    model = build_model_spec("A>B")

    true_k = 0.5
    true_a0 = 2.0

    timepoints = np.linspace(0.0, 5.0, 30)
    a_values = true_a0 * np.exp(-true_k * timepoints)
    b_values = true_a0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    dataset = Dataset(
        raw_dataframe=dataframe,
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

    settings = FitSettings(
        species_mapping={
            "A": "A",
            "B": "B",
        }
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_condition_specs=initial_condition_specs,
        settings=settings,
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.fitted_initial_conditions["A"] == pytest.approx(true_a0, rel=1e-2)
    assert result.fitted_initial_conditions["B"] == 0.0
    assert result.statistics["rmse"] < 1e-3
