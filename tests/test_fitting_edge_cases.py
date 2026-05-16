import numpy as np
import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.optimizer import fit_model
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.model.model_spec import build_model_spec


def make_decay_dataset(
    true_k: float = 0.5,
    a0: float = 1.0,
    number_of_points: int = 30,
) -> Dataset:
    timepoints = np.linspace(0.0, 5.0, number_of_points)
    a_values = a0 * np.exp(-true_k * timepoints)
    b_values = a0 - a_values

    dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": a_values,
            "B": b_values,
        }
    )

    return Dataset(
        raw_dataframe=dataframe,
        time_column="time",
        signal_columns=["A", "B"],
    )


def test_fit_model_accepts_old_initial_conditions_dict():
    model = build_model_spec("A>B")
    dataset = make_decay_dataset(true_k=0.5)

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.0,
                upper_bound=10.0,
            )
        ],
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            }
        ),
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(0.5, rel=1e-2)


def test_fit_model_with_fitted_initial_condition_counts_free_variables():
    model = build_model_spec("A>B")
    dataset = make_decay_dataset(true_k=0.5, a0=2.0)

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.0,
                upper_bound=10.0,
            )
        ],
        initial_condition_specs=[
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
        ],
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            }
        ),
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(0.5, rel=1e-2)
    assert result.fitted_initial_conditions["A"] == pytest.approx(2.0, rel=1e-2)
    assert result.statistics["n_parameters"] == 2.0


def test_fit_model_with_tied_parameter_and_fitted_initial_condition():
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

    c0 = 2.0
    c_values = c0 * np.exp(-true_k * timepoints)
    d_values = c0 - c_values

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

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.0,
                upper_bound=10.0,
            ),
            ParameterSpec(
                name="k2f",
                initial_guess=0.2,
                lower_bound=0.0,
                upper_bound=10.0,
                tied_to="k1f",
            ),
        ],
        initial_condition_specs=[
            InitialConditionSpec("A", initial_guess=1.0, fixed=True, fixed_value=1.0),
            InitialConditionSpec("B", initial_guess=0.0, fixed=True, fixed_value=0.0),
            InitialConditionSpec(
                "C",
                initial_guess=1.0,
                lower_bound=0.0,
                upper_bound=5.0,
                fixed=False,
            ),
            InitialConditionSpec("D", initial_guess=0.0, fixed=True, fixed_value=0.0),
        ],
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
                "C": "C",
                "D": "D",
            }
        ),
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.fitted_parameters["k2f"] == pytest.approx(true_k, rel=1e-2)
    assert result.fitted_initial_conditions["C"] == pytest.approx(c0, rel=1e-2)
    assert result.statistics["n_parameters"] == 2.0


def test_fit_model_with_normalized_data():
    model = build_model_spec("A>B")

    true_k = 0.5
    timepoints = np.linspace(0.0, 5.0, 30)

    normalized_a = np.exp(-true_k * timepoints)
    normalized_b = 1.0 - normalized_a

    raw_dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": 10.0 * normalized_a,
            "B": 10.0 * normalized_b,
        }
    )

    normalized_dataframe = pd.DataFrame(
        {
            "time": timepoints,
            "A": normalized_a,
            "B": normalized_b,
        }
    )

    dataset = Dataset(
        raw_dataframe=raw_dataframe,
        normalized_dataframe=normalized_dataframe,
        normalization_method="manual",
        time_column="time",
        signal_columns=["A", "B"],
    )

    result = fit_model(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec(
                name="k1f",
                initial_guess=0.1,
                lower_bound=0.0,
                upper_bound=10.0,
            )
        ],
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        settings=FitSettings(
            species_mapping={
                "A": "A",
                "B": "B",
            },
            use_normalized_data=True,
        ),
    )

    assert result.success
    assert result.fitted_parameters["k1f"] == pytest.approx(true_k, rel=1e-2)
    assert result.statistics["rmse"] < 1e-3


def test_fit_model_rejects_invalid_loss():
    model = build_model_spec("A>B")
    dataset = make_decay_dataset(true_k=0.5)

    with pytest.raises(ValueError):
        fit_model(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec(
                    name="k1f",
                    initial_guess=0.1,
                    lower_bound=0.0,
                    upper_bound=10.0,
                )
            ],
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            settings=FitSettings(
                species_mapping={"A": "A"},
                loss="not_a_loss",
            ),
        )
