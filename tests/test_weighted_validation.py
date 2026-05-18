import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.observable_spec import ObservableSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.validation import validate_fit_inputs
from odefit.model.model_spec import build_model_spec


def make_dataset():
    return Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A": [1.0, 0.5, 0.25],
                "amide": [2.0, 1.0, 0.5],
                "unused": [10.0, 10.0, 10.0],
            }
        ),
        time_column="time",
        signal_columns=["A", "amide", "unused"],
    )


def test_validate_fit_inputs_accepts_signal_weights_for_species_mapping():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    validate_fit_inputs(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec("k1f", initial_guess=0.1),
        ],
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        },
        settings=FitSettings(
            species_mapping={"A": "A"},
            signal_weights={"A": 2.0},
        ),
    )


def test_validate_fit_inputs_accepts_signal_weights_for_observable_mapping():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    validate_fit_inputs(
        model=model,
        dataset=dataset,
        parameter_specs=[
            ParameterSpec("k1f", initial_guess=0.1),
        ],
        initial_condition_specs=[
            InitialConditionSpec("A", initial_guess=1.0, fixed=True),
            InitialConditionSpec("B", initial_guess=0.0, fixed=True),
        ],
        observable_specs=[
            ObservableSpec("amide", "A"),
        ],
        settings=FitSettings(
            species_mapping={},
            signal_weights={"amide": 2.0},
        ),
    )


def test_validate_fit_inputs_rejects_negative_signal_weight():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec("k1f", initial_guess=0.1),
            ],
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            settings=FitSettings(
                species_mapping={"A": "A"},
                signal_weights={"A": -1.0},
            ),
        )


def test_validate_fit_inputs_rejects_zero_signal_weight():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec("k1f", initial_guess=0.1),
            ],
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            settings=FitSettings(
                species_mapping={"A": "A"},
                signal_weights={"A": 0.0},
            ),
        )


def test_validate_fit_inputs_rejects_weight_for_unknown_column():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec("k1f", initial_guess=0.1),
            ],
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            settings=FitSettings(
                species_mapping={"A": "A"},
                signal_weights={"missing": 1.0},
            ),
        )


def test_validate_fit_inputs_rejects_weight_for_unmapped_column():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec("k1f", initial_guess=0.1),
            ],
            initial_conditions={
                "A": 1.0,
                "B": 0.0,
            },
            settings=FitSettings(
                species_mapping={"A": "A"},
                signal_weights={"unused": 1.0},
            ),
        )
