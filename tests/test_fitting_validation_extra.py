import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.initial_condition_spec import InitialConditionSpec
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.validation import (
    resolve_initial_condition_specs,
    validate_fit_inputs,
)
from odefit.model.model_spec import build_model_spec


def make_dataset(normalized: bool = False):
    raw_dataframe = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "A": [1.0, 0.8, 0.6],
            "B": [0.0, 0.1, 0.2],
        }
    )

    normalized_dataframe = None

    if normalized:
        normalized_dataframe = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A": [1.0, 0.8, 0.6],
                "B": [0.0, 0.1, 0.2],
            }
        )

    return Dataset(
        raw_dataframe=raw_dataframe,
        normalized_dataframe=normalized_dataframe,
        normalization_method="manual" if normalized else None,
        time_column="time",
        signal_columns=["A", "B"],
    )


def test_resolve_initial_condition_specs_from_dict():
    specs = resolve_initial_condition_specs(
        initial_conditions={
            "A": 1.0,
            "B": 0.0,
        }
    )

    assert len(specs) == 2
    assert specs[0].species == "A"
    assert specs[0].fixed is True
    assert specs[0].fixed_value == 1.0


def test_resolve_initial_condition_specs_rejects_both_styles():
    with pytest.raises(ValueError):
        resolve_initial_condition_specs(
            initial_conditions={"A": 1.0},
            initial_condition_specs=[
                InitialConditionSpec("A", initial_guess=1.0),
            ],
        )


def test_resolve_initial_condition_specs_rejects_missing_inputs():
    with pytest.raises(ValueError):
        resolve_initial_condition_specs()


def test_validate_fit_inputs_rejects_missing_fit_settings():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[ParameterSpec("k1f", initial_guess=0.1)],
            initial_conditions={"A": 1.0, "B": 0.0},
            settings=None,
        )


def test_validate_fit_inputs_rejects_missing_initial_condition_spec():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
    ]

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[ParameterSpec("k1f", initial_guess=0.1)],
            initial_condition_specs=initial_condition_specs,
            settings=FitSettings(species_mapping={"A": "A"}),
        )


def test_validate_fit_inputs_rejects_extra_initial_condition_spec():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    initial_condition_specs = [
        InitialConditionSpec("A", initial_guess=1.0, fixed=True),
        InitialConditionSpec("B", initial_guess=0.0, fixed=True),
        InitialConditionSpec("C", initial_guess=0.0, fixed=True),
    ]

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[ParameterSpec("k1f", initial_guess=0.1)],
            initial_condition_specs=initial_condition_specs,
            settings=FitSettings(species_mapping={"A": "A"}),
        )


def test_validate_fit_inputs_rejects_empty_species_mapping():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[ParameterSpec("k1f", initial_guess=0.1)],
            initial_conditions={"A": 1.0, "B": 0.0},
            settings=FitSettings(species_mapping={}),
        )


def test_validate_fit_inputs_accepts_normalized_data_when_available():
    model = build_model_spec("A>B")
    dataset = make_dataset(normalized=True)

    validate_fit_inputs(
        model=model,
        dataset=dataset,
        parameter_specs=[ParameterSpec("k1f", initial_guess=0.1)],
        initial_conditions={"A": 1.0, "B": 0.0},
        settings=FitSettings(
            species_mapping={"A": "A"},
            use_normalized_data=True,
        ),
    )


def test_validate_fit_inputs_rejects_zero_free_variables():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=[
                ParameterSpec(
                    "k1f",
                    initial_guess=0.1,
                    fixed=True,
                    fixed_value=0.1,
                )
            ],
            initial_conditions={"A": 1.0, "B": 0.0},
            settings=FitSettings(species_mapping={"A": "A"}),
        )
