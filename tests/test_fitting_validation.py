import pandas as pd
import pytest

from odefit.data.dataset import Dataset
from odefit.fitting.fit_settings import FitSettings
from odefit.fitting.parameter_spec import ParameterSpec
from odefit.fitting.validation import validate_fit_inputs
from odefit.model.model_spec import build_model_spec


def make_dataset():
    return Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "A": [1.0, 0.8, 0.6],
                "B": [0.0, 0.1, 0.2],
            }
        ),
        time_column="time",
        signal_columns=["A", "B"],
    )


def test_validate_fit_inputs_accepts_valid_inputs():
    model = build_model_spec("A>B")
    dataset = make_dataset()

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

    validate_fit_inputs(
        model=model,
        dataset=dataset,
        parameter_specs=parameter_specs,
        initial_conditions=initial_conditions,
        settings=settings,
    )


def test_missing_parameter_spec_raises_error():
    model = build_model_spec("A-B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(species_mapping={"A": "A"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_extra_parameter_spec_raises_error():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
        ParameterSpec(name="k_extra", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(species_mapping={"A": "A"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_invalid_data_column_mapping_raises_error():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(species_mapping={"C": "A"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_invalid_model_species_mapping_raises_error():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(species_mapping={"A": "C"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_missing_initial_condition_raises_error():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
    }

    settings = FitSettings(species_mapping={"A": "A"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_normalized_data_missing_raises_error():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(
        species_mapping={"A": "A"},
        use_normalized_data=True,
    )

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_all_parameters_fixed_raises_error():
    model = build_model_spec("A>B")
    dataset = make_dataset()

    parameter_specs = [
        ParameterSpec(
            name="k1f",
            initial_guess=0.1,
            fixed=True,
            fixed_value=0.1,
        )
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(species_mapping={"A": "A"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )


def test_too_few_residuals_raises_error():
    model = build_model_spec("A>B")

    dataset = Dataset(
        raw_dataframe=pd.DataFrame(
            {
                "time": [0.0],
                "A": [1.0],
            }
        ),
        time_column="time",
        signal_columns=["A"],
    )

    parameter_specs = [
        ParameterSpec(name="k1f", initial_guess=0.1),
    ]

    initial_conditions = {
        "A": 1.0,
        "B": 0.0,
    }

    settings = FitSettings(species_mapping={"A": "A"})

    with pytest.raises(ValueError):
        validate_fit_inputs(
            model=model,
            dataset=dataset,
            parameter_specs=parameter_specs,
            initial_conditions=initial_conditions,
            settings=settings,
        )
